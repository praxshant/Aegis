"""
engine.py — Intelligence Engine (Aegis)
----------------------------------------
Derives real CustomerIntelligence from transaction history using the
Aegis-trained HistGradientBoosting propensity model.

ARCHITECTURE:
  Customer + Observation Date
        ↓
  Raw Transactions (from DB)
        ↓
  build_features_for_customer() [Aegis-exact feature engineering]
        ↓
  NextPurchasePropensityModel.predict_proba()   [existing Aegis model]
        ↓
  Segment lookup (from Aegis cluster_labels)
        ↓
  Behavioral Signals (from Aegis action_policy logic)
        ↓
  CustomerIntelligence [COMPLETE or INCOMPLETE]

RULE: Never fabricate a propensity score.
RULE: If inference is impossible → status = INCOMPLETE → no financial action.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from backend.intelligence.schemas import (
    CustomerIntelligence,
    IntelligenceStatus,
    EvidenceItem,
)
from backend.models.propensity_features import (
    build_features_for_customer,
    FeatureBuildError,
)
from backend.models.propensity_model import (
    NextPurchasePropensityModel,
    PropensityModelError,
    get_propensity_model,
)
from backend.models.next_basket_features import assemble_features_for_next_basket
from backend.models.next_basket_model import get_next_basket_model, NextBasketModelError
from backend.policy.gate import PolicyGate

logger = logging.getLogger("IntelligenceEngine")


def _format_next_basket_intent(
    customer_id: str, 
    scored_frame: pd.DataFrame, 
    score_col: str, 
    model_name: str = "instacart_next_basket", 
    model_version: str = "1.1.0", 
    top_n: int = 5
) -> dict:
    """Formats the scored next basket candidates into the Aegis intent schema."""
    user_rows = scored_frame[scored_frame["user_id"] == 1].sort_values(score_col, ascending=False).head(top_n)
    
    if user_rows.empty:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "model": model_name,
            "model_version": model_version,
            "unavailable_reason": "No products scored.",
        }

    # Extract catalog info which was joined during feature assembly
    top_products = []
    for r in user_rows.itertuples():
        # Handle cases where product catalog wasn't fully joined or is missing names
        p_name = getattr(r, "product_name", f"Product {r.product_id}")
        if pd.isna(p_name):
            p_name = f"Product {r.product_id}"
            
        top_products.append({
            "product_id": int(r.product_id),
            "product_name": str(p_name),
            "probability": round(float(getattr(r, score_col)), 4),
        })
        
    top_departments = []
    if "department" in user_rows.columns:
        top_departments = user_rows["department"].value_counts().head(3).index.tolist()
        
    top_aisles = []
    if "aisle" in user_rows.columns:
        top_aisles = user_rows["aisle"].value_counts().head(3).index.tolist()
        
    mean_top_n = round(float(user_rows[score_col].mean()), 4)
    
    if mean_top_n >= 0.5:
        signal = "likely_to_repurchase_from_usual_basket"
    elif mean_top_n >= 0.2:
        signal = "moderate_repurchase_signal"
    else:
        signal = "weak_repurchase_signal"

    return {
        "available": True,
        "status": "COMPLETE",
        "model": model_name,
        "model_version": model_version,
        "top_products": top_products,
        "top_departments": top_departments,
        "top_aisles": top_aisles,
        "mean_top_n_probability": mean_top_n,
        "signal": signal,
    }


class IntelligenceEngine:
    """
    Derives real CustomerIntelligence from transaction history using the
    existing Aegis-trained propensity model.
    """

    def __init__(
        self,
        policy_gate: Optional[PolicyGate] = None,
        propensity_model: Optional[NextPurchasePropensityModel] = None,
    ):
        self.policy_gate = policy_gate or PolicyGate()
        self.signals_config = self.policy_gate.signals_config

        # The model is a singleton — get_propensity_model() returns the cached instance.
        # If the model is not yet loaded this will load it. If loading fails, it raises
        # PropensityModelError which the API must handle.
        try:
            self._model = propensity_model or get_propensity_model()
        except PropensityModelError as e:
            logger.error(f"Propensity model unavailable: {e}")
            self._model = None
            
        # Also initialize NextBasketModel
        try:
            self._next_basket_model = get_next_basket_model()
        except NextBasketModelError as e:
            logger.error(f"Next Basket model unavailable: {e}")
            self._next_basket_model = None

    # ── Public interface ───────────────────────────────────────────────────

    def get_intelligence(
        self,
        customer_id: str,
        transactions: pd.DataFrame,
        observation_date: Optional[pd.Timestamp] = None,
        segment: Optional[str] = None,
    ) -> CustomerIntelligence:
        """
        Produce a CustomerIntelligence object for a single customer.

        Args:
            customer_id:       Customer identifier.
            transactions:      All raw transactions for this customer
                               (columns per REQUIRED_TRANSACTION_COLUMNS).
            observation_date:  Point-in-time cutoff. Defaults to today.
                               Only transactions <= observation_date are used.
            segment:           Pre-computed Aegis segment label (optional).
                               If None, defaults to 'Unknown'.

        Returns:
            CustomerIntelligence with status=COMPLETE or status=INCOMPLETE.
            INCOMPLETE means no autonomous financial action should be taken.
        """
        if observation_date is None:
            observation_date = pd.Timestamp.now().normalize()

        obs_date_str = observation_date.strftime("%Y-%m-%d")

        # ── Check model availability ───────────────────────────────────────
        if self._model is None:
            logger.warning(f"Propensity model unavailable for customer {customer_id}")
            return CustomerIntelligence(
                customer_id=customer_id,
                observation_date=obs_date_str,
                status=IntelligenceStatus.INCOMPLETE,
                incomplete_reason="Propensity model is not available. Cannot generate intelligence.",
            )

        # ── Build features ─────────────────────────────────────────────────
        try:
            feature_df = build_features_for_customer(
                customer_id=customer_id,
                transactions=transactions,
                observation_date=observation_date,
            )
        except FeatureBuildError as e:
            logger.warning(f"Cannot build features for {customer_id}: {e}")
            return CustomerIntelligence(
                customer_id=customer_id,
                observation_date=obs_date_str,
                status=IntelligenceStatus.INCOMPLETE,
                incomplete_reason=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error building features for {customer_id}: {e}")
            return CustomerIntelligence(
                customer_id=customer_id,
                observation_date=obs_date_str,
                status=IntelligenceStatus.INCOMPLETE,
                incomplete_reason=f"Feature engineering error: {e}",
            )

        # ── Run inference ──────────────────────────────────────────────────
        try:
            propensity_score, propensity_band = self._model.predict_band(feature_df)
        except (PropensityModelError, Exception) as e:
            logger.error(f"Model inference failed for {customer_id}: {e}")
            return CustomerIntelligence(
                customer_id=customer_id,
                observation_date=obs_date_str,
                status=IntelligenceStatus.INCOMPLETE,
                incomplete_reason=f"Model inference failed: {e}",
            )

        # ── Behavioral signals (ported from Aegis IntelligenceEngine) ─────
        row = feature_df.iloc[0]
        signals = self._generate_signals(row)

        # ── Policy pre-evaluation ──────────────────────────────────────────
        seg = segment or "Unknown"
        policy_result = self.policy_gate.evaluate_pre_action(seg, propensity_band, signals)
        history_days = float(row.get("FEATURE_customer_lifetime_days", 1.0))
        confidence = self.policy_gate.get_confidence(
            policy_result["policy_rule_id"], signals, history_days
        )

        # ── Evidence list ──────────────────────────────────────────────────
        why_list = []
        evidence_list = []
        for sig_name, sig_data in signals.items():
            if sig_data["active"]:
                why_list.append(sig_data["message"])
                evidence_list.append(
                    EvidenceItem(
                        signal=sig_name,
                        value=sig_data["value"],
                        threshold=sig_data["threshold"],
                        message=sig_data["message"],
                    )
                )

        evidence_status = "EVIDENCE_PRESENT" if evidence_list else (
            "INSUFFICIENT_HISTORY" if history_days < 30 else "NO_ACTIVE_SIGNALS"
        )

        active_signals = {k: v["active"] for k, v in signals.items()}

        # ── Feature snapshot (human-readable subset for agent context) ─────
        feature_snapshot = {
            "recency_days": int(row.get("FEATURE_recency_days", 0)),
            "lifetime_days": int(row.get("FEATURE_customer_lifetime_days", 0)),
            "purchase_order_count": int(row.get("FEATURE_purchase_order_count", 0)),
            "total_net_spend": round(float(row.get("FEATURE_total_net_spend", 0)), 2),
            "average_order_value": round(float(row.get("FEATURE_average_order_value", 0)), 2),
            "net_spend_90d": round(float(row.get("FEATURE_net_spend_90d", 0)), 2),
            "purchase_order_count_90d": int(row.get("FEATURE_purchase_order_count_90d", 0)),
            "return_rate": round(float(row.get("FEATURE_return_rate", 0)), 4),
            "is_active_30d": int(row.get("FEATURE_is_active_30d", 0)),
            "is_active_90d": int(row.get("FEATURE_is_active_90d", 0)),
        }

        # ── Next Basket prediction (Phase 1 Integration) ───────────────────
        next_basket_payload = None
        
        # Try lazy load if previously unavailable
        if self._next_basket_model is None:
            try:
                self._next_basket_model = get_next_basket_model()
            except NextBasketModelError:
                pass

        if self._next_basket_model is not None:
            # 1. Load product catalog if available
            product_catalog = None
            catalog_path = Path("artifacts/data/instacart_product_subset.parquet")
            if catalog_path.exists():
                try:
                    product_catalog = pd.read_parquet(catalog_path)
                except Exception as e:
                    logger.warning(f"Failed to load product catalog: {e}")
            
            # 2. Build Next-Basket Features
            nb_status, nb_frame, nb_reason = assemble_features_for_next_basket(
                customer_id=customer_id,
                tx_df=transactions,
                product_catalog=product_catalog
            )
            
            if nb_status == "COMPLETE" and nb_frame is not None:
                # 3. Predict
                try:
                    scored_frame = self._next_basket_model.predict_next_basket(nb_frame)
                    
                    # Also join department/aisle names for formatting if catalog has them
                    if product_catalog is not None and "product_name" in product_catalog.columns:
                        cat_names = product_catalog[["instacart_product_id", "product_name", "department", "aisle"]].drop_duplicates()
                        cat_names = cat_names.rename(columns={"instacart_product_id": "product_id"})
                        scored_frame = scored_frame.merge(cat_names, on="product_id", how="left")
                        
                    intent_dict = _format_next_basket_intent(customer_id, scored_frame, "score")
                    next_basket_payload = intent_dict
                except Exception as e:
                    logger.error(f"Next-basket inference failed: {e}")
                    next_basket_payload = {
                        "available": False,
                        "status": "ERROR",
                        "unavailable_reason": f"Inference failed: {str(e)}"
                    }
            else:
                next_basket_payload = {
                    "available": False,
                    "status": "UNAVAILABLE",
                    "unavailable_reason": nb_reason
                }
        else:
            next_basket_payload = {
                "available": False,
                "status": "UNAVAILABLE",
                "unavailable_reason": "Next-Basket model is not loaded."
            }

        logger.info(
            f"Intelligence COMPLETE for {customer_id}: "
            f"score={propensity_score:.4f}, band={propensity_band}, segment={seg}"
        )

        return CustomerIntelligence(
            customer_id=customer_id,
            observation_date=obs_date_str,
            status=IntelligenceStatus.COMPLETE,
            segment=seg,
            purchase_probability_30d=round(propensity_score, 4),
            propensity_band=propensity_band,
            propensity_source="model_inference",
            recommendation=policy_result["recommendation"],
            recommendation_confidence=confidence,
            policy_rule_id=policy_result["policy_rule_id"],
            why=why_list,
            evidence=evidence_list,
            evidence_status=evidence_status,
            behavioral_signals=active_signals,
            feature_snapshot=feature_snapshot,
            next_basket=next_basket_payload,
        )

    # ── Private: behavioral signal generation (Aegis-exact logic) ─────────

    def _generate_signals(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate deterministic behavioral signals from a feature row.
        Logic ported from Aegis ml/intelligence_engine.py _generate_signals().
        """
        signals: Dict[str, Any] = {}
        history_days = max(float(row.get("FEATURE_customer_lifetime_days", 1)), 1.0)
        orders = max(float(row.get("FEATURE_purchase_order_count", 0)), 1.0)

        # 1. recency_elevated
        recency = float(row.get("FEATURE_recency_days", 0))
        normal_interval = history_days / orders
        recency_ratio = recency / max(normal_interval, 1.0)
        threshold = self.signals_config.get("recency_elevated", {}).get("threshold", 2.0)
        is_elevated = bool(recency_ratio >= threshold and history_days >= 30 and orders >= 2)
        signals["recency_elevated"] = {
            "active": is_elevated,
            "value": round(recency_ratio, 2),
            "threshold": threshold,
            "message": (
                f"Recency is {recency_ratio:.1f}x the customer's normal purchase interval."
                if is_elevated else ""
            ),
        }

        # 2. spend_declining
        spend_90d = float(row.get("FEATURE_net_spend_90d", 0))
        total_spend = float(row.get("FEATURE_total_net_spend", 0))
        baseline_90d = (total_spend / max(history_days, 90.0)) * 90.0
        spend_ratio = spend_90d / max(baseline_90d, 1.0)
        threshold = self.signals_config.get("spend_declining", {}).get("threshold", 0.70)
        is_declining = bool(spend_ratio <= threshold and history_days >= 90 and total_spend > 0)
        signals["spend_declining"] = {
            "active": is_declining,
            "value": round(spend_ratio, 2),
            "threshold": threshold,
            "message": (
                f"90-day spend is {int((1 - spend_ratio) * 100)}% below the customer's historical baseline."
                if is_declining else ""
            ),
        }

        # 3. frequency_declining
        freq_90d = float(row.get("FEATURE_purchase_order_count_90d", 0))
        baseline_90d_freq = (orders / max(history_days, 90.0)) * 90.0
        freq_ratio = freq_90d / max(baseline_90d_freq, 1.0)
        threshold = self.signals_config.get("frequency_declining", {}).get("threshold", 0.50)
        is_freq_declining = bool(freq_ratio <= threshold and history_days >= 90 and orders >= 3)
        signals["frequency_declining"] = {
            "active": is_freq_declining,
            "value": round(freq_ratio, 2),
            "threshold": threshold,
            "message": (
                "Recent 90-day purchase frequency is materially below historical cadence."
                if is_freq_declining else ""
            ),
        }

        # 4. return_risk
        return_rate = float(row.get("FEATURE_return_rate", 0))
        threshold = self.signals_config.get("return_risk", {}).get("threshold", 0.15)
        has_return_risk = bool(
            return_rate > threshold and float(row.get("FEATURE_return_line_count", 0)) > 0
        )
        signals["return_risk"] = {
            "active": has_return_risk,
            "value": round(return_rate, 3),
            "threshold": threshold,
            "message": (
                f"Return rate of {return_rate:.1%} is above normal operating thresholds."
                if has_return_risk else ""
            ),
        }

        return signals
