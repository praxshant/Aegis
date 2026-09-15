"""
propensity_features.py
----------------------
Canonical feature contract for the Aegis propensity inference pipeline.

SOURCE: Aegis / backend/training/prepared/online_retail_ii/temporal/02_temporal_dataset.ipynb
        function: calc_features(group)

RULE: Every feature definition here must exactly reproduce Aegis's calc_features().
      Observation date is mandatory. Features are calculated ONLY from transactions
      where transaction_timestamp <= observation_date (no future leakage).

Model expects exactly these 19 features in this order.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("propensity_features")

# ── Exact feature list as recorded in model.feature_names_in_ ──────────────
PROPENSITY_FEATURES = [
    "FEATURE_recency_days",
    "FEATURE_customer_lifetime_days",
    "FEATURE_purchase_order_count",
    "FEATURE_purchase_line_count",
    "FEATURE_return_line_count",
    "FEATURE_cancellation_line_count",
    "FEATURE_total_gross_purchase_spend",
    "FEATURE_total_return_amount",
    "FEATURE_total_net_spend",
    "FEATURE_average_order_value",
    "FEATURE_purchase_frequency",
    "FEATURE_return_rate",
    "FEATURE_unique_products_purchased",
    "FEATURE_net_spend_30d",
    "FEATURE_net_spend_90d",
    "FEATURE_purchase_order_count_30d",
    "FEATURE_purchase_order_count_90d",
    "FEATURE_is_active_30d",
    "FEATURE_is_active_90d",
]

# ── Required columns in the raw transactions DataFrame ─────────────────────
REQUIRED_TRANSACTION_COLUMNS = {
    "customer_id",
    "transaction_timestamp",
    "transaction_type",   # 'purchase' | 'return' | 'cancellation'
    "invoice",            # unique invoice/order identifier
    "product_id",
    "gross_amount",
    "return_amount",
    "net_amount",
}


class FeatureBuildError(Exception):
    """Raised when features cannot be built (e.g., no purchase history)."""
    pass


def build_features_for_customer(
    customer_id: str,
    transactions: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the exact 19 PROPENSITY_FEATURES for a single customer as of observation_date.

    Args:
        customer_id:       The customer identifier.
        transactions:      DataFrame of ALL raw transactions for this customer.
                           Must contain REQUIRED_TRANSACTION_COLUMNS.
        observation_date:  The point-in-time cutoff. Only transactions on or before
                           this date are used. This is STRICTLY enforced.

    Returns:
        A single-row DataFrame with exactly PROPENSITY_FEATURES columns, in order.

    Raises:
        FeatureBuildError: If the customer has no purchase history on/before observation_date,
                           making inference impossible.
        ValueError:        If required columns are missing from the transactions DataFrame.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    missing_cols = REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
    if missing_cols:
        raise ValueError(f"transactions DataFrame missing required columns: {missing_cols}")

    # ── Enforce observation date (no future leakage) ──────────────────────
    observation_date = pd.Timestamp(observation_date)

    # Coerce timestamp column to datetime — SQLite returns strings on read-back
    transactions = transactions.copy()
    transactions["transaction_timestamp"] = pd.to_datetime(
        transactions["transaction_timestamp"], errors="coerce", utc=False
    )
    # Strip timezone info if present (SQLite has no tz concept)
    if transactions["transaction_timestamp"].dt.tz is not None:
        transactions["transaction_timestamp"] = transactions["transaction_timestamp"].dt.tz_localize(None)

    history = transactions[
        (transactions["customer_id"] == customer_id) &
        (transactions["transaction_timestamp"] <= observation_date)
    ].copy()

    purchases = history[history["transaction_type"] == "purchase"]

    if len(purchases) == 0:
        raise FeatureBuildError(
            f"Customer {customer_id} has no purchase history on or before {observation_date.date()}. "
            "Cannot compute propensity features — inference not possible."
        )

    # ── Financials (all-time, as of observation_date) ─────────────────────
    gross_spend = history["gross_amount"].sum()
    net_spend = history["net_amount"].sum()
    return_amount = history["return_amount"].sum()

    # ── Counts ────────────────────────────────────────────────────────────
    purchase_order_count = purchases["invoice"].nunique()
    purchase_line_count = len(purchases)
    return_line_count = int((history["transaction_type"] == "return").sum())
    cancel_line_count = int((history["transaction_type"] == "cancellation").sum())
    total_lines = purchase_line_count + return_line_count + cancel_line_count

    # ── Dates: recency and lifetime ───────────────────────────────────────
    # Aegis definition: recency = days since LAST purchase (not last transaction)
    last_purchase_date = purchases["transaction_timestamp"].max()
    first_purchase_date = purchases["transaction_timestamp"].min()

    recency_days = (observation_date - last_purchase_date).days
    # lifetime = max(days between first and last purchase, 1) — per Aegis
    lifetime_days = max(int((last_purchase_date - first_purchase_date).days), 1)

    # ── Derived aggregates ────────────────────────────────────────────────
    # average_order_value = gross_spend / purchase_order_count (Aegis uses gross, not net)
    aov = gross_spend / purchase_order_count if purchase_order_count > 0 else 0.0

    # purchase_frequency = orders per 30-day month (lifetime in days / 30 gives months)
    freq = purchase_order_count / (lifetime_days / 30.0) if lifetime_days > 0 else 0.0

    # return_rate = return_lines / total_lines
    return_rate = return_line_count / total_lines if total_lines > 0 else 0.0

    # unique products purchased — from purchase lines only (Aegis: purchases['product_id'].nunique())
    unique_products = purchases["product_id"].nunique()

    # ── Rolling windows (relative to observation_date) ────────────────────
    w30_start = observation_date - pd.Timedelta(days=30)
    w90_start = observation_date - pd.Timedelta(days=90)

    hist_30d = history[history["transaction_timestamp"] > w30_start]
    hist_90d = history[history["transaction_timestamp"] > w90_start]

    net_spend_30d = hist_30d["net_amount"].sum()
    net_spend_90d = hist_90d["net_amount"].sum()

    po_30d = int(hist_30d[hist_30d["transaction_type"] == "purchase"]["invoice"].nunique())
    po_90d = int(hist_90d[hist_90d["transaction_type"] == "purchase"]["invoice"].nunique())

    is_active_30d = 1 if po_30d > 0 else 0
    is_active_90d = 1 if po_90d > 0 else 0

    # ── Assemble feature row ──────────────────────────────────────────────
    feature_values = {
        "FEATURE_recency_days":              float(recency_days),
        "FEATURE_customer_lifetime_days":    float(lifetime_days),
        "FEATURE_purchase_order_count":      float(purchase_order_count),
        "FEATURE_purchase_line_count":       float(purchase_line_count),
        "FEATURE_return_line_count":         float(return_line_count),
        "FEATURE_cancellation_line_count":   float(cancel_line_count),
        "FEATURE_total_gross_purchase_spend": float(gross_spend),
        "FEATURE_total_return_amount":       float(return_amount),
        "FEATURE_total_net_spend":           float(net_spend),
        "FEATURE_average_order_value":       float(aov),
        "FEATURE_purchase_frequency":        float(freq),
        "FEATURE_return_rate":               float(return_rate),
        "FEATURE_unique_products_purchased": float(unique_products),
        "FEATURE_net_spend_30d":             float(net_spend_30d),
        "FEATURE_net_spend_90d":             float(net_spend_90d),
        "FEATURE_purchase_order_count_30d":  float(po_30d),
        "FEATURE_purchase_order_count_90d":  float(po_90d),
        "FEATURE_is_active_30d":             float(is_active_30d),
        "FEATURE_is_active_90d":             float(is_active_90d),
    }

    # Return as single-row DataFrame with features in exact training order
    row_df = pd.DataFrame([feature_values])[PROPENSITY_FEATURES]
    return row_df


def validate_feature_vector(feature_df: pd.DataFrame) -> None:
    """
    Validates that a feature DataFrame matches the model's expected contract.
    Raises ValueError if any issue is detected.
    """
    if list(feature_df.columns) != PROPENSITY_FEATURES:
        raise ValueError(
            f"Feature mismatch. Expected {PROPENSITY_FEATURES}, got {list(feature_df.columns)}"
        )
    if feature_df.shape[0] != 1:
        raise ValueError(f"Expected a single-row feature DataFrame, got {feature_df.shape[0]} rows.")

    for col in PROPENSITY_FEATURES:
        val = feature_df[col].iloc[0]
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"Feature {col} must be numeric, got {type(val)}")
