import json
import os
from typing import Dict, Any, Optional

class PolicyGate:
    """
    Deterministic action policy engine.
    1. Pre-evaluates customer state to provide a baseline recommendation (like Aegis).
    2. Validates LLM proposals against strict financial/business rules.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config/policy_rules.json in Aegis root
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "policy_rules.json")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.signals_config = self.config.get("signals", {})
        self.rules = self.config.get("rules", [])
        self.confidence_rules = self.config.get("confidence_rules", {})
        self.limits = self.config.get("limits", {})
        
        # Sort rules by priority descending
        self.rules = sorted(self.rules, key=lambda x: x.get("priority", 0), reverse=True)

    def evaluate_pre_action(self, segment: str, propensity_band: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate customer state against loaded rules and return the highest priority match.
        Used to build the intelligence object for the LLM.
        """
        for rule in self.rules:
            if self._matches(rule, segment, propensity_band, signals):
                return {
                    "recommendation": rule["recommendation"],
                    "policy_rule_id": rule["rule_id"],
                    "priority": rule.get("priority", 0)
                }
        return {
            "recommendation": "Standard engagement",
            "policy_rule_id": "FALLBACK_00",
            "priority": -1
        }

    def _matches(self, rule: Dict[str, Any], segment: str, propensity_band: str, signals: Dict[str, Any]) -> bool:
        conditions = rule.get("conditions", {})
        if not conditions:
            return True
            
        if "segment" in conditions and conditions["segment"] != segment:
            return False
            
        if "propensity_bands" in conditions and propensity_band not in conditions["propensity_bands"]:
            return False
            
        if "signals" in conditions:
            for sig_name, required_val in conditions["signals"].items():
                if signals.get(sig_name, {}).get("active") != required_val:
                    return False
        return True
        
    def get_confidence(self, matched_rule_id: str, signals: Dict[str, Any], history_days: float) -> str:
        if matched_rule_id in ["DEFAULT_01", "FALLBACK_00"]:
            return "Low"
            
        high_cfg = self.confidence_rules.get("HIGH", {})
        if history_days >= high_cfg.get("min_history_days", 90):
            return "High"
            
        med_cfg = self.confidence_rules.get("MEDIUM", {})
        if history_days >= med_cfg.get("min_history_days", 30):
            return "Medium"
            
        return "Low"

    def authorize_action(self, proposal: Dict[str, Any], intelligence: Dict[str, Any],
                         amount_inr: Optional[float] = None,
                         campaign: Optional[Dict[str, Any]] = None,
                         velocity: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validates an LLM's action proposal against deterministic financial rules.
        amount_inr: final payable amount (post-discount); enforced vs MAX_PAYMENT_AMOUNT_INR.
        campaign:   the SERVER-selected campaign dict (1C). When provided, the per-campaign
                    discount cap and segment/band eligibility are enforced — the LLM never
                    selects the campaign, so it cannot steer the payable amount.
        velocity:   {"action_count", "spend_inr", "hours_since_last"} for this customer over
                    the trailing window (1D); enforced vs the velocity/budget/cooldown limits.
        Returns {"approved": bool, "reason": str}
        """
        # Ensure we have a valid action type
        action_type = proposal.get("action_type", "")
        if not action_type:
            return {"approved": False, "reason": "MISSING_ACTION_TYPE"}

        discount_pct = proposal.get("requested_discount_pct", 0)

        # A negative "discount" inflates the charge ABOVE base (final = base*(1 - d/100)).
        # The LLM proposal is untrusted, so reject it deterministically here — otherwise a
        # prompt-injected -100% would double the payable amount up to the ₹ ceiling (defeats 1C).
        if discount_pct < 0:
            return {"approved": False, "reason": "INVALID_DISCOUNT (negative)"}

        # Check MAX_DISCOUNT
        max_discount = self.limits.get("MAX_DISCOUNT_PCT", 0)
        if discount_pct > max_discount:
            return {"approved": False, "reason": f"DISCOUNT_LIMIT_EXCEEDED (Max {max_discount}%)"}

        # 1C: per-campaign discount cap + eligibility (defense-in-depth on the server-chosen campaign)
        if campaign is not None:
            camp_max = campaign.get("max_discount_pct")
            if camp_max is not None and discount_pct > camp_max:
                return {"approved": False, "reason": f"CAMPAIGN_DISCOUNT_LIMIT_EXCEEDED (Max {camp_max}%)"}
            seg = intelligence.get("segment")
            elig_segs = campaign.get("eligible_segments")
            if elig_segs is not None and seg not in elig_segs:
                return {"approved": False, "reason": f"SEGMENT_NOT_ELIGIBLE ({seg})"}
            band = intelligence.get("propensity_band")
            elig_bands = campaign.get("eligible_propensity_bands")
            if elig_bands is not None and band not in elig_bands:
                return {"approved": False, "reason": f"PROPENSITY_BAND_NOT_ELIGIBLE ({band})"}

        # Check MAX_PAYMENT_AMOUNT (the campaign controls base amount; cap the charge)
        max_amount = self.limits.get("MAX_PAYMENT_AMOUNT_INR")
        if amount_inr is not None and max_amount is not None and amount_inr > max_amount:
            return {"approved": False, "reason": f"AMOUNT_LIMIT_EXCEEDED (₹{amount_inr:.0f} > Max ₹{max_amount})"}

        # Check confidence threshold if required by limits
        req_confidence = self.limits.get("REQUIRED_CONFIDENCE", "Low")
        conf_levels = {"Low": 1, "Medium": 2, "High": 3}
        intel_conf = intelligence.get("recommendation_confidence", "Low")

        if conf_levels.get(intel_conf, 0) < conf_levels.get(req_confidence, 0):
            return {"approved": False, "reason": f"INSUFFICIENT_CONFIDENCE (Required {req_confidence})"}

        # 1D: velocity / budget / cooldown — enforced vs prior actions for this customer
        if velocity is not None:
            window = self.limits.get("ACTION_WINDOW_DAYS", 30)
            max_actions = self.limits.get("MAX_ACTIONS_PER_CUSTOMER_WINDOW")
            if max_actions is not None and velocity.get("action_count", 0) >= max_actions:
                return {"approved": False,
                        "reason": f"VELOCITY_LIMIT_EXCEEDED ({velocity.get('action_count')} in {window}d, max {max_actions})"}

            max_spend = self.limits.get("MAX_SPEND_PER_CUSTOMER_WINDOW_INR")
            if max_spend is not None and amount_inr is not None:
                projected = velocity.get("spend_inr", 0) + amount_inr
                if projected > max_spend:
                    return {"approved": False,
                            "reason": f"BUDGET_LIMIT_EXCEEDED (₹{projected:.0f} > Max ₹{max_spend} in {window}d)"}

            cooldown_h = self.limits.get("COOLDOWN_HOURS")
            hsl = velocity.get("hours_since_last")
            if cooldown_h is not None and hsl is not None and hsl < cooldown_h:
                return {"approved": False, "reason": f"COOLDOWN_ACTIVE ({hsl:.1f}h since last < {cooldown_h}h)"}

        return {"approved": True, "reason": "APPROVED"}
