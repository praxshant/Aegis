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

    def authorize_action(self, proposal: Dict[str, Any], intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates an LLM's action proposal.
        Returns {"approved": bool, "reason": str}
        """
        # Ensure we have a valid action type
        action_type = proposal.get("action_type", "")
        if not action_type:
            return {"approved": False, "reason": "MISSING_ACTION_TYPE"}

        discount_pct = proposal.get("requested_discount_pct", 0)
        
        # Check MAX_DISCOUNT
        max_discount = self.limits.get("MAX_DISCOUNT_PCT", 0)
        if discount_pct > max_discount:
            return {"approved": False, "reason": f"DISCOUNT_LIMIT_EXCEEDED (Max {max_discount}%)"}
            
        # Check confidence threshold if required by limits
        req_confidence = self.limits.get("REQUIRED_CONFIDENCE", "Low")
        conf_levels = {"Low": 1, "Medium": 2, "High": 3}
        intel_conf = intelligence.get("recommendation_confidence", "Low")
        
        if conf_levels.get(intel_conf, 0) < conf_levels.get(req_confidence, 0):
            return {"approved": False, "reason": f"INSUFFICIENT_CONFIDENCE (Required {req_confidence})"}
            
        return {"approved": True, "reason": "APPROVED"}
