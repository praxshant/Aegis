import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.policy.gate import PolicyGate

def test_policy_authorization():
    gate = PolicyGate()
    
    # 1. Valid proposal
    proposal = {
        "action_type": "CREATE_PAYMENT_LINK",
        "requested_discount_pct": 20.0
    }
    intelligence = {
        "recommendation_confidence": "Medium"
    }
    
    result = gate.authorize_action(proposal, intelligence)
    assert result["approved"] is True
    
    # 2. Exceeds max discount
    proposal_high_discount = {
        "action_type": "CREATE_PAYMENT_LINK",
        "requested_discount_pct": 50.0
    }
    result_hd = gate.authorize_action(proposal_high_discount, intelligence)
    assert result_hd["approved"] is False
    assert "DISCOUNT_LIMIT_EXCEEDED" in result_hd["reason"]
    
    # 3. Insufficient confidence
    proposal_low_conf = {
        "action_type": "CREATE_PAYMENT_LINK",
        "requested_discount_pct": 20.0
    }
    intelligence_low_conf = {
        "recommendation_confidence": "Low"
    }
    result_lc = gate.authorize_action(proposal_low_conf, intelligence_low_conf)
    assert result_lc["approved"] is False
    assert "INSUFFICIENT_CONFIDENCE" in result_lc["reason"]

    # 4. Exceeds max payment amount (ceiling enforced only when amount_inr is passed)
    ok_proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 20.0}
    over = gate.limits.get("MAX_PAYMENT_AMOUNT_INR", 2000) + 1
    result_amt = gate.authorize_action(ok_proposal, intelligence, amount_inr=over)
    assert result_amt["approved"] is False
    assert "AMOUNT_LIMIT_EXCEEDED" in result_amt["reason"]
    # Within ceiling still approves
    assert gate.authorize_action(ok_proposal, intelligence, amount_inr=100)["approved"] is True


# --- 1C: server-selected campaign — per-campaign discount cap + eligibility (defense-in-depth) ---

def _campaign(**over):
    c = {
        "base_amount_inr": 1000,
        "max_discount_pct": 10,
        "eligible_segments": ["Dormant Multi-Buyers"],
        "eligible_propensity_bands": ["High", "Medium"],
    }
    c.update(over)
    return c

def test_campaign_discount_cap():
    gate = PolicyGate()
    # 20% is under the global 25% but over this campaign's 10% cap.
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 20.0}
    intel = {"recommendation_confidence": "Medium", "segment": "Dormant Multi-Buyers", "propensity_band": "High"}
    res = gate.authorize_action(proposal, intel, amount_inr=800, campaign=_campaign())
    assert res["approved"] is False
    assert "CAMPAIGN_DISCOUNT_LIMIT_EXCEEDED" in res["reason"]

def test_campaign_segment_and_band_eligibility():
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 5.0}
    # Wrong segment
    intel_seg = {"recommendation_confidence": "Medium", "segment": "Active High-Value", "propensity_band": "High"}
    res_seg = gate.authorize_action(proposal, intel_seg, amount_inr=950, campaign=_campaign())
    assert res_seg["approved"] is False
    assert "SEGMENT_NOT_ELIGIBLE" in res_seg["reason"]
    # Right segment, wrong band
    intel_band = {"recommendation_confidence": "Medium", "segment": "Dormant Multi-Buyers", "propensity_band": "Low"}
    res_band = gate.authorize_action(proposal, intel_band, amount_inr=950, campaign=_campaign())
    assert res_band["approved"] is False
    assert "PROPENSITY_BAND_NOT_ELIGIBLE" in res_band["reason"]
    # Eligible on both → approved
    intel_ok = {"recommendation_confidence": "Medium", "segment": "Dormant Multi-Buyers", "propensity_band": "High"}
    assert gate.authorize_action(proposal, intel_ok, amount_inr=950, campaign=_campaign())["approved"] is True


# --- 1D: velocity / budget / cooldown ---

def test_velocity_limit():
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 10.0}
    intel = {"recommendation_confidence": "Medium"}
    max_actions = gate.limits["MAX_ACTIONS_PER_CUSTOMER_WINDOW"]
    velocity = {"action_count": max_actions, "spend_inr": 0.0, "hours_since_last": 100.0}
    res = gate.authorize_action(proposal, intel, amount_inr=100, velocity=velocity)
    assert res["approved"] is False
    assert "VELOCITY_LIMIT_EXCEEDED" in res["reason"]

def test_budget_limit():
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 10.0}
    intel = {"recommendation_confidence": "Medium"}
    max_spend = gate.limits["MAX_SPEND_PER_CUSTOMER_WINDOW_INR"]
    # prior spend just under the cap; this action tips it over
    velocity = {"action_count": 0, "spend_inr": max_spend - 100, "hours_since_last": 100.0}
    res = gate.authorize_action(proposal, intel, amount_inr=200, velocity=velocity)
    assert res["approved"] is False
    assert "BUDGET_LIMIT_EXCEEDED" in res["reason"]

def test_cooldown_active():
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 10.0}
    intel = {"recommendation_confidence": "Medium"}
    cooldown = gate.limits["COOLDOWN_HOURS"]
    velocity = {"action_count": 0, "spend_inr": 0.0, "hours_since_last": cooldown - 1}
    res = gate.authorize_action(proposal, intel, amount_inr=100, velocity=velocity)
    assert res["approved"] is False
    assert "COOLDOWN_ACTIVE" in res["reason"]

def test_velocity_within_limits_approves():
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": 10.0}
    intel = {"recommendation_confidence": "Medium"}
    velocity = {"action_count": 0, "spend_inr": 0.0, "hours_since_last": None}  # no prior action
    assert gate.authorize_action(proposal, intel, amount_inr=100, velocity=velocity)["approved"] is True


def test_negative_discount_rejected():
    # A negative discount inflates the charge above base (final = base*(1 - d/100)).
    # An untrusted/prompt-injected -100% must be denied, not silently applied.
    gate = PolicyGate()
    proposal = {"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": -100.0}
    intel = {"recommendation_confidence": "Medium"}
    res = gate.authorize_action(proposal, intel, amount_inr=100)
    assert res["approved"] is False
    assert "INVALID_DISCOUNT" in res["reason"]


