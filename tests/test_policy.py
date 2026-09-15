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
