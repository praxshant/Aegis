import pytest
import pandas as pd
from backend.intelligence.engine import IntelligenceEngine
from backend.policy.gate import PolicyGate
from backend.intelligence.schemas import IntelligenceStatus

def test_intelligence_complete():
    ie = IntelligenceEngine(PolicyGate())
    
    txns = pd.DataFrame([
        {
            "customer_id": "C001",
            "transaction_timestamp": pd.Timestamp.now() - pd.Timedelta(days=10),
            "transaction_type": "purchase",
            "invoice": "INV1",
            "product_id": "P1",
            "gross_amount": 100.0,
            "return_amount": 0.0,
            "net_amount": 100.0,
        }
    ])
    
    intel = ie.get_intelligence("C001", txns)
    
    assert intel.status == IntelligenceStatus.COMPLETE
    assert intel.purchase_probability_30d is not None
    assert intel.propensity_band is not None
    assert intel.propensity_source == "model_inference"

def test_intelligence_incomplete_no_history():
    ie = IntelligenceEngine(PolicyGate())
    
    txns = pd.DataFrame() # Empty txns
    
    intel = ie.get_intelligence("C001", txns)
    
    assert intel.status == IntelligenceStatus.INCOMPLETE
    assert intel.incomplete_reason is not None
    assert intel.purchase_probability_30d is None
