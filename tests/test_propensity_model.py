import pytest
import pandas as pd
from backend.models.propensity_model import NextPurchasePropensityModel, PropensityModelError, get_propensity_model
from backend.models.propensity_features import PROPENSITY_FEATURES

def test_propensity_model_singleton_loads(monkeypatch):
    model = get_propensity_model()
    assert model._loaded is True
    assert model._hash_verified is True
    
def test_propensity_model_inference():
    model = get_propensity_model()
    
    # Create valid feature row
    dummy_features = {f: 0.0 for f in PROPENSITY_FEATURES}
    dummy_features["FEATURE_purchase_order_count"] = 5.0
    dummy_features["FEATURE_recency_days"] = 30.0
    dummy_features["FEATURE_customer_lifetime_days"] = 365.0
    dummy_features["FEATURE_total_net_spend"] = 1000.0
    
    df = pd.DataFrame([dummy_features])[PROPENSITY_FEATURES]
    
    score, band = model.predict_band(df)
    
    assert 0.0 <= score <= 1.0
    assert band in ["1_LOW", "2_MODERATE", "3_HIGH", "4_VERY_HIGH"]

def test_propensity_model_invalid_features():
    model = get_propensity_model()
    
    # Missing a feature
    df = pd.DataFrame([{"FEATURE_recency_days": 10.0}])
    
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict_score(df)
