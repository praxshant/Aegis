import pytest
import pandas as pd
from backend.models.propensity_features import build_features_for_customer, PROPENSITY_FEATURES, FeatureBuildError

def test_feature_contract_order_and_types():
    txns = pd.DataFrame([
        {
            "customer_id": "C001",
            "transaction_timestamp": pd.Timestamp("2024-01-01"),
            "transaction_type": "purchase",
            "invoice": "INV1",
            "product_id": "P1",
            "gross_amount": 100.0,
            "return_amount": 0.0,
            "net_amount": 100.0,
        }
    ])
    
    obs_date = pd.Timestamp("2024-02-01")
    features = build_features_for_customer("C001", txns, obs_date)
    
    assert list(features.columns) == PROPENSITY_FEATURES
    assert features.shape == (1, 19)

def test_observation_date_leakage_prevention():
    txns = pd.DataFrame([
        {
            "customer_id": "C001",
            "transaction_timestamp": pd.Timestamp("2024-01-01"), # Past
            "transaction_type": "purchase",
            "invoice": "INV1",
            "product_id": "P1",
            "gross_amount": 100.0,
            "return_amount": 0.0,
            "net_amount": 100.0,
        },
        {
            "customer_id": "C001",
            "transaction_timestamp": pd.Timestamp("2024-03-01"), # Future
            "transaction_type": "purchase",
            "invoice": "INV2",
            "product_id": "P2",
            "gross_amount": 500.0,
            "return_amount": 0.0,
            "net_amount": 500.0,
        }
    ])
    
    # Obs date before the second transaction
    obs_date = pd.Timestamp("2024-02-01")
    features = build_features_for_customer("C001", txns, obs_date)
    
    # Future transaction should be ignored
    assert features["FEATURE_total_gross_purchase_spend"].iloc[0] == 100.0
    assert features["FEATURE_purchase_order_count"].iloc[0] == 1.0

def test_missing_purchase_history():
    txns = pd.DataFrame([
        {
            "customer_id": "C001",
            "transaction_timestamp": pd.Timestamp("2024-01-01"),
            "transaction_type": "return", # Only a return, no purchases
            "invoice": "INV1",
            "product_id": "P1",
            "gross_amount": 0.0,
            "return_amount": 50.0,
            "net_amount": -50.0,
        }
    ])
    
    obs_date = pd.Timestamp("2024-02-01")
    
    with pytest.raises(FeatureBuildError, match="has no purchase history"):
        build_features_for_customer("C001", txns, obs_date)
