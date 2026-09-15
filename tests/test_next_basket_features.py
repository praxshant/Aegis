import pytest
import pandas as pd
from backend.models.next_basket_features import assemble_features_for_next_basket, NEXT_BASKET_FEATURES

def test_feature_assembly_unavailable_no_catalog():
    tx_df = pd.DataFrame([
        {"transaction_timestamp": "2026-08-01", "transaction_type": "purchase", "invoice": "I1", "product_id": "P1"},
        {"transaction_timestamp": "2026-08-05", "transaction_type": "purchase", "invoice": "I2", "product_id": "P1"}
    ])
    status, frame, reason = assemble_features_for_next_basket("C1", tx_df, product_catalog=None)
    assert status == "UNAVAILABLE"
    assert "product_catalog is required" in reason

def test_feature_assembly_insufficient_history():
    tx_df = pd.DataFrame([
        {"transaction_timestamp": "2026-08-01", "transaction_type": "purchase", "invoice": "I1", "product_id": "P1"}
    ])
    catalog = pd.DataFrame([
        {"aegis_product_id": "P1", "instacart_product_id": 1, "product_name": "N1", "aisle_id": 1, "department_id": 1}
    ])
    status, frame, reason = assemble_features_for_next_basket("C1", tx_df, product_catalog=catalog)
    assert status == "UNAVAILABLE"
    assert "Insufficient order history" in reason

def test_feature_assembly_success():
    tx_df = pd.DataFrame([
        {"transaction_timestamp": "2026-08-01", "transaction_type": "purchase", "invoice": "I1", "product_id": "P1"},
        {"transaction_timestamp": "2026-08-05", "transaction_type": "purchase", "invoice": "I2", "product_id": "P2"}
    ])
    catalog = pd.DataFrame([
        {"aegis_product_id": "P1", "instacart_product_id": 1, "product_name": "N1", "aisle_id": 1, "department_id": 1},
        {"aegis_product_id": "P2", "instacart_product_id": 2, "product_name": "N2", "aisle_id": 1, "department_id": 1}
    ])
    status, frame, reason = assemble_features_for_next_basket("C1", tx_df, product_catalog=catalog)
    assert status == "COMPLETE"
    assert frame is not None
    assert all(c in frame.columns for c in NEXT_BASKET_FEATURES)
