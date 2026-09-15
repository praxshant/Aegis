import pytest
import pandas as pd
from pathlib import Path
from backend.intelligence.engine import IntelligenceEngine
from backend.data.database_manager import DatabaseManager

def test_engine_next_basket_integration():
    db = DatabaseManager()
    
    # We test on CUST_002 (Bob) who has plenty of history
    tx_df = db.query_to_dataframe("SELECT * FROM transactions WHERE customer_id = ?", ("CUST_002",))
    if tx_df.empty:
        pytest.skip("Test database not initialized with synthetic data.")
        
    ie = IntelligenceEngine()
    intel = ie.get_intelligence("CUST_002", tx_df, segment="Dormant Multi-Buyers")
    
    assert intel.status == "COMPLETE"
    assert intel.next_basket is not None
    
    nb = intel.next_basket
    # If the catalog doesn't exist, it will be UNAVAILABLE, but not an error
    assert nb.status in ["COMPLETE", "UNAVAILABLE"]
    
    # If complete, verify schema
    if nb.status == "COMPLETE":
        assert nb.available is True
        assert len(nb.top_products) > 0
        assert hasattr(nb.top_products[0], "probability")
        assert hasattr(nb.top_products[0], "product_name")
