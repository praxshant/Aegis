from backend.intelligence.engine import IntelligenceEngine
from backend.data.database_manager import DatabaseManager


def test_engine_next_basket_integration(tmp_path, monkeypatch):
    # Seed the synthetic fixtures into a throwaway DB (this test used to read the real
    # backend/data/aegis.db and silently skip whenever it hadn't been seeded — i.e. in CI).
    db_path = str(tmp_path / "nb.db")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + db_path)
    from tests.fixtures.synthetic_dev_data import generate_fixtures
    generate_fixtures()  # also writes the Instacart catalog mapping the model needs

    db = DatabaseManager(db_path)
    # We test on CUST_002 (Bob) who has plenty of history
    tx_df = db.query_to_dataframe("SELECT * FROM transactions WHERE customer_id = ?", ("CUST_002",))
    assert not tx_df.empty

    ie = IntelligenceEngine()
    intel = ie.get_intelligence("CUST_002", tx_df, segment="Dormant Multi-Buyers")

    assert intel.status == "COMPLETE"
    assert intel.policy_rule_id == "WINBACK_01"
    nb = intel.next_basket
    assert nb is not None and nb.status == "COMPLETE" and nb.available is True
    assert len(nb.top_products) > 0
    assert 0.0 <= nb.top_products[0].probability <= 1.0
    assert nb.top_products[0].product_name
