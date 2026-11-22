"""Regression tests for the Nov-2026 bug sweep: schema migration, ingestion semantics,
segment derivation, the hash-chained audit ledger, and policy/auth edge cases."""
import os
import sqlite3
import hmac
import hashlib

import pandas as pd
import pytest

from backend.data.database_manager import DatabaseManager, resolve_db_path, PROJECT_ROOT
from backend.audit.ledger import AuditLedger
from backend.policy.gate import PolicyGate
from backend.razorpay.adapter import verify_webhook_signature, MockRazorpayAdapter
from backend.intelligence.engine import derive_segment
from backend.auth.auth_manager import AuthManager, LoginThrottle


# ── Data layer ──────────────────────────────────────────────────────────────
def test_explicit_path_beats_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///somewhere/else.db")
    assert resolve_db_path(":memory:") == ":memory:"
    assert resolve_db_path() == str(PROJECT_ROOT / "somewhere" / "else.db")  # relative → project root


def test_legacy_table_is_migrated(tmp_path):
    """A customers table re-created by pandas to_sql(replace) lacks status & co.;
    startup must add them so GET /customers doesn't fail with 'no such column'."""
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE customers (customer_id TEXT, first_name TEXT)")
        conn.execute("INSERT INTO customers VALUES ('C1', 'Ann')")
    db = DatabaseManager(str(path))
    cols = set(db.query_to_dataframe("PRAGMA table_info(customers)")["name"])
    assert {"status", "ingestion_run_id", "opted_out", "propensity_score", "propensity_band"} <= cols
    assert db.query_to_dataframe("SELECT COALESCE(status,'active') AS s FROM customers").iloc[0, 0] == "active"


def test_insert_dataframe_replace_keeps_schema():
    db = DatabaseManager(":memory:")
    db.insert_dataframe(pd.DataFrame([{"customer_id": "A"}]), "customers")
    db.insert_dataframe(pd.DataFrame([{"customer_id": "B"}]), "customers", if_exists="replace")
    df = db.query_to_dataframe("SELECT customer_id, status FROM customers")
    assert df["customer_id"].tolist() == ["B"] and df["status"].tolist() == ["active"]


# ── Ingestion ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ingestion():
    from backend.ingestion.ingestion_service import IngestionService
    db = DatabaseManager(":memory:")
    return IngestionService(db), db


def _uci_frame():
    now = pd.Timestamp.now().normalize()
    rows = [
        {"Invoice": "5001", "StockCode": "85123A", "Quantity": "6", "InvoiceDate": (now - pd.Timedelta(days=200)).strftime("%m/%d/%Y %H:%M"),
         "Price": "2.55", "Customer ID": "13085.0"},
        {"Invoice": "5002", "StockCode": "22423", "Quantity": "2", "InvoiceDate": (now - pd.Timedelta(days=100)).strftime("%m/%d/%Y %H:%M"),
         "Price": "10.00", "Customer ID": "13085.0"},
        {"Invoice": "C5003", "StockCode": "22423", "Quantity": "-1", "InvoiceDate": (now - pd.Timedelta(days=90)).strftime("%m/%d/%Y %H:%M"),
         "Price": "10.00", "Customer ID": "13085.0"},
        {"Invoice": "5004", "StockCode": "22423", "Quantity": "-2", "InvoiceDate": (now - pd.Timedelta(days=80)).strftime("%m/%d/%Y %H:%M"),
         "Price": "10.00", "Customer ID": "13085.0"},
        {"Invoice": "5005", "StockCode": "", "Quantity": "1", "InvoiceDate": "not a date", "Price": "1", "Customer ID": ""},
    ]
    return pd.DataFrame(rows, dtype=str)


def test_ingestion_follows_training_semantics(ingestion):
    svc, db = ingestion
    run_id = svc.start_ingestion_run("uci.csv", 5)
    svc.process_chunk(run_id, _uci_frame())
    tx = db.query_to_dataframe(
        "SELECT invoice, transaction_type, gross_amount, return_amount, net_amount FROM transactions "
        "WHERE customer_id = '13085' ORDER BY invoice")
    assert len(tx) == 4  # numeric id normalised: '13085.0' → '13085'
    by_inv = tx.set_index("invoice")
    assert by_inv.loc["5001", "gross_amount"] == pytest.approx(15.3)      # qty × unit price
    assert by_inv.loc["C5003", "transaction_type"] == "cancellation"
    assert by_inv.loc["C5003", "return_amount"] == pytest.approx(10.0)
    assert by_inv.loc["5004", "transaction_type"] == "return"
    assert by_inv.loc["5004", "net_amount"] == pytest.approx(-20.0)
    run = db.query_to_dataframe("SELECT valid_rows, invalid_rows FROM ingestion_runs WHERE run_id=?", (run_id,)).iloc[0]
    assert run["valid_rows"] == 4 and run["invalid_rows"] == 1

    # Re-uploading the same rows is idempotent, and the customer now belongs to the new run
    run2 = svc.start_ingestion_run("uci.csv", 5)
    svc.process_chunk(run2, _uci_frame())
    assert len(db.query_to_dataframe("SELECT 1 FROM transactions WHERE customer_id='13085'")) == 4
    owner = db.query_to_dataframe("SELECT ingestion_run_id FROM customers WHERE customer_id='13085'").iloc[0, 0]
    assert owner == run2


def test_batch_intelligence_assigns_a_real_segment(ingestion):
    svc, db = ingestion
    run_id = svc.start_ingestion_run("uci.csv", 5)
    svc.process_chunk(run_id, _uci_frame())
    assert svc.generate_batch_intelligence(run_id) == 1
    seg = db.query_to_dataframe("SELECT segment_name FROM customer_segments WHERE customer_id='13085'")
    assert seg.iloc[0, 0] == "Dormant Multi-Buyers"
    assert db.query_to_dataframe("SELECT propensity_band FROM customers WHERE customer_id='13085'").iloc[0, 0]


def test_duplicate_alias_columns_do_not_crash(ingestion):
    svc, _ = ingestion
    df = pd.DataFrame([{"order_id": "1", "invoice": "9", "customer_id": "A", "sku": "P", "date": "2026-01-01",
                        "order_date": "2026-01-02", "amount": "5"}])
    out = svc.detect_schema_and_rename(df)
    assert list(out.columns).count("invoice") == 1
    assert list(out.columns).count("transaction_timestamp") == 1


@pytest.mark.parametrize("orders,active,expected", [
    (7, 1, "Active High-Value"), (3, 1, "Recent Repeat Buyers"), (1, 1, "New Customers"),
    (8, 0, "Dormant Multi-Buyers"), (1, 0, "One-Time Dormant"),
])
def test_derive_segment(orders, active, expected):
    row = pd.Series({"FEATURE_purchase_order_count": orders, "FEATURE_is_active_90d": active})
    assert derive_segment(row) == expected


# ── Audit ledger ────────────────────────────────────────────────────────────
def test_audit_chain_detects_tampering(tmp_path):
    db = DatabaseManager(str(tmp_path / "audit.db"))
    ledger = AuditLedger(db)
    for i in range(3):
        assert ledger.record_event(f"act_{i}", "C1", "EVT", "tester", "SUCCESS", {"i": i, "n": pd.NA})
    assert ledger.verify_chain()["valid"] and ledger.verify_chain()["checked"] == 3

    with pytest.raises(Exception):  # append-only at the storage layer
        db.execute_query("UPDATE audit_ledger SET status='FORGED'")

    with sqlite3.connect(db.db_path) as conn:  # an attacker with raw file access drops the guard
        conn.execute("DROP TRIGGER audit_ledger_no_update")
        conn.execute("UPDATE audit_ledger SET status='FORGED' WHERE action_id='act_1'")
    result = ledger.verify_chain()
    assert not result["valid"] and result["checked"] == 1


# ── Policy / payments ───────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc"])
def test_gate_rejects_non_numeric_discount(bad):
    res = PolicyGate().authorize_action({"action_type": "CREATE_PAYMENT_LINK", "requested_discount_pct": bad},
                                        {"recommendation_confidence": "High"})
    assert not res["approved"] and res["reason"].startswith("INVALID_DISCOUNT")


def test_webhook_signature():
    body, secret = b'{"event":"payment_link.paid"}', "whsec"
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(body, good, secret)
    assert not verify_webhook_signature(body, "0" * 64, secret)
    assert not verify_webhook_signature(body, good, "")


def test_mock_link_description_is_clipped():
    link = MockRazorpayAdapter().create_payment_link(100, "C1", "act_1", "x" * 5000)
    assert len(link["description"]) <= 250


def test_live_key_requires_explicit_opt_in(monkeypatch):
    pytest.importorskip("razorpay")
    from backend.razorpay.adapter import get_razorpay_adapter
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abc")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "s")
    monkeypatch.delenv("RAZORPAY_ENV", raising=False)
    assert get_razorpay_adapter().mode == "mock"


# ── Auth ────────────────────────────────────────────────────────────────────
def test_overlong_password_is_a_failed_login_not_a_crash():
    am = AuthManager(db_manager=DatabaseManager(":memory:"))
    h = am.hash_password("short-pass")
    assert am.verify_password("x" * 100, h) is False
    with pytest.raises(ValueError):
        am.hash_password("x" * 100)


def test_login_throttle():
    t = LoginThrottle(max_failures=2, window_s=60)
    t.check("u")
    t.fail("u"); t.fail("u")
    with pytest.raises(Exception) as exc:
        t.check("u")
    assert getattr(exc.value, "status_code", None) == 429
    t.reset("u")
    t.check("u")
