"""WS1-1D: get_customer_velocity must count only status='EXECUTED' actions.
A FAILED attempt created no live charge, and an in-flight EXECUTION_REQUESTED row
must not burn the customer's budget/cooldown — otherwise a failed retry could block
a legitimate offer, or a failure could count against the ₹ spend cap."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.database_manager import DatabaseManager
from backend.reconciliation.reconciler import Reconciler


def _insert(db, action_id, status, amount):
    db.execute_query(
        "INSERT INTO action_execution "
        "(action_id, customer_id, action_type, requested_discount_pct, amount_inr, status, policy_rule_id, reconciliation_status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (action_id, "CUST_X", "CREATE_PAYMENT_LINK", 10.0, amount, status, "WINBACK_01", "PENDING"),
    )


def test_velocity_counts_executed_only():
    db = DatabaseManager(":memory:")
    recon = Reconciler(db, razorpay_adapter=None)  # velocity path never touches the adapter

    _insert(db, "act_ok1", "EXECUTED", 500.0)
    _insert(db, "act_ok2", "EXECUTED", 300.0)
    _insert(db, "act_fail", "FAILED", 1000.0)          # no live charge — must not count
    _insert(db, "act_inflight", "EXECUTION_REQUESTED", 700.0)  # in-flight — must not count

    v = recon.get_customer_velocity("CUST_X", window_days=30)
    assert v["action_count"] == 2, v
    assert v["spend_inr"] == 800.0, v            # 500 + 300 only; FAILED/in-flight excluded
    assert v["hours_since_last"] is not None     # a prior EXECUTED action exists


def test_velocity_empty_for_unknown_customer():
    db = DatabaseManager(":memory:")
    recon = Reconciler(db, razorpay_adapter=None)
    v = recon.get_customer_velocity("NOBODY", window_days=30)
    assert v["action_count"] == 0
    assert v["spend_inr"] == 0.0
    assert v["hours_since_last"] is None


if __name__ == "__main__":
    test_velocity_counts_executed_only()
    test_velocity_empty_for_unknown_customer()
    print("reconciler velocity checks passed")
