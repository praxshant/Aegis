import logging
from typing import Dict, Any, List
import datetime
import pandas as pd
from backend.data.database_manager import DatabaseManager
from backend.razorpay.adapter import RazorpayAdapter

logger = logging.getLogger("Reconciler")

class Reconciler:
    """
    Handles safe state transitions for actions and reconciles unknown statuses
    with the Razorpay execution layer.
    """
    def __init__(self, db: DatabaseManager, razorpay_adapter: RazorpayAdapter):
        self.db = db
        self.rzp = razorpay_adapter

    def record_execution_request(self, action_id: str, customer_id: str, action_type: str,
                                 discount_pct: float, policy_rule_id: str, amount_inr: float = None):
        query = """
            INSERT INTO action_execution (action_id, customer_id, action_type, requested_discount_pct, amount_inr, status, policy_rule_id, reconciliation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (action_id, customer_id, action_type, discount_pct, amount_inr, "EXECUTION_REQUESTED", policy_rule_id, "PENDING")
        self.db.execute_query(query, params)

    def get_customer_velocity(self, customer_id: str, window_days: int) -> Dict[str, Any]:
        """Prior-action stats for velocity/budget/cooldown gating (1D):
        action_count + spend_inr over the trailing window, and hours since the last action.
        Counts only status='EXECUTED' rows — a FAILED link created no live charge, and an
        in-flight EXECUTION_REQUESTED row must not burn the customer's budget/cooldown."""
        win = self.db.query_to_dataframe(
            """
            SELECT COUNT(*) AS action_count, COALESCE(SUM(amount_inr), 0) AS spend_inr
            FROM action_execution
            WHERE customer_id = ? AND status = 'EXECUTED' AND created_at >= datetime('now', ?)
            """,
            (customer_id, f"-{int(window_days)} days"),
        )
        last = self.db.query_to_dataframe(
            "SELECT (julianday('now') - julianday(MAX(created_at))) * 24.0 AS hours_since_last "
            "FROM action_execution WHERE customer_id = ? AND status = 'EXECUTED'",
            (customer_id,),
        )
        hsl = last.iloc[0]["hours_since_last"]
        return {
            "action_count": int(win.iloc[0]["action_count"]),
            "spend_inr": float(win.iloc[0]["spend_inr"]),
            "hours_since_last": None if pd.isna(hsl) else float(hsl),
        }

    def update_execution_status(self, action_id: str, status: str, rzp_order_id: str = None):
        query = """
            UPDATE action_execution 
            SET status = ?, razorpay_order_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE action_id = ?
        """
        params = (status, rzp_order_id, action_id)
        self.db.execute_query(query, params)

    def reconcile_pending_actions(self) -> List[Dict[str, Any]]:
        """
        Finds all actions that are requested or unknown, and checks Razorpay for their true status.
        """
        query = "SELECT * FROM action_execution WHERE reconciliation_status = 'PENDING' AND razorpay_order_id IS NOT NULL"
        df = self.db.query_to_dataframe(query)
        
        results = []
        for _, row in df.iterrows():
            action_id = row['action_id']
            link_id = row['razorpay_order_id']
            
            # Fetch real status from Razorpay
            real_status = self.rzp.fetch_payment_link_status(link_id)
            
            terminal_states = ["paid", "expired", "cancelled", "failed"]
            new_status = "RECONCILED" if real_status in terminal_states else "PENDING"
            
            # Update DB
            update_query = """
                UPDATE action_execution 
                SET reconciliation_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE action_id = ?
            """
            self.db.execute_query(update_query, (new_status, action_id))
            
            results.append({
                "action_id": action_id,
                "old_status": row['reconciliation_status'],
                "new_status": new_status,
                "razorpay_status": real_status
            })
            
            logger.info(f"Reconciled action {action_id}: Razorpay status is {real_status}")
            
        return results
