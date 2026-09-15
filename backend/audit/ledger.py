import uuid
import datetime
import json
import logging
from typing import Any, Dict, Optional
from backend.data.database_manager import DatabaseManager

logger = logging.getLogger("AuditLedger")

class AuditLedger:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def record_event(self, action_id: str, customer_id: str, event_type: str, actor: str, status: str, metadata: Dict[str, Any] = None):
        """
        Records an event in the immutable audit ledger.
        """
        event_id = f"evt_{uuid.uuid4().hex}"
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        query = """
            INSERT INTO audit_ledger (event_id, action_id, customer_id, event_type, actor, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (event_id, action_id, customer_id, event_type, actor, status, meta_str)
        
        try:
            self.db.execute_query(query, params)
            logger.info(f"Audit Event Recorded: {event_type} for action {action_id} (Status: {status})")
        except Exception as e:
            logger.error(f"Failed to record audit event: {e}")

    def get_events_for_action(self, action_id: str) -> list:
        query = "SELECT * FROM audit_ledger WHERE action_id = ? ORDER BY timestamp ASC"
        df = self.db.query_to_dataframe(query, (action_id,))
        return df.to_dict('records')

    def get_events_for_customer(self, customer_id: str) -> list:
        query = "SELECT * FROM audit_ledger WHERE customer_id = ? ORDER BY timestamp DESC"
        df = self.db.query_to_dataframe(query, (customer_id,))
        return df.to_dict('records')
