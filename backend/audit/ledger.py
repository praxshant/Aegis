import uuid
import json
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from backend.data.database_manager import DatabaseManager

logger = logging.getLogger("AuditLedger")

GENESIS_HASH = "0" * 64

# Serialises appends within this process so two threads never read the same chain head.
# BEGIN IMMEDIATE below does the same across processes sharing the SQLite file.
_append_lock = threading.Lock()


def _canonical(event: Dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_event_hash(prev_hash: str, event: Dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + _canonical(event)).encode("utf-8")).hexdigest()


class AuditLedger:
    """Append-only audit ledger with a SHA-256 hash chain.

    Every row stores prev_hash (the previous row's event_hash) and its own event_hash,
    computed over the row's content. Editing, deleting, or re-ordering any row breaks the
    chain, which verify_chain() reports. UPDATE/DELETE are also blocked by triggers."""

    _HASHED_FIELDS = ("event_id", "timestamp", "action_id", "customer_id", "event_type", "actor", "status", "metadata")

    def __init__(self, db: DatabaseManager):
        self.db = db

    def record_event(self, action_id: str, customer_id: str, event_type: str, actor: str, status: str,
                     metadata: Dict[str, Any] = None) -> Optional[str]:
        """Records an event in the immutable audit ledger. Returns the event_id, or None if the
        write failed (an audit failure is logged, never raised into the business flow)."""
        event = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "action_id": action_id,
            "customer_id": customer_id,
            "event_type": event_type,
            "actor": actor,
            "status": status,
            # default=str: metadata can carry numpy/Decimal/datetime values from upstream
            "metadata": json.dumps(metadata, default=str) if metadata else "{}",
        }
        try:
            with _append_lock, self.db.connection() as conn:
                if self.db._persistent_conn is None:
                    conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT event_hash FROM audit_ledger WHERE event_hash IS NOT NULL ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                prev_hash = row[0] if row else GENESIS_HASH
                event_hash = compute_event_hash(prev_hash, event)
                conn.execute(
                    """INSERT INTO audit_ledger
                       (event_id, timestamp, action_id, customer_id, event_type, actor, status, metadata, prev_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(event[f] for f in self._HASHED_FIELDS) + (prev_hash, event_hash),
                )
            logger.info(f"Audit Event Recorded: {event_type} for action {action_id} (Status: {status})")
            return event["event_id"]
        except Exception as e:
            logger.error(f"Failed to record audit event {event_type} for action {action_id}: {e}")
            return None

    def verify_chain(self) -> Dict[str, Any]:
        """Walk the ledger in insertion order and recompute every hash.
        Rows written before hash-chaining existed (event_hash NULL) are counted as legacy."""
        df = self.db.query_to_dataframe(
            "SELECT rowid AS _rowid, event_id, timestamp, action_id, customer_id, event_type, actor, status, "
            "metadata, prev_hash, event_hash FROM audit_ledger ORDER BY rowid ASC"
        )
        expected_prev = GENESIS_HASH
        checked = legacy = 0
        for rec in df.to_dict("records"):
            if rec["event_hash"] is None:
                legacy += 1
                continue
            event = {f: rec[f] for f in self._HASHED_FIELDS}
            if rec["prev_hash"] != expected_prev:
                return {"valid": False, "checked": checked, "legacy_unhashed": legacy,
                        "broken_at": rec["event_id"], "reason": "prev_hash does not match the previous event"}
            if compute_event_hash(rec["prev_hash"], event) != rec["event_hash"]:
                return {"valid": False, "checked": checked, "legacy_unhashed": legacy,
                        "broken_at": rec["event_id"], "reason": "event content does not match its hash"}
            expected_prev = rec["event_hash"]
            checked += 1
        return {"valid": True, "checked": checked, "legacy_unhashed": legacy, "head": expected_prev}

    def get_events_for_action(self, action_id: str) -> list:
        query = "SELECT * FROM audit_ledger WHERE action_id = ? ORDER BY rowid ASC"
        df = self.db.query_to_dataframe(query, (action_id,))
        return df.to_dict('records')

    def get_events_for_customer(self, customer_id: str, limit: int = 500) -> list:
        # rowid breaks ties between events written in the same second
        query = "SELECT * FROM audit_ledger WHERE customer_id = ? ORDER BY rowid DESC LIMIT ?"
        df = self.db.query_to_dataframe(query, (customer_id, int(limit)))
        return df.to_dict('records')
