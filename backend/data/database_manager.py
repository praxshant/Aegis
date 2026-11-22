import sqlite3
import pandas as pd
from typing import Dict, Optional
import os
from contextlib import contextmanager
from pathlib import Path
import logging

# Anchor relative DB paths on the project root, not the process CWD, so the API,
# the Streamlit app, and the CLI scripts all open the same file wherever they start.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = "backend/data/aegis.db"


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_db_path(db_path: Optional[str] = None) -> str:
    """An explicit db_path wins; otherwise DATABASE_URL; otherwise the default file.
    (Env no longer overrides an explicit path — a test asking for ':memory:' must never
    be silently pointed at the real database.)"""
    raw = db_path if db_path is not None else (os.getenv("DATABASE_URL", "").strip() or DEFAULT_DB_PATH)
    for prefix in ("sqlite:///", "sqlite://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw in ("", ":memory:"):
        return ":memory:"
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


# ── Schema (single source of truth for file and :memory: databases) ──────────
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        registration_date DATE,
        age INTEGER,
        gender TEXT,
        location TEXT,
        opted_out BOOLEAN DEFAULT 0,
        status TEXT DEFAULT 'active',
        ingestion_run_id TEXT DEFAULT NULL,
        FEATURE_customer_lifetime_days REAL,
        FEATURE_purchase_order_count REAL,
        FEATURE_recency_days REAL,
        FEATURE_net_spend_90d REAL,
        FEATURE_total_net_spend REAL,
        propensity_score REAL,
        propensity_band TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        customer_id TEXT,
        transaction_timestamp DATETIME,
        transaction_type TEXT,
        invoice TEXT,
        product_id TEXT,
        gross_amount REAL,
        return_amount REAL,
        net_amount REAL,
        category TEXT,
        merchant TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )""",
    """CREATE TABLE IF NOT EXISTS customer_segments (
        customer_id TEXT PRIMARY KEY,
        segment_id INTEGER,
        segment_name TEXT,
        last_updated DATE,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )""",
    # Append-only, hash-chained: event_hash = sha256(prev_hash + canonical event).
    """CREATE TABLE IF NOT EXISTS audit_ledger (
        event_id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        action_id TEXT,
        customer_id TEXT,
        event_type TEXT,
        actor TEXT,
        status TEXT,
        metadata TEXT,
        prev_hash TEXT,
        event_hash TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS action_execution (
        action_id TEXT PRIMARY KEY,
        customer_id TEXT,
        action_type TEXT,
        requested_discount_pct REAL,
        amount_inr REAL,
        status TEXT,
        policy_rule_id TEXT,
        razorpay_order_id TEXT,
        payment_status TEXT,
        reconciliation_status TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS ingestion_runs (
        run_id TEXT PRIMARY KEY,
        filename TEXT,
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed_at DATETIME,
        status TEXT,
        total_rows INTEGER DEFAULT 0,
        valid_rows INTEGER DEFAULT 0,
        invalid_rows INTEGER DEFAULT 0,
        duplicate_rows INTEGER DEFAULT 0,
        processed_rows INTEGER DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        progress_percentage REAL DEFAULT 0.0,
        import_mode TEXT DEFAULT 'merge',
        customer_count INTEGER DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS ingestion_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        row_index INTEGER,
        error_reason TEXT,
        raw_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES ingestion_runs (run_id)
    )""",
    # Auth users (WS1 trust layer).
    """CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'viewer',
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )""",
]

# Columns added after a table's first release. Older databases (and tables that were
# re-created by pandas' to_sql(if_exists="replace")) are upgraded in place on startup.
_MIGRATIONS = {
    "customers": [
        ("opted_out", "BOOLEAN DEFAULT 0"),
        ("status", "TEXT DEFAULT 'active'"),
        ("ingestion_run_id", "TEXT DEFAULT NULL"),
        ("age", "INTEGER"),
        ("gender", "TEXT"),
        ("location", "TEXT"),
        ("propensity_score", "REAL"),
        ("propensity_band", "TEXT"),
        ("FEATURE_customer_lifetime_days", "REAL"),
        ("FEATURE_purchase_order_count", "REAL"),
        ("FEATURE_recency_days", "REAL"),
        ("FEATURE_net_spend_90d", "REAL"),
        ("FEATURE_total_net_spend", "REAL"),
    ],
    "action_execution": [
        ("amount_inr", "REAL"),
        ("payment_status", "TEXT"),
    ],
    "audit_ledger": [
        ("prev_hash", "TEXT"),
        ("event_hash", "TEXT"),
    ],
    "ingestion_runs": [
        ("import_mode", "TEXT DEFAULT 'merge'"),
        ("customer_count", "INTEGER DEFAULT 0"),
    ],
}

_POST_MIGRATION = [
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
    "CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action_id ON audit_ledger(action_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_customer_id ON audit_ledger(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_customer_id ON action_execution(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_action_rzp_link ON action_execution(razorpay_order_id)",
    # The ledger is append-only at the storage layer, not just by convention.
    """CREATE TRIGGER IF NOT EXISTS audit_ledger_no_update BEFORE UPDATE ON audit_ledger
       BEGIN SELECT RAISE(ABORT, 'audit_ledger is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS audit_ledger_no_delete BEFORE DELETE ON audit_ledger
       BEGIN SELECT RAISE(ABORT, 'audit_ledger is append-only'); END""",
]


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.logger = setup_logger("DatabaseManager")
        self.db_path = resolve_db_path(db_path)
        self.database_url = f"sqlite:///{self.db_path}"
        self.db_type = "sqlite"  # Enforcing SQLite for Aegis MVP

        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._persistent_conn = None
        else:
            # One shared connection, or every query would see a fresh empty database.
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._initialize_database()

    # ── Connections ───────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)

    @contextmanager
    def connection(self):
        """Yield a connection; commit on success, roll back on error, always close.
        (sqlite3's own `with conn:` commits but never closes — it leaked a handle per query.)"""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if conn is not self._persistent_conn:
                conn.close()

    # Backwards-compatible alias used by older call sites.
    _get_connection = connection

    def _initialize_database(self):
        with self.connection() as conn:
            if self._persistent_conn is None:
                try:
                    # WAL lets readers proceed while ingestion/audit writes are in flight.
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError as e:
                    self.logger.warning(f"Could not enable WAL mode: {e}")
            cur = conn.cursor()
            for stmt in _SCHEMA:
                cur.execute(stmt)
            for table, cols in _MIGRATIONS.items():
                existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
                for col_name, col_type in cols:
                    if col_name not in existing:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            for stmt in _POST_MIGRATION:
                cur.execute(stmt)

    # ── Query helpers ─────────────────────────────────────────────────────
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """Insert rows into an existing table. if_exists='replace' deletes the table's rows
        first but keeps its schema (pandas' own 'replace' drops the table and re-creates it
        without primary keys or the status/propensity columns the API depends on)."""
        with self.connection() as conn:
            if if_exists == "replace":
                conn.execute(f"DELETE FROM {table_name}")
            df.to_sql(table_name, conn, if_exists="append", index=False)
        self.logger.info(f"Inserted {len(df)} records into {table_name}")

    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        with self.connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def execute_query(self, query: str, params: Optional[tuple] = None):
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            return cursor.rowcount

    def get_customer_data(self, customer_id: Optional[str] = None) -> pd.DataFrame:
        query = """
            SELECT c.*, cs.segment_name
            FROM customers c
            LEFT JOIN customer_segments cs ON c.customer_id = cs.customer_id
        """
        if customer_id:
            query += " WHERE c.customer_id = ?"
            return self.query_to_dataframe(query, (customer_id,))
        return self.query_to_dataframe(query)

    def get_transaction_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        query = """
            SELECT t.*, c.first_name, c.last_name, cs.segment_name
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            LEFT JOIN customer_segments cs ON t.customer_id = cs.customer_id
        """
        params = []
        if start_date and end_date:
            # transactions has transaction_timestamp; there is no transaction_date column
            query += " WHERE t.transaction_timestamp BETWEEN ? AND ?"
            params = [start_date, end_date]
        return self.query_to_dataframe(query, tuple(params) if params else None)

    def get_database_stats(self) -> Dict[str, int]:
        stats = {}
        tables = ['customers', 'transactions', 'audit_ledger', 'action_execution']
        for table in tables:
            query = f"SELECT COUNT(*) FROM {table}"
            try:
                count = self.query_to_dataframe(query).iloc[0, 0]
                stats[table] = int(count)
            except Exception:
                stats[table] = 0
        return stats
