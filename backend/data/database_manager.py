import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from contextlib import contextmanager
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class DatabaseManager:
    def __init__(self, db_path: str = "backend/data/aegis.db"):
        self.logger = setup_logger("DatabaseManager")
        env_url = os.getenv("DATABASE_URL", "")
        
        if env_url:
            self.database_url = env_url
        else:
            if db_path == ":memory:":
                self.database_url = "sqlite:///:memory:"
            else:
                self.database_url = f"sqlite:///{db_path}"
                
        self.db_type = "sqlite" # Enforcing SQLite for Aegis MVP
        self.db_path = db_path if not env_url else self.database_url.replace("sqlite:///", "").replace("sqlite://", "")

        # Create directory for SQLite file DBs
        real_path = self.db_path
        db_dir = os.path.dirname(real_path)
        if real_path not in (":memory:", "") and db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._persistent_conn = sqlite3.connect(":memory:") if self.database_url.endswith(":memory:") else None
        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            if self._persistent_conn is not None:
                conn = self._persistent_conn
            else:
                conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
                conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn is not None and conn is not self._persistent_conn:
                conn.close()
    
    def _initialize_database(self):
        if self._persistent_conn is not None:
            conn = self._persistent_conn
            cursor = conn.cursor()
        else:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customers (
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
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
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
                    )
                """)
                
                # Create customer_segments table (Aegis Intelligence Output)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS customer_segments (
                        customer_id TEXT PRIMARY KEY,
                        segment_id INTEGER,
                        segment_name TEXT,
                        last_updated DATE,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                """)

                # Create Audit Ledger table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_ledger (
                        event_id TEXT PRIMARY KEY,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        action_id TEXT,
                        customer_id TEXT,
                        event_type TEXT,
                        actor TEXT,
                        status TEXT,
                        metadata TEXT
                    )
                """)

                # Create Action Execution table (Reconciliation)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS action_execution (
                        action_id TEXT PRIMARY KEY,
                        customer_id TEXT,
                        action_type TEXT,
                        requested_discount_pct REAL,
                        amount_inr REAL,
                        status TEXT,
                        policy_rule_id TEXT,
                        razorpay_order_id TEXT,
                        reconciliation_status TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create Ingestion Runs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_runs (
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
                    )
                """)
                
                # Create Ingestion Errors table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        row_index INTEGER,
                        error_reason TEXT,
                        raw_data TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (run_id) REFERENCES ingestion_runs (run_id)
                    )
                """)

                # Auth users (WS1 trust layer). Only the users table is ported from
                # CICOP — api_keys/user_sessions/audit_logs are YAGNI for Aegis.
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'viewer',
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

                # Add opted_out to customers if missing
                try:
                    cursor.execute("ALTER TABLE customers ADD COLUMN opted_out BOOLEAN DEFAULT 0")
                except sqlite3.OperationalError:
                    pass # Column likely exists
                
                for col_name, col_type in [
                    ("propensity_score", "REAL"), 
                    ("propensity_band", "TEXT"),
                    ("FEATURE_customer_lifetime_days", "REAL"),
                    ("FEATURE_purchase_order_count", "REAL"),
                    ("FEATURE_recency_days", "REAL"),
                    ("FEATURE_net_spend_90d", "REAL"),
                    ("FEATURE_total_net_spend", "REAL")
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE customers ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass
                
                # Backfill amount_inr on action_execution for pre-1D DBs (budget cap needs it)
                try:
                    cursor.execute("ALTER TABLE action_execution ADD COLUMN amount_inr REAL")
                except sqlite3.OperationalError:
                    pass

                # Indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action_id ON audit_ledger(action_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_customer_id ON action_execution(customer_id)")
                
                conn.commit()
                return

        # Handle persistent memory init if needed
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                email TEXT,
                registration_date DATE,
                age INTEGER,
                gender TEXT,
                location TEXT,
                opted_out BOOLEAN DEFAULT 0,
                FEATURE_customer_lifetime_days REAL,
                FEATURE_purchase_order_count REAL,
                FEATURE_recency_days REAL,
                FEATURE_net_spend_90d REAL,
                FEATURE_total_net_spend REAL,
                propensity_score REAL,
                propensity_band TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
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
            )
        """)
        # Create customer_segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_segments (
                customer_id TEXT PRIMARY KEY,
                segment_id INTEGER,
                segment_name TEXT,
                last_updated DATE,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        """)
        # Create Audit Ledger table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_ledger (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action_id TEXT,
                customer_id TEXT,
                event_type TEXT,
                actor TEXT,
                status TEXT,
                metadata TEXT
            )
        """)
        # Create Action Execution table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_execution (
                action_id TEXT PRIMARY KEY,
                customer_id TEXT,
                action_type TEXT,
                requested_discount_pct REAL,
                amount_inr REAL,
                status TEXT,
                policy_rule_id TEXT,
                razorpay_order_id TEXT,
                reconciliation_status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Ingestion Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
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
                progress_percentage REAL DEFAULT 0.0
            )
        """)
        
        # Create Ingestion Errors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                row_index INTEGER,
                error_reason TEXT,
                raw_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES ingestion_runs (run_id)
            )
        """)

        # Auth users (WS1 trust layer) — mirror of Block A for :memory: DBs.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        # Add opted_out to customers if missing
        try:
            cursor.execute("ALTER TABLE customers ADD COLUMN opted_out BOOLEAN DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        for col_name, col_type in [
            ("propensity_score", "REAL"), 
            ("propensity_band", "TEXT"),
            ("FEATURE_customer_lifetime_days", "REAL"),
            ("FEATURE_purchase_order_count", "REAL"),
            ("FEATURE_recency_days", "REAL"),
            ("FEATURE_net_spend_90d", "REAL"),
            ("FEATURE_total_net_spend", "REAL")
        ]:
            try:
                cursor.execute(f"ALTER TABLE customers ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

        # Backfill amount_inr on action_execution for pre-1D DBs (budget cap needs it)
        try:
            cursor.execute("ALTER TABLE action_execution ADD COLUMN amount_inr REAL")
        except sqlite3.OperationalError:
            pass

        conn.commit()
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            self.logger.info(f"Inserted {len(df)} records into {table_name}")
    
    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        with (self._persistent_conn if self._persistent_conn else sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
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
            query += " WHERE t.transaction_date BETWEEN ? AND ?"
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
