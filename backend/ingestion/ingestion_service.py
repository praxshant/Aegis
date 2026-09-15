import pandas as pd
import uuid
import datetime
import json
import logging
from typing import Dict, Any, Optional
from backend.data.database_manager import DatabaseManager
from backend.intelligence.engine import IntelligenceEngine

logger = logging.getLogger("IngestionService")

class IngestionService:
    def __init__(self, db: DatabaseManager, ie: Optional[IntelligenceEngine] = None):
        self.db = db
        self.ie = ie or IntelligenceEngine()
        
    def start_ingestion_run(self, filename: str, total_rows: int = 0) -> str:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        query = """
            INSERT INTO ingestion_runs (run_id, filename, status, total_rows)
            VALUES (?, ?, ?, ?)
        """
        self.db.execute_query(query, (run_id, filename, "PROCESSING", total_rows))
        return run_id
        
    def update_ingestion_progress(self, run_id: str, valid_rows: int, invalid_rows: int, duplicate_rows: int, processed_rows: int, status: str = "PROCESSING"):
        query = """
            UPDATE ingestion_runs
            SET valid_rows = valid_rows + ?,
                invalid_rows = invalid_rows + ?,
                duplicate_rows = duplicate_rows + ?,
                processed_rows = processed_rows + ?,
                status = ?
            WHERE run_id = ?
        """
        self.db.execute_query(query, (valid_rows, invalid_rows, duplicate_rows, processed_rows, status, run_id))
        
        # Update percentage if total_rows > 0
        self.db.execute_query("""
            UPDATE ingestion_runs
            SET progress_percentage = CASE WHEN total_rows > 0 THEN (processed_rows * 100.0) / total_rows ELSE 0 END
            WHERE run_id = ?
        """, (run_id,))
        
    def finish_ingestion_run(self, run_id: str, status: str = "COMPLETED"):
        query = """
            UPDATE ingestion_runs
            SET status = ?, completed_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
        """
        self.db.execute_query(query, (status, run_id))
        
    def record_errors(self, run_id: str, errors: list):
        if not errors: return
        query = "INSERT INTO ingestion_errors (run_id, row_index, error_reason, raw_data) VALUES (?, ?, ?, ?)"
        for err in errors:
            self.db.execute_query(query, (run_id, err['row_index'], err['reason'], json.dumps(err['raw_data'])))
        self.db.execute_query("UPDATE ingestion_runs SET error_count = error_count + ? WHERE run_id = ?", (len(errors), run_id))

    def detect_schema_and_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "customer_id": ["customer id", "customer", "customer_identifier", "buyer_id", "user_id"],
            "invoice": ["invoiceno", "order_id", "order", "transaction_id", "receipt_id"],
            "product_id": ["stockcode", "sku", "sku_id", "product", "item_id"],
            "transaction_timestamp": ["invoicedate", "timestamp", "order_date", "purchase_date", "date"],
            "quantity": ["qty", "count"],
            "gross_amount": ["amount", "total", "price", "revenue"],
            "category": ["department", "category_name", "product_category"],
            "product_name": ["name", "description", "item_name"],
            "unit_price": ["unitprice", "price_per_item"]
        }
        
        rename_map = {}
        cols_simplified = {c: c.lower().strip().replace('_', '').replace(' ', '') for c in df.columns}
        
        for c, cs in cols_simplified.items():
            for target, matches in aliases.items():
                target_sim = target.lower().replace('_', '').replace(' ', '')
                matches_sim = [m.lower().replace('_', '').replace(' ', '') for m in matches]
                if cs == target_sim or cs in matches_sim:
                    rename_map[c] = target
                    break
                    
        df = df.rename(columns=rename_map)
        
        # Derived values
        if "gross_amount" not in df.columns and "unit_price" in df.columns and "quantity" in df.columns:
            # Handle possible string prices
            df["unit_price"] = pd.to_numeric(df["unit_price"].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
            df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce')
            df["gross_amount"] = df["unit_price"] * df["quantity"]
            
        return df

    def process_chunk(self, run_id: str, df: pd.DataFrame, offset: int = 0):
        df = self.detect_schema_and_rename(df)
        
        required_cols = ["customer_id", "invoice", "product_id", "transaction_timestamp", "gross_amount"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after mapping: {missing}")
            
        # Defaults
        if "return_amount" not in df.columns:
            df["return_amount"] = 0.0
        if "quantity" not in df.columns:
            df["quantity"] = 1
        if "product_name" not in df.columns:
            df["product_name"] = "Unknown"
        if "category" not in df.columns:
            df["category"] = "General"
            
        # Add generated transaction_id if missing (using hash)
        if "transaction_id" not in df.columns:
            import hashlib
            df["transaction_id"] = df.apply(
                lambda r: "tx_" + hashlib.md5(
                    (str(r['customer_id'])+str(r['invoice'])+str(r['product_id'])+str(r['transaction_timestamp'])).encode()
                ).hexdigest()[:8], 
                axis=1
            )

        errors = []
        valid_rows = []
        
        # Basic Validation
        for i, row in df.iterrows():
            is_valid = True
            reason = ""
            if pd.isna(row["customer_id"]) or str(row["customer_id"]).strip() == "":
                is_valid, reason = False, "INVALID_CUSTOMER_ID"
            elif pd.isna(row["product_id"]) or str(row["product_id"]).strip() == "":
                is_valid, reason = False, "MISSING_PRODUCT_ID"
            elif pd.isna(row["invoice"]) or str(row["invoice"]).strip() == "":
                is_valid, reason = False, "MISSING_INVOICE"
            elif pd.isna(row["transaction_timestamp"]):
                is_valid, reason = False, "INVALID_TIMESTAMP"
            elif pd.isna(row["gross_amount"]):
                is_valid, reason = False, "INVALID_AMOUNT"
                
            if is_valid:
                valid_rows.append(row.to_dict())
            else:
                errors.append({"row_index": offset + i, "reason": reason, "raw_data": row.to_dict()})
                
        if errors:
            self.record_errors(run_id, errors)

        if not valid_rows:
            self.update_ingestion_progress(run_id, 0, len(errors), 0, len(df))
            return
            
        valid_df = pd.DataFrame(valid_rows)
        
        # Clean data types
        valid_df["transaction_timestamp"] = pd.to_datetime(valid_df["transaction_timestamp"], errors='coerce')
        valid_df = valid_df.dropna(subset=["transaction_timestamp"]) # Final safety check
        
        # 1. Update Customers (Idempotent Insert/Ignore)
        cust_df = valid_df[["customer_id"]].drop_duplicates()
        cust_df["first_name"] = "User"
        cust_df["last_name"] = cust_df["customer_id"]
        cust_df["email"] = cust_df["customer_id"] + "@example.com"
        cust_df["registration_date"] = valid_df["transaction_timestamp"].min().strftime('%Y-%m-%d')
        cust_df["opted_out"] = 0
        
        with (self.db._persistent_conn if self.db._persistent_conn else self.db._get_connection()) as conn:
            cursor = conn.cursor()
            # Insert customers if not exists
            for _, c_row in cust_df.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO customers (customer_id, first_name, last_name, email, registration_date, opted_out, status, ingestion_run_id)
                        VALUES (?, ?, ?, ?, ?, ?, 'pending_activation', ?)
                    """, (c_row["customer_id"], c_row["first_name"], c_row["last_name"], c_row["email"], c_row["registration_date"], c_row["opted_out"], run_id))
                except Exception:
                    # Customer already exists — update ingestion_run_id if they don't have one yet
                    try:
                        cursor.execute("""
                            UPDATE customers SET ingestion_run_id = ?, status = COALESCE(status, 'active')
                            WHERE customer_id = ? AND ingestion_run_id IS NULL
                        """, (run_id, c_row["customer_id"]))
                    except Exception:
                        pass
                    
            # Insert Transactions (Idempotent Insert/Ignore)
            inserted_tx = 0
            duplicate_tx = 0
            for _, t_row in valid_df.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO transactions (transaction_id, customer_id, transaction_timestamp, transaction_type, invoice, product_id, gross_amount, return_amount, net_amount, category, merchant)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(t_row["transaction_id"]), str(t_row["customer_id"]), str(t_row["transaction_timestamp"]), 
                        "purchase",
                        str(t_row["invoice"]), str(t_row["product_id"]), float(t_row["gross_amount"]), 
                        float(t_row["return_amount"]), float(t_row["gross_amount"] - t_row["return_amount"]), 
                        str(t_row["category"]), "Ingested_Merchant"
                    ))
                    inserted_tx += 1
                except Exception:
                    # SQLite raises IntegrityError on PRIMARY KEY violation
                    duplicate_tx += 1
            
            conn.commit()

        self.update_ingestion_progress(run_id, inserted_tx, len(errors), duplicate_tx, len(df))

    def process_file_in_background(self, run_id: str, filepath: str):
        try:
            # Count total rows for progress bar
            with open(filepath, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1 # Exclude header
                
            self.db.execute_query("UPDATE ingestion_runs SET total_rows = ? WHERE run_id = ?", (total_rows, run_id))
            
            chunk_size = 10000
            offset = 0
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                self.process_chunk(run_id, chunk, offset)
                offset += len(chunk)
                
            self.finish_ingestion_run(run_id, "PROCESSING_INTELLIGENCE")
            self.generate_batch_intelligence(run_id)
            self.finish_ingestion_run(run_id, "COMPLETED")
        except Exception as e:
            logger.error(f"Ingestion {run_id} failed: {e}")
            self.finish_ingestion_run(run_id, f"FAILED: {str(e)}")

    def generate_batch_intelligence(self, run_id: str):
        # Fetch customers impacted by this run
        cust_query = "SELECT DISTINCT customer_id FROM customers WHERE opted_out = 0"
        customers = self.db.query_to_dataframe(cust_query)["customer_id"].tolist()
        
        # Load all transactions once for efficiency, or per customer if memory is constrained
        # For MVP, we query per customer to reuse the existing `IntelligenceEngine`
        with (self.db._persistent_conn if self.db._persistent_conn else self.db._get_connection()) as conn:
            cursor = conn.cursor()
            for cid in customers:
                try:
                    tx_df = self.db.query_to_dataframe("SELECT * FROM transactions WHERE customer_id = ?", (cid,))
                    intel = self.ie.get_intelligence(cid, tx_df)
                    
                    if intel.status == "COMPLETE":
                        # Persist propensity and band
                        cursor.execute("""
                            UPDATE customers
                            SET propensity_score = ?, propensity_band = ?
                            WHERE customer_id = ?
                        """, (intel.purchase_probability_30d, intel.propensity_band, cid))
                        
                        # Persist segment
                        cursor.execute("""
                            INSERT OR REPLACE INTO customer_segments (customer_id, segment_name, last_updated)
                            VALUES (?, ?, CURRENT_DATE)
                        """, (cid, intel.segment))
                        
                except Exception as e:
                    logger.warning(f"Failed to generate intelligence for {cid}: {e}")
            conn.commit()
