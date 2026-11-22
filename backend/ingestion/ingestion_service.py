import codecs
import datetime
import hashlib
import json
import logging
import os
import uuid
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from backend.data.database_manager import DatabaseManager
from backend.intelligence.engine import IntelligenceEngine

logger = logging.getLogger("IngestionService")

# Column aliases → canonical name. Matching ignores case, spaces and underscores.
# "price" is a UNIT price (UCI Online Retail II "Price"), so the line amount is
# price × quantity — mapping it to gross_amount silently dropped the quantity.
_ALIASES: Dict[str, List[str]] = {
    "customer_id": ["customer id", "customer", "customerid", "customer_identifier", "buyer_id", "user_id"],
    "invoice": ["invoiceno", "invoice_no", "invoice_id", "order_id", "order", "order_number", "receipt_id"],
    "product_id": ["stockcode", "stock_code", "sku", "sku_id", "product", "item_id"],
    "transaction_timestamp": ["invoicedate", "timestamp", "order_date", "purchase_date", "date", "transaction_date"],
    "quantity": ["qty", "count", "units"],
    "gross_amount": ["amount", "total", "revenue", "line_total", "sales"],
    "unit_price": ["unitprice", "price", "price_per_item", "unit_cost"],
    "return_amount": ["refund_amount", "refund"],
    "transaction_type": ["type", "txn_type"],
    "transaction_id": ["transaction_line_id", "line_id", "txn_id"],
    "category": ["department", "category_name", "product_category"],
    "product_name": ["name", "description", "item_name"],
}
_REQUIRED = ["customer_id", "invoice", "product_id", "transaction_timestamp"]
_VALID_TYPES = {"purchase", "return", "cancellation"}
_MAX_STORED_ERRORS = 1000      # per run; error_count stays exact
_INTEL_COMMIT_EVERY = 200      # keep write transactions short so the API is never locked out


def _simplify(name: str) -> str:
    return str(name).lower().strip().replace("_", "").replace(" ", "")


def _normalize_id(value) -> Optional[str]:
    """'13085.0' / 13085.0 → '13085'; strips whitespace; NaN/blank → None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    return s


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _detect_encoding(path: str) -> str:
    """utf-8 (BOM-tolerant) if the whole file decodes, else latin-1 (Excel/UCI exports)."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                decoder.decode(block)
            decoder.decode(b"", final=True)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


class IngestionService:
    def __init__(self, db: DatabaseManager, ie: Optional[IntelligenceEngine] = None):
        self.db = db
        self.ie = ie or IntelligenceEngine()
        self._run_customers: Dict[str, Set[str]] = {}

    # ── Run bookkeeping ──────────────────────────────────────────────────
    def start_ingestion_run(self, filename: str, total_rows: int = 0) -> str:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        query = """
            INSERT INTO ingestion_runs (run_id, filename, status, total_rows)
            VALUES (?, ?, ?, ?)
        """
        self.db.execute_query(query, (run_id, filename, "PROCESSING", total_rows))
        return run_id

    def update_ingestion_progress(self, run_id: str, valid_rows: int, invalid_rows: int, duplicate_rows: int,
                                  processed_rows: int, status: str = "PROCESSING"):
        self.db.execute_query("""
            UPDATE ingestion_runs
            SET valid_rows = valid_rows + ?,
                invalid_rows = invalid_rows + ?,
                duplicate_rows = duplicate_rows + ?,
                processed_rows = processed_rows + ?,
                status = ?,
                progress_percentage = CASE WHEN total_rows > 0
                    THEN MIN(100.0, ((processed_rows + ?) * 100.0) / total_rows) ELSE 0 END
            WHERE run_id = ?
        """, (valid_rows, invalid_rows, duplicate_rows, processed_rows, status, processed_rows, run_id))

    def finish_ingestion_run(self, run_id: str, status: str = "COMPLETED"):
        query = """
            UPDATE ingestion_runs
            SET status = ?, completed_at = CURRENT_TIMESTAMP,
                progress_percentage = CASE WHEN ? = 'COMPLETED' THEN 100.0 ELSE progress_percentage END
            WHERE run_id = ?
        """
        self.db.execute_query(query, (status, status, run_id))

    def record_errors(self, run_id: str, errors: list):
        if not errors:
            return
        stored = self.db.query_to_dataframe(
            "SELECT COUNT(*) AS n FROM ingestion_errors WHERE run_id = ?", (run_id,)).iloc[0, 0]
        room = max(0, _MAX_STORED_ERRORS - int(stored))
        with self.db.connection() as conn:
            if room:
                conn.executemany(
                    "INSERT INTO ingestion_errors (run_id, row_index, error_reason, raw_data) VALUES (?, ?, ?, ?)",
                    [(run_id, int(e["row_index"]), e["reason"], json.dumps(e["raw_data"], default=str))
                     for e in errors[:room]],
                )
            conn.execute("UPDATE ingestion_runs SET error_count = error_count + ? WHERE run_id = ?",
                         (len(errors), run_id))

    # ── Schema mapping ───────────────────────────────────────────────────
    def detect_schema_and_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map arbitrary column names onto the canonical schema. An exact canonical name
        wins over an alias, and each canonical column is claimed at most once (two source
        columns mapping to one name used to produce duplicate columns and crash)."""
        simplified = {c: _simplify(c) for c in df.columns}
        rename_map: Dict[str, str] = {}
        claimed: Set[str] = set()
        for exact_pass in (True, False):
            for col, cs in simplified.items():
                if col in rename_map:
                    continue
                for target, aliases in _ALIASES.items():
                    if target in claimed:
                        continue
                    hit = cs == _simplify(target) if exact_pass else cs in {_simplify(a) for a in aliases}
                    if hit:
                        rename_map[col] = target
                        claimed.add(target)
                        break
        # Leave unmapped columns alone, but never let one collide with a canonical name.
        df = df.rename(columns=rename_map)
        return df.loc[:, ~df.columns.duplicated()]

    def _prepare_chunk(self, df: pd.DataFrame, offset: int):
        """Returns (valid_df, errors, duplicate_count) with the training-time semantics of
        notebooks-CICOP/01_data_preparation: C-invoices are cancellations, negative
        quantities are returns; purchase gross = qty × unit price; returns/cancellations
        have zero gross and a positive return_amount; net = gross − return."""
        df = self.detect_schema_and_rename(df).reset_index(drop=True)
        missing = [c for c in _REQUIRED if c not in df.columns]
        if "gross_amount" not in df.columns and "unit_price" not in df.columns:
            missing.append("gross_amount or unit_price")
        if missing:
            raise ValueError(f"Missing required columns after mapping: {missing}")

        raw = df.copy()
        out = pd.DataFrame(index=df.index)
        out["customer_id"] = df["customer_id"].map(_normalize_id)
        out["invoice"] = df["invoice"].map(_normalize_id)
        out["product_id"] = df["product_id"].map(_normalize_id)

        # utc=True copes with mixed offsets; naive values are taken as-is (SQLite has no tz)
        ts = pd.to_datetime(df["transaction_timestamp"], errors="coerce", format="mixed", utc=True)
        out["transaction_timestamp"] = ts.dt.tz_localize(None)

        qty = _to_number(df["quantity"]) if "quantity" in df.columns else pd.Series(1.0, index=df.index)
        qty = qty.fillna(1.0)
        if "gross_amount" in df.columns:
            amount = _to_number(df["gross_amount"])
            unit_price = None
        else:
            unit_price = _to_number(df["unit_price"])
            amount = unit_price * qty

        if "transaction_type" in df.columns:
            given = df["transaction_type"].astype(str).str.strip().str.lower()
        else:
            given = pd.Series("", index=df.index)
        derived = np.where(out["invoice"].fillna("").str.upper().str.startswith("C"), "cancellation",
                           np.where((qty < 0) | (amount < 0), "return", "purchase"))
        out["transaction_type"] = np.where(given.isin(_VALID_TYPES), given, derived)

        is_purchase = out["transaction_type"] == "purchase"
        out["gross_amount"] = np.where(is_purchase, amount.abs(), 0.0)
        ret = amount.abs()
        if "return_amount" in df.columns:
            explicit = _to_number(df["return_amount"]).abs()
            ret = explicit.where(explicit > 0, ret)
        out["return_amount"] = np.where(is_purchase, 0.0, ret)
        out["net_amount"] = out["gross_amount"] - out["return_amount"]
        out["quantity"] = qty
        out["category"] = (df["category"].astype(str) if "category" in df.columns else "General")
        out["transaction_id"] = (df["transaction_id"].map(_normalize_id) if "transaction_id" in df.columns else None)

        # Row validation (vectorised; first failing reason wins)
        reason = pd.Series("", index=df.index)
        checks = [
            (out["customer_id"].isna(), "INVALID_CUSTOMER_ID"),
            (out["product_id"].isna(), "MISSING_PRODUCT_ID"),
            (out["invoice"].isna(), "MISSING_INVOICE"),
            (out["transaction_timestamp"].isna(), "INVALID_TIMESTAMP"),
            (amount.isna(), "INVALID_AMOUNT"),
        ]
        if unit_price is not None:
            checks.append((unit_price < 0, "INVALID_AMOUNT"))  # negative prices are data errors
        for mask, why in checks:
            reason = reason.mask((reason == "") & mask.fillna(False), why)

        bad = reason != ""
        errors = [{"row_index": offset + int(i), "reason": reason[i], "raw_data": raw.loc[i].to_dict()}
                  for i in df.index[bad]]
        valid = out[~bad].copy()

        before = len(valid)
        valid = valid.drop_duplicates(subset=["customer_id", "invoice", "product_id", "transaction_timestamp",
                                              "quantity", "gross_amount", "return_amount", "transaction_type"])
        duplicates = before - len(valid)

        # Deterministic line id: re-uploading the same file is idempotent. (The old 8-hex
        # md5 collided past ~65k rows and silently dropped real transactions as duplicates.)
        missing_id = valid["transaction_id"].isna()
        if missing_id.any():
            keys = (valid["customer_id"] + "|" + valid["invoice"] + "|" + valid["product_id"] + "|"
                    + valid["transaction_timestamp"].astype(str) + "|" + valid["quantity"].astype(str) + "|"
                    + valid["gross_amount"].astype(str) + "|" + valid["return_amount"].astype(str))
            valid.loc[missing_id, "transaction_id"] = keys[missing_id].map(
                lambda k: "tx_" + hashlib.sha1(k.encode("utf-8")).hexdigest()[:20])
        return valid, errors, duplicates

    # ── Chunk processing ─────────────────────────────────────────────────
    def process_chunk(self, run_id: str, df: pd.DataFrame, offset: int = 0):
        valid, errors, duplicates = self._prepare_chunk(df, offset)
        if errors:
            self.record_errors(run_id, errors)
        if valid.empty:
            self.update_ingestion_progress(run_id, 0, len(errors), duplicates, len(df))
            return

        first_seen = valid.groupby("customer_id")["transaction_timestamp"].min().dt.strftime("%Y-%m-%d")
        cust_ids = list(first_seen.index)
        self._run_customers.setdefault(run_id, set()).update(cust_ids)

        with self.db.connection() as conn:
            existing: Set[str] = set()
            for i in range(0, len(cust_ids), 900):  # stay under SQLite's bound-parameter limit
                part = cust_ids[i:i + 900]
                rows = conn.execute(
                    f"SELECT customer_id FROM customers WHERE customer_id IN ({','.join('?' * len(part))})", part
                ).fetchall()
                existing.update(r[0] for r in rows)

            new_rows = [(cid, "User", cid, f"{cid}@example.com", first_seen[cid], 0, run_id)
                        for cid in cust_ids if cid not in existing]
            conn.executemany("""
                INSERT INTO customers (customer_id, first_name, last_name, email, registration_date, opted_out,
                                       status, ingestion_run_id)
                VALUES (?, ?, ?, ?, ?, ?, 'pending_activation', ?)
            """, new_rows)
            # Customers already on file now belong to this run too, so activating it in
            # 'replace' mode keeps them instead of archiving customers present in the new data.
            conn.executemany("UPDATE customers SET ingestion_run_id = ? WHERE customer_id = ?",
                             [(run_id, cid) for cid in cust_ids if cid in existing])

            cur = conn.executemany("""
                INSERT INTO transactions (transaction_id, customer_id, transaction_timestamp, transaction_type,
                                          invoice, product_id, gross_amount, return_amount, net_amount, category, merchant)
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM transactions WHERE transaction_id = ?)
            """, [(
                r.transaction_id, r.customer_id, r.transaction_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                r.transaction_type, r.invoice, r.product_id, float(r.gross_amount), float(r.return_amount),
                float(r.net_amount), str(r.category), "Ingested_Merchant", r.transaction_id,
            ) for r in valid.itertuples(index=False)])
            inserted = max(cur.rowcount, 0)

        already_loaded = len(valid) - inserted
        self.update_ingestion_progress(run_id, inserted, len(errors), duplicates + already_loaded, len(df))

    def process_file_in_background(self, run_id: str, filepath: str):
        try:
            with open(filepath, "rb") as f:
                total_rows = max(sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b"")) - 1, 0)
            self.db.execute_query("UPDATE ingestion_runs SET total_rows = ? WHERE run_id = ?", (total_rows, run_id))

            encoding = _detect_encoding(filepath)
            offset = 0
            # dtype=str keeps identifiers intact (no 13085 → 13085.0, no lost leading zeros)
            for chunk in pd.read_csv(filepath, chunksize=10000, dtype=str, encoding=encoding):
                self.process_chunk(run_id, chunk, offset)
                offset += len(chunk)

            self.finish_ingestion_run(run_id, "PROCESSING_INTELLIGENCE")
            scored = self.generate_batch_intelligence(run_id)
            self.db.execute_query("UPDATE ingestion_runs SET customer_count = ? WHERE run_id = ?",
                                  (len(self._run_customers.get(run_id, ())), run_id))
            logger.info(f"Ingestion {run_id}: scored {scored} customers")
            self.finish_ingestion_run(run_id, "COMPLETED")
        except Exception as e:
            logger.exception(f"Ingestion {run_id} failed")
            self.finish_ingestion_run(run_id, f"FAILED: {str(e)[:500]}")
        finally:
            self._run_customers.pop(run_id, None)
            try:
                os.remove(filepath)  # uploads are transient; don't let them fill the disk
            except OSError:
                pass

    def generate_batch_intelligence(self, run_id: str, customer_ids: Optional[Iterable[str]] = None) -> int:
        """Score (propensity, band, segment, feature snapshot) the customers touched by this
        run — not every customer in the database on every upload."""
        if customer_ids is None:
            customer_ids = self._run_customers.get(run_id)
        if customer_ids is None:
            customer_ids = self.db.query_to_dataframe(
                "SELECT customer_id FROM customers WHERE ingestion_run_id = ?", (run_id,))["customer_id"].tolist()
        customer_ids = list(customer_ids)
        if not customer_ids:
            return 0

        opted_out = set(self.db.query_to_dataframe(
            "SELECT customer_id FROM customers WHERE opted_out = 1")["customer_id"].astype(str))
        stored = self.db.query_to_dataframe("SELECT customer_id, segment_name FROM customer_segments")
        stored_seg = dict(zip(stored["customer_id"].astype(str), stored["segment_name"]))

        pending, scored = [], 0
        for cid in customer_ids:
            if cid in opted_out:
                continue
            try:
                tx_df = self.db.query_to_dataframe("SELECT * FROM transactions WHERE customer_id = ?", (cid,))
                intel = self.ie.get_intelligence(cid, tx_df, segment=stored_seg.get(cid))
                if intel.status == "COMPLETE":
                    pending.append((cid, intel))
            except Exception as e:
                logger.warning(f"Failed to generate intelligence for {cid}: {e}")
            if len(pending) >= _INTEL_COMMIT_EVERY:
                scored += self._persist_intelligence(pending)
                pending = []
        scored += self._persist_intelligence(pending)
        return scored

    def _persist_intelligence(self, batch) -> int:
        if not batch:
            return 0
        with self.db.connection() as conn:
            for cid, intel in batch:
                fs = intel.feature_snapshot or {}
                conn.execute("""
                    UPDATE customers
                    SET propensity_score = ?, propensity_band = ?,
                        FEATURE_recency_days = ?, FEATURE_customer_lifetime_days = ?,
                        FEATURE_purchase_order_count = ?, FEATURE_net_spend_90d = ?, FEATURE_total_net_spend = ?
                    WHERE customer_id = ?
                """, (intel.purchase_probability_30d, intel.propensity_band, fs.get("recency_days"),
                      fs.get("lifetime_days"), fs.get("purchase_order_count"), fs.get("net_spend_90d"),
                      fs.get("total_net_spend"), cid))
                # DELETE+INSERT rather than INSERT OR REPLACE: tables re-created by older
                # tooling may have lost the primary key that OR REPLACE relies on.
                conn.execute("DELETE FROM customer_segments WHERE customer_id = ?", (cid,))
                conn.execute("""
                    INSERT INTO customer_segments (customer_id, segment_name, last_updated)
                    VALUES (?, ?, CURRENT_DATE)
                """, (cid, intel.segment))
        return len(batch)
