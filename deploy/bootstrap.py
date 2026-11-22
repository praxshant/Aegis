"""Container boot step: make sure the database has operator accounts and demo data.

Free hosts (Hugging Face Spaces, Render free) wipe the disk on every restart, so this
runs before the API starts and is idempotent:

  * AEGIS_ADMIN_PASSWORD set  -> upsert user AEGIS_ADMIN_USERNAME (default "admin"), role admin
  * AEGIS_DEMO_PASSWORD set   -> upsert user AEGIS_DEMO_USERNAME (default "demo"), role operator
  * empty customers table     -> seed the 3 fixture profiles + a 100-customer demo dataset
                                 (AEGIS_SEED_DEMO=0 to skip)
"""
import logging
import os
import shutil
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from backend.data.database_manager import DatabaseManager  # noqa: E402
from backend.auth.auth_manager import AuthManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("bootstrap")


def upsert_user(db: DatabaseManager, am: AuthManager, username: str, password: str, role: str) -> None:
    pw_hash = am.hash_password(password)
    if db.query_to_dataframe("SELECT 1 FROM users WHERE username = ?", (username,)).empty:
        db.execute_query(
            "INSERT INTO users (user_id, username, email, password_hash, role, is_active) VALUES (?,?,?,?,?,1)",
            (f"usr_{uuid.uuid4().hex[:12]}", username, f"{username}@aegis.local", pw_hash, role))
        log.info(f"Created user '{username}' (role={role})")
    else:
        db.execute_query("UPDATE users SET password_hash=?, role=?, is_active=1 WHERE username=?",
                         (pw_hash, role, username))
        log.info(f"Updated user '{username}' (role={role})")


def seed_demo_data(db: DatabaseManager) -> None:
    if int(db.query_to_dataframe("SELECT COUNT(*) FROM customers").iloc[0, 0]) > 0:
        log.info("Customers already present; skipping demo seed.")
        return
    from tests.fixtures.synthetic_dev_data import generate_fixtures
    from generate_final_demo_dataset import generate_dataset
    from backend.ingestion.ingestion_service import IngestionService

    generate_fixtures()  # CUST_001-003 + the Instacart catalog mapping the next-basket model needs
    tmp_dir = tempfile.mkdtemp(prefix="aegis-seed-")
    try:
        csv_path = generate_dataset(out_path=os.path.join(tmp_dir, "demo_dataset.csv"))
        svc = IngestionService(db)
        run_id = svc.start_ingestion_run("demo_dataset.csv")
        svc.process_file_in_background(run_id, csv_path)  # synchronous here; deletes the CSV
        status = db.query_to_dataframe("SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)).iloc[0, 0]
        if status != "COMPLETED":
            raise RuntimeError(f"demo ingestion ended with {status}")
        n = db.execute_query("UPDATE customers SET status='active' WHERE ingestion_run_id = ?", (run_id,))
        db.execute_query("UPDATE ingestion_runs SET import_mode='merge' WHERE run_id = ?", (run_id,))
        log.info(f"Seeded demo dataset: {n} customers active (run {run_id})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    db = DatabaseManager()
    am = AuthManager(db_manager=db)
    log.info(f"Database: {db.db_path}")

    admin_pw = os.getenv("AEGIS_ADMIN_PASSWORD", "")
    if admin_pw:
        upsert_user(db, am, os.getenv("AEGIS_ADMIN_USERNAME", "admin"), admin_pw, "admin")
    else:
        log.warning("AEGIS_ADMIN_PASSWORD is not set: no admin login exists, the dashboard is read-only.")
    demo_pw = os.getenv("AEGIS_DEMO_PASSWORD", "")
    if demo_pw:
        upsert_user(db, am, os.getenv("AEGIS_DEMO_USERNAME", "demo"), demo_pw, "operator")

    if os.getenv("AEGIS_SEED_DEMO", "1") != "0":
        seed_demo_data(db)


if __name__ == "__main__":
    main()
