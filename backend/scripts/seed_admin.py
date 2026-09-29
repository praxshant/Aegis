"""Seed or update an Aegis user. No credentials ship in the repo (unlike CICOP's
'admin123' default) — you supply the password here.

Usage:
    python backend/scripts/seed_admin.py --username admin --role admin
    # password from --password, or (safer, no shell history) AEGIS_SEED_PASSWORD env
"""
import argparse
import os
import sys
import uuid

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.database_manager import DatabaseManager
from backend.auth.auth_manager import AuthManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default=os.getenv("AEGIS_SEED_PASSWORD"))
    ap.add_argument("--email", default=None)
    ap.add_argument("--role", default="admin")
    args = ap.parse_args()
    if not args.password:
        ap.error("password required via --password or AEGIS_SEED_PASSWORD env")

    db = DatabaseManager()
    am = AuthManager(db_manager=db)
    pw_hash = am.hash_password(args.password)

    existing = db.query_to_dataframe("SELECT user_id FROM users WHERE username = ?", (args.username,))
    if not existing.empty:
        db.execute_query(
            "UPDATE users SET password_hash=?, role=?, is_active=1 WHERE username=?",
            (pw_hash, args.role, args.username),
        )
        print(f"Updated user '{args.username}' (role={args.role}).")
    else:
        uid = f"usr_{uuid.uuid4().hex[:12]}"
        db.execute_query(
            "INSERT INTO users (user_id, username, email, password_hash, role, is_active) VALUES (?,?,?,?,?,1)",
            (uid, args.username, args.email or f"{args.username}@aegis.local", pw_hash, args.role),
        )
        print(f"Created user '{args.username}' (role={args.role}, id={uid}).")


if __name__ == "__main__":
    main()
