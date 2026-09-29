"""WS1-1A auth checks — token/password primitives, DB-backed authenticate, and the
FastAPI route guard in isolation (no heavy app startup)."""
import os

os.environ["JWT_SECRET_KEY"] = "test-secret-not-for-prod"  # stable signing key for the test process

import pytest

from backend.data.database_manager import DatabaseManager
from backend.auth.auth_manager import AuthManager


def _fresh_auth(db_path=":memory:"):
    db = DatabaseManager(db_path)  # users table exists in the :memory: schema block too
    am = AuthManager(db_manager=db)
    uid = "usr_test01"
    db.execute_query(
        "INSERT INTO users (user_id, username, email, password_hash, role, is_active) VALUES (?,?,?,?,?,1)",
        (uid, "alice", "alice@aegis.local", am.hash_password("s3cret"), "admin"),
    )
    return am, db, uid


def test_password_hash_roundtrip():
    am, _, _ = _fresh_auth()
    h = am.hash_password("hunter2")
    assert h != "hunter2" and am.verify_password("hunter2", h)
    assert not am.verify_password("wrong", h)


def test_token_roundtrip():
    am, _, uid = _fresh_auth()
    payload = am.verify_token(am.create_access_token({"sub": uid}))
    assert payload["sub"] == uid and payload["type"] == "access"


def test_authenticate_user():
    am, db, _ = _fresh_auth()
    assert am.authenticate_user("alice", "s3cret")["role"] == "admin"
    assert am.authenticate_user("alice", "wrong") is None
    assert am.authenticate_user("ghost", "s3cret") is None
    # inactive users can't authenticate
    db.execute_query("UPDATE users SET is_active=0 WHERE username='alice'")
    assert am.authenticate_user("alice", "s3cret") is None


def test_route_guard(tmp_path):
    pytest.importorskip("httpx")  # starlette TestClient needs httpx
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient

    # File DB (not :memory:) — TestClient runs the route in a worker thread, and the
    # app path uses a file DB with thread-local connections anyway.
    am, _, uid = _fresh_auth(str(tmp_path / "auth.db"))
    app = FastAPI()

    @app.get("/protected")
    def protected(user: dict = Depends(am.get_current_user)):
        return {"who": user["username"]}

    client = TestClient(app)
    assert client.get("/protected").status_code in (401, 403)  # no token
    assert client.get("/protected", headers={"Authorization": "Bearer garbage"}).status_code == 401
    token = am.create_access_token({"sub": uid})
    ok = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.json()["who"] == "alice"


if __name__ == "__main__":
    test_password_hash_roundtrip()
    test_token_roundtrip()
    test_authenticate_user()
    print("auth primitive checks passed")
