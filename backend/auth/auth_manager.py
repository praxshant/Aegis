import os
import time
import logging
import threading
import secrets as _secrets
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Iterable

import jwt
import bcrypt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.data.database_manager import DatabaseManager

logger = logging.getLogger("AegisAuth")

# Roles allowed to trigger spend or change state. 'viewer' (the users-table default)
# can sign in and read, but must not be able to create payment links.
OPERATOR_ROLES = ("admin", "operator")

_BCRYPT_MAX_BYTES = 72  # bcrypt>=5 raises ValueError past this instead of truncating


class LoginThrottle:
    """In-process brute-force guard: after `max_failures` failed logins for one username
    within `window_s`, further attempts get 429 until the window rolls forward."""

    def __init__(self, max_failures: int = 5, window_s: int = 300):
        self.max_failures = max_failures
        self.window_s = window_s
        self._failures: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        q = self._failures[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def check(self, key: str) -> None:
        with self._lock:
            if len(self._prune(key, time.monotonic())) >= self.max_failures:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                    detail="Too many failed login attempts. Try again in a few minutes.")

    def fail(self, key: str) -> None:
        with self._lock:
            self._prune(key, time.monotonic()).append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class AuthManager:
    """JWT (HS256) + bcrypt auth, ported from CICOP. Users live in the `users`
    table created by DatabaseManager. Differs from the CICOP original in two ways:
    the secret never falls back to a shipped constant, and the DatabaseManager is
    injectable so tests/app can share one connection."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        secret = os.getenv("JWT_SECRET_KEY")
        if not secret:
            # CICOP shipped a hardcoded "change-this-in-production" default — a forgeable
            # signing key. Fail safe to an ephemeral per-process key instead.
            # ponytail: ephemeral secret; set JWT_SECRET_KEY so tokens survive a restart.
            secret = _secrets.token_urlsafe(48)
            logger.warning(
                "JWT_SECRET_KEY not set — using an ephemeral per-process secret. "
                "Tokens will not survive a restart. Set JWT_SECRET_KEY in .env for production."
            )
        self.secret_key = secret
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.db_manager = db_manager or DatabaseManager()
        self.security = HTTPBearer()
        self.throttle = LoginThrottle()

    def hash_password(self, password: str) -> str:
        if len(password.encode("utf-8")) > _BCRYPT_MAX_BYTES:
            raise ValueError(f"Password must be at most {_BCRYPT_MAX_BYTES} bytes (bcrypt limit).")
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except ValueError:
            # >72-byte input or a malformed stored hash: a failed login, not a 500
            return False

    def _encode(self, data: Dict[str, Any], lifetime: timedelta, token_type: str) -> str:
        to_encode = data.copy()
        to_encode.update({"exp": datetime.now(timezone.utc) + lifetime, "type": token_type})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_access_token(self, data: Dict[str, Any]) -> str:
        return self._encode(data, timedelta(minutes=self.access_token_expire_minutes), "access")

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        return self._encode(data, timedelta(days=self.refresh_token_expire_days), "refresh")

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    def _load_user(self, user_id: str) -> Dict[str, Any]:
        query = "SELECT user_id, username, email, role, is_active FROM users WHERE user_id = ?"
        df = self.db_manager.query_to_dataframe(query, (user_id,))
        if df is None or df.empty:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        user = df.iloc[0].to_dict()
        if not user.get("is_active"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
        # plain Python types so the dict is JSON-serialisable (pandas hands back numpy ints)
        return {"user_id": str(user["user_id"]), "username": str(user["username"]),
                "email": None if user.get("email") is None else str(user["email"]),
                "role": str(user["role"]), "is_active": bool(user["is_active"])}

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        query = "SELECT user_id, username, email, password_hash, role, is_active FROM users WHERE username = ?"
        df = self.db_manager.query_to_dataframe(query, (username,))
        if df is None or df.empty:
            return None
        user = df.iloc[0].to_dict()
        if not user.get("is_active"):
            return None
        if not self.verify_password(password, user["password_hash"]):
            return None
        try:
            self.db_manager.execute_query("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                                          (user["user_id"],))
        except Exception as e:  # bookkeeping only; never block a valid login
            logger.warning(f"Could not record last_login: {e}")
        return {"user_id": user["user_id"], "username": user["username"],
                "email": user["email"], "role": user["role"]}

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        payload = self.verify_token(refresh_token)
        if payload.get("type") != "refresh" or not payload.get("sub"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user = self._load_user(payload["sub"])  # deactivated users can't refresh
        return {"access_token": self.create_access_token({"sub": user["user_id"]}), "token_type": "bearer"}

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        payload = self.verify_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return self._load_user(user_id)

    def require_roles(self, roles: Iterable[str]):
        allowed = set(roles)

        def role_checker(current_user: dict = Depends(self.get_current_user)):
            if current_user["role"] not in allowed:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
            return current_user
        return role_checker

    def require_role(self, required_role: str):
        return self.require_roles({required_role, "admin"})


# Global singleton — uses its own DatabaseManager (same aegis.db file as the app).
auth_manager = AuthManager()
