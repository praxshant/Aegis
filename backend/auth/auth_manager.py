import os
import logging
import secrets as _secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
import bcrypt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.data.database_manager import DatabaseManager

logger = logging.getLogger("AegisAuth")


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

    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

    def create_access_token(self, data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
                          "type": "access"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        to_encode.update({"exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire_days),
                          "type": "refresh"})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

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
        return {"user_id": user["user_id"], "username": user["username"],
                "email": user["email"], "role": user["role"]}

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        payload = self.verify_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        query = "SELECT user_id, username, email, role, is_active FROM users WHERE user_id = ?"
        df = self.db_manager.query_to_dataframe(query, (user_id,))
        if df is None or df.empty:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        user = df.iloc[0].to_dict()
        if not user.get("is_active"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
        return user

    def require_role(self, required_role: str):
        def role_checker(current_user: dict = Depends(self.get_current_user)):
            if current_user["role"] != required_role and current_user["role"] != "admin":
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
            return current_user
        return role_checker


# Global singleton — uses its own DatabaseManager (same aegis.db file as the app).
auth_manager = AuthManager()
