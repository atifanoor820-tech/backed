"""Accounts and sign-in: password hashing, sessions, and the admin account.

Passwords are stored as PBKDF2-SHA256 hashes. A session token is random; only its SHA-256 is stored.
"""

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from agent.config import get_settings
from database.models import AuthSession, User
from database.session import session_scope

_ITERATIONS = 310_000


class AuthError(ValueError):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, digest)


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def public(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "created_at": u.created_at.isoformat()}


def create_user(email: str, password: str, name: str = "", role: str = "user") -> dict:
    email = _norm_email(email)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise AuthError("Enter a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    with session_scope() as s:
        if s.scalar(select(User.id).where(User.email == email)):
            raise AuthError("An account with this email already exists. Log in instead.")
        u = User(id=str(uuid.uuid4()), email=email, name=(name or "").strip()[:100],
                 password_hash=hash_password(password), role=role)
        s.add(u)
        s.flush()
        return public(u)


def authenticate(email: str, password: str) -> dict:
    with session_scope() as s:
        u = s.scalar(select(User).where(User.email == _norm_email(email)))
        if u is None or not verify_password(password, u.password_hash):
            raise AuthError("Wrong email or password.")
        return public(u)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    with session_scope() as s:
        s.add(AuthSession(token_hash=_hash_token(token), user_id=user_id, created_at=now,
                          expires_at=now + timedelta(days=get_settings().session_days)))
    return token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    with session_scope() as s:
        sess = s.get(AuthSession, _hash_token(token))
        if sess is None:
            return None
        expires = sess.expires_at if sess.expires_at.tzinfo else sess.expires_at.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            s.delete(sess)
            return None
        u = s.get(User, sess.user_id)
        return public(u) if u else None


def end_session(token: str | None) -> None:
    if token:
        with session_scope() as s:
            s.execute(delete(AuthSession).where(AuthSession.token_hash == _hash_token(token)))


def list_users() -> list[dict]:
    with session_scope() as s:
        return [public(u) for u in s.scalars(select(User).order_by(User.created_at))]


def ensure_admin() -> dict | None:
    """Create or update the admin from ABBAS_ADMIN_EMAIL / ABBAS_ADMIN_PASSWORD. On first creation the admin
    also gets the trading accounts and profiles saved before sign-in existed (the old shared Dashi config)."""
    st = get_settings()
    email, password = _norm_email(st.admin_email), st.admin_password
    if not email or not password:
        return None
    with session_scope() as s:
        u = s.scalar(select(User).where(User.email == email))
        created = u is None
        if created:
            u = User(id=str(uuid.uuid4()), email=email, name="Admin", password_hash=hash_password(password), role="admin")
            s.add(u)
        else:
            u.role = "admin"
            if not verify_password(password, u.password_hash):
                u.password_hash = hash_password(password)
        s.flush()
        admin = public(u)
    if created:
        from agent import config_store
        from agent.dashi_config import CONFIG_KEY

        legacy = config_store.load(CONFIG_KEY)
        if legacy and config_store.load(f"{CONFIG_KEY}:{admin['id']}") is None:
            config_store.save(f"{CONFIG_KEY}:{admin['id']}", legacy)
    return admin
