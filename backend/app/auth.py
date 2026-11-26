"""
Nidaan-Rx v2.0 — Strict PBKDF2-SHA256 + HMAC-SHA256 JWT Auth with Server-Side Revocation.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException

SECRET_FILE = Path("data/.auth_secret")
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days
SERVER_BOOT_ID = secrets.token_hex(8)


def _get_secret() -> bytes:
    env_secret = os.getenv("NIDAAN_AUTH_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text(encoding="utf-8").strip().encode("utf-8")
    generated = secrets.token_hex(32)
    SECRET_FILE.write_text(generated, encoding="utf-8")
    return generated.encode("utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        alg, iters, salt_hex, dk_hex = stored_hash.split("$")
        if alg != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iters),
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def create_access_token(
    user_id: str,
    email: str,
    full_name: str,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> str:
    now = int(time.time())
    header = _b64u_encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8")
    )
    payload = _b64u_encode(
        json.dumps(
            {
                "jti": uuid.uuid4().hex,
                "boot": SERVER_BOOT_ID,
                "sub": user_id,
                "email": email,
                "name": full_name,
                "iat": now,
                "exp": now + int(ttl_seconds),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    sig = _b64u_encode(hmac.new(_get_secret(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def verify_access_token(token: str) -> dict[str, Any] | None:
    try:
        header_b64, payload_b64, sig_b64 = token.strip().split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = _b64u_encode(hmac.new(_get_secret(), signing_input, hashlib.sha256).digest())
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        payload = json.loads(_b64u_decode(payload_b64).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        if not payload.get("sub"):
            return None
        if payload.get("boot") != SERVER_BOOT_ID:
            return None
        return payload
    except Exception:
        return None


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def require_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in required to access this feature.")
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please sign in again.")

    # Check server-side revocation list + confirm user still exists
    from app.main import get_case_repo

    repo = get_case_repo()
    jti = payload.get("jti")
    if jti and repo.is_token_revoked(str(jti)):
        raise HTTPException(status_code=401, detail="Session has been logged out. Please sign in again.")
    if repo.get_user_by_id(str(payload["sub"])) is None:
        raise HTTPException(status_code=401, detail="Account not found. Please sign in again.")
    return payload
