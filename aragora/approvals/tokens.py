"""Signed approval action tokens for chat interactions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SECRET: bytes | None = None
_SECRET_INSECURE: bool = False


@dataclass(frozen=True)
class ApprovalActionToken:
    """Decoded approval action token."""

    kind: str
    target_id: str
    action: str
    issued_at: int
    expires_at: int | None = None
    nonce: str | None = None
    insecure: bool = False

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return int(time.time()) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "action": self.action,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
            "insecure": self.insecure,
        }


def _get_secret() -> tuple[bytes, bool]:
    """Return signing secret and whether it is insecure/ephemeral."""
    global _SECRET, _SECRET_INSECURE
    if _SECRET is not None:
        return _SECRET, _SECRET_INSECURE

    env_secret = os.environ.get("ARAGORA_APPROVAL_ACTION_SECRET")
    if env_secret:
        _SECRET = env_secret.encode("utf-8")
        _SECRET_INSECURE = False
        return _SECRET, _SECRET_INSECURE

    # Fallback: generate ephemeral secret (tokens invalid after restart).
    _SECRET = secrets.token_bytes(32)
    _SECRET_INSECURE = True
    logger.warning(
        "ARAGORA_APPROVAL_ACTION_SECRET not set; using ephemeral secret. "
        "Approval tokens will be invalid after restart."
    )
    return _SECRET, _SECRET_INSECURE


def _sign(payload: bytes, secret: bytes) -> str:
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return digest[:16]


def _encode_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    return token


def _decode_payload(encoded: str) -> dict[str, Any] | None:
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode(encoded + padding)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def encode_approval_action(
    *,
    kind: str,
    target_id: str,
    action: str,
    ttl_seconds: int | None = 3600,
    issued_at: int | None = None,
    expires_at: int | None = None,
    nonce: str | None = None,
) -> str | None:
    """Encode a signed approval action token."""
    if not kind or not target_id or not action:
        return None

    issued = issued_at or int(time.time())
    exp = expires_at
    if exp is None and ttl_seconds:
        exp = issued + int(ttl_seconds)

    payload = {
        "k": kind,
        "t": target_id,
        "a": action,
        "iat": issued,
        "n": nonce or secrets.token_hex(4),
    }
    if exp is not None:
        payload["exp"] = exp

    secret, _insecure = _get_secret()
    encoded = _encode_payload(payload)
    sig = _sign(encoded.encode("utf-8"), secret)
    return f"{encoded}.{sig}"


def decode_approval_action(
    token: str,
    *,
    allow_expired: bool = False,
) -> ApprovalActionToken | None:
    """Decode and verify an approval action token."""
    if not token or "." not in token:
        return None

    encoded, sig = token.rsplit(".", 1)
    payload = _decode_payload(encoded)
    if payload is None:
        return None

    secret, insecure = _get_secret()
    expected = _sign(encoded.encode("utf-8"), secret)
    if not hmac.compare_digest(expected, sig):
        logger.warning("Approval token signature mismatch")
        return None

    kind = str(payload.get("k") or "")
    target_id = str(payload.get("t") or "")
    action = str(payload.get("a") or "")
    issued_at = int(payload.get("iat") or 0)
    expires_at = payload.get("exp")
    if expires_at is not None:
        try:
            expires_at = int(expires_at)
        except (TypeError, ValueError):
            expires_at = None

    token_obj = ApprovalActionToken(
        kind=kind,
        target_id=target_id,
        action=action,
        issued_at=issued_at,
        expires_at=expires_at,
        nonce=payload.get("n"),
        insecure=insecure,
    )

    if token_obj.is_expired and not allow_expired:
        return None

    return token_obj


__all__ = [
    "ApprovalActionToken",
    "encode_approval_action",
    "decode_approval_action",
]
