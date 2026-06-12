"""Ed25519 detached signing for Open Decision Receipts (ODR-2, #8225).

Signing construction (spec ``docs/specs/OPEN_DECISION_RECEIPT.md`` §5-§6):

    message   = SHA-256( JCS(odr document with ``signatures`` removed) )
    signature = Ed25519-sign(private_key, message)

The signature is *detached*: it rides in the receipt's reserved
``signatures[]`` array (``alg``, ``key_id``, ``signature``, ``signed_at``)
without changing the content digest those signatures cover —
``odr_content_digest`` already excludes the array.

A third party verifies with **only the receipt JSON and the public key**:
recompute the JCS digest, base64-decode the signature, Ed25519-verify.
No shared secret and no Aragora deployment are required, which is the
difference between this path and the HMAC store-integrity signing in
:mod:`aragora.gauntlet.signing` (which remains for internal use).

Key handling follows the post-incident credential architecture: the
private key is a 32-byte Ed25519 seed held in AWS Secrets Manager under
``ARAGORA_ODR_SIGNING_KEY`` (base64 or hex), loaded through
:func:`aragora.config.secrets.get_secret` — never committed and never
passed around as a raw environment value. The public key is safe to
publish anywhere; ``key_id`` is derived from it so verifiers can match
keys without extra metadata. Rotation guidance lives in
``docs/keys/README.md``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.config.secrets import get_secret
from aragora.gauntlet.odr_export import odr_content_digest
from aragora.gauntlet.signing import CRYPTO_AVAILABLE, Ed25519Signer

logger = logging.getLogger(__name__)

ODR_SIGNING_KEY_SECRET = "ARAGORA_ODR_SIGNING_KEY"
ODR_SIGNATURE_ALG = "Ed25519"

__all__ = [
    "ODR_SIGNATURE_ALG",
    "ODR_SIGNING_KEY_SECRET",
    "ODRVerification",
    "derive_key_id",
    "export_private_seed_b64",
    "generate_odr_keypair",
    "load_odr_signer",
    "odr_signing_message",
    "public_key_b64",
    "sign_odr",
    "signer_from_seed",
    "verify_odr",
]


def odr_signing_message(odr: dict[str, Any]) -> bytes:
    """The exact bytes an ODR signature covers.

    Raw SHA-256 digest of the JCS canonical bytes of the document with the
    ``signatures`` array removed (spec §5-§6). Byte-stable across key order,
    whitespace, and platforms.
    """
    return bytes.fromhex(odr_content_digest(odr))


def derive_key_id(public_key_raw: bytes) -> str:
    """Stable key identifier derived from the raw 32-byte public key."""
    return f"ed25519-{hashlib.sha256(public_key_raw).hexdigest()[:16]}"


def _require_crypto() -> None:
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography package required for Ed25519 ODR signing")


def signer_from_seed(seed: bytes) -> Ed25519Signer:
    """Build an :class:`Ed25519Signer` from a 32-byte private seed."""
    _require_crypto()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    if len(seed) != 32:
        raise ValueError(f"Ed25519 seed must be exactly 32 bytes, got {len(seed)}")
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key()
    raw_pub = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return Ed25519Signer(
        private_key=private_key,
        public_key=public_key,
        key_id=derive_key_id(raw_pub),
    )


def generate_odr_keypair() -> Ed25519Signer:
    """Generate a fresh keypair with a public-key-derived ``key_id``."""
    _require_crypto()
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    raw_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return Ed25519Signer(
        private_key=private_key,
        public_key=private_key.public_key(),
        key_id=derive_key_id(raw_pub),
    )


def public_key_b64(signer: Ed25519Signer) -> str:
    """Base64 of the raw 32-byte public key — the publishable artifact."""
    from cryptography.hazmat.primitives import serialization

    public_key = getattr(signer, "_public_key", None)
    if public_key is None:
        raise ValueError("Signer has no public key")
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def export_private_seed_b64(signer: Ed25519Signer) -> str:
    """Base64 of the raw 32-byte private seed — FOR PROVISIONING ONLY.

    The only legitimate destination for this value is a secrets manager
    (``ARAGORA_ODR_SIGNING_KEY``). Never log it, never write it to the repo
    or a shell profile.
    """
    from cryptography.hazmat.primitives import serialization

    private_key = getattr(signer, "_private_key", None)
    if private_key is None:
        raise ValueError("Signer has no private key")
    seed = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return base64.b64encode(seed).decode("ascii")


def _decode_key_material(value: str) -> bytes:
    """Decode base64- or hex-encoded key material."""
    text = value.strip()
    try:
        return bytes.fromhex(text)
    except ValueError:
        pass
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Key material is neither valid hex nor base64") from exc


def load_odr_signer() -> Ed25519Signer | None:
    """Load the ODR signer from the secrets layer.

    Returns None when no key is configured (signing is opt-in until the
    operator provisions ``ARAGORA_ODR_SIGNING_KEY`` in Secrets Manager).

    Raises:
        ValueError: if the secret exists but is not a valid 32-byte seed.
    """
    encoded = get_secret(ODR_SIGNING_KEY_SECRET)
    if not encoded:
        return None
    seed = _decode_key_material(encoded)
    return signer_from_seed(seed)


def sign_odr(
    odr: dict[str, Any],
    *,
    signer: Ed25519Signer,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``odr`` with a detached Ed25519 signature appended.

    The input document is not mutated; existing signatures are preserved so
    multiple parties can co-sign the same canonical content.
    """
    message = odr_signing_message(odr)
    signature = base64.b64encode(signer.sign(message)).decode("ascii")
    entry = {
        "alg": ODR_SIGNATURE_ALG,
        "key_id": signer.key_id,
        "signature": signature,
        "signed_at": signed_at or datetime.now(timezone.utc).isoformat(),
    }
    signed = dict(odr)
    signed["signatures"] = [dict(s) for s in odr.get("signatures", [])] + [entry]
    return signed


@dataclass
class ODRVerification:
    """Outcome of offline ODR signature verification."""

    valid: bool
    reason: str
    verified_key_ids: list[str] = field(default_factory=list)
    failed_key_ids: list[str] = field(default_factory=list)
    skipped_key_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "verified_key_ids": self.verified_key_ids,
            "failed_key_ids": self.failed_key_ids,
            "skipped_key_ids": self.skipped_key_ids,
        }


def verify_odr(odr: dict[str, Any], *, public_key: str | bytes) -> ODRVerification:
    """Verify ODR signatures with only the receipt JSON and a public key.

    Signatures whose ``key_id`` does not match the supplied key are skipped
    (they may belong to other co-signers); the result is valid when at least
    one signature matches the key and verifies, and no matching signature
    fails.
    """
    _require_crypto()
    from cryptography.hazmat.primitives.asymmetric import ed25519

    raw = public_key if isinstance(public_key, bytes) else _decode_key_material(public_key)
    if len(raw) != 32:
        raise ValueError(f"Ed25519 public key must be exactly 32 bytes, got {len(raw)}")
    key = ed25519.Ed25519PublicKey.from_public_bytes(raw)
    key_id = derive_key_id(raw)
    verifier = Ed25519Signer(public_key=key, key_id=key_id)

    signatures = odr.get("signatures") or []
    if not signatures:
        return ODRVerification(valid=False, reason="receipt carries no signatures")

    message = odr_signing_message(odr)
    verified: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for entry in signatures:
        entry_key_id = str(entry.get("key_id", ""))
        if entry.get("alg") != ODR_SIGNATURE_ALG or entry_key_id != key_id:
            skipped.append(entry_key_id)
            continue
        try:
            signature = base64.b64decode(str(entry.get("signature", "")), validate=True)
        except (binascii.Error, ValueError):
            failed.append(entry_key_id)
            continue
        if verifier.verify(message, signature):
            verified.append(entry_key_id)
        else:
            failed.append(entry_key_id)

    if failed:
        reason = "signature verification failed"
    elif verified:
        reason = "all matching signatures verified"
    else:
        reason = "no signatures match the supplied public key"
    return ODRVerification(
        valid=bool(verified) and not failed,
        reason=reason,
        verified_key_ids=verified,
        failed_key_ids=failed,
        skipped_key_ids=skipped,
    )
