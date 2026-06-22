"""Ed25519 detached signing for Open Decision Receipts (ODR-2, issue #8225).

The verifier shipped first: ``aragora-verify`` (PR #8388) and the
``/api/v2/receipts/*/verify-signature`` endpoints can already *verify* a
signed ODR receipt, but until now nothing on main could *produce* one — a
consumer with no producer. This module is that producer.

It is deliberately written against the verifier's exact contract
(``aragora_verify.verifier`` / ``aragora_verify.jcs``) so producer and
consumer are guaranteed compatible:

    digest_hex = SHA-256( JCS(receipt without "signatures") )      # hex
    message    = bytes.fromhex(digest_hex)                          # 32 raw bytes
    signature  = Ed25519_sign(private_key, message)                # 64 raw bytes
    key_id     = "ed25519-" + SHA-256(raw_public_key).hexdigest()[:16]
    entry      = {"alg": "Ed25519", "key_id": key_id, "signature": base64(signature)}

The digest is computed with :func:`aragora.gauntlet.odr_export.odr_content_digest`,
which the verifier's own docstring states it "mirrors exactly". Excluding the
``signatures`` array from the digest is what makes the signatures *detached*:
attaching one never changes the bytes it covers.

Key management (per the post-incident security architecture):
    The private key is NEVER read from a raw environment variable or committed
    to the repo. It is resolved from AWS Secrets Manager via
    :mod:`aragora.config.secrets` (PEM in the secret named by
    ``ARAGORA_ODR_SIGNING_KEY_SECRET``, default ``aragora/odr-signing-key``).
    Only the *public* key is published (repo + a ``.well-known`` endpoint).
    A loader from explicit PEM bytes is provided for tests and offline tooling.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import TYPE_CHECKING, Any

from aragora.gauntlet.odr_export import odr_content_digest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

logger = logging.getLogger(__name__)

#: Algorithm token recorded in each signature entry (schema-required; the
#: verifier accepts ODR signatures as Ed25519).
ODR_SIGNATURE_ALG = "Ed25519"

#: Name of the AWS Secrets Manager secret holding the PEM private key.
DEFAULT_SIGNING_KEY_SECRET = "aragora/odr-signing-key"
SIGNING_KEY_SECRET_ENV = "ARAGORA_ODR_SIGNING_KEY_SECRET"


class OdrSigningError(Exception):
    """Raised when a private key cannot be loaded or a receipt cannot be signed."""


def _load_ed25519():  # noqa: ANN202 - lazy import keeps the error actionable
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OdrSigningError(
            "the 'cryptography' package is required for ODR signing; "
            "install it (already an Aragora dependency) to sign receipts"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization, InvalidSignature


def compute_key_id(public_key: Ed25519PublicKey) -> str:
    """``ed25519-`` + first 16 hex of SHA-256(raw public key).

    Mirrors ``aragora_verify.verifier.compute_key_id`` exactly so a signed
    receipt's ``key_id`` matches what the verifier derives from the public key.
    """
    _, _, serialization, _ = _load_ed25519()
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return "ed25519-" + hashlib.sha256(raw).hexdigest()[:16]


def load_private_key_from_pem(pem: str | bytes) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from PEM (for tests/offline tooling).

    Production callers should prefer :func:`load_signing_key_from_secrets`,
    which never lets the key material transit a raw environment variable.
    """
    Ed25519PrivateKey, _, serialization, _ = _load_ed25519()
    data = pem.encode("utf-8") if isinstance(pem, str) else pem
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as exc:
        raise OdrSigningError("could not parse Ed25519 private key from PEM") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise OdrSigningError(
            f"private key is not Ed25519 (got {type(key).__name__}); "
            "ODR signatures use Ed25519"
        )
    return key


def load_signing_key_from_secrets(
    secret_name: str | None = None,
) -> Ed25519PrivateKey:
    """Resolve the ODR signing key from AWS Secrets Manager.

    The key PEM is fetched via :mod:`aragora.config.secrets` (the same path
    used for every other Aragora secret), never from a raw env var. The env
    var only *names* which secret to read.
    """
    name = secret_name or os.environ.get(SIGNING_KEY_SECRET_ENV) or DEFAULT_SIGNING_KEY_SECRET
    try:
        from aragora.config.secrets import get_secret
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OdrSigningError("aragora.config.secrets is unavailable") from exc
    pem = get_secret(name)
    if not pem:
        raise OdrSigningError(
            f"ODR signing key secret '{name}' is empty or unset; "
            "provision the PEM private key in Secrets Manager"
        )
    return load_private_key_from_pem(pem)


def generate_signing_key() -> Ed25519PrivateKey:
    """Generate a fresh Ed25519 private key (key-rotation / bootstrap tooling)."""
    Ed25519PrivateKey, _, _, _ = _load_ed25519()
    return Ed25519PrivateKey.generate()


def public_key_pem(private_key: Ed25519PrivateKey) -> str:
    """Return the PEM SubjectPublicKeyInfo for the private key's public half.

    This is the artifact to publish (repo + ``.well-known``) so any third party
    can verify receipts offline of Aragora.
    """
    _, _, serialization, _ = _load_ed25519()
    pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem.decode("utf-8")


def sign_odr_receipt(
    odr: dict[str, Any],
    private_key: Ed25519PrivateKey,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Attach an Ed25519 detached signature to an ODR receipt.

    Args:
        odr: An ODR profile dict (as produced by
            :func:`aragora.gauntlet.odr_export.decision_receipt_to_odr`). The
            input is not mutated; a new dict is returned.
        private_key: The Ed25519 signing key.
        replace: When True, drop any existing signatures before appending
            (re-sign). When False (default), append alongside existing ones —
            the digest excludes ``signatures``, so this never invalidates a
            prior signature.

    Returns:
        A copy of ``odr`` with a ``{"alg", "key_id", "signature"}`` entry
        appended to its ``signatures`` array. The signature covers
        ``bytes.fromhex(odr_content_digest(odr))`` — exactly what the verifier
        re-derives and checks.
    """
    signed = dict(odr)

    existing = odr.get("signatures")
    signatures: list[Any] = [] if replace or not isinstance(existing, list) else list(existing)

    # The digest excludes the signatures array (detached) — compute it against
    # the payload as the verifier will, regardless of what's already attached.
    digest_hex = odr_content_digest(signed)
    message = bytes.fromhex(digest_hex)

    signature_bytes = private_key.sign(message)
    entry = {
        "alg": ODR_SIGNATURE_ALG,
        "key_id": compute_key_id(private_key.public_key()),
        "signature": base64.b64encode(signature_bytes).decode("ascii"),
    }
    signatures.append(entry)
    signed["signatures"] = signatures
    return signed


__all__ = [
    "ODR_SIGNATURE_ALG",
    "OdrSigningError",
    "compute_key_id",
    "generate_signing_key",
    "load_private_key_from_pem",
    "load_signing_key_from_secrets",
    "public_key_pem",
    "sign_odr_receipt",
]
