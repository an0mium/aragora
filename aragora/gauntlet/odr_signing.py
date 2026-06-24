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
import copy
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


# --- Post-quantum (ML-DSA / FIPS 204) hybrid signing (#8602) -----------------
#
# Decision receipts are harvest-now-decrypt-later critical: their signatures
# must verify for years, so an Ed25519-only signature is born quantum-vulnerable.
# We add a *hybrid* path that attaches BOTH an Ed25519 and an ML-DSA detached
# signature over the same content digest. Existing verifiers keep working on the
# Ed25519 entry; the ML-DSA entry provides quantum-resistant assurance. ML-DSA-65
# is FIPS 204 NIST security level 3 (the recommended general-purpose set).
ODR_PQC_SIGNATURE_ALG = "ML-DSA-65"


def _load_mldsa():  # noqa: ANN202 - lazy import keeps the error actionable
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import mldsa
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OdrSigningError(
            "post-quantum ODR signing requires `cryptography` >= 49 built against "
            "OpenSSL 3.5+/AWS-LC/BoringSSL (the `mldsa` module); upgrade `cryptography` "
            "to enable ML-DSA hybrid signatures"
        ) from exc
    return mldsa, InvalidSignature


def pqc_available() -> bool:
    """Whether the runtime ``cryptography`` build exposes ML-DSA (FIPS 204).

    Mirrors the graceful-degradation pattern used elsewhere: callers can hybrid-
    sign when this is True and fall back to Ed25519-only when it is False.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import mldsa  # noqa: F401

        return True
    except ImportError:
        return False


def compute_mldsa_key_id(public_key: Any) -> str:
    """``ml-dsa-65-`` + first 16 hex of SHA-256(raw ML-DSA public key).

    Parallels :func:`compute_key_id` for Ed25519 so an ML-DSA signature entry's
    ``key_id`` is reproducible from the published public key.
    """
    raw = public_key.public_bytes_raw()
    return "ml-dsa-65-" + hashlib.sha256(raw).hexdigest()[:16]


def generate_mldsa_signing_key() -> Any:
    """Generate a fresh ML-DSA-65 private key (key-rotation / bootstrap tooling)."""
    mldsa, _ = _load_mldsa()
    return mldsa.MLDSA65PrivateKey.generate()


def load_mldsa_key_from_seed(seed: bytes) -> Any:
    """Load an ML-DSA-65 private key from its 32-byte seed (held in Secrets Manager)."""
    if len(seed) != 32:
        raise OdrSigningError(f"ML-DSA-65 seed must be exactly 32 bytes, got {len(seed)}")
    mldsa, _ = _load_mldsa()
    return mldsa.MLDSA65PrivateKey.from_seed_bytes(seed)


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
            f"private key is not Ed25519 (got {type(key).__name__}); ODR signatures use Ed25519"
        )
    return key


def _secret_id_label(secret_id: str) -> str:
    """Return a log-safe secret identifier label."""
    if "-----BEGIN" in secret_id or "\n" in secret_id or len(secret_id) > 160:
        return "<redacted-secret-id>"
    return secret_id


def _secret_id_contains_key_material(secret_id: str) -> bool:
    return "-----BEGIN" in secret_id or "PRIVATE KEY" in secret_id or "\n" in secret_id


def _load_pem_secret_from_aws(secret_id: str) -> str:
    """Fetch a standalone PEM secret from AWS Secrets Manager.

    This intentionally does not call ``get_secret(secret_id)`` because that API
    looks up a key inside Aragora's configured JSON secret bundle and may fall
    back to environment variables in non-strict local mode. ODR signing keys are
    standalone custody material: the environment may name the SecretId, but it
    must never carry the raw private key.
    """
    secret_label = _secret_id_label(secret_id)
    if _secret_id_contains_key_material(secret_id):
        raise OdrSigningError(
            "ODR signing key secret identifier appears to contain raw key material; "
            "set ARAGORA_ODR_SIGNING_KEY_SECRET to an AWS Secrets Manager SecretId"
        )

    try:
        from aragora.config import secrets as secret_config
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise OdrSigningError("aragora.config.secrets is unavailable") from exc

    config = secret_config.SecretsConfig.from_env()
    if not config.use_aws:
        raise OdrSigningError(
            "AWS Secrets Manager is not enabled for ODR signing; set "
            "ARAGORA_USE_SECRETS_MANAGER=true and provision the PEM private key "
            f"in secret '{secret_label}'"
        )

    manager = secret_config.SecretManager(config)
    regions = config.aws_regions or [config.aws_region]
    last_error: Exception | None = None
    for region in regions:
        client = manager._get_aws_client(region)  # noqa: SLF001 - reuse repo AWS client setup.
        if client is None:
            continue
        try:
            response = client.get_secret_value(SecretId=secret_id)
        except (secret_config.ClientError, secret_config.BotoCoreError) as exc:
            last_error = exc
            continue
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            last_error = exc
            continue

        secret_string = response.get("SecretString")
        if isinstance(secret_string, str) and secret_string.strip():
            return secret_string

        secret_binary = response.get("SecretBinary")
        if isinstance(secret_binary, bytes) and secret_binary:
            try:
                return secret_binary.decode("utf-8")
            except UnicodeDecodeError:
                last_error = OdrSigningError(
                    f"ODR signing key secret '{secret_label}' binary value is not UTF-8 PEM"
                )
                continue

        last_error = OdrSigningError(f"ODR signing key secret '{secret_label}' is empty")
        continue

    detail = f" (last error: {type(last_error).__name__})" if last_error else ""
    raise OdrSigningError(
        f"ODR signing key secret '{secret_label}' could not be read from AWS Secrets Manager{detail}"
    )


def load_signing_key_from_secrets(
    secret_name: str | None = None,
) -> Ed25519PrivateKey:
    """Resolve the ODR signing key from AWS Secrets Manager.

    The key PEM is fetched via :mod:`aragora.config.secrets` (the same path
    used for every other Aragora secret), never from a raw env var. The env
    var only *names* which secret to read.
    """
    name = secret_name or os.environ.get(SIGNING_KEY_SECRET_ENV) or DEFAULT_SIGNING_KEY_SECRET
    pem = _load_pem_secret_from_aws(name)
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
    signed = copy.deepcopy(odr)
    signatures = _existing_signatures(signed, replace=replace)
    signatures.append(
        {
            "alg": ODR_SIGNATURE_ALG,
            "key_id": compute_key_id(private_key.public_key()),
            "signature": _b64(private_key.sign(_odr_signing_message(signed))),
        }
    )
    signed["signatures"] = signatures
    return signed


def sign_odr_receipt_mldsa(
    odr: dict[str, Any],
    private_key: Any,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Attach a post-quantum **ML-DSA-65** (FIPS 204) detached signature.

    Identical envelope to :func:`sign_odr_receipt` (same content digest, same
    ``{"alg", "key_id", "signature"}`` shape) — just a quantum-resistant
    algorithm. Appends by default so it can sit alongside an Ed25519 entry.
    """
    signed = copy.deepcopy(odr)
    signatures = _existing_signatures(signed, replace=replace)
    signatures.append(
        {
            "alg": ODR_PQC_SIGNATURE_ALG,
            "key_id": compute_mldsa_key_id(private_key.public_key()),
            "signature": _b64(private_key.sign(_odr_signing_message(signed))),
        }
    )
    signed["signatures"] = signatures
    return signed


def sign_odr_hybrid(
    odr: dict[str, Any],
    *,
    ed25519_key: Ed25519PrivateKey,
    mldsa_key: Any,
    replace: bool = False,
) -> dict[str, Any]:
    """Attach BOTH a classical Ed25519 and a post-quantum ML-DSA detached signature.

    Both signatures cover the same content digest, so the receipt stays verifiable
    by existing Ed25519 tooling **and** gains quantum-resistant assurance — the
    harvest-now-decrypt-later defense for long-lived audit receipts (#8602).
    """
    signed = sign_odr_receipt(odr, ed25519_key, replace=replace)
    return sign_odr_receipt_mldsa(signed, mldsa_key, replace=False)


def verify_odr_signature_ed25519(odr: dict[str, Any], *, public_key: Ed25519PublicKey) -> bool:
    """In-package Ed25519 verify (round-trip / offline tooling). Returns True if any
    Ed25519 entry whose key_id matches ``public_key`` verifies over the content digest."""
    _, _, _, InvalidSignature = _load_ed25519()
    return _verify_entries(
        odr,
        alg=ODR_SIGNATURE_ALG,
        key_id=compute_key_id(public_key),
        verify=public_key.verify,
        invalid_signature=InvalidSignature,
    )


def verify_odr_signature_mldsa(odr: dict[str, Any], *, public_key: Any) -> bool:
    """In-package ML-DSA-65 verify. Returns True if any ML-DSA entry whose key_id
    matches ``public_key`` verifies over the content digest."""
    _, InvalidSignature = _load_mldsa()
    return _verify_entries(
        odr,
        alg=ODR_PQC_SIGNATURE_ALG,
        key_id=compute_mldsa_key_id(public_key),
        verify=public_key.verify,
        invalid_signature=InvalidSignature,
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _odr_signing_message(odr: dict[str, Any]) -> bytes:
    """The bytes a detached ODR signature covers: ``bytes.fromhex(content_digest)``.

    The digest excludes the ``signatures`` array, so it is identical before and
    after any signatures are attached — every algorithm signs the same message.
    """
    try:
        return bytes.fromhex(odr_content_digest(odr))
    except (TypeError, ValueError) as exc:
        raise OdrSigningError("could not compute ODR content digest for signing") from exc


def _existing_signatures(signed: dict[str, Any], *, replace: bool) -> list[Any]:
    existing = signed.get("signatures")
    if replace or not isinstance(existing, list):
        return []
    invalid_index = next(
        (i for i, entry in enumerate(existing) if not _is_signature_entry_compatible(entry)),
        None,
    )
    if invalid_index is not None:
        raise OdrSigningError(
            f"existing signatures[{invalid_index}] is not a valid ODR signature entry; "
            "use replace=True to drop existing signatures before signing"
        )
    return existing


def _verify_entries(
    odr: dict[str, Any],
    *,
    alg: str,
    key_id: str,
    verify: Any,
    invalid_signature: type[Exception],
) -> bool:
    message = _odr_signing_message(odr)
    for entry in odr.get("signatures") or []:
        if not isinstance(entry, dict) or entry.get("alg") != alg or entry.get("key_id") != key_id:
            continue
        try:
            verify(base64.b64decode(entry["signature"]), message)
            return True
        except (invalid_signature, ValueError, TypeError, KeyError):
            continue
    return False


def _is_signature_entry_compatible(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("alg") not in (ODR_SIGNATURE_ALG, ODR_PQC_SIGNATURE_ALG):
        return False
    for field in ("key_id", "signature"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            return False
    signed_at = entry.get("signed_at")
    return signed_at is None or isinstance(signed_at, str)


__all__ = [
    "ODR_PQC_SIGNATURE_ALG",
    "ODR_SIGNATURE_ALG",
    "OdrSigningError",
    "compute_key_id",
    "compute_mldsa_key_id",
    "generate_mldsa_signing_key",
    "generate_signing_key",
    "load_mldsa_key_from_seed",
    "load_private_key_from_pem",
    "load_signing_key_from_secrets",
    "pqc_available",
    "public_key_pem",
    "sign_odr_hybrid",
    "sign_odr_receipt",
    "sign_odr_receipt_mldsa",
    "verify_odr_signature_ed25519",
    "verify_odr_signature_mldsa",
]
