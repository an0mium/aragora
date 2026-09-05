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

The digest is computed with :func:`aragora.gauntlet.odr_jcs.odr_content_digest`
(re-exported by :mod:`aragora.gauntlet.odr_export`), which the verifier's own
docstring states it "mirrors exactly". Excluding the
``signatures`` array from the digest is what makes the signatures *detached*:
attaching one never changes the bytes it covers.

Key management (per the post-incident security architecture):
    The private key is NEVER read from a raw environment variable or committed
    to the repo. It is read from the PKCS#8 Ed25519 PEM file named by
    ``ARAGORA_ODR_SIGNING_KEY_FILE``, or resolved from AWS Secrets Manager via
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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.gauntlet.odr_jcs import odr_content_digest

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
SIGNING_KEY_FILE_ENV = "ARAGORA_ODR_SIGNING_KEY_FILE"


class OdrSigningError(Exception):
    """Raised when a private key cannot be loaded or a receipt cannot be signed."""


class OdrSigningUnconfiguredError(OdrSigningError):
    """No signing key is configured — an EXPECTED deployment state.

    Raised only when the deployment genuinely has no key: Secrets Manager is
    not enabled, or the signing secret does not exist in any configured
    region. Every other loader failure (unreadable secret, bad AWS setup,
    invalid key material) stays a plain :class:`OdrSigningError` so callers
    that degrade to unsigned output on *unconfigured* deployments still fail
    closed when a configured key is broken.
    """


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
    from cryptography.exceptions import UnsupportedAlgorithm

    data = pem.encode("utf-8") if isinstance(pem, str) else pem
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
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


def _load_pem_secret_from_aws(secret_id: str, *, explicitly_named: bool = False) -> str:
    """Fetch a standalone PEM secret from AWS Secrets Manager.

    This intentionally does not call ``get_secret(secret_id)`` because that API
    looks up a key inside Aragora's configured JSON secret bundle and may fall
    back to environment variables in non-strict local mode. ODR signing keys are
    standalone custody material: the environment may name the SecretId, but it
    must never carry the raw private key.

    ``explicitly_named`` marks a secret id the operator chose (env var or
    argument) rather than the built-in default. An explicitly named secret
    that does not exist is a configuration ERROR (typo, deleted secret) and
    must fail closed, never be treated as "not configured".
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
        message = (
            "AWS Secrets Manager is not enabled for ODR signing; set "
            "ARAGORA_USE_SECRETS_MANAGER=true and provision the PEM private key "
            f"in secret '{secret_label}'"
        )
        if explicitly_named:
            # The operator explicitly named a signing secret: signing is
            # intended, so an unusable secrets backend must fail closed.
            raise OdrSigningError(message)
        raise OdrSigningUnconfiguredError(message)

    manager = secret_config.SecretManager(config)
    regions = config.aws_regions or [config.aws_region]
    last_error: Exception | None = None
    all_not_found = True
    for region in regions:
        client = manager._get_aws_client(region)  # noqa: SLF001 - reuse repo AWS client setup.
        if client is None:
            all_not_found = False
            continue
        try:
            response = client.get_secret_value(SecretId=secret_id)
        except secret_config.ClientError as exc:
            last_error = exc
            code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code != "ResourceNotFoundException":
                all_not_found = False
            continue
        except secret_config.BotoCoreError as exc:
            last_error = exc
            all_not_found = False
            continue
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            last_error = exc
            all_not_found = False
            continue

        secret_string = response.get("SecretString")
        if isinstance(secret_string, str) and secret_string.strip():
            return secret_string

        all_not_found = False
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

    if all_not_found and last_error is not None and not explicitly_named:
        # Every configured region answered ResourceNotFound for the DEFAULT
        # secret name: the key was never provisioned. This is the expected
        # pre-provisioning deployment state, distinct from a
        # configured-but-unreadable key. An explicitly named secret that is
        # missing falls through to the hard error below (typo/deletion must
        # fail closed).
        raise OdrSigningUnconfiguredError(
            f"ODR signing key secret '{secret_label}' does not exist in any "
            "configured AWS region (not provisioned yet)"
        )

    detail = f" (last error: {type(last_error).__name__})" if last_error else ""
    raise OdrSigningError(
        f"ODR signing key secret '{secret_label}' could not be read from AWS Secrets Manager{detail}"
    )


def load_signing_key_from_secrets(
    secret_name: str | None = None,
) -> Ed25519PrivateKey:
    """Resolve a key from a configured file, otherwise AWS Secrets Manager.

    Empty file configuration is equivalent to unset. A non-empty path takes
    precedence over the secret name and must be usable, never silently unsigned.
    Environment variables name custody locations, never raw key material.
    """
    key_file = os.environ.get(SIGNING_KEY_FILE_ENV)
    if key_file:
        try:
            data = Path(key_file).read_bytes()
            return load_private_key_from_pem(data)
        except (OSError, ValueError, OdrSigningError):
            # Do not echo a path (it could contain mistakenly pasted key bytes)
            # or chain parser exceptions into producer logs.
            raise OdrSigningError(
                "ODR signing key file is configured but could not be used; "
                "expected a readable PKCS#8 Ed25519 private-key PEM"
            ) from None
    explicit = secret_name or os.environ.get(SIGNING_KEY_SECRET_ENV)
    name = explicit or DEFAULT_SIGNING_KEY_SECRET
    pem = _load_pem_secret_from_aws(name, explicitly_named=bool(explicit))
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
    issuer: str | None = None,
    role: str = "emitter",
    signed_at: str | None = None,
    expires_at: str | None = None,
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
        issuer: Non-empty producer identifier; opts into descriptive metadata.
        role: One of emitter, reviewer, attestor, or notary.
        signed_at: Signing timestamp; defaults to current UTC time.
        expires_at: Optional expiry timestamp.

    Returns:
        A copy of ``odr`` with a signature and producer metadata entry
        appended to its ``signatures`` array. The signature covers
        ``bytes.fromhex(odr_content_digest(odr))`` — exactly what the verifier
        re-derives and checks.

    Metadata inside ``signatures`` is descriptive, not digest-covered. It must
    not be treated as cryptographically authenticated identity or time.
    """
    if issuer is not None and (not isinstance(issuer, str) or not issuer):
        raise OdrSigningError("signature issuer must be a non-empty string")
    if role not in ("emitter", "reviewer", "attestor", "notary"):
        raise OdrSigningError("signature role must be emitter, reviewer, attestor, or notary")
    for name, value in (("signed_at", signed_at), ("expires_at", expires_at)):
        if value is not None and not isinstance(value, str):
            raise OdrSigningError(f"signature {name} must be a string")
    if issuer is None and (role != "emitter" or signed_at is not None or expires_at is not None):
        raise OdrSigningError("signature metadata requires an explicit issuer")
    if issuer is not None:
        signed_at = signed_at if signed_at is not None else datetime.now(timezone.utc).isoformat()
        times: dict[str, datetime] = {}
        for name, value in (("signed_at", signed_at), ("expires_at", expires_at)):
            if value is None:
                continue
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("timezone required")
            except ValueError:
                raise OdrSigningError(
                    f"signature {name} must be an ISO 8601 timestamp with timezone"
                ) from None
            times[name] = parsed
        if expires_at is not None and times["expires_at"] <= times["signed_at"]:
            raise OdrSigningError("signature expires_at must be after signed_at")
    signed = copy.deepcopy(odr)

    existing = signed.get("signatures")
    signatures: list[Any] = []
    if not replace and isinstance(existing, list):
        invalid_index = next(
            (
                index
                for index, entry in enumerate(existing)
                if not _is_signature_entry_compatible(entry)
            ),
            None,
        )
        if invalid_index is not None:
            raise OdrSigningError(
                f"existing signatures[{invalid_index}] is not a valid ODR signature entry; "
                "use replace=True to drop existing signatures before signing"
            )
        signatures = existing

    # The digest excludes the signatures array (detached) — compute it against
    # the payload as the verifier will, regardless of what's already attached.
    try:
        digest_hex = odr_content_digest(signed)
        message = bytes.fromhex(digest_hex)
    except (TypeError, ValueError) as exc:
        raise OdrSigningError("could not compute ODR content digest for signing") from exc

    signature_bytes = private_key.sign(message)
    entry = {
        "alg": ODR_SIGNATURE_ALG,
        "key_id": compute_key_id(private_key.public_key()),
        "signature": base64.b64encode(signature_bytes).decode("ascii"),
    }
    # Legacy callers keep the published v0.1 entry shape. File-custody producers
    # explicitly opt in by supplying an issuer; other callers can do the same.
    if issuer is not None and signed_at is not None:
        entry.update(
            issuer=issuer,
            role=role,
            signed_at=signed_at,
        )
        if expires_at is not None:
            entry["expires_at"] = expires_at
    signatures.append(entry)
    signed["signatures"] = signatures
    return signed


def _is_signature_entry_compatible(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("alg") != ODR_SIGNATURE_ALG:
        return False
    for field in ("key_id", "signature"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            return False
    if "issuer" in entry and (not isinstance(entry["issuer"], str) or not entry["issuer"]):
        return False
    if "role" in entry and entry["role"] not in ("emitter", "reviewer", "attestor", "notary"):
        return False
    return all(
        name not in entry or isinstance(entry[name], str) for name in ("signed_at", "expires_at")
    )


__all__ = [
    "ODR_SIGNATURE_ALG",
    "OdrSigningError",
    "OdrSigningUnconfiguredError",
    "compute_key_id",
    "generate_signing_key",
    "load_private_key_from_pem",
    "load_signing_key_from_secrets",
    "public_key_pem",
    "sign_odr_receipt",
]
