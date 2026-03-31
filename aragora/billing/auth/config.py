"""
JWT Authentication Configuration.

Provides security configuration, secret management, and validation
for JWT token handling.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import sys
import time

from aragora.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _get_secret_value(name: str, default: str = "") -> str:
    """Get secret value from secrets manager or environment."""
    try:
        from aragora.config.secrets import get_secret

        return get_secret(name, default) or default
    except ImportError:
        return os.environ.get(name, default)


# Environment configuration
ARAGORA_ENVIRONMENT = os.environ.get("ARAGORA_ENVIRONMENT", "development")

# Lazy-loaded secrets - populated on first access to avoid import-time issues
# When running under systemd, env vars may not be fully propagated during imports
_jwt_secret_cache: str | None = None
_jwt_secret_previous_cache: str | None = None


def _get_jwt_secret() -> str:
    """Get JWT secret with lazy loading and caching."""
    global _jwt_secret_cache
    if _jwt_secret_cache is None:
        _jwt_secret_cache = _get_secret_value("ARAGORA_JWT_SECRET", "")
    return _jwt_secret_cache


def _get_jwt_secret_previous() -> str:
    """Get previous JWT secret with lazy loading and caching."""
    global _jwt_secret_previous_cache
    if _jwt_secret_previous_cache is None:
        _jwt_secret_previous_cache = _get_secret_value("ARAGORA_JWT_SECRET_PREVIOUS", "")
    return _jwt_secret_previous_cache


def __getattr__(name: str) -> str:
    """Provide lazy module attributes for JWT secrets."""
    if name == "JWT_SECRET":
        return _get_jwt_secret()
    if name == "JWT_SECRET_PREVIOUS":
        return _get_jwt_secret_previous()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Unix timestamp when secret was rotated (for limiting previous secret validity)
JWT_SECRET_ROTATED_AT = os.environ.get("ARAGORA_JWT_SECRET_ROTATED_AT", "")
# How long previous secret remains valid after rotation (default: 24 hours)
JWT_ROTATION_GRACE_HOURS = int(os.environ.get("ARAGORA_JWT_ROTATION_GRACE_HOURS", "24"))
JWT_ALGORITHM = "HS256"
ALLOWED_ALGORITHMS = frozenset(["HS256"])  # Explicitly allowed algorithms
JWT_EXPIRY_HOURS = int(os.environ.get("ARAGORA_JWT_EXPIRY_HOURS", "24"))
REFRESH_TOKEN_EXPIRY_DAYS = int(os.environ.get("ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS", "30"))

# Security constraints
MIN_SECRET_LENGTH = 32
MAX_ACCESS_TOKEN_HOURS = 168  # 7 days max
MAX_REFRESH_TOKEN_DAYS = 90  # 90 days max


def is_production() -> bool:
    """Check if running in production environment.

    Conservative detection - treats any production-like environment as production
    to prevent security misconfigurations.
    """
    env = ARAGORA_ENVIRONMENT.lower()
    production_indicators = ["production", "prod", "live", "prd"]
    return any(indicator in env for indicator in production_indicators)


def validate_security_config() -> None:
    """Validate security configuration at module load.

    Keeps import-time validation side-effect free.

    This function intentionally avoids Secrets Manager-backed lookups. Several
    unrelated product paths import billing auth helpers transitively, and
    hitting AWS during module import can stall handler imports by seconds when
    the network is unavailable. Runtime secret enforcement still happens in
    `get_secret()`, which is only called when JWT operations are actually used.
    """
    if "pytest" in sys.modules:
        return

    if JWT_ROTATION_GRACE_HOURS < 0:
        logger.warning("jwt_rotation_grace_hours_negative=%s", JWT_ROTATION_GRACE_HOURS)

    if JWT_EXPIRY_HOURS < 1 or JWT_EXPIRY_HOURS > MAX_ACCESS_TOKEN_HOURS:
        logger.warning(
            "jwt_expiry_hours_out_of_range=%s (allowed 1-%s)",
            JWT_EXPIRY_HOURS,
            MAX_ACCESS_TOKEN_HOURS,
        )

    if REFRESH_TOKEN_EXPIRY_DAYS < 1 or REFRESH_TOKEN_EXPIRY_DAYS > MAX_REFRESH_TOKEN_DAYS:
        logger.warning(
            "refresh_token_expiry_days_out_of_range=%s (allowed 1-%s)",
            REFRESH_TOKEN_EXPIRY_DAYS,
            MAX_REFRESH_TOKEN_DAYS,
        )

    jwt_secret_prev = os.environ.get("ARAGORA_JWT_SECRET_PREVIOUS", "")
    if jwt_secret_prev and len(jwt_secret_prev) < MIN_SECRET_LENGTH:
        logger.warning(
            "jwt_previous_secret_too_short length=%s min=%s",
            len(jwt_secret_prev),
            MIN_SECRET_LENGTH,
        )

    if jwt_secret_prev and not JWT_SECRET_ROTATED_AT:
        logger.warning("jwt_previous_secret_without_rotation_timestamp")

    jwt_secret = os.environ.get("ARAGORA_JWT_SECRET", "")
    if is_production() and jwt_secret and len(jwt_secret) < MIN_SECRET_LENGTH:
        raise ConfigurationError(
            component="JWT Authentication",
            reason=f"ARAGORA_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters in production. "
            f"Current length: {len(jwt_secret)}",
        )
    if not is_production() and jwt_secret and len(jwt_secret) < MIN_SECRET_LENGTH:
        logger.warning(
            "jwt_secret_too_short_non_production length=%s min=%s",
            len(jwt_secret),
            MIN_SECRET_LENGTH,
        )


def validate_secret_strength(secret: str) -> bool:
    """Validate JWT secret meets minimum entropy requirements."""
    return len(secret) >= MIN_SECRET_LENGTH


def _derive_non_production_secret() -> str:
    """Derive a stable local-only JWT secret for non-production fallbacks."""
    machine_id = os.environ.get("HOSTNAME", "") + os.environ.get("USER", "")
    if not machine_id:
        machine_id = "local-development"
    return hashlib.sha256(f"aragora-jwt-{machine_id}".encode()).hexdigest()


def get_secret() -> bytes:
    """
    Get JWT secret with strict validation.

    Production requires ARAGORA_JWT_SECRET. Non-production may fall back to
    ARAGORA_SECRET_KEY or a stable machine-local derived secret to keep local
    signup and auth flows working without manual secret bootstrapping.

    Raises:
        RuntimeError: If secret is missing or weak in production.
    """
    global _jwt_secret_cache
    running_under_pytest = "pytest" in sys.modules

    jwt_secret = _get_jwt_secret()
    secret_source = "ARAGORA_JWT_SECRET"
    if not jwt_secret:
        if running_under_pytest:
            # Allow ephemeral secret only in test environments
            _jwt_secret_cache = base64.b64encode(os.urandom(32)).decode("utf-8")
            jwt_secret = _jwt_secret_cache
            secret_source = "pytest-ephemeral"
            logger.debug("TEST MODE: Using ephemeral JWT secret")
        elif is_production():
            logger.error("[JWT_DEBUG] get_secret: ARAGORA_JWT_SECRET is NOT SET!")
            raise ConfigurationError(
                component="JWT Authentication",
                reason="ARAGORA_JWT_SECRET must be set. "
                'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"',
            )
        else:
            jwt_secret = _get_secret_value("ARAGORA_SECRET_KEY", "")
            if jwt_secret:
                secret_source = "ARAGORA_SECRET_KEY"
                logger.warning(
                    "JWT auth: ARAGORA_JWT_SECRET is not set. Using ARAGORA_SECRET_KEY in non-production."
                )
            else:
                jwt_secret = _derive_non_production_secret()
                secret_source = "derived-non-production"
                logger.warning(
                    "JWT auth: No ARAGORA_JWT_SECRET or ARAGORA_SECRET_KEY set. "
                    "Using derived secret in non-production."
                )
            _jwt_secret_cache = jwt_secret

    if not validate_secret_strength(jwt_secret):
        if running_under_pytest:
            logger.debug("TEST MODE: JWT secret is weak (< %s chars)", MIN_SECRET_LENGTH)
        elif is_production():
            logger.error(
                "[JWT_DEBUG] get_secret: Secret too weak! Length=%s, required=%s",
                len(jwt_secret),
                MIN_SECRET_LENGTH,
            )
            raise ConfigurationError(
                component="JWT Authentication",
                reason=f"ARAGORA_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters. "
                f"Current length: {len(jwt_secret)}",
            )
        else:
            logger.warning(
                "JWT auth: non-production secret is weak (source=%s length=%s min=%s)",
                secret_source,
                len(jwt_secret),
                MIN_SECRET_LENGTH,
            )

    # Log secret fingerprint (first 4 chars of hash) for debugging without exposing secret
    secret_fingerprint = hashlib.sha256(jwt_secret.encode()).hexdigest()[:8]
    logger.info(
        "[JWT_DEBUG] get_secret: Using secret with fingerprint=%s, length=%s source=%s",
        secret_fingerprint,
        len(jwt_secret),
        secret_source,
    )

    return jwt_secret.encode("utf-8")


def get_previous_secret() -> bytes | None:
    """
    Get previous JWT secret for rotation support.

    Returns the previous secret only if:
    1. It meets minimum length requirements
    2. The rotation timestamp is within the grace period

    This prevents leaked old secrets from being exploitable indefinitely.
    """
    jwt_secret_prev = _get_jwt_secret_previous()
    if not jwt_secret_prev or len(jwt_secret_prev) < MIN_SECRET_LENGTH:
        return None

    # Check rotation timestamp if set
    if JWT_SECRET_ROTATED_AT:
        try:
            rotated_at = int(JWT_SECRET_ROTATED_AT)
            grace_seconds = JWT_ROTATION_GRACE_HOURS * 3600
            if time.time() - rotated_at > grace_seconds:
                logger.debug(
                    "jwt_previous_secret_expired: rotated %s+ hours ago", JWT_ROTATION_GRACE_HOURS
                )
                return None
        except ValueError:
            logger.warning(
                "jwt_previous_secret: invalid ARAGORA_JWT_SECRET_ROTATED_AT format, "
                "expected Unix timestamp"
            )
            # In production, reject previous secret if timestamp is invalid
            if is_production():
                return None

    return jwt_secret_prev.encode("utf-8")


# Backward compatibility alias
_validate_secret_strength = validate_secret_strength

__all__ = [
    # Environment
    "ARAGORA_ENVIRONMENT",
    "JWT_ALGORITHM",
    "ALLOWED_ALGORITHMS",
    "JWT_EXPIRY_HOURS",
    "REFRESH_TOKEN_EXPIRY_DAYS",
    # Constraints
    "MIN_SECRET_LENGTH",
    "MAX_ACCESS_TOKEN_HOURS",
    "MAX_REFRESH_TOKEN_DAYS",
    # Functions
    "is_production",
    "validate_security_config",
    "validate_secret_strength",
    "_validate_secret_strength",  # Backward compatibility
    "get_secret",
    "get_previous_secret",
]
