"""
JWT Authentication Configuration.

Provides security configuration, secret management, and validation
for JWT token handling.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
from typing import Optional

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
# Use secrets manager for sensitive values
JWT_SECRET = _get_secret_value("ARAGORA_JWT_SECRET", "")
JWT_SECRET_PREVIOUS = _get_secret_value("ARAGORA_JWT_SECRET_PREVIOUS", "")
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

    Enforces strict requirements in production and logs warnings in non-prod.
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

    if JWT_SECRET_PREVIOUS and len(JWT_SECRET_PREVIOUS) < MIN_SECRET_LENGTH:
        logger.warning(
            "jwt_previous_secret_too_short length=%s min=%s",
            len(JWT_SECRET_PREVIOUS),
            MIN_SECRET_LENGTH,
        )

    if JWT_SECRET_PREVIOUS and not JWT_SECRET_ROTATED_AT:
        logger.warning("jwt_previous_secret_without_rotation_timestamp")

    if is_production():
        if not JWT_SECRET:
            raise ConfigurationError(
                component="JWT Authentication",
                reason="ARAGORA_JWT_SECRET must be set in production. "
                'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"',
            )
        if len(JWT_SECRET) < MIN_SECRET_LENGTH:
            raise ConfigurationError(
                component="JWT Authentication",
                reason=f"ARAGORA_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters in production. "
                f"Current length: {len(JWT_SECRET)}",
            )


def validate_secret_strength(secret: str) -> bool:
    """Validate JWT secret meets minimum entropy requirements."""
    return len(secret) >= MIN_SECRET_LENGTH


def get_secret() -> bytes:
    """
    Get JWT secret with strict validation.

    ARAGORA_JWT_SECRET must be set in all environments except pytest.
    This prevents issues with:
    - Load balancing (different instances need same secret)
    - Server restarts invalidating all tokens

    Raises:
        RuntimeError: If secret is missing or weak (except in pytest).
    """
    global JWT_SECRET
    running_under_pytest = "pytest" in sys.modules

    if not JWT_SECRET:
        if running_under_pytest:
            # Allow ephemeral secret only in test environments
            JWT_SECRET = base64.b64encode(os.urandom(32)).decode("utf-8")
            logger.debug("TEST MODE: Using ephemeral JWT secret")
        else:
            raise ConfigurationError(
                component="JWT Authentication",
                reason="ARAGORA_JWT_SECRET must be set. "
                'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"',
            )

    if not validate_secret_strength(JWT_SECRET):
        if running_under_pytest:
            logger.debug(f"TEST MODE: JWT secret is weak (< {MIN_SECRET_LENGTH} chars)")
        else:
            raise ConfigurationError(
                component="JWT Authentication",
                reason=f"ARAGORA_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters. "
                f"Current length: {len(JWT_SECRET)}",
            )

    return JWT_SECRET.encode("utf-8")


def get_previous_secret() -> Optional[bytes]:
    """
    Get previous JWT secret for rotation support.

    Returns the previous secret only if:
    1. It meets minimum length requirements
    2. The rotation timestamp is within the grace period

    This prevents leaked old secrets from being exploitable indefinitely.
    """
    if not JWT_SECRET_PREVIOUS or len(JWT_SECRET_PREVIOUS) < MIN_SECRET_LENGTH:
        return None

    # Check rotation timestamp if set
    if JWT_SECRET_ROTATED_AT:
        try:
            rotated_at = int(JWT_SECRET_ROTATED_AT)
            grace_seconds = JWT_ROTATION_GRACE_HOURS * 3600
            if time.time() - rotated_at > grace_seconds:
                logger.debug(
                    f"jwt_previous_secret_expired: rotated {JWT_ROTATION_GRACE_HOURS}+ hours ago"
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

    return JWT_SECRET_PREVIOUS.encode("utf-8")


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
