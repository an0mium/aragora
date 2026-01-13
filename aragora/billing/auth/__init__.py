"""
JWT Authentication Module.

This module provides JWT-based authentication for user sessions,
including token creation, validation, and revocation.

Re-exports all public APIs from submodules for backward compatibility.
"""

from .blacklist import (
    TokenBlacklist,
    get_persistent_blacklist,
    get_token_blacklist,
    is_token_revoked_persistent,
    revoke_token_persistent,
)
from .config import (
    ALLOWED_ALGORITHMS,
    ARAGORA_ENVIRONMENT,
    MIN_SECRET_LENGTH,
    _validate_secret_strength,
    get_previous_secret,
    get_secret,
    is_production,
    validate_secret_strength,
    validate_security_config,
)
from .context import (
    UserAuthContext,
    extract_user_from_request,
)
from .tokens import (
    JWTPayload,
    TokenPair,
    _base64url_decode,
    _base64url_encode,
    create_access_token,
    create_mfa_pending_token,
    create_refresh_token,
    create_token_pair,
    decode_jwt,
    validate_access_token,
    validate_mfa_pending_token,
    validate_refresh_token,
)

__all__ = [
    # Blacklist
    "TokenBlacklist",
    "get_token_blacklist",
    "get_persistent_blacklist",
    "revoke_token_persistent",
    "is_token_revoked_persistent",
    # Config
    "ARAGORA_ENVIRONMENT",
    "MIN_SECRET_LENGTH",
    "ALLOWED_ALGORITHMS",
    "is_production",
    "validate_security_config",
    "validate_secret_strength",
    "_validate_secret_strength",  # Backward compat
    "get_secret",
    "get_previous_secret",
    # Context
    "UserAuthContext",
    "extract_user_from_request",
    # Tokens
    "JWTPayload",
    "TokenPair",
    "decode_jwt",
    "create_access_token",
    "create_refresh_token",
    "validate_access_token",
    "validate_refresh_token",
    "create_token_pair",
    "create_mfa_pending_token",
    "validate_mfa_pending_token",
    # Internal utilities (for testing)
    "_base64url_encode",
    "_base64url_decode",
]

# Validate security configuration at module load
validate_security_config()
