"""
JWT Authentication for User Sessions.

This module has been refactored into submodules for better maintainability:
- aragora.billing.auth.blacklist: Token revocation
- aragora.billing.auth.config: Security configuration
- aragora.billing.auth.tokens: Token creation and validation
- aragora.billing.auth.context: User authentication context

All public APIs are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Re-export everything from the auth submodule for backward compatibility
from aragora.billing.auth import (
    ALLOWED_ALGORITHMS,
    ARAGORA_ENVIRONMENT,
    MIN_SECRET_LENGTH,
    JWTPayload,
    TokenBlacklist,
    TokenPair,
    UserAuthContext,
    _base64url_decode,
    _base64url_encode,
    _validate_secret_strength,
    create_access_token,
    create_mfa_pending_token,
    create_refresh_token,
    create_token_pair,
    decode_jwt,
    extract_user_from_request,
    get_persistent_blacklist,
    get_token_blacklist,
    is_token_revoked_persistent,
    revoke_token_persistent,
    validate_access_token,
    validate_mfa_pending_token,
    validate_refresh_token,
)

__all__ = [
    "JWTPayload",
    "UserAuthContext",
    "TokenPair",
    "TokenBlacklist",
    "get_token_blacklist",
    "get_persistent_blacklist",
    "revoke_token_persistent",
    "is_token_revoked_persistent",
    "create_access_token",
    "create_refresh_token",
    "decode_jwt",
    "validate_access_token",
    "validate_refresh_token",
    "extract_user_from_request",
    "create_token_pair",
    # MFA pending token
    "create_mfa_pending_token",
    "validate_mfa_pending_token",
    # Security configuration
    "ARAGORA_ENVIRONMENT",
    "MIN_SECRET_LENGTH",
    "ALLOWED_ALGORITHMS",
    # Internal utilities (for testing backward compat)
    "_validate_secret_strength",
    "_base64url_encode",
    "_base64url_decode",
]
