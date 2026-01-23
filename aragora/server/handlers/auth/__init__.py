"""
Auth handlers subpackage.

This package contains authentication-related handlers split by domain:
- handler: Main AuthHandler class for authentication endpoints
- validation: Email and password validation utilities
- store: In-memory user store for development/testing
- sso_handlers: SSO/OIDC authentication handlers
- signup_handlers: Self-service signup and invitation handlers
"""

from aragora.billing.jwt_auth import extract_user_from_request

from .handler import AuthHandler
from .store import InMemoryUserStore
from .validation import (
    EMAIL_PATTERN,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    validate_email,
    validate_password,
)
from .sso_handlers import (
    get_sso_handlers,
    handle_sso_login,
    handle_sso_callback,
    handle_sso_refresh,
    handle_sso_logout,
    handle_list_providers,
    handle_get_sso_config,
)
from .signup_handlers import (
    get_signup_handlers,
    handle_signup,
    handle_verify_email,
    handle_resend_verification,
    handle_setup_organization,
    handle_invite,
    handle_check_invite,
    handle_accept_invite,
)

__all__ = [
    # Core auth
    "AuthHandler",
    "EMAIL_PATTERN",
    "MIN_PASSWORD_LENGTH",
    "MAX_PASSWORD_LENGTH",
    "validate_email",
    "validate_password",
    "InMemoryUserStore",
    "extract_user_from_request",
    # SSO handlers
    "get_sso_handlers",
    "handle_sso_login",
    "handle_sso_callback",
    "handle_sso_refresh",
    "handle_sso_logout",
    "handle_list_providers",
    "handle_get_sso_config",
    # Signup handlers
    "get_signup_handlers",
    "handle_signup",
    "handle_verify_email",
    "handle_resend_verification",
    "handle_setup_organization",
    "handle_invite",
    "handle_check_invite",
    "handle_accept_invite",
]
