"""
Auth handlers subpackage.

This package contains authentication-related handlers split by domain:
- handler: Main AuthHandler class for authentication endpoints
- validation: Email and password validation utilities
- store: In-memory user store for development/testing
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

__all__ = [
    "AuthHandler",
    "EMAIL_PATTERN",
    "MIN_PASSWORD_LENGTH",
    "MAX_PASSWORD_LENGTH",
    "validate_email",
    "validate_password",
    "InMemoryUserStore",
    "extract_user_from_request",
]
