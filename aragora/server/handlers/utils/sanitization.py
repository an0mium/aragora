"""
Response sanitization utilities.

Provides utilities for removing sensitive data from API responses
before returning them to clients. This complements the encryption
utilities in aragora.storage.encrypted_fields which handle storage.

Usage:
    from aragora.server.handlers.utils.sanitization import sanitize_response

    # Remove sensitive fields from a user dict:
    safe_user = sanitize_response(user.to_dict())

    # Or use the decorator:
    @sanitize_output
    async def get_user(user_id: str) -> dict:
        return user.to_dict()
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Fields that should NEVER appear in API responses
# These overlap with encrypted_fields.SENSITIVE_FIELDS but are specifically
# for output sanitization (not all encrypted fields need removal from responses)
RESPONSE_SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        # Password-related
        "password",
        "password_hash",
        "password_salt",
        "hashed_password",
        # API keys and tokens (raw values)
        "api_key",
        "api_key_hash",
        "api_key_salt",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "bot_token",
        # Secrets
        "secret",
        "client_secret",
        "signing_secret",
        "webhook_secret",
        "encryption_key",
        "private_key",
        # MFA secrets
        "mfa_secret",
        "totp_secret",
        "mfa_backup_codes",
        # Platform-specific credentials
        "slack_signing_secret",
        "discord_token",
        "telegram_token",
        "github_token",
        "twilio_auth_token",
        "sendgrid_api_key",
        "smtp_password",
        "ses_secret_access_key",
        "stripe_secret_key",
        "stripe_webhook_secret",
        # Database credentials
        "db_password",
        "database_password",
        "connection_string",
        # OAuth
        "oauth_token",
        "oauth_secret",
    }
)

# Additional fields that may contain sensitive data in nested structures
# These are checked recursively
NESTED_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "credentials",
        "secrets",
        "auth",
        "authentication",
    }
)


def sanitize_response(
    data: dict[str, Any] | list | Any,
    additional_fields: set[str] | None = None,
    recursive: bool = True,
    redact_value: str | None = None,
) -> dict[str, Any] | list | Any:
    """
    Remove sensitive fields from API response data.

    Args:
        data: Dictionary, list, or other value to sanitize
        additional_fields: Extra field names to remove beyond defaults
        recursive: Whether to sanitize nested dicts/lists (default True)
        redact_value: If provided, replace sensitive values with this
                      instead of removing them entirely. Useful for
                      indicating a field exists but is hidden.

    Returns:
        Sanitized copy of data with sensitive fields removed

    Example:
        >>> user = {"id": "123", "email": "a@b.com", "password_hash": "xxx"}
        >>> sanitize_response(user)
        {'id': '123', 'email': 'a@b.com'}

        >>> sanitize_response(user, redact_value="[REDACTED]")
        {'id': '123', 'email': 'a@b.com', 'password_hash': '[REDACTED]'}
    """
    if data is None:
        return None

    # Handle lists
    if isinstance(data, list):
        if recursive:
            return [
                sanitize_response(item, additional_fields, recursive, redact_value) for item in data
            ]
        return data

    # Handle non-dict types
    if not isinstance(data, dict):
        return data

    # Build set of fields to remove
    fields_to_remove = RESPONSE_SENSITIVE_FIELDS
    if additional_fields:
        fields_to_remove = fields_to_remove | additional_fields

    # Create sanitized copy
    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if this is a sensitive field
        if key in fields_to_remove or key_lower in fields_to_remove:
            if redact_value is not None:
                result[key] = redact_value
            # else: omit the field entirely
            continue

        # Check for nested sensitive containers
        if key_lower in NESTED_SENSITIVE_KEYS and isinstance(value, dict):
            if redact_value is not None:
                result[key] = redact_value
            continue

        # Recursively sanitize nested structures
        if recursive and isinstance(value, (dict, list)):
            result[key] = sanitize_response(value, additional_fields, recursive, redact_value)
        else:
            result[key] = value

    return result


def sanitize_user_response(user_data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize user data for API responses.

    Removes all password, token, and credential fields.
    This is a convenience wrapper for common user response patterns.

    Args:
        user_data: User dictionary (e.g., from user.to_dict())

    Returns:
        Sanitized user data safe for API response
    """
    # User-specific additional fields
    user_fields = {
        "password_reset_token",
        "password_reset_expires",
        "email_verification_token",
        "session_token",
    }
    result = sanitize_response(user_data, additional_fields=user_fields)
    # Type narrowing: input is dict, so output is dict
    assert isinstance(result, dict)
    return result


def sanitize_integration_response(integration_data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize integration/connector data for API responses.

    Removes OAuth tokens, API keys, and secrets while preserving
    configuration metadata.

    Args:
        integration_data: Integration configuration dict

    Returns:
        Sanitized integration data safe for API response
    """
    # Integration-specific additional fields
    integration_fields = {
        "oauth_access_token",
        "oauth_refresh_token",
        "api_secret",
        "app_secret",
        "verification_token",
    }
    result = sanitize_response(integration_data, additional_fields=integration_fields)
    # Type narrowing: input is dict, so output is dict
    assert isinstance(result, dict)
    return result


def sanitize_payment_response(payment_data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize payment/billing data for API responses.

    Masks card numbers, removes full account numbers while
    preserving transaction metadata.

    Args:
        payment_data: Payment or billing dictionary

    Returns:
        Sanitized payment data safe for API response
    """
    # Payment-specific additional fields
    payment_fields = {
        "card_number",
        "cvv",
        "cvc",
        "account_number",
        "routing_number",
        "bank_account",
    }

    result = sanitize_response(payment_data, additional_fields=payment_fields)
    # Type narrowing: input is dict, so output is dict
    assert isinstance(result, dict)

    # Mask last 4 digits if present
    if "card_last_four" not in result:
        if "masked_card" in payment_data:
            result["card_last_four"] = payment_data["masked_card"]

    return result


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def sanitize_output(
    additional_fields: set[str] | None = None,
    redact_value: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically sanitize function output.

    Can be used on sync or async functions that return dicts.

    Args:
        additional_fields: Extra fields to remove
        redact_value: If provided, replace instead of remove

    Example:
        @sanitize_output()
        async def get_user(user_id: str) -> dict:
            return user.to_dict()

        @sanitize_output(additional_fields={"internal_notes"})
        def get_invoice(invoice_id: str) -> dict:
            return invoice.to_dict()
    """

    def decorator(func: F) -> F:
        import asyncio

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                if isinstance(result, (dict, list)):
                    return sanitize_response(result, additional_fields, True, redact_value)
                return result

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, (dict, list)):
                    return sanitize_response(result, additional_fields, True, redact_value)
                return result

            return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "RESPONSE_SENSITIVE_FIELDS",
    "sanitize_response",
    "sanitize_user_response",
    "sanitize_integration_response",
    "sanitize_payment_response",
    "sanitize_output",
]
