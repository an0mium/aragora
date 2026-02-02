"""
CRM Input Validation - Constants and Functions.

This module contains validation utilities for CRM handler endpoints:
- Pattern constants for validating IDs and emails
- Field length limits
- Validation functions for various input types

Stability: STABLE
"""

from __future__ import annotations

import re
from typing import Any


# =============================================================================
# Input Validation Constants
# =============================================================================

# Platform ID validation: alphanumeric and underscores only
SAFE_PLATFORM_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,49}$")

# Contact/Company/Deal ID validation: alphanumeric, hyphens, underscores
SAFE_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$")

# Email validation pattern (basic but covers most cases)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Max lengths for input validation
MAX_EMAIL_LENGTH = 254
MAX_NAME_LENGTH = 128
MAX_PHONE_LENGTH = 32
MAX_COMPANY_NAME_LENGTH = 256
MAX_JOB_TITLE_LENGTH = 128
MAX_DOMAIN_LENGTH = 253
MAX_DEAL_NAME_LENGTH = 256
MAX_STAGE_LENGTH = 64
MAX_PIPELINE_LENGTH = 64
MAX_CREDENTIAL_VALUE_LENGTH = 1024
MAX_SEARCH_QUERY_LENGTH = 256


# =============================================================================
# Validation Functions
# =============================================================================


def validate_platform_id(platform: str) -> tuple[bool, str | None]:
    """Validate a platform ID.

    Args:
        platform: Platform identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not platform:
        return False, "Platform is required"
    if len(platform) > 50:
        return False, "Platform name too long (max 50 characters)"
    if not SAFE_PLATFORM_PATTERN.match(platform):
        return False, "Invalid platform format (alphanumeric and underscores only)"
    return True, None


def validate_resource_id(resource_id: str, resource_type: str = "ID") -> tuple[bool, str | None]:
    """Validate a resource ID (contact, company, deal, etc.).

    Args:
        resource_id: Resource identifier to validate
        resource_type: Type name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not resource_id:
        return False, f"{resource_type} is required"
    if len(resource_id) > 128:
        return False, f"{resource_type} too long (max 128 characters)"
    if not SAFE_RESOURCE_ID_PATTERN.match(resource_id):
        return False, f"Invalid {resource_type.lower()} format"
    return True, None


def validate_email(email: str | None, required: bool = False) -> tuple[bool, str | None]:
    """Validate an email address.

    Args:
        email: Email to validate
        required: Whether email is required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        if required:
            return False, "Email is required"
        return True, None
    if len(email) > MAX_EMAIL_LENGTH:
        return False, f"Email too long (max {MAX_EMAIL_LENGTH} characters)"
    if not EMAIL_PATTERN.match(email):
        return False, "Invalid email format"
    return True, None


def validate_string_field(
    value: str | None,
    field_name: str,
    max_length: int,
    required: bool = False,
) -> tuple[bool, str | None]:
    """Validate a string field with length constraints.

    Args:
        value: Value to validate
        field_name: Field name for error messages
        max_length: Maximum allowed length
        required: Whether field is required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        if required:
            return False, f"{field_name} is required"
        return True, None
    if len(value) > max_length:
        return False, f"{field_name} too long (max {max_length} characters)"
    return True, None


def validate_amount(amount: Any) -> tuple[bool, str | None, float | None]:
    """Validate a monetary amount.

    Args:
        amount: Amount value to validate

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    if amount is None:
        return True, None, None
    try:
        amt = float(amount)
        if amt < 0:
            return False, "Amount cannot be negative", None
        if amt > 1_000_000_000_000:  # 1 trillion max
            return False, "Amount too large", None
        return True, None, amt
    except (ValueError, TypeError):
        return False, "Invalid amount format", None


def validate_probability(probability: Any) -> tuple[bool, str | None, float | None]:
    """Validate a probability value (0-100).

    Args:
        probability: Probability value to validate

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    if probability is None:
        return True, None, None
    try:
        prob = float(probability)
        if prob < 0 or prob > 100:
            return False, "Probability must be between 0 and 100", None
        return True, None, prob
    except (ValueError, TypeError):
        return False, "Invalid probability format", None


__all__ = [
    # Constants
    "SAFE_PLATFORM_PATTERN",
    "SAFE_RESOURCE_ID_PATTERN",
    "EMAIL_PATTERN",
    "MAX_EMAIL_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_PHONE_LENGTH",
    "MAX_COMPANY_NAME_LENGTH",
    "MAX_JOB_TITLE_LENGTH",
    "MAX_DOMAIN_LENGTH",
    "MAX_DEAL_NAME_LENGTH",
    "MAX_STAGE_LENGTH",
    "MAX_PIPELINE_LENGTH",
    "MAX_CREDENTIAL_VALUE_LENGTH",
    "MAX_SEARCH_QUERY_LENGTH",
    # Validation functions
    "validate_platform_id",
    "validate_resource_id",
    "validate_email",
    "validate_string_field",
    "validate_amount",
    "validate_probability",
]
