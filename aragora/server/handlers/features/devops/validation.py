"""Input validation constants and helpers for DevOps endpoints."""

from __future__ import annotations

import re
from typing import Any


# Valid PagerDuty ID pattern (alphanumeric with specific prefixes)
PAGERDUTY_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")

# Valid urgency values
VALID_URGENCIES = frozenset({"high", "low"})

# Valid incident statuses
VALID_INCIDENT_STATUSES = frozenset({"triggered", "acknowledged", "resolved"})

# Maximum lengths for text fields
MAX_TITLE_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 10000
MAX_NOTE_CONTENT_LENGTH = 5000
MAX_RESOLUTION_LENGTH = 2000

# Maximum items in lists
MAX_USER_IDS = 20
MAX_SOURCE_INCIDENT_IDS = 50


def validate_pagerduty_id(id_value: str, field_name: str = "id") -> tuple[bool, str | None]:
    """Validate a PagerDuty ID format.

    Args:
        id_value: The ID to validate
        field_name: Name of the field for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not id_value:
        return False, f"{field_name} is required"
    if not isinstance(id_value, str):
        return False, f"{field_name} must be a string"
    if len(id_value) > 20:
        return False, f"{field_name} is too long"
    if not PAGERDUTY_ID_PATTERN.match(id_value):
        return False, f"{field_name} has invalid format"
    return True, None


def validate_urgency(urgency: str | None) -> str:
    """Validate and normalize urgency value."""
    if urgency is None:
        return "high"
    urgency_lower = str(urgency).lower().strip()
    if urgency_lower in VALID_URGENCIES:
        return urgency_lower
    return "high"  # Default to high if invalid


def validate_string_field(
    value: Any,
    field_name: str,
    required: bool = False,
    max_length: int = 500,
) -> tuple[str | None, str | None]:
    """Validate a string field.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        required: Whether the field is required
        max_length: Maximum allowed length

    Returns:
        Tuple of (sanitized_value, error_message)
    """
    if value is None or value == "":
        if required:
            return None, f"{field_name} is required"
        return None, None

    if not isinstance(value, str):
        try:
            value = str(value)
        except (ValueError, TypeError):
            return None, f"{field_name} must be a string"

    # Strip whitespace
    value = value.strip()

    # Check length
    if len(value) > max_length:
        return None, f"{field_name} exceeds maximum length of {max_length}"

    return value, None


def validate_id_list(
    values: Any,
    field_name: str,
    max_items: int = 20,
) -> tuple[list[str] | None, str | None]:
    """Validate a list of IDs.

    Args:
        values: The list to validate
        field_name: Name of the field for error messages
        max_items: Maximum number of items allowed

    Returns:
        Tuple of (validated_list, error_message)
    """
    if values is None:
        return None, None

    if not isinstance(values, list):
        return None, f"{field_name} must be a list"

    if len(values) > max_items:
        return None, f"{field_name} exceeds maximum of {max_items} items"

    validated = []
    for i, item in enumerate(values):
        is_valid, err = validate_pagerduty_id(str(item), f"{field_name}[{i}]")
        if not is_valid:
            return None, err
        validated.append(str(item))

    return validated, None


# Backward-compatible aliases (prefixed with underscore in original module)
_validate_pagerduty_id = validate_pagerduty_id
_validate_urgency = validate_urgency
_validate_string_field = validate_string_field
_validate_id_list = validate_id_list
