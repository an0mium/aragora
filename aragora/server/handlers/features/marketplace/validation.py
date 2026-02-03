"""Marketplace input validation functions and constants."""

from __future__ import annotations

import re
from typing import Any

from aragora.server.validation.core import sanitize_string

from .models import TemplateCategory

# =============================================================================
# Constants for Input Validation
# =============================================================================

# Safe pattern for template IDs and deployment IDs: alphanumeric, hyphens, underscores
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

MAX_TEMPLATE_NAME_LENGTH = 200
MAX_DEPLOYMENT_NAME_LENGTH = 200
MAX_REVIEW_LENGTH = 2000
MAX_SEARCH_QUERY_LENGTH = 500
MAX_CONFIG_KEYS = 50
MAX_CONFIG_SIZE = MAX_CONFIG_KEYS
MIN_RATING = 1
MAX_RATING = 5
DEFAULT_LIMIT = 50
MIN_LIMIT = 1
MAX_LIMIT = 200
MAX_OFFSET = 10000

# Constant aliases
SAFE_TEMPLATE_ID_PATTERN = SAFE_ID_PATTERN

# =============================================================================
# Input Validation Functions
# =============================================================================


def _validate_id(value: str, label: str = "ID") -> tuple[bool, str]:
    """Validate an ID string (template_id or deployment_id).

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not value or not isinstance(value, str):
        return False, f"{label} is required"
    if len(value) > 128:
        return False, f"{label} must be at most 128 characters"
    if not SAFE_ID_PATTERN.match(value):
        return False, f"{label} contains invalid characters"
    return True, ""


def _validate_pagination(query: dict[str, Any]) -> tuple[int, int, str]:
    """Validate and clamp pagination parameters.

    Returns:
        Tuple of (limit, offset, error_message). error_message is empty if valid.
    """
    try:
        limit = int(query.get("limit", DEFAULT_LIMIT))
    except (ValueError, TypeError):
        return DEFAULT_LIMIT, 0, "limit must be an integer"

    try:
        offset = int(query.get("offset", 0))
    except (ValueError, TypeError):
        return DEFAULT_LIMIT, 0, "offset must be an integer"

    limit, offset = _clamp_pagination(limit, offset)

    return limit, offset, ""


def _validate_rating_value(value: Any) -> tuple[bool, int, str]:
    """Validate a rating value.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return False, 0, "Rating is required"
    if not isinstance(value, int):
        return False, 0, "Rating must be an integer"
    if value < MIN_RATING or value > MAX_RATING:
        return False, 0, f"Rating must be between {MIN_RATING} and {MAX_RATING}"
    return True, value, ""


def _validate_review_internal(value: Any) -> tuple[bool, str | None, str]:
    """Validate a review string.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, None, ""
    if not isinstance(value, str):
        return False, None, "Review must be a string"
    if len(value) > MAX_REVIEW_LENGTH:
        return False, None, f"Review must be at most {MAX_REVIEW_LENGTH} characters"
    try:
        from aragora.server.handlers.features import marketplace as marketplace_module

        return True, marketplace_module.sanitize_string(value), ""
    except Exception:
        return True, sanitize_string(value), ""


def _validate_deployment_name_internal(value: Any, fallback: str) -> tuple[bool, str, str]:
    """Validate a deployment name.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, fallback, ""
    if not isinstance(value, str):
        return False, "", "Deployment name must be a string"
    if len(value) > MAX_DEPLOYMENT_NAME_LENGTH:
        return False, "", f"Deployment name must be at most {MAX_DEPLOYMENT_NAME_LENGTH} characters"
    sanitized = sanitize_string(value, MAX_DEPLOYMENT_NAME_LENGTH)
    if not sanitized:
        return True, fallback, ""
    return True, sanitized, ""


def _validate_config(value: Any) -> tuple[bool, dict[str, Any], str]:
    """Validate deployment config.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None:
        return True, {}, ""
    if not isinstance(value, dict):
        return False, {}, "Config must be a dictionary"
    if len(value) > MAX_CONFIG_SIZE:
        return False, {}, f"Config must have at most {MAX_CONFIG_SIZE} keys"
    return True, value, ""


def _validate_search_query(value: Any) -> tuple[bool, str, str]:
    """Validate a search query string.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    if value is None or value == "":
        return True, "", ""
    if not isinstance(value, str):
        return False, "", "Search query must be a string"
    if len(value) > MAX_SEARCH_QUERY_LENGTH:
        return False, "", f"Search query must be at most {MAX_SEARCH_QUERY_LENGTH} characters"
    return True, sanitize_string(value, MAX_SEARCH_QUERY_LENGTH).lower(), ""


def _validate_category_filter(value: Any) -> tuple[bool, str | None, str]:
    """Validate a category filter value.

    Returns:
        Tuple of (is_valid, sanitized_value, error_message).
    """
    valid, category, err = _validate_category(value)
    if not valid:
        return False, None, err
    return True, category.value if category else None, ""


# =============================================================================
# Aliases for backward compatibility with existing tests
# =============================================================================


def _validate_template_id(value: str) -> tuple[bool, str | None]:
    """Validate a template ID (backward-compatible signature).

    Returns (is_valid, error_or_None).
    """
    valid, err = _validate_id(value, "Template ID")
    return valid, err if not valid else None


def _validate_deployment_id(value: str) -> tuple[bool, str | None]:
    """Validate a deployment ID (backward-compatible signature).

    Returns (is_valid, error_or_None).
    """
    valid, err = _validate_id(value, "Deployment ID")
    return valid, err if not valid else None


def _validate_deployment_name(value: Any, fallback: str = "") -> tuple[bool, str | None]:
    """Validate a deployment name (backward-compatible 2-tuple signature).

    Returns (is_valid, error_or_None).
    """
    valid, _, err = _validate_deployment_name_internal(value, fallback)
    return valid, err if not valid else None


def _validate_review(value: Any) -> tuple[bool, str | None]:
    """Validate a review (backward-compatible 2-tuple signature).

    Returns (is_valid, error_or_None).
    """
    valid, _, err = _validate_review_internal(value)
    return valid, err if not valid else None


def _validate_rating(value: Any) -> tuple[bool, int, str]:
    """Validate a rating value (alias for _validate_rating_value)."""
    return _validate_rating_value(value)


def _validate_category(value: Any) -> tuple[bool, TemplateCategory | None, str]:
    """Validate a category filter (backward-compatible, returns enum).

    Returns (is_valid, TemplateCategory_or_None, error_message).
    """
    if value is None or value == "":
        return True, None, ""
    if not isinstance(value, str):
        return False, None, "Category must be a string"
    # Case-insensitive lookup
    lower = value.lower()
    valid_categories = {cat.value: cat for cat in TemplateCategory}
    if lower not in valid_categories:
        return (
            False,
            None,
            f"Invalid category. Must be one of: {', '.join(sorted(valid_categories))}",
        )
    return True, valid_categories[lower], ""


def _clamp_pagination(limit: Any, offset: Any) -> tuple[int, int]:
    """Clamp pagination values (backward-compatible direct args)."""
    try:
        limit_int = int(limit) if limit is not None else DEFAULT_LIMIT
    except (ValueError, TypeError):
        limit_int = DEFAULT_LIMIT
    try:
        offset_int = int(offset) if offset is not None else 0
    except (ValueError, TypeError):
        offset_int = 0
    return (
        max(MIN_LIMIT, min(limit_int, MAX_LIMIT)),
        max(0, min(offset_int, MAX_OFFSET)),
    )
