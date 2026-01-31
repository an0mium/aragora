"""
Query parameter extraction utilities for handler methods.

Provides type-safe parameter extraction from query strings with support
for defaults, bounds, and list value handling.
"""

import logging
from typing import Any
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

# Threshold above which we warn about large page sizes
LARGE_PAGE_WARNING_THRESHOLD = 500


def parse_query_params(query_string: str) -> dict[str, Any]:
    """Parse query string into a dictionary."""
    if not query_string:
        return {}
    params = parse_qs(query_string)
    # Convert single-value lists to just values
    return {k: v[0] if len(v) == 1 else v for k, v in params.items()}


def get_int_param(params: dict[str, Any], key: str, default: int = 0) -> int:
    """Safely get an integer parameter, handling list values from query strings."""
    try:
        value = params.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return int(value)
    except (ValueError, TypeError):
        return default


def get_float_param(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely get a float parameter, handling list values from query strings."""
    try:
        value = params.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_bool_param(params: dict[str, Any], key: str, default: bool = False) -> bool:
    """Safely get a boolean parameter, handling various input types."""
    value = params.get(key)
    if value is None:
        return default
    # Already a boolean
    if isinstance(value, bool):
        return value
    # List from query string - get first element
    if isinstance(value, list):
        value = value[0] if value else default
        if isinstance(value, bool):
            return value
    # Convert to string and check truthy values
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    # Numbers (1/0) or other types
    return bool(value)


def get_string_param(params: dict[str, Any], key: str, default: str | None = None) -> str | None:
    """Safely get a string parameter, handling list values from query strings."""
    value = params.get(key, default)
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


def get_clamped_int_param(
    params: dict[str, Any],
    key: str,
    default: int,
    min_val: int,
    max_val: int,
) -> int:
    """Get integer parameter clamped to [min_val, max_val].

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Integer value clamped to the specified range
    """
    val = get_int_param(params, key, default)
    return min(max(val, min_val), max_val)


def get_bounded_float_param(
    params: dict[str, Any],
    key: str,
    default: float,
    min_val: float,
    max_val: float,
) -> float:
    """Get float parameter bounded to [min_val, max_val].

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Float value bounded to the specified range
    """
    val = get_float_param(params, key, default)
    return min(max(val, min_val), max_val)


def get_bounded_string_param(
    params: dict[str, Any],
    key: str,
    default: str | None = None,
    max_length: int = 500,
) -> str | None:
    """Get string parameter with length limit.

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        max_length: Maximum allowed string length

    Returns:
        String value truncated to max_length, or None
    """
    val = get_string_param(params, key, default)
    if val is None:
        return None
    return val[:max_length]


def get_pagination_params(
    params: dict[str, Any],
    default_limit: int = 100,
    max_limit: int = 1000,
    warn_threshold: int | None = None,
    context: str | None = None,
) -> tuple[int, int]:
    """Get standard pagination parameters (limit, offset) with bounds.

    Safely extracts 'limit' and 'offset' query parameters with:
    - limit: bounded to [1, max_limit], defaults to default_limit
    - offset: bounded to [0, 1_000_000], defaults to 0

    Args:
        params: Query parameters dict (from request.query or parsed query string)
        default_limit: Default limit if not specified (default: 100)
        max_limit: Maximum allowed limit (default: 1000)
        warn_threshold: Log warning if requested limit exceeds this value
                        (defaults to LARGE_PAGE_WARNING_THRESHOLD=500)
        context: Optional context string for warning messages (e.g., endpoint name)

    Returns:
        Tuple of (limit, offset) as bounded integers

    Example:
        limit, offset = get_pagination_params(dict(request.query))
    """
    # Get the raw requested limit before clamping (for warning)
    raw_limit = get_int_param(params, "limit", default_limit)

    limit = get_clamped_int_param(params, "limit", default_limit, 1, max_limit)
    offset = get_clamped_int_param(params, "offset", 0, 0, 1_000_000)

    # Warn about large page sizes that could cause memory pressure
    threshold = warn_threshold if warn_threshold is not None else LARGE_PAGE_WARNING_THRESHOLD
    if raw_limit > threshold:
        ctx = f" ({context})" if context else ""
        logger.warning(
            "Large page size requested%s: limit=%d (clamped to %d). "
            "Consider using pagination for better performance.",
            ctx,
            raw_limit,
            limit,
        )

    return limit, offset
