"""
HTTP utility functions for request handling.

Provides validation, type conversion, and async execution utilities
used across the unified server.
"""

import logging

# Import run_async from central utils for re-export
from aragora.utils.async_utils import run_async

logger = logging.getLogger(__name__)


# Query parameter whitelist (security: reject unknown params to prevent injection)
# Maps param name -> validation rule:
#   - None: numeric/short params (no length limit, validated elsewhere)
#   - set: restricted to specific values
#   - int: max length for string params (DoS protection)
ALLOWED_QUERY_PARAMS = {
    # Pagination (numeric, validated by int parsing)
    "limit": None,
    "offset": None,
    # Filtering (string, need length limits)
    "domain": 100,
    "loop_id": 100,
    "topic": 500,
    "query": 1000,
    # Export
    "table": {"summary", "debates", "proposals", "votes", "critiques", "messages"},
    # Agent queries
    "agent": 100,
    "agent_a": 100,
    "agent_b": 100,
    "sections": {"identity", "performance", "relationships", "all"},
    # Calibration
    "buckets": None,
    # Memory
    "tiers": 100,
    "min_importance": None,
    # Genesis
    "event_type": {"mutation", "crossover", "selection", "extinction", "speciation"},
    # Logs
    "lines": None,
}


def validate_query_params(query: dict) -> tuple[bool, str]:
    """Validate query parameters against whitelist.

    Returns (is_valid, error_message).

    Validation rules:
    - None: no length validation (for numeric params)
    - set: value must be in the set
    - int: max length for string params
    """
    for param, values in query.items():
        if param not in ALLOWED_QUERY_PARAMS:
            return False, f"Unknown query parameter: {param}"

        allowed = ALLOWED_QUERY_PARAMS[param]
        if allowed is None:
            # No validation needed (numeric params validated elsewhere)
            continue

        if isinstance(allowed, set):
            # Check if value is in the allowed set
            for val in values:
                if val not in allowed:
                    return False, f"Invalid value for {param}: {val}"
        elif isinstance(allowed, int):
            # Check length limit
            for val in values:
                if len(val) > allowed:
                    return False, f"Parameter {param} exceeds max length ({allowed})"

    return True, ""


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# run_async is imported from aragora.utils.async_utils (see top of file)

# Backward compatibility aliases (prefixed with underscore)
_validate_query_params = validate_query_params
_safe_float = safe_float
_safe_int = safe_int
_run_async = run_async


__all__ = [
    "ALLOWED_QUERY_PARAMS",
    "validate_query_params",
    "safe_float",
    "safe_int",
    "run_async",
    # Backward compatibility
    "_validate_query_params",
    "_safe_float",
    "_safe_int",
    "_run_async",
]
