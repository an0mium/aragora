"""
Safe data access utilities for handler modules.

Provides defensive data access patterns for safely navigating
dicts, parsing JSON, and handling None values gracefully.
"""

import json
from typing import Any


def safe_get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object.

    Handles None values, missing keys, and non-dict inputs gracefully.

    Args:
        data: Dict or dict-like object (can be None)
        key: Key to look up
        default: Default value if key not found or data is None

    Returns:
        The value at key, or default

    Example:
        # Before:
        value = data.get("key", []) if data else []

        # After:
        value = safe_get(data, "key", [])
    """
    if data is None:
        return default
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def safe_get_nested(data: Any, keys: list[str], default: Any = None) -> Any:
    """Safely navigate nested dict structures.

    Args:
        data: Root dict or dict-like object (can be None)
        keys: List of keys to traverse
        default: Default value if any key not found

    Returns:
        The nested value, or default

    Example:
        # Before:
        value = data.get("outer", {}).get("inner", {}).get("deep", [])

        # After:
        value = safe_get_nested(data, ["outer", "inner", "deep"], [])
    """
    current = data
    for key in keys:
        if current is None:
            return default
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def safe_json_parse(data: Any, default: Any = None) -> Any:
    """Safely parse JSON string or return dict/list as-is.

    Handles the common pattern where a value might be stored as either
    a JSON string or already parsed as a dict/list.

    Args:
        data: JSON string, dict, list, or None
        default: Default value if parsing fails or data is None

    Returns:
        Parsed dict/list, or default on failure

    Example:
        # Before (6 lines):
        raw = debate.get("grounded_verdict")
        if isinstance(raw, str):
            try:
                verdict = json.loads(raw)
            except json.JSONDecodeError:
                verdict = None

        # After (1 line):
        verdict = safe_json_parse(debate.get("grounded_verdict"))
    """
    if data is None:
        return default
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, (str, bytes, bytearray)):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


__all__ = [
    "safe_get",
    "safe_get_nested",
    "safe_json_parse",
]
