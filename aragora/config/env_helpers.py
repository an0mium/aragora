"""
Environment variable helper utilities.

This module provides common utilities for reading configuration values
from environment variables with type conversion and fallback defaults.

These helpers are used across multiple config modules to avoid duplication.
"""

import os
from typing import List, Optional


def env_str(key: str, default: str = "") -> str:
    """Get string from environment with fallback.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Environment value or default
    """
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    """Get integer from environment with fallback.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Integer value from environment or default
    """
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    """Get float from environment with fallback.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Float value from environment or default
    """
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment with fallback.

    Recognizes "true", "1", "yes", "on" as True (case-insensitive).
    All other values are treated as False.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    """Get list from comma-separated environment variable.

    Args:
        key: Environment variable name
        default: Default list if not set
        separator: Separator character (default: comma)

    Returns:
        List of stripped strings
    """
    if default is None:
        default = []
    value = os.environ.get(key, "")
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


# Aliases for backward compatibility (private naming convention)
_env_str = env_str
_env_int = env_int
_env_float = env_float
_env_bool = env_bool


__all__ = [
    "env_str",
    "env_int",
    "env_float",
    "env_bool",
    "env_list",
    # Private aliases for backward compatibility
    "_env_str",
    "_env_int",
    "_env_float",
    "_env_bool",
]
