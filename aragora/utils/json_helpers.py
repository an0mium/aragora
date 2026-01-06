"""Safe JSON parsing utilities."""

import json
import logging
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_json_loads(data: str | None, default: T = None) -> T | Any:
    """Safely parse JSON string with fallback to default.

    Args:
        data: JSON string to parse (can be None)
        default: Value to return on failure (defaults to {} if None)

    Returns:
        Parsed JSON or default value
    """
    if not data:
        return default if default is not None else {}
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON data: %s", e)
        return default if default is not None else {}
