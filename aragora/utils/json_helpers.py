"""Safe JSON parsing and extraction utilities.

This module provides utilities for safely handling JSON data,
especially in the context of LLM responses which may contain
JSON embedded in markdown or other text.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_json_loads(data: str | None, default: T = None, context: str | None = None) -> T | Any:
    """Safely parse JSON string with fallback to default.

    Args:
        data: JSON string to parse (can be None)
        default: Value to return on failure (defaults to {} if None)
        context: Optional context for logging (e.g., "consensus:abc123")

    Returns:
        Parsed JSON or default value
    """
    if not data:
        return default if default is not None else {}
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        ctx_str = f" [{context}]" if context else ""
        logger.warning("Failed to parse JSON data%s: %s", ctx_str, e)
        return default if default is not None else {}


def _extract_balanced_json(text: str, open_char: str, close_char: str) -> str | None:
    """Extract a balanced JSON structure (object or array) from text.

    Uses brace counting to find the first complete JSON structure,
    handling nested objects correctly.

    Args:
        text: Text potentially containing JSON
        open_char: Opening character ('{' or '[')
        close_char: Closing character ('}' or ']')

    Returns:
        Extracted JSON string, or None if no balanced structure found
    """
    start = text.find(open_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain other content.

    Useful for parsing LLM responses that may wrap JSON in markdown
    code blocks or include explanatory text.

    Uses balanced brace counting to correctly extract nested JSON
    structures without collapsing multiple JSON blocks.

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted JSON string, or original text if no JSON found

    Examples:
        >>> extract_json_from_text('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_text('Here is the result: {"key": "value"}')
        '{"key": "value"}'
    """
    # Try to find JSON in code blocks first (```json or ```)
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    # Try to find JSON array in code blocks
    array_block_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if array_block_match:
        return array_block_match.group(1)

    # Try to find raw JSON object using balanced brace extraction
    json_obj = _extract_balanced_json(text, "{", "}")
    if json_obj:
        return json_obj

    # Try to find raw JSON array using balanced bracket extraction
    json_arr = _extract_balanced_json(text, "[", "]")
    if json_arr:
        return json_arr

    return text


def extract_and_parse_json(
    text: str,
    default: T = None,
    context: str | None = None,
) -> T | Any:
    """Extract JSON from text and parse it safely.

    Combines extract_json_from_text and safe_json_loads for convenience.

    Args:
        text: Text potentially containing JSON
        default: Value to return if extraction or parsing fails
        context: Optional context for logging

    Returns:
        Parsed JSON data or default value
    """
    extracted = extract_json_from_text(text)
    return safe_json_loads(extracted, default=default, context=context)


def parse_json_env(
    env_var: str,
    default: T = None,
    context: str | None = None,
) -> T | Any:
    """Safely parse JSON from an environment variable.

    Args:
        env_var: Name of the environment variable
        default: Value to return if env var is missing or invalid
        context: Optional context for logging

    Returns:
        Parsed JSON data or default value

    Examples:
        >>> os.environ["MY_CONFIG"] = '{"key": "value"}'
        >>> parse_json_env("MY_CONFIG")
        {'key': 'value'}
    """
    value = os.environ.get(env_var)
    if value is None:
        return default if default is not None else {}

    ctx = context or f"env:{env_var}"
    return safe_json_loads(value, default=default, context=ctx)


def validate_json_keys(
    data: dict[str, Any],
    required_keys: list[str],
    context: str | None = None,
) -> list[str]:
    """Validate that a JSON dict contains required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of keys that must be present
        context: Optional context for error messages

    Returns:
        List of error messages (empty if valid)

    Examples:
        >>> validate_json_keys({"a": 1}, ["a", "b"])
        ["Missing required key: 'b'"]
    """
    errors = []
    ctx_str = f" [{context}]" if context else ""

    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key{ctx_str}: '{key}'")

    return errors


def validate_vote_response(text: str) -> dict[str, Any] | None:
    """Validate and parse a vote response from an agent.

    Expected structure:
    {
        "position": "...",
        "confidence": 0.0-1.0,
        "rationale": "..."
    }

    Args:
        text: Raw text response from agent

    Returns:
        Validated vote dict or None if invalid
    """
    data = extract_and_parse_json(text, default=None, context="vote_response")

    if not isinstance(data, dict):
        logger.warning("Vote response is not a dict: %s", type(data))
        return None

    errors = validate_json_keys(data, ["position"], context="vote_response")
    if errors:
        logger.warning("Invalid vote response: %s", errors)
        return None

    # Normalize confidence to float if present
    if "confidence" in data:
        try:
            data["confidence"] = float(data["confidence"])
            if not 0.0 <= data["confidence"] <= 1.0:
                data["confidence"] = max(0.0, min(1.0, data["confidence"]))
        except (ValueError, TypeError):
            data["confidence"] = 0.5

    return data


def validate_critique_response(text: str) -> dict[str, Any] | None:
    """Validate and parse a critique response from an agent.

    Expected structure:
    {
        "target_agent": "...",
        "severity": 0.0-1.0,
        "critique": "...",
        "suggestions": [...]
    }

    Args:
        text: Raw text response from agent

    Returns:
        Validated critique dict or None if invalid
    """
    data = extract_and_parse_json(text, default=None, context="critique_response")

    if not isinstance(data, dict):
        logger.warning("Critique response is not a dict: %s", type(data))
        return None

    errors = validate_json_keys(data, ["critique"], context="critique_response")
    if errors:
        logger.warning("Invalid critique response: %s", errors)
        return None

    # Normalize severity to float if present
    if "severity" in data:
        try:
            data["severity"] = float(data["severity"])
            if not 0.0 <= data["severity"] <= 1.0:
                data["severity"] = max(0.0, min(1.0, data["severity"]))
        except (ValueError, TypeError):
            data["severity"] = 0.5

    return data
