"""
Core validation utilities for request processing.

Provides ValidationResult dataclass and fundamental validation functions
for JSON bodies, content types, and field-level validation.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Max JSON body size (1MB by default, lower than file upload limit)
MAX_JSON_BODY_SIZE = 1 * 1024 * 1024


@dataclass
class ValidationResult:
    """Result of validation check."""

    is_valid: bool
    error: Optional[str] = None
    data: Optional[Any] = None


def validate_json_body(
    body: bytes,
    max_size: int = MAX_JSON_BODY_SIZE,
) -> ValidationResult:
    """Validate JSON body for size and format.

    Args:
        body: Raw request body bytes
        max_size: Maximum allowed size in bytes

    Returns:
        ValidationResult with parsed data or error
    """
    if len(body) > max_size:
        return ValidationResult(
            is_valid=False, error=f"Request body too large. Max size: {max_size // 1024}KB"
        )

    if len(body) == 0:
        return ValidationResult(is_valid=False, error="Request body is empty")

    try:
        data = json.loads(body.decode("utf-8"))
        return ValidationResult(is_valid=True, data=data)
    except json.JSONDecodeError as e:
        return ValidationResult(is_valid=False, error=f"Invalid JSON: {str(e)}")
    except UnicodeDecodeError:
        return ValidationResult(is_valid=False, error="Invalid UTF-8 encoding in request body")


def validate_content_type(
    content_type: str, expected: str = "application/json"
) -> ValidationResult:
    """Validate Content-Type header.

    Args:
        content_type: The Content-Type header value
        expected: Expected content type prefix

    Returns:
        ValidationResult with success or error
    """
    if not content_type:
        return ValidationResult(
            is_valid=False, error=f"Missing Content-Type header. Expected: {expected}"
        )

    if not content_type.lower().startswith(expected.lower()):
        return ValidationResult(
            is_valid=False, error=f"Invalid Content-Type: {content_type}. Expected: {expected}"
        )

    return ValidationResult(is_valid=True)


def validate_required_fields(data: dict, fields: list[str]) -> ValidationResult:
    """Validate that required fields are present.

    Args:
        data: Parsed JSON data
        fields: List of required field names

    Returns:
        ValidationResult with success or error
    """
    missing = [f for f in fields if f not in data or data[f] is None]

    if missing:
        return ValidationResult(
            is_valid=False, error=f"Missing required fields: {', '.join(missing)}"
        )

    return ValidationResult(is_valid=True)


def validate_string_field(
    data: dict,
    field: str,
    min_length: int = 0,
    max_length: int = 1000,
    pattern: Optional[re.Pattern] = None,
    required: bool = True,
) -> ValidationResult:
    """Validate a string field.

    Args:
        data: Parsed JSON data
        field: Field name to validate
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Optional regex pattern to match
        required: Whether the field is required

    Returns:
        ValidationResult with success or error
    """
    value = data.get(field)

    if value is None:
        if required:
            return ValidationResult(is_valid=False, error=f"Missing required field: {field}")
        return ValidationResult(is_valid=True)

    if not isinstance(value, str):
        return ValidationResult(is_valid=False, error=f"Field '{field}' must be a string")

    if len(value) < min_length:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at least {min_length} characters"
        )

    if len(value) > max_length:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at most {max_length} characters"
        )

    if pattern and not pattern.match(value):
        return ValidationResult(is_valid=False, error=f"Field '{field}' has invalid format")

    return ValidationResult(is_valid=True)


def validate_string(
    value: Optional[str],
    field_name: str,
    pattern: Optional[re.Pattern] = None,
    min_length: int = 0,
    max_length: int = 1000,
) -> tuple[bool, Optional[str]]:
    """Validate a string value directly.

    Unlike validate_string_field which operates on a dict, this validates
    a single value and returns a tuple for easier inline usage.

    Args:
        value: String value to validate
        field_name: Field name for error messages
        pattern: Optional regex pattern to match
        min_length: Minimum string length
        max_length: Maximum string length

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if value is None:
        return False, f"Missing required field: {field_name}"

    if not isinstance(value, str):
        return False, f"Field '{field_name}' must be a string"

    if len(value) < min_length:
        return False, f"Field '{field_name}' must be at least {min_length} characters"

    if len(value) > max_length:
        return False, f"Field '{field_name}' must be at most {max_length} characters"

    if pattern and not pattern.match(value):
        return False, f"Field '{field_name}' has invalid format"

    return True, None


def validate_int_field(
    data: dict,
    field: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    required: bool = True,
) -> ValidationResult:
    """Validate an integer field.

    Args:
        data: Parsed JSON data
        field: Field name to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        required: Whether the field is required

    Returns:
        ValidationResult with success or error
    """
    value = data.get(field)

    if value is None:
        if required:
            return ValidationResult(is_valid=False, error=f"Missing required field: {field}")
        return ValidationResult(is_valid=True)

    if not isinstance(value, int) or isinstance(value, bool):
        return ValidationResult(is_valid=False, error=f"Field '{field}' must be an integer")

    if min_value is not None and value < min_value:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at least {min_value}"
        )

    if max_value is not None and value > max_value:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at most {max_value}"
        )

    return ValidationResult(is_valid=True)


def validate_float_field(
    data: dict,
    field: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required: bool = True,
) -> ValidationResult:
    """Validate a float field.

    Args:
        data: Parsed JSON data
        field: Field name to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        required: Whether the field is required

    Returns:
        ValidationResult with success or error
    """
    value = data.get(field)

    if value is None:
        if required:
            return ValidationResult(is_valid=False, error=f"Missing required field: {field}")
        return ValidationResult(is_valid=True)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ValidationResult(is_valid=False, error=f"Field '{field}' must be a number")

    if min_value is not None and value < min_value:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at least {min_value}"
        )

    if max_value is not None and value > max_value:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be at most {max_value}"
        )

    return ValidationResult(is_valid=True)


def validate_list_field(
    data: dict,
    field: str,
    min_length: int = 0,
    max_length: int = 100,
    item_type: Optional[type] = None,
    required: bool = True,
) -> ValidationResult:
    """Validate a list field.

    Args:
        data: Parsed JSON data
        field: Field name to validate
        min_length: Minimum list length
        max_length: Maximum list length
        item_type: Expected type of list items
        required: Whether the field is required

    Returns:
        ValidationResult with success or error
    """
    value = data.get(field)

    if value is None:
        if required:
            return ValidationResult(is_valid=False, error=f"Missing required field: {field}")
        return ValidationResult(is_valid=True)

    if not isinstance(value, list):
        return ValidationResult(is_valid=False, error=f"Field '{field}' must be a list")

    if len(value) < min_length:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must have at least {min_length} items"
        )

    if len(value) > max_length:
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must have at most {max_length} items"
        )

    if item_type is not None:
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                return ValidationResult(
                    is_valid=False,
                    error=f"Field '{field}[{i}]' must be of type {item_type.__name__}",
                )

    return ValidationResult(is_valid=True)


def validate_enum_field(
    data: dict,
    field: str,
    allowed_values: set,
    required: bool = True,
) -> ValidationResult:
    """Validate a field against allowed values.

    Args:
        data: Parsed JSON data
        field: Field name to validate
        allowed_values: Set of allowed values
        required: Whether the field is required

    Returns:
        ValidationResult with success or error
    """
    value = data.get(field)

    if value is None:
        if required:
            return ValidationResult(is_valid=False, error=f"Missing required field: {field}")
        return ValidationResult(is_valid=True)

    if value not in allowed_values:
        allowed_str = ", ".join(str(v) for v in sorted(allowed_values))
        return ValidationResult(
            is_valid=False, error=f"Field '{field}' must be one of: {allowed_str}"
        )

    return ValidationResult(is_valid=True)


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize a string by stripping and truncating.

    Args:
        value: String to sanitize
        max_length: Maximum length to truncate to

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_length]
