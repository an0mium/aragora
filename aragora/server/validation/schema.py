"""
JSON schema validation for API requests.

Provides schema definitions for common endpoints and a function
to validate data against these schemas.
"""

import re

from .core import (
    ValidationResult,
    validate_string_field,
    validate_int_field,
    validate_float_field,
    validate_list_field,
    validate_enum_field,
)
from .entities import SAFE_AGENT_PATTERN


# =============================================================================
# Endpoint-Specific Validation Schemas
# =============================================================================

DEBATE_START_SCHEMA = {
    "task": {"type": "string", "min_length": 1, "max_length": 2000, "required": True},
    "agents": {"type": "list", "min_length": 2, "max_length": 10, "item_type": str, "required": False},
    "mode": {"type": "string", "max_length": 64, "required": False},
    "rounds": {"type": "int", "min_value": 1, "max_value": 20, "required": False},
}

VERIFICATION_SCHEMA = {
    "claim": {"type": "string", "min_length": 1, "max_length": 5000, "required": True},
    "context": {"type": "string", "max_length": 10000, "required": False},
}

PROBE_RUN_SCHEMA = {
    "agent_name": {"type": "string", "min_length": 1, "max_length": 64, "pattern": SAFE_AGENT_PATTERN, "required": True},
    "probe_types": {"type": "list", "max_length": 10, "item_type": str, "required": False},
    "probes_per_type": {"type": "int", "min_value": 1, "max_value": 10, "required": False},
    "model_type": {"type": "string", "max_length": 64, "required": False},
}

FORK_REQUEST_SCHEMA = {
    "branch_point": {"type": "int", "min_value": 0, "max_value": 100, "required": True},
    "modified_context": {"type": "string", "max_length": 5000, "required": False},
}

MEMORY_CLEANUP_SCHEMA = {
    "tier": {"type": "enum", "allowed_values": {"fast", "medium", "slow", "glacial"}, "required": False},
    "archive": {"type": "string", "max_length": 10, "required": False},  # "true" or "false"
    "max_age_hours": {"type": "float", "min_value": 0.0, "max_value": 8760.0, "required": False},  # Max 1 year
}

# Agent configuration schema
AGENT_CONFIG_SCHEMA = {
    "name": {"type": "string", "min_length": 1, "max_length": 64, "pattern": SAFE_AGENT_PATTERN, "required": True},
    "model": {"type": "string", "max_length": 100, "required": False},
    "temperature": {"type": "float", "min_value": 0.0, "max_value": 2.0, "required": False},
    "max_tokens": {"type": "int", "min_value": 1, "max_value": 100000, "required": False},
    "system_prompt": {"type": "string", "max_length": 10000, "required": False},
}

# Batch debate submission schema
BATCH_SUBMIT_SCHEMA = {
    "debates": {"type": "list", "min_length": 1, "max_length": 100, "item_type": dict, "required": True},
    "priority": {"type": "enum", "allowed_values": {"low", "normal", "high"}, "required": False},
    "callback_url": {"type": "string", "max_length": 500, "required": False},
}

# User/auth schemas
USER_REGISTER_SCHEMA = {
    "email": {"type": "string", "min_length": 5, "max_length": 255, "required": True},
    "password": {"type": "string", "min_length": 8, "max_length": 128, "required": True},
    "name": {"type": "string", "max_length": 100, "required": False},
}

USER_LOGIN_SCHEMA = {
    "email": {"type": "string", "min_length": 5, "max_length": 255, "required": True},
    "password": {"type": "string", "min_length": 1, "max_length": 128, "required": True},
}

# Organization schemas
ORG_CREATE_SCHEMA = {
    "name": {"type": "string", "min_length": 1, "max_length": 100, "required": True},
    "slug": {"type": "string", "max_length": 100, "required": False},
}

ORG_INVITE_SCHEMA = {
    "email": {"type": "string", "min_length": 5, "max_length": 255, "required": True},
    "role": {"type": "enum", "allowed_values": {"member", "admin"}, "required": False},
}

# Gauntlet run schema
GAUNTLET_RUN_SCHEMA = {
    "input_content": {"type": "string", "min_length": 1, "max_length": 50000, "required": True},
    "input_type": {"type": "enum", "allowed_values": {"spec", "code", "text", "url", "file"}, "required": False},
    "agents": {"type": "list", "max_length": 10, "item_type": str, "required": False},
    "persona": {"type": "string", "max_length": 100, "required": False},
    "profile": {"type": "string", "max_length": 100, "required": False},
}


def validate_against_schema(data: dict, schema: dict) -> ValidationResult:
    """Validate data against a schema definition.

    Args:
        data: Parsed JSON data
        schema: Schema definition dict

    Returns:
        ValidationResult with success or error

    Schema format:
        {
            "field_name": {
                "type": "string" | "int" | "float" | "list" | "enum",
                "required": bool,
                # Type-specific options:
                "min_length": int,  # For strings/lists
                "max_length": int,  # For strings/lists
                "pattern": re.Pattern,  # For strings
                "min_value": number,  # For int/float
                "max_value": number,  # For int/float
                "item_type": type,  # For lists
                "allowed_values": set,  # For enums
            },
            ...
        }

    Example:
        >>> result = validate_against_schema(
        ...     {"task": "Test", "rounds": 3},
        ...     DEBATE_START_SCHEMA
        ... )
        >>> if not result.is_valid:
        ...     return error_response(400, result.error)
    """
    for field, rules in schema.items():
        field_type = rules.get("type", "string")
        required = rules.get("required", True)

        if field_type == "string":
            result = validate_string_field(
                data, field,
                min_length=rules.get("min_length", 0),
                max_length=rules.get("max_length", 1000),
                pattern=rules.get("pattern"),
                required=required,
            )
        elif field_type == "int":
            result = validate_int_field(
                data, field,
                min_value=rules.get("min_value"),
                max_value=rules.get("max_value"),
                required=required,
            )
        elif field_type == "float":
            result = validate_float_field(
                data, field,
                min_value=rules.get("min_value"),
                max_value=rules.get("max_value"),
                required=required,
            )
        elif field_type == "list":
            result = validate_list_field(
                data, field,
                min_length=rules.get("min_length", 0),
                max_length=rules.get("max_length", 100),
                item_type=rules.get("item_type"),
                required=required,
            )
        elif field_type == "enum":
            result = validate_enum_field(
                data, field,
                allowed_values=rules.get("allowed_values", set()),
                required=required,
            )
        else:
            continue  # Unknown type, skip

        if not result.is_valid:
            return result

    return ValidationResult(is_valid=True, data=data)
