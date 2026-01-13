"""
JSON schema validation for API requests.

Provides schema definitions for common endpoints and a function
to validate data against these schemas.
"""

from .core import (
    ValidationResult,
    validate_enum_field,
    validate_float_field,
    validate_int_field,
    validate_list_field,
    validate_string_field,
)
from .entities import SAFE_AGENT_PATTERN

# =============================================================================
# Endpoint-Specific Validation Schemas
# =============================================================================

DEBATE_START_SCHEMA = {
    "task": {
        "type": "string",
        "min_length": 1,
        "max_length": 2000,
        "required": False,
    },  # Can use 'question' too
    "question": {"type": "string", "min_length": 1, "max_length": 2000, "required": False},
    "agents": {
        "type": "list",
        "min_length": 2,
        "max_length": 10,
        "item_type": str,
        "required": False,
    },
    "mode": {"type": "string", "max_length": 64, "required": False},
    "rounds": {"type": "int", "min_value": 1, "max_value": 20, "required": False},
    "consensus": {"type": "string", "max_length": 64, "required": False},
}

DEBATE_UPDATE_SCHEMA = {
    "title": {"type": "string", "max_length": 500, "required": False},
    "status": {
        "type": "enum",
        "allowed_values": {"active", "paused", "concluded", "archived"},
        "required": False,
    },
    "tags": {"type": "list", "max_length": 20, "item_type": str, "required": False},
}

VERIFICATION_SCHEMA = {
    "claim": {"type": "string", "min_length": 1, "max_length": 5000, "required": True},
    "context": {"type": "string", "max_length": 10000, "required": False},
}

PROBE_RUN_SCHEMA = {
    "agent_name": {
        "type": "string",
        "min_length": 1,
        "max_length": 64,
        "pattern": SAFE_AGENT_PATTERN,
        "required": True,
    },
    "probe_types": {"type": "list", "max_length": 10, "item_type": str, "required": False},
    "probes_per_type": {"type": "int", "min_value": 1, "max_value": 10, "required": False},
    "model_type": {"type": "string", "max_length": 64, "required": False},
}

FORK_REQUEST_SCHEMA = {
    "branch_point": {"type": "int", "min_value": 0, "max_value": 100, "required": True},
    "modified_context": {"type": "string", "max_length": 5000, "required": False},
}

MEMORY_CLEANUP_SCHEMA = {
    "tier": {
        "type": "enum",
        "allowed_values": {"fast", "medium", "slow", "glacial"},
        "required": False,
    },
    "archive": {"type": "string", "max_length": 10, "required": False},  # "true" or "false"
    "max_age_hours": {
        "type": "float",
        "min_value": 0.0,
        "max_value": 8760.0,
        "required": False,
    },  # Max 1 year
}

# Agent configuration schema
AGENT_CONFIG_SCHEMA = {
    "name": {
        "type": "string",
        "min_length": 1,
        "max_length": 64,
        "pattern": SAFE_AGENT_PATTERN,
        "required": True,
    },
    "model": {"type": "string", "max_length": 100, "required": False},
    "temperature": {"type": "float", "min_value": 0.0, "max_value": 2.0, "required": False},
    "max_tokens": {"type": "int", "min_value": 1, "max_value": 100000, "required": False},
    "system_prompt": {"type": "string", "max_length": 10000, "required": False},
}

# Batch debate submission schema
BATCH_SUBMIT_SCHEMA = {
    "items": {
        "type": "list",
        "min_length": 1,
        "max_length": 1000,
        "item_type": dict,
        "required": True,
    },
    "webhook_url": {"type": "string", "max_length": 2000, "required": False},
    "max_parallel": {"type": "int", "min_value": 1, "max_value": 50, "required": False},
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
    "input_type": {
        "type": "enum",
        "allowed_values": {"spec", "code", "text", "url", "file"},
        "required": False,
    },
    "agents": {"type": "list", "max_length": 10, "item_type": str, "required": False},
    "persona": {"type": "string", "max_length": 100, "required": False},
    "profile": {"type": "string", "max_length": 100, "required": False},
}

# Billing checkout schema
CHECKOUT_SESSION_SCHEMA = {
    "tier": {
        "type": "enum",
        "allowed_values": {"starter", "professional", "enterprise"},
        "required": True,
    },
    "success_url": {"type": "string", "min_length": 1, "max_length": 2000, "required": True},
    "cancel_url": {"type": "string", "min_length": 1, "max_length": 2000, "required": True},
}

# Social publishing schema (all optional since body can be empty)
SOCIAL_PUBLISH_SCHEMA = {
    "include_audio_link": {"type": "string", "max_length": 10, "required": False},  # "true"/"false"
    "thread_mode": {"type": "string", "max_length": 10, "required": False},
    "title": {"type": "string", "max_length": 200, "required": False},
    "description": {"type": "string", "max_length": 5000, "required": False},
    "tags": {"type": "list", "max_length": 20, "item_type": str, "required": False},
}

# Plugin execution schema
PLUGIN_RUN_SCHEMA = {
    "input": {"type": "string", "max_length": 100000, "required": False},  # Can also be dict
    "config": {"type": "string", "max_length": 10000, "required": False},  # Config dict
    "working_dir": {"type": "string", "max_length": 500, "required": False},
}

# Plugin install schema
PLUGIN_INSTALL_SCHEMA = {
    "config": {"type": "string", "max_length": 10000, "required": False},
    "enabled": {"type": "string", "max_length": 10, "required": False},  # "true"/"false"
}

# Sharing update schema
SHARE_UPDATE_SCHEMA = {
    "visibility": {
        "type": "enum",
        "allowed_values": {"private", "team", "public"},
        "required": False,
    },
    "expires_in_hours": {
        "type": "int",
        "min_value": 0,
        "max_value": 8760,
        "required": False,
    },  # Max 1 year
    "allow_comments": {"type": "string", "max_length": 10, "required": False},  # bool as string
    "allow_forking": {"type": "string", "max_length": 10, "required": False},  # bool as string
}

# Email configuration schema
EMAIL_CONFIG_SCHEMA = {
    "smtp_host": {"type": "string", "max_length": 255, "required": False},
    "smtp_port": {"type": "int", "min_value": 1, "max_value": 65535, "required": False},
    "smtp_username": {"type": "string", "max_length": 255, "required": False},
    "smtp_password": {"type": "string", "max_length": 255, "required": False},
    "from_email": {"type": "string", "max_length": 255, "required": False},
    "from_name": {"type": "string", "max_length": 100, "required": False},
}

# Telegram configuration schema
TELEGRAM_CONFIG_SCHEMA = {
    "bot_token": {"type": "string", "min_length": 1, "max_length": 100, "required": True},
    "chat_id": {"type": "string", "min_length": 1, "max_length": 50, "required": True},
}

# Notification send schema
NOTIFICATION_SEND_SCHEMA = {
    "type": {"type": "enum", "allowed_values": {"all", "email", "telegram"}, "required": False},
    "subject": {"type": "string", "max_length": 200, "required": False},
    "message": {"type": "string", "min_length": 1, "max_length": 10000, "required": True},
    "html_message": {"type": "string", "max_length": 50000, "required": False},
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
                data,
                field,
                min_length=rules.get("min_length", 0),
                max_length=rules.get("max_length", 1000),
                pattern=rules.get("pattern"),
                required=required,
            )
        elif field_type == "int":
            result = validate_int_field(
                data,
                field,
                min_value=rules.get("min_value"),
                max_value=rules.get("max_value"),
                required=required,
            )
        elif field_type == "float":
            result = validate_float_field(
                data,
                field,
                min_value=rules.get("min_value"),
                max_value=rules.get("max_value"),
                required=required,
            )
        elif field_type == "list":
            result = validate_list_field(
                data,
                field,
                min_length=rules.get("min_length", 0),
                max_length=rules.get("max_length", 100),
                item_type=rules.get("item_type"),
                required=required,
            )
        elif field_type == "enum":
            result = validate_enum_field(
                data,
                field,
                allowed_values=rules.get("allowed_values", set()),
                required=required,
            )
        else:
            continue  # Unknown type, skip

        if not result.is_valid:
            return result

    return ValidationResult(is_valid=True, data=data)
