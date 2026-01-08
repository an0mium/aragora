"""
Request validation package for Aragora server.

Provides JSON schema validation, query parameter parsing, entity ID validation,
and validation decorators for handler methods.

Modules:
- core: ValidationResult, field validators, JSON body validation
- entities: Path segment validators (debate_id, agent_name, etc.)
- query_params: Query parameter parsing with bounds checking
- schema: Schema definitions and validate_against_schema
- decorators: Handler validation decorators

Usage:
    from aragora.server.validation import (
        # Core validation
        ValidationResult,
        validate_json_body,
        validate_content_type,
        validate_required_fields,

        # Entity validators
        validate_debate_id,
        validate_agent_name,
        SAFE_ID_PATTERN,

        # Query params
        safe_query_int,
        validate_sort_params,

        # Schema validation
        validate_against_schema,
        DEBATE_START_SCHEMA,

        # Decorators
        validate_request,
        validate_post_body,
    )
"""

# Core validation
from .core import (
    MAX_JSON_BODY_SIZE,
    ValidationResult,
    validate_json_body,
    validate_content_type,
    validate_required_fields,
    validate_string_field,
    validate_int_field,
    validate_float_field,
    validate_list_field,
    validate_enum_field,
    sanitize_string,
)

# Entity validators
from .entities import (
    SAFE_ID_PATTERN,
    SAFE_ID_PATTERN_WITH_DOTS,
    SAFE_SLUG_PATTERN,
    SAFE_AGENT_PATTERN,
    validate_path_segment,
    validate_id,
    validate_agent_name,
    validate_debate_id,
    validate_plugin_name,
    validate_loop_id,
    validate_replay_id,
    validate_genome_id,
    validate_agent_name_with_version,
    validate_no_path_traversal,
    sanitize_id,
)

# Query parameter parsing
from .query_params import (
    DEFAULT_QUERY_STRING_MAX_LENGTH,
    ALLOWED_SORT_COLUMNS,
    ALLOWED_SORT_DIRECTIONS,
    ALLOWED_FILTER_OPERATORS,
    parse_int_param,
    parse_float_param,
    parse_bool_param,
    parse_string_param,
    safe_query_int,
    safe_query_float,
    validate_sort_param,
    validate_sort_direction,
    validate_sort_params,
    safe_query_string,
    validate_filter_operator,
    validate_search_query,
)

# Schema validation
from .schema import (
    DEBATE_START_SCHEMA,
    VERIFICATION_SCHEMA,
    PROBE_RUN_SCHEMA,
    FORK_REQUEST_SCHEMA,
    MEMORY_CLEANUP_SCHEMA,
    validate_against_schema,
)

# Decorators
from .decorators import (
    validate_request,
    validate_post_body,
    validate_query_params,
)

__all__ = [
    # Core
    "MAX_JSON_BODY_SIZE",
    "ValidationResult",
    "validate_json_body",
    "validate_content_type",
    "validate_required_fields",
    "validate_string_field",
    "validate_int_field",
    "validate_float_field",
    "validate_list_field",
    "validate_enum_field",
    "sanitize_string",
    # Entities
    "SAFE_ID_PATTERN",
    "SAFE_ID_PATTERN_WITH_DOTS",
    "SAFE_SLUG_PATTERN",
    "SAFE_AGENT_PATTERN",
    "validate_path_segment",
    "validate_id",
    "validate_agent_name",
    "validate_debate_id",
    "validate_plugin_name",
    "validate_loop_id",
    "validate_replay_id",
    "validate_genome_id",
    "validate_agent_name_with_version",
    "validate_no_path_traversal",
    "sanitize_id",
    # Query params
    "DEFAULT_QUERY_STRING_MAX_LENGTH",
    "ALLOWED_SORT_COLUMNS",
    "ALLOWED_SORT_DIRECTIONS",
    "ALLOWED_FILTER_OPERATORS",
    "parse_int_param",
    "parse_float_param",
    "parse_bool_param",
    "parse_string_param",
    "safe_query_int",
    "safe_query_float",
    "validate_sort_param",
    "validate_sort_direction",
    "validate_sort_params",
    "safe_query_string",
    "validate_filter_operator",
    "validate_search_query",
    # Schema
    "DEBATE_START_SCHEMA",
    "VERIFICATION_SCHEMA",
    "PROBE_RUN_SCHEMA",
    "FORK_REQUEST_SCHEMA",
    "MEMORY_CLEANUP_SCHEMA",
    "validate_against_schema",
    # Decorators
    "validate_request",
    "validate_post_body",
    "validate_query_params",
]
