"""
OpenAPI Schema Generator for Aragora API.

This module provides OpenAPI 3.0 specification generation and export.
The original implementation is preserved in the parent module for
backward compatibility.

Usage:
    from aragora.server.openapi import generate_openapi_schema, save_openapi_schema

    # Get schema as dict
    schema = generate_openapi_schema()

    # Save to file
    path, count = save_openapi_schema("docs/api/openapi.json")

Submodules:
    - schemas: Common schema definitions and response helpers
"""

# Re-export from original module for backward compatibility
from aragora.server.openapi_impl import (
    API_VERSION,
    ALL_ENDPOINTS,
    COMMON_SCHEMAS,
    generate_openapi_schema,
    get_openapi_json,
    get_openapi_yaml,
    handle_openapi_request,
    save_openapi_schema,
    get_endpoint_count,
    generate_postman_collection,
    get_postman_json,
    save_postman_collection,
    handle_postman_request,
    _openapi_to_postman_request,
)

# Export from schemas submodule
from aragora.server.openapi.schemas import (
    STANDARD_ERRORS,
    ok_response,
    array_response,
    error_response,
)

__all__ = [
    # Version
    "API_VERSION",
    # Schemas
    "COMMON_SCHEMAS",
    "STANDARD_ERRORS",
    "ALL_ENDPOINTS",
    # Response helpers
    "ok_response",
    "array_response",
    "error_response",
    # Generator functions
    "generate_openapi_schema",
    "get_openapi_json",
    "get_openapi_yaml",
    "handle_openapi_request",
    "save_openapi_schema",
    "get_endpoint_count",
    # Postman
    "generate_postman_collection",
    "get_postman_json",
    "save_postman_collection",
    "handle_postman_request",
    "_openapi_to_postman_request",
]
