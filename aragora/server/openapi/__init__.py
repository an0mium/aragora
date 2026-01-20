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

from typing import TYPE_CHECKING

# Export from schemas submodule (no circular import - schemas doesn't import openapi_impl)
from aragora.server.openapi.schemas import (
    COMMON_SCHEMAS,
    STANDARD_ERRORS,
    ok_response,
    array_response,
    error_response,
)

# Use lazy imports to avoid circular import with openapi_impl
# openapi_impl imports from openapi.schemas, and if we import from openapi_impl here,
# we get a circular import since this __init__.py loads when openapi.schemas is accessed

if TYPE_CHECKING:
    from aragora.server.openapi_impl import (
        API_VERSION,
        ALL_ENDPOINTS,
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
    )
    from aragora.server.postman_generator import _openapi_to_postman_request

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

# Lazy import implementation to break circular import
_lazy_imports = {
    "API_VERSION": "aragora.server.openapi_impl",
    "ALL_ENDPOINTS": "aragora.server.openapi_impl",
    "generate_openapi_schema": "aragora.server.openapi_impl",
    "get_openapi_json": "aragora.server.openapi_impl",
    "get_openapi_yaml": "aragora.server.openapi_impl",
    "handle_openapi_request": "aragora.server.openapi_impl",
    "save_openapi_schema": "aragora.server.openapi_impl",
    "get_endpoint_count": "aragora.server.openapi_impl",
    "generate_postman_collection": "aragora.server.openapi_impl",
    "get_postman_json": "aragora.server.openapi_impl",
    "save_postman_collection": "aragora.server.openapi_impl",
    "handle_postman_request": "aragora.server.openapi_impl",
    "_openapi_to_postman_request": "aragora.server.postman_generator",
}


def __getattr__(name: str):
    """Lazy import handler to avoid circular imports."""
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
