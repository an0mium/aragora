"""
OpenAPI Response Helpers.

Utility functions for creating standardized response definitions.
"""

from typing import Any


def ok_response(description: str, schema_ref: str | None = None) -> dict[str, Any]:
    """Create a successful response definition."""
    resp: dict[str, Any] = {"description": description}
    if schema_ref:
        resp["content"] = {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_ref}"}}
        }
    return resp


def array_response(description: str, schema_ref: str) -> dict[str, Any]:
    """Create an array response definition."""
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"$ref": f"#/components/schemas/{schema_ref}"},
                        },
                        "total": {"type": "integer"},
                    },
                }
            }
        },
    }


def error_response(status: str, description: str) -> dict[str, Any]:
    """Create an error response definition."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


# Standard error responses used across endpoints
STANDARD_ERRORS = {
    "400": error_response("400", "Bad request"),
    "401": error_response("401", "Unauthorized"),
    "404": error_response("404", "Not found"),
    "402": error_response("402", "Quota exceeded"),
    "429": error_response("429", "Rate limited"),
    "500": error_response("500", "Server error"),
}


__all__ = [
    "ok_response",
    "array_response",
    "error_response",
    "STANDARD_ERRORS",
]
