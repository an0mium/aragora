"""
OpenAPI Helper Functions.

Response builders and standard error definitions.
"""


def _ok_response(description: str, schema_ref: str | None = None) -> dict:
    """Build a successful response definition."""
    resp: dict = {"description": description}
    if schema_ref:
        resp["content"] = {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_ref}"}}
        }
    return resp


def _array_response(description: str, schema_ref: str) -> dict:
    """Build an array response definition."""
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


def _error_response(status: str, description: str) -> dict:
    """Build an error response definition."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


STANDARD_ERRORS = {
    "400": _error_response("400", "Bad request"),
    "401": _error_response("401", "Unauthorized"),
    "404": _error_response("404", "Not found"),
    "402": _error_response("402", "Quota exceeded"),
    "429": _error_response("429", "Rate limited"),
    "500": _error_response("500", "Server error"),
}

__all__ = ["_ok_response", "_array_response", "_error_response", "STANDARD_ERRORS"]
