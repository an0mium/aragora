"""
OpenAPI Helper Functions.

Response builders, standard error definitions, and rate limit documentation.
"""

from typing import Any

# =============================================================================
# Rate Limit Tiers
# =============================================================================

RATE_LIMIT_TIERS = {
    "free": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "debates_per_day": 10,
        "concurrent_debates": 1,
    },
    "pro": {
        "requests_per_minute": 300,
        "requests_per_hour": 10000,
        "debates_per_day": 100,
        "concurrent_debates": 5,
    },
    "enterprise": {
        "requests_per_minute": 1000,
        "requests_per_hour": 50000,
        "debates_per_day": -1,  # Unlimited
        "concurrent_debates": 20,
    },
}

# Rate limit headers returned in responses
RATE_LIMIT_HEADERS = {
    "X-RateLimit-Limit": "Maximum requests allowed in window",
    "X-RateLimit-Remaining": "Requests remaining in current window",
    "X-RateLimit-Reset": "Unix timestamp when the rate limit resets",
    "X-RateLimit-RetryAfter": "Seconds to wait before retrying (only on 429)",
}

# Standard headers included in all API responses (OpenAPI schema format)
STANDARD_RESPONSE_HEADERS: dict[str, dict[str, Any]] = {
    "X-Request-ID": {
        "description": "Unique request identifier for tracing and debugging",
        "schema": {"type": "string", "format": "uuid"},
    },
    "X-Response-Time": {
        "description": "Server processing time in milliseconds",
        "schema": {"type": "integer"},
    },
}

# Rate limit headers in OpenAPI schema format
RATE_LIMIT_RESPONSE_HEADERS: dict[str, dict[str, Any]] = {
    **STANDARD_RESPONSE_HEADERS,
    "X-RateLimit-Limit": {
        "description": "Maximum requests allowed in current window",
        "schema": {"type": "integer"},
    },
    "X-RateLimit-Remaining": {
        "description": "Requests remaining in current window",
        "schema": {"type": "integer"},
    },
    "X-RateLimit-Reset": {
        "description": "Unix timestamp when rate limit resets",
        "schema": {"type": "integer"},
    },
}

# =============================================================================
# Error Examples
# =============================================================================

ERROR_EXAMPLES: dict[str, dict[str, Any]] = {
    "400": {
        "invalid_json": {
            "summary": "Invalid JSON body",
            "value": {
                "error": "Invalid JSON in request body",
                "code": "INVALID_JSON",
                "trace_id": "req_abc123xyz",
            },
        },
        "missing_field": {
            "summary": "Missing required field",
            "value": {
                "error": "Missing required field: task",
                "code": "MISSING_FIELD",
                "field": "task",
                "trace_id": "req_abc123xyz",
            },
        },
        "invalid_value": {
            "summary": "Invalid field value",
            "value": {
                "error": "Invalid value for 'rounds': must be between 1 and 12",
                "code": "INVALID_VALUE",
                "field": "rounds",
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "401": {
        "missing_token": {
            "summary": "No authentication token",
            "value": {
                "error": "Authentication required",
                "code": "AUTH_REQUIRED",
                "trace_id": "req_abc123xyz",
            },
        },
        "invalid_token": {
            "summary": "Invalid or expired token",
            "value": {
                "error": "Invalid or expired authentication token",
                "code": "INVALID_TOKEN",
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "403": {
        "insufficient_permissions": {
            "summary": "Insufficient permissions",
            "value": {
                "error": "You do not have permission to access this resource",
                "code": "FORBIDDEN",
                "required_role": "admin",
                "trace_id": "req_abc123xyz",
            },
        },
        "resource_owner": {
            "summary": "Not resource owner",
            "value": {
                "error": "You do not have permission to modify this debate",
                "code": "NOT_OWNER",
                "resource_type": "debate",
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "404": {
        "not_found": {
            "summary": "Resource not found",
            "value": {
                "error": "Debate not found",
                "code": "NOT_FOUND",
                "resource_type": "debate",
                "resource_id": "deb_abc123",
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "402": {
        "quota_exceeded": {
            "summary": "Quota exceeded",
            "value": {
                "error": "Daily debate limit exceeded",
                "code": "QUOTA_EXCEEDED",
                "limit": 10,
                "used": 10,
                "resets_at": "2024-01-16T00:00:00Z",
                "upgrade_url": "https://aragora.ai/pricing",
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "429": {
        "rate_limited": {
            "summary": "Rate limit exceeded",
            "value": {
                "error": "Rate limit exceeded",
                "code": "RATE_LIMITED",
                "limit": 60,
                "window": "1 minute",
                "retry_after": 45,
                "trace_id": "req_abc123xyz",
            },
        },
    },
    "500": {
        "internal_error": {
            "summary": "Internal server error",
            "value": {
                "error": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
                "trace_id": "req_abc123xyz",
                "support_url": "https://github.com/an0mium/aragora/issues",
            },
        },
    },
}

# =============================================================================
# Response Builders
# =============================================================================


_VALID_JSON_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _is_complete_schema(schema: dict[str, Any]) -> bool:
    """Check if a dict is a complete JSON Schema (not a properties map).

    A properties map looks like ``{"name": {"type": "string"}, ...}`` where the
    keys are property names and values are sub-schemas.  A complete schema has a
    ``"type"`` key whose value is a valid JSON Schema type string (or list of
    type strings for nullable types in OpenAPI 3.1), or uses composition
    keywords like ``$ref``, ``oneOf``, ``anyOf``, ``allOf``.
    """
    # Composition keywords indicate a complete schema
    if any(k in schema for k in ("$ref", "oneOf", "anyOf", "allOf")):
        return True
    type_val = schema.get("type")
    if type_val is None:
        return False
    # OpenAPI 3.1 allows type to be a list, e.g. ["string", "null"]
    if isinstance(type_val, list):
        return all(isinstance(t, str) and t in _VALID_JSON_SCHEMA_TYPES for t in type_val)
    return isinstance(type_val, str) and type_val in _VALID_JSON_SCHEMA_TYPES


def _ok_response(
    description: str,
    schema: str | dict[str, Any] | None = None,
    *,
    include_rate_limit_headers: bool = False,
) -> dict[str, Any]:
    """Build a successful response definition.

    Args:
        description: Response description
        schema: Either a schema reference string (e.g., "AgentResponse")
                or an inline schema dict (e.g., {"type": "object", ...})
        include_rate_limit_headers: Whether to include rate limit headers
    """
    resp: dict[str, Any] = {
        "description": description,
        "headers": (
            RATE_LIMIT_RESPONSE_HEADERS if include_rate_limit_headers else STANDARD_RESPONSE_HEADERS
        ),
    }
    if schema:
        if isinstance(schema, str):
            # Schema reference
            resp["content"] = {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{schema}"}}
            }
        elif _is_complete_schema(schema):
            # Complete schema dict (has "type" key with valid JSON Schema type value,
            # or uses $ref / oneOf / anyOf / allOf)
            resp["content"] = {"application/json": {"schema": schema}}
        else:
            # Properties map (e.g. {"success": {"type": "boolean"}})
            resp["content"] = {
                "application/json": {"schema": {"type": "object", "properties": schema}}
            }
    return resp


def _array_response(
    description: str,
    schema: str | dict[str, Any],
    *,
    include_rate_limit_headers: bool = False,
) -> dict[str, Any]:
    """Build an array response definition.

    Args:
        description: Response description
        schema: Either a schema reference string (e.g., "Device")
                or an inline schema dict (e.g., {"device_id": {"type": "string"}, ...})
        include_rate_limit_headers: Whether to include rate limit headers
    """
    items_schema: dict[str, Any]
    if isinstance(schema, str):
        # Schema reference
        items_schema = {"$ref": f"#/components/schemas/{schema}"}
    else:
        # Inline schema dict - wrap in object type with properties
        items_schema = {"type": "object", "properties": schema}

    return {
        "description": description,
        "headers": (
            RATE_LIMIT_RESPONSE_HEADERS if include_rate_limit_headers else STANDARD_RESPONSE_HEADERS
        ),
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": items_schema,
                        },
                        "total": {"type": "integer"},
                    },
                }
            }
        },
    }


def _error_response(status: str, description: str) -> dict[str, Any]:
    """Build an error response definition with examples."""
    examples = ERROR_EXAMPLES.get(status, {})
    response: dict[str, Any] = {
        "description": description,
        "headers": STANDARD_RESPONSE_HEADERS,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/Error"},
            }
        },
    }
    if examples:
        response["content"]["application/json"]["examples"] = examples
    return response


def _rate_limited_endpoint(
    operation: dict[str, Any],
    tier: str = "free",
    custom_limit: int | None = None,
    window: str = "minute",
) -> dict[str, Any]:
    """Add rate limit documentation to an endpoint operation.

    Args:
        operation: The endpoint operation dict to enhance
        tier: The rate limit tier (free, pro, enterprise)
        custom_limit: Override the tier's default limit
        window: Rate limit window (minute, hour, day)

    Returns:
        Enhanced operation dict with rate limit documentation
    """
    limits = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["free"])
    limit_key = f"requests_per_{window}"
    limit = custom_limit or limits.get(limit_key, 60)

    # Add rate limit info to description
    rate_info = f"\n\n**Rate Limit:** {limit} requests per {window} ({tier} tier)"
    if "description" in operation:
        operation["description"] += rate_info
    else:
        operation["description"] = rate_info.strip()

    # Add rate limit headers to responses
    for status_code, response in operation.get("responses", {}).items():
        if status_code.startswith("2"):
            if "headers" not in response:
                response["headers"] = {}
            response["headers"].update(
                {
                    "X-RateLimit-Limit": {
                        "description": RATE_LIMIT_HEADERS["X-RateLimit-Limit"],
                        "schema": {"type": "integer"},
                    },
                    "X-RateLimit-Remaining": {
                        "description": RATE_LIMIT_HEADERS["X-RateLimit-Remaining"],
                        "schema": {"type": "integer"},
                    },
                    "X-RateLimit-Reset": {
                        "description": RATE_LIMIT_HEADERS["X-RateLimit-Reset"],
                        "schema": {"type": "integer"},
                    },
                }
            )

    return operation


# =============================================================================
# Standard Errors
# =============================================================================

STANDARD_ERRORS = {
    "400": _error_response("400", "Bad request - Invalid input or malformed JSON"),
    "401": _error_response("401", "Unauthorized - Authentication required or token invalid"),
    "403": _error_response("403", "Forbidden - Insufficient permissions for this operation"),
    "404": _error_response("404", "Not found - The requested resource does not exist"),
    "409": _error_response("409", "Conflict - The request could not be completed"),
    "402": _error_response("402", "Payment required - Quota exceeded, upgrade required"),
    "429": _error_response("429", "Too many requests - Rate limit exceeded"),
    "500": _error_response("500", "Internal server error - Unexpected error occurred"),
}

# =============================================================================
# Authentication Documentation
# =============================================================================

AUTH_REQUIREMENTS: dict[str, dict[str, str | list[dict[str, list[str]]]]] = {
    "none": {
        "description": "No authentication required",
        "security": [],
    },
    "optional": {
        "description": "Authentication optional - provides additional features when authenticated",
        "security": [{}],
    },
    "required": {
        "description": "Authentication required via Bearer token",
        "security": [{"bearerAuth": []}],
    },
    "admin": {
        "description": "Admin role required",
        "security": [{"bearerAuth": ["admin"]}],
    },
}

__all__ = [
    "_ok_response",
    "_array_response",
    "_error_response",
    "_rate_limited_endpoint",
    "STANDARD_ERRORS",
    "ERROR_EXAMPLES",
    "RATE_LIMIT_TIERS",
    "RATE_LIMIT_HEADERS",
    "STANDARD_RESPONSE_HEADERS",
    "RATE_LIMIT_RESPONSE_HEADERS",
    "AUTH_REQUIREMENTS",
]
