"""
OpenAPI Helper Functions.

Response builders, standard error definitions, and rate limit documentation.
"""

from typing import Any, Dict, List, Union

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
                "error": "Invalid value for 'rounds': must be between 1 and 10",
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
                "support_url": "https://github.com/anthropics/aragora/issues",
            },
        },
    },
}


# =============================================================================
# Response Builders
# =============================================================================


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
    """Build an error response definition with examples."""
    examples = ERROR_EXAMPLES.get(status, {})
    response: dict[str, Any] = {
        "description": description,
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
    operation: dict,
    tier: str = "free",
    custom_limit: int | None = None,
    window: str = "minute",
) -> dict:
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
    "402": _error_response("402", "Payment required - Quota exceeded, upgrade required"),
    "429": _error_response("429", "Too many requests - Rate limit exceeded"),
    "500": _error_response("500", "Internal server error - Unexpected error occurred"),
}

# =============================================================================
# Authentication Documentation
# =============================================================================

AUTH_REQUIREMENTS: Dict[str, Dict[str, Union[str, List[Dict[str, List[str]]]]]] = {
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
    "AUTH_REQUIREMENTS",
]
