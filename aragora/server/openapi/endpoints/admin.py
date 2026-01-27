"""Admin endpoint definitions for OpenAPI documentation.

Administrative endpoints for managing organizations, users, system metrics,
and the Nomic self-improvement loop.
"""

from typing import Any


def _org_schema() -> dict[str, Any]:
    """Organization object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "plan": {"type": "string", "enum": ["free", "pro", "enterprise"]},
            "member_count": {"type": "integer"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    }


def _user_admin_schema() -> dict[str, Any]:
    """Admin user object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "suspended", "deleted"]},
            "organization_id": {"type": "string", "nullable": True},
            "last_login": {"type": "string", "format": "date-time", "nullable": True},
            "created_at": {"type": "string", "format": "date-time"},
        },
    }


ADMIN_ENDPOINTS = {
    # =========================================================================
    # Organization Management
    # =========================================================================
    "/api/v1/admin/organizations": {
        "get": {
            "tags": ["Admin"],
            "summary": "List organizations",
            "operationId": "adminListOrganizations",
            "description": "List all organizations. Requires admin privileges.",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
                {
                    "name": "plan",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of organizations",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "organizations": {
                                        "type": "array",
                                        "items": _org_schema(),
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # User Management
    # =========================================================================
    "/api/v1/admin/users": {
        "get": {
            "tags": ["Admin"],
            "summary": "List users",
            "operationId": "adminListUsers",
            "description": "List all users across all organizations. Requires admin privileges.",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["active", "suspended", "deleted"]},
                },
                {
                    "name": "organization_id",
                    "in": "query",
                    "schema": {"type": "string"},
                    "description": "Filter by organization",
                },
            ],
            "responses": {
                "200": {
                    "description": "List of users",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "users": {
                                        "type": "array",
                                        "items": _user_admin_schema(),
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # System Statistics
    # =========================================================================
    "/api/v1/admin/stats": {
        "get": {
            "tags": ["Admin"],
            "summary": "Get system statistics",
            "operationId": "adminGetStats",
            "description": "Returns aggregate system statistics including user counts, debate metrics, and usage trends.",
            "responses": {
                "200": {
                    "description": "System statistics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "total_users": {"type": "integer"},
                                    "active_users_24h": {"type": "integer"},
                                    "total_debates": {"type": "integer"},
                                    "debates_24h": {"type": "integer"},
                                    "total_organizations": {"type": "integer"},
                                    "api_calls_24h": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/system/metrics": {
        "get": {
            "tags": ["Admin"],
            "summary": "Get system metrics",
            "operationId": "adminGetSystemMetrics",
            "description": "Returns detailed system metrics including memory, CPU, and database stats.",
            "responses": {
                "200": {
                    "description": "System metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "memory_mb": {"type": "number"},
                                    "cpu_percent": {"type": "number"},
                                    "db_connections": {"type": "integer"},
                                    "cache_hit_rate": {"type": "number"},
                                    "queue_depth": {"type": "integer"},
                                    "uptime_seconds": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/revenue": {
        "get": {
            "tags": ["Admin"],
            "summary": "Get revenue metrics",
            "operationId": "adminGetRevenue",
            "description": "Returns revenue and billing metrics. Requires superadmin privileges.",
            "parameters": [
                {
                    "name": "period",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["day", "week", "month", "year"]},
                    "description": "Reporting period",
                },
            ],
            "responses": {
                "200": {
                    "description": "Revenue metrics",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "mrr": {
                                        "type": "number",
                                        "description": "Monthly recurring revenue",
                                    },
                                    "arr": {
                                        "type": "number",
                                        "description": "Annual recurring revenue",
                                    },
                                    "subscribers_by_plan": {
                                        "type": "object",
                                        "additionalProperties": {"type": "integer"},
                                    },
                                    "churn_rate": {"type": "number"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Superadmin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # Impersonation
    # =========================================================================
    "/api/v1/admin/impersonate/{user_id}": {
        "post": {
            "tags": ["Admin"],
            "summary": "Impersonate user",
            "operationId": "adminImpersonateUser",
            "description": "Start an impersonation session for a user. All actions are logged.",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Impersonation session started",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "impersonation_token": {"type": "string"},
                                    "target_user": _user_admin_schema(),
                                    "expires_at": {"type": "string", "format": "date-time"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
                "404": {"description": "User not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # Nomic Loop Control
    # =========================================================================
    "/api/v1/admin/nomic/status": {
        "get": {
            "tags": ["Admin", "Nomic"],
            "summary": "Get Nomic loop status",
            "operationId": "adminGetNomicStatus",
            "description": "Returns the current status of the Nomic self-improvement loop.",
            "responses": {
                "200": {
                    "description": "Nomic loop status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "running": {"type": "boolean"},
                                    "paused": {"type": "boolean"},
                                    "current_phase": {"type": "string"},
                                    "cycle_count": {"type": "integer"},
                                    "last_activity": {"type": "string", "format": "date-time"},
                                    "circuit_breakers": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "object",
                                            "properties": {
                                                "state": {"type": "string"},
                                                "failure_count": {"type": "integer"},
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/nomic/circuit-breakers": {
        "get": {
            "tags": ["Admin", "Nomic"],
            "summary": "Get circuit breaker status",
            "operationId": "adminGetCircuitBreakers",
            "description": "Returns the status of all circuit breakers in the system.",
            "responses": {
                "200": {
                    "description": "Circuit breaker status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "breakers": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "state": {
                                                    "type": "string",
                                                    "enum": ["closed", "open", "half_open"],
                                                },
                                                "failure_count": {"type": "integer"},
                                                "last_failure": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "nullable": True,
                                                },
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/nomic/reset": {
        "post": {
            "tags": ["Admin", "Nomic"],
            "summary": "Reset Nomic loop",
            "operationId": "adminResetNomic",
            "description": "Reset the Nomic loop to initial state. Use with caution.",
            "responses": {
                "200": {
                    "description": "Nomic loop reset",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Superadmin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/nomic/pause": {
        "post": {
            "tags": ["Admin", "Nomic"],
            "summary": "Pause Nomic loop",
            "operationId": "adminPauseNomic",
            "description": "Pause the Nomic loop at the current phase.",
            "responses": {
                "200": {
                    "description": "Nomic loop paused",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "paused_at_phase": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
                "409": {"description": "Loop already paused"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/nomic/resume": {
        "post": {
            "tags": ["Admin", "Nomic"],
            "summary": "Resume Nomic loop",
            "operationId": "adminResumeNomic",
            "description": "Resume the Nomic loop from paused state.",
            "responses": {
                "200": {
                    "description": "Nomic loop resumed",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "resumed_at_phase": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
                "409": {"description": "Loop not paused"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/v1/admin/nomic/circuit-breakers/reset": {
        "post": {
            "tags": ["Admin", "Nomic"],
            "summary": "Reset circuit breakers",
            "operationId": "adminResetCircuitBreakers",
            "description": "Reset all circuit breakers to closed state.",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "breaker_name": {
                                    "type": "string",
                                    "description": "Specific breaker to reset (optional)",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Circuit breakers reset",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "reset_count": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Authentication required"},
                "403": {"description": "Admin privileges required"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
