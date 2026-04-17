"""Admin endpoint definitions for OpenAPI documentation.

Administrative endpoints for managing organizations, users, system metrics,
and the Nomic self-improvement loop.
"""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


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
            "organization_id": {"type": ["string", "null"]},
            "last_login": {"type": ["string", "null"], "format": "date-time"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    }


def _mfa_compliance_response_schema() -> dict[str, Any]:
    """Admin MFA compliance report schema."""
    return {
        "type": "object",
        "properties": {
            "total_admins": {"type": "integer"},
            "mfa_enabled_count": {"type": "integer"},
            "mfa_disabled_count": {"type": "integer"},
            "in_grace_period": {"type": "integer"},
            "compliance_pct": {"type": "number"},
            "non_compliant_users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "role": {"type": "string"},
                        "in_grace_period": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _mfa_compliance_operation(*, operation_id: str) -> dict[str, Any]:
    """OpenAPI operation for the admin MFA compliance report."""
    return {
        "tags": ["Admin", "MFA", "Compliance"],
        "summary": "Get admin MFA compliance report",
        "operationId": operation_id,
        "description": "Returns a compliance report showing how many admin users have MFA enabled.",
        "responses": {
            "200": {
                "description": "MFA compliance report",
                "content": {
                    "application/json": {
                        "schema": _mfa_compliance_response_schema(),
                    }
                },
            },
            "401": {"description": "Authentication required"},
            "403": {"description": "Admin privileges required"},
            "501": {"description": "User store does not support listing users"},
            "503": {"description": "User service unavailable"},
        },
        "security": [{"bearerAuth": []}],
    }


def _system_health_response_schema() -> dict[str, Any]:
    """Admin system health envelope schema."""
    return {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "additionalProperties": True,
            }
        },
    }


def _system_health_operation(
    *, operation_id: str, summary: str, description: str
) -> dict[str, Any]:
    """OpenAPI operation for admin system-health routes."""
    return {
        "tags": ["Admin", "System"],
        "summary": summary,
        "operationId": operation_id,
        "description": description,
        "responses": {
            "200": {
                "description": "System health payload",
                "content": {
                    "application/json": {
                        "schema": _system_health_response_schema(),
                    }
                },
            },
            "401": {"description": "Authentication required"},
            "403": {"description": "Admin privileges required"},
        },
        "security": [{"bearerAuth": []}],
    }


def _feature_flag_value_schema() -> dict[str, Any]:
    """Flexible schema for feature-flag values/defaults."""
    return {
        "anyOf": [
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array"},
            {"type": "object", "additionalProperties": True},
            {"type": "null"},
        ]
    }


def _feature_flag_schema() -> dict[str, Any]:
    """Admin feature-flag payload schema."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": _feature_flag_value_schema(),
            "default": _feature_flag_value_schema(),
            "type": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "status": {"type": "string"},
            "env_var": {"type": ["string", "null"]},
            "deprecated_since": {"type": ["string", "null"]},
            "removed_in": {"type": ["string", "null"]},
            "replacement": {"type": ["string", "null"]},
            "usage": {
                "type": "object",
                "properties": {
                    "access_count": {"type": "integer"},
                    "last_accessed": {"type": ["string", "null"]},
                    "access_locations": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
            },
        },
    }


def _feature_flag_list_response_schema() -> dict[str, Any]:
    """List feature flags response schema."""
    return {
        "type": "object",
        "properties": {
            "flags": {"type": "array", "items": _feature_flag_schema()},
            "total": {"type": "integer"},
            "stats": {"type": "object", "additionalProperties": True},
        },
    }


def _feature_flag_name_parameter() -> dict[str, Any]:
    """Shared feature-flag name path parameter."""
    return {
        "name": "name",
        "in": "path",
        "required": True,
        "description": "Feature flag name",
        "schema": {"type": "string"},
    }


def _feature_flag_list_operation(*, operation_id: str) -> dict[str, Any]:
    """OpenAPI operation for listing admin feature flags."""
    return {
        "tags": ["Admin"],
        "summary": "List admin feature flags",
        "operationId": operation_id,
        "description": "Returns all admin-manageable feature flags with their current values.",
        "responses": {
            "200": {
                "description": "Feature flags",
                "content": {
                    "application/json": {
                        "schema": _feature_flag_list_response_schema(),
                    }
                },
            },
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "503": {"description": "Feature flag system not available"},
        },
        "security": [{"bearerAuth": []}],
    }


def _feature_flag_detail_operation(*, operation_id: str) -> dict[str, Any]:
    """OpenAPI operation for fetching one admin feature flag."""
    return {
        "tags": ["Admin"],
        "summary": "Get admin feature flag",
        "operationId": operation_id,
        "description": "Returns one admin-manageable feature flag and its usage metadata.",
        "parameters": [_feature_flag_name_parameter()],
        "responses": {
            "200": {
                "description": "Feature flag",
                "content": {
                    "application/json": {
                        "schema": _feature_flag_schema(),
                    }
                },
            },
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "404": STANDARD_ERRORS["404"],
            "503": {"description": "Feature flag system not available"},
        },
        "security": [{"bearerAuth": []}],
    }


def _feature_flag_update_operation(*, operation_id: str) -> dict[str, Any]:
    """OpenAPI operation for updating one admin feature flag."""
    return {
        "tags": ["Admin"],
        "summary": "Set admin feature flag",
        "operationId": operation_id,
        "description": "Sets one admin-manageable feature flag to a new value.",
        "parameters": [_feature_flag_name_parameter()],
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["value"],
                        "properties": {
                            "value": _feature_flag_value_schema(),
                        },
                    }
                }
            },
        },
        "responses": {
            "200": {
                "description": "Feature flag updated",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": _feature_flag_value_schema(),
                                "previous_default": _feature_flag_value_schema(),
                                "updated": {"type": "boolean"},
                            },
                        }
                    }
                },
            },
            "400": STANDARD_ERRORS["400"],
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "404": STANDARD_ERRORS["404"],
            "503": {"description": "Feature flag system not available"},
        },
        "security": [{"bearerAuth": []}],
    }


ADMIN_ENDPOINTS = {
    # =========================================================================
    # Admin MFA Compliance
    # =========================================================================
    "/api/v1/admin/mfa/compliance": {
        "get": _mfa_compliance_operation(operation_id="adminGetMfaCompliance")
    },
    "/api/admin/mfa/compliance": {
        "get": {
            **_mfa_compliance_operation(operation_id="adminGetMfaComplianceLegacy"),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    # =========================================================================
    # Admin System Health
    # =========================================================================
    "/api/v1/admin/system-health": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthOverview",
            summary="Get admin system health overview",
            description="Returns the aggregated admin system-health overview.",
        )
    },
    "/api/admin/system-health": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthOverviewLegacy",
                summary="Get admin system health overview",
                description="Returns the aggregated admin system-health overview.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/system-health/circuit-breakers": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthCircuitBreakers",
            summary="Get admin system health circuit breakers",
            description="Returns circuit-breaker health for admin system-health views.",
        )
    },
    "/api/admin/system-health/circuit-breakers": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthCircuitBreakersLegacy",
                summary="Get admin system health circuit breakers",
                description="Returns circuit-breaker health for admin system-health views.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/system-health/slos": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthSlos",
            summary="Get admin system health SLOs",
            description="Returns SLO compliance health for admin system-health views.",
        )
    },
    "/api/admin/system-health/slos": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthSlosLegacy",
                summary="Get admin system health SLOs",
                description="Returns SLO compliance health for admin system-health views.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/system-health/adapters": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthAdapters",
            summary="Get admin system health adapters",
            description="Returns adapter health for admin system-health views.",
        )
    },
    "/api/admin/system-health/adapters": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthAdaptersLegacy",
                summary="Get admin system health adapters",
                description="Returns adapter health for admin system-health views.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/system-health/agents": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthAgents",
            summary="Get admin system health agents",
            description="Returns agent-pool health for admin system-health views.",
        )
    },
    "/api/admin/system-health/agents": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthAgentsLegacy",
                summary="Get admin system health agents",
                description="Returns agent-pool health for admin system-health views.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/system-health/budget": {
        "get": _system_health_operation(
            operation_id="adminGetSystemHealthBudget",
            summary="Get admin system health budget",
            description="Returns budget health for admin system-health views.",
        )
    },
    "/api/admin/system-health/budget": {
        "get": {
            **_system_health_operation(
                operation_id="adminGetSystemHealthBudgetLegacy",
                summary="Get admin system health budget",
                description="Returns budget health for admin system-health views.",
            ),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    # =========================================================================
    # Admin Feature Flags
    # =========================================================================
    "/api/v1/admin/feature-flags": {
        "get": _feature_flag_list_operation(operation_id="adminListFeatureFlags")
    },
    "/api/admin/feature-flags": {
        "get": {
            **_feature_flag_list_operation(operation_id="adminListFeatureFlagsLegacy"),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        }
    },
    "/api/v1/admin/feature-flags/{name}": {
        "get": _feature_flag_detail_operation(operation_id="adminGetFeatureFlag"),
        "put": _feature_flag_update_operation(operation_id="adminSetFeatureFlag"),
    },
    "/api/admin/feature-flags/{name}": {
        "get": {
            **_feature_flag_detail_operation(operation_id="adminGetFeatureFlagLegacy"),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        },
        "put": {
            **_feature_flag_update_operation(operation_id="adminSetFeatureFlagLegacy"),
            "deprecated": True,
            "x-preserve-legacy-operation-id": True,
        },
    },
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
                                                    "type": ["string", "null"],
                                                    "format": "date-time",
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
