"""Response schema definitions for endpoints missing them.

This module adds proper response schemas to endpoints whose schemas were
either missing or lost during the OpenAPI generation pipeline (e.g. due to
_filter_unhandled_paths stripping unversioned /api/ paths before
_autogenerate_missing_paths re-adds them without schemas).

These definitions use /api/v1/ versioned paths to survive the pipeline.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
_DASHBOARD_ENDPOINTS = {
    "/api/v1/dashboard/overview": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Dashboard overview",
            "operationId": "getDashboardOverview",
            "description": "Get dashboard overview with key metrics.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Dashboard overview data",
                    {
                        "total_debates": {"type": "integer", "description": "Total debates"},
                        "active_debates": {"type": "integer", "description": "Active debates"},
                        "completed_debates": {
                            "type": "integer",
                            "description": "Completed debates",
                        },
                        "total_agents": {"type": "integer", "description": "Total agents"},
                        "recent_activity": {"type": "array", "items": {"type": "object"}},
                        "health_status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "unhealthy"],
                        },
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/stats": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Dashboard stats",
            "operationId": "getDashboardStats",
            "description": "Get dashboard statistics.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Dashboard statistics",
                    {
                        "debates_today": {"type": "integer"},
                        "debates_this_week": {"type": "integer"},
                        "average_duration_ms": {"type": "number"},
                        "consensus_rate": {"type": "number"},
                        "agent_utilization": {"type": "number"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/activity": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Recent activity",
            "operationId": "getDashboardActivity",
            "description": "Get recent activity feed.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Activity feed",
                    {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"},
                                    "timestamp": {"type": "string", "format": "date-time"},
                                    "actor": {"type": "string"},
                                },
                            },
                        },
                        "total": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/debates/{debate_id}": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Dashboard debate detail",
            "operationId": "getDashboardDebate",
            "description": "Get debate details for dashboard view.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Debate details",
                    {
                        "id": {"type": "string"},
                        "task": {"type": "string"},
                        "status": {"type": "string"},
                        "rounds": {"type": "integer"},
                        "agents": {"type": "array", "items": {"type": "string"}},
                        "created_at": {"type": "string", "format": "date-time"},
                        "consensus": {"type": "object"},
                        "summary": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/inbox-summary": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Inbox summary",
            "operationId": "getDashboardInboxSummary",
            "description": "Get inbox summary for dashboard.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Inbox summary",
                    {
                        "unread_count": {"type": "integer"},
                        "priority_count": {"type": "integer"},
                        "recent_items": {"type": "array", "items": {"type": "object"}},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/labels": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Dashboard labels",
            "operationId": "getDashboardLabels",
            "description": "Get labels used in dashboard.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Labels list",
                    {
                        "labels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "color": {"type": "string"},
                                    "count": {"type": "integer"},
                                },
                            },
                        }
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/pending-actions": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Pending actions",
            "operationId": "getDashboardPendingActions",
            "description": "Get pending actions for the current user.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Pending actions",
                    {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"},
                                    "priority": {"type": "string"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                        "total": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/pending-actions/{action_id}/complete": {
        "post": {
            "tags": ["Dashboard"],
            "summary": "Complete pending action",
            "operationId": "completeDashboardPendingAction",
            "description": "Mark a pending action as complete.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "action_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Action completed",
                    {
                        "action_id": {"type": "string"},
                        "completed": {"type": "boolean"},
                        "completed_at": {"type": "string", "format": "date-time"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/quality-metrics": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Quality metrics",
            "operationId": "getDashboardQualityMetrics",
            "description": "Get debate quality metrics.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Quality metrics",
                    {
                        "consensus_rate": {"type": "number"},
                        "average_rounds": {"type": "number"},
                        "agent_accuracy": {"type": "number"},
                        "debate_quality_score": {"type": "number"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/quick-actions": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Quick actions",
            "operationId": "getDashboardQuickActions",
            "description": "Get available quick actions.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Quick actions",
                    {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "icon": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                        }
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/quick-actions/{action_id}": {
        "post": {
            "tags": ["Dashboard"],
            "summary": "Execute quick action",
            "operationId": "executeDashboardQuickAction",
            "description": "Execute a quick action.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "action_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Action executed",
                    {
                        "action_id": {"type": "string"},
                        "result": {"type": "object"},
                        "success": {"type": "boolean"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/search": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Dashboard search",
            "operationId": "searchDashboard",
            "description": "Search across dashboard resources.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "q",
                    "in": "query",
                    "description": "Search query",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Search results",
                    {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "title": {"type": "string"},
                                    "snippet": {"type": "string"},
                                    "score": {"type": "number"},
                                },
                            },
                        },
                        "total": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/stat-cards": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Stat cards",
            "operationId": "getDashboardStatCards",
            "description": "Get stat card data for dashboard.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Stat cards",
                    {
                        "cards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "value": {"type": "number"},
                                    "change": {"type": "number"},
                                    "trend": {
                                        "type": "string",
                                        "enum": ["up", "down", "stable"],
                                    },
                                },
                            },
                        }
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/team-performance": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Team performance",
            "operationId": "getDashboardTeamPerformance",
            "description": "Get team performance overview.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Team performance",
                    {
                        "teams": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "team_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "debates_completed": {"type": "integer"},
                                    "success_rate": {"type": "number"},
                                },
                            },
                        }
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/team-performance/{team_id}": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Team performance detail",
            "operationId": "getDashboardTeamPerformanceDetail",
            "description": "Get detailed team performance.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "team_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Team performance detail",
                    {
                        "team_id": {"type": "string"},
                        "name": {"type": "string"},
                        "members": {"type": "array", "items": {"type": "object"}},
                        "stats": {"type": "object"},
                        "recent_debates": {"type": "array", "items": {"type": "object"}},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/top-senders": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Top senders",
            "operationId": "getDashboardTopSenders",
            "description": "Get top email senders.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Top senders",
                    {
                        "senders": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string"},
                                    "name": {"type": "string"},
                                    "count": {"type": "integer"},
                                    "priority_score": {"type": "number"},
                                },
                            },
                        }
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/urgent": {
        "get": {
            "tags": ["Dashboard"],
            "summary": "Urgent items",
            "operationId": "getDashboardUrgent",
            "description": "Get urgent items requiring attention.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Urgent items",
                    {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "title": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                        "total": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/dashboard/urgent/{item_id}/dismiss": {
        "post": {
            "tags": ["Dashboard"],
            "summary": "Dismiss urgent item",
            "operationId": "dismissDashboardUrgentItem",
            "description": "Dismiss an urgent item.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "item_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Item dismissed",
                    {
                        "item_id": {"type": "string"},
                        "dismissed": {"type": "boolean"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}

# ---------------------------------------------------------------------------
# Receipts (v2)
# ---------------------------------------------------------------------------
_RECEIPT_ENDPOINTS = {
    "/api/v2/receipts": {
        "get": {
            "tags": ["Receipts"],
            "summary": "List receipts",
            "operationId": "listReceipts",
            "description": "List decision receipts.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Receipt list",
                    {
                        "receipts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "debate_id": {"type": "string"},
                                    "hash": {"type": "string"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                    "status": {"type": "string"},
                                },
                            },
                        },
                        "total": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/search": {
        "get": {
            "tags": ["Receipts", "Search"],
            "summary": "Search receipts",
            "operationId": "searchReceipts",
            "description": "Search decision receipts.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Search results",
                    {
                        "results": {"type": "array", "items": {"type": "object"}},
                        "total": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/stats": {
        "get": {
            "tags": ["Receipts", "Statistics"],
            "summary": "Receipt statistics",
            "operationId": "getReceiptStats",
            "description": "Get receipt statistics.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response(
                    "Receipt stats",
                    {
                        "total_receipts": {"type": "integer"},
                        "verified_count": {"type": "integer"},
                        "by_status": {"type": "object"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/{receipt_id}": {
        "get": {
            "tags": ["Receipts"],
            "summary": "Get receipt",
            "operationId": "getReceiptById",
            "description": "Get a specific receipt by ID.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Receipt details",
                    {
                        "id": {"type": "string"},
                        "debate_id": {"type": "string"},
                        "hash": {"type": "string"},
                        "signature": {"type": "string"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "metadata": {"type": "object"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/{receipt_id}/export": {
        "get": {
            "tags": ["Receipts", "Export"],
            "summary": "Export receipt",
            "operationId": "exportReceipt",
            "description": "Export a receipt in the requested format.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Exported receipt",
                    {
                        "receipt_id": {"type": "string"},
                        "format": {"type": "string"},
                        "data": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/{receipt_id}/share": {
        "post": {
            "tags": ["Receipts", "Sharing"],
            "summary": "Share receipt",
            "operationId": "shareReceipt",
            "description": "Share a receipt publicly.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Receipt shared",
                    {
                        "receipt_id": {"type": "string"},
                        "shared": {"type": "boolean"},
                        "share_url": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/receipts/{receipt_id}/verify": {
        "get": {
            "tags": ["Receipts", "Verification"],
            "summary": "Verify receipt",
            "operationId": "verifyReceiptById",
            "description": "Verify receipt integrity.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Verification result",
                    {
                        "receipt_id": {"type": "string"},
                        "valid": {"type": "boolean"},
                        "hash_match": {"type": "boolean"},
                        "verified_at": {"type": "string", "format": "date-time"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Receipts", "Verification"],
            "summary": "Verify receipt integrity",
            "operationId": "verifyReceiptIntegrity",
            "description": "Verify receipt integrity with payload.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Integrity verification result",
                    {
                        "receipt_id": {"type": "string"},
                        "valid": {"type": "boolean"},
                        "integrity_verified": {"type": "boolean"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/receipts/{receipt_id}/verify-signature": {
        "post": {
            "tags": ["Receipts", "Verification"],
            "summary": "Verify receipt signature",
            "operationId": "verifyReceiptSignature",
            "description": "Verify the cryptographic signature of a receipt.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "receipt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Signature verification result",
                    {
                        "receipt_id": {"type": "string"},
                        "valid": {"type": "boolean"},
                        "signer": {"type": "string"},
                        "algorithm": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}

# ---------------------------------------------------------------------------
# System / OpenAPI spec
# ---------------------------------------------------------------------------
_SYSTEM_SCHEMA_ENDPOINTS = {
    "/api/v1/openapi": {
        "get": {
            "tags": ["System"],
            "summary": "Get OpenAPI specification",
            "operationId": "getOpenAPISpec",
            "description": "Returns the full OpenAPI 3.1 specification for all Aragora API endpoints.",
            "responses": {
                "200": {
                    "description": "OpenAPI 3.1 specification",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "openapi": {
                                        "type": "string",
                                        "description": "OpenAPI version",
                                    },
                                    "info": {
                                        "type": "object",
                                        "description": "API metadata",
                                    },
                                    "paths": {
                                        "type": "object",
                                        "description": "API path definitions",
                                    },
                                    "components": {
                                        "type": "object",
                                        "description": "Reusable components",
                                    },
                                },
                            }
                        }
                    },
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Monitoring / Prometheus metrics
# ---------------------------------------------------------------------------
_MONITORING_SCHEMA_ENDPOINTS = {
    "/metrics": {
        "get": {
            "tags": ["Monitoring"],
            "summary": "Prometheus metrics",
            "operationId": "getPrometheusMetrics",
            "description": "Prometheus exposition format metrics for monitoring.",
            "responses": {
                "200": {
                    "description": "Prometheus metrics in OpenMetrics format",
                    "content": {
                        "text/plain": {
                            "schema": {
                                "type": "string",
                                "description": "Prometheus exposition format metrics",
                            }
                        }
                    },
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Podcast XML feed
# ---------------------------------------------------------------------------
_MEDIA_SCHEMA_ENDPOINTS = {
    "/api/v1/podcast/feed.xml": {
        "get": {
            "tags": ["Media"],
            "summary": "Podcast RSS feed",
            "operationId": "getPodcastFeed",
            "description": "Returns podcast episodes as an RSS/Atom XML feed.",
            "responses": {
                "200": {
                    "description": "RSS/Atom podcast feed in XML format",
                    "content": {
                        "application/xml": {
                            "schema": {
                                "type": "string",
                                "description": "XML-formatted RSS/Atom podcast feed",
                            }
                        }
                    },
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Combined export
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA_ENDPOINTS = {
    **_DASHBOARD_ENDPOINTS,
    **_RECEIPT_ENDPOINTS,
    **_SYSTEM_SCHEMA_ENDPOINTS,
    **_MONITORING_SCHEMA_ENDPOINTS,
    **_MEDIA_SCHEMA_ENDPOINTS,
}

__all__ = ["RESPONSE_SCHEMA_ENDPOINTS"]
