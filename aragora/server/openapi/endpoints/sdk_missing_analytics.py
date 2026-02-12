"""SDK missing endpoints: Planned analytics query/reporting endpoints (partially implemented).

This module contains endpoint definitions for analytics features that are
planned or partially implemented: agent performance metrics, analytics provider
connections, custom analytics queries, report generation, and analytics data
management.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

# =============================================================================
# Response Schemas
# =============================================================================

_PERFORMANCE_METRICS_SCHEMA = {
    "agent_id": {"type": "string", "description": "Unique agent identifier"},
    "response_time_ms": {
        "type": "number",
        "description": "Average response time in milliseconds",
    },
    "success_rate": {
        "type": "number",
        "description": "Success rate as decimal (0-1)",
    },
    "total_debates": {"type": "integer", "description": "Total debates participated in"},
    "wins": {"type": "integer", "description": "Number of debates won"},
    "elo_rating": {"type": "number", "description": "Current ELO rating"},
    "period": {
        "type": "string",
        "enum": ["day", "week", "month", "all_time"],
        "description": "Time period for metrics",
    },
    "timestamp": {"type": "string", "format": "date-time"},
}

_ANALYTICS_CONNECTION_SCHEMA = {
    "connection_id": {"type": "string", "description": "Connection identifier"},
    "provider": {
        "type": "string",
        "enum": ["prometheus", "grafana", "datadog", "newrelic", "custom"],
    },
    "status": {"type": "string", "enum": ["connected", "disconnected", "error"]},
    "connected_at": {"type": "string", "format": "date-time"},
}

_QUERY_RESULT_SCHEMA = {
    "query_id": {"type": "string", "description": "Unique query identifier"},
    "data": {
        "type": "array",
        "items": {"type": "object"},
        "description": "Query result rows",
    },
    "columns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Column names in result",
    },
    "row_count": {"type": "integer", "description": "Number of rows returned"},
    "execution_time_ms": {"type": "number", "description": "Query execution time"},
}

_REPORT_SCHEMA = {
    "report_id": {"type": "string", "description": "Generated report identifier"},
    "report_type": {
        "type": "string",
        "enum": ["performance", "usage", "cost", "agent_comparison", "debate_summary"],
    },
    "format": {"type": "string", "enum": ["json", "csv", "pdf", "html"]},
    "download_url": {"type": "string", "format": "uri", "description": "URL to download report"},
    "expires_at": {"type": "string", "format": "date-time"},
    "generated_at": {"type": "string", "format": "date-time"},
}

_DELETE_RESULT_SCHEMA = {
    "deleted": {"type": "boolean", "description": "Whether deletion was successful"},
    "resource_id": {"type": "string", "description": "ID of deleted resource"},
    "deleted_at": {"type": "string", "format": "date-time"},
}

# =============================================================================
# Endpoints
# =============================================================================

SDK_MISSING_ANALYTICS_ENDPOINTS: dict = {
    "/api/analytics/agents/{id}/performance": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Get agent performance metrics",
            "description": "Retrieve performance metrics for a specific agent including response times, success rates, and ELO ratings.",
            "operationId": "getAgentsPerformance",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Agent identifier",
                },
                {
                    "name": "period",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string", "enum": ["day", "week", "month", "all_time"]},
                    "description": "Time period for metrics",
                },
            ],
            "responses": {
                "200": _ok_response("Agent performance metrics", _PERFORMANCE_METRICS_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/analytics/connect": {
        "post": {
            "tags": ["Analytics"],
            "summary": "Connect analytics provider",
            "description": "Connect an external analytics provider (Prometheus, Grafana, DataDog, etc.) for metrics export.",
            "operationId": "postAnalyticsConnect",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["provider", "endpoint"],
                            "properties": {
                                "provider": {
                                    "type": "string",
                                    "enum": [
                                        "prometheus",
                                        "grafana",
                                        "datadog",
                                        "newrelic",
                                        "custom",
                                    ],
                                },
                                "endpoint": {"type": "string", "format": "uri"},
                                "api_key": {"type": "string"},
                                "options": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Analytics connection established", _ANALYTICS_CONNECTION_SCHEMA
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/analytics/query": {
        "post": {
            "tags": ["Analytics"],
            "summary": "Execute analytics query",
            "description": "Execute a custom analytics query against collected metrics and debate data.",
            "operationId": "postAnalyticsQuery",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {
                                "query": {"type": "string", "description": "Query string"},
                                "start_time": {"type": "string", "format": "date-time"},
                                "end_time": {"type": "string", "format": "date-time"},
                                "limit": {"type": "integer", "default": 100, "maximum": 10000},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Query results", _QUERY_RESULT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/analytics/reports/generate": {
        "post": {
            "tags": ["Analytics"],
            "summary": "Generate analytics report",
            "description": "Generate a formatted analytics report (PDF, CSV, JSON) for the specified time period and metrics.",
            "operationId": "postReportsGenerate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["report_type"],
                            "properties": {
                                "report_type": {
                                    "type": "string",
                                    "enum": [
                                        "performance",
                                        "usage",
                                        "cost",
                                        "agent_comparison",
                                        "debate_summary",
                                    ],
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["json", "csv", "pdf", "html"],
                                    "default": "json",
                                },
                                "start_date": {"type": "string", "format": "date"},
                                "end_date": {"type": "string", "format": "date"},
                                "include_charts": {"type": "boolean", "default": True},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Generated report", _REPORT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/analytics/{id}": {
        "delete": {
            "tags": ["Analytics"],
            "summary": "Delete analytics data",
            "description": "Delete analytics data for a specific resource or time period.",
            "operationId": "deleteAnalytics",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Analytics data identifier",
                }
            ],
            "responses": {
                "200": _ok_response("Analytics data deleted", _DELETE_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}

__all__ = ["SDK_MISSING_ANALYTICS_ENDPOINTS"]
