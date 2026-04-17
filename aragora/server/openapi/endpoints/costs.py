"""Cost visibility endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

_DEBATE_ID_PARAM = {
    "name": "debate_id",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Debate session ID.",
}

_DEBATE_COST_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "debate_id": {"type": "string"},
                "total_cost_usd": {"type": "number"},
                "total_tokens_in": {"type": "integer"},
                "total_tokens_out": {"type": "integer"},
                "api_calls": {"type": "integer"},
                "avg_latency_ms": {"type": "number"},
                "by_agent": {"type": "array", "items": {"type": "object"}},
                "by_model": {"type": "array", "items": {"type": "object"}},
                "budget": {"type": "object", "additionalProperties": True},
            },
        }
    },
}

_DEBATE_COST_LINE_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "debate_id": {"type": "string"},
                "line_items": {"type": "array", "items": {"type": "object"}},
                "total_count": {"type": "integer"},
                "returned_count": {"type": "integer"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
                "page_total_cost_usd": {"type": "number"},
                "page_total_tokens": {"type": "integer"},
            },
        }
    },
}

_DEBATE_COST_PERFORMANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {
            "type": "object",
            "properties": {
                "debate_id": {"type": "string"},
                "api_calls": {"type": "integer"},
                "total_cost_usd": {"type": "number"},
                "total_tokens": {"type": "integer"},
                "duration_seconds": {"type": "number"},
                "throughput": {"type": "object", "additionalProperties": True},
                "latency": {"type": "object", "additionalProperties": True},
                "cost_efficiency": {"type": "object", "additionalProperties": True},
                "by_operation": {"type": "array", "items": {"type": "object"}},
                "time_range": {"type": "object", "additionalProperties": True},
                "message": {"type": "string"},
            },
        }
    },
}

COSTS_ENDPOINTS = {
    "/api/costs": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get cost summary",
            "operationId": "listCosts",
            "description": "Fetch cost dashboard summary data.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "range", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Cost summary", "CostSummaryResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/costs/breakdown": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get cost breakdown",
            "operationId": "listCostsBreakdown",
            "description": "Fetch cost breakdown by provider or feature.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "range", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "group_by",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["provider", "feature", "model"]},
                },
            ],
            "responses": {
                "200": _ok_response("Cost breakdown", "CostBreakdownResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/costs/timeline": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get cost timeline",
            "operationId": "listCostsTimeline",
            "description": "Fetch cost timeline data.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "range", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Cost timeline", "CostTimelineResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/costs/alerts": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get budget alerts",
            "operationId": "listCostsAlerts",
            "description": "Fetch active budget alerts.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Alerts", "CostAlertsResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/costs/budget": {
        "post": {
            "tags": ["Costs"],
            "summary": "Set budget limits",
            "operationId": "createCostsBudget",
            "description": "Set workspace budget limit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CostBudgetRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Budget updated", "CostBudgetResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/costs/alerts/{alert_id}/dismiss": {
        "post": {
            "tags": ["Costs"],
            "summary": "Dismiss alert",
            "operationId": "createCostsAlertsDismis",
            "description": "Dismiss a budget alert.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "alert_id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Alert dismissed", "CostDismissAlertResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Usage tracking
    "/api/v1/costs/usage": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get usage tracking",
            "operationId": "getCostsUsage",
            "description": "Get detailed usage tracking data for the workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "range",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["24h", "7d", "30d", "90d"]},
                },
                {
                    "name": "group_by",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["provider", "model", "operation"]},
                },
            ],
            "responses": {
                "200": _ok_response("Usage data", "CostUsageResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Budget management
    "/api/v1/costs/budgets": {
        "get": {
            "tags": ["Costs"],
            "summary": "List budgets",
            "operationId": "listCostsBudgets",
            "description": "List all budgets for the workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "active_only",
                    "in": "query",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": _ok_response("Budgets list", "CostBudgetsListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Costs"],
            "summary": "Create budget",
            "operationId": "createCostsBudgets",
            "description": "Create a new budget for the workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CostBudgetCreateRequest"}
                    }
                },
            },
            "responses": {
                "201": _ok_response("Budget created", "CostBudgetResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    # Constraints check
    "/api/v1/costs/constraints/check": {
        "post": {
            "tags": ["Costs"],
            "summary": "Check cost constraints",
            "operationId": "postCostsConstraintsCheck",
            "description": "Pre-flight check if an operation would exceed budget constraints.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "workspace_id": {"type": "string"},
                                "estimated_cost_usd": {"type": "number"},
                                "operation": {"type": "string"},
                            },
                            "required": ["estimated_cost_usd"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Constraint check result", "CostConstraintCheckResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Cost estimation
    "/api/v1/costs/estimate": {
        "post": {
            "tags": ["Costs"],
            "summary": "Estimate operation cost",
            "operationId": "postCostsEstimate",
            "description": "Estimate the cost of an operation before executing it.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string"},
                                "tokens_input": {"type": "integer"},
                                "tokens_output": {"type": "integer"},
                                "model": {"type": "string"},
                                "provider": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Cost estimate", "CostEstimateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/efficiency": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get efficiency metrics",
            "operationId": "getCostsEfficiency",
            "description": "Get cost efficiency metrics including cost per token and model utilization.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "range", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Efficiency metrics",
                    {
                        "workspace_id": {"type": "string"},
                        "range": {"type": "string"},
                        "metrics": {"type": "object", "additionalProperties": True},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/export": {
        "get": {
            "tags": ["Costs"],
            "summary": "Export cost data",
            "operationId": "getCostsExport",
            "description": "Export usage data as CSV or JSON.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "format",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["json", "csv"], "default": "json"},
                },
                {"name": "range", "in": "query", "schema": {"type": "string"}},
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Export generated",
                    {
                        "format": {"type": "string"},
                        "filename": {"type": "string"},
                        "content": {"type": "string"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/forecast": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get cost forecast",
            "operationId": "getCostsForecast",
            "description": "Get cost forecast for the specified number of days.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "days",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Forecast",
                    {
                        "workspace_id": {"type": "string"},
                        "days": {"type": "integer"},
                        "predicted_cost": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/recommendations": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get recommendations",
            "operationId": "getCostsRecommendations",
            "description": "Get cost optimization recommendations for the workspace.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["pending", "applied", "dismissed"]},
                },
                {"name": "type", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Recommendations",
                    {
                        "recommendations": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                    },
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Detailed forecast
    "/api/v1/costs/forecast/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get detailed forecast",
            "operationId": "getCostsForecastDetailed",
            "description": "Get detailed cost forecast with daily breakdowns and confidence intervals.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "days",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 90, "default": 30},
                },
                {
                    "name": "include_confidence",
                    "in": "query",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": _ok_response("Detailed forecast", "CostForecastDetailedResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Detailed recommendations
    "/api/v1/costs/recommendations/detailed": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get detailed recommendations",
            "operationId": "getCostsRecommendationsDetailed",
            "description": "Get detailed cost optimization recommendations with implementation steps.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "include_implementation",
                    "in": "query",
                    "schema": {"type": "boolean", "default": True},
                },
                {"name": "min_savings", "in": "query", "schema": {"type": "number", "default": 0}},
            ],
            "responses": {
                "200": _ok_response(
                    "Detailed recommendations", "CostRecommendationsDetailedResponse"
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    # Alert creation
    "/api/v1/costs/alerts": {
        "post": {
            "tags": ["Costs"],
            "summary": "Create cost alert",
            "operationId": "postCostsAlerts",
            "description": "Create a new cost alert with custom thresholds.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "workspace_id": {"type": "string"},
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["budget_threshold", "spike_detection", "daily_limit"],
                                },
                                "threshold": {"type": "number"},
                                "notification_channels": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name"],
                        }
                    }
                },
            },
            "responses": {
                "201": _ok_response("Alert created", "CostAlertCreateResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/debates/{debate_id}": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get debate session costs",
            "operationId": "getDebateSessionCosts",
            "description": "Get total cost, token usage, model breakdowns, and budget status for a debate session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_DEBATE_ID_PARAM],
            "responses": {
                "200": _ok_response("Debate session costs", _DEBATE_COST_SUMMARY_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "503": {"description": "Cost tracker unavailable"},
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/debates/{debate_id}/line-items": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get debate cost line items",
            "operationId": "listDebateCostLineItems",
            "description": "List individual API-call cost line items for a debate session.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                _DEBATE_ID_PARAM,
                {
                    "name": "sort_by",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["cost", "timestamp", "tokens"],
                        "default": "timestamp",
                    },
                },
                {
                    "name": "order",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 0, "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response("Debate cost line items", _DEBATE_COST_LINE_ITEMS_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "503": {"description": "Cost tracker unavailable"},
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/costs/debates/{debate_id}/performance": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get debate cost performance",
            "operationId": "getDebateCostPerformance",
            "description": "Get latency, throughput, and cost-efficiency metrics for a debate session's API usage.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [_DEBATE_ID_PARAM],
            "responses": {
                "200": _ok_response("Debate cost performance", _DEBATE_COST_PERFORMANCE_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "503": {"description": "Cost tracker unavailable"},
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["COSTS_ENDPOINTS"]
