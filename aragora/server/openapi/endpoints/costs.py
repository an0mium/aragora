"""Cost visibility endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

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
}


__all__ = ["COSTS_ENDPOINTS"]
