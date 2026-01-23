"""Cost visibility endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


COSTS_ENDPOINTS = {
    "/api/costs": {
        "get": {
            "tags": ["Costs"],
            "summary": "Get cost summary",
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
}


__all__ = ["COSTS_ENDPOINTS"]
