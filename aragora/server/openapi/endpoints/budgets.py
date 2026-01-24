"""
Budget Management API Endpoints.

Provides budget CRUD, enforcement, alerts, and override management.
"""

from aragora.server.openapi.schemas import STANDARD_ERRORS

BUDGET_ENDPOINTS = {
    "/api/v1/budgets": {
        "get": {
            "tags": ["Budgets"],
            "summary": "List organization budgets",
            "description": "Get all budgets for the authenticated organization.",
            "operationId": "listBudgets",
            "parameters": [
                {
                    "name": "active_only",
                    "in": "query",
                    "description": "Only return active budgets",
                    "schema": {"type": "boolean", "default": True},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of budgets",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BudgetListResponse"}
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Budgets"],
            "summary": "Create a new budget",
            "description": "Create a new budget for the organization with spending limits and alert thresholds.",
            "operationId": "createBudget",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/BudgetCreateRequest"},
                        "example": {
                            "name": "Monthly AI Budget",
                            "amount_usd": 500.00,
                            "period": "monthly",
                            "description": "Budget for AI debate operations",
                            "auto_suspend": True,
                        },
                    }
                },
            },
            "responses": {
                "201": {
                    "description": "Budget created",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Budget"}}
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/summary": {
        "get": {
            "tags": ["Budgets"],
            "summary": "Get budget summary",
            "description": "Get aggregated budget summary for the organization including total spend and remaining budget.",
            "operationId": "getBudgetSummary",
            "responses": {
                "200": {
                    "description": "Budget summary",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BudgetSummary"}
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/check": {
        "post": {
            "tags": ["Budgets"],
            "summary": "Pre-flight budget check",
            "description": "Check if an operation with the estimated cost would be allowed within budget limits. Use this before starting expensive operations.",
            "operationId": "checkBudget",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/BudgetCheckRequest"},
                        "example": {"estimated_cost_usd": 0.50},
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Budget check result",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BudgetCheckResponse"},
                            "example": {
                                "allowed": True,
                                "reason": "OK",
                                "action": None,
                                "estimated_cost_usd": 0.50,
                            },
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}": {
        "get": {
            "tags": ["Budgets"],
            "summary": "Get budget details",
            "description": "Get detailed information about a specific budget including current spend and thresholds.",
            "operationId": "getBudget",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "responses": {
                "200": {
                    "description": "Budget details",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Budget"}}
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
        "patch": {
            "tags": ["Budgets"],
            "summary": "Update a budget",
            "description": "Update budget configuration such as name, amount, or status.",
            "operationId": "updateBudget",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/BudgetUpdateRequest"},
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Updated budget",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Budget"}}
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Budgets"],
            "summary": "Delete a budget",
            "description": "Close a budget (soft delete). The budget history is preserved.",
            "operationId": "deleteBudget",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "responses": {
                "200": {
                    "description": "Budget deleted",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "deleted": {"type": "boolean"},
                                    "budget_id": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}/alerts": {
        "get": {
            "tags": ["Budgets"],
            "summary": "Get budget alerts",
            "description": "Get all alerts triggered for this budget, including acknowledged and unacknowledged.",
            "operationId": "getBudgetAlerts",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "responses": {
                "200": {
                    "description": "List of alerts",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BudgetAlertListResponse"}
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}/alerts/{alert_id}/acknowledge": {
        "post": {
            "tags": ["Budgets"],
            "summary": "Acknowledge an alert",
            "description": "Mark a budget alert as acknowledged by the current user.",
            "operationId": "acknowledgeBudgetAlert",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
                {
                    "name": "alert_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Alert ID",
                },
            ],
            "responses": {
                "200": {
                    "description": "Alert acknowledged",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "acknowledged": {"type": "boolean"},
                                    "alert_id": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}/override": {
        "post": {
            "tags": ["Budgets"],
            "summary": "Add budget override",
            "description": "Grant a user temporary or permanent permission to bypass budget limits.",
            "operationId": "addBudgetOverride",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/BudgetOverrideRequest"},
                        "example": {
                            "user_id": "user-123",
                            "duration_hours": 24,
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Override added",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/BudgetOverrideResponse"}
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}/override/{user_id}": {
        "delete": {
            "tags": ["Budgets"],
            "summary": "Remove budget override",
            "description": "Remove a user's budget override permission.",
            "operationId": "removeBudgetOverride",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "User ID to remove override for",
                },
            ],
            "responses": {
                "200": {
                    "description": "Override removed",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "override_removed": {"type": "boolean"},
                                    "budget_id": {"type": "string"},
                                    "user_id": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/budgets/{budget_id}/reset": {
        "post": {
            "tags": ["Budgets"],
            "summary": "Reset budget period",
            "description": "Reset the budget's spent amount for a new period. Use this to manually start a new budget cycle.",
            "operationId": "resetBudgetPeriod",
            "parameters": [
                {
                    "name": "budget_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Budget ID",
                },
            ],
            "responses": {
                "200": {
                    "description": "Budget reset",
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Budget"}}
                    },
                },
                **STANDARD_ERRORS,
            },
            "security": [{"bearerAuth": []}],
        },
    },
}

__all__ = ["BUDGET_ENDPOINTS"]
