"""
Cost and Budget OpenAPI Schema Definitions.

Schemas for cost tracking, budget management, and alerts.
"""

from typing import Any

COST_SCHEMAS: dict[str, Any] = {
    # Cost tracking schemas
    "CostBreakdownItem": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "cost": {"type": "number"},
            "percentage": {"type": "number"},
        },
    },
    "CostDailyItem": {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "cost": {"type": "number"},
            "tokens": {"type": "integer"},
        },
    },
    "CostAlert": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "message": {"type": "string"},
            "severity": {"type": "string"},
            "timestamp": {"type": "string"},
        },
    },
    "CostSummaryResponse": {
        "type": "object",
        "properties": {
            "totalCost": {"type": "number"},
            "budget": {"type": "number"},
            "tokensUsed": {"type": "integer"},
            "apiCalls": {"type": "integer"},
            "lastUpdated": {"type": "string"},
            "costByProvider": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CostBreakdownItem"},
            },
            "costByFeature": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CostBreakdownItem"},
            },
            "dailyCosts": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CostDailyItem"},
            },
            "alerts": {"type": "array", "items": {"$ref": "#/components/schemas/CostAlert"}},
        },
    },
    "CostBreakdownResponse": {
        "type": "object",
        "properties": {
            "groupBy": {"type": "string"},
            "breakdown": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CostBreakdownItem"},
            },
            "total": {"type": "number"},
        },
    },
    "CostTimelineResponse": {
        "type": "object",
        "properties": {
            "timeline": {"type": "array", "items": {"$ref": "#/components/schemas/CostDailyItem"}},
            "total": {"type": "number"},
            "average": {"type": "number"},
        },
    },
    "CostAlertsResponse": {
        "type": "object",
        "properties": {
            "alerts": {"type": "array", "items": {"$ref": "#/components/schemas/CostAlert"}},
        },
    },
    "CostBudgetRequest": {
        "type": "object",
        "properties": {
            "budget": {"type": "number"},
            "workspace_id": {"type": "string"},
        },
        "required": ["budget"],
    },
    "CostBudgetResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "budget": {"type": "number"},
            "workspace_id": {"type": "string"},
        },
        "required": ["success"],
    },
    "CostDismissAlertResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
        },
        "required": ["success"],
    },
    # Budget management schemas
    "BudgetPeriod": {
        "type": "string",
        "description": "Budget period type",
        "enum": ["daily", "weekly", "monthly", "quarterly", "annual", "unlimited"],
    },
    "BudgetStatus": {
        "type": "string",
        "description": "Budget status",
        "enum": ["active", "warning", "critical", "exceeded", "suspended", "paused", "closed"],
    },
    "BudgetAction": {
        "type": "string",
        "description": "Action when budget threshold is reached",
        "enum": ["notify", "warn", "soft_limit", "hard_limit", "suspend"],
    },
    "BudgetThreshold": {
        "type": "object",
        "description": "Budget alert threshold configuration",
        "properties": {
            "percentage": {
                "type": "number",
                "description": "Threshold percentage (0.0 - 1.0)",
                "minimum": 0,
                "maximum": 1,
            },
            "action": {"$ref": "#/components/schemas/BudgetAction"},
        },
        "required": ["percentage", "action"],
    },
    "Budget": {
        "type": "object",
        "description": "Budget configuration",
        "properties": {
            "id": {"type": "string"},
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string", "nullable": True},
            "limit_usd": {"type": "number"},
            "period": {"$ref": "#/components/schemas/BudgetPeriod"},
            "status": {"$ref": "#/components/schemas/BudgetStatus"},
            "current_spend_usd": {"type": "number"},
            "current_period_start": {"type": "string", "format": "date-time"},
            "current_period_end": {"type": "string", "format": "date-time"},
            "thresholds": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BudgetThreshold"},
            },
            "scope": {"type": "object"},
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
            "created_by": {"type": "string"},
        },
        "required": ["id", "workspace_id", "name", "limit_usd", "period", "status"],
    },
    "BudgetCreateRequest": {
        "type": "object",
        "description": "Request to create a budget",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "limit_usd": {"type": "number"},
            "period": {"$ref": "#/components/schemas/BudgetPeriod"},
            "thresholds": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BudgetThreshold"},
            },
            "scope": {"type": "object"},
        },
        "required": ["name", "limit_usd", "period"],
    },
    "BudgetUpdateRequest": {
        "type": "object",
        "description": "Request to update a budget",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "limit_usd": {"type": "number"},
            "period": {"$ref": "#/components/schemas/BudgetPeriod"},
            "status": {"$ref": "#/components/schemas/BudgetStatus"},
            "thresholds": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BudgetThreshold"},
            },
        },
    },
    "BudgetListResponse": {
        "type": "object",
        "description": "Response containing list of budgets",
        "properties": {
            "budgets": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Budget"},
            },
            "total": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["budgets", "total"],
    },
    "BudgetSummary": {
        "type": "object",
        "description": "Summary of budget status across workspace",
        "properties": {
            "total_budget_usd": {"type": "number"},
            "total_spend_usd": {"type": "number"},
            "active_budgets": {"type": "integer"},
            "warning_budgets": {"type": "integer"},
            "exceeded_budgets": {"type": "integer"},
            "utilization_percentage": {"type": "number"},
            "trend": {"type": "string", "enum": ["increasing", "stable", "decreasing"]},
        },
    },
    "BudgetCheckRequest": {
        "type": "object",
        "description": "Request to check budget for a specific operation",
        "properties": {
            "operation_type": {"type": "string"},
            "estimated_cost_usd": {"type": "number"},
            "model": {"type": "string"},
            "tokens": {"type": "integer"},
        },
        "required": ["operation_type"],
    },
    "BudgetCheckResponse": {
        "type": "object",
        "description": "Response to budget check request",
        "properties": {
            "allowed": {"type": "boolean"},
            "budget_id": {"type": "string", "nullable": True},
            "remaining_usd": {"type": "number", "nullable": True},
            "current_utilization": {"type": "number", "nullable": True},
            "reason": {"type": "string", "nullable": True},
        },
        "required": ["allowed"],
    },
    "BudgetAlert": {
        "type": "object",
        "description": "Budget alert notification",
        "properties": {
            "id": {"type": "string"},
            "budget_id": {"type": "string"},
            "budget_name": {"type": "string"},
            "threshold_percentage": {"type": "number"},
            "actual_percentage": {"type": "number"},
            "action_taken": {"$ref": "#/components/schemas/BudgetAction"},
            "message": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"},
            "acknowledged": {"type": "boolean"},
            "acknowledged_at": {"type": "string", "format": "date-time", "nullable": True},
            "acknowledged_by": {"type": "string", "nullable": True},
        },
        "required": ["id", "budget_id", "budget_name", "threshold_percentage", "action_taken"],
    },
    "BudgetAlertListResponse": {
        "type": "object",
        "description": "Response containing list of budget alerts",
        "properties": {
            "alerts": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BudgetAlert"},
            },
            "total": {"type": "integer"},
            "unacknowledged_count": {"type": "integer"},
        },
        "required": ["alerts", "total"],
    },
    "BudgetOverrideRequest": {
        "type": "object",
        "description": "Request to add a temporary budget override",
        "properties": {
            "budget_id": {"type": "string"},
            "override_limit_usd": {"type": "number"},
            "duration_hours": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["budget_id", "override_limit_usd", "reason"],
    },
    "BudgetOverrideResponse": {
        "type": "object",
        "description": "Response for budget override request",
        "properties": {
            "override_added": {"type": "boolean"},
            "budget_id": {"type": "string"},
            "user_id": {"type": "string"},
            "duration_hours": {"type": ["number", "null"]},
        },
        "required": ["override_added", "budget_id", "user_id"],
    },
}


__all__ = ["COST_SCHEMAS"]
