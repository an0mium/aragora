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
    # --- Response schemas for v1 cost endpoints ---
    "CostUsageResponse": {
        "type": "object",
        "description": "Detailed usage tracking data",
        "properties": {
            "workspace_id": {"type": "string"},
            "time_range": {"type": "string"},
            "group_by": {"type": "string"},
            "total_cost_usd": {"type": "number"},
            "total_tokens_in": {"type": "integer"},
            "total_tokens_out": {"type": "integer"},
            "total_api_calls": {"type": "integer"},
            "usage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost_usd": {"type": "number"},
                        "api_calls": {"type": "integer"},
                    },
                },
            },
            "period_start": {"type": "string", "format": "date-time"},
            "period_end": {"type": "string", "format": "date-time"},
        },
    },
    "CostBudgetsListResponse": {
        "type": "object",
        "description": "List of budgets for a workspace",
        "properties": {
            "budgets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "workspace_id": {"type": "string"},
                        "name": {"type": "string"},
                        "monthly_limit_usd": {"type": "number", "nullable": True},
                        "daily_limit_usd": {"type": "number", "nullable": True},
                        "current_monthly_spend": {"type": "number"},
                        "current_daily_spend": {"type": "number"},
                        "active": {"type": "boolean"},
                    },
                },
            },
            "count": {"type": "integer"},
            "workspace_id": {"type": "string"},
        },
        "required": ["budgets", "count"],
    },
    "CostBudgetCreateRequest": {
        "type": "object",
        "description": "Request to create a new budget",
        "properties": {
            "workspace_id": {"type": "string"},
            "name": {"type": "string"},
            "monthly_limit_usd": {"type": "number"},
            "daily_limit_usd": {"type": "number"},
            "alert_thresholds": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Alert threshold percentages (e.g. [50, 75, 90, 100])",
            },
        },
        "required": ["monthly_limit_usd"],
    },
    "CostConstraintCheckResponse": {
        "type": "object",
        "description": "Result of a pre-flight budget constraint check",
        "properties": {
            "allowed": {"type": "boolean"},
            "reason": {"type": "string"},
            "workspace_id": {"type": "string"},
            "estimated_cost_usd": {"type": "number"},
            "operation": {"type": "string"},
            "remaining_monthly_budget": {"type": "number", "nullable": True},
        },
        "required": ["allowed", "reason"],
    },
    "CostEstimateResponse": {
        "type": "object",
        "description": "Estimated cost for an operation",
        "properties": {
            "estimated_cost_usd": {"type": "number"},
            "breakdown": {
                "type": "object",
                "properties": {
                    "input_tokens": {"type": "integer"},
                    "output_tokens": {"type": "integer"},
                    "input_cost_usd": {"type": "number"},
                    "output_cost_usd": {"type": "number"},
                },
            },
            "pricing": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "provider": {"type": "string"},
                    "input_per_1m": {"type": "number"},
                    "output_per_1m": {"type": "number"},
                },
            },
            "operation": {"type": "string"},
        },
        "required": ["estimated_cost_usd"],
    },
    "CostForecastDetailedResponse": {
        "type": "object",
        "description": "Detailed cost forecast with daily breakdowns",
        "properties": {
            "workspace_id": {"type": "string"},
            "forecast_days": {"type": "integer"},
            "summary": {"type": "object"},
            "daily_forecasts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "format": "date"},
                        "projected_cost_usd": {"type": "number"},
                        "confidence_low": {"type": "number"},
                        "confidence_high": {"type": "number"},
                    },
                },
            },
            "confidence_level": {"type": "number"},
        },
    },
    "CostRecommendationsDetailedResponse": {
        "type": "object",
        "description": "Detailed cost optimization recommendations with implementation steps",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "estimated_savings_usd": {"type": "number"},
                        "implementation_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "difficulty": {"type": "string"},
                        "time_to_implement": {"type": "string"},
                    },
                },
            },
            "count": {"type": "integer"},
            "summary": {"type": "object"},
            "workspace_id": {"type": "string"},
            "total_potential_savings_usd": {"type": "number"},
        },
    },
    "CostAlertCreateResponse": {
        "type": "object",
        "description": "Response after creating a cost alert",
        "properties": {
            "success": {"type": "boolean"},
            "alert": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "threshold": {"type": "number"},
                    "notification_channels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "active": {"type": "boolean"},
                    "created_at": {"type": "string", "format": "date-time"},
                },
            },
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
