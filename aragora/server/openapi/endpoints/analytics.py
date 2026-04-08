"""Analytics and insights endpoint definitions."""

from aragora.server.openapi.helpers import STANDARD_ERRORS, _ok_response

_SPEND_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_name": {"type": "string", "description": "Agent type or configured agent name"},
        "cost_usd": {"type": "string", "description": "Total spend attributed to the agent"},
        "percentage": {
            "type": "number",
            "description": "Percentage of the workspace spend attributed to the agent",
        },
    },
}

_SPEND_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "debate_id": {"type": "string", "description": "Debate or decision identifier"},
        "cost_usd": {"type": "string", "description": "Total spend attributed to the decision"},
    },
}

_SPEND_BUDGET_SCHEMA = {
    "type": "object",
    "properties": {
        "budget_id": {"type": "string"},
        "name": {"type": "string"},
        "amount_usd": {"type": "number"},
        "spent_usd": {"type": "number"},
        "remaining_usd": {"type": "number"},
        "period": {"type": "string"},
        "period_start": {"type": "number"},
        "period_start_iso": {"type": ["string", "null"], "format": "date-time"},
        "period_end": {"type": "number"},
        "period_end_iso": {"type": ["string", "null"], "format": "date-time"},
        "status": {"type": "string"},
        "auto_suspend": {"type": "boolean"},
    },
}

ANALYTICS_ENDPOINTS = {
    "/api/analytics/disagreements": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Disagreement analysis",
            "operationId": "listAnalyticsDisagreements",
            "description": "Get metrics on agent disagreement patterns across debates.",
            "responses": {"200": _ok_response("Disagreement statistics", "DisagreementStats")},
        },
    },
    "/api/analytics/role-rotation": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Role rotation stats",
            "operationId": "listAnalyticsRoleRotation",
            "description": "Get statistics on how agents rotate between proposer, critic, and judge roles.",
            "responses": {"200": _ok_response("Role rotation data", "RoleRotationStats")},
        },
    },
    "/api/analytics/early-stops": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Early stop statistics",
            "operationId": "listAnalyticsEarlyStops",
            "description": "Get data on debates that ended early due to early consensus or other conditions.",
            "responses": {"200": _ok_response("Early stop data", "EarlyStopStats")},
        },
    },
    "/api/ranking/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Ranking statistics",
            "operationId": "listRankingStats",
            "description": "Get aggregate ELO ranking statistics across all agents.",
            "responses": {"200": _ok_response("Ranking stats", "RankingStats")},
        },
    },
    "/api/memory/stats": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Memory statistics",
            "operationId": "listMemoryStats",
            "description": "Get statistics on memory system usage and performance.",
            "responses": {"200": _ok_response("Memory stats", "MemoryStats")},
        },
    },
    "/api/flips/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent position flips",
            "operationId": "listFlipsRecent",
            "description": "Get recent instances where agents changed their positions during debate.",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Recent flips", "FlipsRecent")},
        },
    },
    "/api/flips/summary": {
        "get": {
            "tags": ["Insights"],
            "summary": "Flip summary",
            "operationId": "listFlipsSummary",
            "description": "Get summary statistics on position flips and conviction changes.",
            "responses": {"200": _ok_response("Flip summary statistics", "FlipsSummary")},
        },
    },
    "/api/insights/recent": {
        "get": {
            "tags": ["Insights"],
            "summary": "Recent insights",
            "operationId": "listInsightsRecent",
            "description": "Get recent insights extracted from debates.",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}}
            ],
            "responses": {"200": _ok_response("Recent insights", "InsightsRecent")},
        },
    },
    "/api/insights/extract-detailed": {
        "post": {
            "tags": ["Insights"],
            "summary": "Extract detailed insights",
            "operationId": "createInsightsExtractDetailed",
            "description": "Computationally expensive insight extraction (requires auth).",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Debate IDs to analyze",
                                },
                                "depth": {
                                    "type": "string",
                                    "enum": ["basic", "detailed", "comprehensive"],
                                    "description": "Analysis depth",
                                },
                            },
                        }
                    }
                }
            },
            "responses": {"200": _ok_response("Detailed insights", "InsightsDetailed")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/moments/summary": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments summary",
            "operationId": "listMomentsSummary",
            "description": "Get summary of key moments across debates.",
            "responses": {"200": _ok_response("Moments summary", "MomentsSummary")},
        },
    },
    "/api/moments/timeline": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments timeline",
            "operationId": "listMomentsTimeline",
            "description": "Get chronological timeline of significant debate moments.",
            "responses": {"200": _ok_response("Timeline data", "MomentsTimeline")},
        },
    },
    "/api/moments/trending": {
        "get": {
            "tags": ["Insights"],
            "summary": "Trending moments",
            "operationId": "listMomentsTrending",
            "description": "Get currently trending debate moments based on engagement.",
            "responses": {"200": _ok_response("Trending moments", "MomentsTrending")},
        },
    },
    "/api/moments/by-type/{type}": {
        "get": {
            "tags": ["Insights"],
            "summary": "Moments by type",
            "operationId": "getMomentsByType",
            "description": "Get debate moments filtered by type (e.g., breakthrough, conflict, consensus).",
            "parameters": [
                {"name": "type", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {"200": _ok_response("Moments of specified type", "MomentsByType")},
        },
    },
    # =========================================================================
    # Analytics Metrics (v1) endpoints
    # =========================================================================
    "/api/v1/analytics/debates/overview": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Debate overview stats",
            "operationId": "getAnalyticsDebatesOverview",
            "description": "Get debate overview statistics for a time range.",
            "parameters": [
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Debate overview", "AnalyticsDebatesOverview")},
        },
    },
    "/api/v1/analytics/debates/trends": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Debate trends",
            "operationId": "getAnalyticsDebatesTrends",
            "description": "Get debate trends over time.",
            "parameters": [
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "granularity", "in": "query", "schema": {"type": "string"}},
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Debate trends", "AnalyticsDebatesTrends")},
        },
    },
    "/api/v1/analytics/debates/topics": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Debate topics",
            "operationId": "getAnalyticsDebatesTopics",
            "description": "Get topic distribution for debates.",
            "parameters": [
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Debate topics", "AnalyticsDebatesTopics")},
        },
    },
    "/api/v1/analytics/debates/outcomes": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Debate outcomes",
            "operationId": "getAnalyticsDebatesOutcomes",
            "description": "Get debate outcome distribution.",
            "parameters": [
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Debate outcomes", "AnalyticsDebatesOutcomes")},
        },
    },
    "/api/v1/analytics/agents/leaderboard": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Agent leaderboard",
            "operationId": "getAnalyticsAgentsLeaderboard",
            "description": "Get agent leaderboard with rankings and win rates.",
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "domain", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Agent leaderboard", "AnalyticsAgentsLeaderboard")},
        },
    },
    "/api/v1/analytics/agents/{agent_id}/performance": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Agent performance",
            "operationId": "getAnalyticsAgentPerformance",
            "description": "Get individual agent performance stats.",
            "parameters": [
                {"name": "agent_id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Agent performance", "AnalyticsAgentPerformance")},
        },
    },
    "/api/v1/analytics/agents/comparison": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Agent comparison",
            "operationId": "getAnalyticsAgentsComparison",
            "description": "Compare multiple agents.",
            "parameters": [
                {"name": "agents", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Agent comparison", "AnalyticsAgentsComparison")},
        },
    },
    "/api/v1/analytics/agents/trends": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Agent trends",
            "operationId": "getAnalyticsAgentsTrends",
            "description": "Get agent performance trends over time.",
            "parameters": [
                {"name": "agents", "in": "query", "schema": {"type": "string"}},
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "granularity", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Agent trends", "AnalyticsAgentsTrends")},
        },
    },
    "/api/v1/analytics/spend/summary": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Spend analytics summary",
            "operationId": "getAnalyticsSpendSummary",
            "description": (
                "Get aggregate spend, usage, and budget utilization for a workspace or "
                "organization. Mirrors the live spend analytics dashboard summary card."
            ),
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Spend summary",
                    {
                        "type": "object",
                        "properties": {
                            "total_spend_usd": {
                                "type": "string",
                                "description": "Formatted total spend for the scope",
                            },
                            "total_api_calls": {
                                "type": "integer",
                                "description": "Total API calls recorded for the scope",
                            },
                            "total_tokens": {
                                "type": "integer",
                                "description": "Combined input and output token usage",
                            },
                            "budget_limit_usd": {
                                "type": "number",
                                "description": "Combined active budget limit for the org",
                            },
                            "budget_spent_usd": {
                                "type": "number",
                                "description": "Combined spend across active org budgets",
                            },
                            "utilization_pct": {
                                "type": "number",
                                "description": "Percent of available budget already spent",
                            },
                            "trend_direction": {
                                "type": "string",
                                "enum": ["stable", "increasing", "decreasing"],
                                "description": "Projected spend direction relative to current spend",
                            },
                            "avg_cost_per_decision": {
                                "type": "number",
                                "description": "Average spend per recorded API call",
                            },
                        },
                    },
                    include_rate_limit_headers=True,
                ),
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/analytics/spend/trends": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Spend trends",
            "operationId": "getAnalyticsSpendTrends",
            "description": "Get daily, weekly, or monthly spend trend points for an organization.",
            "parameters": [
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "period",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
                },
                {
                    "name": "days",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 365},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Spend trends",
                    {
                        "type": "object",
                        "properties": {
                            "org_id": {"type": "string"},
                            "period": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                            },
                            "days": {"type": "integer"},
                            "data_points": {
                                "type": "array",
                                "items": {"type": "object", "additionalProperties": True},
                                "description": "Budget-manager trend rows for the selected period",
                            },
                        },
                    },
                    include_rate_limit_headers=True,
                ),
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/analytics/spend/by-agent": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Spend by agent",
            "operationId": "getAnalyticsSpendByAgent",
            "description": "Get per-agent spend attribution for a workspace.",
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Spend by agent",
                    {
                        "type": "object",
                        "properties": {
                            "workspace_id": {"type": "string"},
                            "total_usd": {
                                "type": "string",
                                "description": "Formatted total spend for the workspace",
                            },
                            "agents": {
                                "type": "array",
                                "items": _SPEND_AGENT_SCHEMA,
                            },
                        },
                    },
                    include_rate_limit_headers=True,
                ),
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/analytics/spend/by-decision": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Spend by decision",
            "operationId": "getAnalyticsSpendByDecision",
            "description": "Get the most expensive debates or decisions for a workspace.",
            "parameters": [
                {"name": "workspace_id", "in": "query", "schema": {"type": "string"}},
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Spend by decision",
                    {
                        "type": "object",
                        "properties": {
                            "workspace_id": {"type": "string"},
                            "decisions": {
                                "type": "array",
                                "items": _SPEND_DECISION_SCHEMA,
                            },
                            "count": {"type": "integer"},
                        },
                    },
                    include_rate_limit_headers=True,
                ),
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/analytics/spend/budget": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Spend budget status",
            "operationId": "getAnalyticsSpendBudget",
            "description": "Get budget totals, remaining funds, and exhaustion forecast for an org.",
            "parameters": [
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response(
                    "Spend budget status",
                    {
                        "type": "object",
                        "properties": {
                            "org_id": {"type": "string"},
                            "budgets": {
                                "type": "array",
                                "items": _SPEND_BUDGET_SCHEMA,
                            },
                            "total_budget_usd": {"type": "number"},
                            "total_spent_usd": {"type": "number"},
                            "total_remaining_usd": {"type": "number"},
                            "utilization_pct": {"type": "number"},
                            "forecast_exhaustion_days": {"type": ["number", "null"]},
                        },
                    },
                    include_rate_limit_headers=True,
                ),
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/analytics/usage/tokens": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Token usage",
            "operationId": "getAnalyticsUsageTokens",
            "description": "Get token usage summaries for an organization.",
            "parameters": [
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
                {"name": "granularity", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Token usage", "AnalyticsUsageTokens")},
        },
    },
    "/api/v1/analytics/usage/costs": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Usage costs",
            "operationId": "getAnalyticsUsageCosts",
            "description": "Get cost breakdown for an organization.",
            "parameters": [
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Usage costs", "AnalyticsUsageCosts")},
        },
    },
    "/api/v1/analytics/usage/active_users": {
        "get": {
            "tags": ["Analytics"],
            "summary": "Active users",
            "operationId": "getAnalyticsActiveUsers",
            "description": "Get active user counts for an organization.",
            "parameters": [
                {"name": "org_id", "in": "query", "schema": {"type": "string"}},
                {"name": "time_range", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {"200": _ok_response("Active users", "AnalyticsActiveUsers")},
        },
    },
}
