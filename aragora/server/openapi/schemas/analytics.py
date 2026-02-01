"""
Analytics OpenAPI Schema Definitions.

Schemas for debate analytics, insights, moments, and campaigns.
"""

from typing import Any

ANALYTICS_SCHEMAS: dict[str, Any] = {
    # Debate analytics schemas
    "DisagreementStats": {
        "type": "object",
        "description": "Statistics about debate disagreements",
        "properties": {
            "total_debates": {"type": "integer", "description": "Total debates analyzed"},
            "with_disagreements": {"type": "integer", "description": "Debates with disagreements"},
            "unanimous": {"type": "integer", "description": "Unanimous debates"},
            "disagreement_types": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
                "description": "Count by disagreement type",
            },
        },
    },
    "RoleRotationStats": {
        "type": "object",
        "description": "Statistics about agent role rotation",
        "properties": {
            "total_rotations": {"type": "integer"},
            "by_agent": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "proposer": {"type": "integer"},
                        "critic": {"type": "integer"},
                        "judge": {"type": "integer"},
                    },
                },
            },
        },
    },
    "EarlyStopStats": {
        "type": "object",
        "description": "Statistics about early debate stops",
        "properties": {
            "total_early_stops": {"type": "integer"},
            "by_reason": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
            "average_rounds_saved": {"type": "number"},
        },
    },
    "RankingStats": {
        "type": "object",
        "description": "Aggregate ELO ranking statistics",
        "properties": {
            "total_agents": {"type": "integer"},
            "average_elo": {"type": "number"},
            "highest_elo": {"type": "number"},
            "lowest_elo": {"type": "number"},
            "total_matches": {"type": "integer"},
        },
    },
    # Position flip schemas
    "PositionFlip": {
        "type": "object",
        "description": "A position change by an agent during debate",
        "properties": {
            "debate_id": {"type": "string"},
            "agent": {"type": "string"},
            "round": {"type": "integer"},
            "old_position": {"type": "string"},
            "new_position": {"type": "string"},
            "reason": {"type": "string"},
            "conviction_delta": {"type": "number"},
            "timestamp": {"type": "string", "format": "date-time"},
        },
    },
    "FlipsRecent": {
        "type": "object",
        "description": "Recent position flips response",
        "properties": {
            "flips": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/PositionFlip"},
            },
            "total": {"type": "integer"},
        },
    },
    "FlipsSummary": {
        "type": "object",
        "description": "Summary statistics on position flips",
        "properties": {
            "total_flips": {"type": "integer"},
            "by_agent": {"type": "object", "additionalProperties": {"type": "integer"}},
            "by_debate": {"type": "object", "additionalProperties": {"type": "integer"}},
            "average_conviction_delta": {"type": "number"},
            "flip_rate": {"type": "number", "description": "Percentage of debates with flips"},
        },
    },
    # Insight schemas
    "Insight": {
        "type": "object",
        "description": "An insight extracted from debate",
        "properties": {
            "id": {"type": "string"},
            "debate_id": {"type": "string"},
            "content": {"type": "string"},
            "type": {"type": "string", "enum": ["observation", "conclusion", "recommendation"]},
            "confidence": {"type": "number"},
            "supporting_evidence": {"type": "array", "items": {"type": "string"}},
            "extracted_at": {"type": "string", "format": "date-time"},
        },
    },
    "InsightsRecent": {
        "type": "object",
        "description": "Recent insights response",
        "properties": {
            "insights": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Insight"},
            },
            "total": {"type": "integer"},
        },
    },
    "InsightsDetailed": {
        "type": "object",
        "description": "Detailed insight extraction result",
        "properties": {
            "insights": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Insight"},
            },
            "themes": {"type": "array", "items": {"type": "string"}},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "processing_time_ms": {"type": "integer"},
        },
    },
    # Moment schemas
    "DebateMoment": {
        "type": "object",
        "description": "A significant moment in a debate",
        "properties": {
            "id": {"type": "string"},
            "debate_id": {"type": "string"},
            "type": {
                "type": "string",
                "enum": ["breakthrough", "conflict", "consensus", "insight", "flip"],
            },
            "round": {"type": "integer"},
            "description": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
            "significance_score": {"type": "number"},
            "timestamp": {"type": "string", "format": "date-time"},
        },
    },
    "MomentsSummary": {
        "type": "object",
        "description": "Summary of key moments across debates",
        "properties": {
            "total_moments": {"type": "integer"},
            "by_type": {"type": "object", "additionalProperties": {"type": "integer"}},
            "top_debates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "debate_id": {"type": "string"},
                        "moment_count": {"type": "integer"},
                    },
                },
            },
        },
    },
    "MomentsTimeline": {
        "type": "object",
        "description": "Chronological timeline of moments",
        "properties": {
            "moments": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DebateMoment"},
            },
            "start_time": {"type": "string", "format": "date-time"},
            "end_time": {"type": "string", "format": "date-time"},
        },
    },
    "MomentsTrending": {
        "type": "object",
        "description": "Currently trending debate moments",
        "properties": {
            "moments": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DebateMoment"},
            },
            "trending_period_hours": {"type": "integer"},
        },
    },
    "MomentsByType": {
        "type": "object",
        "description": "Moments filtered by type",
        "properties": {
            "type": {"type": "string"},
            "moments": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DebateMoment"},
            },
            "total": {"type": "integer"},
        },
    },
    # Campaign schemas
    "UnifiedCampaign": {
        "type": "object",
        "description": "Unified campaign representation across advertising platforms",
        "properties": {
            "id": {
                "type": "string",
                "description": "Platform-specific campaign ID",
            },
            "platform": {
                "type": "string",
                "description": "Advertising platform name",
                "enum": ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"],
            },
            "name": {
                "type": "string",
                "description": "Campaign name",
            },
            "status": {
                "type": "string",
                "description": "Campaign status",
                "enum": ["ENABLED", "PAUSED", "REMOVED"],
            },
            "objective": {
                "type": "string",
                "description": "Campaign objective/goal",
                "nullable": True,
            },
            "daily_budget": {
                "type": "number",
                "description": "Daily budget in account currency",
                "nullable": True,
            },
            "total_budget": {
                "type": "number",
                "description": "Total campaign budget",
                "nullable": True,
            },
            "start_date": {
                "type": "string",
                "format": "date",
                "description": "Campaign start date",
                "nullable": True,
            },
            "end_date": {
                "type": "string",
                "format": "date",
                "description": "Campaign end date",
                "nullable": True,
            },
            "created_at": {
                "type": "string",
                "format": "date-time",
                "description": "When the campaign was created",
                "nullable": True,
            },
            "updated_at": {
                "type": "string",
                "format": "date-time",
                "description": "When the campaign was last updated",
                "nullable": True,
            },
        },
        "required": ["id", "platform", "name", "status"],
    },
    # =========================================================================
    # Analytics Metrics (v1) schemas
    # =========================================================================
    "AnalyticsDebatesOverview": {
        "type": "object",
        "description": "Overview statistics for debates",
        "properties": {
            "time_range": {"type": "string"},
            "total_debates": {"type": "integer"},
            "debates_this_period": {"type": "integer"},
            "debates_previous_period": {"type": "integer"},
            "growth_rate": {"type": "number"},
            "consensus_reached": {"type": "integer"},
            "consensus_rate": {"type": "number"},
            "avg_rounds": {"type": "number"},
            "avg_agents_per_debate": {"type": "number"},
            "avg_confidence": {"type": "number"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsDebateTrendPoint": {
        "type": "object",
        "description": "Single time bucket for debate trends",
        "properties": {
            "period": {"type": "string"},
            "total": {"type": "integer"},
            "consensus_reached": {"type": "integer"},
            "consensus_rate": {"type": "number"},
            "avg_rounds": {"type": "number"},
        },
    },
    "AnalyticsDebatesTrends": {
        "type": "object",
        "description": "Debate trends over time",
        "properties": {
            "time_range": {"type": "string"},
            "granularity": {"type": "string"},
            "data_points": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/AnalyticsDebateTrendPoint"},
            },
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsDebateTopic": {
        "type": "object",
        "description": "Topic distribution entry",
        "properties": {
            "topic": {"type": "string"},
            "count": {"type": "integer"},
            "percentage": {"type": "number"},
            "consensus_rate": {"type": "number"},
        },
    },
    "AnalyticsDebatesTopics": {
        "type": "object",
        "description": "Topic distribution across debates",
        "properties": {
            "time_range": {"type": "string"},
            "topics": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/AnalyticsDebateTopic"},
            },
            "total_debates": {"type": "integer"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsOutcomeConfidenceBucket": {
        "type": "object",
        "description": "Outcome stats for a confidence bucket",
        "properties": {
            "count": {"type": "integer"},
            "consensus_rate": {"type": "number"},
        },
    },
    "AnalyticsDebatesOutcomes": {
        "type": "object",
        "description": "Debate outcome distribution",
        "properties": {
            "time_range": {"type": "string"},
            "outcomes": {
                "type": "object",
                "properties": {
                    "consensus": {"type": "integer"},
                    "majority": {"type": "integer"},
                    "dissent": {"type": "integer"},
                    "no_resolution": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            "total_debates": {"type": "integer"},
            "by_confidence": {
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/components/schemas/AnalyticsOutcomeConfidenceBucket"
                },
            },
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsLeaderboardEntry": {
        "type": "object",
        "description": "Agent leaderboard entry",
        "properties": {
            "rank": {"type": "integer"},
            "agent_name": {"type": "string"},
            "elo": {"type": "number"},
            "wins": {"type": "integer"},
            "losses": {"type": "integer"},
            "draws": {"type": "integer"},
            "win_rate": {"type": "number"},
            "games_played": {"type": "integer"},
            "calibration_score": {"type": "number"},
        },
    },
    "AnalyticsAgentsLeaderboard": {
        "type": "object",
        "description": "Agent leaderboard response",
        "properties": {
            "leaderboard": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/AnalyticsLeaderboardEntry"},
            },
            "total_agents": {"type": "integer"},
            "domain": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsDomainPerformanceEntry": {
        "type": "object",
        "description": "Performance metrics for a domain",
        "properties": {
            "elo": {"type": "number"},
            "wins": {"type": "integer"},
            "losses": {"type": "integer"},
        },
    },
    "AnalyticsEloHistoryPoint": {
        "type": "object",
        "description": "ELO history data point",
        "properties": {
            "timestamp": {"type": "string"},
            "elo": {"type": "number"},
        },
    },
    "AnalyticsAgentPerformance": {
        "type": "object",
        "description": "Agent performance details",
        "properties": {
            "agent_id": {"type": "string"},
            "agent_name": {"type": "string"},
            "time_range": {"type": "string"},
            "elo": {"type": "number"},
            "elo_change": {"type": "number"},
            "rank": {"type": "integer"},
            "wins": {"type": "integer"},
            "losses": {"type": "integer"},
            "draws": {"type": "integer"},
            "win_rate": {"type": "number"},
            "games_played": {"type": "integer"},
            "debates_count": {"type": "integer"},
            "consensus_contribution_rate": {"type": "number"},
            "calibration_score": {"type": "number"},
            "calibration_accuracy": {"type": "number"},
            "domain_performance": {
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/components/schemas/AnalyticsDomainPerformanceEntry"
                },
            },
            "recent_matches": {"type": "array", "items": {"type": "object"}},
            "elo_history": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/AnalyticsEloHistoryPoint"},
            },
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsAgentComparisonEntry": {
        "type": "object",
        "description": "Agent comparison entry",
        "properties": {
            "agent_name": {"type": "string"},
            "elo": {"type": "number"},
            "wins": {"type": "integer"},
            "losses": {"type": "integer"},
            "draws": {"type": "integer"},
            "win_rate": {"type": "number"},
            "games_played": {"type": "integer"},
            "calibration_score": {"type": "number"},
            "error": {"type": "string"},
        },
    },
    "AnalyticsAgentsComparison": {
        "type": "object",
        "description": "Agent comparison response",
        "properties": {
            "agents": {"type": "array", "items": {"type": "string"}},
            "comparison": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/AnalyticsAgentComparisonEntry"},
            },
            "head_to_head": {"type": "object", "additionalProperties": {"type": "object"}},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsAgentTrendPoint": {
        "type": "object",
        "description": "Agent trend data point",
        "properties": {
            "period": {"type": "string"},
            "elo": {"type": "number"},
            "games": {"type": "integer"},
        },
    },
    "AnalyticsAgentsTrends": {
        "type": "object",
        "description": "Agent trend series over time",
        "properties": {
            "agents": {"type": "array", "items": {"type": "string"}},
            "time_range": {"type": "string"},
            "granularity": {"type": "string"},
            "trends": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/AnalyticsAgentTrendPoint"},
                },
            },
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsTokenSummary": {
        "type": "object",
        "description": "Token usage summary",
        "properties": {
            "total_tokens_in": {"type": "number"},
            "total_tokens_out": {"type": "number"},
            "total_tokens": {"type": "number"},
            "avg_tokens_per_day": {"type": "number"},
        },
    },
    "AnalyticsUsageTokens": {
        "type": "object",
        "description": "Token usage response",
        "properties": {
            "org_id": {"type": "string"},
            "time_range": {"type": "string"},
            "granularity": {"type": "string"},
            "summary": {"$ref": "#/components/schemas/AnalyticsTokenSummary"},
            "trends": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "period": {"type": "string"},
                        "tokens_in": {"type": "number"},
                        "tokens_out": {"type": "number"},
                    },
                },
            },
            "by_agent": {"type": "object", "additionalProperties": {"type": "number"}},
            "by_model": {"type": "object", "additionalProperties": {"type": "number"}},
            "message": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsCostSummary": {
        "type": "object",
        "description": "Cost usage summary",
        "properties": {
            "total_cost_usd": {"type": "string"},
            "avg_cost_per_day": {"type": "string"},
            "avg_cost_per_debate": {"type": "string"},
            "total_api_calls": {"type": "integer"},
        },
    },
    "AnalyticsUsageCosts": {
        "type": "object",
        "description": "Cost usage response",
        "properties": {
            "org_id": {"type": "string"},
            "time_range": {"type": "string"},
            "summary": {"$ref": "#/components/schemas/AnalyticsCostSummary"},
            "by_provider": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "cost": {"type": "string"},
                        "percentage": {"type": "number"},
                    },
                },
            },
            "by_model": {"type": "object", "additionalProperties": {"type": "object"}},
            "message": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
    "AnalyticsActiveUserCounts": {
        "type": "object",
        "description": "Active user counts",
        "properties": {
            "daily": {"type": "integer"},
            "weekly": {"type": "integer"},
            "monthly": {"type": "integer"},
        },
    },
    "AnalyticsUserGrowth": {
        "type": "object",
        "description": "User growth statistics",
        "properties": {
            "new_users": {"type": "integer"},
            "churned_users": {"type": "integer"},
            "net_growth": {"type": "integer"},
        },
    },
    "AnalyticsActivityDistribution": {
        "type": "object",
        "description": "User activity distribution",
        "properties": {
            "power_users": {"type": "integer"},
            "regular_users": {"type": "integer"},
            "occasional_users": {"type": "integer"},
        },
    },
    "AnalyticsActiveUsers": {
        "type": "object",
        "description": "Active users response",
        "properties": {
            "org_id": {"type": "string"},
            "time_range": {"type": "string"},
            "active_users": {"$ref": "#/components/schemas/AnalyticsActiveUserCounts"},
            "user_growth": {"$ref": "#/components/schemas/AnalyticsUserGrowth"},
            "activity_distribution": {"$ref": "#/components/schemas/AnalyticsActivityDistribution"},
            "message": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
        },
    },
}


__all__ = ["ANALYTICS_SCHEMAS"]
