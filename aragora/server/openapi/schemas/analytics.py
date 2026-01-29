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
}


__all__ = ["ANALYTICS_SCHEMAS"]
