"""
Debate OpenAPI Schema Definitions.

Schemas for debate creation, status, consensus, and related entities.
"""

from typing import Any

DEBATE_SCHEMAS: dict[str, Any] = {
    "DebateStatus": {
        "type": "string",
        "enum": [
            "created",
            "starting",
            "pending",
            "running",
            "in_progress",
            "completed",
            "failed",
            "cancelled",
            "paused",
            "active",
            "concluded",
            "archived",
        ],
    },
    "ConsensusResult": {
        "type": "object",
        "properties": {
            "reached": {"type": "boolean"},
            "agreement": {"type": "number"},
            "confidence": {"type": "number"},
            "final_answer": {"type": "string"},
            "conclusion": {"type": "string"},
            "supporting_agents": {"type": "array", "items": {"type": "string"}},
            "dissenting_agents": {"type": "array", "items": {"type": "string"}},
        },
    },
    "DebateCreateRequest": {
        "type": "object",
        "description": "Request body for creating a new debate",
        "properties": {
            "task": {
                "type": "string",
                "description": "The topic or question for the debate",
                "example": "Should we adopt microservices architecture for our e-commerce platform?",
                "minLength": 10,
                "maxLength": 2000,
            },
            "question": {
                "type": "string",
                "description": "Alias for task (deprecated, use task instead)",
                "deprecated": True,
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of agent names to participate. If empty, auto_select is used.",
                "example": ["claude", "gpt-4", "gemini"],
                "minItems": 2,
                "maxItems": 8,
            },
            "rounds": {
                "type": "integer",
                "description": "Maximum number of debate rounds",
                "default": 9,
                "minimum": 1,
                "maximum": 12,
                "example": 9,
            },
            "consensus": {
                "type": "string",
                "description": "Consensus strategy to use",
                "enum": [
                    "majority",
                    "unanimous",
                    "supermajority",
                    "weighted",
                    "hybrid",
                    "judge",
                    "none",
                ],
                "default": "judge",
                "example": "judge",
            },
            "context": {
                "type": "string",
                "description": "Additional context or background information",
                "example": "We have 1M daily active users and need 99.9% uptime.",
                "maxLength": 10000,
            },
            "auto_select": {
                "type": "boolean",
                "description": "Automatically select optimal agents based on topic",
                "default": True,
            },
            "auto_select_config": {
                "type": "object",
                "description": "Configuration for auto-selection algorithm",
                "properties": {
                    "min_agents": {"type": "integer", "default": 3},
                    "max_agents": {"type": "integer", "default": 5},
                    "diversity_weight": {"type": "number", "default": 0.3},
                    "expertise_weight": {"type": "number", "default": 0.7},
                },
            },
            "use_trending": {
                "type": "boolean",
                "description": "Include trending context from news/social media",
                "default": False,
            },
            "trending_category": {
                "type": "string",
                "description": "Category filter for trending content",
                "enum": ["tech", "science", "politics", "business", "health"],
            },
        },
        "required": ["task"],
        "example": {
            "task": "Should we adopt microservices architecture for our e-commerce platform?",
            "agents": ["claude", "gpt-4", "gemini"],
            "rounds": 9,
            "consensus": "judge",
            "context": "We have 1M daily active users and need 99.9% uptime.",
        },
    },
    "DebateCreateResponse": {
        "type": "object",
        "description": "Response when a debate is successfully created",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "Whether the debate was created successfully",
                "example": True,
            },
            "debate_id": {
                "type": "string",
                "description": "Unique identifier for the created debate",
                "example": "deb_abc123xyz",
            },
            "status": {
                "$ref": "#/components/schemas/DebateStatus",
                "description": "Current status of the debate",
            },
            "task": {
                "type": "string",
                "description": "The debate topic (echoed back)",
                "example": "Should we adopt microservices architecture?",
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agents participating in the debate",
                "example": ["claude", "gpt-4", "gemini"],
            },
            "websocket_url": {
                "type": "string",
                "description": "WebSocket URL to stream debate progress",
                "example": "wss://api.aragora.ai/ws/debates/deb_abc123xyz",
            },
            "estimated_duration": {
                "type": "integer",
                "description": "Estimated debate duration in seconds",
                "example": 120,
            },
            "error": {
                "type": "string",
                "description": "Error message if success is false",
            },
        },
        "required": ["success"],
        "example": {
            "success": True,
            "debate_id": "deb_abc123xyz",
            "status": "running",
            "task": "Should we adopt microservices architecture?",
            "agents": ["claude", "gpt-4", "gemini"],
            "websocket_url": "wss://api.aragora.ai/ws/debates/deb_abc123xyz",
            "estimated_duration": 120,
        },
    },
    "Debate": {
        "type": "object",
        "properties": {
            "debate_id": {"type": "string"},
            "id": {"type": "string"},
            "slug": {"type": "string"},
            "task": {"type": "string"},
            "topic": {"type": "string", "description": "Alias for task"},
            "context": {"type": "string"},
            "status": {"$ref": "#/components/schemas/DebateStatus"},
            "outcome": {"type": "string"},
            "final_answer": {"type": "string"},
            "consensus": {"$ref": "#/components/schemas/ConsensusResult"},
            "consensus_proof": {"type": "object"},
            "consensus_reached": {"type": "boolean"},
            "confidence": {"type": "number"},
            "rounds_used": {"type": "integer"},
            "duration_seconds": {"type": "number"},
            "agents": {"type": "array", "items": {"type": "string"}},
            "rounds": {"type": "array", "items": {"$ref": "#/components/schemas/Round"}},
            "created_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "metadata": {"type": "object"},
        },
    },
    "Message": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant"]},
            "content": {"type": "string"},
            "agent": {"type": "string"},
            "agent_id": {"type": "string"},
            "round": {"type": "integer"},
            "timestamp": {"type": "string", "format": "date-time"},
        },
    },
    "Round": {
        "type": "object",
        "properties": {
            "round_number": {"type": "integer", "description": "Round number (1-indexed)"},
            "messages": {"type": "array", "items": {"$ref": "#/components/schemas/Message"}},
            "votes": {"type": "object", "description": "Agent votes for this round"},
            "summary": {"type": "string", "description": "Round summary"},
        },
    },
    "Consensus": {
        "type": "object",
        "properties": {
            "reached": {"type": "boolean"},
            "topic": {"type": "string"},
            "verdict": {"type": "string"},
            "confidence": {"type": "number"},
            "participating_agents": {"type": "array", "items": {"type": "string"}},
        },
    },
    "SimilarDebate": {
        "type": "object",
        "description": "A debate similar to the query topic",
        "properties": {
            "debate_id": {"type": "string"},
            "topic": {"type": "string"},
            "similarity_score": {"type": "number", "description": "0-1 similarity"},
            "verdict": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"},
            "agents": {"type": "array", "items": {"type": "string"}},
        },
    },
    "SimilarDebatesResponse": {
        "type": "object",
        "properties": {
            "debates": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/SimilarDebate"},
            },
            "query": {"type": "string"},
            "total": {"type": "integer"},
        },
    },
    "SettledQuestion": {
        "type": "object",
        "description": "A question with strong consensus",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "debate_count": {"type": "integer"},
            "last_debated": {"type": "string", "format": "date-time"},
            "supporting_debates": {"type": "array", "items": {"type": "string"}},
        },
    },
    "SettledQuestionsResponse": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/SettledQuestion"},
            },
            "total": {"type": "integer"},
            "threshold": {"type": "number"},
        },
    },
    "ConsensusStats": {
        "type": "object",
        "description": "Aggregate consensus statistics",
        "properties": {
            "total_debates": {"type": "integer"},
            "consensus_rate": {"type": "number"},
            "avg_time_to_consensus_ms": {"type": "integer"},
            "avg_rounds_to_consensus": {"type": "number"},
            "by_domain": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "consensus_rate": {"type": "number"},
                    },
                },
            },
        },
    },
    "DissentingView": {
        "type": "object",
        "description": "A significant dissenting position",
        "properties": {
            "debate_id": {"type": "string"},
            "agent": {"type": "string"},
            "position": {"type": "string"},
            "reasoning": {"type": "string"},
            "strength_score": {"type": "number"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    },
    "DissentingViewsResponse": {
        "type": "object",
        "properties": {
            "dissents": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DissentingView"},
            },
            "total": {"type": "integer"},
        },
    },
    "ContrarianView": {
        "type": "object",
        "description": "A view opposing established consensus",
        "properties": {
            "topic": {"type": "string"},
            "consensus_position": {"type": "string"},
            "contrarian_position": {"type": "string"},
            "agent": {"type": "string"},
            "argument_strength": {"type": "number"},
        },
    },
    "ContrarianViewsResponse": {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ContrarianView"},
            },
            "total": {"type": "integer"},
        },
    },
    "RiskWarning": {
        "type": "object",
        "description": "A consensus risk warning",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["low_confidence", "shifting", "contradictory", "bias"],
            },
            "topic": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "debate_ids": {"type": "array", "items": {"type": "string"}},
        },
    },
    "RiskWarningsResponse": {
        "type": "object",
        "properties": {
            "warnings": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/RiskWarning"},
            },
            "total": {"type": "integer"},
        },
    },
    "DomainConsensusResponse": {
        "type": "object",
        "description": "Consensus data for a specific domain",
        "properties": {
            "domain": {"type": "string"},
            "total_debates": {"type": "integer"},
            "consensus_rate": {"type": "number"},
            "settled_questions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/SettledQuestion"},
            },
            "top_agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "contribution_count": {"type": "integer"},
                    },
                },
            },
        },
    },
    "DebateSummary": {
        "type": "object",
        "description": "Summary of a debate decision",
        "properties": {
            "debate_id": {"type": "string"},
            "summary": {"type": "string"},
            "confidence": {"type": "number"},
            "consensus_reached": {"type": "boolean"},
        },
        "required": ["debate_id", "summary"],
    },
}


__all__ = ["DEBATE_SCHEMAS"]
