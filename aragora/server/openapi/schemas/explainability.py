"""
Explainability OpenAPI Schema Definitions.

Schemas for decision explanations, evidence chains, and counterfactuals.
"""

from typing import Any

EXPLAINABILITY_SCHEMAS: dict[str, Any] = {
    "DecisionExplanation": {
        "type": "object",
        "description": "Full explanation of a debate decision",
        "properties": {
            "debate_id": {"type": "string", "description": "Debate ID"},
            "narrative": {"type": "string", "description": "Natural language narrative"},
            "confidence": {"type": "number", "description": "Overall confidence (0-1)"},
            "factors": {
                "type": "array",
                "description": "Contributing factors",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "contribution": {"type": "number"},
                        "description": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            },
            "counterfactuals": {
                "type": "array",
                "description": "What-if scenarios",
                "items": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "outcome": {"type": "string"},
                        "probability": {"type": "number"},
                    },
                },
            },
            "provenance": {
                "type": "array",
                "description": "Decision provenance chain",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "action": {"type": "string"},
                        "agent": {"type": "string"},
                        "confidence": {"type": "number"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
            },
            "generated_at": {"type": "string", "format": "date-time"},
        },
        "required": ["debate_id", "narrative"],
    },
    "EvidenceLink": {
        "type": "object",
        "description": "Evidence link supporting a decision",
        "properties": {
            "id": {"type": "string"},
            "content": {"type": "string"},
            "source": {"type": "string"},
            "relevance_score": {"type": "number"},
            "quality_scores": {"type": "object"},
            "cited_by": {"type": "array", "items": {"type": "string"}},
            "grounding_type": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time", "nullable": True},
            "metadata": {"type": "object"},
        },
        "required": ["id", "content", "source", "relevance_score"],
    },
    "EvidenceChain": {
        "type": "object",
        "description": "Evidence chain response for a debate",
        "properties": {
            "debate_id": {"type": "string"},
            "evidence_count": {"type": "integer"},
            "evidence_quality_score": {"type": "number"},
            "evidence": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/EvidenceLink"},
            },
        },
        "required": ["debate_id", "evidence_count", "evidence"],
    },
    "VotePivot": {
        "type": "object",
        "description": "Vote pivot that influenced a decision",
        "properties": {
            "agent": {"type": "string"},
            "choice": {"type": "string"},
            "confidence": {"type": "number"},
            "weight": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "influence_score": {"type": "number"},
            "calibration_adjustment": {"type": "number", "nullable": True},
            "elo_rating": {"type": "number", "nullable": True},
            "flip_detected": {"type": "boolean"},
            "metadata": {"type": "object"},
        },
        "required": ["agent", "choice", "confidence", "weight", "reasoning_summary"],
    },
    "VotePivots": {
        "type": "object",
        "description": "Vote pivot analysis for a debate",
        "properties": {
            "debate_id": {"type": "string"},
            "total_votes": {"type": "integer"},
            "pivotal_votes": {"type": "integer"},
            "agent_agreement_score": {"type": "number"},
            "votes": {"type": "array", "items": {"$ref": "#/components/schemas/VotePivot"}},
        },
        "required": ["debate_id", "votes"],
    },
    "Counterfactual": {
        "type": "object",
        "description": "Counterfactual scenario for a debate decision",
        "properties": {
            "condition": {"type": "string"},
            "outcome_change": {"type": "string"},
            "likelihood": {"type": "number"},
            "sensitivity": {"type": "number"},
            "affected_agents": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
        "required": ["condition", "outcome_change", "likelihood", "sensitivity"],
    },
    "Counterfactuals": {
        "type": "object",
        "description": "Counterfactual analysis response",
        "properties": {
            "debate_id": {"type": "string"},
            "counterfactual_count": {"type": "integer"},
            "counterfactuals": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Counterfactual"},
            },
        },
        "required": ["debate_id", "counterfactuals"],
    },
    "ExplainabilityBatch": {
        "type": "object",
        "description": "Batch explainability job creation response",
        "properties": {
            "batch_id": {"type": "string"},
            "status": {"type": "string"},
            "total_debates": {"type": "integer"},
            "status_url": {"type": "string"},
            "results_url": {"type": "string"},
        },
        "required": ["batch_id", "status", "total_debates"],
    },
    "ExplainabilityBatchStatus": {
        "type": "object",
        "description": "Batch explainability job status",
        "properties": {
            "batch_id": {"type": "string"},
            "status": {"type": "string"},
            "total_debates": {"type": "integer"},
            "processed_count": {"type": "integer"},
            "success_count": {"type": "integer"},
            "error_count": {"type": "integer"},
            "created_at": {"type": "number"},
            "started_at": {"type": "number", "nullable": True},
            "completed_at": {"type": "number", "nullable": True},
            "progress_pct": {"type": "number"},
        },
        "required": ["batch_id", "status", "total_debates"],
    },
    "BatchDebateResult": {
        "type": "object",
        "description": "Result for a single debate in a batch",
        "properties": {
            "debate_id": {"type": "string"},
            "status": {"type": "string"},
            "processing_time_ms": {"type": "number"},
            "explanation": {"type": "object"},
            "error": {"type": "string"},
        },
        "required": ["debate_id", "status"],
    },
    "ExplainabilityBatchResults": {
        "type": "object",
        "description": "Batch explainability results with pagination",
        "properties": {
            "batch_id": {"type": "string"},
            "status": {"type": "string"},
            "total_debates": {"type": "integer"},
            "processed_count": {"type": "integer"},
            "success_count": {"type": "integer"},
            "error_count": {"type": "integer"},
            "created_at": {"type": "number"},
            "started_at": {"type": "number", "nullable": True},
            "completed_at": {"type": "number", "nullable": True},
            "progress_pct": {"type": "number"},
            "results": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BatchDebateResult"},
            },
            "pagination": {
                "type": "object",
                "properties": {
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "total": {"type": "integer"},
                    "has_more": {"type": "boolean"},
                },
            },
        },
        "required": ["batch_id", "status", "total_debates", "results"],
    },
}


__all__ = ["EXPLAINABILITY_SCHEMAS"]
