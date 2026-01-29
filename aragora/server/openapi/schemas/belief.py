"""
Belief Network OpenAPI Schema Definitions.

Schemas for belief networks, cruxes, and agent relationships.
"""

from typing import Any

BELIEF_SCHEMAS: dict[str, Any] = {
    "BeliefCrux": {
        "type": "object",
        "description": "A crux point of disagreement between agents",
        "properties": {
            "id": {"type": "string"},
            "proposition": {"type": "string", "description": "The crux proposition"},
            "importance": {"type": "number", "description": "Importance score 0-1"},
            "agents_for": {"type": "array", "items": {"type": "string"}},
            "agents_against": {"type": "array", "items": {"type": "string"}},
            "resolution_impact": {
                "type": "string",
                "description": "How resolving this crux would affect the debate",
            },
        },
    },
    "BeliefCruxesResponse": {
        "type": "object",
        "properties": {
            "debate_id": {"type": "string"},
            "cruxes": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/BeliefCrux"},
            },
            "total": {"type": "integer"},
        },
    },
    "LoadBearingClaim": {
        "type": "object",
        "description": "A claim foundational to the argument structure",
        "properties": {
            "id": {"type": "string"},
            "claim": {"type": "string"},
            "agent": {"type": "string"},
            "dependents_count": {
                "type": "integer",
                "description": "Number of arguments that depend on this claim",
            },
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
    },
    "LoadBearingClaimsResponse": {
        "type": "object",
        "properties": {
            "debate_id": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/LoadBearingClaim"},
            },
            "total": {"type": "integer"},
        },
    },
    "BeliefGraphStats": {
        "type": "object",
        "description": "Graph-based statistics for a belief network",
        "properties": {
            "debate_id": {"type": "string"},
            "node_count": {"type": "integer"},
            "edge_count": {"type": "integer"},
            "max_depth": {"type": "integer"},
            "clustering_coefficient": {"type": "number"},
            "most_connected_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "connections": {"type": "integer"},
                    },
                },
            },
        },
    },
    "Calibration": {
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "score": {"type": "number", "description": "Calibration score (0-1)"},
            "bucket_stats": {"type": "array", "items": {"type": "object"}},
            "overconfidence_index": {"type": "number"},
        },
    },
    "Relationship": {
        "type": "object",
        "properties": {
            "agent_a": {"type": "string"},
            "agent_b": {"type": "string"},
            "alliance_score": {"type": "number"},
            "rivalry_score": {"type": "number"},
            "total_interactions": {"type": "integer"},
        },
    },
}


__all__ = ["BELIEF_SCHEMAS"]
