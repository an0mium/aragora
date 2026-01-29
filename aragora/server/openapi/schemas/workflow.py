"""
Workflow OpenAPI Schema Definitions.

Schemas for workflow definitions, templates, checkpoints, and receipts.
"""

from typing import Any

WORKFLOW_SCHEMAS: dict[str, Any] = {
    "StepDefinition": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "type": {"type": "string"},
            "config": {"type": "object"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
    },
    "TransitionRule": {
        "type": "object",
        "properties": {
            "from_step": {"type": "string"},
            "to_step": {"type": "string"},
            "condition": {"type": "string"},
        },
    },
    "Workflow": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "status": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "transitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/TransitionRule"},
            },
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
        },
    },
    "WorkflowList": {
        "type": "object",
        "properties": {
            "workflows": {"type": "array", "items": {"$ref": "#/components/schemas/Workflow"}},
            "total": {"type": "integer"},
        },
    },
    "WorkflowUpdate": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "transitions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/TransitionRule"},
            },
        },
    },
    "WorkflowTemplate": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "steps": {"type": "array", "items": {"$ref": "#/components/schemas/StepDefinition"}},
            "parameters": {"type": "object"},
        },
    },
    "WorkflowTemplateList": {
        "type": "object",
        "properties": {
            "templates": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/WorkflowTemplate"},
            },
            "total": {"type": "integer"},
        },
    },
    "ExecutionList": {
        "type": "object",
        "properties": {
            "executions": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
        },
    },
    "DecisionReceipt": {
        "type": "object",
        "description": "A decision receipt documenting a gauntlet outcome",
        "properties": {
            "id": {"type": "string", "description": "Receipt ID"},
            "debate_id": {"type": "string", "description": "Associated debate ID"},
            "verdict": {"type": "string", "description": "Final verdict"},
            "confidence": {"type": "number", "description": "Confidence score (0-1)"},
            "consensus_reached": {"type": "boolean"},
            "participating_agents": {
                "type": "array",
                "items": {"type": "string"},
            },
            "dissenting_agents": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence": {"type": "array", "items": {"type": "object"}},
            "reasoning": {"type": "string"},
            "hash": {"type": "string", "description": "Receipt integrity hash"},
            "created_at": {"type": "string", "format": "date-time"},
            "metadata": {"type": "object"},
        },
        "required": ["id", "debate_id", "verdict"],
    },
    "ReceiptList": {
        "type": "object",
        "description": "Paginated list of decision receipts",
        "properties": {
            "receipts": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DecisionReceipt"},
            },
            "total": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
            "has_more": {"type": "boolean"},
        },
        "required": ["receipts", "total"],
    },
    "RiskHeatmap": {
        "type": "object",
        "description": "Risk heatmap visualization data",
        "properties": {
            "id": {"type": "string", "description": "Heatmap ID"},
            "gauntlet_id": {"type": "string", "description": "Associated gauntlet ID"},
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Risk categories",
            },
            "scores": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Risk scores per category",
            },
            "matrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "description": "2D risk matrix",
            },
            "overall_risk": {"type": "number", "description": "Overall risk score"},
            "created_at": {"type": "string", "format": "date-time"},
            "metadata": {"type": "object"},
        },
        "required": ["id", "categories", "scores"],
    },
    "HeatmapList": {
        "type": "object",
        "description": "Paginated list of risk heatmaps",
        "properties": {
            "heatmaps": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/RiskHeatmap"},
            },
            "total": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["heatmaps", "total"],
    },
    "PatternTemplate": {
        "type": "object",
        "description": "A workflow pattern template (e.g., Hive Mind, MapReduce)",
        "properties": {
            "id": {"type": "string", "description": "Pattern template ID"},
            "name": {"type": "string", "description": "Pattern name"},
            "description": {"type": "string", "description": "Pattern description"},
            "pattern_type": {
                "type": "string",
                "enum": ["hive_mind", "map_reduce", "review_cycle"],
            },
            "parameters": {"type": "object", "description": "Pattern parameters schema"},
            "example_config": {"type": "object", "description": "Example configuration"},
            "created_at": {"type": "string", "format": "date-time"},
        },
        "required": ["id", "name", "pattern_type"],
    },
    "PatternTemplateList": {
        "type": "object",
        "description": "Paginated list of pattern templates",
        "properties": {
            "templates": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/PatternTemplate"},
            },
            "total": {"type": "integer"},
        },
        "required": ["templates", "total"],
    },
    "CheckpointMetadata": {
        "type": "object",
        "description": "Metadata for a debate or workflow checkpoint",
        "properties": {
            "id": {"type": "string", "description": "Checkpoint ID"},
            "debate_id": {"type": "string", "description": "Associated debate ID"},
            "workflow_id": {"type": "string", "description": "Associated workflow ID"},
            "name": {"type": "string", "description": "Checkpoint name"},
            "description": {"type": "string"},
            "state": {"type": "object", "description": "Checkpoint state data"},
            "round_number": {"type": "integer"},
            "created_at": {"type": "string", "format": "date-time"},
            "created_by": {"type": "string"},
            "size_bytes": {"type": "integer"},
        },
        "required": ["id", "created_at"],
    },
    "CheckpointList": {
        "type": "object",
        "description": "Paginated list of checkpoints",
        "properties": {
            "checkpoints": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CheckpointMetadata"},
            },
            "total": {"type": "integer"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["checkpoints", "total"],
    },
    "RestoreResult": {
        "type": "object",
        "description": "Result of a checkpoint restore operation",
        "properties": {
            "success": {"type": "boolean"},
            "checkpoint_id": {"type": "string"},
            "debate_id": {"type": "string"},
            "workflow_id": {"type": "string"},
            "restored_at": {"type": "string", "format": "date-time"},
            "state_restored": {"type": "boolean"},
            "message": {"type": "string"},
        },
        "required": ["success", "checkpoint_id"],
    },
}


__all__ = ["WORKFLOW_SCHEMAS"]
