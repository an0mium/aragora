"""
OpenAPI Schema Definitions.

Contains common schema components used across all API endpoints.
"""

from typing import Any

# =============================================================================
# Common Schema Components
# =============================================================================

COMMON_SCHEMAS: dict[str, Any] = {
    "Error": {
        "type": "object",
        "description": "Standard error response format",
        "properties": {
            "error": {
                "type": "string",
                "description": "Human-readable error message",
                "example": "Invalid request: missing required field 'task'",
            },
            "code": {
                "type": "string",
                "description": "Machine-readable error code for programmatic handling",
                "enum": [
                    "INVALID_JSON",
                    "MISSING_FIELD",
                    "INVALID_VALUE",
                    "AUTH_REQUIRED",
                    "INVALID_TOKEN",
                    "FORBIDDEN",
                    "NOT_OWNER",
                    "NOT_FOUND",
                    "QUOTA_EXCEEDED",
                    "RATE_LIMITED",
                    "INTERNAL_ERROR",
                    "SERVICE_UNAVAILABLE",
                    "AGENT_TIMEOUT",
                    "CONSENSUS_FAILED",
                ],
                "example": "MISSING_FIELD",
            },
            "trace_id": {
                "type": "string",
                "description": "Unique request ID for debugging and support",
                "example": "req_abc123xyz789",
            },
            "field": {
                "type": "string",
                "description": "Name of the field that caused the error (for validation errors)",
                "example": "task",
            },
            "resource_type": {
                "type": "string",
                "description": "Type of resource involved in the error",
                "example": "debate",
            },
            "resource_id": {
                "type": "string",
                "description": "ID of the resource involved in the error",
                "example": "deb_abc123",
            },
            "limit": {
                "type": "integer",
                "description": "The limit that was exceeded (for quota/rate errors)",
                "example": 60,
            },
            "retry_after": {
                "type": "integer",
                "description": "Seconds to wait before retrying (for rate limit errors)",
                "example": 45,
            },
            "resets_at": {
                "type": "string",
                "format": "date-time",
                "description": "When the quota/rate limit resets",
                "example": "2024-01-16T00:00:00Z",
            },
            "upgrade_url": {
                "type": "string",
                "format": "uri",
                "description": "URL to upgrade plan (for quota errors)",
                "example": "https://aragora.ai/pricing",
            },
            "support_url": {
                "type": "string",
                "format": "uri",
                "description": "URL for support/issue reporting",
                "example": "https://github.com/anthropics/aragora/issues",
            },
        },
        "required": ["error"],
    },
    "PaginatedResponse": {
        "type": "object",
        "properties": {
            "total": {"type": "integer", "description": "Total items available"},
            "offset": {"type": "integer", "description": "Current offset"},
            "limit": {"type": "integer", "description": "Page size"},
            "has_more": {"type": "boolean", "description": "More items available"},
        },
    },
    "Agent": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Agent name"},
            "elo": {"type": "number", "description": "ELO rating"},
            "matches": {"type": "integer", "description": "Total matches played"},
            "wins": {"type": "integer", "description": "Total wins"},
            "losses": {"type": "integer", "description": "Total losses"},
            "calibration_score": {"type": "number", "description": "Calibration accuracy (0-1)"},
        },
    },
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
                "default": 3,
                "minimum": 1,
                "maximum": 10,
                "example": 3,
            },
            "consensus": {
                "type": "string",
                "description": "Consensus strategy to use",
                "enum": ["majority", "unanimous", "weighted", "semantic"],
                "default": "majority",
                "example": "majority",
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
            "rounds": 3,
            "consensus": "majority",
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
    "HealthCheck": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
            "version": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "checks": {"type": "object", "additionalProperties": {"type": "object"}},
            "response_time_ms": {"type": "number"},
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
    "OAuthProvider": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Provider identifier (e.g., 'google', 'github')",
            },
            "name": {"type": "string", "description": "Display name for the provider"},
        },
        "required": ["id", "name"],
    },
    "OAuthProviders": {
        "type": "object",
        "properties": {
            "providers": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/OAuthProvider"},
                "description": "List of available OAuth providers",
            },
        },
        "required": ["providers"],
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
    "Workspace": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Workspace ID"},
            "organization_id": {"type": "string", "description": "Parent organization ID"},
            "name": {"type": "string", "description": "Workspace name"},
            "created_at": {"type": "string", "format": "date-time"},
            "created_by": {"type": "string"},
            "encrypted": {"type": "boolean", "description": "Whether workspace data is encrypted"},
            "retention_days": {"type": "integer", "description": "Data retention period in days"},
            "sensitivity_level": {"type": "string", "description": "Data sensitivity level"},
            "document_count": {"type": "integer"},
            "storage_bytes": {"type": "integer"},
        },
        "required": ["id", "organization_id", "name"],
    },
    "WorkspaceList": {
        "type": "object",
        "properties": {
            "workspaces": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Workspace"},
            },
            "total": {"type": "integer"},
        },
        "required": ["workspaces", "total"],
    },
    # Retention Policies
    "RetentionPolicy": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "retention_days": {"type": "integer"},
            "data_types": {"type": "array", "items": {"type": "string"}},
            "enabled": {"type": "boolean"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    },
    "RetentionPolicyList": {
        "type": "object",
        "properties": {
            "policies": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/RetentionPolicy"},
            },
            "total": {"type": "integer"},
        },
    },
    # Workflow schemas
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
    # Gauntlet/Receipt schemas
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
    # Pattern template schemas
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
    # Checkpoint schemas
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
    # Explainability schemas
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
    # Control plane schemas
    "ControlPlaneAgent": {
        "type": "object",
        "description": "Control plane agent record",
        "properties": {
            "agent_id": {"type": "string"},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string"},
            "model": {"type": "string"},
            "provider": {"type": "string"},
            "metadata": {"type": "object"},
            "registered_at": {"type": "number"},
            "last_heartbeat": {"type": "number"},
            "current_task_id": {"type": "string", "nullable": True},
            "tasks_completed": {"type": "integer"},
            "tasks_failed": {"type": "integer"},
            "avg_latency_ms": {"type": "number"},
            "region_id": {"type": "string"},
            "available_regions": {"type": "array", "items": {"type": "string"}},
            "region_latency_ms": {"type": "object"},
            "last_heartbeat_by_region": {"type": "object"},
        },
        "required": ["agent_id", "capabilities", "status"],
    },
    "ControlPlaneAgentList": {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ControlPlaneAgent"},
            },
            "total": {"type": "integer"},
        },
        "required": ["agents", "total"],
    },
    "ControlPlaneTask": {
        "type": "object",
        "description": "Control plane task record",
        "properties": {
            "id": {"type": "string"},
            "task_type": {"type": "string"},
            "payload": {"type": "object"},
            "required_capabilities": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string"},
            "priority": {"type": "string"},
            "created_at": {"type": "number"},
            "assigned_at": {"type": "number", "nullable": True},
            "started_at": {"type": "number", "nullable": True},
            "completed_at": {"type": "number", "nullable": True},
            "assigned_agent": {"type": "string", "nullable": True},
            "timeout_seconds": {"type": "number", "nullable": True},
            "max_retries": {"type": "integer"},
            "retries": {"type": "integer"},
            "result": {"type": "object", "nullable": True},
            "error": {"type": "string", "nullable": True},
            "metadata": {"type": "object"},
            "target_region": {"type": "string", "nullable": True},
            "fallback_regions": {"type": "array", "items": {"type": "string"}},
            "assigned_region": {"type": "string", "nullable": True},
            "region_routing_mode": {"type": "string"},
            "origin_region": {"type": "string", "nullable": True},
        },
        "required": ["id", "task_type", "status"],
    },
    "ControlPlaneTaskCreated": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
    "ControlPlaneTaskClaimResponse": {
        "type": "object",
        "properties": {
            "task": {
                "oneOf": [
                    {"$ref": "#/components/schemas/ControlPlaneTask"},
                    {"type": "null"},
                ]
            }
        },
    },
    "ControlPlaneQueueJob": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "name": {"type": "string"},
            "status": {"type": "string"},
            "progress": {"type": "number"},
            "started_at": {"type": "string", "nullable": True},
            "created_at": {"type": "string", "nullable": True},
            "document_count": {"type": "integer"},
            "agents_assigned": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "string"},
        },
    },
    "ControlPlaneQueue": {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ControlPlaneQueueJob"},
            },
            "total": {"type": "integer"},
        },
        "required": ["jobs", "total"],
    },
    "ControlPlaneMetrics": {
        "type": "object",
        "properties": {
            "active_jobs": {"type": "integer"},
            "queued_jobs": {"type": "integer"},
            "completed_jobs": {"type": "integer"},
            "agents_available": {"type": "integer"},
            "agents_busy": {"type": "integer"},
            "total_agents": {"type": "integer"},
            "documents_processed_today": {"type": "integer"},
            "audits_completed_today": {"type": "integer"},
            "tokens_used_today": {"type": "integer"},
        },
    },
    "ControlPlaneStats": {
        "type": "object",
        "description": "Scheduler and registry stats",
        "properties": {
            "scheduler": {"type": "object"},
            "registry": {"type": "object"},
        },
    },
    "ControlPlaneHealth": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "agents": {"type": "object"},
        },
    },
    # Codebase analysis schemas
    "VulnerabilityReference": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "source": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    },
    "VulnerabilityFinding": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string"},
            "cvss_score": {"type": "number", "nullable": True},
            "package_name": {"type": "string", "nullable": True},
            "package_ecosystem": {"type": "string", "nullable": True},
            "vulnerable_versions": {"type": "array", "items": {"type": "string"}},
            "patched_versions": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string"},
            "references": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityReference"},
            },
            "cwe_ids": {"type": "array", "items": {"type": "string"}},
            "fix_available": {"type": "boolean"},
            "recommended_version": {"type": "string", "nullable": True},
        },
    },
    "DependencyInfo": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "ecosystem": {"type": "string"},
            "direct": {"type": "boolean"},
            "dev_dependency": {"type": "boolean"},
            "license": {"type": "string", "nullable": True},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "has_vulnerabilities": {"type": "boolean"},
            "highest_severity": {"type": "string", "nullable": True},
        },
    },
    "CodebaseScanSummary": {
        "type": "object",
        "properties": {
            "total_dependencies": {"type": "integer"},
            "vulnerable_dependencies": {"type": "integer"},
            "critical_count": {"type": "integer"},
            "high_count": {"type": "integer"},
            "medium_count": {"type": "integer"},
            "low_count": {"type": "integer"},
        },
    },
    "CodebaseScanResult": {
        "type": "object",
        "properties": {
            "scan_id": {"type": "string"},
            "repository": {"type": "string"},
            "branch": {"type": "string", "nullable": True},
            "commit_sha": {"type": "string", "nullable": True},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time", "nullable": True},
            "status": {"type": "string"},
            "error": {"type": "string", "nullable": True},
            "dependencies": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DependencyInfo"},
            },
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "summary": {"$ref": "#/components/schemas/CodebaseScanSummary"},
        },
    },
    "CodebaseScanStartResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_id": {"type": "string"},
            "status": {"type": "string"},
            "repository": {"type": "string"},
        },
    },
    "CodebaseScanResultResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scan_result": {"$ref": "#/components/schemas/CodebaseScanResult"},
        },
    },
    "CodebaseScanListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "scans": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
    },
    "CodebaseVulnerabilityListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "scan_id": {"type": "string"},
        },
    },
    "CodebasePackageVulnerabilityResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "package": {"type": "string"},
            "ecosystem": {"type": "string"},
            "version": {"type": "string", "nullable": True},
            "vulnerabilities": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/VulnerabilityFinding"},
            },
            "total": {"type": "integer"},
        },
    },
    "CodebaseCVEResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "vulnerability": {"$ref": "#/components/schemas/VulnerabilityFinding"},
        },
    },
    "CodebaseMetricsStartResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "analysis_id": {"type": "string"},
            "status": {"type": "string"},
            "repository": {"type": "string"},
        },
    },
    "CodebaseMetricsReportResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "report": {"type": "object"},
        },
    },
    "CodebaseHotspot": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "function_name": {"type": "string", "nullable": True},
            "class_name": {"type": "string", "nullable": True},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "complexity": {"type": "number"},
            "lines_of_code": {"type": "integer"},
            "risk_score": {"type": "number"},
        },
    },
    "CodebaseHotspotListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "hotspots": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/CodebaseHotspot"},
            },
            "total": {"type": "integer"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseDuplicateListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "duplicates": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseFileMetricsResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "file": {"type": "object"},
            "analysis_id": {"type": "string"},
        },
    },
    "CodebaseMetricsHistoryResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "analyses": {"type": "array", "items": {"type": "object"}},
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        },
    },
    # Decision and deliberation schemas
    "DecisionRequest": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "content": {"type": "string"},
            "decision_type": {"type": "string"},
            "source": {"type": "string"},
            "response_channels": {"type": "array", "items": {"type": "object"}},
            "context": {"type": "object"},
            "config": {"type": "object"},
            "priority": {"type": "string"},
            "attachments": {"type": "array", "items": {"type": "object"}},
            "evidence": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["content"],
    },
    "DecisionResult": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "decision_type": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "consensus_reached": {"type": "boolean"},
            "reasoning": {"type": "string", "nullable": True},
            "evidence_used": {"type": "array", "items": {"type": "object"}},
            "agent_contributions": {"type": "array", "items": {"type": "object"}},
            "duration_seconds": {"type": "number"},
            "completed_at": {"type": "string", "format": "date-time"},
            "success": {"type": "boolean"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "decision_type", "answer", "confidence"],
    },
    "DecisionStatus": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    "DecisionSummary": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id"],
    },
    "DecisionList": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/DecisionSummary"},
            },
            "total": {"type": "integer"},
        },
        "required": ["decisions", "total"],
    },
    "DeliberationRequest": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "content": {"type": "string"},
            "decision_type": {"type": "string"},
            "async": {"type": "boolean"},
            "priority": {"type": "string"},
            "timeout_seconds": {"type": "number"},
            "required_capabilities": {"type": "array", "items": {"type": "string"}},
            "response_channels": {"type": "array", "items": {"type": "object"}},
            "metadata": {"type": "object"},
        },
        "required": ["content"],
    },
    "DeliberationQueuedResponse": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "request_id": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["task_id", "request_id", "status"],
    },
    "DeliberationSyncResponse": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "decision_type": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "consensus_reached": {"type": "boolean"},
            "reasoning": {"type": "string", "nullable": True},
            "evidence_used": {"type": "array", "items": {"type": "object"}},
            "duration_seconds": {"type": "number"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    "DeliberationRecord": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "result": {"$ref": "#/components/schemas/DecisionResult"},
            "completed_at": {"type": "string", "format": "date-time"},
            "error": {"type": "string", "nullable": True},
            "metrics": {"type": "object"},
        },
        "required": ["request_id", "status"],
    },
    "DeliberationStatus": {
        "type": "object",
        "properties": {
            "request_id": {"type": "string"},
            "status": {"type": "string"},
            "completed_at": {"type": "string", "nullable": True},
        },
        "required": ["request_id", "status"],
    },
    "GitHubReviewComment": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "file_path": {"type": "string"},
            "line": {"type": "integer"},
            "body": {"type": "string"},
            "side": {"type": "string"},
            "suggestion": {"type": "string", "nullable": True},
            "severity": {"type": "string"},
            "category": {"type": "string"},
        },
        "required": ["id", "file_path", "line", "body"],
    },
    "GitHubPRReviewResult": {
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "pr_number": {"type": "integer"},
            "repository": {"type": "string"},
            "status": {"type": "string"},
            "verdict": {"type": "string", "nullable": True},
            "summary": {"type": "string", "nullable": True},
            "comments": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GitHubReviewComment"},
            },
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time", "nullable": True},
            "error": {"type": "string", "nullable": True},
            "metrics": {"type": "object"},
        },
        "required": ["review_id", "pr_number", "repository", "status", "comments", "started_at"],
    },
    "GitHubPRDetails": {
        "type": "object",
        "properties": {
            "number": {"type": "integer"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "state": {"type": "string"},
            "author": {"type": "string"},
            "base_branch": {"type": "string"},
            "head_branch": {"type": "string"},
            "changed_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "status": {"type": "string"},
                        "additions": {"type": "integer"},
                        "deletions": {"type": "integer"},
                        "patch": {"type": "string"},
                    },
                },
            },
            "commits": {"type": "array", "items": {"type": "object"}},
            "labels": {"type": "array", "items": {"type": "string"}},
            "created_at": {"type": "string", "format": "date-time", "nullable": True},
            "updated_at": {"type": "string", "format": "date-time", "nullable": True},
        },
        "required": ["number", "title", "state", "author", "base_branch", "head_branch"],
    },
    "GitHubPRReviewTriggerRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "owner/repo"},
            "pr_number": {"type": "integer"},
            "review_type": {"type": "string", "enum": ["comprehensive", "quick", "security"]},
            "workspace_id": {"type": "string"},
        },
        "required": ["repository", "pr_number"],
    },
    "GitHubPRReviewTriggerResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "review_id": {"type": "string"},
            "status": {"type": "string"},
            "pr_number": {"type": "integer"},
            "repository": {"type": "string"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRDetailsResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "pr": {"$ref": "#/components/schemas/GitHubPRDetails"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRReviewStatusResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "review": {"$ref": "#/components/schemas/GitHubPRReviewResult"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
    "GitHubPRReviewListResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "reviews": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/GitHubPRReviewResult"},
            },
            "total": {"type": "integer"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success", "reviews", "total"],
    },
    "GitHubPRSubmitReviewRequest": {
        "type": "object",
        "properties": {
            "repository": {"type": "string", "description": "owner/repo"},
            "event": {
                "type": "string",
                "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"],
            },
            "body": {"type": "string"},
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "position": {"type": "integer"},
                        "body": {"type": "string"},
                    },
                    "required": ["path", "position", "body"],
                },
            },
        },
        "required": ["repository", "event"],
    },
    "GitHubPRSubmitReviewResponse": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "demo": {"type": "boolean"},
            "data": {"type": "object"},
            "error": {"type": "string", "nullable": True},
        },
        "required": ["success"],
    },
}


# =============================================================================
# Response Helpers
# =============================================================================


def ok_response(description: str, schema_ref: str | None = None) -> dict:
    """Create a successful response definition."""
    resp: dict = {"description": description}
    if schema_ref:
        resp["content"] = {
            "application/json": {"schema": {"$ref": f"#/components/schemas/{schema_ref}"}}
        }
    return resp


def array_response(description: str, schema_ref: str) -> dict:
    """Create an array response definition."""
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"$ref": f"#/components/schemas/{schema_ref}"},
                        },
                        "total": {"type": "integer"},
                    },
                }
            }
        },
    }


def error_response(status: str, description: str) -> dict:
    """Create an error response definition."""
    return {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }


# Standard error responses used across endpoints
STANDARD_ERRORS = {
    "400": error_response("400", "Bad request"),
    "401": error_response("401", "Unauthorized"),
    "404": error_response("404", "Not found"),
    "402": error_response("402", "Quota exceeded"),
    "429": error_response("429", "Rate limited"),
    "500": error_response("500", "Server error"),
}


__all__ = [
    "COMMON_SCHEMAS",
    "STANDARD_ERRORS",
    "ok_response",
    "array_response",
    "error_response",
]
