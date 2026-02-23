"""
Control Plane OpenAPI Schema Definitions.

Schemas for agent registry, task scheduling, and control plane operations.
"""

from typing import Any

CONTROL_PLANE_SCHEMAS: dict[str, Any] = {
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
            "current_task_id": {"type": ["string", "null"]},
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
            "assigned_at": {"type": ["number", "null"]},
            "started_at": {"type": ["number", "null"]},
            "completed_at": {"type": ["number", "null"]},
            "assigned_agent": {"type": ["string", "null"]},
            "timeout_seconds": {"type": ["number", "null"]},
            "max_retries": {"type": "integer"},
            "retries": {"type": "integer"},
            "result": {"type": ["object", "null"]},
            "error": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
            "target_region": {"type": ["string", "null"]},
            "fallback_regions": {"type": "array", "items": {"type": "string"}},
            "assigned_region": {"type": ["string", "null"]},
            "region_routing_mode": {"type": "string"},
            "origin_region": {"type": ["string", "null"]},
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
            "started_at": {"type": ["string", "null"]},
            "created_at": {"type": ["string", "null"]},
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
    "AgentRegistration": {
        "type": "object",
        "description": "Agent registration data",
        "properties": {
            "agent_id": {"type": "string"},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "model": {"type": "string"},
            "provider": {"type": "string"},
            "registered_at": {"type": "string", "format": "date-time"},
            "status": {"type": "string"},
        },
    },
    "TaskStatus": {
        "type": "object",
        "description": "Task status record",
        "properties": {
            "task_id": {"type": "string"},
            "status": {"type": "string"},
            "progress": {"type": "number"},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "result": {"type": ["object", "null"]},
            "error": {"type": ["string", "null"]},
        },
    },
    "PolicyEvaluation": {
        "type": "object",
        "description": "Policy evaluation result",
        "properties": {
            "policy_id": {"type": "string"},
            "allowed": {"type": "boolean"},
            "reason": {"type": "string"},
            "conditions_met": {"type": "array", "items": {"type": "string"}},
            "evaluated_at": {"type": "string", "format": "date-time"},
        },
    },
}


__all__ = ["CONTROL_PLANE_SCHEMAS"]
