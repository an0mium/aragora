"""
MCP Tool Definitions for Aragora.

These functions can be used standalone or registered with an MCP server.
Tool implementations are in the tools_module/ subpackage.
"""

from __future__ import annotations

import logging

# Import all tool implementations from the tools_module subpackage
from aragora.mcp.tools_module import (
    # Debate tools
    run_debate_tool,
    get_debate_tool,
    search_debates_tool,
    fork_debate_tool,
    get_forks_tool,
    # Gauntlet tools
    run_gauntlet_tool,
    # Agent tools
    list_agents_tool,
    get_agent_history_tool,
    get_agent_lineage_tool,
    breed_agents_tool,
    # Memory tools
    query_memory_tool,
    store_memory_tool,
    get_memory_pressure_tool,
    # Checkpoint tools
    create_checkpoint_tool,
    list_checkpoints_tool,
    resume_checkpoint_tool,
    delete_checkpoint_tool,
    # Verification tools
    get_consensus_proofs_tool,
    verify_consensus_tool,
    generate_proof_tool,
    # Evidence tools
    search_evidence_tool,
    cite_evidence_tool,
    verify_citation_tool,
    # Trending tools
    list_trending_topics_tool,
    # Audit tools
    list_audit_presets_tool,
    list_audit_types_tool,
    get_audit_preset_tool,
    create_audit_session_tool,
    run_audit_tool,
    get_audit_status_tool,
    get_audit_findings_tool,
    update_finding_status_tool,
    run_quick_audit_tool,
    # Knowledge tools
    query_knowledge_tool,
    store_knowledge_tool,
    get_knowledge_stats_tool,
    get_decision_receipt_tool,
    # Workflow tools
    run_workflow_tool,
    get_workflow_status_tool,
    list_workflow_templates_tool,
    cancel_workflow_tool,
    # External integration tools
    trigger_external_webhook_tool,
    list_integrations_tool,
    test_integration_tool,
    get_integration_events_tool,
    # Control plane tools
    register_agent_tool,
    unregister_agent_tool,
    list_registered_agents_tool,
    get_agent_health_tool,
    submit_task_tool,
    get_task_status_tool,
    cancel_task_tool,
    list_pending_tasks_tool,
    get_control_plane_status_tool,
    trigger_health_check_tool,
    get_resource_utilization_tool,
    # Canvas tools
    canvas_create_tool,
    canvas_get_tool,
    canvas_add_node_tool,
    canvas_add_edge_tool,
    canvas_execute_action_tool,
    canvas_list_tool,
    canvas_delete_node_tool,
)

logger = logging.getLogger(__name__)


# Tool metadata for MCP registration
TOOLS_METADATA = [
    {
        "name": "run_debate",
        "description": "Run a multi-agent AI debate on a topic",
        "function": run_debate_tool,
        "parameters": {
            "question": {"type": "string", "required": True},
            "agents": {"type": "string", "default": "anthropic-api,openai-api"},
            "rounds": {"type": "integer", "default": 3},
            "consensus": {"type": "string", "default": "majority"},
        },
    },
    {
        "name": "run_gauntlet",
        "description": "Stress-test content through adversarial analysis",
        "function": run_gauntlet_tool,
        "parameters": {
            "content": {"type": "string", "required": True},
            "content_type": {"type": "string", "default": "spec"},
            "profile": {"type": "string", "default": "quick"},
        },
    },
    {
        "name": "list_agents",
        "description": "List available AI agents",
        "function": list_agents_tool,
        "parameters": {},
    },
    {
        "name": "get_debate",
        "description": "Get results of a previous debate",
        "function": get_debate_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "search_debates",
        "description": "Search debates by topic, date, or agents",
        "function": search_debates_tool,
        "parameters": {
            "query": {"type": "string"},
            "agent": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "consensus_only": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "get_agent_history",
        "description": "Get agent debate history and performance stats",
        "function": get_agent_history_tool,
        "parameters": {
            "agent_name": {"type": "string", "required": True},
            "include_debates": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "get_consensus_proofs",
        "description": "Retrieve formal verification proofs from debates",
        "function": get_consensus_proofs_tool,
        "parameters": {
            "debate_id": {"type": "string"},
            "proof_type": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "list_trending_topics",
        "description": "Get trending topics from Pulse for debates",
        "function": list_trending_topics_tool,
        "parameters": {
            "platform": {"type": "string", "default": "all"},
            "category": {"type": "string"},
            "min_score": {"type": "number", "default": 0.5},
            "limit": {"type": "integer", "default": 10},
        },
    },
    # Memory tools
    {
        "name": "query_memory",
        "description": "Query memories from the continuum memory system",
        "function": query_memory_tool,
        "parameters": {
            "query": {"type": "string", "required": True},
            "tier": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
            "min_importance": {"type": "number", "default": 0.0},
        },
    },
    {
        "name": "store_memory",
        "description": "Store a memory in the continuum memory system",
        "function": store_memory_tool,
        "parameters": {
            "content": {"type": "string", "required": True},
            "tier": {"type": "string", "default": "medium"},
            "importance": {"type": "number", "default": 0.5},
            "tags": {"type": "string", "default": ""},
        },
    },
    {
        "name": "get_memory_pressure",
        "description": "Get current memory pressure and utilization",
        "function": get_memory_pressure_tool,
        "parameters": {},
    },
    # Fork tools
    {
        "name": "fork_debate",
        "description": "Fork a debate to explore counterfactual scenarios",
        "function": fork_debate_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "branch_point": {"type": "integer", "default": -1},
            "modified_context": {"type": "string", "default": ""},
        },
    },
    {
        "name": "get_forks",
        "description": "Get all forks of a debate",
        "function": get_forks_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "include_nested": {"type": "boolean", "default": False},
        },
    },
    # Genesis tools
    {
        "name": "get_agent_lineage",
        "description": "Get the evolutionary lineage of an agent",
        "function": get_agent_lineage_tool,
        "parameters": {
            "agent_name": {"type": "string", "required": True},
            "depth": {"type": "integer", "default": 5},
        },
    },
    {
        "name": "breed_agents",
        "description": "Breed two agents to create a new offspring agent",
        "function": breed_agents_tool,
        "parameters": {
            "parent_a": {"type": "string", "required": True},
            "parent_b": {"type": "string", "required": True},
            "mutation_rate": {"type": "number", "default": 0.1},
        },
    },
    # Checkpoint tools
    {
        "name": "create_checkpoint",
        "description": "Create a checkpoint for a debate to enable resume later",
        "function": create_checkpoint_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "label": {"type": "string", "default": ""},
            "storage_backend": {"type": "string", "default": "file"},
        },
    },
    {
        "name": "list_checkpoints",
        "description": "List checkpoints for a debate or all debates",
        "function": list_checkpoints_tool,
        "parameters": {
            "debate_id": {"type": "string", "default": ""},
            "include_expired": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "resume_checkpoint",
        "description": "Resume a debate from a checkpoint",
        "function": resume_checkpoint_tool,
        "parameters": {
            "checkpoint_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "delete_checkpoint",
        "description": "Delete a checkpoint",
        "function": delete_checkpoint_tool,
        "parameters": {
            "checkpoint_id": {"type": "string", "required": True},
        },
    },
    # Verification tools
    {
        "name": "verify_consensus",
        "description": "Verify the consensus of a completed debate using formal methods",
        "function": verify_consensus_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "backend": {"type": "string", "default": "z3"},
        },
    },
    {
        "name": "generate_proof",
        "description": "Generate a formal proof for a claim without verification",
        "function": generate_proof_tool,
        "parameters": {
            "claim": {"type": "string", "required": True},
            "output_format": {"type": "string", "default": "lean4"},
            "context": {"type": "string", "default": ""},
        },
    },
    # Evidence tools
    {
        "name": "search_evidence",
        "description": "Search for evidence across configured sources",
        "function": search_evidence_tool,
        "parameters": {
            "query": {"type": "string", "required": True},
            "sources": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "cite_evidence",
        "description": "Add a citation to evidence in a debate message",
        "function": cite_evidence_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "evidence_id": {"type": "string", "required": True},
            "message_index": {"type": "integer", "required": True},
            "citation_text": {"type": "string", "default": ""},
        },
    },
    {
        "name": "verify_citation",
        "description": "Verify that a citation URL is valid and accessible",
        "function": verify_citation_tool,
        "parameters": {
            "url": {"type": "string", "required": True},
        },
    },
    # Audit tools
    {
        "name": "list_audit_presets",
        "description": "List available audit presets (Legal Due Diligence, Financial Audit, Code Security)",
        "function": list_audit_presets_tool,
        "parameters": {},
    },
    {
        "name": "list_audit_types",
        "description": "List registered audit types (security, compliance, consistency, quality)",
        "function": list_audit_types_tool,
        "parameters": {},
    },
    {
        "name": "get_audit_preset",
        "description": "Get details of a specific audit preset including custom rules",
        "function": get_audit_preset_tool,
        "parameters": {
            "preset_name": {
                "type": "string",
                "required": True,
                "description": "Name of preset (e.g., 'Legal Due Diligence')",
            },
        },
    },
    {
        "name": "create_audit_session",
        "description": "Create a new document audit session",
        "function": create_audit_session_tool,
        "parameters": {
            "document_ids": {
                "type": "string",
                "required": True,
                "description": "Comma-separated document IDs",
            },
            "audit_types": {"type": "string", "default": "security,compliance,consistency,quality"},
            "preset": {"type": "string", "description": "Optional preset name to use"},
            "name": {"type": "string", "description": "Optional session name"},
        },
    },
    {
        "name": "run_audit",
        "description": "Start running an audit session",
        "function": run_audit_tool,
        "parameters": {
            "session_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_audit_status",
        "description": "Get status and progress of an audit session",
        "function": get_audit_status_tool,
        "parameters": {
            "session_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_audit_findings",
        "description": "Get findings from an audit session with optional filtering",
        "function": get_audit_findings_tool,
        "parameters": {
            "session_id": {"type": "string", "required": True},
            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
            "status": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
    },
    {
        "name": "update_finding_status",
        "description": "Update the workflow status of a finding",
        "function": update_finding_status_tool,
        "parameters": {
            "finding_id": {"type": "string", "required": True},
            "status": {
                "type": "string",
                "required": True,
                "enum": [
                    "open",
                    "triaging",
                    "investigating",
                    "remediating",
                    "resolved",
                    "false_positive",
                    "accepted_risk",
                ],
            },
            "comment": {"type": "string", "default": ""},
        },
    },
    {
        "name": "run_quick_audit",
        "description": "Run a quick audit using a preset and return findings summary",
        "function": run_quick_audit_tool,
        "parameters": {
            "document_ids": {
                "type": "string",
                "required": True,
                "description": "Comma-separated document IDs",
            },
            "preset": {
                "type": "string",
                "default": "Code Security",
                "description": "Preset to use",
            },
        },
    },
    # Knowledge tools
    {
        "name": "query_knowledge",
        "description": "Query the Knowledge Mound for relevant information",
        "function": query_knowledge_tool,
        "parameters": {
            "query": {"type": "string", "required": True},
            "node_types": {"type": "string", "default": "all"},
            "min_confidence": {"type": "number", "default": 0.0},
            "limit": {"type": "integer", "default": 10},
            "include_relationships": {"type": "boolean", "default": False},
        },
    },
    {
        "name": "store_knowledge",
        "description": "Store a new knowledge node in the Knowledge Mound",
        "function": store_knowledge_tool,
        "parameters": {
            "content": {"type": "string", "required": True},
            "node_type": {"type": "string", "default": "fact"},
            "confidence": {"type": "number", "default": 0.8},
            "tier": {"type": "string", "default": "medium"},
            "topics": {"type": "string", "default": ""},
            "source_debate_id": {"type": "string"},
        },
    },
    {
        "name": "get_knowledge_stats",
        "description": "Get statistics about the Knowledge Mound",
        "function": get_knowledge_stats_tool,
        "parameters": {},
    },
    {
        "name": "get_decision_receipt",
        "description": "Get a formal decision receipt for a completed debate",
        "function": get_decision_receipt_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "format": {"type": "string", "default": "json"},
            "include_proofs": {"type": "boolean", "default": True},
            "include_evidence": {"type": "boolean", "default": True},
        },
    },
    # Workflow tools
    {
        "name": "run_workflow",
        "description": "Execute a workflow from a template",
        "function": run_workflow_tool,
        "parameters": {
            "template": {"type": "string", "required": True},
            "inputs": {"type": "string", "default": ""},
            "async_execution": {"type": "boolean", "default": False},
            "timeout_seconds": {"type": "integer", "default": 300},
        },
    },
    {
        "name": "get_workflow_status",
        "description": "Get the status of a workflow execution",
        "function": get_workflow_status_tool,
        "parameters": {
            "execution_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "list_workflow_templates",
        "description": "List available workflow templates",
        "function": list_workflow_templates_tool,
        "parameters": {
            "category": {"type": "string", "default": "all"},
        },
    },
    {
        "name": "cancel_workflow",
        "description": "Cancel a running workflow execution",
        "function": cancel_workflow_tool,
        "parameters": {
            "execution_id": {"type": "string", "required": True},
            "reason": {"type": "string", "default": ""},
        },
    },
    # External integration tools
    {
        "name": "trigger_external_webhook",
        "description": "Trigger an external automation webhook (Zapier, Make, n8n)",
        "function": trigger_external_webhook_tool,
        "parameters": {
            "platform": {"type": "string", "required": True},
            "event_type": {"type": "string", "required": True},
            "data": {"type": "string", "default": "{}"},
        },
    },
    {
        "name": "list_integrations",
        "description": "List configured external integrations",
        "function": list_integrations_tool,
        "parameters": {
            "platform": {"type": "string", "default": "all"},
            "workspace_id": {"type": "string"},
        },
    },
    {
        "name": "test_integration",
        "description": "Test an integration connection",
        "function": test_integration_tool,
        "parameters": {
            "platform": {"type": "string", "required": True},
            "integration_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_integration_events",
        "description": "Get available event types for an integration platform",
        "function": get_integration_events_tool,
        "parameters": {
            "platform": {"type": "string", "required": True},
        },
    },
    # Control plane tools
    {
        "name": "register_agent",
        "description": "Register an agent with the control plane",
        "function": register_agent_tool,
        "parameters": {
            "agent_id": {"type": "string", "required": True},
            "capabilities": {"type": "string", "default": "debate"},
            "model": {"type": "string", "default": "unknown"},
            "provider": {"type": "string", "default": "unknown"},
        },
    },
    {
        "name": "unregister_agent",
        "description": "Unregister an agent from the control plane",
        "function": unregister_agent_tool,
        "parameters": {
            "agent_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "list_registered_agents",
        "description": "List all agents registered with the control plane",
        "function": list_registered_agents_tool,
        "parameters": {
            "capability": {"type": "string", "default": ""},
            "only_available": {"type": "boolean", "default": True},
        },
    },
    {
        "name": "get_agent_health",
        "description": "Get detailed health status for a specific agent",
        "function": get_agent_health_tool,
        "parameters": {
            "agent_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "submit_task",
        "description": "Submit a task to the control plane for execution",
        "function": submit_task_tool,
        "parameters": {
            "task_type": {"type": "string", "required": True},
            "payload": {"type": "string", "default": "{}"},
            "required_capabilities": {"type": "string", "default": ""},
            "priority": {"type": "string", "default": "normal"},
            "timeout_seconds": {"type": "integer", "default": 300},
        },
    },
    {
        "name": "get_task_status",
        "description": "Get the status of a task in the control plane",
        "function": get_task_status_tool,
        "parameters": {
            "task_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "cancel_task",
        "description": "Cancel a pending or running task",
        "function": cancel_task_tool,
        "parameters": {
            "task_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "list_pending_tasks",
        "description": "List tasks in the pending queue",
        "function": list_pending_tasks_tool,
        "parameters": {
            "task_type": {"type": "string", "default": ""},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "get_control_plane_status",
        "description": "Get overall control plane health and status",
        "function": get_control_plane_status_tool,
        "parameters": {},
    },
    {
        "name": "trigger_health_check",
        "description": "Trigger a health check for an agent or all agents",
        "function": trigger_health_check_tool,
        "parameters": {
            "agent_id": {"type": "string", "default": ""},
        },
    },
    {
        "name": "get_resource_utilization",
        "description": "Get resource utilization metrics for the control plane",
        "function": get_resource_utilization_tool,
        "parameters": {},
    },
    # Canvas tools
    {
        "name": "canvas_create",
        "description": "Create a new interactive canvas for visual collaboration",
        "function": canvas_create_tool,
        "parameters": {
            "name": {"type": "string", "default": "Untitled Canvas"},
            "description": {"type": "string", "default": ""},
            "owner_id": {"type": "string", "default": ""},
            "workspace_id": {"type": "string", "default": ""},
        },
    },
    {
        "name": "canvas_get",
        "description": "Get the state of a canvas including nodes and edges",
        "function": canvas_get_tool,
        "parameters": {
            "canvas_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "canvas_add_node",
        "description": "Add a node to a canvas (text, agent, debate, knowledge, workflow, browser)",
        "function": canvas_add_node_tool,
        "parameters": {
            "canvas_id": {"type": "string", "required": True},
            "node_type": {"type": "string", "default": "text"},
            "label": {"type": "string", "default": ""},
            "x": {"type": "integer", "default": 100},
            "y": {"type": "integer", "default": 100},
            "data": {"type": "string", "default": "{}"},
        },
    },
    {
        "name": "canvas_add_edge",
        "description": "Add an edge between two nodes on a canvas",
        "function": canvas_add_edge_tool,
        "parameters": {
            "canvas_id": {"type": "string", "required": True},
            "source_id": {"type": "string", "required": True},
            "target_id": {"type": "string", "required": True},
            "edge_type": {"type": "string", "default": "default"},
            "label": {"type": "string", "default": ""},
        },
    },
    {
        "name": "canvas_execute_action",
        "description": "Execute an action on a canvas (start_debate, run_workflow, query_knowledge, clear_canvas)",
        "function": canvas_execute_action_tool,
        "parameters": {
            "canvas_id": {"type": "string", "required": True},
            "action": {"type": "string", "required": True},
            "params": {"type": "string", "default": "{}"},
        },
    },
    {
        "name": "canvas_list",
        "description": "List available canvases",
        "function": canvas_list_tool,
        "parameters": {
            "owner_id": {"type": "string", "default": ""},
            "workspace_id": {"type": "string", "default": ""},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "canvas_delete_node",
        "description": "Delete a node from a canvas",
        "function": canvas_delete_node_tool,
        "parameters": {
            "canvas_id": {"type": "string", "required": True},
            "node_id": {"type": "string", "required": True},
        },
    },
]


__all__ = [
    "run_debate_tool",
    "run_gauntlet_tool",
    "list_agents_tool",
    "get_debate_tool",
    "search_debates_tool",
    "get_agent_history_tool",
    "get_consensus_proofs_tool",
    "list_trending_topics_tool",
    # Memory tools
    "query_memory_tool",
    "store_memory_tool",
    "get_memory_pressure_tool",
    # Fork tools
    "fork_debate_tool",
    "get_forks_tool",
    # Genesis tools
    "get_agent_lineage_tool",
    "breed_agents_tool",
    # Checkpoint tools
    "create_checkpoint_tool",
    "list_checkpoints_tool",
    "resume_checkpoint_tool",
    "delete_checkpoint_tool",
    # Verification tools
    "verify_consensus_tool",
    "generate_proof_tool",
    # Evidence tools
    "search_evidence_tool",
    "cite_evidence_tool",
    "verify_citation_tool",
    # Audit tools
    "list_audit_presets_tool",
    "list_audit_types_tool",
    "get_audit_preset_tool",
    "create_audit_session_tool",
    "run_audit_tool",
    "get_audit_status_tool",
    "get_audit_findings_tool",
    "update_finding_status_tool",
    "run_quick_audit_tool",
    # Knowledge tools
    "query_knowledge_tool",
    "store_knowledge_tool",
    "get_knowledge_stats_tool",
    "get_decision_receipt_tool",
    # Workflow tools
    "run_workflow_tool",
    "get_workflow_status_tool",
    "list_workflow_templates_tool",
    "cancel_workflow_tool",
    # External integration tools
    "trigger_external_webhook_tool",
    "list_integrations_tool",
    "test_integration_tool",
    "get_integration_events_tool",
    # Control plane tools
    "register_agent_tool",
    "unregister_agent_tool",
    "list_registered_agents_tool",
    "get_agent_health_tool",
    "submit_task_tool",
    "get_task_status_tool",
    "cancel_task_tool",
    "list_pending_tasks_tool",
    "get_control_plane_status_tool",
    "trigger_health_check_tool",
    "get_resource_utilization_tool",
    # Canvas tools
    "canvas_create_tool",
    "canvas_get_tool",
    "canvas_add_node_tool",
    "canvas_add_edge_tool",
    "canvas_execute_action_tool",
    "canvas_list_tool",
    "canvas_delete_node_tool",
    "TOOLS_METADATA",
]
