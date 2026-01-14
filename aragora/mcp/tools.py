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
    "TOOLS_METADATA",
]
