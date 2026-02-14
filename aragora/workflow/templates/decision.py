"""
SME Decision Workflow Template.

Provides a gold-path decision workflow template that wires together:
debate → plan_generation → approval_gate → execution

This template is registered in the WORKFLOW_TEMPLATES catalog for
discovery via CLI (--list-templates) and API (GET /api/workflow-templates).
"""

from __future__ import annotations

from typing import Any


def create_sme_decision_template() -> dict[str, Any]:
    """Create the SME decision workflow template.

    A 7-node DAG: context_read → debate → plan_generation → human_review →
    approval_gate → execution → store_decision.
    Configurable consensus threshold for auto-approval.

    Returns:
        Template dict compatible with WORKFLOW_TEMPLATES catalog.
    """
    return {
        "name": "SME Decision Pipeline",
        "description": (
            "Full decision pipeline for SME teams: multi-agent debate, "
            "plan generation, optional human approval, and execution. "
            "Auto-approves when consensus exceeds 80%."
        ),
        "category": "decision",
        "tags": ["sme", "decision", "pipeline", "gold-path"],
        "agents": ["claude", "codex", "deepseek"],
        "rounds": 2,
        "consensus_threshold": 0.8,
        "auto_approve_above": 0.8,
        "steps": [
            {
                "id": "context_read",
                "name": "Load Decision Context",
                "type": "memory_read",
                "description": "Retrieve prior decisions and context from knowledge base",
                "config": {
                    "query_template": "Find prior decisions related to: {topic}",
                    "domains": ["decision/sme"],
                },
            },
            {
                "id": "debate",
                "name": "Multi-Agent Debate",
                "type": "debate",
                "config": {
                    "agents": ["claude", "codex", "deepseek"],
                    "rounds": 2,
                    "consensus": "majority",
                },
            },
            {
                "id": "plan_generation",
                "name": "Generate Decision Plan",
                "type": "task",
                "config": {
                    "task_type": "plan_from_debate",
                    "approval_mode": "risk_based",
                },
            },
            {
                "id": "human_review",
                "name": "Human Review",
                "type": "human_checkpoint",
                "description": "Decision-maker reviews the proposed plan",
                "config": {
                    "approval_type": "review",
                    "checklist": [
                        "Verify debate consensus is sound",
                        "Confirm plan addresses key risks",
                        "Approve execution scope",
                    ],
                    "timeout_hours": 24,
                },
            },
            {
                "id": "approval_gate",
                "name": "Approval Gate",
                "type": "decision",
                "config": {
                    "conditions": [
                        {
                            "name": "auto_approve",
                            "expression": "consensus_confidence >= 0.8",
                            "next_step": "execution",
                        },
                    ],
                    "default_branch": "human_review",
                    "timeout_seconds": 86400,
                },
            },
            {
                "id": "execution",
                "name": "Execute Plan",
                "type": "task",
                "config": {
                    "task_type": "execute_plan",
                    "execution_mode": "workflow",
                },
            },
            {
                "id": "store_decision",
                "name": "Store Decision",
                "type": "memory_write",
                "description": "Persist decision outcome for future reference",
                "config": {
                    "domain": "decision/sme",
                    "confidence": 0.85,
                },
            },
        ],
        "transitions": [
            {"from": "context_read", "to": "debate"},
            {"from": "debate", "to": "plan_generation"},
            {"from": "plan_generation", "to": "human_review"},
            {"from": "human_review", "to": "approval_gate", "condition": "approved"},
            {"from": "human_review", "to": "plan_generation", "condition": "rejected"},
            {"from": "approval_gate", "to": "execution"},
            {"from": "execution", "to": "store_decision"},
        ],
        "metadata": {
            "author": "aragora",
            "version": "1.0.0",
            "min_agents": 2,
            "max_agents": 5,
            "estimated_duration_minutes": 5,
        },
    }


def create_quick_decision_template() -> dict[str, Any]:
    """Create a quick decision template for simple yes/no decisions.

    Minimal: 2 agents, 1 round, auto-approve at 70% consensus.

    Returns:
        Template dict compatible with WORKFLOW_TEMPLATES catalog.
    """
    return {
        "name": "Quick Decision",
        "description": (
            "Fast decision pipeline for simple questions. "
            "2 agents, 1 round, auto-approve at 70% consensus."
        ),
        "category": "decision",
        "tags": ["sme", "quick", "decision"],
        "agents": ["claude", "codex"],
        "rounds": 1,
        "consensus_threshold": 0.7,
        "auto_approve_above": 0.7,
        "steps": [
            {
                "id": "context_read",
                "name": "Load Context",
                "type": "memory_read",
                "description": "Retrieve relevant context for the decision",
                "config": {
                    "query_template": "Find context for: {topic}",
                    "domains": ["decision/quick"],
                },
            },
            {
                "id": "debate",
                "name": "Quick Debate",
                "type": "debate",
                "config": {
                    "agents": ["claude", "codex"],
                    "rounds": 1,
                    "consensus": "majority",
                },
            },
            {
                "id": "human_review",
                "name": "Quick Review",
                "type": "human_checkpoint",
                "description": "Brief human review of the decision outcome",
                "config": {
                    "approval_type": "review",
                    "checklist": [
                        "Confirm decision aligns with intent",
                    ],
                    "timeout_hours": 12,
                },
            },
            {
                "id": "execution",
                "name": "Execute",
                "type": "task",
                "config": {
                    "task_type": "execute_plan",
                    "execution_mode": "workflow",
                },
            },
            {
                "id": "store_result",
                "name": "Store Result",
                "type": "memory_write",
                "description": "Persist quick decision result",
                "config": {
                    "domain": "decision/quick",
                    "confidence": 0.8,
                },
            },
        ],
        "transitions": [
            {"from": "context_read", "to": "debate"},
            {"from": "debate", "to": "human_review"},
            {"from": "human_review", "to": "execution", "condition": "approved"},
            {"from": "human_review", "to": "debate", "condition": "rejected"},
            {"from": "execution", "to": "store_result"},
        ],
        "metadata": {
            "author": "aragora",
            "version": "1.0.0",
            "estimated_duration_minutes": 2,
        },
    }


# Templates to register in the catalog
DECISION_TEMPLATES = {
    "decision/sme-decision": create_sme_decision_template(),
    "decision/quick-decision": create_quick_decision_template(),
}
