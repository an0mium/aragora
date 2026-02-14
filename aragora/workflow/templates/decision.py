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

    A 4-node DAG: debate → plan_generation → approval_gate → execution.
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
        "agents": ["anthropic-api", "openai-api", "deepseek"],
        "rounds": 2,
        "consensus_threshold": 0.8,
        "auto_approve_above": 0.8,
        "steps": [
            {
                "id": "debate",
                "name": "Multi-Agent Debate",
                "type": "debate",
                "config": {
                    "agents": ["anthropic-api", "openai-api", "deepseek"],
                    "rounds": 2,
                    "consensus": "majority",
                },
                "next_steps": ["plan_generation"],
            },
            {
                "id": "plan_generation",
                "name": "Generate Decision Plan",
                "type": "task",
                "config": {
                    "task_type": "plan_from_debate",
                    "approval_mode": "risk_based",
                },
                "next_steps": ["approval_gate"],
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
                    "default_branch": "human_approval",
                    "timeout_seconds": 86400,
                },
                "next_steps": ["execution"],
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
        "agents": ["anthropic-api", "openai-api"],
        "rounds": 1,
        "consensus_threshold": 0.7,
        "auto_approve_above": 0.7,
        "steps": [
            {
                "id": "debate",
                "name": "Quick Debate",
                "type": "debate",
                "config": {
                    "agents": ["anthropic-api", "openai-api"],
                    "rounds": 1,
                    "consensus": "majority",
                },
                "next_steps": ["execution"],
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
