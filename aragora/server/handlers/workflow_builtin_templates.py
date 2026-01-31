"""
Built-in workflow templates for the Visual Workflow Builder.

This module contains:
- Python-defined workflow templates (contract review, code review)
- YAML template loading functionality
- Template registration utilities

Extracted from workflows.py to reduce file size and improve maintainability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.workflow.persistent_store import PersistentWorkflowStore

from aragora.workflow.types import (
    WorkflowDefinition,
    WorkflowCategory,
    StepDefinition,
    TransitionRule,
    Position,
    VisualNodeData,
    NodeCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Built-in Template Definitions
# =============================================================================


def create_contract_review_template() -> WorkflowDefinition:
    """Create contract review workflow template.

    A multi-agent review workflow for contract documents with:
    - Key term extraction
    - Legal analysis debate
    - Risk-based routing (human review for high-risk, auto-approve for low-risk)
    - Knowledge Mound storage

    Returns:
        WorkflowDefinition configured as a template
    """
    return WorkflowDefinition(
        id="template_contract_review",
        name="Contract Review",
        description="Multi-agent review of contract documents with legal analysis",
        category=WorkflowCategory.LEGAL,
        tags=["legal", "contracts", "review", "compliance"],
        is_template=True,
        icon="document-text",
        steps=[
            StepDefinition(
                id="extract",
                name="Extract Key Terms",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Extract key terms and clauses from: {document}",
                },
                description="Extract important terms and clauses from the contract",
                visual=VisualNodeData(
                    position=Position(100, 100),
                    category=NodeCategory.AGENT,
                    color="#4299e1",
                ),
                next_steps=["analyze"],
            ),
            StepDefinition(
                id="analyze",
                name="Legal Analysis",
                step_type="debate",
                config={
                    "topic": "Analyze legal implications of: {step.extract}",
                    "agents": ["legal_analyst", "risk_assessor"],
                    "rounds": 2,
                },
                description="Multi-agent debate on legal implications",
                visual=VisualNodeData(
                    position=Position(100, 250),
                    category=NodeCategory.DEBATE,
                    color="#38b2ac",
                ),
                next_steps=["risk_check"],
            ),
            StepDefinition(
                id="risk_check",
                name="Risk Assessment",
                step_type="decision",
                config={
                    "conditions": [
                        {
                            "name": "high_risk",
                            "expression": "step.analyze.get('consensus', {}).get('risk_level', 0) > 0.7",
                            "next_step": "human_review",
                        },
                    ],
                    "default_branch": "auto_approve",
                },
                description="Route based on risk assessment",
                visual=VisualNodeData(
                    position=Position(100, 400),
                    category=NodeCategory.CONTROL,
                    color="#ed8936",
                ),
            ),
            StepDefinition(
                id="human_review",
                name="Human Review",
                step_type="human_checkpoint",
                config={
                    "title": "Contract Review Required",
                    "description": "High-risk contract requires human approval",
                    "checklist": [
                        {"label": "Reviewed risk assessment", "required": True},
                        {"label": "Verified compliance terms", "required": True},
                        {"label": "Approved indemnification clause", "required": True},
                    ],
                    "timeout_seconds": 86400,
                },
                description="Human approval for high-risk contracts",
                visual=VisualNodeData(
                    position=Position(250, 550),
                    category=NodeCategory.HUMAN,
                    color="#f56565",
                ),
                next_steps=["store_result"],
            ),
            StepDefinition(
                id="auto_approve",
                name="Auto-Approve",
                step_type="task",
                config={
                    "task_type": "transform",
                    "transform": "{'approved': True, 'method': 'auto', 'analysis': outputs.get('analyze', {})}",
                },
                description="Auto-approve low-risk contracts",
                visual=VisualNodeData(
                    position=Position(-50, 550),
                    category=NodeCategory.TASK,
                    color="#48bb78",
                ),
                next_steps=["store_result"],
            ),
            StepDefinition(
                id="store_result",
                name="Store Analysis",
                step_type="memory_write",
                config={
                    "content": "Contract analysis: {step.analyze.synthesis}",
                    "source_type": "CONSENSUS",
                    "domain": "legal/contracts",
                },
                description="Store analysis in Knowledge Mound",
                visual=VisualNodeData(
                    position=Position(100, 700),
                    category=NodeCategory.MEMORY,
                    color="#9f7aea",
                ),
            ),
        ],
        transitions=[
            TransitionRule(
                id="high_risk_route",
                from_step="risk_check",
                to_step="human_review",
                condition="step_output.get('decision') == 'human_review'",
            ),
            TransitionRule(
                id="low_risk_route",
                from_step="risk_check",
                to_step="auto_approve",
                condition="step_output.get('decision') == 'auto_approve'",
            ),
        ],
    )


def create_code_review_template() -> WorkflowDefinition:
    """Create code review workflow template.

    A security-focused code review workflow with:
    - Static analysis via Codex agent
    - Security debate with adversarial topology
    - Report generation

    Returns:
        WorkflowDefinition configured as a template
    """
    return WorkflowDefinition(
        id="template_code_review",
        name="Code Security Review",
        description="Multi-agent security review of code changes",
        category=WorkflowCategory.CODE,
        tags=["code", "security", "review", "OWASP"],
        is_template=True,
        icon="code",
        steps=[
            StepDefinition(
                id="scan",
                name="Static Analysis",
                step_type="agent",
                config={
                    "agent_type": "codex",
                    "prompt_template": "Perform static security analysis on: {code_diff}",
                },
                description="Run static security analysis",
                visual=VisualNodeData(
                    position=Position(100, 100),
                    category=NodeCategory.AGENT,
                    color="#4299e1",
                ),
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Security Debate",
                step_type="debate",
                config={
                    "topic": "Review security implications: {step.scan}",
                    "agents": ["security_analyst", "penetration_tester"],
                    "rounds": 2,
                    "topology": "adversarial",
                },
                description="Multi-agent security debate",
                visual=VisualNodeData(
                    position=Position(100, 250),
                    category=NodeCategory.DEBATE,
                    color="#38b2ac",
                ),
                next_steps=["summarize"],
            ),
            StepDefinition(
                id="summarize",
                name="Generate Report",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate security report from: {step.debate}",
                },
                description="Generate security report",
                visual=VisualNodeData(
                    position=Position(100, 400),
                    category=NodeCategory.AGENT,
                    color="#4299e1",
                ),
            ),
        ],
    )


# =============================================================================
# Template Registration
# =============================================================================


def register_builtin_templates(store: "PersistentWorkflowStore") -> None:
    """Register all built-in Python-defined templates.

    Args:
        store: The persistent workflow store to register templates in
    """
    contract_template = create_contract_review_template()
    code_template = create_code_review_template()

    store.save_template(contract_template)
    store.save_template(code_template)

    logger.debug("Registered built-in templates: contract_review, code_review")


# =============================================================================
# YAML Template Loading
# =============================================================================


def load_yaml_templates(store: "PersistentWorkflowStore") -> int:
    """Load workflow templates from YAML files into persistent store.

    Templates are loaded from the workflow/templates directory. Only new
    templates (not already in the database) are added.

    Args:
        store: The persistent workflow store to load templates into

    Returns:
        Number of new templates loaded
    """
    try:
        from aragora.workflow.template_loader import load_templates

        templates = load_templates()
        loaded = 0
        for template_id, template in templates.items():
            # Check if already in database
            existing = store.get_template(template_id)
            if not existing:
                store.save_template(template)
                loaded += 1
        if loaded > 0:
            logger.info(f"Loaded {loaded} new YAML templates into database")
        return loaded
    except ImportError as e:
        logger.debug(f"Template loader not available: {e}")
        return 0
    except (OSError, IOError) as e:
        logger.warning(f"Failed to read YAML templates from disk: {e}")
        return 0
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse YAML templates: {e}")
        return 0


def initialize_templates(store: "PersistentWorkflowStore") -> None:
    """Initialize all templates (built-in and YAML) on module load.

    This should be called once when the workflows module initializes.

    Args:
        store: The persistent workflow store to initialize templates in
    """
    register_builtin_templates(store)
    load_yaml_templates(store)
