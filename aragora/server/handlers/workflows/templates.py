"""
Workflow template management.

Provides operations for listing templates, creating workflows from templates,
and registering built-in templates.
"""

from __future__ import annotations

from typing import Any

from .core import (
    logger,
    _get_store,
    WorkflowDefinition,
    WorkflowCategory,
    StepDefinition,
    TransitionRule,
)
from .crud import create_workflow


async def list_templates(
    category: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List available workflow templates."""
    store = _get_store()
    templates = store.list_templates(category=category, tags=tags)
    return [t.to_dict() for t in templates]


async def get_template(template_id: str) -> dict[str, Any] | None:
    """Get a workflow template by ID."""
    store = _get_store()
    template = store.get_template(template_id)
    return template.to_dict() if template else None


async def create_workflow_from_template(
    template_id: str,
    name: str,
    tenant_id: str = "default",
    created_by: str = "",
    customizations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new workflow from a template."""
    store = _get_store()
    template = store.get_template(template_id)
    if not template:
        raise ValueError(f"Template not found: {template_id}")

    # Increment usage count
    store.increment_template_usage(template_id)

    # Clone template
    workflow = template.clone(new_name=name)

    # Apply customizations
    if customizations:
        workflow_dict = workflow.to_dict()
        workflow_dict.update(customizations)
        workflow = WorkflowDefinition.from_dict(workflow_dict)

    return await create_workflow(workflow.to_dict(), tenant_id, created_by)


def register_template(workflow: WorkflowDefinition) -> None:
    """Register a workflow as a template.

    Note: For PostgreSQL backends, this is a no-op at import time. Templates
    registered via this function will be loaded during server startup.
    """
    workflow.is_template = True
    from aragora.storage.factory import get_storage_backend, StorageBackend

    backend = get_storage_backend()
    if backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
        logger.debug("Deferring template registration for %s to server startup", workflow.id)
        return

    store = _get_store()
    store.save_template(workflow)


# =============================================================================
# Built-in Templates
# =============================================================================


def _create_contract_review_template() -> WorkflowDefinition:
    """Create contract review workflow template."""
    from aragora.workflow.types import (
        Position,
        VisualNodeData,
        NodeCategory,
    )

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


def _create_code_review_template() -> WorkflowDefinition:
    """Create code review workflow template."""
    from aragora.workflow.types import Position, VisualNodeData, NodeCategory

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


def _register_builtin_templates() -> None:
    """Register built-in templates (Python-defined)."""
    register_template(_create_contract_review_template())
    register_template(_create_code_review_template())


# Load YAML templates from disk
def _load_yaml_templates() -> None:
    """Load workflow templates from YAML files into persistent store.

    Note: For PostgreSQL backends, template loading is deferred to server startup
    (via load_yaml_templates_async) to avoid connection pool issues at import time.
    """
    try:
        from aragora.workflow.template_loader import load_templates
        from aragora.storage.factory import get_storage_backend, StorageBackend

        backend = get_storage_backend()
        if backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
            logger.debug(
                "Skipping YAML template loading at import time for PostgreSQL backend. "
                "Templates will be loaded during server startup."
            )
            return

        store = _get_store()

        templates = load_templates()
        loaded = 0
        for template_id, template in templates.items():
            # Check if already in database
            existing = store.get_template(template_id)
            if not existing:
                store.save_template(template)
                loaded += 1
        if loaded > 0:
            logger.info("Loaded %s new YAML templates into database", loaded)
    except ImportError as e:
        logger.debug("Template loader not available: %s", e)
    except OSError as e:
        logger.warning("Failed to read YAML templates from disk: %s", e)
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Failed to parse YAML templates: %s", e)


async def load_yaml_templates_async() -> None:
    """Load workflow templates from YAML files (async version for PostgreSQL).

    Call this during server startup to load templates when using PostgreSQL backend.
    """
    try:
        from aragora.workflow.template_loader import load_templates
        from aragora.workflow.persistent_store import get_async_workflow_store

        store = await get_async_workflow_store()
        templates = load_templates()
        loaded = 0
        for template_id, template in templates.items():
            existing = await store.get_template(template_id)  # type: ignore[misc]  # WorkflowStoreType union: async/sync method variance
            if not existing:
                await store.save_template(template)
                loaded += 1
        if loaded > 0:
            logger.info("Loaded %s new YAML templates into database (async)", loaded)
    except ImportError as e:
        logger.debug("Template loader not available: %s", e)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("Failed to load YAML templates (async): %s", e)


async def register_builtin_templates_async() -> None:
    """Register built-in Python-defined templates (async version for PostgreSQL).

    Call this during server startup when using PostgreSQL backend.
    """
    try:
        from aragora.workflow.persistent_store import get_async_workflow_store

        store = await get_async_workflow_store()

        # Register Python-defined templates
        templates = [
            _create_contract_review_template(),
            _create_code_review_template(),
        ]
        for template in templates:
            template.is_template = True
            existing = await store.get_template(template.id)  # type: ignore[misc]  # WorkflowStoreType union: async/sync method variance
            if not existing:
                await store.save_template(template)
                logger.debug("Registered built-in template: %s", template.id)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.warning("Failed to register built-in templates (async): %s", e)


def initialize_templates() -> None:
    """Initialize all templates (called on module import)."""
    _register_builtin_templates()
    _load_yaml_templates()


__all__ = [
    "list_templates",
    "get_template",
    "create_workflow_from_template",
    "register_template",
    "load_yaml_templates_async",
    "register_builtin_templates_async",
    "initialize_templates",
    # Internal template creators (for async registration)
    "_create_contract_review_template",
    "_create_code_review_template",
]
