"""
Workflow HTTP API handlers for the Visual Workflow Builder.

Provides CRUD operations and execution control for workflows:
- /api/workflows - List, create workflows
- /api/workflows/:id - Get, update, delete workflow
- /api/workflows/:id/execute - Execute workflow
- /api/workflows/:id/versions - Version history
- /api/workflow-templates - Workflow template gallery
- /api/workflow-approvals - Human approval management
- /api/workflow-executions - List all workflow executions (for runtime dashboard)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aragora.server.http_utils import run_async as _run_async

from aragora.workflow.types import (
    WorkflowDefinition,
    WorkflowCategory,
    StepDefinition,
    TransitionRule,
)
from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.persistent_store import get_workflow_store, PersistentWorkflowStore
from aragora.audit.unified import audit_data

# RBAC imports
try:
    from aragora.rbac import (
        AuthorizationContext,
        check_permission,
        PermissionDeniedError,
        get_role_permissions,
    )
    from aragora.billing.auth import extract_user_from_request

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

# Metrics imports
try:
    from aragora.observability.metrics import record_rbac_check

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_rbac_check(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)


# =============================================================================
# Persistent Storage (SQLite-backed)
# =============================================================================


def _get_store() -> PersistentWorkflowStore:
    """Get the persistent workflow store."""
    return get_workflow_store()  # type: ignore[return-value]


_engine = WorkflowEngine()


# In-memory template store for built-in and YAML templates
class _TemplateStore:
    """In-memory storage for workflow templates."""

    def __init__(self) -> None:
        self.templates: Dict[str, WorkflowDefinition] = {}


_store = _TemplateStore()


# =============================================================================
# CRUD Operations
# =============================================================================


async def list_workflows(
    tenant_id: str = "default",
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    List workflows with optional filtering.

    Returns:
        {
            "workflows": [...],
            "total_count": int,
            "limit": int,
            "offset": int,
        }
    """
    store = _get_store()
    workflows, total = store.list_workflows(
        tenant_id=tenant_id,
        category=category,
        tags=tags,
        search=search,
        limit=limit,
        offset=offset,
    )

    return {
        "workflows": [w.to_dict() for w in workflows],
        "total_count": total,
        "limit": limit,
        "offset": offset,
    }


async def get_workflow(workflow_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
    """Get a workflow by ID."""
    store = _get_store()
    workflow = store.get_workflow(workflow_id, tenant_id)
    return workflow.to_dict() if workflow else None


async def create_workflow(
    data: Dict[str, Any],
    tenant_id: str = "default",
    created_by: str = "",
) -> Dict[str, Any]:
    """
    Create a new workflow.

    Args:
        data: Workflow definition data
        tenant_id: Tenant ID for isolation
        created_by: User ID of creator

    Returns:
        Created workflow definition
    """
    store = _get_store()

    # Generate ID if not provided
    workflow_id = data.get("id") or f"wf_{uuid.uuid4().hex[:12]}"
    data["id"] = workflow_id
    data["tenant_id"] = tenant_id
    data["created_by"] = created_by
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["updated_at"] = data["created_at"]

    workflow = WorkflowDefinition.from_dict(data)

    # Validate
    is_valid, errors = workflow.validate()
    if not is_valid:
        raise ValueError(f"Invalid workflow: {', '.join(errors)}")

    # Save to persistent store
    store.save_workflow(workflow)

    # Save initial version
    store.save_version(workflow)

    logger.info(f"Created workflow {workflow_id}: {workflow.name}")
    audit_data(
        user_id=created_by or "system",
        resource_type="workflow",
        resource_id=workflow_id,
        action="create",
        workflow_name=workflow.name,
        tenant_id=tenant_id,
    )
    return workflow.to_dict()


async def update_workflow(
    workflow_id: str,
    data: Dict[str, Any],
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Update an existing workflow.

    Args:
        workflow_id: ID of workflow to update
        data: Updated workflow data
        tenant_id: Tenant ID for isolation

    Returns:
        Updated workflow definition or None if not found
    """
    store = _get_store()
    existing = store.get_workflow(workflow_id, tenant_id)
    if not existing:
        return None

    # Preserve metadata
    data["id"] = workflow_id
    data["tenant_id"] = tenant_id
    data["created_by"] = existing.created_by
    data["created_at"] = existing.created_at.isoformat() if existing.created_at else None
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Increment version
    old_version = existing.version
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    data["version"] = ".".join(parts)

    workflow = WorkflowDefinition.from_dict(data)

    # Validate
    is_valid, errors = workflow.validate()
    if not is_valid:
        raise ValueError(f"Invalid workflow: {', '.join(errors)}")

    # Save to persistent store
    store.save_workflow(workflow)

    # Save version history
    store.save_version(workflow)

    logger.info(f"Updated workflow {workflow_id} to version {workflow.version}")
    audit_data(
        user_id="system",
        resource_type="workflow",
        resource_id=workflow_id,
        action="update",
        new_version=workflow.version,
        tenant_id=tenant_id,
    )
    return workflow.to_dict()


async def delete_workflow(workflow_id: str, tenant_id: str = "default") -> bool:
    """
    Delete a workflow.

    Args:
        workflow_id: ID of workflow to delete
        tenant_id: Tenant ID for isolation

    Returns:
        True if deleted, False if not found
    """
    store = _get_store()
    deleted = store.delete_workflow(workflow_id, tenant_id)

    if deleted:
        logger.info(f"Deleted workflow {workflow_id}")
        audit_data(
            user_id="system",
            resource_type="workflow",
            resource_id=workflow_id,
            action="delete",
            tenant_id=tenant_id,
        )

    return deleted


# =============================================================================
# Version Management
# =============================================================================


async def get_workflow_versions(
    workflow_id: str,
    tenant_id: str = "default",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get version history for a workflow."""
    store = _get_store()
    return store.get_versions(workflow_id, tenant_id, limit)


async def restore_workflow_version(
    workflow_id: str,
    version: str,
    tenant_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """Restore a workflow to a specific version."""
    store = _get_store()
    old_workflow = store.get_version(workflow_id, version)

    if old_workflow:
        # Create new version from old
        restored = old_workflow.clone(new_id=workflow_id, new_name=old_workflow.name)
        return await update_workflow(workflow_id, restored.to_dict(), tenant_id)

    return None


# =============================================================================
# Execution
# =============================================================================


async def execute_workflow(
    workflow_id: str,
    inputs: Optional[Dict[str, Any]] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """
    Execute a workflow.

    Args:
        workflow_id: ID of workflow to execute
        inputs: Input parameters for the workflow
        tenant_id: Tenant ID for isolation

    Returns:
        Execution result
    """
    store = _get_store()
    workflow = store.get_workflow(workflow_id, tenant_id)
    if not workflow:
        raise ValueError(f"Workflow not found: {workflow_id}")

    execution_id = f"exec_{uuid.uuid4().hex[:12]}"

    # Store execution state
    execution = {
        "id": execution_id,
        "workflow_id": workflow_id,
        "tenant_id": tenant_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs or {},
    }
    store.save_execution(execution)

    try:
        result = await _engine.execute(workflow, inputs, execution_id)

        execution.update(
            {
                "status": "completed" if result.success else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "outputs": result.final_output,
                "steps": [s.__dict__ for s in result.steps],  # type: ignore[misc]
                "error": result.error,  # type: ignore[dict-item]
                "duration_ms": result.total_duration_ms,  # type: ignore[dict-item]
            }
        )
        store.save_execution(execution)

        audit_data(
            user_id="system",
            resource_type="workflow_execution",
            resource_id=execution_id,
            action="execute",
            workflow_id=workflow_id,
            status=execution["status"],
            tenant_id=tenant_id,
        )

        return execution

    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Invalid workflow configuration or inputs: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except (OSError, IOError) as e:
        logger.error(f"Storage error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Connection error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise


async def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get execution status and result."""
    store = _get_store()
    return store.get_execution(execution_id)


async def list_executions(
    workflow_id: Optional[str] = None,
    tenant_id: str = "default",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List workflow executions."""
    store = _get_store()
    executions, _ = store.list_executions(
        workflow_id=workflow_id,
        tenant_id=tenant_id,
        limit=limit,
    )
    return executions


async def terminate_execution(execution_id: str) -> bool:
    """Request termination of a running execution."""
    store = _get_store()
    execution = store.get_execution(execution_id)
    if not execution:
        return False

    if execution.get("status") != "running":
        return False

    _engine.request_termination("User requested")
    execution["status"] = "terminated"
    execution["completed_at"] = datetime.now(timezone.utc).isoformat()
    store.save_execution(execution)

    return True


# =============================================================================
# Templates
# =============================================================================


async def list_templates(
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """List available workflow templates."""
    store = _get_store()
    templates = store.list_templates(category=category, tags=tags)
    return [t.to_dict() for t in templates]


async def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Get a workflow template by ID."""
    store = _get_store()
    template = store.get_template(template_id)
    return template.to_dict() if template else None


async def create_workflow_from_template(
    template_id: str,
    name: str,
    tenant_id: str = "default",
    created_by: str = "",
    customizations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    """Register a workflow as a template."""
    workflow.is_template = True
    store = _get_store()
    store.save_template(workflow)


# =============================================================================
# Human Approvals
# =============================================================================


async def list_pending_approvals(
    workflow_id: Optional[str] = None,
    tenant_id: str = "default",
) -> List[Dict[str, Any]]:
    """List pending human approvals."""
    from aragora.workflow.nodes.human_checkpoint import get_pending_approvals

    approvals = get_pending_approvals(workflow_id)
    return [a.to_dict() for a in approvals]


async def resolve_approval(
    request_id: str,
    status: str,
    responder_id: str,
    notes: str = "",
    checklist_updates: Optional[Dict[str, bool]] = None,
) -> bool:
    """Resolve a human approval request."""
    from aragora.workflow.nodes.human_checkpoint import (
        resolve_approval as _resolve,
        ApprovalStatus,
    )

    try:
        approval_status = ApprovalStatus[status.upper()]
    except KeyError:
        raise ValueError(f"Invalid status: {status}")

    return _resolve(request_id, approval_status, responder_id, notes, checklist_updates)


async def get_approval(request_id: str) -> Optional[Dict[str, Any]]:
    """Get an approval request by ID."""
    from aragora.workflow.nodes.human_checkpoint import get_approval_request

    approval = get_approval_request(request_id)
    return approval.to_dict() if approval else None


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


# Register built-in templates (Python-defined)
register_template(_create_contract_review_template())
register_template(_create_code_review_template())


# Load YAML templates from disk
def _load_yaml_templates() -> None:
    """Load workflow templates from YAML files into persistent store."""
    try:
        from aragora.workflow.template_loader import load_templates

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
            logger.info(f"Loaded {loaded} new YAML templates into database")
    except ImportError as e:
        logger.debug(f"Template loader not available: {e}")
    except (OSError, IOError) as e:
        logger.warning(f"Failed to read YAML templates from disk: {e}")
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse YAML templates: {e}")


# Load templates on module import
_load_yaml_templates()


# =============================================================================
# HTTP Route Handlers (for integration with unified_server)
# =============================================================================


class WorkflowHandlers:
    """HTTP handlers for workflow API."""

    @staticmethod
    async def handle_list_workflows(params: Dict[str, Any]) -> Dict[str, Any]:
        """GET /api/workflows"""
        return await list_workflows(
            tenant_id=params.get("tenant_id", "default"),
            category=params.get("category"),
            tags=params.get("tags"),
            search=params.get("search"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
        )

    @staticmethod
    async def handle_get_workflow(
        workflow_id: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """GET /api/workflows/:id"""
        return await get_workflow(workflow_id, params.get("tenant_id", "default"))

    @staticmethod
    async def handle_create_workflow(
        data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/workflows"""
        return await create_workflow(
            data,
            tenant_id=params.get("tenant_id", "default"),
            created_by=params.get("user_id", ""),
        )

    @staticmethod
    async def handle_update_workflow(
        workflow_id: str, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """PUT /api/workflows/:id"""
        return await update_workflow(workflow_id, data, params.get("tenant_id", "default"))

    @staticmethod
    async def handle_delete_workflow(workflow_id: str, params: Dict[str, Any]) -> bool:
        """DELETE /api/workflows/:id"""
        return await delete_workflow(workflow_id, params.get("tenant_id", "default"))

    @staticmethod
    async def handle_execute_workflow(
        workflow_id: str, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/workflows/:id/execute"""
        return await execute_workflow(
            workflow_id,
            inputs=data.get("inputs"),
            tenant_id=params.get("tenant_id", "default"),
        )

    @staticmethod
    async def handle_list_templates(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """GET /api/workflow-templates"""
        return await list_templates(
            category=params.get("category"),
            tags=params.get("tags"),
        )

    @staticmethod
    async def handle_list_approvals(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """GET /api/workflow-approvals"""
        return await list_pending_approvals(
            workflow_id=params.get("workflow_id"),
            tenant_id=params.get("tenant_id", "default"),
        )

    @staticmethod
    async def handle_resolve_approval(
        request_id: str, data: Dict[str, Any], params: Dict[str, Any]
    ) -> bool:
        """POST /api/workflow-approvals/:id/resolve"""
        return await resolve_approval(
            request_id,
            status=data.get("status", "approved"),
            responder_id=params.get("user_id", ""),
            notes=data.get("notes", ""),
            checklist_updates=data.get("checklist"),
        )


# =============================================================================
# BaseHandler Integration for Unified Server
# =============================================================================

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    PaginatedHandlerMixin,
    error_response,
    json_response,
    get_int_param,
    get_string_param,
)


class WorkflowHandler(BaseHandler, PaginatedHandlerMixin):
    """
    HTTP request handler for workflow API endpoints.

    Integrates with the unified server's dispatch mechanism using BaseHandler.
    Provides REST API for managing and executing workflows.

    Routes:
        GET    /api/workflows              - List workflows
        POST   /api/workflows              - Create workflow
        GET    /api/workflows/{id}         - Get workflow details
        PATCH  /api/workflows/{id}         - Update workflow
        DELETE /api/workflows/{id}         - Delete workflow
        POST   /api/workflows/{id}/execute - Execute workflow
        POST   /api/workflows/{id}/simulate - Dry-run workflow
        GET    /api/workflows/{id}/status  - Get execution status
        GET    /api/workflows/{id}/versions - Get version history
        GET    /api/workflow-templates     - List templates
        POST   /api/workflow-approvals/{id}/resolve - Resolve approval
    """

    ROUTES = [
        "/api/v1/workflows",
        "/api/v1/workflows/*",
        "/api/v1/workflow-templates",
        "/api/v1/workflow-approvals",
        "/api/v1/workflow-approvals/*",
        "/api/v1/workflow-executions",
        "/api/v1/workflow-executions/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return (
            path.startswith("/api/v1/workflows")
            or path.startswith("/api/v1/workflow-templates")
            or path.startswith("/api/v1/workflow-approvals")
            or path.startswith("/api/v1/workflow-executions")
        )

    def _get_auth_context(self, handler: Any) -> Optional["AuthorizationContext"]:
        """Build AuthorizationContext from validated JWT token.

        SECURITY: Only accepts JWT-based authentication. Header-based auth
        has been removed to prevent identity spoofing and privilege escalation.

        Returns:
            AuthorizationContext if JWT is valid and RBAC available,
            None if RBAC not available (allows request in dev mode),
            "unauthenticated" sentinel if JWT is missing/invalid
        """
        if not RBAC_AVAILABLE:
            return None

        # JWT authentication required (secure)
        jwt_context = extract_user_from_request(handler)
        if not jwt_context.authenticated or not jwt_context.user_id:
            logger.warning(
                "workflows: JWT authentication required. " "Request rejected - no valid token."
            )
            # Return special sentinel to distinguish from RBAC-not-available
            return "unauthenticated"  # type: ignore

        roles = {jwt_context.role} if jwt_context.role else {"member"}
        permissions: set[str] = set()
        for role in roles:
            permissions |= get_role_permissions(role, include_inherited=True)

        return AuthorizationContext(
            user_id=jwt_context.user_id,
            org_id=jwt_context.org_id,
            roles=roles,
            permissions=permissions,
        )

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: Optional[str] = None
    ) -> Optional[HandlerResult]:
        """Check if the request has the required permission.

        Returns None if allowed, or an error response if denied.
        If RBAC is not available, allows the request (development mode).
        """
        if not RBAC_AVAILABLE:
            logger.debug(f"RBAC not available, allowing {permission_key}")
            return None

        context = self._get_auth_context(handler)
        if context is None:
            return None  # RBAC not configured

        # Handle unauthenticated sentinel
        if context == "unauthenticated":
            return error_response("Authentication required. Please provide a valid JWT token.", 401)

        try:
            decision = check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                logger.warning(
                    f"Permission denied: {permission_key} for user {context.user_id}: {decision.reason}"
                )
                record_rbac_check(permission_key, allowed=False, handler="WorkflowHandler")
                return error_response(f"Permission denied: {decision.reason}", 403)
            record_rbac_check(permission_key, allowed=True)
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {permission_key} for user {context.user_id}: {e}")
            record_rbac_check(permission_key, allowed=False, handler="WorkflowHandler")
            return error_response(f"Permission denied: {str(e)}", 403)
        return None

    def _get_tenant_id(self, handler: Any, query_params: dict) -> str:
        """Extract tenant_id from auth context or query params."""
        if RBAC_AVAILABLE:
            context = self._get_auth_context(handler)
            # Only use org_id from valid AuthorizationContext (not "unauthenticated" sentinel)
            if (
                context
                and context != "unauthenticated"
                and hasattr(context, "org_id")
                and context.org_id
            ):
                return context.org_id
        return get_string_param(query_params, "tenant_id", "default")

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if not self.can_handle(path):
            return None

        # GET /api/workflow-executions
        if path == "/api/v1/workflow-executions":
            return self._handle_list_executions(query_params, handler)

        # GET /api/workflow-executions/{id}
        if path.startswith("/api/v1/workflow-executions/"):
            execution_id = path.split("/")[-1]
            if execution_id:
                return self._handle_get_execution(execution_id, query_params, handler)

        # GET /api/workflow-templates
        if path == "/api/v1/workflow-templates":
            return self._handle_list_templates(query_params, handler)

        # GET /api/workflow-approvals
        if path == "/api/v1/workflow-approvals":
            return self._handle_list_approvals(query_params, handler)

        # GET /api/workflows/{id}/versions
        if path.endswith("/versions"):
            workflow_id = self._extract_id(path, suffix="/versions")
            if workflow_id:
                return self._handle_get_versions(workflow_id, query_params, handler)

        # GET /api/workflows/{id}/status
        if path.endswith("/status"):
            workflow_id = self._extract_id(path, suffix="/status")
            if workflow_id:
                return self._handle_get_status(workflow_id, query_params, handler)

        # GET /api/workflows/{id}
        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_get_workflow(workflow_id, query_params, handler)

        # GET /api/workflows
        if path == "/api/v1/workflows":
            return self._handle_list_workflows(query_params, handler)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if not self.can_handle(path):
            return None

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # POST /api/workflows/{id}/execute
        if path.endswith("/execute"):
            workflow_id = self._extract_id(path, suffix="/execute")
            if workflow_id:
                return self._handle_execute(workflow_id, body, query_params, handler)

        # POST /api/workflows/{id}/simulate
        if path.endswith("/simulate"):
            workflow_id = self._extract_id(path, suffix="/simulate")
            if workflow_id:
                return self._handle_simulate(workflow_id, body, query_params, handler)

        # POST /api/workflow-approvals/{id}/resolve
        if "/workflow-approvals/" in path and path.endswith("/resolve"):
            parts = path.split("/")
            if len(parts) >= 4:
                request_id = parts[3]
                return self._handle_resolve_approval(request_id, body, query_params, handler)

        # POST /api/workflows (create)
        if path == "/api/v1/workflows":
            return self._handle_create_workflow(body, query_params, handler)

        return None

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle PATCH requests."""
        if not self.can_handle(path):
            return None

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_update_workflow(workflow_id, body, query_params, handler)

        return None

    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle PUT requests (same as PATCH for workflows)."""
        return self.handle_patch(path, query_params, handler)

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        if not self.can_handle(path):
            return None

        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_delete_workflow(workflow_id, query_params, handler)

        return None

    # =========================================================================
    # Path Helpers
    # =========================================================================

    def _extract_id(self, path: str, suffix: str = "") -> Optional[str]:
        """Extract workflow ID from path."""
        if suffix and path.endswith(suffix):
            path = path[: -len(suffix)]

        parts = path.strip("/").split("/")
        # /api/v1/workflows/{id} (4 parts: api, v1, workflows, id)
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "workflows":
            return parts[3]
        return None

    # =========================================================================
    # Request Handlers
    # =========================================================================

    def _handle_list_workflows(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflows."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.read"):
            return error

        try:
            limit, offset = self.get_pagination(query_params)
            tenant_id = self._get_tenant_id(handler, query_params)
            result = _run_async(
                list_workflows(
                    tenant_id=tenant_id,
                    category=get_string_param(query_params, "category", None),
                    search=get_string_param(query_params, "search", None),
                    limit=limit,
                    offset=offset,
                )
            )
            return json_response(result)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error listing workflows: {e}")
            return error_response("Internal data error", 500)

    def _handle_get_workflow(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = _run_async(
                get_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if result:
                return json_response(result)
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error getting workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_create_workflow(
        self, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.create"):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            # Get user_id from auth context if available
            auth_context = self._get_auth_context(handler) if RBAC_AVAILABLE else None
            created_by = (
                auth_context.user_id
                if auth_context
                else get_string_param(query_params, "user_id", "")
            )

            result = _run_async(
                create_workflow(
                    body,
                    tenant_id=tenant_id,
                    created_by=created_by,
                )
            )
            return json_response(result, status=201)
        except ValueError as e:
            return error_response(str(e), 400)
        except (OSError, IOError) as e:
            logger.error(f"Storage error creating workflow: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error creating workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_update_workflow(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle PATCH /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.update", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = _run_async(
                update_workflow(
                    workflow_id,
                    body,
                    tenant_id=tenant_id,
                )
            )
            if result:
                return json_response(result)
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except ValueError as e:
            return error_response(str(e), 400)
        except (OSError, IOError) as e:
            logger.error(f"Storage error updating workflow: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error updating workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_delete_workflow(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.delete", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            deleted = _run_async(
                delete_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if deleted:
                return json_response({"deleted": True, "id": workflow_id})
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except (OSError, IOError) as e:
            logger.error(f"Storage error deleting workflow: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error deleting workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_execute(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows/{id}/execute."""
        # RBAC check - execute requires specific permission
        if error := self._check_permission(handler, "workflows.execute", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = _run_async(
                execute_workflow(
                    workflow_id,
                    inputs=body.get("inputs"),
                    tenant_id=tenant_id,
                )
            )
            return json_response(result)
        except ValueError as e:
            return error_response(str(e), 404)
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error executing workflow: {e}")
            return error_response("Execution service unavailable", 503)
        except (OSError, IOError) as e:
            logger.error(f"Storage error executing workflow: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error executing workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_simulate(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows/{id}/simulate (dry-run)."""
        # RBAC check - simulate only needs read permission
        if error := self._check_permission(handler, "workflows.read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            workflow_dict = _run_async(
                get_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if not workflow_dict:
                return error_response(f"Workflow not found: {workflow_id}", 404)

            workflow = WorkflowDefinition.from_dict(workflow_dict)
            is_valid, errors = workflow.validate()

            # Build execution plan
            plan = []
            visited = set()
            current = workflow.entry_step

            while current and current not in visited:
                step = workflow.get_step(current)
                if step:
                    plan.append(
                        {
                            "step_id": step.id,
                            "step_name": step.name,
                            "step_type": step.step_type,
                            "optional": step.optional,
                            "timeout": step.timeout_seconds,
                        }
                    )
                    visited.add(current)
                    current = step.next_steps[0] if step.next_steps else None
                else:
                    break

            return json_response(
                {
                    "workflow_id": workflow_id,
                    "is_valid": is_valid,
                    "validation_errors": errors,
                    "execution_plan": plan,
                    "estimated_steps": len(workflow.steps),
                }
            )

        except (OSError, IOError) as e:
            logger.error(f"Storage error simulating workflow: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error simulating workflow: {e}")
            return error_response("Internal data error", 500)

    def _handle_get_status(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}/status."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.read", workflow_id):
            return error

        try:
            executions = _run_async(list_executions(workflow_id=workflow_id, limit=1))
            if executions:
                return json_response(executions[0])
            return json_response(
                {
                    "workflow_id": workflow_id,
                    "status": "no_executions",
                    "message": "No executions found for this workflow",
                }
            )
        except (OSError, IOError) as e:
            logger.error(f"Storage error getting workflow status: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error getting workflow status: {e}")
            return error_response("Internal data error", 500)

    def _handle_get_versions(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}/versions."""
        # RBAC check
        if error := self._check_permission(handler, "workflows.read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            versions = _run_async(
                get_workflow_versions(
                    workflow_id,
                    tenant_id=tenant_id,
                    limit=get_int_param(query_params, "limit", 20),
                )
            )
            return json_response({"versions": versions, "workflow_id": workflow_id})
        except (OSError, IOError) as e:
            logger.error(f"Storage error getting workflow versions: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error getting workflow versions: {e}")
            return error_response("Internal data error", 500)

    def _handle_list_templates(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-templates."""
        # RBAC check - templates require read permission
        if error := self._check_permission(handler, "workflows.read"):
            return error

        try:
            templates = _run_async(
                list_templates(
                    category=get_string_param(query_params, "category", None),
                )
            )
            return json_response({"templates": templates, "count": len(templates)})
        except (OSError, IOError) as e:
            logger.error(f"Storage error listing templates: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error listing templates: {e}")
            return error_response("Internal data error", 500)

    def _handle_list_approvals(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-approvals."""
        # RBAC check - approvals require read permission
        if error := self._check_permission(handler, "workflows.read"):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            approvals = _run_async(
                list_pending_approvals(
                    workflow_id=get_string_param(query_params, "workflow_id", None),
                    tenant_id=tenant_id,
                )
            )
            return json_response({"approvals": approvals, "count": len(approvals)})
        except (OSError, IOError) as e:
            logger.error(f"Storage error listing approvals: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error listing approvals: {e}")
            return error_response("Internal data error", 500)

    def _handle_list_executions(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-executions.

        Returns all workflow executions across all workflows, filtered by status.
        Used by the runtime monitoring dashboard.
        """
        # RBAC check
        if error := self._check_permission(handler, "workflows.read"):
            return error

        try:
            status_filter = get_string_param(query_params, "status", None)
            workflow_id = get_string_param(query_params, "workflow_id", None)
            limit = get_int_param(query_params, "limit", 50)
            tenant_id = self._get_tenant_id(handler, query_params)

            executions = _run_async(
                list_executions(
                    workflow_id=workflow_id,
                    tenant_id=tenant_id,
                    limit=limit,
                )
            )

            # Apply status filter if provided
            if status_filter:
                executions = [e for e in executions if e.get("status") == status_filter]

            return json_response(
                {
                    "executions": executions,
                    "count": len(executions),
                }
            )
        except (OSError, IOError) as e:
            logger.error(f"Storage error listing executions: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error listing executions: {e}")
            return error_response("Internal data error", 500)

    def _handle_get_execution(
        self, execution_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflow-executions/{id}.

        Returns a single execution's details and status.
        Used for polling workflow execution progress.
        """
        # RBAC check
        if error := self._check_permission(handler, "workflows.read", execution_id):
            return error

        try:
            execution = _run_async(get_execution(execution_id))
            if not execution:
                return error_response(f"Execution not found: {execution_id}", 404)

            return json_response(execution)
        except (OSError, IOError) as e:
            logger.error(f"Storage error getting execution: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error getting execution: {e}")
            return error_response("Internal data error", 500)

    def _handle_resolve_approval(
        self, request_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflow-approvals/{id}/resolve."""
        # RBAC check - resolving approvals requires approve permission
        if error := self._check_permission(handler, "workflows.approve", request_id):
            return error

        try:
            # Get responder from auth context if available
            auth_context = self._get_auth_context(handler) if RBAC_AVAILABLE else None
            responder_id = (
                auth_context.user_id
                if auth_context
                else get_string_param(query_params, "user_id", "")
            )

            resolved = _run_async(
                resolve_approval(
                    request_id,
                    status=body.get("status", "approved"),
                    responder_id=responder_id,
                    notes=body.get("notes", ""),
                    checklist_updates=body.get("checklist"),
                )
            )
            if resolved:
                return json_response({"resolved": True, "request_id": request_id})
            return error_response(f"Approval request not found: {request_id}", 404)
        except ValueError as e:
            return error_response(str(e), 400)
        except (OSError, IOError) as e:
            logger.error(f"Storage error resolving approval: {e}")
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Data error resolving approval: {e}")
            return error_response("Internal data error", 500)
