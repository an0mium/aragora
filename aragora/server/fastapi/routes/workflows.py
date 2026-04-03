"""
Workflow Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/workflow/ (aiohttp handler)

Provides async workflow management endpoints:
- GET  /api/v2/workflows                         - List workflows
- GET  /api/v2/workflows/{workflow_id}            - Get workflow details
- POST /api/v2/workflows                          - Create workflow
- POST /api/v2/workflows/{workflow_id}/execute     - Execute workflow
- GET  /api/v2/workflows/{workflow_id}/status      - Get execution status

Migration Notes:
    This module replaces the legacy workflow handler endpoints with native
    FastAPI routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.workflows import (
    create_workflow as create_workflow_record,
    execute_workflow as execute_workflow_record,
    get_workflow as get_workflow_record,
    list_executions as list_workflow_executions,
    list_pending_approvals as list_pending_workflow_approvals,
    list_workflows as list_workflow_records,
    resolve_approval as resolve_workflow_approval,
)
from aragora.server.handlers.workflows.templates import (
    get_template as get_workflow_template_record,
)

from ..dependencies.auth import get_auth_context, require_permission
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Workflows"])


# =============================================================================
# Pydantic Models
# =============================================================================


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class WorkflowSummary(BaseModel):
    """Summary of a workflow for list views."""

    id: str
    name: str
    description: str = ""
    status: str = "pending"
    template: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    node_count: int = 0

    model_config = {"extra": "allow"}


class WorkflowListResponse(BaseModel):
    """Response for workflow listing."""

    workflows: list[WorkflowSummary]
    total: int
    limit: int
    offset: int


class WorkflowNodeDetail(BaseModel):
    """Detail for a single workflow node."""

    id: str
    type: str = ""
    name: str = ""
    status: str = "pending"
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class WorkflowDetail(BaseModel):
    """Full workflow details."""

    id: str
    name: str
    description: str = ""
    status: str = "pending"
    template: str | None = None
    nodes: list[WorkflowNodeDetail] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    model_config = {"extra": "allow"}


class CreateWorkflowRequest(BaseModel):
    """Request body for POST /workflows."""

    name: str = Field(..., min_length=1, max_length=200, description="Workflow name")
    description: str = Field("", max_length=2000, description="Workflow description")
    template: str | None = Field(None, description="Template name to base workflow on")
    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Workflow nodes")
    edges: list[dict[str, str]] = Field(default_factory=list, description="Edges between nodes")
    config: dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")


class CreateWorkflowResponse(BaseModel):
    """Response for POST /workflows."""

    success: bool
    workflow_id: str
    workflow: WorkflowDetail


class ExecuteWorkflowRequest(BaseModel):
    """Request body for POST /workflows/{workflow_id}/execute."""

    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for workflow execution"
    )
    async_execution: bool = Field(True, description="Run asynchronously (default true)")


class ExecuteWorkflowResponse(BaseModel):
    """Response for POST /workflows/{workflow_id}/execute."""

    success: bool
    workflow_id: str
    execution_id: str
    status: str = "pending"


class WorkflowStatusResponse(BaseModel):
    """Response for GET /workflows/{workflow_id}/status."""

    workflow_id: str
    status: str
    progress: float = 0.0
    current_node: str | None = None
    completed_nodes: list[str] = Field(default_factory=list)
    failed_nodes: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class TemplateSummary(BaseModel):
    """Summary of a workflow template."""

    name: str
    description: str = ""
    category: str = ""
    node_count: int = 0
    tags: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class TemplateListResponse(BaseModel):
    """Response for template listing."""

    templates: list[TemplateSummary]
    total: int


class HistoryEntry(BaseModel):
    """A single workflow execution history entry."""

    execution_id: str
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None

    model_config = {"extra": "allow"}


class WorkflowHistoryResponse(BaseModel):
    """Response for workflow execution history."""

    workflow_id: str
    executions: list[HistoryEntry]
    total: int


class ApproveStepRequest(BaseModel):
    """Request body for POST /workflows/{workflow_id}/approve."""

    step_id: str = Field(..., description="ID of the pending step to approve")
    comment: str = Field("", description="Optional approval comment")


class ApproveStepResponse(BaseModel):
    """Response for step approval."""

    success: bool
    workflow_id: str
    step_id: str
    status: str = "approved"


# =============================================================================
# Dependencies
# =============================================================================


async def get_workflow_engine(request: Request):
    """Dependency to get the workflow engine from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        engine = ctx.get("workflow_engine")
        if engine:
            return engine

    # Fall back to global workflow engine
    try:
        from aragora.workflow.engine import get_workflow_engine as _get_engine

        return _get_engine()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("Workflow engine not available: %s", e)
        return None


# =============================================================================
# Helpers
# =============================================================================


def _workflow_to_summary(wf: Any) -> WorkflowSummary:
    """Convert a workflow object to a summary."""
    steps = _workflow_steps(wf)
    if isinstance(wf, dict):
        return WorkflowSummary(
            id=wf.get("id", wf.get("workflow_id", "")),
            name=wf.get("name", ""),
            description=wf.get("description", ""),
            status=_workflow_status_value(wf),
            template=_workflow_template_name(wf),
            created_at=wf.get("created_at"),
            updated_at=wf.get("updated_at"),
            node_count=len(steps),
        )
    return WorkflowSummary(
        id=getattr(wf, "id", getattr(wf, "workflow_id", "")),
        name=getattr(wf, "name", ""),
        description=getattr(wf, "description", ""),
        status=_workflow_status_value(wf),
        template=_workflow_template_name(wf),
        created_at=str(getattr(wf, "created_at", "")) if hasattr(wf, "created_at") else None,
        updated_at=str(getattr(wf, "updated_at", "")) if hasattr(wf, "updated_at") else None,
        node_count=len(steps),
    )


def _workflow_to_detail(wf: Any) -> WorkflowDetail:
    """Convert a workflow object to full detail."""
    nodes = [_workflow_node_detail(node) for node in _workflow_steps(wf)]
    edges = [_workflow_edge_detail(edge) for edge in _workflow_edges(wf)]
    return WorkflowDetail(
        id=str(_workflow_field(wf, "id", _workflow_field(wf, "workflow_id", ""))),
        name=_workflow_field(wf, "name", ""),
        description=_workflow_field(wf, "description", ""),
        status=_workflow_status_value(wf),
        template=_workflow_template_name(wf),
        nodes=nodes,
        edges=edges,
        config=_workflow_config(wf),
        created_at=_workflow_timestamp(wf, "created_at"),
        updated_at=_workflow_timestamp(wf, "updated_at"),
        started_at=_workflow_timestamp(wf, "started_at"),
        completed_at=_workflow_timestamp(wf, "completed_at"),
        result=_workflow_field(wf, "result", None),
        error=_workflow_field(wf, "error", None),
    )


def _workflow_field(wf: Any, field: str, default: Any = None) -> Any:
    """Read a field from a dict or object workflow representation."""
    if isinstance(wf, dict):
        return wf.get(field, default)
    return getattr(wf, field, default)


def _workflow_timestamp(wf: Any, field: str) -> str | None:
    """Return a workflow timestamp as a string if present."""
    value = _workflow_field(wf, field)
    if value is None:
        return None
    return str(value)


def _workflow_metadata(wf: Any) -> dict[str, Any]:
    """Return workflow metadata when present."""
    metadata = _workflow_field(wf, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _workflow_config(wf: Any) -> dict[str, Any]:
    """Return workflow config, including metadata-backed config for persisted definitions."""
    config = _workflow_field(wf, "config")
    if isinstance(config, dict):
        return config
    metadata = _workflow_metadata(wf)
    metadata_config = metadata.get("config")
    return metadata_config if isinstance(metadata_config, dict) else {}


def _workflow_template_name(wf: Any) -> str | None:
    """Return the workflow template identifier when present."""
    template = _workflow_field(wf, "template")
    if template:
        return str(template)
    template_id = _workflow_field(wf, "template_id")
    if template_id:
        return str(template_id)
    metadata_template = _workflow_metadata(wf).get("template")
    return str(metadata_template) if metadata_template else None


def _workflow_status_value(wf: Any) -> str:
    """Return a workflow status or a default placeholder."""
    status = _workflow_field(wf, "status")
    return str(status) if status else "pending"


def _workflow_steps(wf: Any) -> list[Any]:
    """Return workflow nodes/steps across route and persisted representations."""
    steps = _workflow_field(wf, "nodes")
    if isinstance(steps, list):
        return steps
    steps = _workflow_field(wf, "steps")
    return steps if isinstance(steps, list) else []


def _workflow_edges(wf: Any) -> list[Any]:
    """Return workflow edges/transitions across route and persisted representations."""
    edges = _workflow_field(wf, "edges")
    if isinstance(edges, list):
        return edges
    transitions = _workflow_field(wf, "transitions")
    return transitions if isinstance(transitions, list) else []


def _workflow_node_detail(node: Any) -> WorkflowNodeDetail:
    """Normalize a node/step into the route response shape."""
    if isinstance(node, dict):
        return WorkflowNodeDetail(
            id=str(node.get("id", "")),
            type=str(node.get("type") or node.get("step_type") or ""),
            name=str(node.get("name", "")),
            status=str(node.get("status", "pending")),
            config=node.get("config", {}) if isinstance(node.get("config"), dict) else {},
        )
    return WorkflowNodeDetail(
        id=str(getattr(node, "id", "")),
        type=str(getattr(node, "type", getattr(node, "step_type", ""))),
        name=str(getattr(node, "name", "")),
        status=str(getattr(node, "status", "pending")),
        config=getattr(node, "config", {}) if isinstance(getattr(node, "config", {}), dict) else {},
    )


def _workflow_edge_detail(edge: Any) -> dict[str, str]:
    """Normalize an edge/transition into the route response shape."""
    if isinstance(edge, dict):
        from_step = edge.get("from") or edge.get("source") or edge.get("from_step")
        to_step = edge.get("to") or edge.get("target") or edge.get("to_step")
        result = {}
        if from_step:
            result["from"] = str(from_step)
        if to_step:
            result["to"] = str(to_step)
        label = edge.get("label")
        if label:
            result["label"] = str(label)
        return result

    from_step = getattr(edge, "from", getattr(edge, "from_step", None))
    to_step = getattr(edge, "to", getattr(edge, "to_step", None))
    result = {}
    if from_step:
        result["from"] = str(from_step)
    if to_step:
        result["to"] = str(to_step)
    label = getattr(edge, "label", None)
    if label:
        result["label"] = str(label)
    return result


def _engine_supports(engine: Any, *methods: str) -> bool:
    """Check whether an engine provides any of the requested callable methods."""
    return any(callable(getattr(engine, method, None)) for method in methods)


def _tenant_id_from_auth(auth: AuthorizationContext | None) -> str:
    """Resolve the tenant scope from auth when available."""
    org_id = getattr(auth, "org_id", None) if auth is not None else None
    return str(org_id) if org_id else "default"


def _created_by_from_auth(auth: AuthorizationContext | None) -> str:
    """Resolve a creator identifier from auth context."""
    user_id = getattr(auth, "user_id", "") if auth is not None else ""
    return "" if user_id == "anonymous" else str(user_id or "")


def _workflow_request_to_definition_payload(
    body: CreateWorkflowRequest,
) -> dict[str, Any]:
    """Convert the route payload's nodes/edges model into persisted workflow definitions."""
    next_steps: dict[str, list[str]] = defaultdict(list)
    transitions: list[dict[str, Any]] = []

    for idx, edge in enumerate(body.edges):
        from_step = edge.get("from") or edge.get("source") or edge.get("from_step")
        to_step = edge.get("to") or edge.get("target") or edge.get("to_step")
        if not from_step or not to_step:
            continue
        from_step = str(from_step)
        to_step = str(to_step)
        next_steps[from_step].append(to_step)
        transitions.append(
            {
                "id": edge.get("id") or f"tr_{from_step}_to_{to_step}_{idx}",
                "from_step": from_step,
                "to_step": to_step,
                "condition": str(edge.get("condition", "")),
                "priority": int(edge.get("priority", 0) or 0),
                "label": str(edge.get("label", "")),
            }
        )

    steps: list[dict[str, Any]] = []
    for idx, node in enumerate(body.nodes):
        step_id = str(node.get("id") or f"step_{idx + 1}")
        steps.append(
            {
                "id": step_id,
                "name": str(node.get("name") or step_id),
                "step_type": str(node.get("step_type") or node.get("type") or "task"),
                "config": node.get("config", {}) if isinstance(node.get("config"), dict) else {},
                "description": str(node.get("description", "")),
                "next_steps": list(next_steps.get(step_id, [])),
            }
        )

    metadata: dict[str, Any] = {}
    if body.config:
        metadata["config"] = body.config
    if body.template:
        metadata["template"] = body.template

    payload: dict[str, Any] = {
        "name": body.name,
        "description": body.description,
        "steps": steps,
        "transitions": transitions,
        "metadata": metadata,
        "template_id": body.template,
    }
    if steps:
        payload["entry_step"] = steps[0]["id"]
    return payload


async def _build_create_workflow_payload(body: CreateWorkflowRequest) -> dict[str, Any]:
    """Build the persisted workflow definition payload for route requests."""
    if body.template and not body.nodes and not body.edges:
        template = await get_workflow_template_record(body.template)
        if template is None:
            raise NotFoundError(f"Workflow template {body.template} not found")

        payload = dict(template)
        payload.pop("id", None)
        payload["name"] = body.name
        if body.description:
            payload["description"] = body.description
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if body.config:
            metadata["config"] = body.config
        metadata["template"] = body.template
        payload["metadata"] = metadata
        payload["template_id"] = body.template
        return payload

    return _workflow_request_to_definition_payload(body)


def _execution_duration_seconds(execution: dict[str, Any]) -> float:
    """Convert execution duration into seconds."""
    if "duration_seconds" in execution:
        try:
            return float(execution["duration_seconds"])
        except (TypeError, ValueError):
            return 0.0
    duration_ms = execution.get("duration_ms")
    if duration_ms is None:
        return 0.0
    try:
        return float(duration_ms) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _status_from_execution(
    workflow_id: str,
    execution: dict[str, Any],
) -> WorkflowStatusResponse:
    """Build a status response from the latest persisted execution record."""
    steps = execution.get("steps", [])
    completed: list[str] = []
    failed: list[str] = []
    current_node = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id") or step.get("id") or "")
        step_status = str(step.get("status", ""))
        if step_status == "completed":
            completed.append(step_id)
        elif step_status == "failed":
            failed.append(step_id)
        elif step_status == "running" and current_node is None:
            current_node = step_id

    total_steps = len([step for step in steps if isinstance(step, dict)])
    progress = len(completed) / total_steps if total_steps > 0 else 0.0

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=str(execution.get("status", "pending")),
        progress=round(progress, 3),
        current_node=current_node,
        completed_nodes=completed,
        failed_nodes=failed,
        started_at=execution.get("started_at"),
        completed_at=execution.get("completed_at"),
        error=execution.get("error"),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request,
    auth: AuthorizationContext = Depends(get_auth_context),
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    status: str | None = Query(None, description="Filter by status"),
    engine=Depends(get_workflow_engine),
) -> WorkflowListResponse:
    """
    List all workflows with pagination.

    Returns a paginated list of workflow summaries.
    """
    if not engine:
        return WorkflowListResponse(workflows=[], total=0, limit=limit, offset=offset)

    try:
        workflows_raw: list[Any] = []

        if _engine_supports(engine, "list_workflows"):
            workflows_raw = engine.list_workflows(limit=limit, offset=offset, status=status)
        elif _engine_supports(engine, "list"):
            all_wf = engine.list()
            if status:
                all_wf = [
                    w
                    for w in all_wf
                    if (w.get("status") if isinstance(w, dict) else getattr(w, "status", ""))
                    == status
                ]
            workflows_raw = all_wf[offset : offset + limit]
        else:
            result = await list_workflow_records(
                tenant_id=_tenant_id_from_auth(auth),
                limit=limit,
                offset=offset,
            )
            workflows_raw = result.get("workflows", [])
            if status:
                workflows_raw = [wf for wf in workflows_raw if _workflow_status_value(wf) == status]

        # Get total count
        if _engine_supports(engine, "count_workflows"):
            total = engine.count_workflows(status=status)
        else:
            total = len(workflows_raw)

        workflows = [_workflow_to_summary(wf) for wf in workflows_raw]

        return WorkflowListResponse(
            workflows=workflows,
            total=total,
            limit=limit,
            offset=offset,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing workflows: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list workflows")


@router.get("/workflows/templates", response_model=TemplateListResponse)
async def list_workflow_templates(
    request: Request,
    category: str | None = Query(None, description="Filter by category"),
) -> TemplateListResponse:
    """List available workflow templates."""
    try:
        templates: list[TemplateSummary] = []

        try:
            from aragora.workflow.templates import list_templates

            raw_templates = list_templates(category=category)
            for t in raw_templates:
                if isinstance(t, dict):
                    templates.append(
                        TemplateSummary(
                            name=t.get("name", t.get("id", "")),
                            description=t.get("description", ""),
                            category=t.get("category", ""),
                            node_count=len(t.get("nodes", [])),
                            tags=t.get("tags", []),
                        )
                    )
                else:
                    templates.append(
                        TemplateSummary(
                            name=getattr(t, "name", getattr(t, "id", "")),
                            description=getattr(t, "description", ""),
                            category=getattr(t, "category", ""),
                            node_count=len(getattr(t, "nodes", [])),
                            tags=getattr(t, "tags", []),
                        )
                    )
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Workflow templates not available: %s", e)

        return TemplateListResponse(templates=templates, total=len(templates))

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing workflow templates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list workflow templates")


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
async def get_workflow(
    workflow_id: str,
    auth: AuthorizationContext = Depends(get_auth_context),
    engine=Depends(get_workflow_engine),
) -> WorkflowDetail:
    """
    Get workflow details by ID.

    Returns full workflow details including nodes, edges, and execution state.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        wf = None

        if _engine_supports(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif _engine_supports(engine, "get"):
            wf = engine.get(workflow_id)
        else:
            wf = await get_workflow_record(workflow_id, tenant_id=_tenant_id_from_auth(auth))

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        return _workflow_to_detail(wf)

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting workflow %s: %s", workflow_id, e)
        raise HTTPException(status_code=500, detail="Failed to get workflow")


@router.post("/workflows", response_model=CreateWorkflowResponse, status_code=201)
async def create_workflow(
    body: CreateWorkflowRequest,
    auth: AuthorizationContext = Depends(require_permission("workflows:write")),
    engine=Depends(get_workflow_engine),
) -> CreateWorkflowResponse:
    """
    Create a new workflow.

    Creates a workflow from scratch or from a template.
    Requires `workflows:write` permission.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        import uuid

        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        wf_data = await _build_create_workflow_payload(body)
        wf_data.setdefault("id", workflow_id)
        wf_data.setdefault("status", "pending")

        created: Any = None
        if _engine_supports(engine, "create_workflow"):
            created = engine.create_workflow(wf_data)
        elif _engine_supports(engine, "create"):
            created = engine.create(wf_data)
        else:
            created = await create_workflow_record(
                wf_data,
                tenant_id=_tenant_id_from_auth(auth),
                created_by=_created_by_from_auth(auth),
            )

        if isinstance(created, dict) and created.get("id"):
            workflow_id = str(created["id"])

        workflow_view = dict(wf_data)
        if isinstance(created, dict):
            workflow_view.update(created)
        workflow_view.setdefault("id", workflow_id)
        workflow_view.setdefault("status", "pending")
        workflow_view.setdefault("template", body.template)
        workflow_view.setdefault("config", body.config)
        if not _workflow_steps(workflow_view) and body.nodes:
            workflow_view["nodes"] = body.nodes
        if not _workflow_edges(workflow_view) and body.edges:
            workflow_view["edges"] = body.edges

        logger.info("Created workflow: %s (name=%s)", workflow_id, body.name)

        return CreateWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            workflow=_workflow_to_detail(workflow_view),
        )

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating workflow: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create workflow")


@router.post("/workflows/{workflow_id}/execute", response_model=ExecuteWorkflowResponse)
async def execute_workflow(
    request: Request,
    workflow_id: str,
    body: ExecuteWorkflowRequest,
    auth: AuthorizationContext = Depends(require_permission("workflows:execute")),
    engine=Depends(get_workflow_engine),
) -> ExecuteWorkflowResponse:
    """
    Execute a workflow.

    Starts execution of a workflow with the provided input data.
    Requires `workflows:execute` permission.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        # Verify workflow exists
        wf = None
        use_store_fallback = not _engine_supports(engine, "get_workflow", "get")
        if _engine_supports(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif _engine_supports(engine, "get"):
            wf = engine.get(workflow_id)
        else:
            wf = await get_workflow_record(workflow_id, tenant_id=_tenant_id_from_auth(auth))

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        # Execute the workflow
        if not use_store_fallback and _engine_supports(engine, "execute"):
            import uuid

            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            result = engine.execute(
                workflow_id,
                input_data=body.input_data,
                async_execution=body.async_execution,
            )
            if isinstance(result, dict) and "execution_id" in result:
                execution_id = result["execution_id"]
            status = "running" if body.async_execution else "completed"
        elif not use_store_fallback and _engine_supports(engine, "run"):
            import uuid

            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            result = engine.run(workflow_id, input_data=body.input_data)
            if isinstance(result, dict) and "execution_id" in result:
                execution_id = result["execution_id"]
            status = "completed"
        else:
            ctx = getattr(request.app.state, "context", None)
            event_emitter = ctx.get("event_emitter") if isinstance(ctx, dict) else None
            execution = await execute_workflow_record(
                workflow_id,
                inputs=body.input_data,
                tenant_id=_tenant_id_from_auth(auth),
                user_id=_created_by_from_auth(auth) or None,
                org_id=getattr(auth, "org_id", None),
                event_emitter=event_emitter,
            )
            execution_id = str(execution.get("id") or execution.get("execution_id") or "")
            status = str(execution.get("status", "completed"))

        logger.info("Executing workflow %s (execution_id=%s)", workflow_id, execution_id)

        return ExecuteWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=status,
        )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error executing workflow %s: %s", workflow_id, e)
        raise HTTPException(status_code=500, detail="Failed to execute workflow")


@router.get("/workflows/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    auth: AuthorizationContext = Depends(get_auth_context),
    engine=Depends(get_workflow_engine),
) -> WorkflowStatusResponse:
    """
    Get workflow execution status.

    Returns current execution status including progress and node states.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        wf = None

        if _engine_supports(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif _engine_supports(engine, "get"):
            wf = engine.get(workflow_id)
        else:
            wf = await get_workflow_record(workflow_id, tenant_id=_tenant_id_from_auth(auth))

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        if not _engine_supports(engine, "get_workflow", "get"):
            executions = await list_workflow_executions(
                workflow_id=workflow_id,
                tenant_id=_tenant_id_from_auth(auth),
                limit=1,
            )
            if executions:
                return _status_from_execution(workflow_id, executions[0])

        # Extract status info
        status = _workflow_status_value(wf)
        nodes = _workflow_steps(wf)
        completed = [
            _workflow_field(n, "id", "")
            for n in nodes
            if str(_workflow_field(n, "status", "")) == "completed"
        ]
        failed = [
            _workflow_field(n, "id", "")
            for n in nodes
            if str(_workflow_field(n, "status", "")) == "failed"
        ]
        total_nodes = len(nodes)
        progress = len(completed) / total_nodes if total_nodes > 0 else 0.0

        # Find current executing node
        current_node = None
        for n in nodes:
            n_status = _workflow_field(n, "status", "")
            if n_status == "running":
                current_node = _workflow_field(n, "id", None)
                break

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=status,
            progress=round(progress, 3),
            current_node=current_node,
            completed_nodes=completed,
            failed_nodes=failed,
            started_at=_workflow_timestamp(wf, "started_at"),
            completed_at=_workflow_timestamp(wf, "completed_at"),
            error=_workflow_field(wf, "error", None),
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting workflow status %s: %s", workflow_id, e)
        raise HTTPException(status_code=500, detail="Failed to get workflow status")


# =============================================================================
# New Endpoints (History, Approve)
# =============================================================================


@router.get("/workflows/{workflow_id}/history", response_model=WorkflowHistoryResponse)
async def get_workflow_history(
    workflow_id: str,
    auth: AuthorizationContext = Depends(get_auth_context),
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
    engine=Depends(get_workflow_engine),
) -> WorkflowHistoryResponse:
    """Get execution history for a workflow."""
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        # Verify workflow exists
        wf = None
        if _engine_supports(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif _engine_supports(engine, "get"):
            wf = engine.get(workflow_id)
        else:
            wf = await get_workflow_record(workflow_id, tenant_id=_tenant_id_from_auth(auth))

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        executions: list[HistoryEntry] = []

        # Try to get execution history
        raw_history: list[Any] = []
        if _engine_supports(engine, "get_execution_history"):
            raw_history = engine.get_execution_history(workflow_id, limit=limit)
        elif _engine_supports(engine, "get_history"):
            raw_history = engine.get_history(workflow_id, limit=limit)
        elif _engine_supports(engine, "list_executions"):
            raw_history = engine.list_executions(workflow_id, limit=limit)
        else:
            raw_history = await list_workflow_executions(
                workflow_id=workflow_id,
                tenant_id=_tenant_id_from_auth(auth),
                limit=limit,
            )

        for entry in raw_history:
            if isinstance(entry, dict):
                executions.append(
                    HistoryEntry(
                        execution_id=entry.get("execution_id", entry.get("id", "")),
                        status=entry.get("status", "completed"),
                        started_at=entry.get("started_at"),
                        completed_at=entry.get("completed_at"),
                        duration_seconds=_execution_duration_seconds(entry),
                        result=entry.get("result", entry.get("outputs")),
                        error=entry.get("error"),
                    )
                )
            else:
                executions.append(
                    HistoryEntry(
                        execution_id=getattr(entry, "execution_id", getattr(entry, "id", "")),
                        status=getattr(entry, "status", "completed"),
                        started_at=str(getattr(entry, "started_at", ""))
                        if hasattr(entry, "started_at")
                        else None,
                        completed_at=str(getattr(entry, "completed_at", ""))
                        if hasattr(entry, "completed_at")
                        else None,
                        duration_seconds=getattr(entry, "duration_seconds", 0.0),
                        result=getattr(entry, "result", None),
                        error=getattr(entry, "error", None),
                    )
                )

        return WorkflowHistoryResponse(
            workflow_id=workflow_id,
            executions=executions,
            total=len(executions),
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting workflow history %s: %s", workflow_id, e)
        raise HTTPException(status_code=500, detail="Failed to get workflow history")


@router.post("/workflows/{workflow_id}/approve", response_model=ApproveStepResponse)
async def approve_workflow_step(
    workflow_id: str,
    body: ApproveStepRequest,
    auth: AuthorizationContext = Depends(require_permission("workflows:execute")),
    engine=Depends(get_workflow_engine),
) -> ApproveStepResponse:
    """Approve a pending workflow step. Requires workflows:execute permission."""
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        # Verify workflow exists
        wf = None
        if _engine_supports(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif _engine_supports(engine, "get"):
            wf = engine.get(workflow_id)
        else:
            wf = await get_workflow_record(workflow_id, tenant_id=_tenant_id_from_auth(auth))

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        # Try to approve the step
        approved = False
        if _engine_supports(engine, "approve_step"):
            approved = engine.approve_step(workflow_id, body.step_id, comment=body.comment)
        elif _engine_supports(engine, "approve"):
            approved = engine.approve(workflow_id, body.step_id, comment=body.comment)
        else:
            approvals = await list_pending_workflow_approvals(
                workflow_id=workflow_id,
                tenant_id=_tenant_id_from_auth(auth),
            )
            approval = next(
                (
                    candidate
                    for candidate in approvals
                    if candidate.get("step_id") == body.step_id and candidate.get("id")
                ),
                None,
            )
            if approval is None:
                raise NotFoundError(
                    f"No pending approval found for workflow {workflow_id} step {body.step_id}"
                )
            approved = await resolve_workflow_approval(
                request_id=str(approval["id"]),
                status="approved",
                responder_id=_created_by_from_auth(auth) or "system",
                notes=body.comment,
            )

        logger.info("Approved step %s in workflow %s", body.step_id, workflow_id)

        return ApproveStepResponse(
            success=bool(approved),
            workflow_id=workflow_id,
            step_id=body.step_id,
            status="approved" if approved else "pending",
        )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(
            "Error approving step %s in workflow %s: %s",
            body.step_id,
            workflow_id,
            e,
        )
        raise HTTPException(status_code=500, detail="Failed to approve workflow step")
