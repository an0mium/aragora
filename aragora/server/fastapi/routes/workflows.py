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
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
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
    if isinstance(wf, dict):
        return WorkflowSummary(
            id=wf.get("id", wf.get("workflow_id", "")),
            name=wf.get("name", ""),
            description=wf.get("description", ""),
            status=wf.get("status", "pending"),
            template=wf.get("template"),
            created_at=wf.get("created_at"),
            updated_at=wf.get("updated_at"),
            node_count=len(wf.get("nodes", [])),
        )
    return WorkflowSummary(
        id=getattr(wf, "id", getattr(wf, "workflow_id", "")),
        name=getattr(wf, "name", ""),
        description=getattr(wf, "description", ""),
        status=getattr(wf, "status", "pending"),
        template=getattr(wf, "template", None),
        created_at=str(getattr(wf, "created_at", "")) if hasattr(wf, "created_at") else None,
        updated_at=str(getattr(wf, "updated_at", "")) if hasattr(wf, "updated_at") else None,
        node_count=len(getattr(wf, "nodes", [])),
    )


def _workflow_to_detail(wf: Any) -> WorkflowDetail:
    """Convert a workflow object to full detail."""
    if isinstance(wf, dict):
        nodes = []
        for n in wf.get("nodes", []):
            if isinstance(n, dict):
                nodes.append(WorkflowNodeDetail(**{k: n[k] for k in n if k in WorkflowNodeDetail.model_fields}))
            else:
                nodes.append(WorkflowNodeDetail(
                    id=getattr(n, "id", ""),
                    type=getattr(n, "type", ""),
                    name=getattr(n, "name", ""),
                    status=getattr(n, "status", "pending"),
                    config=getattr(n, "config", {}),
                ))

        return WorkflowDetail(
            id=wf.get("id", wf.get("workflow_id", "")),
            name=wf.get("name", ""),
            description=wf.get("description", ""),
            status=wf.get("status", "pending"),
            template=wf.get("template"),
            nodes=nodes,
            edges=wf.get("edges", []),
            config=wf.get("config", {}),
            created_at=wf.get("created_at"),
            updated_at=wf.get("updated_at"),
            started_at=wf.get("started_at"),
            completed_at=wf.get("completed_at"),
            result=wf.get("result"),
            error=wf.get("error"),
        )

    nodes = []
    for n in getattr(wf, "nodes", []):
        if isinstance(n, dict):
            nodes.append(WorkflowNodeDetail(**{k: n[k] for k in n if k in WorkflowNodeDetail.model_fields}))
        else:
            nodes.append(WorkflowNodeDetail(
                id=getattr(n, "id", ""),
                type=getattr(n, "type", ""),
                name=getattr(n, "name", ""),
                status=getattr(n, "status", "pending"),
                config=getattr(n, "config", {}),
            ))

    return WorkflowDetail(
        id=getattr(wf, "id", getattr(wf, "workflow_id", "")),
        name=getattr(wf, "name", ""),
        description=getattr(wf, "description", ""),
        status=getattr(wf, "status", "pending"),
        template=getattr(wf, "template", None),
        nodes=nodes,
        edges=[
            e if isinstance(e, dict) else e.__dict__
            for e in getattr(wf, "edges", [])
        ],
        config=getattr(wf, "config", {}),
        created_at=str(getattr(wf, "created_at", "")) if hasattr(wf, "created_at") else None,
        updated_at=str(getattr(wf, "updated_at", "")) if hasattr(wf, "updated_at") else None,
        started_at=str(getattr(wf, "started_at", "")) if hasattr(wf, "started_at") else None,
        completed_at=str(getattr(wf, "completed_at", "")) if hasattr(wf, "completed_at") else None,
        result=getattr(wf, "result", None),
        error=getattr(wf, "error", None),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request,
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

        if hasattr(engine, "list_workflows"):
            workflows_raw = engine.list_workflows(limit=limit, offset=offset, status=status)
        elif hasattr(engine, "list"):
            all_wf = engine.list()
            if status:
                all_wf = [
                    w for w in all_wf
                    if (w.get("status") if isinstance(w, dict) else getattr(w, "status", ""))
                    == status
                ]
            workflows_raw = all_wf[offset: offset + limit]

        # Get total count
        if hasattr(engine, "count_workflows"):
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
                    templates.append(TemplateSummary(
                        name=t.get("name", t.get("id", "")),
                        description=t.get("description", ""),
                        category=t.get("category", ""),
                        node_count=len(t.get("nodes", [])),
                        tags=t.get("tags", []),
                    ))
                else:
                    templates.append(TemplateSummary(
                        name=getattr(t, "name", getattr(t, "id", "")),
                        description=getattr(t, "description", ""),
                        category=getattr(t, "category", ""),
                        node_count=len(getattr(t, "nodes", [])),
                        tags=getattr(t, "tags", []),
                    ))
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Workflow templates not available: %s", e)

        return TemplateListResponse(templates=templates, total=len(templates))

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing workflow templates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list workflow templates")


@router.get("/workflows/{workflow_id}", response_model=WorkflowDetail)
async def get_workflow(
    workflow_id: str,
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

        if hasattr(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif hasattr(engine, "get"):
            wf = engine.get(workflow_id)

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

        wf_data: dict[str, Any] = {
            "id": workflow_id,
            "name": body.name,
            "description": body.description,
            "template": body.template,
            "nodes": body.nodes,
            "edges": body.edges,
            "config": body.config,
            "status": "pending",
        }

        # Load template if specified
        if body.template:
            try:
                from aragora.workflow.templates import get_template

                template = get_template(body.template)
                if template:
                    # Merge template defaults with user overrides
                    template_data = template if isinstance(template, dict) else template.__dict__
                    if not body.nodes:
                        wf_data["nodes"] = template_data.get("nodes", [])
                    if not body.edges:
                        wf_data["edges"] = template_data.get("edges", [])
            except (ImportError, RuntimeError, ValueError) as e:
                logger.debug("Template %s not available: %s", body.template, e)

        # Create the workflow
        if hasattr(engine, "create_workflow"):
            created = engine.create_workflow(wf_data)
            if isinstance(created, dict) and "id" in created:
                workflow_id = created["id"]
        elif hasattr(engine, "create"):
            created = engine.create(wf_data)
            if isinstance(created, dict) and "id" in created:
                workflow_id = created["id"]

        logger.info("Created workflow: %s (name=%s)", workflow_id, body.name)

        return CreateWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            workflow=WorkflowDetail(
                id=workflow_id,
                name=body.name,
                description=body.description,
                status="pending",
                template=body.template,
                nodes=[],
                edges=body.edges,
                config=body.config,
            ),
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating workflow: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create workflow")


@router.post("/workflows/{workflow_id}/execute", response_model=ExecuteWorkflowResponse)
async def execute_workflow(
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
        if hasattr(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif hasattr(engine, "get"):
            wf = engine.get(workflow_id)

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        import uuid

        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        # Execute the workflow
        if hasattr(engine, "execute"):
            result = engine.execute(
                workflow_id,
                input_data=body.input_data,
                async_execution=body.async_execution,
            )
            if isinstance(result, dict) and "execution_id" in result:
                execution_id = result["execution_id"]
        elif hasattr(engine, "run"):
            result = engine.run(workflow_id, input_data=body.input_data)
            if isinstance(result, dict) and "execution_id" in result:
                execution_id = result["execution_id"]

        logger.info("Executing workflow %s (execution_id=%s)", workflow_id, execution_id)

        return ExecuteWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            execution_id=execution_id,
            status="running" if body.async_execution else "completed",
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

        if hasattr(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif hasattr(engine, "get"):
            wf = engine.get(workflow_id)

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        # Extract status info
        if isinstance(wf, dict):
            status = wf.get("status", "pending")
            nodes = wf.get("nodes", [])
            completed = [
                n.get("id", "") for n in nodes
                if (n.get("status") if isinstance(n, dict) else getattr(n, "status", ""))
                == "completed"
            ]
            failed = [
                n.get("id", "") for n in nodes
                if (n.get("status") if isinstance(n, dict) else getattr(n, "status", ""))
                == "failed"
            ]
            total_nodes = len(nodes)
            progress = len(completed) / total_nodes if total_nodes > 0 else 0.0
        else:
            status = getattr(wf, "status", "pending")
            nodes = getattr(wf, "nodes", [])
            completed = [
                getattr(n, "id", "") for n in nodes
                if getattr(n, "status", "") == "completed"
            ]
            failed = [
                getattr(n, "id", "") for n in nodes
                if getattr(n, "status", "") == "failed"
            ]
            total_nodes = len(nodes)
            progress = len(completed) / total_nodes if total_nodes > 0 else 0.0

        # Find current executing node
        current_node = None
        for n in nodes:
            n_status = n.get("status") if isinstance(n, dict) else getattr(n, "status", "")
            if n_status == "running":
                current_node = n.get("id") if isinstance(n, dict) else getattr(n, "id", None)
                break

        started_at = (
            wf.get("started_at") if isinstance(wf, dict)
            else (str(getattr(wf, "started_at", "")) if hasattr(wf, "started_at") else None)
        )
        completed_at = (
            wf.get("completed_at") if isinstance(wf, dict)
            else (str(getattr(wf, "completed_at", "")) if hasattr(wf, "completed_at") else None)
        )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=status,
            progress=round(progress, 3),
            current_node=current_node,
            completed_nodes=completed,
            failed_nodes=failed,
            started_at=started_at,
            completed_at=completed_at,
            error=wf.get("error") if isinstance(wf, dict) else getattr(wf, "error", None),
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting workflow status %s: %s", workflow_id, e)
        raise HTTPException(status_code=500, detail="Failed to get workflow status")


# =============================================================================
# New Endpoints (History, Approve)
# =============================================================================


@router.get(
    "/workflows/{workflow_id}/history", response_model=WorkflowHistoryResponse
)
async def get_workflow_history(
    workflow_id: str,
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
    engine=Depends(get_workflow_engine),
) -> WorkflowHistoryResponse:
    """Get execution history for a workflow."""
    if not engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        # Verify workflow exists
        wf = None
        if hasattr(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif hasattr(engine, "get"):
            wf = engine.get(workflow_id)

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        executions: list[HistoryEntry] = []

        # Try to get execution history
        raw_history: list[Any] = []
        if hasattr(engine, "get_execution_history"):
            raw_history = engine.get_execution_history(workflow_id, limit=limit)
        elif hasattr(engine, "get_history"):
            raw_history = engine.get_history(workflow_id, limit=limit)
        elif hasattr(engine, "list_executions"):
            raw_history = engine.list_executions(workflow_id, limit=limit)

        for entry in raw_history:
            if isinstance(entry, dict):
                executions.append(HistoryEntry(
                    execution_id=entry.get("execution_id", entry.get("id", "")),
                    status=entry.get("status", "completed"),
                    started_at=entry.get("started_at"),
                    completed_at=entry.get("completed_at"),
                    duration_seconds=entry.get("duration_seconds", 0.0),
                    result=entry.get("result"),
                    error=entry.get("error"),
                ))
            else:
                executions.append(HistoryEntry(
                    execution_id=getattr(
                        entry, "execution_id", getattr(entry, "id", "")
                    ),
                    status=getattr(entry, "status", "completed"),
                    started_at=str(getattr(entry, "started_at", ""))
                    if hasattr(entry, "started_at") else None,
                    completed_at=str(getattr(entry, "completed_at", ""))
                    if hasattr(entry, "completed_at") else None,
                    duration_seconds=getattr(entry, "duration_seconds", 0.0),
                    result=getattr(entry, "result", None),
                    error=getattr(entry, "error", None),
                ))

        return WorkflowHistoryResponse(
            workflow_id=workflow_id,
            executions=executions,
            total=len(executions),
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(
            "Error getting workflow history %s: %s", workflow_id, e
        )
        raise HTTPException(
            status_code=500, detail="Failed to get workflow history"
        )


@router.post(
    "/workflows/{workflow_id}/approve", response_model=ApproveStepResponse
)
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
        if hasattr(engine, "get_workflow"):
            wf = engine.get_workflow(workflow_id)
        elif hasattr(engine, "get"):
            wf = engine.get(workflow_id)

        if not wf:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        # Try to approve the step
        approved = False
        if hasattr(engine, "approve_step"):
            approved = engine.approve_step(
                workflow_id, body.step_id, comment=body.comment
            )
        elif hasattr(engine, "approve"):
            approved = engine.approve(
                workflow_id, body.step_id, comment=body.comment
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="Workflow engine does not support step approval",
            )

        logger.info(
            "Approved step %s in workflow %s", body.step_id, workflow_id
        )

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
        raise HTTPException(
            status_code=500, detail="Failed to approve workflow step"
        )
