"""
Workflows API resource for the Aragora client.

Provides methods for workflow management:
- Create, list, and manage workflows
- Execute workflows and track execution status
- Work with workflow templates
- Handle approvals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class Workflow:
    """A workflow definition."""

    id: str
    name: str
    description: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    workspace_id: Optional[str] = None
    is_active: bool = True


@dataclass
class WorkflowExecution:
    """A workflow execution instance."""

    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: Optional[int] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class WorkflowTemplate:
    """A workflow template from the marketplace."""

    id: str
    name: str
    description: str
    category: str
    pattern: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    inputs_schema: Dict[str, Any] = field(default_factory=dict)
    is_public: bool = True


@dataclass
class WorkflowApproval:
    """A pending workflow approval request."""

    id: str
    workflow_id: str
    execution_id: str
    step_name: str
    approvers: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class WorkflowsAPI:
    """API interface for workflow management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Workflow CRUD
    # =========================================================================

    def list(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Workflow]:
        """
        List workflows.

        Args:
            workspace_id: Filter by workspace
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of workflows
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id

        response = self._client._get("/api/workflows", params)
        workflows = response.get("workflows", [])
        return [Workflow(**w) for w in workflows]

    async def list_async(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Workflow]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id

        response = await self._client._get_async("/api/workflows", params)
        workflows = response.get("workflows", [])
        return [Workflow(**w) for w in workflows]

    def get(self, workflow_id: str) -> Workflow:
        """Get a workflow by ID."""
        response = self._client._get(f"/api/workflows/{workflow_id}")
        return Workflow(**response)

    async def get_async(self, workflow_id: str) -> Workflow:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/workflows/{workflow_id}")
        return Workflow(**response)

    def create(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: Optional[str] = None,
        triggers: Optional[List[Dict[str, Any]]] = None,
        workspace_id: Optional[str] = None,
    ) -> Workflow:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            steps: List of workflow steps
            description: Optional description
            triggers: Optional trigger definitions
            workspace_id: Workspace to create in

        Returns:
            Created workflow
        """
        body: Dict[str, Any] = {
            "name": name,
            "steps": steps,
        }
        if description:
            body["description"] = description
        if triggers:
            body["triggers"] = triggers
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = self._client._post("/api/workflows", body)
        return Workflow(**response)

    async def create_async(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: Optional[str] = None,
        triggers: Optional[List[Dict[str, Any]]] = None,
        workspace_id: Optional[str] = None,
    ) -> Workflow:
        """Async version of create()."""
        body: Dict[str, Any] = {
            "name": name,
            "steps": steps,
        }
        if description:
            body["description"] = description
        if triggers:
            body["triggers"] = triggers
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = await self._client._post_async("/api/workflows", body)
        return Workflow(**response)

    def update(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Workflow:
        """Update a workflow."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if steps is not None:
            body["steps"] = steps
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active

        response = self._client._patch(f"/api/workflows/{workflow_id}", body)
        return Workflow(**response)

    async def update_async(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Workflow:
        """Async version of update()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if steps is not None:
            body["steps"] = steps
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active

        response = await self._client._patch_async(f"/api/workflows/{workflow_id}", body)
        return Workflow(**response)

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        self._client._delete(f"/api/workflows/{workflow_id}")
        return True

    async def delete_async(self, workflow_id: str) -> bool:
        """Async version of delete()."""
        await self._client._delete_async(f"/api/workflows/{workflow_id}")
        return True

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow_id: The workflow to execute
            inputs: Input parameters for the workflow

        Returns:
            Workflow execution instance
        """
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs

        response = self._client._post(f"/api/workflows/{workflow_id}/execute", body)
        return WorkflowExecution(
            id=response.get("execution_id", response.get("id", "")),
            workflow_id=workflow_id,
            status=WorkflowStatus(response.get("status", "running")),
            inputs=inputs or {},
        )

    async def execute_async(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Async version of execute()."""
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs

        response = await self._client._post_async(f"/api/workflows/{workflow_id}/execute", body)
        return WorkflowExecution(
            id=response.get("execution_id", response.get("id", "")),
            workflow_id=workflow_id,
            status=WorkflowStatus(response.get("status", "running")),
            inputs=inputs or {},
        )

    def get_execution(self, execution_id: str) -> WorkflowExecution:
        """Get execution details."""
        response = self._client._get(f"/api/v1/workflow-executions/{execution_id}")
        return WorkflowExecution(**response)

    async def get_execution_async(self, execution_id: str) -> WorkflowExecution:
        """Async version of get_execution()."""
        response = await self._client._get_async(f"/api/v1/workflow-executions/{execution_id}")
        return WorkflowExecution(**response)

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowExecution]:
        """List workflow executions."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        if status:
            params["status"] = status.value

        response = self._client._get("/api/v1/workflow-executions", params)
        executions = response.get("executions", [])
        return [WorkflowExecution(**e) for e in executions]

    async def list_executions_async(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowExecution]:
        """Async version of list_executions()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        if status:
            params["status"] = status.value

        response = await self._client._get_async("/api/v1/workflow-executions", params)
        executions = response.get("executions", [])
        return [WorkflowExecution(**e) for e in executions]

    def get_status(self, workflow_id: str) -> WorkflowExecution:
        """Get the latest execution status for a workflow."""
        response = self._client._get(f"/api/v1/workflows/{workflow_id}/status")
        return WorkflowExecution(**response)

    async def get_status_async(self, workflow_id: str) -> WorkflowExecution:
        """Async version of get_status()."""
        response = await self._client._get_async(f"/api/v1/workflows/{workflow_id}/status")
        return WorkflowExecution(**response)

    def simulate(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate (dry-run) a workflow without executing.

        Args:
            workflow_id: The workflow to simulate
            inputs: Input parameters

        Returns:
            Simulation result showing what would happen
        """
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs

        return self._client._post(f"/api/v1/workflows/{workflow_id}/simulate", body)

    async def simulate_async(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of simulate()."""
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs

        return await self._client._post_async(f"/api/v1/workflows/{workflow_id}/simulate", body)

    # =========================================================================
    # Templates
    # =========================================================================

    def list_templates(
        self,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowTemplate]:
        """List workflow templates."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern

        response = self._client._get("/api/workflow/templates", params)
        templates = response.get("templates", [])
        return [WorkflowTemplate(**t) for t in templates]

    async def list_templates_async(
        self,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowTemplate]:
        """Async version of list_templates()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern

        response = await self._client._get_async("/api/workflow/templates", params)
        templates = response.get("templates", [])
        return [WorkflowTemplate(**t) for t in templates]

    def get_template(self, template_id: str) -> WorkflowTemplate:
        """Get a workflow template by ID."""
        response = self._client._get(f"/api/workflow/templates/{template_id}")
        return WorkflowTemplate(**response)

    async def get_template_async(self, template_id: str) -> WorkflowTemplate:
        """Async version of get_template()."""
        response = await self._client._get_async(f"/api/workflow/templates/{template_id}")
        return WorkflowTemplate(**response)

    def run_template(
        self,
        template_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a workflow template directly.

        Args:
            template_id: The template to run
            inputs: Input parameters
            workspace_id: Workspace context

        Returns:
            Execution result
        """
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs
        if workspace_id:
            body["workspace_id"] = workspace_id

        return self._client._post(f"/api/workflow/templates/{template_id}/run", body)

    async def run_template_async(
        self,
        template_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of run_template()."""
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs
        if workspace_id:
            body["workspace_id"] = workspace_id

        return await self._client._post_async(f"/api/workflow/templates/{template_id}/run", body)

    def list_categories(self) -> List[str]:
        """List available workflow categories."""
        response = self._client._get("/api/workflow/categories")
        return response.get("categories", [])

    async def list_categories_async(self) -> List[str]:
        """Async version of list_categories()."""
        response = await self._client._get_async("/api/workflow/categories")
        return response.get("categories", [])

    def list_patterns(self) -> List[str]:
        """List available workflow patterns."""
        response = self._client._get("/api/workflow/patterns")
        return response.get("patterns", [])

    async def list_patterns_async(self) -> List[str]:
        """Async version of list_patterns()."""
        response = await self._client._get_async("/api/workflow/patterns")
        return response.get("patterns", [])

    # =========================================================================
    # Approvals
    # =========================================================================

    def list_approvals(
        self,
        workflow_id: Optional[str] = None,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowApproval]:
        """List pending workflow approvals."""
        params: Dict[str, Any] = {
            "status": status,
            "limit": limit,
            "offset": offset,
        }
        if workflow_id:
            params["workflow_id"] = workflow_id

        response = self._client._get("/api/v1/workflow-approvals", params)
        approvals = response.get("approvals", [])
        return [WorkflowApproval(**a) for a in approvals]

    async def list_approvals_async(
        self,
        workflow_id: Optional[str] = None,
        status: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowApproval]:
        """Async version of list_approvals()."""
        params: Dict[str, Any] = {
            "status": status,
            "limit": limit,
            "offset": offset,
        }
        if workflow_id:
            params["workflow_id"] = workflow_id

        response = await self._client._get_async("/api/v1/workflow-approvals", params)
        approvals = response.get("approvals", [])
        return [WorkflowApproval(**a) for a in approvals]

    def resolve_approval(
        self,
        approval_id: str,
        decision: str,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve a workflow approval.

        Args:
            approval_id: The approval to resolve
            decision: approve or reject
            comment: Optional comment

        Returns:
            Resolution result
        """
        body: Dict[str, Any] = {"decision": decision}
        if comment:
            body["comment"] = comment

        return self._client._post(f"/api/v1/workflow-approvals/{approval_id}/resolve", body)

    async def resolve_approval_async(
        self,
        approval_id: str,
        decision: str,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of resolve_approval()."""
        body: Dict[str, Any] = {"decision": decision}
        if comment:
            body["comment"] = comment

        return await self._client._post_async(
            f"/api/v1/workflow-approvals/{approval_id}/resolve", body
        )


__all__ = [
    "WorkflowsAPI",
    "Workflow",
    "WorkflowExecution",
    "WorkflowTemplate",
    "WorkflowApproval",
    "WorkflowStatus",
]
