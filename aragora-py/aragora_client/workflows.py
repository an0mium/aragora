"""Workflows API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class WorkflowStep(BaseModel):
    """Workflow step definition."""

    id: str
    name: str
    type: str
    config: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] | None = None


class Workflow(BaseModel):
    """Workflow model."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None
    status: str = "draft"
    steps: list[WorkflowStep] | None = None
    config: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    version: int = 1


class WorkflowTemplate(BaseModel):
    """Workflow template model."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None
    industry: str | None = None
    steps: list[WorkflowStep] | None = None
    config: dict[str, Any] | None = None
    variables: list[dict[str, Any]] | None = None


class WorkflowExecution(BaseModel):
    """Workflow execution status."""

    id: str
    workflow_id: str
    status: str = "pending"
    current_step: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None


class WorkflowCheckpoint(BaseModel):
    """Workflow checkpoint."""

    id: str
    execution_id: str
    step_id: str
    state: dict[str, Any] | None = None
    created_at: str | None = None


class WorkflowsAPI:
    """API for workflow operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # ==========================================================================
    # Workflow CRUD
    # ==========================================================================

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        category: str | None = None,
    ) -> list[Workflow]:
        """List workflows.

        Args:
            limit: Maximum number of workflows to return
            offset: Pagination offset
            status: Filter by status
            category: Filter by category

        Returns:
            List of workflows
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if category:
            params["category"] = category

        data = await self._client._get("/api/workflows", params=params)
        return [Workflow.model_validate(w) for w in data.get("workflows", [])]

    async def get(self, workflow_id: str) -> Workflow:
        """Get a workflow by ID.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow details
        """
        data = await self._client._get(f"/api/workflows/{workflow_id}")
        return Workflow.model_validate(data)

    async def create(
        self,
        name: str,
        *,
        description: str | None = None,
        category: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Workflow:
        """Create a new workflow.

        Args:
            name: Workflow name
            description: Workflow description
            category: Workflow category
            steps: Workflow step definitions
            config: Workflow configuration

        Returns:
            Created workflow
        """
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if category:
            body["category"] = category
        if steps:
            body["steps"] = steps
        if config:
            body["config"] = config

        data = await self._client._post("/api/workflows", body)
        return Workflow.model_validate(data)

    async def update(
        self,
        workflow_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Workflow:
        """Update a workflow.

        Args:
            workflow_id: Workflow ID
            name: Updated name
            description: Updated description
            steps: Updated steps
            config: Updated configuration

        Returns:
            Updated workflow
        """
        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if steps is not None:
            updates["steps"] = steps
        if config is not None:
            updates["config"] = config

        data = await self._client._put(f"/api/workflows/{workflow_id}", updates)
        return Workflow.model_validate(data)

    async def delete(self, workflow_id: str) -> None:
        """Delete a workflow.

        Args:
            workflow_id: Workflow ID
        """
        await self._client._delete(f"/api/workflows/{workflow_id}")

    # ==========================================================================
    # Workflow Execution
    # ==========================================================================

    async def execute(
        self,
        workflow_id: str,
        *,
        inputs: dict[str, Any] | None = None,
        async_execution: bool = False,
    ) -> WorkflowExecution:
        """Execute a workflow.

        Args:
            workflow_id: Workflow ID to execute
            inputs: Input parameters for the workflow
            async_execution: Whether to execute asynchronously

        Returns:
            Execution status
        """
        body: dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs
        if async_execution:
            body["async"] = True

        data = await self._client._post(f"/api/workflows/{workflow_id}/execute", body)
        return WorkflowExecution.model_validate(data)

    async def get_execution(self, execution_id: str) -> WorkflowExecution:
        """Get workflow execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Execution status
        """
        data = await self._client._get(f"/api/workflow-executions/{execution_id}")
        return WorkflowExecution.model_validate(data)

    async def list_executions(
        self,
        *,
        workflow_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WorkflowExecution]:
        """List workflow executions.

        Args:
            workflow_id: Filter by workflow ID
            status: Filter by status
            limit: Maximum number of executions to return
            offset: Pagination offset

        Returns:
            List of workflow executions
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        if status:
            params["status"] = status

        data = await self._client._get("/api/workflow-executions", params=params)
        return [WorkflowExecution.model_validate(e) for e in data.get("executions", [])]

    async def cancel_execution(self, execution_id: str) -> WorkflowExecution:
        """Cancel a running workflow execution.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            Updated execution status
        """
        data = await self._client._post(
            f"/api/workflow-executions/{execution_id}/cancel", {}
        )
        return WorkflowExecution.model_validate(data)

    # ==========================================================================
    # Templates
    # ==========================================================================

    async def list_templates(
        self,
        *,
        category: str | None = None,
        industry: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WorkflowTemplate]:
        """List workflow templates.

        Args:
            category: Filter by category
            industry: Filter by industry
            limit: Maximum number of templates to return
            offset: Pagination offset

        Returns:
            List of workflow templates
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if industry:
            params["industry"] = industry

        data = await self._client._get("/api/workflow-templates", params=params)
        return [WorkflowTemplate.model_validate(t) for t in data.get("templates", [])]

    async def get_template(self, template_id: str) -> WorkflowTemplate:
        """Get a workflow template by ID.

        Args:
            template_id: Template ID

        Returns:
            Workflow template
        """
        data = await self._client._get(f"/api/workflow-templates/{template_id}")
        return WorkflowTemplate.model_validate(data)

    async def run_template(
        self,
        template_id: str,
        *,
        variables: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> WorkflowExecution:
        """Run a workflow from a template.

        Args:
            template_id: Template ID
            variables: Template variable values
            inputs: Workflow inputs

        Returns:
            Execution status
        """
        body: dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        if inputs:
            body["inputs"] = inputs

        data = await self._client._post(
            f"/api/workflow-templates/{template_id}/run", body
        )
        return WorkflowExecution.model_validate(data)

    async def list_categories(self) -> list[str]:
        """List available workflow categories.

        Returns:
            List of category names
        """
        data = await self._client._get("/api/workflow-categories")
        return data.get("categories", [])

    async def list_patterns(self) -> list[str]:
        """List available workflow patterns.

        Returns:
            List of pattern names
        """
        data = await self._client._get("/api/workflow-patterns")
        return data.get("patterns", [])

    # ==========================================================================
    # Checkpoints
    # ==========================================================================

    async def list_checkpoints(
        self,
        execution_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WorkflowCheckpoint]:
        """List checkpoints for a workflow execution.

        Args:
            execution_id: Execution ID
            limit: Maximum number of checkpoints to return
            offset: Pagination offset

        Returns:
            List of checkpoints
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/workflow-executions/{execution_id}/checkpoints", params=params
        )
        return [
            WorkflowCheckpoint.model_validate(c) for c in data.get("checkpoints", [])
        ]

    async def restore_checkpoint(
        self,
        execution_id: str,
        checkpoint_id: str,
    ) -> WorkflowExecution:
        """Restore a workflow execution to a checkpoint.

        Args:
            execution_id: Execution ID
            checkpoint_id: Checkpoint ID to restore to

        Returns:
            Restored execution status
        """
        data = await self._client._post(
            f"/api/workflow-executions/{execution_id}/checkpoints/{checkpoint_id}/restore",
            {},
        )
        return WorkflowExecution.model_validate(data)

    # ==========================================================================
    # Versions
    # ==========================================================================

    async def list_versions(
        self,
        workflow_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Workflow]:
        """List workflow versions.

        Args:
            workflow_id: Workflow ID
            limit: Maximum number of versions to return
            offset: Pagination offset

        Returns:
            List of workflow versions
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/workflows/{workflow_id}/versions", params=params
        )
        return [Workflow.model_validate(v) for v in data.get("versions", [])]

    async def get_version(self, workflow_id: str, version: int) -> Workflow:
        """Get a specific workflow version.

        Args:
            workflow_id: Workflow ID
            version: Version number

        Returns:
            Workflow at specified version
        """
        data = await self._client._get(
            f"/api/workflows/{workflow_id}/versions/{version}"
        )
        return Workflow.model_validate(data)
