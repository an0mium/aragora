"""
Workflows Namespace API

Provides methods for creating and executing automated workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class WorkflowsAPI:
    """
    Synchronous Workflows API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> execution = client.workflows.execute("workflow-123", {"input": "value"})
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List all workflows.

        Args:
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip

        Returns:
            List of workflows
        """
        return self._client.request(
            "GET",
            "/api/v1/workflows",
            params={"limit": limit, "offset": offset},
        )

    def get(self, workflow_id: str) -> dict[str, Any]:
        """
        Get a workflow by ID.

        Args:
            workflow_id: The workflow ID

        Returns:
            Workflow details
        """
        return self._client.request("GET", f"/api/v1/workflows/{workflow_id}")

    def create(
        self,
        name: str,
        steps: list[dict[str, Any]],
        description: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            steps: List of workflow steps
            description: Optional description
            **kwargs: Additional workflow options

        Returns:
            Created workflow
        """
        data = {"name": name, "steps": steps, **kwargs}
        if description:
            data["description"] = description

        return self._client.request("POST", "/api/v1/workflows", json=data)

    def update(
        self,
        workflow_id: str,
        **updates,
    ) -> dict[str, Any]:
        """
        Update a workflow.

        Args:
            workflow_id: The workflow ID
            **updates: Fields to update

        Returns:
            Updated workflow
        """
        return self._client.request(
            "PUT",
            f"/api/v1/workflows/{workflow_id}",
            json=updates,
        )

    def delete(self, workflow_id: str) -> dict[str, Any]:
        """
        Delete a workflow.

        Args:
            workflow_id: The workflow ID

        Returns:
            Deletion result
        """
        return self._client.request("DELETE", f"/api/v1/workflows/{workflow_id}")

    def execute(
        self,
        workflow_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow_id: The workflow ID
            inputs: Input data for the workflow

        Returns:
            Execution result with execution_id
        """
        return self._client.request(
            "POST",
            f"/api/v1/workflows/{workflow_id}/execute",
            json={"inputs": inputs or {}},
        )

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        """
        Get workflow execution status.

        Args:
            execution_id: The execution ID

        Returns:
            Execution details
        """
        return self._client.request("GET", f"/api/v1/workflows/executions/{execution_id}")

    def list_executions(
        self,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List workflow executions.

        Args:
            workflow_id: Optional filter by workflow ID
            limit: Maximum number of executions to return
            offset: Number of executions to skip

        Returns:
            List of executions
        """
        params = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id

        return self._client.request(
            "GET",
            "/api/v1/workflows/executions",
            params=params,
        )

    def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        """
        Cancel a workflow execution.

        Args:
            execution_id: The execution ID

        Returns:
            Cancellation result
        """
        return self._client.request(
            "POST",
            f"/api/v1/workflows/executions/{execution_id}/cancel",
        )

    def list_templates(self) -> dict[str, Any]:
        """
        List available workflow templates.

        Returns:
            List of templates
        """
        return self._client.request("GET", "/api/v1/workflows/templates")

    def create_from_template(
        self,
        template_id: str,
        name: str,
        **overrides,
    ) -> dict[str, Any]:
        """
        Create a workflow from a template.

        Args:
            template_id: The template ID
            name: Name for the new workflow
            **overrides: Override template defaults

        Returns:
            Created workflow
        """
        return self._client.request(
            "POST",
            f"/api/v1/workflows/templates/{template_id}/create",
            json={"name": name, **overrides},
        )


class AsyncWorkflowsAPI:
    """
    Asynchronous Workflows API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     execution = await client.workflows.execute("workflow-123", {"input": "value"})
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all workflows."""
        return await self._client.request(
            "GET",
            "/api/v1/workflows",
            params={"limit": limit, "offset": offset},
        )

    async def get(self, workflow_id: str) -> dict[str, Any]:
        """Get a workflow by ID."""
        return await self._client.request("GET", f"/api/v1/workflows/{workflow_id}")

    async def create(
        self,
        name: str,
        steps: list[dict[str, Any]],
        description: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a new workflow."""
        data = {"name": name, "steps": steps, **kwargs}
        if description:
            data["description"] = description

        return await self._client.request("POST", "/api/v1/workflows", json=data)

    async def update(
        self,
        workflow_id: str,
        **updates,
    ) -> dict[str, Any]:
        """Update a workflow."""
        return await self._client.request(
            "PUT",
            f"/api/v1/workflows/{workflow_id}",
            json=updates,
        )

    async def delete(self, workflow_id: str) -> dict[str, Any]:
        """Delete a workflow."""
        return await self._client.request("DELETE", f"/api/v1/workflows/{workflow_id}")

    async def execute(
        self,
        workflow_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow."""
        return await self._client.request(
            "POST",
            f"/api/v1/workflows/{workflow_id}/execute",
            json={"inputs": inputs or {}},
        )

    async def get_execution(self, execution_id: str) -> dict[str, Any]:
        """Get workflow execution status."""
        return await self._client.request("GET", f"/api/v1/workflows/executions/{execution_id}")

    async def list_executions(
        self,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List workflow executions."""
        params = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id

        return await self._client.request(
            "GET",
            "/api/v1/workflows/executions",
            params=params,
        )

    async def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        """Cancel a workflow execution."""
        return await self._client.request(
            "POST",
            f"/api/v1/workflows/executions/{execution_id}/cancel",
        )

    async def list_templates(self) -> dict[str, Any]:
        """List available workflow templates."""
        return await self._client.request("GET", "/api/v1/workflows/templates")

    async def create_from_template(
        self,
        template_id: str,
        name: str,
        **overrides,
    ) -> dict[str, Any]:
        """Create a workflow from a template."""
        return await self._client.request(
            "POST",
            f"/api/v1/workflows/templates/{template_id}/create",
            json={"name": name, **overrides},
        )
