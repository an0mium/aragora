"""
Workflows Namespace API

Provides methods for creating and executing automated workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations

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
        steps: _List[dict[str, Any]],
        description: str | None = None,
        **kwargs: Any,
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
        **updates: Any,
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

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        """
        Get workflow execution status.

        Args:
            execution_id: The execution ID

        Returns:
            Execution details
        """
        return self._client.request("GET", f"/api/v1/workflow-executions/{execution_id}")

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
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id

        return self._client.request(
            "GET",
            "/api/v1/workflow-executions",
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
            "DELETE",
            f"/api/v1/workflow-executions/{execution_id}",
        )

    def list_templates(self) -> dict[str, Any]:
        """
        List available workflow templates.

        Returns:
            List of templates
        """
        return self._client.request("GET", "/api/v1/workflow-templates")

    def list_library_templates(self, **params: Any) -> dict[str, Any]:
        """List workflow library templates (/api/v1/workflow/templates)."""
        return self._client.request("GET", "/api/v1/workflow/templates", params=params)

    def get_library_template(self, template_id: str) -> dict[str, Any]:
        """Get a workflow library template."""
        return self._client.request("GET", f"/api/v1/workflow/templates/{template_id}")

    def get_library_package(self, template_id: str) -> dict[str, Any]:
        """Get a workflow library template package."""
        return self._client.request("GET", f"/api/v1/workflow/templates/{template_id}/package")

    def list_template_categories(self) -> dict[str, Any]:
        """List workflow template categories."""
        return self._client.request("GET", "/api/v1/workflow/categories")

    def list_template_patterns(self) -> dict[str, Any]:
        """List workflow template patterns."""
        return self._client.request("GET", "/api/v1/workflow/patterns")

    def list_pattern_templates(self) -> dict[str, Any]:
        """List workflow pattern templates."""
        return self._client.request("GET", "/api/v1/workflow/pattern-templates")

    def get_pattern_template(self, template_id: str) -> dict[str, Any]:
        """Get a workflow pattern template by ID."""
        return self._client.request("GET", f"/api/v1/workflow/pattern-templates/{template_id}")

    # =========================================================================
    # Workflows/Templates & Workflows/Executions (alternate paths)
    # =========================================================================

    def list_workflow_templates(self, **params: Any) -> dict[str, Any]:
        """
        List workflow templates via /workflows/templates path.

        GET /api/v1/workflows/templates

        Returns:
            List of workflow templates
        """
        return self._client.request("GET", "/api/v1/workflows/templates", params=params or None)

    def get_workflow_template(self, template_id: str) -> dict[str, Any]:
        """
        Get a workflow template by ID via /workflows/templates path.

        GET /api/v1/workflows/templates/:template_id

        Args:
            template_id: Template identifier

        Returns:
            Template details
        """
        return self._client.request("GET", f"/api/v1/workflows/templates/{template_id}")

    def list_workflow_executions(
        self,
        *,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List workflow executions via /workflows/executions path.

        GET /api/v1/workflows/executions

        Args:
            workflow_id: Optional filter by workflow ID
            limit: Maximum executions to return
            offset: Pagination offset

        Returns:
            List of executions
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        return self._client.request("GET", "/api/v1/workflows/executions", params=params)

    def get_workflow_execution(self, execution_id: str) -> dict[str, Any]:
        """
        Get a workflow execution by ID via /workflows/executions path.

        GET /api/v1/workflows/executions/:execution_id

        Args:
            execution_id: Execution identifier

        Returns:
            Execution details including status and steps
        """
        return self._client.request("GET", f"/api/v1/workflows/executions/{execution_id}")

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
        steps: _List[dict[str, Any]],
        description: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new workflow."""
        data = {"name": name, "steps": steps, **kwargs}
        if description:
            data["description"] = description

        return await self._client.request("POST", "/api/v1/workflows", json=data)

    async def update(
        self,
        workflow_id: str,
        **updates: Any,
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

    async def get_execution(self, execution_id: str) -> dict[str, Any]:
        """Get workflow execution status."""
        return await self._client.request("GET", f"/api/v1/workflow-executions/{execution_id}")

    async def list_executions(
        self,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List workflow executions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id

        return await self._client.request(
            "GET",
            "/api/v1/workflow-executions",
            params=params,
        )

    async def cancel_execution(self, execution_id: str) -> dict[str, Any]:
        """Cancel a workflow execution."""
        return await self._client.request(
            "DELETE",
            f"/api/v1/workflow-executions/{execution_id}",
        )

    async def list_templates(self) -> dict[str, Any]:
        """List available workflow templates."""
        return await self._client.request("GET", "/api/v1/workflow-templates")

    async def list_library_templates(self, **params: Any) -> dict[str, Any]:
        """List workflow library templates (/api/v1/workflow/templates)."""
        return await self._client.request("GET", "/api/v1/workflow/templates", params=params)

    async def get_library_template(self, template_id: str) -> dict[str, Any]:
        """Get a workflow library template."""
        return await self._client.request("GET", f"/api/v1/workflow/templates/{template_id}")

    async def get_library_package(self, template_id: str) -> dict[str, Any]:
        """Get a workflow library template package."""
        return await self._client.request(
            "GET", f"/api/v1/workflow/templates/{template_id}/package"
        )

    async def list_template_categories(self) -> dict[str, Any]:
        """List workflow template categories."""
        return await self._client.request("GET", "/api/v1/workflow/categories")

    async def list_template_patterns(self) -> dict[str, Any]:
        """List workflow template patterns."""
        return await self._client.request("GET", "/api/v1/workflow/patterns")

    async def list_pattern_templates(self) -> dict[str, Any]:
        """List workflow pattern templates."""
        return await self._client.request("GET", "/api/v1/workflow/pattern-templates")

    async def get_pattern_template(self, template_id: str) -> dict[str, Any]:
        """Get a workflow pattern template by ID."""
        return await self._client.request(
            "GET", f"/api/v1/workflow/pattern-templates/{template_id}"
        )

    # =========================================================================
    # Workflows/Templates & Workflows/Executions (alternate paths)
    # =========================================================================

    async def list_workflow_templates(self, **params: Any) -> dict[str, Any]:
        """List workflow templates. GET /api/v1/workflows/templates"""
        return await self._client.request(
            "GET", "/api/v1/workflows/templates", params=params or None
        )

    async def get_workflow_template(self, template_id: str) -> dict[str, Any]:
        """Get a workflow template by ID. GET /api/v1/workflows/templates/:template_id"""
        return await self._client.request(
            "GET", f"/api/v1/workflows/templates/{template_id}"
        )

    async def list_workflow_executions(
        self,
        *,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List workflow executions. GET /api/v1/workflows/executions"""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workflow_id:
            params["workflow_id"] = workflow_id
        return await self._client.request(
            "GET", "/api/v1/workflows/executions", params=params
        )

    async def get_workflow_execution(self, execution_id: str) -> dict[str, Any]:
        """Get a workflow execution. GET /api/v1/workflows/executions/:execution_id"""
        return await self._client.request(
            "GET", f"/api/v1/workflows/executions/{execution_id}"
        )
