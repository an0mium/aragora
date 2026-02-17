"""
Computer Use Namespace API.

Provides methods for computer use orchestration:
- Task creation and execution
- Task status monitoring
- Action statistics
- Policy management

Endpoints:
    POST   /api/v1/computer-use/tasks            - Create and run task
    GET    /api/v1/computer-use/tasks            - List tasks
    GET    /api/v1/computer-use/tasks/{id}       - Get task status
    POST   /api/v1/computer-use/tasks/{id}/cancel - Cancel task
    GET    /api/v1/computer-use/policies         - List policies
    POST   /api/v1/computer-use/policies         - Create policy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

class ComputerUseAPI:
    """
    Synchronous Computer Use API.

    Provides methods for orchestrating computer use tasks with
    screenshot analysis, clicking, typing, and scrolling.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Create a task
        >>> task = client.computer_use.create_task(
        ...     goal="Open the settings page and enable dark mode",
        ...     max_steps=10,
        ... )
        >>> # Check task status
        >>> status = client.computer_use.get_task(task["task_id"])
        >>> print(f"Status: {status['task']['status']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Tasks
    # =========================================================================

    def create_task(
        self,
        goal: str,
        max_steps: int = 10,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Create and run a computer use task.

        Args:
            goal: The goal to accomplish.
            max_steps: Maximum steps to take (default 10).
            dry_run: If True, simulate without actually performing actions.

        Returns:
            Dict with task_id, status, message.
        """
        return self._client.request(
            "POST",
            "/api/v1/computer-use/tasks",
            json={"goal": goal, "max_steps": max_steps, "dry_run": dry_run},
        )

    def list_tasks(
        self,
        limit: int = 20,
        status: TaskStatus | None = None,
    ) -> dict[str, Any]:
        """
        List recent tasks.

        Args:
            limit: Maximum tasks to return (1-100).
            status: Filter by task status.

        Returns:
            Dict with tasks array and total count.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/computer-use/tasks", params=params)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """
        Get task status and details.

        Args:
            task_id: Task ID.

        Returns:
            Dict with task info including status, steps, result.
        """
        return self._client.request("GET", f"/api/v1/computer-use/tasks/{task_id}")

    def list_policies(self) -> dict[str, Any]:
        """
        List active policies.

        GET /api/v1/computer-use/policies

        Returns:
            Dict with policies array and total count.
        """
        return self._client.request("GET", "/api/v1/computer-use/policies")

    def get_policy(self, policy_id: str) -> dict[str, Any]:
        """
        Get a specific computer use policy.

        GET /api/v1/computer-use/policies/:policy_id

        Args:
            policy_id: Policy identifier.

        Returns:
            Dict with policy details.
        """
        return self._client.request("GET", f"/api/v1/computer-use/policies/{policy_id}")

    def create_policy(
        self,
        name: str,
        description: str | None = None,
        allowed_actions: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a computer use policy.

        POST /api/v1/computer-use/policies

        Args:
            name: Policy name.
            description: Optional description.
            allowed_actions: List of allowed action types.
            blocked_domains: List of blocked domains.

        Returns:
            Dict with policy_id and success message.
        """
        data: dict[str, Any] = {"name": name}
        if description:
            data["description"] = description
        if allowed_actions:
            data["allowed_actions"] = allowed_actions
        if blocked_domains:
            data["blocked_domains"] = blocked_domains

        return self._client.request("POST", "/api/v1/computer-use/policies", json=data)

    # =========================================================================
    # Actions
    # =========================================================================

    def get_action_stats(self) -> dict[str, Any]:
        """
        Get action statistics.

        GET /api/v1/computer-use/actions/stats

        Returns:
            Dict with action statistics.
        """
        return self._client.request("GET", "/api/v1/computer-use/actions/stats")

    def list_actions(self) -> dict[str, Any]:
        """
        List available computer use actions.

        GET /api/v1/computer-use/actions

        Returns:
            Dict with available action types.
        """
        return self._client.request("GET", "/api/v1/computer-use/actions")

    def get_action(self, action_id: str) -> dict[str, Any]:
        """
        Get a specific action's details.

        GET /api/v1/computer-use/actions/:action_id

        Args:
            action_id: Action identifier.

        Returns:
            Dict with action details.
        """
        return self._client.request("GET", f"/api/v1/computer-use/actions/{action_id}")

    # =========================================================================
    # Approvals
    # =========================================================================

    def list_approvals(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        List approval requests.

        GET /api/v1/computer-use/approvals

        Args:
            status: Filter by approval status (pending, approved, denied).
            limit: Maximum approvals to return.

        Returns:
            Dict with approvals array.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/computer-use/approvals", params=params)

    def get_approval(self, request_id: str) -> dict[str, Any]:
        """
        Get a specific approval request.

        GET /api/v1/computer-use/approvals/:request_id

        Args:
            request_id: Approval request identifier.

        Returns:
            Dict with approval request details.
        """
        return self._client.request("GET", f"/api/v1/computer-use/approvals/{request_id}")


class AsyncComputerUseAPI:
    """Asynchronous Computer Use API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Tasks
    # =========================================================================

    async def create_task(
        self,
        goal: str,
        max_steps: int = 10,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Create and run a computer use task."""
        return await self._client.request(
            "POST",
            "/api/v1/computer-use/tasks",
            json={"goal": goal, "max_steps": max_steps, "dry_run": dry_run},
        )

    async def list_tasks(
        self,
        limit: int = 20,
        status: TaskStatus | None = None,
    ) -> dict[str, Any]:
        """List recent tasks."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/computer-use/tasks", params=params)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get task status and details."""
        return await self._client.request("GET", f"/api/v1/computer-use/tasks/{task_id}")

    async def list_policies(self) -> dict[str, Any]:
        """List active policies. GET /api/v1/computer-use/policies"""
        return await self._client.request("GET", "/api/v1/computer-use/policies")

    async def get_policy(self, policy_id: str) -> dict[str, Any]:
        """Get a specific policy. GET /api/v1/computer-use/policies/:policy_id"""
        return await self._client.request("GET", f"/api/v1/computer-use/policies/{policy_id}")

    async def create_policy(
        self,
        name: str,
        description: str | None = None,
        allowed_actions: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a computer use policy. POST /api/v1/computer-use/policies"""
        data: dict[str, Any] = {"name": name}
        if description:
            data["description"] = description
        if allowed_actions:
            data["allowed_actions"] = allowed_actions
        if blocked_domains:
            data["blocked_domains"] = blocked_domains

        return await self._client.request("POST", "/api/v1/computer-use/policies", json=data)

    # =========================================================================
    # Actions
    # =========================================================================

    async def get_action_stats(self) -> dict[str, Any]:
        """Get action statistics. GET /api/v1/computer-use/actions/stats"""
        return await self._client.request("GET", "/api/v1/computer-use/actions/stats")

    async def list_actions(self) -> dict[str, Any]:
        """List available computer use actions. GET /api/v1/computer-use/actions"""
        return await self._client.request("GET", "/api/v1/computer-use/actions")

    async def get_action(self, action_id: str) -> dict[str, Any]:
        """Get a specific action's details. GET /api/v1/computer-use/actions/:action_id"""
        return await self._client.request("GET", f"/api/v1/computer-use/actions/{action_id}")

    # =========================================================================
    # Approvals
    # =========================================================================

    async def list_approvals(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List approval requests. GET /api/v1/computer-use/approvals"""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request(
            "GET", "/api/v1/computer-use/approvals", params=params
        )

    async def get_approval(self, request_id: str) -> dict[str, Any]:
        """Get an approval request. GET /api/v1/computer-use/approvals/:request_id"""
        return await self._client.request(
            "GET", f"/api/v1/computer-use/approvals/{request_id}"
        )

