"""
Computer Use Namespace API.

Provides methods for computer use orchestration:
- Task creation, execution, cancellation, and deletion
- Action execution, listing, and deletion
- Action statistics
- Policy CRUD (create, read, update, delete)
- Approval workflow (list, get, approve, deny)

Endpoints:
    POST   /api/v1/computer-use/tasks                       - Create and run task
    GET    /api/v1/computer-use/tasks                       - List tasks
    GET    /api/v1/computer-use/tasks/{id}                  - Get task status
    POST   /api/v1/computer-use/tasks/{id}/cancel           - Cancel task
    DELETE /api/v1/computer-use/tasks/{id}                  - Delete task record
    GET    /api/v1/computer-use/actions                     - List actions
    POST   /api/v1/computer-use/actions                     - Execute an action
    GET    /api/v1/computer-use/actions/{id}                - Get action details
    DELETE /api/v1/computer-use/actions/{id}                - Delete action record
    GET    /api/v1/computer-use/actions/stats               - Get action stats
    GET    /api/v1/computer-use/policies                    - List policies
    POST   /api/v1/computer-use/policies                    - Create policy
    GET    /api/v1/computer-use/policies/{id}               - Get policy details
    PUT    /api/v1/computer-use/policies/{id}               - Update policy
    DELETE /api/v1/computer-use/policies/{id}               - Delete policy
    GET    /api/v1/computer-use/approvals                   - List approvals
    GET    /api/v1/computer-use/approvals/{id}              - Get approval details
    POST   /api/v1/computer-use/approvals/{id}/approve      - Approve request
    POST   /api/v1/computer-use/approvals/{id}/deny         - Deny request
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
ActionType = Literal["click", "type", "screenshot", "scroll", "key"]

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

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """
        Cancel a running task.

        POST /api/v1/computer-use/tasks/:task_id/cancel

        Args:
            task_id: Task ID to cancel.

        Returns:
            Dict with success confirmation.
        """
        return self._client.request("POST", f"/api/v1/computer-use/tasks/{task_id}/cancel")

    def delete_task(self, task_id: str) -> dict[str, Any]:
        """
        Delete a task record.

        DELETE /api/v1/computer-use/tasks/:task_id

        Only completed, failed, or cancelled tasks can be deleted.
        Running tasks must be cancelled first.

        Args:
            task_id: Task ID to delete.

        Returns:
            Dict with deleted confirmation and task_id.
        """
        return self._client.request("DELETE", f"/api/v1/computer-use/tasks/{task_id}")

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

    def update_policy(
        self,
        policy_id: str,
        name: str | None = None,
        description: str | None = None,
        allowed_actions: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_steps: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Update a computer use policy.

        PUT /api/v1/computer-use/policies/:policy_id

        Args:
            policy_id: Policy identifier.
            name: Updated policy name.
            description: Updated description.
            allowed_actions: Updated list of allowed action types.
            blocked_domains: Updated list of blocked domains.
            max_steps: Maximum number of steps per task.
            timeout_seconds: Timeout in seconds for task execution.

        Returns:
            Dict with policy_id and success message.
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if allowed_actions is not None:
            data["allowed_actions"] = allowed_actions
        if blocked_domains is not None:
            data["blocked_domains"] = blocked_domains
        if max_steps is not None:
            data["max_steps"] = max_steps
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds

        return self._client.request("PUT", f"/api/v1/computer-use/policies/{policy_id}", json=data)

    def delete_policy(self, policy_id: str) -> dict[str, Any]:
        """
        Delete a computer use policy.

        DELETE /api/v1/computer-use/policies/:policy_id

        The default policy cannot be deleted.

        Args:
            policy_id: Policy identifier.

        Returns:
            Dict with deleted confirmation and policy_id.
        """
        return self._client.request("DELETE", f"/api/v1/computer-use/policies/{policy_id}")

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

    def execute_action(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a single computer use action.

        POST /api/v1/computer-use/actions

        The action is validated against the active policy before execution.

        Args:
            action_type: Type of action (click, type, screenshot, scroll, key).
            parameters: Action-specific parameters (e.g., coordinates for click).
            task_id: Optional associated task ID.

        Returns:
            Dict with action_id, action_type, success, message.
        """
        data: dict[str, Any] = {"action_type": action_type}
        if parameters is not None:
            data["parameters"] = parameters
        if task_id is not None:
            data["task_id"] = task_id

        return self._client.request("POST", "/api/v1/computer-use/actions", json=data)

    def delete_action(self, action_id: str) -> dict[str, Any]:
        """
        Delete a computer use action record.

        DELETE /api/v1/computer-use/actions/:action_id

        Only completed or failed actions can be deleted.

        Args:
            action_id: Action identifier.

        Returns:
            Dict with deleted confirmation and action_id.
        """
        return self._client.request("DELETE", f"/api/v1/computer-use/actions/{action_id}")

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

    def approve_approval(self, request_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Approve a pending approval request.

        POST /api/v1/computer-use/approvals/:request_id/approve

        Args:
            request_id: Approval request identifier.
            reason: Optional reason for approval.

        Returns:
            Dict with approved confirmation and request_id.
        """
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return self._client.request(
            "POST",
            f"/api/v1/computer-use/approvals/{request_id}/approve",
            json=data if data else None,
        )

    def deny_approval(self, request_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Deny a pending approval request.

        POST /api/v1/computer-use/approvals/:request_id/deny

        Args:
            request_id: Approval request identifier.
            reason: Optional reason for denial.

        Returns:
            Dict with denied confirmation and request_id.
        """
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return self._client.request(
            "POST",
            f"/api/v1/computer-use/approvals/{request_id}/deny",
            json=data if data else None,
        )


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

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a running task. POST /api/v1/computer-use/tasks/:task_id/cancel"""
        return await self._client.request("POST", f"/api/v1/computer-use/tasks/{task_id}/cancel")

    async def delete_task(self, task_id: str) -> dict[str, Any]:
        """Delete a task record. DELETE /api/v1/computer-use/tasks/:task_id"""
        return await self._client.request("DELETE", f"/api/v1/computer-use/tasks/{task_id}")

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

    async def update_policy(
        self,
        policy_id: str,
        name: str | None = None,
        description: str | None = None,
        allowed_actions: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_steps: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Update a computer use policy. PUT /api/v1/computer-use/policies/:policy_id"""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if allowed_actions is not None:
            data["allowed_actions"] = allowed_actions
        if blocked_domains is not None:
            data["blocked_domains"] = blocked_domains
        if max_steps is not None:
            data["max_steps"] = max_steps
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds

        return await self._client.request(
            "PUT", f"/api/v1/computer-use/policies/{policy_id}", json=data
        )

    async def delete_policy(self, policy_id: str) -> dict[str, Any]:
        """Delete a computer use policy. DELETE /api/v1/computer-use/policies/:policy_id"""
        return await self._client.request("DELETE", f"/api/v1/computer-use/policies/{policy_id}")

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

    async def execute_action(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single computer use action. POST /api/v1/computer-use/actions"""
        data: dict[str, Any] = {"action_type": action_type}
        if parameters is not None:
            data["parameters"] = parameters
        if task_id is not None:
            data["task_id"] = task_id

        return await self._client.request("POST", "/api/v1/computer-use/actions", json=data)

    async def delete_action(self, action_id: str) -> dict[str, Any]:
        """Delete a computer use action record. DELETE /api/v1/computer-use/actions/:action_id"""
        return await self._client.request("DELETE", f"/api/v1/computer-use/actions/{action_id}")

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

    async def approve_approval(self, request_id: str, reason: str | None = None) -> dict[str, Any]:
        """Approve a pending approval request. POST /api/v1/computer-use/approvals/:request_id/approve"""
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return await self._client.request(
            "POST",
            f"/api/v1/computer-use/approvals/{request_id}/approve",
            json=data if data else None,
        )

    async def deny_approval(self, request_id: str, reason: str | None = None) -> dict[str, Any]:
        """Deny a pending approval request. POST /api/v1/computer-use/approvals/:request_id/deny"""
        data: dict[str, Any] = {}
        if reason is not None:
            data["reason"] = reason

        return await self._client.request(
            "POST",
            f"/api/v1/computer-use/approvals/{request_id}/deny",
            json=data if data else None,
        )

