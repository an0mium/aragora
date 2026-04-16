"""
Tasks Namespace API

Provides methods for task management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class TasksAPI:
    """Synchronous Tasks API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new task.

        Args:
            **kwargs: Task configuration (title, description, assignee, etc.).

        Returns:
            Dict with created task details.
        """
        return self._client.request("POST", "/api/v1/tasks", json=kwargs)

    def get(self, task_id: str) -> dict[str, Any]:
        """Get a task by ID."""
        return self._client.request("GET", f"/api/v2/tasks/{task_id}")

    def list(self, **params: Any) -> dict[str, Any]:
        """List tasks with optional filters."""
        return self._client.request("GET", "/api/v2/tasks", params=params or None)

    def update(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a task."""
        return self._client.request("PUT", f"/api/v2/tasks/{task_id}", json=kwargs)

    def delete(self, task_id: str) -> dict[str, Any]:
        """Delete a task."""
        return self._client.request("DELETE", f"/api/v2/tasks/{task_id}")

    def list_queue(
        self,
        *,
        status: str | None = None,
        work_type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """List developer task queue items."""
        params = {
            key: value
            for key, value in {
                "status": status,
                "work_type": work_type,
                "limit": limit,
            }.items()
            if value is not None
        }
        return self._client.request("GET", "/api/v1/tasks/queue", params=params or None)

    def get_queue_task(self, task_id: str) -> dict[str, Any]:
        """Get a developer task queue item by ID."""
        return self._client.request("GET", f"/api/v1/tasks/queue/{task_id}")

    def get_queue_stats(self) -> dict[str, Any]:
        """Get developer task queue statistics."""
        return self._client.request("GET", "/api/v1/tasks/queue/stats")

    def sync_queue(
        self,
        *,
        include_pending: bool = True,
        include_developer_tasks: bool = True,
        complete_missing: bool = True,
    ) -> dict[str, Any]:
        """Synchronize developer coordination work into the global task queue."""
        return self._client.request(
            "POST",
            "/api/v1/tasks/queue/sync",
            json={
                "include_pending": include_pending,
                "include_developer_tasks": include_developer_tasks,
                "complete_missing": complete_missing,
            },
        )

    def claim_queue_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Claim a task queue item and create a lease."""
        return self._client.request(
            "POST",
            f"/api/v1/tasks/queue/{task_id}/claim",
            json=kwargs,
        )

    def list_leases(self) -> dict[str, Any]:
        """List active task leases."""
        return self._client.request("GET", "/api/v1/tasks/leases")

    def heartbeat_lease(self, lease_id: str, *, ttl_hours: float | None = None) -> dict[str, Any]:
        """Heartbeat a task lease."""
        body: dict[str, Any] = {}
        if ttl_hours is not None:
            body["ttl_hours"] = ttl_hours
        return self._client.request(
            "POST",
            f"/api/v1/tasks/leases/{lease_id}/heartbeat",
            json=body,
        )

    def release_lease(self, lease_id: str) -> dict[str, Any]:
        """Release a task lease."""
        return self._client.request("POST", f"/api/v1/tasks/leases/{lease_id}/release")

    def complete_lease(self, lease_id: str, **kwargs: Any) -> dict[str, Any]:
        """Record completion for a task lease."""
        return self._client.request(
            "POST",
            f"/api/v1/tasks/leases/{lease_id}/complete",
            json=kwargs,
        )

    def list_salvage(self) -> dict[str, Any]:
        """List task salvage candidates."""
        return self._client.request("GET", "/api/v1/tasks/salvage")


class AsyncTasksAPI:
    """Asynchronous Tasks API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new task."""
        return await self._client.request("POST", "/api/v1/tasks", json=kwargs)

    async def get(self, task_id: str) -> dict[str, Any]:
        """Get a task by ID."""
        return await self._client.request("GET", f"/api/v2/tasks/{task_id}")

    async def list(self, **params: Any) -> dict[str, Any]:
        """List tasks with optional filters."""
        return await self._client.request("GET", "/api/v2/tasks", params=params or None)

    async def update(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a task."""
        return await self._client.request("PUT", f"/api/v2/tasks/{task_id}", json=kwargs)

    async def delete(self, task_id: str) -> dict[str, Any]:
        """Delete a task."""
        return await self._client.request("DELETE", f"/api/v2/tasks/{task_id}")

    async def list_queue(
        self,
        *,
        status: str | None = None,
        work_type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """List developer task queue items."""
        params = {
            key: value
            for key, value in {
                "status": status,
                "work_type": work_type,
                "limit": limit,
            }.items()
            if value is not None
        }
        return await self._client.request("GET", "/api/v1/tasks/queue", params=params or None)

    async def get_queue_task(self, task_id: str) -> dict[str, Any]:
        """Get a developer task queue item by ID."""
        return await self._client.request("GET", f"/api/v1/tasks/queue/{task_id}")

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get developer task queue statistics."""
        return await self._client.request("GET", "/api/v1/tasks/queue/stats")

    async def sync_queue(
        self,
        *,
        include_pending: bool = True,
        include_developer_tasks: bool = True,
        complete_missing: bool = True,
    ) -> dict[str, Any]:
        """Synchronize developer coordination work into the global task queue."""
        return await self._client.request(
            "POST",
            "/api/v1/tasks/queue/sync",
            json={
                "include_pending": include_pending,
                "include_developer_tasks": include_developer_tasks,
                "complete_missing": complete_missing,
            },
        )

    async def claim_queue_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        """Claim a task queue item and create a lease."""
        return await self._client.request(
            "POST",
            f"/api/v1/tasks/queue/{task_id}/claim",
            json=kwargs,
        )

    async def list_leases(self) -> dict[str, Any]:
        """List active task leases."""
        return await self._client.request("GET", "/api/v1/tasks/leases")

    async def heartbeat_lease(
        self, lease_id: str, *, ttl_hours: float | None = None
    ) -> dict[str, Any]:
        """Heartbeat a task lease."""
        body: dict[str, Any] = {}
        if ttl_hours is not None:
            body["ttl_hours"] = ttl_hours
        return await self._client.request(
            "POST",
            f"/api/v1/tasks/leases/{lease_id}/heartbeat",
            json=body,
        )

    async def release_lease(self, lease_id: str) -> dict[str, Any]:
        """Release a task lease."""
        return await self._client.request("POST", f"/api/v1/tasks/leases/{lease_id}/release")

    async def complete_lease(self, lease_id: str, **kwargs: Any) -> dict[str, Any]:
        """Record completion for a task lease."""
        return await self._client.request(
            "POST",
            f"/api/v1/tasks/leases/{lease_id}/complete",
            json=kwargs,
        )

    async def list_salvage(self) -> dict[str, Any]:
        """List task salvage candidates."""
        return await self._client.request("GET", "/api/v1/tasks/salvage")
