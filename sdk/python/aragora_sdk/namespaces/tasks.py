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
