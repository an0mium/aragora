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


class AsyncTasksAPI:
    """Asynchronous Tasks API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new task."""
        return await self._client.request("POST", "/api/v1/tasks", json=kwargs)
