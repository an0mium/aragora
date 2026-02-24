"""
Playbooks Namespace API

Provides methods for playbook management and execution:
- List available playbooks
- Get playbook details
- Run playbooks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PlaybooksAPI:
    """Synchronous Playbooks API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, **params: Any) -> dict[str, Any]:
        """List available playbooks.

        Args:
            **params: Filter parameters (category, limit, offset, etc.).

        Returns:
            Dict with playbooks array and pagination.
        """
        return self._client.request("GET", "/api/v1/playbooks", params=params or None)

    def get(self, playbook_id: str) -> dict[str, Any]:
        """Get a playbook by ID.

        Args:
            playbook_id: Playbook identifier.

        Returns:
            Dict with playbook details.
        """
        return self._client.request("GET", f"/api/v1/playbooks/{playbook_id}")

    def get_run_status(self, playbook_id: str) -> dict[str, Any]:
        """Get the run status for a playbook.

        Args:
            playbook_id: Playbook identifier.

        Returns:
            Dict with run status and results.
        """
        return self._client.request("GET", f"/api/v1/playbooks/{playbook_id}/run")

    def run(self, playbook_id: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a playbook.

        Args:
            playbook_id: Playbook identifier.
            **kwargs: Execution parameters.

        Returns:
            Dict with execution result and run_id.
        """
        return self._client.request("POST", f"/api/v1/playbooks/{playbook_id}/run", json=kwargs)


class AsyncPlaybooksAPI:
    """Asynchronous Playbooks API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, **params: Any) -> dict[str, Any]:
        """List available playbooks."""
        return await self._client.request("GET", "/api/v1/playbooks", params=params or None)

    async def get(self, playbook_id: str) -> dict[str, Any]:
        """Get a playbook by ID."""
        return await self._client.request("GET", f"/api/v1/playbooks/{playbook_id}")

    async def get_run_status(self, playbook_id: str) -> dict[str, Any]:
        """Get the run status for a playbook."""
        return await self._client.request("GET", f"/api/v1/playbooks/{playbook_id}/run")

    async def run(self, playbook_id: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a playbook."""
        return await self._client.request(
            "POST", f"/api/v1/playbooks/{playbook_id}/run", json=kwargs
        )
