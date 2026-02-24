"""
Users Namespace API

Provides methods for user self-service operations:
- Account deletion requests
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class UsersAPI:
    """Synchronous Users API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def request_deletion(self, **kwargs: Any) -> dict[str, Any]:
        """Request account deletion.

        Args:
            **kwargs: Deletion request parameters (reason, confirm, etc.).

        Returns:
            Dict with deletion request confirmation and timeline.
        """
        return self._client.request("POST", "/api/v1/users/self/deletion-request", json=kwargs)

    def cancel_deletion(self) -> dict[str, Any]:
        """Cancel a pending account deletion request.

        Returns:
            Dict with cancellation confirmation.
        """
        return self._client.request("DELETE", "/api/v1/users/self/deletion-request")


class AsyncUsersAPI:
    """Asynchronous Users API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def request_deletion(self, **kwargs: Any) -> dict[str, Any]:
        """Request account deletion."""
        return await self._client.request(
            "POST", "/api/v1/users/self/deletion-request", json=kwargs
        )

    async def cancel_deletion(self) -> dict[str, Any]:
        """Cancel a pending account deletion request."""
        return await self._client.request("DELETE", "/api/v1/users/self/deletion-request")
