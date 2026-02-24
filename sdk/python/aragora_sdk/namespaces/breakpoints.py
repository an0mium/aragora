"""
Breakpoints Namespace API

Provides methods for managing debug breakpoints in debate execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class BreakpointsAPI:
    """Synchronous Breakpoints API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new breakpoint.

        Args:
            **kwargs: Breakpoint configuration (debate_id, round, condition, etc.).

        Returns:
            Dict with created breakpoint details.
        """
        return self._client.request("POST", "/api/v1/breakpoints", json=kwargs)

    def list_pending(self) -> dict[str, Any]:
        """List all pending breakpoints."""
        return self._client.request("GET", "/api/v1/breakpoints/pending")

    def get_status(self, breakpoint_id: str) -> dict[str, Any]:
        """Get the status of a specific breakpoint."""
        return self._client.request("GET", f"/api/v1/breakpoints/{breakpoint_id}/status")

    def resolve(self, breakpoint_id: str, **kwargs: Any) -> dict[str, Any]:
        """Resolve a breakpoint (continue or abort).

        Args:
            breakpoint_id: ID of the breakpoint to resolve.
            **kwargs: Resolution details (action, reason, etc.).
        """
        return self._client.request(
            "POST", f"/api/v1/breakpoints/{breakpoint_id}/resolve", json=kwargs
        )


class AsyncBreakpointsAPI:
    """Asynchronous Breakpoints API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new breakpoint."""
        return await self._client.request("POST", "/api/v1/breakpoints", json=kwargs)

    async def list_pending(self) -> dict[str, Any]:
        """List all pending breakpoints."""
        return await self._client.request("GET", "/api/v1/breakpoints/pending")

    async def get_status(self, breakpoint_id: str) -> dict[str, Any]:
        """Get the status of a specific breakpoint."""
        return await self._client.request("GET", f"/api/v1/breakpoints/{breakpoint_id}/status")

    async def resolve(self, breakpoint_id: str, **kwargs: Any) -> dict[str, Any]:
        """Resolve a breakpoint (continue or abort)."""
        return await self._client.request(
            "POST", f"/api/v1/breakpoints/{breakpoint_id}/resolve", json=kwargs
        )
