"""
Readiness Namespace API

Provides methods for system readiness checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ReadinessAPI:
    """Synchronous Readiness API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def check(self) -> dict[str, Any]:
        """Check system readiness status.

        Returns:
            Dict with readiness status for all subsystems.
        """
        return self._client.request("GET", "/api/v1/readiness")

    def health(self) -> dict[str, Any]:
        """Get health check status (lightweight liveness probe)."""
        return self._client.request("GET", "/api/v1/health")

    def detailed(self) -> dict[str, Any]:
        """Get detailed health with subsystem diagnostics."""
        return self._client.request("GET", "/api/v1/health/detailed")


class AsyncReadinessAPI:
    """Asynchronous Readiness API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def check(self) -> dict[str, Any]:
        """Check system readiness status."""
        return await self._client.request("GET", "/api/v1/readiness")

    async def health(self) -> dict[str, Any]:
        """Get health check status (lightweight liveness probe)."""
        return await self._client.request("GET", "/api/v1/health")

    async def detailed(self) -> dict[str, Any]:
        """Get detailed health with subsystem diagnostics."""
        return await self._client.request("GET", "/api/v1/health/detailed")
