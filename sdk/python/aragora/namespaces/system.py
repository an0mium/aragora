"""
System Namespace API.

Provides system administration and monitoring:
- Database maintenance tasks
- History tracking (cycles, events, debates)
- Circuit breaker monitoring
- Authentication statistics
- Debug endpoints
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

MaintenanceTask = Literal["status", "vacuum", "analyze", "checkpoint", "full"]
CircuitBreakerStatus = Literal["closed", "open", "half_open"]


class SystemAPI:
    """
    Synchronous System API.

    Provides methods for system administration:
    - Database maintenance tasks
    - History tracking (cycles, events, debates)
    - Circuit breaker monitoring
    - Authentication statistics
    - Debug endpoints

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Run database maintenance
        >>> result = client.system.run_maintenance("vacuum")
        >>> # Get circuit breaker status
        >>> breakers = client.system.get_circuit_breakers()
        >>> # Get debate history
        >>> history = client.system.get_debate_history(limit=50)
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def debug_test(self) -> dict[str, Any]:
        """
        Run a debug test endpoint.

        Useful for verifying API connectivity.

        Returns:
            Debug test response with status and method.
        """
        return self._client.request("GET", "/api/debug/test")

    def get_cycles(
        self,
        loop_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get cycle history.

        Args:
            loop_id: Optional filter by loop ID.
            limit: Maximum number of results.

        Returns:
            List of cycle entries.
        """
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return self._client.request("GET", "/api/history/cycles", params=params)

    def get_events(
        self,
        loop_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get event history.

        Args:
            loop_id: Optional filter by loop ID.
            limit: Maximum number of results.

        Returns:
            List of event entries.
        """
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return self._client.request("GET", "/api/history/events", params=params)

    def get_debate_history(
        self,
        loop_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get debate history.

        Args:
            loop_id: Optional filter by loop ID.
            limit: Maximum number of results.

        Returns:
            List of debate history entries.
        """
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return self._client.request("GET", "/api/history/debates", params=params)

    def get_history_summary(self, loop_id: str | None = None) -> dict[str, Any]:
        """
        Get history summary statistics.

        Args:
            loop_id: Optional filter by loop ID.

        Returns:
            Summary with total debates, agents, matches.
        """
        params: dict[str, Any] = {}
        if loop_id:
            params["loop_id"] = loop_id
        return self._client.request("GET", "/api/history/summary", params=params)

    def run_maintenance(
        self,
        task: MaintenanceTask = "status",
    ) -> dict[str, Any]:
        """
        Run database maintenance task.

        Args:
            task: Type of maintenance to run.

        Returns:
            Maintenance result with success status.
        """
        return self._client.request("GET", "/api/system/maintenance", params={"task": task})

    def get_auth_stats(self) -> dict[str, Any]:
        """
        Get authentication statistics.

        Returns:
            Auth stats with user counts, sessions, tokens.
        """
        return self._client.request("GET", "/api/auth/stats")

    def revoke_token(
        self,
        token_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Revoke a token.

        Args:
            token_id: Token identifier to revoke.
            user_id: User ID to revoke tokens for.

        Returns:
            Revocation confirmation.
        """
        data: dict[str, Any] = {}
        if token_id:
            data["token_id"] = token_id
        if user_id:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/auth/revoke", json=data)

    def get_circuit_breakers(self) -> dict[str, Any]:
        """
        Get circuit breaker metrics.

        Returns:
            Dictionary of circuit breaker names to metrics.
        """
        return self._client.request("GET", "/api/circuit-breakers")

    def get_prometheus_metrics(self) -> str:
        """
        Get Prometheus metrics.

        Returns:
            Raw Prometheus format metrics string.
        """
        return self._client.request("GET", "/metrics")


class AsyncSystemAPI:
    """Asynchronous System API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def debug_test(self) -> dict[str, Any]:
        """Run a debug test endpoint."""
        return await self._client.request("GET", "/api/debug/test")

    async def get_cycles(
        self,
        loop_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get cycle history."""
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return await self._client.request("GET", "/api/history/cycles", params=params)

    async def get_events(
        self,
        loop_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get event history."""
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return await self._client.request("GET", "/api/history/events", params=params)

    async def get_debate_history(
        self,
        loop_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get debate history."""
        params: dict[str, Any] = {"limit": limit}
        if loop_id:
            params["loop_id"] = loop_id
        return await self._client.request("GET", "/api/history/debates", params=params)

    async def get_history_summary(self, loop_id: str | None = None) -> dict[str, Any]:
        """Get history summary statistics."""
        params: dict[str, Any] = {}
        if loop_id:
            params["loop_id"] = loop_id
        return await self._client.request("GET", "/api/history/summary", params=params)

    async def run_maintenance(
        self,
        task: MaintenanceTask = "status",
    ) -> dict[str, Any]:
        """Run database maintenance task."""
        return await self._client.request("GET", "/api/system/maintenance", params={"task": task})

    async def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return await self._client.request("GET", "/api/auth/stats")

    async def revoke_token(
        self,
        token_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Revoke a token."""
        data: dict[str, Any] = {}
        if token_id:
            data["token_id"] = token_id
        if user_id:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/auth/revoke", json=data)

    async def get_circuit_breakers(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return await self._client.request("GET", "/api/circuit-breakers")

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics."""
        return await self._client.request("GET", "/metrics")
