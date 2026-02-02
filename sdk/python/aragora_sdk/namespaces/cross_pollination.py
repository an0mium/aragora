"""
Cross-Pollination Namespace API

Provides methods for cross-pollination between debates:
- Statistics and metrics
- Subscriber management
- Bridge configuration
- Knowledge mound integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CrossPollinationAPI:
    """
    Synchronous Cross-Pollination API.

    Cross-pollination enables knowledge sharing between debates,
    allowing insights from one debate to inform others.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> stats = client.cross_pollination.get_stats()
        >>> suggestions = client.cross_pollination.suggest(debate_id="...")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Statistics & Metrics
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get cross-pollination statistics.

        Returns:
            Dict with total_pollinations, active_bridges, success_rate, etc.
        """
        return self._client.request("GET", "/api/v1/cross-pollination/stats")

    def get_metrics(self) -> dict[str, Any]:
        """
        Get cross-pollination metrics.

        Returns:
            Dict with detailed metrics
        """
        return self._client.request("GET", "/api/v1/cross-pollination/metrics")

    # ===========================================================================
    # Subscriber Management
    # ===========================================================================

    def list_subscribers(self) -> dict[str, Any]:
        """
        List cross-pollination subscribers.

        Returns:
            Dict with subscribers array
        """
        return self._client.request("GET", "/api/v1/cross-pollination/subscribers")

    def subscribe(
        self,
        debate_id: str,
        topics: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """
        Subscribe a debate to cross-pollination.

        Args:
            debate_id: Debate ID to subscribe
            topics: Topics to subscribe to (all if not specified)
            min_confidence: Minimum confidence threshold

        Returns:
            Subscription result
        """
        data: dict[str, Any] = {"debate_id": debate_id}
        if topics:
            data["topics"] = topics
        if min_confidence is not None:
            data["min_confidence"] = min_confidence

        return self._client.request("POST", "/api/v1/cross-pollination/subscribe", json=data)

    def unsubscribe(self, debate_id: str) -> dict[str, Any]:
        """
        Unsubscribe a debate from cross-pollination.

        Args:
            debate_id: Debate ID to unsubscribe

        Returns:
            Unsubscription result
        """
        return self._client.request("DELETE", f"/api/v1/cross-pollination/subscribers/{debate_id}")

    # ===========================================================================
    # Bridge Configuration
    # ===========================================================================

    def get_bridge(self) -> dict[str, Any]:
        """
        Get bridge configuration.

        Returns:
            Bridge configuration and status
        """
        return self._client.request("GET", "/api/v1/cross-pollination/bridge")

    def configure_bridge(
        self,
        enabled: bool | None = None,
        max_pollinations_per_debate: int | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Configure the cross-pollination bridge.

        Args:
            enabled: Enable/disable bridge
            max_pollinations_per_debate: Max pollinations per debate
            confidence_threshold: Confidence threshold

        Returns:
            Updated configuration
        """
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if max_pollinations_per_debate is not None:
            data["max_pollinations_per_debate"] = max_pollinations_per_debate
        if confidence_threshold is not None:
            data["confidence_threshold"] = confidence_threshold

        return self._client.request("PUT", "/api/v1/cross-pollination/bridge", json=data)

    # ===========================================================================
    # Knowledge Mound Integration
    # ===========================================================================

    def get_km_status(self) -> dict[str, Any]:
        """
        Get Knowledge Mound integration status.

        Returns:
            KM integration status and configuration
        """
        return self._client.request("GET", "/api/v1/cross-pollination/km")

    def check_staleness(self) -> dict[str, Any]:
        """
        Check for stale cross-pollinated knowledge.

        Returns:
            Staleness report with stale items
        """
        return self._client.request("GET", "/api/v1/cross-pollination/km/staleness-check")

    # ===========================================================================
    # Laboratory (Suggestions)
    # ===========================================================================

    def suggest(
        self,
        debate_id: str | None = None,
        topic: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Get cross-pollination suggestions.

        Args:
            debate_id: Debate ID to get suggestions for
            topic: Topic to get suggestions for
            limit: Maximum suggestions to return

        Returns:
            Dict with suggestions array
        """
        params: dict[str, Any] = {"limit": limit}
        if debate_id:
            params["debate_id"] = debate_id
        if topic:
            params["topic"] = topic

        return self._client.request(
            "GET", "/api/v1/laboratory/cross-pollinations/suggest", params=params
        )


class AsyncCrossPollinationAPI:
    """
    Asynchronous Cross-Pollination API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     stats = await client.cross_pollination.get_stats()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Statistics & Metrics
    async def get_stats(self) -> dict[str, Any]:
        """Get cross-pollination statistics."""
        return await self._client.request("GET", "/api/v1/cross-pollination/stats")

    async def get_metrics(self) -> dict[str, Any]:
        """Get cross-pollination metrics."""
        return await self._client.request("GET", "/api/v1/cross-pollination/metrics")

    # Subscriber Management
    async def list_subscribers(self) -> dict[str, Any]:
        """List cross-pollination subscribers."""
        return await self._client.request("GET", "/api/v1/cross-pollination/subscribers")

    async def subscribe(
        self,
        debate_id: str,
        topics: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Subscribe a debate to cross-pollination."""
        data: dict[str, Any] = {"debate_id": debate_id}
        if topics:
            data["topics"] = topics
        if min_confidence is not None:
            data["min_confidence"] = min_confidence

        return await self._client.request("POST", "/api/v1/cross-pollination/subscribe", json=data)

    async def unsubscribe(self, debate_id: str) -> dict[str, Any]:
        """Unsubscribe a debate from cross-pollination."""
        return await self._client.request(
            "DELETE", f"/api/v1/cross-pollination/subscribers/{debate_id}"
        )

    # Bridge Configuration
    async def get_bridge(self) -> dict[str, Any]:
        """Get bridge configuration."""
        return await self._client.request("GET", "/api/v1/cross-pollination/bridge")

    async def configure_bridge(
        self,
        enabled: bool | None = None,
        max_pollinations_per_debate: int | None = None,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Configure the cross-pollination bridge."""
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if max_pollinations_per_debate is not None:
            data["max_pollinations_per_debate"] = max_pollinations_per_debate
        if confidence_threshold is not None:
            data["confidence_threshold"] = confidence_threshold

        return await self._client.request("PUT", "/api/v1/cross-pollination/bridge", json=data)

    # Knowledge Mound Integration
    async def get_km_status(self) -> dict[str, Any]:
        """Get Knowledge Mound integration status."""
        return await self._client.request("GET", "/api/v1/cross-pollination/km")

    async def check_staleness(self) -> dict[str, Any]:
        """Check for stale cross-pollinated knowledge."""
        return await self._client.request("GET", "/api/v1/cross-pollination/km/staleness-check")

    # Laboratory (Suggestions)
    async def suggest(
        self,
        debate_id: str | None = None,
        topic: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get cross-pollination suggestions."""
        params: dict[str, Any] = {"limit": limit}
        if debate_id:
            params["debate_id"] = debate_id
        if topic:
            params["topic"] = topic

        return await self._client.request(
            "GET", "/api/v1/laboratory/cross-pollinations/suggest", params=params
        )
