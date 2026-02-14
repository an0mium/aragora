"""
Consensus Namespace API

Provides methods for consensus tracking and analysis:
- Similar debates
- Settled topics
- Dissents, contrarian views, risk warnings
- Domain history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ConsensusAPI:
    """Synchronous Consensus API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_similar_debates(self, topic: str, limit: int = 10) -> dict[str, Any]:
        """Find debates similar to a given topic."""
        params = {"topic": topic, "limit": limit}
        return self._client.request("GET", "/api/v1/consensus/similar", params=params)

    def get_settled_topics(self, min_confidence: float = 0.8, limit: int = 20) -> dict[str, Any]:
        """Get topics with established consensus."""
        params = {"min_confidence": min_confidence, "limit": limit}
        return self._client.request("GET", "/api/v1/consensus/settled", params=params)

    def get_stats(self, domain: str | None = None) -> dict[str, Any]:
        """Get consensus statistics."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/consensus/stats", params=params)

    def get_dissents(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get recent dissenting views."""
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/consensus/dissents", params=params)

    def get_contrarian_views(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get contrarian perspectives."""
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/consensus/contrarian-views", params=params)

    def get_risk_warnings(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get risk warnings and edge cases."""
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/consensus/risk-warnings", params=params)

    def get_domain_history(self, domain: str, limit: int = 50) -> dict[str, Any]:
        """Get consensus history for a domain."""
        params = {"limit": limit}
        return self._client.request("GET", f"/api/v1/consensus/domain/{domain}", params=params)

    def seed_demo(self) -> dict[str, Any]:
        """Seed demo consensus data (requires auth)."""
        return self._client.request("POST", "/api/v1/consensus/seed-demo")


class AsyncConsensusAPI:
    """Asynchronous Consensus API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_similar_debates(self, topic: str, limit: int = 10) -> dict[str, Any]:
        params = {"topic": topic, "limit": limit}
        return await self._client.request("GET", "/api/v1/consensus/similar", params=params)

    async def get_settled_topics(
        self, min_confidence: float = 0.8, limit: int = 20
    ) -> dict[str, Any]:
        params = {"min_confidence": min_confidence, "limit": limit}
        return await self._client.request("GET", "/api/v1/consensus/settled", params=params)

    async def get_stats(self, domain: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/v1/consensus/stats", params=params)

    async def get_dissents(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/v1/consensus/dissents", params=params)

    async def get_contrarian_views(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return await self._client.request(
            "GET", "/api/v1/consensus/contrarian-views", params=params
        )

    async def get_risk_warnings(
        self,
        topic: str | None = None,
        domain: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if topic:
            params["topic"] = topic
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/v1/consensus/risk-warnings", params=params)

    async def get_domain_history(self, domain: str, limit: int = 50) -> dict[str, Any]:
        params = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/v1/consensus/domain/{domain}", params=params
        )

    async def seed_demo(self) -> dict[str, Any]:
        return await self._client.request("POST", "/api/v1/consensus/seed-demo")
