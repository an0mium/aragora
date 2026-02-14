"""
Learning Namespace API

Provides access to autonomous learning and meta-learning analytics,
including learning patterns, efficiency metrics, and agent evolution.

Features:
- Get meta-learning statistics
- List learning sessions
- Discover learning patterns
- Analyze learning efficiency
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

SessionStatus = Literal["active", "completed", "failed"]

class LearningAPI:
    """
    Synchronous Learning API.

    Provides access to autonomous learning and meta-learning analytics:
    - Get meta-learning statistics
    - List learning sessions
    - Discover learning patterns
    - Analyze learning efficiency

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> stats = client.learning.get_stats()
        >>> patterns = client.learning.list_patterns(min_confidence=0.8)
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_sessions(
        self,
        agent: str | None = None,
        domain: str | None = None,
        status: SessionStatus | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        List learning sessions with optional filtering.

        Args:
            agent: Filter by agent name
            domain: Filter by domain
            status: Filter by status
            limit: Maximum number of sessions
            offset: Number of sessions to skip

        Returns:
            Dict with sessions list and total count
        """
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if domain:
            params["domain"] = domain
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return self._client.request(
            "GET", "/api/v1/learning/sessions", params=params if params else None
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Get a specific learning session by ID.

        Args:
            session_id: The session ID

        Returns:
            Dict with session details
        """
        return self._client.request("GET", f"/api/v1/learning/sessions/{session_id}")

    def list_patterns(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        List detected learning patterns.

        Args:
            min_confidence: Minimum confidence threshold (0-1)
            limit: Maximum number of patterns

        Returns:
            Dict with patterns list and count
        """
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/learning/patterns", params=params if params else None
        )

class AsyncLearningAPI:
    """
    Asynchronous Learning API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     stats = await client.learning.get_stats()
        ...     efficiency = await client.learning.get_efficiency("claude", "coding")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_sessions(
        self,
        agent: str | None = None,
        domain: str | None = None,
        status: SessionStatus | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List learning sessions with optional filtering."""
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if domain:
            params["domain"] = domain
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/api/v1/learning/sessions", params=params if params else None
        )

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get a specific learning session by ID."""
        return await self._client.request("GET", f"/api/v1/learning/sessions/{session_id}")

    async def list_patterns(
        self,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """List detected learning patterns."""
        params: dict[str, Any] = {}
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        if limit:
            params["limit"] = limit
        return await self._client.request(
            "GET", "/api/v1/learning/patterns", params=params if params else None
        )

