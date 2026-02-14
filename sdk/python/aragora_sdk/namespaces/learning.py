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

    def get_pattern(self, pattern_id: str) -> dict[str, Any]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: The pattern ID

        Returns:
            Dict with pattern details
        """
        return self._client.request("GET", f"/api/v1/learning/patterns/{pattern_id}")

    def validate_pattern(
        self,
        pattern_id: str,
    ) -> dict[str, Any]:
        """
        Validate a detected pattern.

        Args:
            pattern_id: The pattern ID to validate

        Returns:
            Dict with validated pattern and confirmation message
        """
        return self._client.request("POST", f"/api/v1/learning/patterns/{pattern_id}/validate")

    def get_metrics(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get learning metrics with filtering.

        Args:
            session_id: Filter by session ID
            agent_id: Filter by agent ID
            limit: Maximum number of metrics to return

        Returns:
            Dict with metrics list and count
        """
        params: dict[str, Any] = {}
        if session_id:
            params["session_id"] = session_id
        if agent_id:
            params["agent_id"] = agent_id
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/learning/metrics", params=params if params else None
        )

    def get_metric_by_type(self, metric_type: str) -> dict[str, Any]:
        """
        Get metrics of a specific type with aggregations.

        Args:
            metric_type: The metric type (accuracy, loss, precision, etc.)

        Returns:
            Dict with metric type, count, average, min, max, and recent values
        """
        return self._client.request("GET", f"/api/v1/learning/metrics/{metric_type}")

    def submit_feedback(
        self,
        feedback_type: str,
        target_type: str,
        target_id: str,
        comment: str = "",
        rating: int | None = None,
    ) -> dict[str, Any]:
        """
        Submit feedback on learning outcomes.

        Args:
            feedback_type: Feedback type (positive, negative, neutral, correction)
            target_type: Target type (session, pattern, knowledge)
            target_id: Target entity ID
            comment: Feedback comment
            rating: Optional 1-5 rating

        Returns:
            Dict with feedback record and confirmation message
        """
        data: dict[str, Any] = {
            "feedback_type": feedback_type,
            "target_type": target_type,
            "target_id": target_id,
            "comment": comment,
        }
        if rating is not None:
            data["rating"] = rating
        return self._client.request("POST", "/api/v1/learning/feedback", json=data)

    def list_knowledge(
        self,
        verified: bool | None = None,
        source_type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        List extracted knowledge items.

        Args:
            verified: Filter by verification status
            source_type: Filter by source type
            limit: Maximum number of items to return

        Returns:
            Dict with knowledge items and count
        """
        params: dict[str, Any] = {}
        if verified is not None:
            params["verified"] = str(verified).lower()
        if source_type:
            params["source_type"] = source_type
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/learning/knowledge", params=params if params else None
        )

    def get_knowledge_item(self, knowledge_id: str) -> dict[str, Any]:
        """
        Get a specific knowledge item by ID.

        Args:
            knowledge_id: The knowledge item ID

        Returns:
            Dict with knowledge item details
        """
        return self._client.request("GET", f"/api/v1/learning/knowledge/{knowledge_id}")

    def extract_knowledge(
        self,
        debate_ids: list[str],
        title: str | None = None,
        content: str | None = None,
        topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Trigger knowledge extraction from debates.

        Args:
            debate_ids: List of debate IDs to extract knowledge from
            title: Optional title for extracted knowledge
            content: Optional content description
            topics: Optional list of topics

        Returns:
            Dict with extracted knowledge record and confirmation message
        """
        data: dict[str, Any] = {"debate_ids": debate_ids}
        if title:
            data["title"] = title
        if content:
            data["content"] = content
        if topics:
            data["topics"] = topics
        return self._client.request("POST", "/api/v1/learning/knowledge/extract", json=data)

    def get_recommendations(
        self,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get learning recommendations.

        Args:
            limit: Maximum number of recommendations to return

        Returns:
            Dict with recommendations list and count
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/learning/recommendations", params=params if params else None
        )

    def get_performance(self) -> dict[str, Any]:
        """
        Get model performance statistics.

        Returns:
            Dict with performance stats including accuracy, loss, session counts
        """
        return self._client.request("GET", "/api/v1/learning/performance")

    def calibrate(
        self,
        agent_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger model calibration.

        Args:
            agent_ids: Optional list of agent IDs to calibrate
            force: Force recalibration even if recent calibration exists

        Returns:
            Dict with calibration ID, metric, and confirmation message
        """
        data: dict[str, Any] = {"force": force}
        if agent_ids:
            data["agent_ids"] = agent_ids
        return self._client.request("POST", "/api/v1/learning/calibrate", json=data)

    def create_session(
        self,
        name: str,
        mode: str = "supervised",
        total_epochs: int = 100,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new training session.

        Args:
            name: Session name
            mode: Learning mode (supervised, reinforcement, self_supervised, transfer, federated)
            total_epochs: Total epochs for training
            config: Optional session configuration

        Returns:
            Dict with created session record and confirmation message
        """
        data: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "total_epochs": total_epochs,
        }
        if config:
            data["config"] = config
        return self._client.request("POST", "/api/v1/learning/sessions", json=data)

    def stop_session(self, session_id: str) -> dict[str, Any]:
        """
        Stop a running training session.

        Args:
            session_id: The session ID to stop

        Returns:
            Dict with stopped session record and confirmation message
        """
        return self._client.request("POST", f"/api/v1/learning/sessions/{session_id}/stop")

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

