"""
Decisions API resource for the Aragora client.

Provides methods for unified decision-making:
- Create decision requests (debate, workflow, gauntlet, quick)
- Get decision results
- Poll decision status
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class DecisionConfig:
    """Configuration for a decision request."""

    agents: List[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])
    rounds: int = 3
    consensus: str = "majority"
    timeout_seconds: int = 300


@dataclass
class DecisionContext:
    """Context for a decision request."""

    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseChannel:
    """A response channel for delivering results."""

    platform: str  # http_api, slack, email, webhook
    target: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """Result of a decision request."""

    request_id: str
    status: str  # pending, processing, completed, failed
    decision_type: str  # debate, workflow, gauntlet, quick, auto
    content: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionStatus:
    """Status of a decision request."""

    request_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    current_stage: Optional[str] = None
    estimated_remaining_seconds: Optional[int] = None


class DecisionsAPI:
    """API interface for unified decision-making."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Decision CRUD
    # =========================================================================

    def create(
        self,
        content: str,
        decision_type: str = "auto",
        config: Optional[DecisionConfig] = None,
        context: Optional[DecisionContext] = None,
        priority: str = "normal",
        response_channels: Optional[List[ResponseChannel]] = None,
    ) -> DecisionResult:
        """
        Create a new decision request.

        Args:
            content: Question or topic for the decision.
            decision_type: Type of decision (debate, workflow, gauntlet, quick, auto).
            config: Decision configuration.
            context: Decision context.
            priority: Priority level (high, normal, low).
            response_channels: Channels for delivering results.

        Returns:
            DecisionResult object.
        """
        body: Dict[str, Any] = {
            "content": content,
            "decision_type": decision_type,
            "priority": priority,
        }

        if config:
            body["config"] = {
                "agents": config.agents,
                "rounds": config.rounds,
                "consensus": config.consensus,
                "timeout_seconds": config.timeout_seconds,
            }

        if context:
            body["context"] = {
                "user_id": context.user_id,
                "workspace_id": context.workspace_id,
                **context.metadata,
            }

        if response_channels:
            body["response_channels"] = [
                {
                    "platform": ch.platform,
                    "target": ch.target,
                    **ch.options,
                }
                for ch in response_channels
            ]

        response = self._client._post("/api/v1/decisions", body)
        return self._parse_result(response)

    async def create_async(
        self,
        content: str,
        decision_type: str = "auto",
        config: Optional[DecisionConfig] = None,
        context: Optional[DecisionContext] = None,
        priority: str = "normal",
        response_channels: Optional[List[ResponseChannel]] = None,
    ) -> DecisionResult:
        """Async version of create()."""
        body: Dict[str, Any] = {
            "content": content,
            "decision_type": decision_type,
            "priority": priority,
        }

        if config:
            body["config"] = {
                "agents": config.agents,
                "rounds": config.rounds,
                "consensus": config.consensus,
                "timeout_seconds": config.timeout_seconds,
            }

        if context:
            body["context"] = {
                "user_id": context.user_id,
                "workspace_id": context.workspace_id,
                **context.metadata,
            }

        if response_channels:
            body["response_channels"] = [
                {
                    "platform": ch.platform,
                    "target": ch.target,
                    **ch.options,
                }
                for ch in response_channels
            ]

        response = await self._client._post_async("/api/v1/decisions", body)
        return self._parse_result(response)

    def get(self, request_id: str) -> DecisionResult:
        """
        Get decision result by ID.

        Args:
            request_id: The decision request ID.

        Returns:
            DecisionResult object.
        """
        response = self._client._get(f"/api/v1/decisions/{request_id}")
        return self._parse_result(response)

    async def get_async(self, request_id: str) -> DecisionResult:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/v1/decisions/{request_id}")
        return self._parse_result(response)

    def get_status(self, request_id: str) -> DecisionStatus:
        """
        Get decision status for polling.

        Args:
            request_id: The decision request ID.

        Returns:
            DecisionStatus object.
        """
        response = self._client._get(f"/api/v1/decisions/{request_id}/status")
        return self._parse_status(response)

    async def get_status_async(self, request_id: str) -> DecisionStatus:
        """Async version of get_status()."""
        response = await self._client._get_async(f"/api/v1/decisions/{request_id}/status")
        return self._parse_status(response)

    def list(
        self,
        status: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[DecisionResult], int]:
        """
        List recent decisions.

        Args:
            status: Filter by status.
            decision_type: Filter by decision type.
            limit: Maximum number of results.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of DecisionResult objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if decision_type:
            params["decision_type"] = decision_type

        response = self._client._get("/api/v1/decisions", params=params)
        results = [self._parse_result(r) for r in response.get("decisions", [])]
        return results, response.get("total", len(results))

    async def list_async(
        self,
        status: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[DecisionResult], int]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if decision_type:
            params["decision_type"] = decision_type

        response = await self._client._get_async("/api/v1/decisions", params=params)
        results = [self._parse_result(r) for r in response.get("decisions", [])]
        return results, response.get("total", len(results))

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def quick_decision(
        self,
        question: str,
        agents: Optional[List[str]] = None,
    ) -> DecisionResult:
        """
        Make a quick decision with minimal configuration.

        Args:
            question: The question to decide.
            agents: Optional list of agents to use.

        Returns:
            DecisionResult object.
        """
        config = DecisionConfig(
            agents=agents or ["anthropic-api", "openai-api"],
            rounds=2,
            consensus="majority",
            timeout_seconds=60,
        )
        return self.create(
            content=question,
            decision_type="quick",
            config=config,
        )

    async def quick_decision_async(
        self,
        question: str,
        agents: Optional[List[str]] = None,
    ) -> DecisionResult:
        """Async version of quick_decision()."""
        config = DecisionConfig(
            agents=agents or ["anthropic-api", "openai-api"],
            rounds=2,
            consensus="majority",
            timeout_seconds=60,
        )
        return await self.create_async(
            content=question,
            decision_type="quick",
            config=config,
        )

    def start_debate(
        self,
        topic: str,
        agents: Optional[List[str]] = None,
        rounds: int = 3,
    ) -> DecisionResult:
        """
        Start a full debate on a topic.

        Args:
            topic: The debate topic.
            agents: Optional list of agents to use.
            rounds: Number of debate rounds.

        Returns:
            DecisionResult object.
        """
        config = DecisionConfig(
            agents=agents or ["anthropic-api", "openai-api", "gemini-api"],
            rounds=rounds,
            consensus="majority",
            timeout_seconds=300,
        )
        return self.create(
            content=topic,
            decision_type="debate",
            config=config,
        )

    async def start_debate_async(
        self,
        topic: str,
        agents: Optional[List[str]] = None,
        rounds: int = 3,
    ) -> DecisionResult:
        """Async version of start_debate()."""
        config = DecisionConfig(
            agents=agents or ["anthropic-api", "openai-api", "gemini-api"],
            rounds=rounds,
            consensus="majority",
            timeout_seconds=300,
        )
        return await self.create_async(
            content=topic,
            decision_type="debate",
            config=config,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_result(self, data: Dict[str, Any]) -> DecisionResult:
        """Parse result data into DecisionResult object."""
        created_at = None
        completed_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return DecisionResult(
            request_id=data.get("request_id", data.get("id", "")),
            status=data.get("status", "pending"),
            decision_type=data.get("decision_type", "auto"),
            content=data.get("content", ""),
            result=data.get("result"),
            error=data.get("error"),
            created_at=created_at,
            completed_at=completed_at,
            metadata=data.get("metadata", {}),
        )

    def _parse_status(self, data: Dict[str, Any]) -> DecisionStatus:
        """Parse status data into DecisionStatus object."""
        return DecisionStatus(
            request_id=data.get("request_id", data.get("id", "")),
            status=data.get("status", "pending"),
            progress=data.get("progress", 0.0),
            current_stage=data.get("current_stage"),
            estimated_remaining_seconds=data.get("estimated_remaining_seconds"),
        )


__all__ = [
    "DecisionsAPI",
    "DecisionResult",
    "DecisionStatus",
    "DecisionConfig",
    "DecisionContext",
    "ResponseChannel",
]
