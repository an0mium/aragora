"""Replay API for the Aragora SDK.

Provides access to debate replays and learning evolution data:
- Replay listing and retrieval
- Paginated event streaming
- Learning evolution patterns
- Meta-learning statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


@dataclass
class ReplaySummary:
    """Summary of a debate replay."""

    id: str
    topic: str
    agents: list[str]
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplaySummary:
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            agents=data.get("agents", []),
            schema_version=data.get("schema_version", "1.0"),
        )


@dataclass
class ReplayMeta:
    """Metadata for a debate replay."""

    topic: str
    agents: list[dict[str, Any]]
    schema_version: str
    created_at: str | None = None
    duration_seconds: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayMeta:
        return cls(
            topic=data.get("topic", ""),
            agents=data.get("agents", []),
            schema_version=data.get("schema_version", "1.0"),
            created_at=data.get("created_at"),
            duration_seconds=data.get("duration_seconds"),
        )


@dataclass
class ReplayEvent:
    """An event in a replay timeline."""

    event_type: str
    timestamp: str
    agent_id: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayEvent:
        return cls(
            event_type=data.get("event_type", data.get("type", "")),
            timestamp=data.get("timestamp", ""),
            agent_id=data.get("agent_id"),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Replay:
    """Full debate replay with events."""

    id: str
    meta: ReplayMeta
    events: list[ReplayEvent]
    event_count: int
    total_events: int
    offset: int
    limit: int
    has_more: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Replay:
        return cls(
            id=data.get("id", ""),
            meta=ReplayMeta.from_dict(data.get("meta", {})),
            events=[ReplayEvent.from_dict(e) for e in data.get("events", [])],
            event_count=data.get("event_count", 0),
            total_events=data.get("total_events", 0),
            offset=data.get("offset", 0),
            limit=data.get("limit", 1000),
            has_more=data.get("has_more", False),
        )


@dataclass
class LearningPattern:
    """A learning pattern from meta-learning."""

    date: str
    issue_type: str
    success_rate: float
    pattern_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningPattern:
        return cls(
            date=data.get("date", ""),
            issue_type=data.get("issue_type", "unknown"),
            success_rate=data.get("success_rate", 0.5),
            pattern_count=data.get("pattern_count", 1),
        )


@dataclass
class AgentEvolution:
    """Agent performance evolution over time."""

    agent: str
    date: str
    acceptance_rate: float
    critique_quality: float
    reputation_score: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEvolution:
        return cls(
            agent=data.get("agent", ""),
            date=data.get("date", ""),
            acceptance_rate=data.get("acceptance_rate", 0.5),
            critique_quality=data.get("critique_quality", 0.5),
            reputation_score=data.get("reputation_score", 0.5),
        )


@dataclass
class DebateStats:
    """Debate statistics for a time period."""

    date: str
    total_debates: int
    consensus_rate: float
    avg_confidence: float
    avg_rounds: float
    avg_duration: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateStats:
        return cls(
            date=data.get("date", ""),
            total_debates=data.get("total_debates", 0),
            consensus_rate=data.get("consensus_rate", 0.0),
            avg_confidence=data.get("avg_confidence", 0.5),
            avg_rounds=data.get("avg_rounds", 3.0),
            avg_duration=data.get("avg_duration", 60.0),
        )


@dataclass
class LearningEvolution:
    """Learning evolution data combining patterns, agents, and debates."""

    patterns: list[LearningPattern]
    patterns_count: int
    agents: list[AgentEvolution]
    agents_count: int
    debates: list[DebateStats]
    debates_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningEvolution:
        return cls(
            patterns=[LearningPattern.from_dict(p) for p in data.get("patterns", [])],
            patterns_count=data.get("patterns_count", 0),
            agents=[AgentEvolution.from_dict(a) for a in data.get("agents", [])],
            agents_count=data.get("agents_count", 0),
            debates=[DebateStats.from_dict(d) for d in data.get("debates", [])],
            debates_count=data.get("debates_count", 0),
        )


@dataclass
class HyperparamAdjustment:
    """Record of a hyperparameter adjustment."""

    hyperparams: dict[str, Any]
    metrics: dict[str, Any] | None
    reason: str | None
    timestamp: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperparamAdjustment:
        return cls(
            hyperparams=data.get("hyperparams", {}),
            metrics=data.get("metrics"),
            reason=data.get("reason"),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class EfficiencyLogEntry:
    """An entry in the efficiency log."""

    cycle: int
    metrics: dict[str, Any]
    timestamp: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EfficiencyLogEntry:
        return cls(
            cycle=data.get("cycle", 0),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class MetaLearningStats:
    """Meta-learning statistics and hyperparameters."""

    status: str
    current_hyperparams: dict[str, Any]
    adjustment_history: list[HyperparamAdjustment]
    efficiency_log: list[EfficiencyLogEntry]
    trend: str | None = None
    evaluations: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaLearningStats:
        return cls(
            status=data.get("status", "unknown"),
            current_hyperparams=data.get("current_hyperparams", {}),
            adjustment_history=[
                HyperparamAdjustment.from_dict(a)
                for a in data.get("adjustment_history", [])
            ],
            efficiency_log=[
                EfficiencyLogEntry.from_dict(e) for e in data.get("efficiency_log", [])
            ],
            trend=data.get("trend"),
            evaluations=data.get("evaluations", 0),
        )


class ReplayAPI:
    """API for replay and learning evolution operations.

    Provides access to debate replays, learning patterns, and meta-learning
    statistics for analyzing system evolution over time.
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Replay Operations
    # =========================================================================

    async def list(self, *, limit: int = 100) -> list[ReplaySummary]:
        """List available debate replays.

        Args:
            limit: Maximum replays to return (1-500)

        Returns:
            List of ReplaySummary objects

        Example:
            replays = await client.replays.list(limit=20)
            for r in replays:
                print(f"{r.id}: {r.topic}")
        """
        params = {"limit": min(max(1, limit), 500)}
        response = await self._client._get("/api/v1/replays", params=params)
        if isinstance(response, list):
            return [ReplaySummary.from_dict(r) for r in response]
        return [ReplaySummary.from_dict(r) for r in response.get("replays", [])]

    async def get(
        self,
        replay_id: str,
        *,
        offset: int = 0,
        limit: int = 1000,
    ) -> Replay:
        """Get a specific replay with events.

        Supports pagination for large replays with many events.

        Args:
            replay_id: The replay ID
            offset: Number of events to skip
            limit: Maximum events to return (1-5000)

        Returns:
            Replay with metadata and events

        Example:
            replay = await client.replays.get("replay_abc123")
            print(f"Topic: {replay.meta.topic}")
            for event in replay.events:
                print(f"  {event.event_type}: {event.content}")

            # Pagination for large replays
            if replay.has_more:
                next_page = await client.replays.get(
                    "replay_abc123",
                    offset=replay.offset + replay.event_count
                )
        """
        params = {
            "offset": max(0, offset),
            "limit": min(max(1, limit), 5000),
        }
        response = await self._client._get(
            f"/api/v1/replays/{replay_id}", params=params
        )
        return Replay.from_dict(response)

    async def get_all_events(self, replay_id: str) -> list[ReplayEvent]:
        """Get all events from a replay, handling pagination automatically.

        Args:
            replay_id: The replay ID

        Returns:
            List of all ReplayEvent objects

        Example:
            events = await client.replays.get_all_events("replay_abc123")
            print(f"Total events: {len(events)}")
        """
        all_events: list[ReplayEvent] = []
        offset = 0
        limit = 5000

        while True:
            replay = await self.get(replay_id, offset=offset, limit=limit)
            all_events.extend(replay.events)
            if not replay.has_more:
                break
            offset += replay.event_count

        return all_events

    # =========================================================================
    # Learning Evolution
    # =========================================================================

    async def get_learning_evolution(self, *, limit: int = 20) -> LearningEvolution:
        """Get learning evolution data.

        Returns patterns, agent performance, and debate statistics
        over time to visualize system learning.

        Args:
            limit: Maximum items per category (1-100)

        Returns:
            LearningEvolution with patterns, agents, and debates

        Example:
            evolution = await client.replays.get_learning_evolution()
            print(f"Patterns: {evolution.patterns_count}")
            for pattern in evolution.patterns:
                print(f"  {pattern.issue_type}: {pattern.success_rate:.1%}")
        """
        params = {"limit": min(max(1, limit), 100)}
        response = await self._client._get("/api/v1/learning/evolution", params=params)
        return LearningEvolution.from_dict(response)

    # =========================================================================
    # Meta-Learning Statistics
    # =========================================================================

    async def get_meta_learning_stats(self, *, limit: int = 20) -> MetaLearningStats:
        """Get meta-learning hyperparameters and efficiency stats.

        Returns current hyperparameters, adjustment history, and
        efficiency metrics for analyzing meta-learning performance.

        Args:
            limit: Maximum history entries to return (1-50)

        Returns:
            MetaLearningStats with hyperparams, history, and efficiency

        Example:
            stats = await client.replays.get_meta_learning_stats()
            print(f"Status: {stats.status}")
            print(f"Trend: {stats.trend}")
            print(f"Current params: {stats.current_hyperparams}")
        """
        params = {"limit": min(max(1, limit), 50)}
        response = await self._client._get("/api/v1/meta-learning/stats", params=params)
        return MetaLearningStats.from_dict(response)
