"""
Stigmergic Coordination - Indirect agent communication through environment.

Implements a pheromone-like signaling system where agents communicate
indirectly by modifying the shared environment (knowledge mound).
Signals decay over time unless reinforced by other agents.

Inspired by termite colony coordination patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Types of stigmergic signals."""

    ATTENTION = "attention"  # "This needs review"
    WARNING = "warning"  # "Caution here"
    SUCCESS = "success"  # "This approach worked"
    FAILURE = "failure"  # "This approach failed"
    QUESTION = "question"  # "Needs clarification"
    INSIGHT = "insight"  # "I discovered something"
    DEPENDENCY = "dependency"  # "This relates to X"
    CONTROVERSY = "controversy"  # "Agents disagreed here"
    CONSENSUS = "consensus"  # "Agents agreed here"


@dataclass
class StigmergicSignal:
    """
    A signal left by an agent for others to discover.

    Implements indirect communication through environment modification.
    Signals decay over time unless reinforced by other agents.
    """

    id: str
    signal_type: SignalType
    target_id: str  # What this signal is about (fact, pattern, document, etc.)
    target_type: str  # "fact", "pattern", "document", "debate", etc.
    content: str  # Signal message

    # Signal strength and decay
    intensity: float = 1.0  # Current strength (0-1)
    decay_rate: float = 0.1  # Per day
    reinforcement_count: int = 0

    # Agent info
    emitter_agent_id: str = ""
    reinforcing_agents: list[str] = field(default_factory=list)
    workspace_id: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)

    @property
    def current_intensity(self) -> float:
        """Calculate current intensity with decay."""
        days_since = (datetime.now() - self.last_reinforced).total_seconds() / 86400
        decay = days_since * self.decay_rate
        base = self.intensity * (1.0 - decay)
        # Reinforcement adds longevity
        reinforcement_bonus = min(0.3, self.reinforcement_count * 0.05)
        return max(0.0, min(1.0, base + reinforcement_bonus))

    @property
    def is_expired(self) -> bool:
        """Check if signal has decayed below threshold."""
        return self.current_intensity < 0.1

    @property
    def age_days(self) -> float:
        """Get age of signal in days."""
        return (datetime.now() - self.created_at).total_seconds() / 86400

    def reinforce(self, agent_id: str) -> None:
        """Reinforce this signal (another agent found it valuable)."""
        self.reinforcement_count += 1
        self.last_reinforced = datetime.now()
        self.intensity = min(1.0, self.intensity + 0.1)
        if agent_id not in self.reinforcing_agents:
            self.reinforcing_agents.append(agent_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "signal_type": self.signal_type.value,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "content": self.content,
            "intensity": self.intensity,
            "decay_rate": self.decay_rate,
            "reinforcement_count": self.reinforcement_count,
            "emitter_agent_id": self.emitter_agent_id,
            "reinforcing_agents": self.reinforcing_agents,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StigmergicSignal:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            signal_type=SignalType(data["signal_type"]),
            target_id=data["target_id"],
            target_type=data["target_type"],
            content=data["content"],
            intensity=data.get("intensity", 1.0),
            decay_rate=data.get("decay_rate", 0.1),
            reinforcement_count=data.get("reinforcement_count", 0),
            emitter_agent_id=data.get("emitter_agent_id", ""),
            reinforcing_agents=data.get("reinforcing_agents", []),
            workspace_id=data.get("workspace_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"]) if data.get("last_reinforced") else datetime.now(),
        )


@dataclass
class PheromoneTrail:
    """
    A sequence of related signals forming a trail.

    Represents emergent paths through the knowledge space
    that multiple agents have followed.
    """

    id: str
    name: str
    signals: list[str]  # Signal IDs in order
    total_intensity: float
    agent_count: int
    workspace_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def strength(self) -> float:
        """Trail strength based on usage and intensity."""
        return self.total_intensity * (1.0 + 0.1 * self.agent_count)


class StigmergyManager:
    """
    Manages stigmergic signals and pheromone trails.

    Enables agents to communicate indirectly through the environment,
    similar to how ants and termites coordinate through pheromones.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize stigmergy manager."""
        self._signals: dict[str, StigmergicSignal] = {}
        self._trails: dict[str, PheromoneTrail] = {}
        self._db_path = db_path

    async def emit_signal(
        self,
        signal_type: SignalType,
        target_id: str,
        target_type: str,
        content: str,
        agent_id: str,
        workspace_id: str = "",
    ) -> str:
        """
        Emit a new signal or reinforce existing one.

        If a similar signal already exists for the same target,
        reinforce it instead of creating a duplicate.

        Args:
            signal_type: Type of signal
            target_id: ID of the target (fact, pattern, etc.)
            target_type: Type of target
            content: Signal message
            agent_id: ID of emitting agent
            workspace_id: Workspace scope

        Returns:
            Signal ID
        """
        # Check for existing signal on same target
        existing = await self._find_similar_signal(
            signal_type, target_id, workspace_id
        )

        if existing:
            # Reinforce existing signal
            existing.reinforce(agent_id)
            return existing.id

        # Create new signal
        signal_id = f"sig_{uuid.uuid4().hex[:12]}"
        signal = StigmergicSignal(
            id=signal_id,
            signal_type=signal_type,
            target_id=target_id,
            target_type=target_type,
            content=content,
            emitter_agent_id=agent_id,
            workspace_id=workspace_id,
        )

        self._signals[signal_id] = signal
        return signal_id

    async def _find_similar_signal(
        self,
        signal_type: SignalType,
        target_id: str,
        workspace_id: str,
    ) -> Optional[StigmergicSignal]:
        """Find an existing signal for the same target."""
        for signal in self._signals.values():
            if (
                signal.signal_type == signal_type
                and signal.target_id == target_id
                and signal.workspace_id == workspace_id
                and not signal.is_expired
            ):
                return signal
        return None

    async def get_signals_for_target(
        self,
        target_id: str,
        signal_types: Optional[list[SignalType]] = None,
        min_intensity: float = 0.1,
    ) -> list[StigmergicSignal]:
        """Get active signals for a target."""
        results = []
        for signal in self._signals.values():
            if signal.target_id != target_id:
                continue
            if signal.current_intensity < min_intensity:
                continue
            if signal_types and signal.signal_type not in signal_types:
                continue
            results.append(signal)

        # Sort by intensity descending
        results.sort(key=lambda s: s.current_intensity, reverse=True)
        return results

    async def get_attention_signals(
        self,
        workspace_id: str,
        min_intensity: float = 0.3,
        limit: int = 20,
    ) -> list[StigmergicSignal]:
        """Get active attention signals for a workspace."""
        attention_types = [SignalType.ATTENTION, SignalType.WARNING, SignalType.CONTROVERSY]

        results = []
        for signal in self._signals.values():
            if signal.workspace_id != workspace_id:
                continue
            if signal.signal_type not in attention_types:
                continue
            if signal.current_intensity < min_intensity:
                continue
            results.append(signal)

        # Sort by intensity descending
        results.sort(key=lambda s: s.current_intensity, reverse=True)
        return results[:limit]

    async def follow_trail(
        self,
        start_id: str,
        signal_types: Optional[list[SignalType]] = None,
        max_depth: int = 10,
    ) -> list[StigmergicSignal]:
        """
        Follow a pheromone trail from a starting point.

        Traverses connected signals via dependency relationships.
        """
        visited = set()
        trail = []

        def _follow(target_id: str, depth: int) -> None:
            if depth >= max_depth or target_id in visited:
                return

            visited.add(target_id)

            # Get signals pointing to this target
            for signal in self._signals.values():
                if signal.target_id == target_id and not signal.is_expired:
                    if signal_types is None or signal.signal_type in signal_types:
                        trail.append(signal)

                        # Follow dependencies
                        if signal.signal_type == SignalType.DEPENDENCY:
                            # The content might reference another target
                            pass  # Could parse content for next target

        _follow(start_id, 0)
        return trail

    async def cleanup_expired(self) -> int:
        """Remove expired signals."""
        expired_ids = [
            signal_id
            for signal_id, signal in self._signals.items()
            if signal.is_expired
        ]

        for signal_id in expired_ids:
            del self._signals[signal_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired signals")

        return len(expired_ids)

    async def get_signal(self, signal_id: str) -> Optional[StigmergicSignal]:
        """Get a signal by ID."""
        return self._signals.get(signal_id)

    async def reinforce_signal(self, signal_id: str, agent_id: str) -> bool:
        """Reinforce an existing signal."""
        signal = self._signals.get(signal_id)
        if signal and not signal.is_expired:
            signal.reinforce(agent_id)
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about signals."""
        active_signals = [s for s in self._signals.values() if not s.is_expired]

        by_type: dict[str, int] = {}
        for signal in active_signals:
            by_type[signal.signal_type.value] = by_type.get(signal.signal_type.value, 0) + 1

        return {
            "total_signals": len(self._signals),
            "active_signals": len(active_signals),
            "by_type": by_type,
            "trails": len(self._trails),
        }
