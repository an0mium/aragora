"""
Position Tracker - Track agent position evolution across debate rounds.

Enables decision explainability by tracking:
- Initial positions
- Position shifts between rounds
- Vote pivots (when agents change their stance)
- Consensus convergence patterns
- Position stability metrics

"Every change of mind tells a story."
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PositionStance(str, Enum):
    """Agent stance on a proposition."""

    STRONGLY_AGREE = "strongly_agree"
    AGREE = "agree"
    LEAN_AGREE = "lean_agree"
    NEUTRAL = "neutral"
    LEAN_DISAGREE = "lean_disagree"
    DISAGREE = "disagree"
    STRONGLY_DISAGREE = "strongly_disagree"

    @classmethod
    def from_confidence(cls, confidence: float, agrees: bool) -> "PositionStance":
        """Convert confidence score to stance."""
        if confidence >= 0.9:
            return cls.STRONGLY_AGREE if agrees else cls.STRONGLY_DISAGREE
        elif confidence >= 0.7:
            return cls.AGREE if agrees else cls.DISAGREE
        elif confidence >= 0.5:
            return cls.LEAN_AGREE if agrees else cls.LEAN_DISAGREE
        else:
            return cls.NEUTRAL

    @property
    def numeric_value(self) -> float:
        """Convert stance to numeric value for analysis."""
        values = {
            PositionStance.STRONGLY_AGREE: 1.0,
            PositionStance.AGREE: 0.75,
            PositionStance.LEAN_AGREE: 0.55,
            PositionStance.NEUTRAL: 0.5,
            PositionStance.LEAN_DISAGREE: 0.45,
            PositionStance.DISAGREE: 0.25,
            PositionStance.STRONGLY_DISAGREE: 0.0,
        }
        return values.get(self, 0.5)


@dataclass
class PositionRecord:
    """Record of an agent's position at a specific point in time."""

    agent: str
    round_number: int
    stance: PositionStance
    confidence: float
    key_argument: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    influenced_by: list[str] = field(default_factory=list)  # Agent names
    evidence_cited: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "round": self.round_number,
            "stance": self.stance.value,
            "confidence": self.confidence,
            "key_argument": self.key_argument,
            "timestamp": self.timestamp.isoformat(),
            "influenced_by": self.influenced_by,
            "evidence_cited": self.evidence_cited,
        }


@dataclass
class PositionPivot:
    """Record of when an agent changed their position."""

    agent: str
    from_round: int
    to_round: int
    from_stance: PositionStance
    to_stance: PositionStance
    from_confidence: float
    to_confidence: float
    trigger_argument: Optional[str] = None
    trigger_agent: Optional[str] = None
    pivot_magnitude: float = 0.0
    pivot_type: str = "shift"  # shift, reversal, strengthening, weakening

    def __post_init__(self) -> None:
        """Calculate pivot magnitude and type."""
        from_val = self.from_stance.numeric_value
        to_val = self.to_stance.numeric_value
        self.pivot_magnitude = abs(to_val - from_val)

        # Determine pivot type
        if from_val < 0.5 and to_val > 0.5 or from_val > 0.5 and to_val < 0.5:
            self.pivot_type = "reversal"
        elif abs(to_val - 0.5) > abs(from_val - 0.5):
            self.pivot_type = "strengthening"
        elif abs(to_val - 0.5) < abs(from_val - 0.5):
            self.pivot_type = "weakening"
        else:
            self.pivot_type = "shift"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "from_round": self.from_round,
            "to_round": self.to_round,
            "from_stance": self.from_stance.value,
            "to_stance": self.to_stance.value,
            "from_confidence": self.from_confidence,
            "to_confidence": self.to_confidence,
            "trigger_argument": self.trigger_argument,
            "trigger_agent": self.trigger_agent,
            "pivot_magnitude": self.pivot_magnitude,
            "pivot_type": self.pivot_type,
        }


@dataclass
class PositionEvolution:
    """Complete evolution of positions for a debate."""

    debate_id: str
    topic: str
    positions: dict[str, list[PositionRecord]] = field(default_factory=dict)  # agent -> positions by round
    pivots: list[PositionPivot] = field(default_factory=list)

    def record_position(
        self,
        agent: str,
        round_number: int,
        stance: PositionStance,
        confidence: float,
        key_argument: str,
        influenced_by: Optional[list[str]] = None,
        evidence_cited: Optional[list[str]] = None,
    ) -> Optional[PositionPivot]:
        """
        Record an agent's position and detect pivots.

        Returns PositionPivot if a pivot was detected.
        """
        record = PositionRecord(
            agent=agent,
            round_number=round_number,
            stance=stance,
            confidence=confidence,
            key_argument=key_argument,
            influenced_by=influenced_by or [],
            evidence_cited=evidence_cited or [],
        )

        if agent not in self.positions:
            self.positions[agent] = []

        # Check for pivot from previous position
        pivot = None
        if self.positions[agent]:
            prev = self.positions[agent][-1]
            if prev.stance != stance:
                pivot = PositionPivot(
                    agent=agent,
                    from_round=prev.round_number,
                    to_round=round_number,
                    from_stance=prev.stance,
                    to_stance=stance,
                    from_confidence=prev.confidence,
                    to_confidence=confidence,
                    trigger_argument=key_argument[:200] if key_argument else None,
                    trigger_agent=influenced_by[0] if influenced_by else None,
                )
                self.pivots.append(pivot)
                logger.info(f"Position pivot detected: {agent} {prev.stance.value} -> {stance.value}")

        self.positions[agent].append(record)
        return pivot

    def get_agent_trajectory(self, agent: str) -> list[PositionRecord]:
        """Get position trajectory for an agent."""
        return self.positions.get(agent, [])

    def get_round_positions(self, round_number: int) -> dict[str, PositionRecord]:
        """Get all agent positions for a specific round."""
        result = {}
        for agent, records in self.positions.items():
            for record in records:
                if record.round_number == round_number:
                    result[agent] = record
                    break
        return result

    def get_pivots_for_agent(self, agent: str) -> list[PositionPivot]:
        """Get all pivots for a specific agent."""
        return [p for p in self.pivots if p.agent == agent]

    def get_reversals(self) -> list[PositionPivot]:
        """Get all full reversals (agree -> disagree or vice versa)."""
        return [p for p in self.pivots if p.pivot_type == "reversal"]

    def calculate_convergence_score(self) -> float:
        """
        Calculate how much positions converged over the debate.

        Returns value 0-1 where 1 means full convergence.
        """
        if not self.positions:
            return 0.0

        # Get final positions
        final_positions = []
        for agent_positions in self.positions.values():
            if agent_positions:
                final_positions.append(agent_positions[-1].stance.numeric_value)

        if len(final_positions) < 2:
            return 1.0

        # Calculate variance of final positions
        mean = sum(final_positions) / len(final_positions)
        variance = sum((p - mean) ** 2 for p in final_positions) / len(final_positions)

        # Convert variance to convergence score (lower variance = higher convergence)
        # Max variance is 0.25 (half at 0, half at 1)
        return max(0.0, 1.0 - (variance / 0.25))

    def calculate_stability_scores(self) -> dict[str, float]:
        """
        Calculate position stability for each agent.

        Returns agent -> stability score (0-1, higher = more stable).
        """
        stability = {}
        for agent, records in self.positions.items():
            if len(records) < 2:
                stability[agent] = 1.0
                continue

            # Count position changes
            changes = 0
            total_magnitude = 0.0
            for i in range(1, len(records)):
                prev_val = records[i - 1].stance.numeric_value
                curr_val = records[i].stance.numeric_value
                diff = abs(curr_val - prev_val)
                if diff > 0.1:  # Threshold for meaningful change
                    changes += 1
                    total_magnitude += diff

            # Stability = 1 - (changes / max_possible_changes) weighted by magnitude
            max_changes = len(records) - 1
            stability[agent] = 1.0 - (total_magnitude / max_changes if max_changes > 0 else 0.0)

        return stability

    def identify_influencers(self) -> dict[str, int]:
        """
        Identify agents who influenced the most pivots.

        Returns agent -> number of pivots they triggered.
        """
        influence_counts: dict[str, int] = {}
        for pivot in self.pivots:
            if pivot.trigger_agent:
                influence_counts[pivot.trigger_agent] = influence_counts.get(pivot.trigger_agent, 0) + 1
        return influence_counts

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "positions": {
                agent: [p.to_dict() for p in records]
                for agent, records in self.positions.items()
            },
            "pivots": [p.to_dict() for p in self.pivots],
            "summary": {
                "convergence_score": self.calculate_convergence_score(),
                "total_pivots": len(self.pivots),
                "reversals": len(self.get_reversals()),
                "stability_scores": self.calculate_stability_scores(),
                "influencers": self.identify_influencers(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class PositionTracker:
    """
    Service for tracking position evolution across debates.

    Usage:
        tracker = PositionTracker()
        evolution = tracker.create_evolution("debate-123", "Should we use microservices?")

        # Record positions as debate progresses
        evolution.record_position(
            agent="claude",
            round_number=1,
            stance=PositionStance.LEAN_AGREE,
            confidence=0.6,
            key_argument="Microservices improve scalability",
        )

        # Later...
        evolution.record_position(
            agent="claude",
            round_number=2,
            stance=PositionStance.STRONGLY_AGREE,
            confidence=0.9,
            key_argument="After considering deployment complexity...",
            influenced_by=["gpt"],
        )

        # Get analysis
        pivots = evolution.pivots
        convergence = evolution.calculate_convergence_score()
    """

    def __init__(self) -> None:
        self._evolutions: dict[str, PositionEvolution] = {}

    def create_evolution(self, debate_id: str, topic: str) -> PositionEvolution:
        """Create a new position evolution tracker for a debate."""
        evolution = PositionEvolution(debate_id=debate_id, topic=topic)
        self._evolutions[debate_id] = evolution
        return evolution

    def get_evolution(self, debate_id: str) -> Optional[PositionEvolution]:
        """Get position evolution for a debate."""
        return self._evolutions.get(debate_id)

    def record_from_message(
        self,
        debate_id: str,
        agent: str,
        round_number: int,
        content: str,
        sentiment_score: float = 0.5,
    ) -> Optional[PositionPivot]:
        """
        Record position from a debate message.

        Uses sentiment score to infer stance.
        """
        evolution = self._evolutions.get(debate_id)
        if not evolution:
            return None

        # Infer stance from sentiment
        if sentiment_score >= 0.7:
            stance = PositionStance.AGREE
        elif sentiment_score >= 0.55:
            stance = PositionStance.LEAN_AGREE
        elif sentiment_score <= 0.3:
            stance = PositionStance.DISAGREE
        elif sentiment_score <= 0.45:
            stance = PositionStance.LEAN_DISAGREE
        else:
            stance = PositionStance.NEUTRAL

        return evolution.record_position(
            agent=agent,
            round_number=round_number,
            stance=stance,
            confidence=abs(sentiment_score - 0.5) * 2,
            key_argument=content[:200],
        )

    def analyze_debate(self, debate_id: str) -> Optional[dict[str, Any]]:
        """
        Generate comprehensive position analysis for a debate.

        Returns structured analysis suitable for explainability API.
        """
        evolution = self._evolutions.get(debate_id)
        if not evolution:
            return None

        return {
            "debate_id": debate_id,
            "topic": evolution.topic,
            "position_trajectories": {
                agent: [
                    {
                        "round": p.round_number,
                        "stance": p.stance.value,
                        "confidence": p.confidence,
                    }
                    for p in records
                ]
                for agent, records in evolution.positions.items()
            },
            "pivot_analysis": {
                "total_pivots": len(evolution.pivots),
                "reversals": len(evolution.get_reversals()),
                "pivots_by_agent": {
                    agent: len(evolution.get_pivots_for_agent(agent))
                    for agent in evolution.positions.keys()
                },
                "pivot_details": [p.to_dict() for p in evolution.pivots],
            },
            "convergence": {
                "score": evolution.calculate_convergence_score(),
                "interpretation": self._interpret_convergence(evolution.calculate_convergence_score()),
            },
            "stability": evolution.calculate_stability_scores(),
            "influence": evolution.identify_influencers(),
        }

    def _interpret_convergence(self, score: float) -> str:
        """Interpret convergence score as human-readable text."""
        if score >= 0.9:
            return "Strong consensus - all agents converged to similar positions"
        elif score >= 0.7:
            return "Good convergence - most agents reached agreement"
        elif score >= 0.5:
            return "Moderate convergence - partial agreement with some divergence"
        elif score >= 0.3:
            return "Weak convergence - significant disagreement remains"
        else:
            return "No convergence - agents maintain opposing positions"


# Global tracker instance
_position_tracker: Optional[PositionTracker] = None


def get_position_tracker() -> PositionTracker:
    """Get or create the global position tracker."""
    global _position_tracker
    if _position_tracker is None:
        _position_tracker = PositionTracker()
    return _position_tracker
