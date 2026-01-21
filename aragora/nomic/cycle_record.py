"""Nomic Cycle Record - captures learning from complete cycles.

Records comprehensive data about each Nomic Loop cycle for:
- Cross-cycle learning (what worked, what didn't)
- Pattern reinforcement (confirming successful approaches)
- Trajectory tracking (agent performance over time)
- Context injection (informing future cycles)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContribution:
    """Contribution metrics for a single agent in a cycle."""

    agent_name: str
    proposals_made: int = 0
    proposals_accepted: int = 0
    critiques_given: int = 0
    critiques_valuable: int = 0
    votes_cast: int = 0
    consensus_aligned: int = 0
    quality_score: float = 0.0  # 0.0-1.0


@dataclass
class SurpriseEvent:
    """An unexpected outcome during the cycle."""

    phase: str
    description: str
    expected: str
    actual: str
    impact: str = "low"  # low, medium, high
    timestamp: Optional[float] = None


@dataclass
class PatternReinforcement:
    """A pattern that was confirmed during the cycle."""

    pattern_type: str  # e.g., "refactor", "bugfix", "security"
    description: str
    success: bool
    confidence: float = 0.5  # 0.0-1.0
    context: str = ""


@dataclass
class NomicCycleRecord:
    """Comprehensive record of a Nomic Loop cycle.

    Captures everything needed for cross-cycle learning:
    - Cycle metadata (id, timing, phases)
    - Debate outcomes (topics, consensus, dissent)
    - Agent contributions and performance
    - Code changes and test results
    - Surprises and pattern reinforcements

    Example:
        record = NomicCycleRecord(
            cycle_id="abc123",
            started_at=time.time(),
        )
        record.topics_debated.append("Add rate limiting")
        record.agent_contributions["claude"] = AgentContribution(
            agent_name="claude",
            proposals_made=3,
            proposals_accepted=2,
        )
        record.completed_at = time.time()
    """

    # Identification
    cycle_id: str
    started_at: float

    # Timing
    completed_at: Optional[float] = None
    phases_completed: List[str] = field(default_factory=list)
    phases_skipped: List[str] = field(default_factory=list)
    duration_seconds: Optional[float] = None

    # Debate outcomes
    topics_debated: List[str] = field(default_factory=list)
    consensus_reached: List[str] = field(default_factory=list)
    consensus_failed: List[str] = field(default_factory=list)
    dissent_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Agent performance
    agent_contributions: Dict[str, AgentContribution] = field(default_factory=dict)
    agent_reliability: Dict[str, float] = field(default_factory=dict)

    # Code changes
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0

    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    test_coverage: Optional[float] = None

    # Learning signals
    surprise_events: List[SurpriseEvent] = field(default_factory=list)
    pattern_reinforcements: List[PatternReinforcement] = field(default_factory=list)

    # Outcome
    success: bool = False
    error_message: Optional[str] = None
    rollback_performed: bool = False

    # Metadata
    commit_sha: Optional[str] = None
    branch_name: Optional[str] = None
    triggering_context: Optional[str] = None

    def mark_complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark the cycle as complete.

        Args:
            success: Whether the cycle succeeded
            error: Optional error message if failed
        """
        import time

        self.completed_at = time.time()
        self.success = success
        self.error_message = error
        if self.started_at:
            self.duration_seconds = self.completed_at - self.started_at

    def add_agent_contribution(
        self,
        agent_name: str,
        proposals_made: int = 0,
        proposals_accepted: int = 0,
        critiques_given: int = 0,
        critiques_valuable: int = 0,
    ) -> None:
        """Add or update agent contribution metrics."""
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = AgentContribution(agent_name=agent_name)

        contrib = self.agent_contributions[agent_name]
        contrib.proposals_made += proposals_made
        contrib.proposals_accepted += proposals_accepted
        contrib.critiques_given += critiques_given
        contrib.critiques_valuable += critiques_valuable

    def add_surprise(
        self,
        phase: str,
        description: str,
        expected: str,
        actual: str,
        impact: str = "low",
    ) -> None:
        """Record a surprising outcome."""
        import time

        self.surprise_events.append(
            SurpriseEvent(
                phase=phase,
                description=description,
                expected=expected,
                actual=actual,
                impact=impact,
                timestamp=time.time(),
            )
        )

    def add_pattern_reinforcement(
        self,
        pattern_type: str,
        description: str,
        success: bool,
        confidence: float = 0.5,
    ) -> None:
        """Record a pattern that was confirmed or refuted."""
        self.pattern_reinforcements.append(
            PatternReinforcement(
                pattern_type=pattern_type,
                description=description,
                success=success,
                confidence=confidence,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "phases_completed": self.phases_completed,
            "phases_skipped": self.phases_skipped,
            "topics_debated": self.topics_debated,
            "consensus_reached": self.consensus_reached,
            "consensus_failed": self.consensus_failed,
            "dissent_patterns": self.dissent_patterns,
            "agent_contributions": {
                k: {
                    "agent_name": v.agent_name,
                    "proposals_made": v.proposals_made,
                    "proposals_accepted": v.proposals_accepted,
                    "critiques_given": v.critiques_given,
                    "critiques_valuable": v.critiques_valuable,
                    "votes_cast": v.votes_cast,
                    "consensus_aligned": v.consensus_aligned,
                    "quality_score": v.quality_score,
                }
                for k, v in self.agent_contributions.items()
            },
            "agent_reliability": self.agent_reliability,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "test_coverage": self.test_coverage,
            "surprise_events": [
                {
                    "phase": s.phase,
                    "description": s.description,
                    "expected": s.expected,
                    "actual": s.actual,
                    "impact": s.impact,
                    "timestamp": s.timestamp,
                }
                for s in self.surprise_events
            ],
            "pattern_reinforcements": [
                {
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "success": p.success,
                    "confidence": p.confidence,
                    "context": p.context,
                }
                for p in self.pattern_reinforcements
            ],
            "success": self.success,
            "error_message": self.error_message,
            "rollback_performed": self.rollback_performed,
            "commit_sha": self.commit_sha,
            "branch_name": self.branch_name,
            "triggering_context": self.triggering_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NomicCycleRecord":
        """Create from dictionary."""
        record = cls(
            cycle_id=data["cycle_id"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds"),
            phases_completed=data.get("phases_completed", []),
            phases_skipped=data.get("phases_skipped", []),
            topics_debated=data.get("topics_debated", []),
            consensus_reached=data.get("consensus_reached", []),
            consensus_failed=data.get("consensus_failed", []),
            dissent_patterns=data.get("dissent_patterns", []),
            agent_reliability=data.get("agent_reliability", {}),
            files_modified=data.get("files_modified", []),
            files_created=data.get("files_created", []),
            lines_added=data.get("lines_added", 0),
            lines_removed=data.get("lines_removed", 0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            tests_skipped=data.get("tests_skipped", 0),
            test_coverage=data.get("test_coverage"),
            success=data.get("success", False),
            error_message=data.get("error_message"),
            rollback_performed=data.get("rollback_performed", False),
            commit_sha=data.get("commit_sha"),
            branch_name=data.get("branch_name"),
            triggering_context=data.get("triggering_context"),
        )

        # Restore agent contributions
        for name, contrib_data in data.get("agent_contributions", {}).items():
            record.agent_contributions[name] = AgentContribution(
                agent_name=contrib_data["agent_name"],
                proposals_made=contrib_data.get("proposals_made", 0),
                proposals_accepted=contrib_data.get("proposals_accepted", 0),
                critiques_given=contrib_data.get("critiques_given", 0),
                critiques_valuable=contrib_data.get("critiques_valuable", 0),
                votes_cast=contrib_data.get("votes_cast", 0),
                consensus_aligned=contrib_data.get("consensus_aligned", 0),
                quality_score=contrib_data.get("quality_score", 0.0),
            )

        # Restore surprise events
        for s in data.get("surprise_events", []):
            record.surprise_events.append(
                SurpriseEvent(
                    phase=s["phase"],
                    description=s["description"],
                    expected=s["expected"],
                    actual=s["actual"],
                    impact=s.get("impact", "low"),
                    timestamp=s.get("timestamp"),
                )
            )

        # Restore pattern reinforcements
        for p in data.get("pattern_reinforcements", []):
            record.pattern_reinforcements.append(
                PatternReinforcement(
                    pattern_type=p["pattern_type"],
                    description=p["description"],
                    success=p["success"],
                    confidence=p.get("confidence", 0.5),
                    context=p.get("context", ""),
                )
            )

        return record
