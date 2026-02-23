"""
NomicCycleAdapter - Bridges Nomic Loop cycle outcomes to the Knowledge Mound.

This adapter enables cross-cycle learning for the self-improvement system:

- Data flow IN: Cycle outcomes (goals, results, failures) are stored as knowledge
- Data flow IN: What worked/failed is tracked for future planning
- Reverse flow: MetaPlanner queries past cycles to avoid repeating failures

The adapter provides:
- Automatic persistence of cycle outcomes after each Nomic Loop cycle
- Retrieval of similar past cycles for informed planning
- Success/failure pattern tracking across cycles
- Learning from past approaches to inform future improvements

"Those who cannot remember the past are condemned to repeat it." - Santayana
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    pass

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.unified.types import (
    KnowledgeSource,
)

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


class CycleStatus(Enum):
    """Status of a Nomic Loop cycle."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some goals succeeded, some failed
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class CurriculumOutcome:
    """Outcome of curriculum-based learning within a cycle.

    Tracks how stepping stones were used to bridge capability gaps.
    """

    curricula_created: int = 0
    stones_attempted: int = 0
    stones_succeeded: int = 0
    skill_gaps: list[str] = field(default_factory=list)
    skills_improved: list[str] = field(default_factory=list)

    # Per-curriculum details (curriculum_id -> results)
    curriculum_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "curricula_created": self.curricula_created,
            "stones_attempted": self.stones_attempted,
            "stones_succeeded": self.stones_succeeded,
            "skill_gaps": self.skill_gaps,
            "skills_improved": self.skills_improved,
            "curriculum_results": self.curriculum_results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumOutcome:
        """Deserialize from dictionary."""
        return cls(
            curricula_created=data.get("curricula_created", 0),
            stones_attempted=data.get("stones_attempted", 0),
            stones_succeeded=data.get("stones_succeeded", 0),
            skill_gaps=data.get("skill_gaps", []),
            skills_improved=data.get("skills_improved", []),
            curriculum_results=data.get("curriculum_results", {}),
        )

    @property
    def stone_success_rate(self) -> float:
        """Calculate stepping stone success rate."""
        if self.stones_attempted == 0:
            return 0.0
        return self.stones_succeeded / self.stones_attempted


@dataclass
class GoalOutcome:
    """Outcome of a single goal within a cycle."""

    goal_id: str
    description: str
    track: str
    status: CycleStatus
    error: str | None = None
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    learnings: list[str] = field(default_factory=list)
    # Per-goal measurement delta (from MetricsCollector)
    metrics_delta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "track": self.track,
            "status": self.status.value,
            "error": self.error,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "learnings": self.learnings,
            "metrics_delta": self.metrics_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalOutcome:
        """Deserialize from dictionary."""
        return cls(
            goal_id=data["goal_id"],
            description=data["description"],
            track=data["track"],
            status=CycleStatus(data["status"]),
            error=data.get("error"),
            files_changed=data.get("files_changed", []),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            learnings=data.get("learnings", []),
            metrics_delta=data.get("metrics_delta", {}),
        )


@dataclass
class NomicCycleOutcome:
    """Complete outcome of a Nomic Loop cycle."""

    cycle_id: str
    objective: str
    status: CycleStatus
    started_at: datetime
    completed_at: datetime
    goal_outcomes: list[GoalOutcome] = field(default_factory=list)

    # Summary metrics
    goals_attempted: int = 0
    goals_succeeded: int = 0
    goals_failed: int = 0
    total_files_changed: int = 0
    total_tests_passed: int = 0
    total_tests_failed: int = 0

    # Learnings
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    agents_used: list[str] = field(default_factory=list)
    tracks_affected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Measurement layer: objective improvement tracking
    metrics_delta: dict[str, Any] = field(default_factory=dict)
    """Baseline vs after metrics comparison from MetricsCollector.
    Keys: tests_passed_delta, tests_failed_delta, test_pass_rate_delta,
    lint_errors_delta, improvement_score, improved, summary."""
    improvement_score: float = 0.0  # 0.0-1.0 overall improvement score
    success_criteria_met: bool | None = None

    # Curriculum learning data (optional)
    curriculum_outcome: CurriculumOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "cycle_id": self.cycle_id,
            "objective": self.objective,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "goal_outcomes": [g.to_dict() for g in self.goal_outcomes],
            "goals_attempted": self.goals_attempted,
            "goals_succeeded": self.goals_succeeded,
            "goals_failed": self.goals_failed,
            "total_files_changed": self.total_files_changed,
            "total_tests_passed": self.total_tests_passed,
            "total_tests_failed": self.total_tests_failed,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "recommendations": self.recommendations,
            "agents_used": self.agents_used,
            "tracks_affected": self.tracks_affected,
            "metadata": self.metadata,
            "metrics_delta": self.metrics_delta,
            "improvement_score": self.improvement_score,
            "success_criteria_met": self.success_criteria_met,
        }
        if self.curriculum_outcome:
            result["curriculum_outcome"] = self.curriculum_outcome.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NomicCycleOutcome:
        """Deserialize from dictionary."""
        curriculum_data = data.get("curriculum_outcome")
        curriculum_outcome = (
            CurriculumOutcome.from_dict(curriculum_data) if curriculum_data else None
        )
        return cls(
            cycle_id=data["cycle_id"],
            objective=data["objective"],
            status=CycleStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            goal_outcomes=[GoalOutcome.from_dict(g) for g in data.get("goal_outcomes", [])],
            goals_attempted=data.get("goals_attempted", 0),
            goals_succeeded=data.get("goals_succeeded", 0),
            goals_failed=data.get("goals_failed", 0),
            total_files_changed=data.get("total_files_changed", 0),
            total_tests_passed=data.get("total_tests_passed", 0),
            total_tests_failed=data.get("total_tests_failed", 0),
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            recommendations=data.get("recommendations", []),
            agents_used=data.get("agents_used", []),
            tracks_affected=data.get("tracks_affected", []),
            metadata=data.get("metadata", {}),
            metrics_delta=data.get("metrics_delta", {}),
            improvement_score=data.get("improvement_score", 0.0),
            success_criteria_met=data.get("success_criteria_met"),
            curriculum_outcome=curriculum_outcome,
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a fraction."""
        if self.goals_attempted == 0:
            return 0.0
        return self.goals_succeeded / self.goals_attempted


@dataclass
class CycleIngestionResult:
    """Result of ingesting a Nomic cycle outcome into Knowledge Mound."""

    cycle_id: str
    items_ingested: int
    learnings_ingested: int
    relationships_created: int
    knowledge_item_ids: list[str]
    errors: list[str]

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.items_ingested > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "items_ingested": self.items_ingested,
            "learnings_ingested": self.learnings_ingested,
            "relationships_created": self.relationships_created,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class SimilarCycle:
    """A past cycle similar to the current planning context."""

    cycle_id: str
    objective: str
    similarity: float  # 0-1, how similar to current objective
    status: CycleStatus
    success_rate: float
    what_worked: list[str]
    what_failed: list[str]
    recommendations: list[str]
    tracks_affected: list[str]
    completed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "objective": self.objective,
            "similarity": self.similarity,
            "status": self.status.value,
            "success_rate": self.success_rate,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "recommendations": self.recommendations,
            "tracks_affected": self.tracks_affected,
            "completed_at": self.completed_at.isoformat(),
        }


class NomicCycleAdapterError(Exception):
    """Base exception for nomic cycle adapter errors."""

    pass


class CycleNotFoundError(NomicCycleAdapterError):
    """Raised when a cycle is not found in the store."""

    pass


class NomicCycleAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Nomic Loop cycle outcomes to the Knowledge Mound.

    Provides methods to:
    - Ingest cycle outcomes as knowledge items
    - Query similar past cycles for informed planning
    - Track what worked and what failed across cycles
    - Generate recommendations based on past learnings

    Usage:
        from aragora.knowledge.mound.adapters import NomicCycleAdapter
        from aragora.knowledge.mound.core import KnowledgeMound

        mound = KnowledgeMound()
        adapter = NomicCycleAdapter(mound)

        # After a Nomic cycle completes
        result = await adapter.ingest_cycle_outcome(outcome, workspace_id="nomic")

        # When planning a new cycle
        similar = await adapter.find_similar_cycles(
            objective="Improve test coverage",
            limit=5
        )
    """

    adapter_name = "nomic_cycle"

    ID_PREFIX = "nomic_"
    LEARNING_PREFIX = "learning_"

    def __init__(
        self,
        mound: Any = None,
        on_event: EventCallback | None = None,
    ):
        """Initialize the adapter.

        Args:
            mound: KnowledgeMound instance (optional, will use singleton if not provided)
            on_event: Callback for emitting events
        """
        super().__init__()
        self._mound = mound
        self._on_event = on_event

    @property
    def mound(self) -> Any:
        """Get the Knowledge Mound instance (lazy initialization)."""
        if self._mound is None:
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                self._mound = get_knowledge_mound(workspace_id="nomic")
            except ImportError:
                logger.warning("Knowledge Mound not available")
        return self._mound

    def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event if callback is registered."""
        if self._on_event:
            try:
                self._on_event(event, data)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                logger.warning("Event emission failed: %s", e)

    def _generate_cycle_id(self, outcome: NomicCycleOutcome) -> str:
        """Generate a unique ID for a cycle outcome."""
        content = f"{outcome.objective}:{outcome.started_at.isoformat()}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"{self.ID_PREFIX}{content_hash}"

    def _generate_learning_id(self, learning: str, cycle_id: str) -> str:
        """Generate a unique ID for a learning."""
        content = f"{cycle_id}:{learning}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"{self.LEARNING_PREFIX}{content_hash}"

    async def ingest_cycle_outcome(
        self,
        outcome: NomicCycleOutcome,
        workspace_id: str = "nomic",
    ) -> CycleIngestionResult:
        """
        Ingest a Nomic cycle outcome into the Knowledge Mound.

        Creates knowledge items for:
        - The cycle summary
        - Individual goal outcomes
        - Learnings (what worked/failed)
        - Recommendations for future cycles

        Args:
            outcome: The cycle outcome to ingest
            workspace_id: Workspace to store in

        Returns:
            CycleIngestionResult with ingestion details
        """
        async with self._resilient_call("ingest_cycle_outcome"):
            return await self._do_ingest_cycle_outcome(outcome, workspace_id)

    async def _do_ingest_cycle_outcome(
        self,
        outcome: NomicCycleOutcome,
        workspace_id: str,
    ) -> CycleIngestionResult:
        """Internal implementation of cycle outcome ingestion."""
        errors: list[str] = []
        knowledge_item_ids: list[str] = []
        items_ingested = 0
        learnings_ingested = 0
        relationships_created = 0

        # Guard: reject non-NomicCycleOutcome objects (e.g. _IndexOutcome)
        if not hasattr(outcome, "status") or not hasattr(outcome, "cycle_id"):
            return CycleIngestionResult(
                cycle_id=getattr(outcome, "cycle_id", "unknown"),
                items_ingested=0,
                learnings_ingested=0,
                relationships_created=0,
                knowledge_item_ids=[],
                errors=["Received incompatible outcome type"],
            )

        mound = self.mound
        if mound is None:
            errors.append("Knowledge Mound not available")
            return CycleIngestionResult(
                cycle_id=outcome.cycle_id,
                items_ingested=0,
                learnings_ingested=0,
                relationships_created=0,
                knowledge_item_ids=[],
                errors=errors,
            )

        try:
            from aragora.knowledge.mound import IngestionRequest

            cycle_id = self._generate_cycle_id(outcome)

            # 1. Ingest the cycle summary
            summary_content = self._build_cycle_summary(outcome)
            summary_request = IngestionRequest(
                content=summary_content,
                workspace_id=workspace_id,
                source_type=KnowledgeSource.INSIGHT,
                document_id=cycle_id,
                confidence=0.9 if outcome.status == CycleStatus.SUCCESS else 0.7,
                topics=["nomic", "self-improvement"] + outcome.tracks_affected,
                metadata={
                    "type": "nomic_cycle",
                    "cycle_id": cycle_id,
                    "objective": outcome.objective,
                    "status": outcome.status.value,
                    "success_rate": outcome.success_rate,
                    "started_at": outcome.started_at.isoformat(),
                    "completed_at": outcome.completed_at.isoformat(),
                    "goals_attempted": outcome.goals_attempted,
                    "goals_succeeded": outcome.goals_succeeded,
                    "goals_failed": outcome.goals_failed,
                    "agents_used": outcome.agents_used,
                    "tracks_affected": outcome.tracks_affected,
                },
            )

            result = await mound.store(summary_request)
            if result and hasattr(result, "item_id"):
                knowledge_item_ids.append(result.item_id)
                items_ingested += 1

            # 2. Ingest learnings (what worked)
            for learning in outcome.what_worked:
                learning_id = self._generate_learning_id(learning, cycle_id)
                learning_request = IngestionRequest(
                    content=f"WHAT WORKED: {learning}",
                    workspace_id=workspace_id,
                    source_type=KnowledgeSource.INSIGHT,
                    document_id=learning_id,
                    confidence=0.85,
                    topics=["nomic", "learning", "success"],
                    metadata={
                        "type": "nomic_learning",
                        "learning_type": "success",
                        "parent_cycle_id": cycle_id,
                        "objective": outcome.objective,
                    },
                )
                result = await mound.store(learning_request)
                if result and not result.deduplicated:
                    learnings_ingested += 1
                    if hasattr(result, "item_id"):
                        knowledge_item_ids.append(result.item_id)

            # 3. Ingest learnings (what failed)
            for learning in outcome.what_failed:
                learning_id = self._generate_learning_id(f"fail:{learning}", cycle_id)
                learning_request = IngestionRequest(
                    content=f"WHAT FAILED: {learning}",
                    workspace_id=workspace_id,
                    source_type=KnowledgeSource.INSIGHT,
                    document_id=learning_id,
                    confidence=0.85,
                    topics=["nomic", "learning", "failure"],
                    metadata={
                        "type": "nomic_learning",
                        "learning_type": "failure",
                        "parent_cycle_id": cycle_id,
                        "objective": outcome.objective,
                    },
                )
                result = await mound.store(learning_request)
                if result and not result.deduplicated:
                    learnings_ingested += 1
                    if hasattr(result, "item_id"):
                        knowledge_item_ids.append(result.item_id)

            # 4. Ingest recommendations
            for rec in outcome.recommendations:
                rec_id = self._generate_learning_id(f"rec:{rec}", cycle_id)
                rec_request = IngestionRequest(
                    content=f"RECOMMENDATION: {rec}",
                    workspace_id=workspace_id,
                    source_type=KnowledgeSource.INSIGHT,
                    document_id=rec_id,
                    confidence=0.8,
                    topics=["nomic", "recommendation"],
                    metadata={
                        "type": "nomic_recommendation",
                        "parent_cycle_id": cycle_id,
                        "objective": outcome.objective,
                    },
                )
                result = await mound.store(rec_request)
                if result and not result.deduplicated:
                    learnings_ingested += 1
                    if hasattr(result, "item_id"):
                        knowledge_item_ids.append(result.item_id)

            # 5. Ingest curriculum outcome if present
            if outcome.curriculum_outcome:
                curriculum_data = outcome.curriculum_outcome
                curriculum_id = self._generate_learning_id(f"curriculum:{cycle_id}", cycle_id)
                curriculum_content = self._build_curriculum_summary(
                    curriculum_data, outcome.objective
                )
                curriculum_request = IngestionRequest(
                    content=curriculum_content,
                    workspace_id=workspace_id,
                    source_type=KnowledgeSource.INSIGHT,
                    document_id=curriculum_id,
                    confidence=0.85,
                    topics=["nomic", "curriculum", "learning"],
                    metadata={
                        "type": "nomic_curriculum",
                        "parent_cycle_id": cycle_id,
                        "objective": outcome.objective,
                        "curricula_created": curriculum_data.curricula_created,
                        "stones_attempted": curriculum_data.stones_attempted,
                        "stones_succeeded": curriculum_data.stones_succeeded,
                        "stone_success_rate": curriculum_data.stone_success_rate,
                        "skill_gaps": curriculum_data.skill_gaps,
                        "skills_improved": curriculum_data.skills_improved,
                    },
                )
                result = await mound.store(curriculum_request)
                if result and not result.deduplicated:
                    learnings_ingested += 1
                    if hasattr(result, "item_id"):
                        knowledge_item_ids.append(result.item_id)

                # Ingest individual skill gaps as searchable items
                for gap in curriculum_data.skill_gaps:
                    gap_id = self._generate_learning_id(f"skill_gap:{gap}", cycle_id)
                    gap_request = IngestionRequest(
                        content=f"SKILL GAP: {gap}",
                        workspace_id=workspace_id,
                        source_type=KnowledgeSource.INSIGHT,
                        document_id=gap_id,
                        confidence=0.8,
                        topics=["nomic", "curriculum", "skill_gap"],
                        metadata={
                            "type": "nomic_skill_gap",
                            "parent_cycle_id": cycle_id,
                            "objective": outcome.objective,
                        },
                    )
                    result = await mound.store(gap_request)
                    if result and not result.deduplicated:
                        learnings_ingested += 1
                        if hasattr(result, "item_id"):
                            knowledge_item_ids.append(result.item_id)

                # Ingest improved skills
                for skill in curriculum_data.skills_improved:
                    skill_id = self._generate_learning_id(f"skill_improved:{skill}", cycle_id)
                    skill_request = IngestionRequest(
                        content=f"SKILL IMPROVED: {skill}",
                        workspace_id=workspace_id,
                        source_type=KnowledgeSource.INSIGHT,
                        document_id=skill_id,
                        confidence=0.85,
                        topics=["nomic", "curriculum", "skill_improvement"],
                        metadata={
                            "type": "nomic_skill_improvement",
                            "parent_cycle_id": cycle_id,
                            "objective": outcome.objective,
                        },
                    )
                    result = await mound.store(skill_request)
                    if result and not result.deduplicated:
                        learnings_ingested += 1
                        if hasattr(result, "item_id"):
                            knowledge_item_ids.append(result.item_id)

            self._emit_event(
                "nomic_cycle_ingested",
                {
                    "cycle_id": cycle_id,
                    "items_ingested": items_ingested,
                    "learnings_ingested": learnings_ingested,
                    "status": outcome.status.value,
                },
            )

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            error_msg = f"Failed to ingest cycle outcome: {e}"
            errors.append(error_msg)
            logger.exception(error_msg)

        return CycleIngestionResult(
            cycle_id=outcome.cycle_id,
            items_ingested=items_ingested,
            learnings_ingested=learnings_ingested,
            relationships_created=relationships_created,
            knowledge_item_ids=knowledge_item_ids,
            errors=errors,
        )

    def _build_cycle_summary(self, outcome: NomicCycleOutcome) -> str:
        """Build a text summary of the cycle for knowledge storage."""
        lines = [
            f"NOMIC CYCLE: {outcome.objective}",
            f"Status: {outcome.status.value.upper()}",
            f"Success Rate: {outcome.success_rate:.0%}",
            f"Goals: {outcome.goals_succeeded}/{outcome.goals_attempted} succeeded",
            f"Files Changed: {outcome.total_files_changed}",
            f"Tests: {outcome.total_tests_passed} passed, {outcome.total_tests_failed} failed",
            "",
        ]

        if outcome.what_worked:
            lines.append("WHAT WORKED:")
            for item in outcome.what_worked[:5]:
                lines.append(f"  - {item}")
            lines.append("")

        if outcome.what_failed:
            lines.append("WHAT FAILED:")
            for item in outcome.what_failed[:5]:
                lines.append(f"  - {item}")
            lines.append("")

        if outcome.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in outcome.recommendations[:3]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def _build_curriculum_summary(
        self,
        curriculum: CurriculumOutcome,
        objective: str,
    ) -> str:
        """Build a text summary of curriculum learning for knowledge storage."""
        lines = [
            f"CURRICULUM LEARNING: {objective}",
            f"Curricula Created: {curriculum.curricula_created}",
            f"Stepping Stones: {curriculum.stones_succeeded}/{curriculum.stones_attempted} succeeded ({curriculum.stone_success_rate:.0%})",
            "",
        ]

        if curriculum.skill_gaps:
            lines.append("SKILL GAPS IDENTIFIED:")
            for gap in curriculum.skill_gaps[:5]:
                lines.append(f"  - {gap}")
            lines.append("")

        if curriculum.skills_improved:
            lines.append("SKILLS IMPROVED:")
            for skill in curriculum.skills_improved[:5]:
                lines.append(f"  - {skill}")
            lines.append("")

        return "\n".join(lines)

    async def find_similar_curricula(
        self,
        task_description: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        workspace_id: str = "nomic",
    ) -> list[dict[str, Any]]:
        """
        Find past curricula similar to the given task description.

        Used by MetaPlanner to find relevant past learning experiences
        when planning how to approach a new task.

        Args:
            task_description: Description of the current task
            limit: Maximum number of curricula to return
            min_similarity: Minimum similarity score (0-1)
            workspace_id: Workspace to search in

        Returns:
            List of curriculum data dicts with similarity scores
        """
        mound = self.mound
        if mound is None:
            return []

        similar_curricula: list[dict[str, Any]] = []

        try:
            query_result = await mound.query(
                query=task_description,
                workspace_id=workspace_id,
                limit=limit * 2,
            )

            for result in query_result.items:
                if not hasattr(result, "metadata"):
                    continue

                metadata = result.metadata
                if metadata.get("type") != "nomic_curriculum":
                    continue

                similarity = getattr(result, "score", 0.5)
                if similarity < min_similarity:
                    continue

                similar_curricula.append(
                    {
                        "objective": metadata.get("objective", ""),
                        "similarity": similarity,
                        "curricula_created": metadata.get("curricula_created", 0),
                        "stones_attempted": metadata.get("stones_attempted", 0),
                        "stones_succeeded": metadata.get("stones_succeeded", 0),
                        "stone_success_rate": metadata.get("stone_success_rate", 0.0),
                        "skill_gaps": metadata.get("skill_gaps", []),
                        "skills_improved": metadata.get("skills_improved", []),
                        "cycle_id": metadata.get("parent_cycle_id", ""),
                    }
                )

            # Sort by similarity
            similar_curricula.sort(key=lambda c: c["similarity"], reverse=True)
            return similar_curricula[:limit]

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find similar curricula: %s", e)
            return []

    async def find_high_roi_goal_types(
        self,
        limit: int = 5,
        workspace_id: str = "nomic",
    ) -> list[dict[str, Any]]:
        """Find goal types that historically produce the best improvements.

        Queries past cycle outcomes, groups by objective keywords, and ranks
        by average improvement_score. This helps MetaPlanner focus on goal
        types that have historically delivered measurable improvements.

        Args:
            limit: Maximum number of patterns to return
            workspace_id: Workspace to search in

        Returns:
            List of dicts: [{pattern, avg_improvement_score, cycle_count, example_objectives}]
        """
        mound = self.mound
        if mound is None:
            return []

        try:
            query_result = await mound.query(
                query="nomic cycle outcome improvement",
                workspace_id=workspace_id,
                limit=50,
            )

            # Group by objective keywords to find patterns
            pattern_data: dict[str, dict[str, Any]] = {}

            for result in query_result.items:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "nomic_cycle":
                    continue

                objective = metadata.get("objective", "")
                improvement = metadata.get("improvement_score", 0.0)
                if not isinstance(improvement, (int, float)):
                    improvement = 0.0

                # Extract keywords from objective for pattern matching
                words = set(objective.lower().split())
                # Use the first 3 significant words as a pattern key
                significant = [
                    w for w in words
                    if len(w) > 3 and w not in {"the", "and", "for", "with", "from", "that", "this"}
                ]
                if not significant:
                    continue

                pattern_key = " ".join(sorted(significant[:3]))

                if pattern_key not in pattern_data:
                    pattern_data[pattern_key] = {
                        "pattern": pattern_key,
                        "total_score": 0.0,
                        "cycle_count": 0,
                        "example_objectives": [],
                    }

                entry = pattern_data[pattern_key]
                entry["total_score"] += improvement
                entry["cycle_count"] += 1
                if len(entry["example_objectives"]) < 3:
                    entry["example_objectives"].append(objective[:100])

            # Compute averages and rank
            ranked: list[dict[str, Any]] = []
            for entry in pattern_data.values():
                if entry["cycle_count"] == 0:
                    continue
                avg = entry["total_score"] / entry["cycle_count"]
                ranked.append({
                    "pattern": entry["pattern"],
                    "avg_improvement_score": round(avg, 3),
                    "cycle_count": entry["cycle_count"],
                    "example_objectives": entry["example_objectives"],
                })

            ranked.sort(key=lambda x: x["avg_improvement_score"], reverse=True)
            return ranked[:limit]

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find high-ROI goal types: %s", e)
            return []

    async def find_recurring_failures(
        self,
        min_occurrences: int = 2,
        limit: int = 10,
        workspace_id: str = "nomic",
    ) -> list[dict[str, Any]]:
        """Detect failure patterns that recur across multiple cycles.

        Queries past cycle failure learnings and groups by similarity to
        identify recurring issues. This helps MetaPlanner avoid known
        failure patterns.

        Args:
            min_occurrences: Minimum number of times a pattern must appear
            limit: Maximum number of failure patterns to return
            workspace_id: Workspace to search in

        Returns:
            List of dicts: [{pattern, occurrences, affected_tracks, example_errors, last_seen}]
        """
        mound = self.mound
        if mound is None:
            return []

        try:
            query_result = await mound.query(
                query="nomic learning failure what failed",
                workspace_id=workspace_id,
                limit=100,
            )

            # Group by failure content similarity
            failure_groups: dict[str, dict[str, Any]] = {}

            for result in query_result.items:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "nomic_learning":
                    continue
                if metadata.get("learning_type") != "failure":
                    continue

                content = getattr(result, "content", "")
                if content.startswith("WHAT FAILED: "):
                    content = content[13:]

                # Normalize content for grouping
                normalized = content.lower().strip()[:80]
                if not normalized:
                    continue

                # Use first 50 chars as a grouping key
                group_key = normalized[:50]

                if group_key not in failure_groups:
                    failure_groups[group_key] = {
                        "pattern": content[:100],
                        "occurrences": 0,
                        "affected_tracks": set(),
                        "example_errors": [],
                        "last_seen": "",
                    }

                entry = failure_groups[group_key]
                entry["occurrences"] += 1

                # Extract track info from parent cycle
                cycle_id = metadata.get("parent_cycle_id", "")
                if cycle_id:
                    # Track names are typically in the objective or metadata
                    objective = metadata.get("objective", "")
                    for track_name in ["sme", "developer", "qa", "core", "security", "self_hosted"]:
                        if track_name in objective.lower():
                            entry["affected_tracks"].add(track_name)

                if len(entry["example_errors"]) < 3:
                    entry["example_errors"].append(content[:150])

            # Filter by min_occurrences and convert sets to lists
            recurring: list[dict[str, Any]] = []
            for entry in failure_groups.values():
                if entry["occurrences"] >= min_occurrences:
                    recurring.append({
                        "pattern": entry["pattern"],
                        "occurrences": entry["occurrences"],
                        "affected_tracks": sorted(entry["affected_tracks"]),
                        "example_errors": entry["example_errors"],
                    })

            recurring.sort(key=lambda x: x["occurrences"], reverse=True)
            return recurring[:limit]

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find recurring failures: %s", e)
            return []

    async def find_similar_cycles(
        self,
        objective: str,
        tracks: list[str] | None = None,
        limit: int = 5,
        min_similarity: float = 0.3,
        workspace_id: str = "nomic",
    ) -> list[SimilarCycle]:
        """
        Find past cycles similar to the given objective.

        Uses semantic search to find cycles with similar objectives,
        optionally filtered by tracks.

        Args:
            objective: The current planning objective
            tracks: Optional list of tracks to filter by
            limit: Maximum number of similar cycles to return
            min_similarity: Minimum similarity score (0-1)
            workspace_id: Workspace to search in

        Returns:
            List of SimilarCycle objects ordered by relevance
        """
        async with self._resilient_call("find_similar_cycles"):
            return await self._do_find_similar_cycles(
                objective, tracks, limit, min_similarity, workspace_id
            )

    async def _do_find_similar_cycles(
        self,
        objective: str,
        tracks: list[str] | None,
        limit: int,
        min_similarity: float,
        workspace_id: str,
    ) -> list[SimilarCycle]:
        """Internal implementation of finding similar cycles."""
        mound = self.mound
        if mound is None:
            return []

        similar_cycles: list[SimilarCycle] = []

        try:
            # Search for cycle summaries with similar objectives
            query_result = await mound.query(
                query=objective,
                workspace_id=workspace_id,
                limit=limit * 2,  # Fetch more to filter
            )

            for result in query_result.items:
                if not hasattr(result, "metadata"):
                    continue

                metadata = result.metadata
                if metadata.get("type") != "nomic_cycle":
                    continue

                # Calculate similarity (use the search score if available)
                similarity = getattr(result, "score", 0.5)
                if similarity < min_similarity:
                    continue

                # Filter by tracks if specified
                cycle_tracks = metadata.get("tracks_affected", [])
                if tracks and not any(t in cycle_tracks for t in tracks):
                    continue

                # Fetch related learnings
                cycle_id = metadata.get("cycle_id", "")
                what_worked = await self._get_learnings(cycle_id, "success", workspace_id)
                what_failed = await self._get_learnings(cycle_id, "failure", workspace_id)
                recommendations = await self._get_recommendations(cycle_id, workspace_id)

                similar_cycles.append(
                    SimilarCycle(
                        cycle_id=cycle_id,
                        objective=metadata.get("objective", ""),
                        similarity=similarity,
                        status=CycleStatus(metadata.get("status", "success")),
                        success_rate=metadata.get("success_rate", 0.0),
                        what_worked=what_worked,
                        what_failed=what_failed,
                        recommendations=recommendations,
                        tracks_affected=cycle_tracks,
                        completed_at=datetime.fromisoformat(
                            metadata.get("completed_at", datetime.now(timezone.utc).isoformat())
                        ),
                    )
                )

            # Sort by similarity and limit
            similar_cycles.sort(key=lambda c: c.similarity, reverse=True)
            return similar_cycles[:limit]

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find similar cycles: %s", e)
            return []

    async def _get_learnings(
        self,
        cycle_id: str,
        learning_type: str,
        workspace_id: str,
    ) -> list[str]:
        """Get learnings of a specific type for a cycle."""
        mound = self.mound
        if mound is None:
            return []

        try:
            query_result = await mound.query(
                query=f"nomic learning {learning_type}",
                workspace_id=workspace_id,
                limit=10,
            )

            learnings = []
            for result in query_result.items:
                metadata = getattr(result, "metadata", {})
                # Filter by learning type and parent cycle
                if metadata.get("type") != "nomic_learning":
                    continue
                if metadata.get("learning_type") != learning_type:
                    continue
                if metadata.get("parent_cycle_id") != cycle_id:
                    continue
                content = getattr(result, "content", "")
                # Strip the prefix
                if content.startswith("WHAT WORKED: "):
                    content = content[13:]
                elif content.startswith("WHAT FAILED: "):
                    content = content[13:]
                learnings.append(content)

            return learnings

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to get learnings: %s", e)
            return []

    async def _get_recommendations(
        self,
        cycle_id: str,
        workspace_id: str,
    ) -> list[str]:
        """Get recommendations for a cycle."""
        mound = self.mound
        if mound is None:
            return []

        try:
            query_result = await mound.query(
                query="nomic recommendation",
                workspace_id=workspace_id,
                limit=5,
            )

            recommendations = []
            for result in query_result.items:
                metadata = getattr(result, "metadata", {})
                # Filter by recommendation type and parent cycle
                if metadata.get("type") != "nomic_recommendation":
                    continue
                if metadata.get("parent_cycle_id") != cycle_id:
                    continue
                content = getattr(result, "content", "")
                # Strip the prefix
                if content.startswith("RECOMMENDATION: "):
                    content = content[16:]
                recommendations.append(content)

            return recommendations

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to get recommendations: %s", e)
            return []

    async def get_cycle_outcome(
        self,
        cycle_id: str,
        workspace_id: str = "nomic",
    ) -> NomicCycleOutcome | None:
        """
        Retrieve a specific cycle outcome by ID.

        Args:
            cycle_id: The cycle ID to retrieve
            workspace_id: Workspace to search in

        Returns:
            NomicCycleOutcome if found, None otherwise
        """
        mound = self.mound
        if mound is None:
            return None

        try:
            query_result = await mound.query(
                query=cycle_id,
                workspace_id=workspace_id,
                limit=10,
            )

            # Find the matching cycle by metadata
            result = None
            for item in query_result.items:
                item_meta = getattr(item, "metadata", {})
                if item_meta.get("type") == "nomic_cycle" and item_meta.get("cycle_id") == cycle_id:
                    result = item
                    break

            if result is None:
                return None

            metadata = getattr(result, "metadata", {})

            return NomicCycleOutcome(
                cycle_id=metadata.get("cycle_id", cycle_id),
                objective=metadata.get("objective", ""),
                status=CycleStatus(metadata.get("status", "success")),
                started_at=datetime.fromisoformat(
                    metadata.get("started_at", datetime.now(timezone.utc).isoformat())
                ),
                completed_at=datetime.fromisoformat(
                    metadata.get("completed_at", datetime.now(timezone.utc).isoformat())
                ),
                goals_attempted=metadata.get("goals_attempted", 0),
                goals_succeeded=metadata.get("goals_succeeded", 0),
                goals_failed=metadata.get("goals_failed", 0),
                agents_used=metadata.get("agents_used", []),
                tracks_affected=metadata.get("tracks_affected", []),
            )

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to get cycle outcome: %s", e)
            return None


# Convenience function
def get_nomic_cycle_adapter(workspace_id: str = "nomic") -> NomicCycleAdapter:
    """Get or create a NomicCycleAdapter instance."""
    return NomicCycleAdapter()


__all__ = [
    "NomicCycleAdapter",
    "NomicCycleOutcome",
    "GoalOutcome",
    "CurriculumOutcome",
    "CycleStatus",
    "CycleIngestionResult",
    "SimilarCycle",
    "NomicCycleAdapterError",
    "CycleNotFoundError",
    "get_nomic_cycle_adapter",
]
