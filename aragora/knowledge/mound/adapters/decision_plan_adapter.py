"""
DecisionPlanAdapter - Bridges Decision Plans to the Knowledge Mound.

This adapter enables knowledge persistence from decision plan lifecycle:

- Data flow IN: Completed plans are stored as knowledge items
- Data flow IN: Plan outcomes with lessons learned are preserved
- Data flow IN: Risk assessments and verification results
- Reverse flow: KM can retrieve past decisions for similar queries

The adapter provides:
- Automatic ingestion of plan outcomes to knowledge items
- Historical decision retrieval for risk enrichment
- Cross-plan learning via lessons learned
- Audit trail integration

"Every implemented decision strengthens institutional memory."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.pipeline.decision_plan import DecisionPlan, PlanOutcome

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
    RelationshipType,
)

# Decision plans use DEBATE source since they derive from debates
PLAN_SOURCE = KnowledgeSource.DEBATE

logger = logging.getLogger(__name__)

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


class DecisionPlanAdapterError(Exception):
    """Base exception for decision plan adapter errors."""

    pass


@dataclass
class PlanIngestionResult:
    """Result of ingesting a decision plan into Knowledge Mound."""

    plan_id: str
    items_ingested: int = 0
    lessons_ingested: int = 0
    relationships_created: int = 0
    knowledge_item_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.items_ingested > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "items_ingested": self.items_ingested,
            "lessons_ingested": self.lessons_ingested,
            "relationships_created": self.relationships_created,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


class DecisionPlanAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Decision Plans to the Knowledge Mound.

    Provides methods to:
    - Ingest completed plans as knowledge items
    - Store lessons learned from outcomes
    - Create relationships between plans and debates
    - Retrieve past decisions for historical enrichment
    """

    adapter_name = "decision_plan"

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        event_callback: EventCallback | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            knowledge_mound: Knowledge Mound instance (optional, uses global if not provided)
            event_callback: Optional callback for emitting events
        """
        super().__init__(**kwargs)
        self._km = knowledge_mound
        self._event_callback = event_callback

    @property
    def knowledge_mound(self) -> Any:
        """Get the Knowledge Mound instance, initializing if needed."""
        if self._km is None:
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                self._km = get_knowledge_mound()
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                logger.debug("Could not get knowledge mound: %s", e)
        return self._km

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event via the callback if configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                logger.debug("Event callback failed: %s", e)

    async def ingest_plan_outcome(
        self,
        plan: DecisionPlan,
        outcome: PlanOutcome,
    ) -> PlanIngestionResult:
        """Ingest a completed plan and its outcome into Knowledge Mound.

        Args:
            plan: The completed DecisionPlan
            outcome: The execution outcome

        Returns:
            PlanIngestionResult with ingestion statistics
        """
        result = PlanIngestionResult(plan_id=plan.id)
        item_ids: list[str] = []
        errors: list[str] = []

        km = self.knowledge_mound
        if km is None:
            errors.append("Knowledge Mound not available")
            result.errors = errors
            return result

        try:
            # Create main plan outcome item
            main_item = self._create_plan_outcome_item(plan, outcome)
            stored_id = await self._store_item(km, main_item)
            if stored_id:
                item_ids.append(stored_id)
                result.items_ingested += 1

            # Create items for each lesson learned
            for i, lesson in enumerate(outcome.lessons):
                lesson_item = self._create_lesson_item(plan, outcome, lesson, i)
                lesson_id = await self._store_item(km, lesson_item)
                if lesson_id:
                    item_ids.append(lesson_id)
                    result.lessons_ingested += 1

                    # Create relationship between main item and lesson
                    if stored_id:
                        await self._create_relationship(
                            km, stored_id, lesson_id, RelationshipType.DERIVED_FROM
                        )
                        result.relationships_created += 1

            # Link to debate if we have a debate item
            if plan.debate_id and stored_id:
                debate_item_id = await self._find_debate_item(km, plan.debate_id)
                if debate_item_id:
                    await self._create_relationship(
                        km, debate_item_id, stored_id, RelationshipType.DERIVED_FROM
                    )
                    result.relationships_created += 1

            result.knowledge_item_ids = item_ids

            # Emit event
            self._emit_event(
                "plan_ingested",
                {
                    "plan_id": plan.id,
                    "debate_id": plan.debate_id,
                    "success": outcome.success,
                    "items_ingested": result.items_ingested,
                },
            )

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Plan ingestion failed: %s", e)
            errors.append("Plan ingestion failed")

        result.errors = errors
        return result

    def _create_plan_outcome_item(
        self,
        plan: DecisionPlan,
        outcome: PlanOutcome,
    ) -> KnowledgeItem:
        """Create a KnowledgeItem from plan outcome."""
        # Build content summary
        status = "SUCCESS" if outcome.success else "FAILURE"
        content = f"""[Decision Plan Outcome: {status}]
Task: {plan.task}
Plan ID: {plan.id}
Debate ID: {plan.debate_id}

Execution Results:
- Tasks: {outcome.tasks_completed}/{outcome.tasks_total} completed
- Verification: {outcome.verification_passed}/{outcome.verification_total} passed
- Duration: {outcome.duration_seconds:.1f}s
- Cost: ${outcome.total_cost_usd:.4f}
"""

        if outcome.error:
            content += f"\nError: {outcome.error}"

        if outcome.lessons:
            content += "\n\nLessons Learned:\n"
            for lesson in outcome.lessons:
                content += f"- {lesson}\n"

        # Calculate confidence from outcome metrics
        if outcome.tasks_total > 0:
            task_ratio = outcome.tasks_completed / outcome.tasks_total
        else:
            task_ratio = 1.0 if outcome.success else 0.0

        if outcome.verification_total > 0:
            verify_ratio = outcome.verification_passed / outcome.verification_total
        else:
            verify_ratio = 1.0 if outcome.success else 0.0

        confidence = (task_ratio + verify_ratio) / 2

        # Map confidence to level
        if confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW

        now = datetime.now(timezone.utc)
        return KnowledgeItem(
            id=f"plan-outcome-{plan.id}",
            content=content,
            source=PLAN_SOURCE,
            source_id=plan.id,
            confidence=confidence_level,
            created_at=now,
            updated_at=now,
            metadata={
                "plan_id": plan.id,
                "debate_id": plan.debate_id,
                "task": plan.task[:500],
                "success": outcome.success,
                "tasks_completed": outcome.tasks_completed,
                "tasks_total": outcome.tasks_total,
                "verification_passed": outcome.verification_passed,
                "verification_total": outcome.verification_total,
                "total_cost_usd": outcome.total_cost_usd,
                "duration_seconds": outcome.duration_seconds,
                "receipt_id": outcome.receipt_id,
                "review_passed": outcome.review_passed,
                "review_model": outcome.review.get("model")
                if isinstance(outcome.review, dict)
                else None,
                "review_duration_seconds": (
                    outcome.review.get("duration_seconds")
                    if isinstance(outcome.review, dict)
                    else None
                ),
                "ingested_at": now.isoformat(),
                "tags": [
                    "decision_plan",
                    "outcome",
                    status.lower(),
                    f"plan:{plan.id}",
                    f"debate:{plan.debate_id}",
                ],
            },
        )

    def _create_lesson_item(
        self,
        plan: DecisionPlan,
        outcome: PlanOutcome,
        lesson: str,
        index: int,
    ) -> KnowledgeItem:
        """Create a KnowledgeItem for a lesson learned."""
        content = f"""[Lesson Learned]
From Plan: {plan.id}
Task: {plan.task[:200]}

{lesson}
"""

        now = datetime.now(timezone.utc)
        return KnowledgeItem(
            id=f"plan-lesson-{plan.id}-{index}",
            content=content,
            source=PLAN_SOURCE,
            source_id=f"{plan.id}-lesson-{index}",
            confidence=ConfidenceLevel.MEDIUM,
            created_at=now,
            updated_at=now,
            metadata={
                "plan_id": plan.id,
                "debate_id": plan.debate_id,
                "lesson_index": index,
                "outcome_success": outcome.success,
                "ingested_at": now.isoformat(),
                "tags": [
                    "lesson_learned",
                    "decision_plan",
                    f"plan:{plan.id}",
                ],
            },
        )

    async def _store_item(self, km: Any, item: KnowledgeItem) -> str | None:
        """Store an item in Knowledge Mound."""
        try:
            if hasattr(km, "add_item"):
                result = await km.add_item(item)
                return getattr(result, "id", None) or str(result)
            elif hasattr(km, "store"):
                return await km.store(item)
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to store item: %s", e)
        return None

    async def _create_relationship(
        self,
        km: Any,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
    ) -> bool:
        """Create a relationship between two items."""
        try:
            if hasattr(km, "add_relationship"):
                await km.add_relationship(source_id, target_id, rel_type)
                return True
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to create relationship: %s", e)
        return False

    async def _find_debate_item(self, km: Any, debate_id: str) -> str | None:
        """Find the KM item ID for a debate."""
        try:
            if hasattr(km, "search"):
                results = await km.search(
                    query=f"debate:{debate_id}",
                    limit=1,
                    filters={"tags": ["debate"]},
                )
                if results:
                    return getattr(results[0], "id", None)
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Failed to find debate item: %s", e)
        return None

    async def query_similar_plans(
        self,
        task: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Query for historically similar decision plans.

        Args:
            task: The task description to find similar plans for
            limit: Maximum number of results

        Returns:
            List of similar plan summaries
        """
        km = self.knowledge_mound
        if km is None:
            return []

        try:
            if hasattr(km, "search"):
                results = await km.search(
                    query=task,
                    limit=limit,
                    filters={"tags": ["decision_plan", "outcome"]},
                )

                return [
                    {
                        "plan_id": getattr(r, "metadata", {}).get("plan_id"),
                        "task": getattr(r, "metadata", {}).get("task"),
                        "success": getattr(r, "metadata", {}).get("success"),
                        "content": getattr(r, "content", "")[:500],
                        "similarity": getattr(r, "score", 0.0),
                    }
                    for r in results
                    if hasattr(r, "metadata")
                ]
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Similar plan query failed: %s", e)

        return []

    async def get_lessons_for_domain(
        self,
        domain: str,
        limit: int = 10,
    ) -> list[str]:
        """Get lessons learned from past plans in a domain.

        Args:
            domain: Domain keyword to filter by
            limit: Maximum number of lessons

        Returns:
            List of lesson strings
        """
        km = self.knowledge_mound
        if km is None:
            return []

        try:
            if hasattr(km, "search"):
                results = await km.search(
                    query=domain,
                    limit=limit,
                    filters={"tags": ["lesson_learned"]},
                )

                lessons = []
                for r in results:
                    content = getattr(r, "content", "")
                    # Extract lesson from content
                    if "[Lesson Learned]" in content:
                        parts = content.split("\n\n")
                        if len(parts) > 1:
                            lessons.append(parts[-1].strip())
                return lessons
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.debug("Lessons query failed: %s", e)

        return []


# Singleton accessor
_decision_plan_adapter: DecisionPlanAdapter | None = None


def get_decision_plan_adapter(
    knowledge_mound: Any | None = None,
) -> DecisionPlanAdapter:
    """Get or create the singleton DecisionPlanAdapter.

    Args:
        knowledge_mound: Optional KM instance

    Returns:
        DecisionPlanAdapter instance
    """
    global _decision_plan_adapter
    if _decision_plan_adapter is None or knowledge_mound is not None:
        _decision_plan_adapter = DecisionPlanAdapter(knowledge_mound=knowledge_mound)
    return _decision_plan_adapter


__all__ = [
    "DecisionPlanAdapter",
    "DecisionPlanAdapterError",
    "PlanIngestionResult",
    "get_decision_plan_adapter",
]
