"""
Decision plan memory feedback - outcome recording and learning loop.

Stability: STABLE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aragora.pipeline.decision_plan.core import DecisionPlan, PlanStatus

logger = logging.getLogger(__name__)

# =========================================================================
# Memory feedback loop
# =========================================================================


@dataclass
class PlanOutcome:
    """Records the outcome of executing a DecisionPlan.

    This is the data structure that feeds back into organizational memory,
    closing the gold-path loop: debate → plan → execute → verify → LEARN.
    """

    plan_id: str
    debate_id: str
    task: str
    success: bool
    tasks_completed: int = 0
    tasks_total: int = 0
    verification_passed: int = 0
    verification_total: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None
    duration_seconds: float = 0.0
    lessons: list[str] = field(default_factory=list)
    receipt_id: str | None = None  # Cryptographic receipt ID for audit trail
    review: dict[str, Any] | None = None
    review_passed: bool | None = None

    @property
    def completion_rate(self) -> float:
        if self.tasks_total == 0:
            return 0.0
        return self.tasks_completed / self.tasks_total

    @property
    def verification_rate(self) -> float:
        if self.verification_total == 0:
            return 0.0
        return self.verification_passed / self.verification_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "debate_id": self.debate_id,
            "task": self.task,
            "success": self.success,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "verification_passed": self.verification_passed,
            "verification_total": self.verification_total,
            "total_cost_usd": self.total_cost_usd,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "completion_rate": self.completion_rate,
            "verification_rate": self.verification_rate,
            "lessons": self.lessons,
            "receipt_id": self.receipt_id,
            "review": self.review,
            "review_passed": self.review_passed,
        }

    def to_memory_content(self) -> str:
        """Format outcome as structured text for memory storage."""
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [
            f"[Decision Plan Outcome: {status}]",
            f"Task: {self.task}",
            f"Debate: {self.debate_id}",
            f"Completion: {self.tasks_completed}/{self.tasks_total} tasks",
            f"Verification: {self.verification_passed}/{self.verification_total} cases",
            f"Cost: ${self.total_cost_usd:.4f}",
        ]
        if self.review_passed is not None:
            review_status = "PASS" if self.review_passed else "FAIL"
            lines.append(f"Review: {review_status}")
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.lessons:
            lines.append("Lessons learned:")
            for lesson in self.lessons:
                lines.append(f"  - {lesson}")
        return "\n".join(lines)


async def record_plan_outcome(
    plan: DecisionPlan,
    outcome: PlanOutcome,
    *,
    continuum_memory: Any | None = None,
    knowledge_mound: Any | None = None,
) -> dict[str, Any]:
    """Write implementation outcome back to organizational memory.

    This closes the gold-path feedback loop by recording what happened
    when a debate's conclusions were actually implemented.

    The recorded data feeds future debates via:
    - ContinuumMemory: Pattern learning (what debate patterns lead to
      successful implementations)
    - KnowledgeMound: Organizational knowledge (lessons learned,
      successful approaches)

    Args:
        plan: The executed DecisionPlan.
        outcome: The PlanOutcome recording what happened.
        continuum_memory: Optional ContinuumMemory instance.
        knowledge_mound: Optional KnowledgeMound instance.

    Returns:
        Dict with write results: {continuum_id, mound_id, errors}.
    """
    import logging

    logger = logging.getLogger(__name__)
    results: dict[str, Any] = {"continuum_id": None, "mound_id": None, "errors": []}

    plan_meta: dict[str, Any] = plan.metadata if isinstance(plan.metadata, dict) else {}
    workspace_id = plan_meta.get("workspace_id") or plan_meta.get("tenant_id")
    org_id = plan_meta.get("org_id")
    owner_id = (
        plan_meta.get("owner_id") or plan_meta.get("user_id") or plan_meta.get("requested_by")
    )
    scope = plan_meta.get("memory_scope") or plan_meta.get("scope") or "workspace"

    # Update plan state
    plan.status = PlanStatus.COMPLETED if outcome.success else PlanStatus.FAILED
    plan.execution_completed_at = datetime.now()
    plan.execution_error = outcome.error
    plan.budget.spent_usd = outcome.total_cost_usd

    # 1. Write to ContinuumMemory (pattern learning)
    if continuum_memory is not None:
        try:
            # Determine importance based on outcome
            # Failures are more important to learn from than successes
            importance = 0.8 if not outcome.success else 0.6
            # Low verification rates are surprising and important
            if outcome.verification_rate < 0.5:
                importance = min(1.0, importance + 0.2)

            from aragora.memory.continuum import MemoryTier

            entry = await continuum_memory.store(
                key=f"plan_outcome:{plan.id}",
                content=outcome.to_memory_content(),
                tier=MemoryTier.SLOW,  # Cross-session learning
                importance=importance,
                metadata={
                    "type": "plan_outcome",
                    "plan_id": plan.id,
                    "debate_id": plan.debate_id,
                    "success": outcome.success,
                    "completion_rate": outcome.completion_rate,
                    "verification_rate": outcome.verification_rate,
                    "cost_usd": outcome.total_cost_usd,
                    "tenant_id": workspace_id,
                    "workspace_id": workspace_id,
                    "org_id": org_id,
                    "owner_id": owner_id,
                    "scope": scope,
                },
            )
            results["continuum_id"] = entry.id
            logger.info(
                "Recorded plan outcome to ContinuumMemory: %s (importance=%.2f)",
                entry.id,
                importance,
            )

            # Update the original debate memory entry with outcome data
            if plan.debate_result and plan.debate_result.debate_id:
                try:
                    continuum_memory.update_outcome(
                        id=f"debate:{plan.debate_result.debate_id}",
                        success=outcome.success,
                        agent_prediction_error=1.0 - outcome.completion_rate,
                    )
                except Exception:
                    logger.debug("Failed to update debate memory outcome", exc_info=True)

        except Exception as e:
            err = f"ContinuumMemory write failed: {e}"
            logger.warning(err)
            results["errors"].append(err)

    # 2. Write to KnowledgeMound (organizational knowledge)
    if knowledge_mound is not None:
        try:
            confidence = outcome.verification_rate if outcome.success else 0.3

            item_id = await knowledge_mound.store_knowledge(
                content=outcome.to_memory_content(),
                source="decision_plan",
                source_id=plan.id,
                confidence=confidence,
                metadata={
                    "type": "implementation_outcome",
                    "plan_id": plan.id,
                    "debate_id": plan.debate_id,
                    "task": plan.task[:200],
                    "success": outcome.success,
                    "lessons": outcome.lessons[:5],
                    "workspace_id": workspace_id,
                    "org_id": org_id,
                    "owner_id": owner_id,
                    "scope": scope,
                },
            )
            results["mound_id"] = item_id
            logger.info("Recorded plan outcome to KnowledgeMound: %s", item_id)

        except Exception as e:
            err = f"KnowledgeMound write failed: {e}"
            logger.warning(err)
            results["errors"].append(err)

    # 3. Extract patterns from debate (organizational learning)
    if outcome.success and plan.debate_result is not None:
        try:
            # Extract patterns from the winning debate approach
            from aragora.evolution.pattern_extractor import PatternExtractor

            extractor = PatternExtractor()

            # Build outcome dict from debate result for pattern extraction
            debate_outcome = _build_debate_outcome_dict(plan.debate_result)
            if debate_outcome:
                patterns = extractor.extract(debate_outcome)

                if patterns and knowledge_mound is not None:
                    stored_count = 0
                    for pattern in patterns[:5]:  # Limit to top 5 patterns
                        try:
                            await knowledge_mound.store_knowledge(
                                content=f"Pattern: {pattern.description}\nExamples: {', '.join(pattern.examples[:3])}",
                                source="pattern_extractor",
                                source_id=f"{plan.id}:{pattern.pattern_type}",
                                confidence=pattern.effectiveness,
                                metadata={
                                    "type": "debate_pattern",
                                    "pattern_type": pattern.pattern_type,
                                    "plan_id": plan.id,
                                    "debate_id": plan.debate_id,
                                    "agent": pattern.agent,
                                    "frequency": pattern.frequency,
                                },
                            )
                            stored_count += 1
                        except Exception:
                            logger.debug("Failed to store pattern in knowledge mound", exc_info=True)

                    if stored_count > 0:
                        logger.info(
                            "Extracted and stored %d patterns from debate %s",
                            stored_count,
                            plan.debate_id,
                        )
                    results["patterns_stored"] = stored_count

        except ImportError:
            pass  # PatternExtractor not available
        except Exception as e:
            logger.debug("Pattern extraction skipped: %s", e)

    # Mark feedback written
    plan.memory_written = len(results["errors"]) == 0

    return results


def _extract_debate_content(debate_result: Any) -> str:
    """Extract text content from DebateResult for pattern analysis."""
    parts: list[str] = []

    # Final answer is the main content
    if hasattr(debate_result, "final_answer") and debate_result.final_answer:
        parts.append(debate_result.final_answer)

    # Add critiques if available
    if hasattr(debate_result, "critiques") and debate_result.critiques:
        for critique in debate_result.critiques[:5]:
            if hasattr(critique, "content"):
                parts.append(critique.content)
            elif isinstance(critique, str):
                parts.append(critique)

    # Add winning arguments if available
    if hasattr(debate_result, "arguments") and debate_result.arguments:
        for arg in debate_result.arguments[:5]:
            if hasattr(arg, "content"):
                parts.append(arg.content)
            elif isinstance(arg, str):
                parts.append(arg)

    return "\n\n".join(parts)


def _build_debate_outcome_dict(debate_result: Any) -> dict[str, Any]:
    """Build a dict from DebateResult for PatternExtractor.extract().

    PatternExtractor expects a dict with 'winner', 'messages', and 'critiques' keys.
    """
    outcome: dict[str, Any] = {}

    # Extract winner
    if hasattr(debate_result, "winner"):
        outcome["winner"] = debate_result.winner
    elif hasattr(debate_result, "winning_agent"):
        outcome["winner"] = debate_result.winning_agent
    else:
        outcome["winner"] = None

    # Extract messages
    messages: list[dict[str, Any]] = []
    if hasattr(debate_result, "messages") and debate_result.messages:
        for msg in debate_result.messages:
            if hasattr(msg, "to_dict"):
                messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                messages.append(msg)
            else:
                messages.append(
                    {
                        "content": getattr(msg, "content", str(msg)),
                        "agent": getattr(msg, "agent", None),
                    }
                )
    outcome["messages"] = messages

    # Extract critiques
    critiques: list[dict[str, Any]] = []
    if hasattr(debate_result, "critiques") and debate_result.critiques:
        for critique in debate_result.critiques:
            if hasattr(critique, "to_dict"):
                critiques.append(critique.to_dict())
            elif isinstance(critique, dict):
                critiques.append(critique)
            else:
                critiques.append(
                    {
                        "content": getattr(critique, "content", str(critique)),
                        "agent": getattr(critique, "agent", None),
                    }
                )
    outcome["critiques"] = critiques

    return outcome
