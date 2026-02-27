"""Cross-cycle learning memory — closes the self-improvement feedback loop.

Records outcomes from each improvement cycle and makes them available to
future planning. This is the critical missing piece that allows the
self-improvement system to learn from its own history.

Three components:

1. **CycleOutcomeAnalyzer** — Analyzes execution results and extracts
   structured learnings (goal type → success pattern, file → risk level).

2. **StrategicMemoryStore** — Persists cross-cycle learnings in a
   queryable format. Supports semantic similarity for finding precedents.

3. **FeedbackBridge** — Routes execution results back into the
   ImprovementQueue for the next MetaPlanner cycle.

Usage::

    store = StrategicMemoryStore()
    analyzer = CycleOutcomeAnalyzer(store)

    # After an improvement cycle completes
    insights = analyzer.analyze_cycle(
        cycle_id="cycle-42",
        goals=[{"title": "Improve test coverage", "track": "qa"}],
        results=[{"success": True, "tests_passed": 15, "tests_failed": 0}],
    )

    # Next cycle: query for similar past work
    precedents = store.find_similar("Improve test coverage")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CycleInsight:
    """A structured learning from an improvement cycle."""

    cycle_id: str
    goal_type: str  # test_coverage | performance | ux | refactor | security | docs
    objective: str  # What was attempted
    outcome: str  # succeeded | failed | partial | reverted
    success_score: float  # 0.0-1.0
    files_changed: list[str] = field(default_factory=list)
    key_learnings: list[str] = field(default_factory=list)
    failure_reason: str = ""
    metrics_delta: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "goal_type": self.goal_type,
            "objective": self.objective,
            "outcome": self.outcome,
            "success_score": self.success_score,
            "files_changed": self.files_changed,
            "key_learnings": self.key_learnings,
            "failure_reason": self.failure_reason,
            "metrics_delta": self.metrics_delta,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleInsight:
        return cls(
            cycle_id=data.get("cycle_id", ""),
            goal_type=data.get("goal_type", ""),
            objective=data.get("objective", ""),
            outcome=data.get("outcome", ""),
            success_score=data.get("success_score", 0.0),
            files_changed=data.get("files_changed", []),
            key_learnings=data.get("key_learnings", []),
            failure_reason=data.get("failure_reason", ""),
            metrics_delta=data.get("metrics_delta", {}),
            duration_seconds=data.get("duration_seconds", 0.0),
            tokens_used=data.get("tokens_used", 0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class GoalTypeStats:
    """Aggregated statistics for a goal type across cycles."""

    goal_type: str
    total_attempts: int = 0
    successes: int = 0
    failures: int = 0
    avg_score: float = 0.0
    common_failure_reasons: list[str] = field(default_factory=list)
    risky_files: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successes / self.total_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_type": self.goal_type,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "avg_score": self.avg_score,
            "common_failure_reasons": self.common_failure_reasons,
            "risky_files": self.risky_files,
        }


class StrategicMemoryStore:
    """Persistent cross-cycle learning store.

    Stores CycleInsights and provides query methods for finding
    relevant precedents during planning.

    Storage is file-based (JSON) for simplicity. Can be extended
    to use KnowledgeMound for semantic search.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            store_path = Path.home() / ".aragora" / "strategic_memory.json"
        self.store_path = store_path
        self._insights: list[CycleInsight] = []
        self._load()

    def record(self, insight: CycleInsight) -> None:
        """Record a cycle insight."""
        self._insights.append(insight)
        self._save()
        logger.info(
            "Recorded cycle insight: %s (%s, score=%.2f)",
            insight.cycle_id,
            insight.outcome,
            insight.success_score,
        )

    def find_similar(
        self,
        objective: str,
        goal_type: str | None = None,
        limit: int = 5,
    ) -> list[CycleInsight]:
        """Find insights from similar past objectives.

        Uses keyword overlap for similarity (lightweight alternative
        to embedding-based search). Returns most recent matches first.
        """
        objective_words = set(objective.lower().split())
        scored: list[tuple[float, CycleInsight]] = []

        for insight in self._insights:
            # Skip if filtering by goal_type and it doesn't match
            if goal_type and insight.goal_type != goal_type:
                continue

            insight_words = set(insight.objective.lower().split())
            if not insight_words:
                continue

            # Jaccard similarity
            intersection = objective_words & insight_words
            union = objective_words | insight_words
            similarity = len(intersection) / len(union) if union else 0.0

            if similarity > 0.1:  # Minimum threshold
                scored.append((similarity, insight))

        # Sort by similarity desc, then recency
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [insight for _, insight in scored[:limit]]

    def get_goal_type_stats(self, goal_type: str) -> GoalTypeStats:
        """Get aggregated statistics for a goal type."""
        relevant = [i for i in self._insights if i.goal_type == goal_type]

        stats = GoalTypeStats(goal_type=goal_type)
        stats.total_attempts = len(relevant)

        if not relevant:
            return stats

        scores = []
        failure_reasons: dict[str, int] = {}
        file_failure_count: dict[str, int] = {}

        for insight in relevant:
            if insight.outcome in ("succeeded", "partial"):
                stats.successes += 1
            else:
                stats.failures += 1
                if insight.failure_reason:
                    failure_reasons[insight.failure_reason] = (
                        failure_reasons.get(insight.failure_reason, 0) + 1
                    )

            scores.append(insight.success_score)

            # Track files that appear in failed cycles
            if insight.outcome == "failed":
                for f in insight.files_changed:
                    file_failure_count[f] = file_failure_count.get(f, 0) + 1

        stats.avg_score = sum(scores) / len(scores) if scores else 0.0

        # Top 3 failure reasons
        stats.common_failure_reasons = sorted(
            failure_reasons,
            key=failure_reasons.get,
            reverse=True,  # type: ignore[arg-type]
        )[:3]

        # Files that appear in 2+ failed cycles
        stats.risky_files = [f for f, count in file_failure_count.items() if count >= 2]

        return stats

    def get_all_stats(self) -> dict[str, GoalTypeStats]:
        """Get stats for all goal types."""
        goal_types = {i.goal_type for i in self._insights}
        return {gt: self.get_goal_type_stats(gt) for gt in goal_types}

    def get_recent(self, limit: int = 10) -> list[CycleInsight]:
        """Get the most recent insights."""
        return sorted(self._insights, key=lambda i: i.timestamp, reverse=True)[:limit]

    def _load(self) -> None:
        """Load insights from disk."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                self._insights = [CycleInsight.from_dict(d) for d in data.get("insights", [])]
                logger.debug("Loaded %d strategic memory insights", len(self._insights))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load strategic memory: %s", exc)
                self._insights = []
        else:
            self._insights = []

    def _save(self) -> None:
        """Save insights to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"insights": [i.to_dict() for i in self._insights]}
        self.store_path.write_text(json.dumps(data, indent=2))


# -- Goal type classification --------------------------------------------------

GOAL_TYPE_KEYWORDS: dict[str, list[str]] = {
    "test_coverage": ["test", "coverage", "testing", "pytest", "spec"],
    "performance": ["performance", "speed", "latency", "optimize", "fast", "slow"],
    "ux": ["user", "ux", "ui", "experience", "frontend", "onboarding", "dashboard"],
    "refactor": ["refactor", "clean", "reorganize", "modularize", "extract"],
    "security": ["security", "auth", "permission", "rbac", "encrypt", "vulnerability"],
    "reliability": ["reliability", "resilience", "circuit", "retry", "fallback", "error"],
    "docs": ["documentation", "readme", "docs", "guide", "reference"],
}


def classify_goal_type(objective: str) -> str:
    """Classify a goal objective into a goal type using keyword matching."""
    objective_lower = objective.lower()
    scores: dict[str, int] = {}

    for goal_type, keywords in GOAL_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in objective_lower)
        if score > 0:
            scores[goal_type] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)  # type: ignore[arg-type]


class CycleOutcomeAnalyzer:
    """Analyzes improvement cycle results and extracts structured learnings.

    Bridges execution results → CycleInsight → StrategicMemoryStore.
    """

    def __init__(self, store: StrategicMemoryStore | None = None) -> None:
        self.store = store or StrategicMemoryStore()

    def analyze_cycle(
        self,
        cycle_id: str,
        goals: list[dict[str, Any]],
        results: list[dict[str, Any]],
        metrics_before: dict[str, float] | None = None,
        metrics_after: dict[str, float] | None = None,
    ) -> list[CycleInsight]:
        """Analyze a completed cycle and record learnings.

        Args:
            cycle_id: Unique identifier for this cycle.
            goals: List of goal dicts with at least 'title' or 'objective'.
            results: List of result dicts with 'success', optionally
                     'files_changed', 'tests_passed', 'tests_failed', 'error'.
            metrics_before: Codebase metrics snapshot before execution.
            metrics_after: Codebase metrics snapshot after execution.

        Returns:
            List of CycleInsight objects (one per goal).
        """
        insights: list[CycleInsight] = []

        for i, goal in enumerate(goals):
            result = results[i] if i < len(results) else {}
            insight = self._analyze_single(cycle_id, goal, result, metrics_before, metrics_after)
            insights.append(insight)
            self.store.record(insight)

        return insights

    def get_planning_context(
        self,
        objective: str,
        goal_type: str | None = None,
    ) -> dict[str, Any]:
        """Get context for planning from strategic memory.

        Returns a dict suitable for injecting into MetaPlanner prompts.
        """
        similar = self.store.find_similar(objective, goal_type=goal_type, limit=3)

        if goal_type is None:
            goal_type = classify_goal_type(objective)

        stats = self.store.get_goal_type_stats(goal_type)

        context: dict[str, Any] = {
            "goal_type": goal_type,
            "goal_type_stats": stats.to_dict(),
            "similar_past_cycles": [s.to_dict() for s in similar],
            "recommendations": [],
        }

        # Generate recommendations from history
        if stats.success_rate < 0.3 and stats.total_attempts >= 3:
            context["recommendations"].append(
                f"Warning: '{goal_type}' goals have a {stats.success_rate:.0%} success rate. "
                "Consider breaking into smaller scope or different approach."
            )

        if stats.risky_files:
            context["recommendations"].append(
                f"Caution: These files frequently cause failures: {', '.join(stats.risky_files[:3])}"
            )

        for past in similar:
            if past.outcome == "failed" and past.key_learnings:
                context["recommendations"].append(
                    f"Past lesson ({past.cycle_id}): {past.key_learnings[0]}"
                )

        return context

    def _analyze_single(
        self,
        cycle_id: str,
        goal: dict[str, Any],
        result: dict[str, Any],
        metrics_before: dict[str, float] | None,
        metrics_after: dict[str, float] | None,
    ) -> CycleInsight:
        """Analyze a single goal's execution result."""
        objective = goal.get("title", goal.get("objective", goal.get("description", "")))
        goal_type = goal.get("goal_type", classify_goal_type(objective))

        success = result.get("success", False)
        tests_passed = result.get("tests_passed", 0)
        tests_failed = result.get("tests_failed", 0)
        error = result.get("error", "")
        files_changed = result.get("files_changed", [])

        # Determine outcome
        if success and tests_failed == 0:
            outcome = "succeeded"
            success_score = 1.0
        elif success and tests_failed > 0:
            outcome = "partial"
            total_tests = tests_passed + tests_failed
            success_score = tests_passed / total_tests if total_tests > 0 else 0.5
        else:
            outcome = "failed"
            success_score = 0.0

        # Extract learnings
        learnings = self._extract_learnings(goal, result, outcome)

        # Calculate metrics delta
        metrics_delta: dict[str, float] = {}
        if metrics_before and metrics_after:
            for key in metrics_after:
                if key in metrics_before:
                    metrics_delta[key] = metrics_after[key] - metrics_before[key]

        return CycleInsight(
            cycle_id=cycle_id,
            goal_type=goal_type,
            objective=objective,
            outcome=outcome,
            success_score=success_score,
            files_changed=files_changed,
            key_learnings=learnings,
            failure_reason=error if outcome == "failed" else "",
            metrics_delta=metrics_delta,
            duration_seconds=result.get("duration_seconds", 0.0),
            tokens_used=result.get("tokens_used", 0),
        )

    def _extract_learnings(
        self,
        goal: dict[str, Any],
        result: dict[str, Any],
        outcome: str,
    ) -> list[str]:
        """Extract structured learnings from a goal/result pair."""
        learnings: list[str] = []

        if outcome == "failed":
            error = result.get("error", "")
            if "import" in error.lower():
                learnings.append("Import errors — check module dependencies before editing")
            if "test" in error.lower():
                learnings.append("Test failures — run affected tests before committing")
            if not error:
                learnings.append("Unknown failure — consider adding better error reporting")

        if outcome == "succeeded":
            files = result.get("files_changed", [])
            if len(files) <= 2:
                learnings.append("Small, focused change — easier to verify")
            tests_passed = result.get("tests_passed", 0)
            if tests_passed > 10:
                learnings.append(f"Well-tested: {tests_passed} tests passed")

        if outcome == "partial":
            learnings.append("Partial success — may need follow-up cycle")

        return learnings


class FeedbackBridge:
    """Routes execution results back into the ImprovementQueue.

    Closes the loop: execution → analysis → improvement suggestions
    for the next MetaPlanner cycle.
    """

    def __init__(
        self,
        analyzer: CycleOutcomeAnalyzer | None = None,
    ) -> None:
        self.analyzer = analyzer or CycleOutcomeAnalyzer()

    def generate_improvement_suggestions(
        self,
        insights: list[CycleInsight],
    ) -> list[dict[str, Any]]:
        """Generate improvement suggestions from cycle insights.

        Returns dicts compatible with ImprovementSuggestion fields.
        """
        suggestions: list[dict[str, Any]] = []

        for insight in insights:
            if insight.outcome == "failed":
                suggestions.append(
                    {
                        "debate_id": insight.cycle_id,
                        "task": f"Retry: {insight.objective}",
                        "suggestion": (
                            f"Previous attempt failed: {insight.failure_reason}. "
                            f"Learnings: {', '.join(insight.key_learnings)}"
                        ),
                        "category": insight.goal_type,
                        "confidence": 0.6,
                    }
                )

            if insight.outcome == "partial":
                suggestions.append(
                    {
                        "debate_id": insight.cycle_id,
                        "task": f"Complete: {insight.objective}",
                        "suggestion": (
                            f"Partial success (score={insight.success_score:.2f}). "
                            "Follow-up needed to address remaining failures."
                        ),
                        "category": insight.goal_type,
                        "confidence": 0.7,
                    }
                )

            # If a goal type consistently fails, suggest meta-improvement
            stats = self.analyzer.store.get_goal_type_stats(insight.goal_type)
            if stats.total_attempts >= 3 and stats.success_rate < 0.3:
                suggestions.append(
                    {
                        "debate_id": insight.cycle_id,
                        "task": f"Improve '{insight.goal_type}' approach",
                        "suggestion": (
                            f"'{insight.goal_type}' goals succeed only {stats.success_rate:.0%} of the time. "
                            f"Common failures: {', '.join(stats.common_failure_reasons[:2])}. "
                            "Consider changing the approach or breaking goals into smaller scope."
                        ),
                        "category": "meta_improvement",
                        "confidence": 0.8,
                    }
                )

        return suggestions

    def bridge_to_queue(
        self,
        insights: list[CycleInsight],
    ) -> int:
        """Bridge insights to the ImprovementQueue.

        Returns the number of suggestions enqueued.
        """
        try:
            from aragora.nomic.improvement_queue import ImprovementQueue, ImprovementSuggestion
        except ImportError:
            logger.debug("ImprovementQueue not available")
            return 0

        suggestions = self.generate_improvement_suggestions(insights)

        queue = ImprovementQueue()

        count = 0
        for s in suggestions:
            queue.enqueue(
                ImprovementSuggestion(
                    debate_id=s["debate_id"],
                    task=s["task"],
                    suggestion=s["suggestion"],
                    category=s["category"],
                    confidence=s["confidence"],
                )
            )
            count += 1

        logger.info("Bridged %d improvement suggestions to queue", count)
        return count
