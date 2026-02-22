"""Goal Generator — converts CodebaseHealthReport into prioritized goals.

Bridges the AutonomousAssessmentEngine output to the SelfImprovePipeline
input format (PrioritizedGoal objects).

Usage:
    from aragora.nomic.assessment_engine import AutonomousAssessmentEngine
    from aragora.nomic.goal_generator import GoalGenerator

    engine = AutonomousAssessmentEngine()
    report = await engine.assess()

    generator = GoalGenerator()
    goals = generator.generate_goals(report)
    ideas = generator.generate_ideas(report)  # For pipeline visualization
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Category → Track mapping
_CATEGORY_TRACK_MAP = {
    "test": "qa",
    "lint": "core",
    "complexity": "core",
    "todo": "developer",
    "regression": "qa",
    "feedback": "core",
    "general": "core",
}

# Category → estimated impact mapping
_CATEGORY_IMPACT_MAP = {
    "test": "high",
    "regression": "high",
    "lint": "medium",
    "complexity": "medium",
    "todo": "low",
    "feedback": "medium",
    "general": "medium",
}


class GoalGenerator:
    """Converts CodebaseHealthReport into PrioritizedGoal objects.

    Transforms assessment improvement candidates into goals that
    SelfImprovePipeline can execute.
    """

    def __init__(self, max_goals: int = 5) -> None:
        self._max_goals = max_goals

    def generate_goals(self, report: Any) -> list[Any]:
        """Convert a CodebaseHealthReport into PrioritizedGoal objects.

        Args:
            report: CodebaseHealthReport from AutonomousAssessmentEngine.

        Returns:
            List of PrioritizedGoal objects, ranked by priority.
        """
        try:
            from aragora.nomic.meta_planner import PrioritizedGoal, Track
        except ImportError:
            logger.warning("PrioritizedGoal not importable, returning empty goals")
            return []

        candidates = getattr(report, "improvement_candidates", [])
        if not candidates:
            logger.info("goal_generator_no_candidates health_score=%.2f",
                        getattr(report, "health_score", 0.0))
            return []

        goals: list[Any] = []
        for i, candidate in enumerate(candidates[:self._max_goals]):
            track_name = _CATEGORY_TRACK_MAP.get(candidate.category, "core")
            try:
                track = Track(track_name)
            except ValueError:
                track = Track.CORE

            impact = _CATEGORY_IMPACT_MAP.get(candidate.category, "medium")

            goal = PrioritizedGoal(
                id=f"auto_{i}_{candidate.category}",
                track=track,
                description=candidate.description,
                rationale=f"Auto-generated from {candidate.source} assessment "
                          f"(priority={candidate.priority:.2f})",
                estimated_impact=impact,
                priority=i + 1,
                file_hints=candidate.files[:10],
            )
            goals.append(goal)

        logger.info(
            "goal_generator_produced goals=%d from_candidates=%d",
            len(goals),
            len(candidates),
        )
        return goals

    def generate_ideas(self, report: Any) -> list[str]:
        """Convert a CodebaseHealthReport into idea strings for pipeline.

        Produces human-readable idea strings suitable for
        IdeaToExecutionPipeline.from_ideas().

        Args:
            report: CodebaseHealthReport from AutonomousAssessmentEngine.

        Returns:
            List of idea strings.
        """
        candidates = getattr(report, "improvement_candidates", [])
        if not candidates:
            return []

        ideas: list[str] = []
        for candidate in candidates[:self._max_goals]:
            files_hint = ""
            if candidate.files:
                files_hint = f" (files: {', '.join(candidate.files[:3])})"

            idea = f"[{candidate.category}] {candidate.description}{files_hint}"
            ideas.append(idea)

        return ideas

    def generate_objective(self, report: Any) -> str:
        """Generate a single objective string summarizing the top priority.

        Suitable for passing to SelfImprovePipeline.run(objective=...).

        Args:
            report: CodebaseHealthReport from AutonomousAssessmentEngine.

        Returns:
            Objective string, or a default if no candidates.
        """
        candidates = getattr(report, "improvement_candidates", [])
        if not candidates:
            return "Maintain codebase health (no issues detected)"

        top = candidates[0]
        return f"[auto-assess] {top.description}"


__all__ = [
    "GoalGenerator",
]
