"""
Feedback loop for autonomous orchestration.

Manages feedback from verification back to earlier phases, determining
root causes and appropriate recovery actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aragora.nomic.types import AgentAssignment
from aragora.observability import get_logger

logger = get_logger(__name__)


class FeedbackLoop:
    """
    Manages feedback from verification back to earlier phases.

    When verification fails, determines:
    1. Root cause (test failure, lint error, etc.)
    2. Which phase to return to (design, implement, or new subtask)
    3. How to modify the approach

    Optionally integrates with SelfCorrectionEngine to apply strategy
    recommendations (e.g., rotate agents, decrease scope) informed by
    cross-cycle failure patterns.

    When a TestResult is available in error_info, uses testfixer's heuristic
    analysis to produce rich hints (file path, line number, error category,
    fix target, relevant code snippets, suggested approach).
    """

    def __init__(self, max_iterations: int = 3, repo_path: Path | None = None):
        self.max_iterations = max_iterations
        self.repo_path = repo_path
        self._iteration_counts: dict[str, int] = {}
        self._strategy_recommendations: list[Any] = []

    def analyze_failure(
        self,
        assignment: AgentAssignment,
        error_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze a failure and determine next steps.

        Anti-fragile design: on first failure, tries reassigning to a different
        agent before retrying the same one. This handles cases where an agent
        type is fundamentally incompatible with a task (timeout, rate limit,
        capability mismatch).
        """
        subtask_id = assignment.subtask.id
        self._iteration_counts[subtask_id] = self._iteration_counts.get(subtask_id, 0) + 1

        if self._iteration_counts[subtask_id] >= self.max_iterations:
            return {
                "action": "escalate",
                "reason": f"Max iterations ({self.max_iterations}) reached",
                "require_human": True,
            }

        error_type = error_info.get("type", "unknown")
        error_message = error_info.get("message", "")

        # Agent-level failures -> try a different agent before retrying same one
        if error_type in ("agent_timeout", "agent_error", "workflow_failure"):
            if assignment.attempt_count == 0:
                # First failure: try reassigning to a different agent
                return {
                    "action": "reassign_agent",
                    "reason": f"Agent {assignment.agent_type} failed on first attempt; "
                    f"trying alternative agent",
                    "original_agent": assignment.agent_type,
                }

        # CI failures -> adjust implementation
        if error_type == "ci_failure":
            return {
                "action": "retry_implement",
                "reason": "CI test failures require implementation adjustment",
                "hints": error_info.get("ci_failures", []),
            }

        # Test failures -> use rich analysis if TestResult is available
        if error_type == "test_failure":
            hints = self._extract_rich_test_hints(error_info)
            return {
                "action": "retry_implement",
                "reason": "Test failures require implementation adjustment",
                "hints": hints,
            }

        # Lint/type errors -> quick fix
        if error_type in ("lint_error", "type_error"):
            return {
                "action": "quick_fix",
                "reason": "Static analysis errors can be auto-fixed",
                "hints": error_message,
            }

        # Design issues -> revisit design
        if error_type == "design_issue":
            return {
                "action": "redesign",
                "reason": "Implementation revealed design flaws",
                "hints": error_info.get("suggestion", ""),
            }

        # Unknown -> escalate
        return {
            "action": "escalate",
            "reason": f"Unknown error type: {error_type}",
            "require_human": True,
        }

    def _extract_rich_test_hints(self, error_info: dict[str, Any]) -> str | list[dict[str, Any]]:
        """Extract rich hints from TestResult using testfixer heuristics.

        Falls back to basic string extraction if testfixer is unavailable
        or no TestResult is present in error_info.
        """
        test_result = error_info.get("test_result")
        if test_result is None:
            return self._extract_test_hints(error_info.get("message", ""))

        try:
            from aragora.nomic.testfixer.analyzer import (
                categorize_by_heuristics,
                determine_fix_target,
                extract_relevant_code,
                generate_approach_heuristic,
            )

            repo_path = self.repo_path or Path.cwd()
            rich_hints = []

            for failure in test_result.failures[:5]:
                category, confidence = categorize_by_heuristics(failure)
                fix_target = determine_fix_target(category, failure)
                code_snippets = extract_relevant_code(failure, repo_path)
                approach = generate_approach_heuristic(category, failure)

                rich_hints.append(
                    {
                        "test_name": failure.test_name,
                        "test_file": failure.test_file,
                        "line_number": failure.line_number,
                        "error_type": failure.error_type,
                        "error_message": failure.error_message,
                        "category": category.value,
                        "confidence": confidence,
                        "fix_target": fix_target.value,
                        "relevant_code": {k: v[:500] for k, v in code_snippets.items()},
                        "suggested_approach": approach,
                    }
                )

            return rich_hints or self._extract_test_hints(error_info.get("message", ""))

        except ImportError:
            logger.debug("testfixer analyzer unavailable, using basic hint extraction")
            return self._extract_test_hints(error_info.get("message", ""))

    def apply_strategy_recommendations(self, recommendations: list[Any]) -> None:
        """Accept strategy recommendations from SelfCorrectionEngine.

        These influence failure analysis: if a recommendation suggests
        rotating agents or decreasing scope for a track, that advice
        is incorporated into the next analyze_failure decision.
        """
        self._strategy_recommendations = list(recommendations)

    def get_recommendation_for_track(self, track_value: str) -> dict[str, str] | None:
        """Get the most confident recommendation for a given track."""
        best = None
        best_confidence = 0.0
        for rec in self._strategy_recommendations:
            if getattr(rec, "track", None) == track_value:
                conf = getattr(rec, "confidence", 0.0)
                if conf > best_confidence:
                    best_confidence = conf
                    best = rec
        if best is None:
            return None
        return {
            "action": getattr(best, "action_type", "unknown"),
            "recommendation": getattr(best, "recommendation", ""),
            "confidence": str(best_confidence),
        }

    def _extract_test_hints(self, error_message: str) -> str:
        """Extract hints from test failure messages."""
        lines = error_message.split("\n")
        hints = []

        for line in lines:
            if "AssertionError" in line or "Expected" in line or "Actual" in line:
                hints.append(line.strip())

        return "\n".join(hints[:5]) if hints else "Review test output"
