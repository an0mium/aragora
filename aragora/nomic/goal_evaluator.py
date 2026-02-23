"""Semantic goal evaluation for self-improvement cycles.

Evaluates whether an execution attempt actually achieved its stated goal,
going beyond simple test pass/fail metrics.

Three scoring dimensions:
1. Scope coverage — were the files in file_scope actually modified?
2. Test delta — did test pass rate improve?
3. Diff relevance — does the diff touch goal-related concepts?

Usage:
    evaluator = GoalEvaluator()
    score = evaluator.evaluate(
        goal="Improve authentication error handling",
        file_scope=["aragora/auth/oidc.py"],
        files_changed=["aragora/auth/oidc.py", "tests/auth/test_oidc.py"],
        diff_summary="...",
        tests_before={"passed": 10, "failed": 2},
        tests_after={"passed": 12, "failed": 0},
    )
    print(score.achievement_score)  # 0.0 - 1.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GoalEvaluation:
    """Result of evaluating goal achievement."""

    goal: str
    achievement_score: float  # 0.0 - 1.0, composite score
    scope_coverage: float  # 0.0 - 1.0, fraction of target files modified
    test_delta: float  # -1.0 to 1.0, improvement in test pass rate
    diff_relevance: float  # 0.0 - 1.0, keyword overlap between goal and diff
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def achieved(self) -> bool:
        """Whether the goal was meaningfully achieved (score >= 0.5)."""
        return self.achievement_score >= 0.5


class GoalEvaluator:
    """Evaluates whether an execution actually achieved its stated goal.

    Uses heuristic scoring across three dimensions:
    - Scope coverage: were the intended files touched?
    - Test delta: did tests improve?
    - Diff relevance: does the diff relate to the goal?

    No LLM calls — pure signal-based evaluation.
    """

    def __init__(
        self,
        scope_weight: float = 0.35,
        test_weight: float = 0.35,
        relevance_weight: float = 0.30,
    ):
        self.scope_weight = scope_weight
        self.test_weight = test_weight
        self.relevance_weight = relevance_weight

    def evaluate(
        self,
        goal: str,
        file_scope: list[str] | None = None,
        files_changed: list[str] | None = None,
        diff_summary: str = "",
        tests_before: dict[str, int] | None = None,
        tests_after: dict[str, int] | None = None,
    ) -> GoalEvaluation:
        """Evaluate goal achievement across all dimensions.

        Args:
            goal: The stated objective.
            file_scope: Files that were supposed to be modified.
            files_changed: Files that were actually modified.
            diff_summary: Git diff text of the changes made.
            tests_before: {"passed": N, "failed": M} before execution.
            tests_after: {"passed": N, "failed": M} after execution.

        Returns:
            GoalEvaluation with scores and details.
        """
        scope = self._score_scope_coverage(file_scope or [], files_changed or [])
        test = self._score_test_delta(tests_before or {}, tests_after or {})
        relevance = self._score_diff_relevance(goal, diff_summary)

        composite = (
            scope * self.scope_weight
            + max(test, 0) * self.test_weight  # only reward improvement
            + relevance * self.relevance_weight
        )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        return GoalEvaluation(
            goal=goal,
            achievement_score=composite,
            scope_coverage=scope,
            test_delta=test,
            diff_relevance=relevance,
            details={
                "file_scope_count": len(file_scope or []),
                "files_changed_count": len(files_changed or []),
                "tests_before": tests_before or {},
                "tests_after": tests_after or {},
            },
        )

    @staticmethod
    def _score_scope_coverage(
        file_scope: list[str],
        files_changed: list[str],
    ) -> float:
        """Score how well the changes covered the intended file scope.

        Returns 1.0 if all target files were modified, 0.0 if none were.
        Also gives partial credit for modifying test files corresponding
        to source files in scope.
        """
        if not file_scope:
            # No target scope defined — give full credit if anything changed
            return 1.0 if files_changed else 0.0

        changed_set = set(files_changed)
        hits: float = 0

        for target in file_scope:
            if target in changed_set:
                hits += 1
            else:
                # Partial credit: check if a test file for this module changed
                # e.g. aragora/auth/oidc.py → tests/auth/test_oidc.py
                module_name = target.rsplit("/", 1)[-1].replace(".py", "")
                test_pattern = f"test_{module_name}"
                if any(test_pattern in f for f in changed_set):
                    hits += 0.5

        return min(1.0, hits / len(file_scope))

    @staticmethod
    def _score_test_delta(
        before: dict[str, int],
        after: dict[str, int],
    ) -> float:
        """Score change in test pass rate.

        Returns value in [-1.0, 1.0]:
        - Positive: tests improved
        - 0.0: no change or no data
        - Negative: tests regressed
        """
        before_passed = before.get("passed", 0)
        before_failed = before.get("failed", 0)
        after_passed = after.get("passed", 0)
        after_failed = after.get("failed", 0)

        before_total = before_passed + before_failed
        after_total = after_passed + after_failed

        if before_total == 0 and after_total == 0:
            return 0.0

        if before_total == 0:
            # No tests before — credit for adding passing tests
            return 1.0 if after_failed == 0 and after_passed > 0 else 0.5

        before_rate = before_passed / before_total if before_total > 0 else 0.0
        after_rate = after_passed / after_total if after_total > 0 else 0.0

        return after_rate - before_rate

    @staticmethod
    def _score_diff_relevance(goal: str, diff_summary: str) -> float:
        """Score how relevant the diff is to the goal using keyword overlap.

        Extracts meaningful words from the goal and checks how many appear
        in the diff. Filters out common stop words and very short tokens.
        """
        if not diff_summary:
            return 0.0

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "in",
            "to",
            "for",
            "and",
            "or",
            "of",
            "on",
            "at",
            "by",
            "with",
            "from",
            "as",
            "it",
            "be",
            "this",
            "that",
            "are",
            "was",
            "were",
            "will",
            "can",
            "do",
            "not",
            "all",
            "each",
            "any",
            "more",
            "also",
            "but",
            "if",
            "so",
        }

        # Extract meaningful words from goal
        goal_words = set(
            w.lower()
            for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", goal)
            if len(w) > 2 and w.lower() not in stop_words
        )

        if not goal_words:
            return 0.5  # Can't assess, neutral

        diff_lower = diff_summary.lower()
        matches = sum(1 for w in goal_words if w in diff_lower)

        return min(1.0, matches / len(goal_words))


__all__ = [
    "GoalEvaluation",
    "GoalEvaluator",
]
