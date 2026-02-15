"""Cross-test regression detection for TestFixer.

After applying a batch of fixes, runs the full test suite and compares
the failure set against the baseline to detect regressions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.testfixer.runner import TestFailure, TestRunner

logger = logging.getLogger(__name__)


@dataclass
class ImpactResult:
    """Result of a cross-test impact analysis."""

    new_failures: list[TestFailure] = field(default_factory=list)
    resolved_failures: list[str] = field(default_factory=list)
    unchanged_failures: list[str] = field(default_factory=list)
    has_regressions: bool = False
    total_tests: int = 0
    total_passed: int = 0

    @property
    def summary(self) -> str:
        parts = []
        if self.new_failures:
            parts.append(f"{len(self.new_failures)} new failure(s)")
        if self.resolved_failures:
            parts.append(f"{len(self.resolved_failures)} resolved")
        if self.unchanged_failures:
            parts.append(f"{len(self.unchanged_failures)} unchanged")
        return ", ".join(parts) or "no changes"


class ImpactAnalyzer:
    """Detect regressions by comparing test results against a baseline.

    Runs the full test suite and compares the set of failing tests
    against a known baseline. Any test that was previously passing
    but now fails is flagged as a regression.
    """

    def __init__(self, runner: TestRunner) -> None:
        self._runner = runner

    async def check_impact(
        self,
        baseline_failures: list[str],
        override_command: str | None = None,
    ) -> ImpactResult:
        """Run full test suite and compare against baseline failures.

        Args:
            baseline_failures: Test names that were already failing before
                the batch fix was applied.
            override_command: Optional test command override for the full
                suite run (e.g. without --maxfail).

        Returns:
            ImpactResult with new, resolved, and unchanged failures.
        """
        test_result = await self._runner.run_full(override_command=override_command)

        current_failing = {f.test_name for f in test_result.failures}
        baseline_set = set(baseline_failures)

        new_failures = [f for f in test_result.failures if f.test_name not in baseline_set]
        resolved = [name for name in baseline_failures if name not in current_failing]
        unchanged = [name for name in baseline_failures if name in current_failing]

        has_regressions = len(new_failures) > 0

        result = ImpactResult(
            new_failures=new_failures,
            resolved_failures=resolved,
            unchanged_failures=unchanged,
            has_regressions=has_regressions,
            total_tests=test_result.total_tests,
            total_passed=test_result.passed,
        )

        logger.info(
            "impact.check new=%d resolved=%d unchanged=%d regressions=%s",
            len(new_failures),
            len(resolved),
            len(unchanged),
            has_regressions,
        )

        return result
