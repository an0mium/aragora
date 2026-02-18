"""Baseline metrics collection for pre/post implementation comparison.

Captures test pass rate and lint error counts before implementation starts,
then compares against a post-implementation snapshot to produce a metrics
delta that populates NomicCycleOutcome.metrics_delta.

Usage:
    collector = BaselineCollector(aragora_path)
    baseline = await collector.collect()
    # ... run implementation ...
    post = await collector.collect()
    delta = baseline.compare(post)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BaselineSnapshot:
    """Pre- or post-implementation metrics snapshot."""

    tests_passed: int
    tests_failed: int
    lint_errors: int
    test_pass_rate: float
    timestamp: float

    def compare(self, after: BaselineSnapshot) -> dict[str, Any]:
        """Compare this baseline to a post-implementation snapshot.

        Returns:
            Dictionary with delta values suitable for NomicCycleOutcome.metrics_delta.
        """
        return {
            "tests_passed_delta": after.tests_passed - self.tests_passed,
            "tests_failed_delta": after.tests_failed - self.tests_failed,
            "lint_errors_delta": after.lint_errors - self.lint_errors,
            "test_pass_rate_delta": after.test_pass_rate - self.test_pass_rate,
            "improved": (
                after.test_pass_rate >= self.test_pass_rate
                and after.lint_errors <= self.lint_errors
            ),
            "improvement_score": self._compute_score(after),
        }

    def _compute_score(self, after: BaselineSnapshot) -> float:
        """Compute a 0.0-1.0 improvement score.

        Combines test pass rate improvement and lint error reduction into
        a single score. Neutral (no change) = 0.5, improvement > 0.5,
        regression < 0.5.
        """
        # Test pass rate delta contribution (0-0.5)
        rate_delta = after.test_pass_rate - self.test_pass_rate
        test_score = 0.25 + min(max(rate_delta * 2.5, -0.25), 0.25)

        # Lint error reduction contribution (0-0.5)
        if self.lint_errors > 0:
            lint_reduction = (self.lint_errors - after.lint_errors) / self.lint_errors
        elif after.lint_errors == 0:
            lint_reduction = 0.0  # No change
        else:
            lint_reduction = -1.0  # Regression from 0 errors

        lint_score = 0.25 + min(max(lint_reduction * 0.25, -0.25), 0.25)

        return min(max(test_score + lint_score, 0.0), 1.0)


class BaselineCollector:
    """Collects baseline metrics by running quick pytest and ruff checks."""

    def __init__(self, aragora_path: Path):
        self.aragora_path = aragora_path

    async def collect(self) -> BaselineSnapshot:
        """Run quick pytest and ruff to capture current metrics.

        Returns:
            BaselineSnapshot with test and lint metrics.
        """
        tests_passed, tests_failed = await self._collect_test_metrics()
        lint_errors = await self._collect_lint_metrics()

        total = tests_passed + tests_failed
        rate = tests_passed / total if total > 0 else 1.0

        return BaselineSnapshot(
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            lint_errors=lint_errors,
            test_pass_rate=rate,
            timestamp=time.time(),
        )

    async def _collect_test_metrics(self) -> tuple[int, int]:
        """Run quick pytest to count pass/fail."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest",
                "tests/", "-x", "-q", "--tb=no",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode() if stdout else ""
            return self._parse_pytest_output(output)
        except (asyncio.TimeoutError, OSError) as e:
            logger.warning("Baseline test collection failed: %s", e)
            return 0, 0

    async def _collect_lint_metrics(self) -> int:
        """Run ruff check to count lint errors."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", "--count", "--quiet", ".",
                cwd=self.aragora_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode().strip() if stdout else ""
            return self._parse_ruff_output(output)
        except (asyncio.TimeoutError, OSError, FileNotFoundError) as e:
            logger.warning("Baseline lint collection failed: %s", e)
            return 0

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int]:
        """Parse pytest -q output for passed/failed counts."""
        import re

        passed = 0
        failed = 0
        # Match "N passed" and "N failed" patterns
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        return passed, failed

    @staticmethod
    def _parse_ruff_output(output: str) -> int:
        """Parse ruff --count output for error count."""
        import re

        # ruff --count outputs lines like "Found 42 errors."
        match = re.search(r"(\d+)", output)
        if match:
            return int(match.group(1))
        return 0


__all__ = ["BaselineCollector", "BaselineSnapshot"]
