"""Batch processing for TestFixer.

Groups multiple test failures into batches by module path and category,
then processes each batch through the existing single-fix loop. Adds
ELO-based agent selection, post-fix bug checking, and cross-test
impact analysis on top of the existing TestFixerOrchestrator.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from aragora.events.types import StreamEventType
from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory
from aragora.nomic.testfixer.orchestrator import FixAttempt, FixLoopConfig, LoopStatus

if TYPE_CHECKING:
    from aragora.nomic.testfixer.orchestrator import TestFixerOrchestrator
    from aragora.nomic.testfixer.runner import TestFailure, TestResult

logger = logging.getLogger(__name__)

# Category severity ordering for batch prioritization (higher = more severe)
_CATEGORY_SEVERITY: dict[FailureCategory, int] = {
    FailureCategory.IMPL_BUG: 10,
    FailureCategory.IMPL_MISSING: 9,
    FailureCategory.IMPL_TYPE: 8,
    FailureCategory.IMPL_API_CHANGE: 8,
    FailureCategory.RACE_CONDITION: 7,
    FailureCategory.TEST_IMPORT: 6,
    FailureCategory.TEST_SETUP: 5,
    FailureCategory.TEST_ASYNC: 5,
    FailureCategory.TEST_MOCK: 4,
    FailureCategory.TEST_ASSERTION: 3,
    FailureCategory.ENV_DEPENDENCY: 2,
    FailureCategory.ENV_CONFIG: 2,
    FailureCategory.ENV_RESOURCE: 2,
    FailureCategory.FLAKY: 1,
    FailureCategory.UNKNOWN: 0,
}


@dataclass
class FailureBatch:
    """A group of related test failures to fix together."""

    id: str
    failures: list[TestFailure]
    analyses: list[FailureAnalysis]
    category: FailureCategory
    module_path: str
    priority: int = 0

    @property
    def size(self) -> int:
        return len(self.failures)


class BatchStatus(str, Enum):
    """Status of a batch fix attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    REVERTED = "reverted"
    SKIPPED = "skipped"


@dataclass
class BatchResult:
    """Result of processing a single batch."""

    batch: FailureBatch
    status: BatchStatus
    attempts: list[FixAttempt] = field(default_factory=list)
    fixes_attempted: int = 0
    fixes_successful: int = 0
    fixes_reverted: int = 0
    bug_check_passed: bool = True
    impact_check_passed: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class BatchFixResult:
    """Result of the full batch fix process."""

    status: LoopStatus
    total_failures_found: int = 0
    batches_created: int = 0
    batches_processed: int = 0
    fixes_attempted: int = 0
    fixes_successful: int = 0
    fixes_reverted: int = 0
    bug_checks_passed: int = 0
    bug_checks_failed: int = 0
    regressions_detected: int = 0
    batch_results: list[BatchResult] = field(default_factory=list)
    final_test_result: TestResult | None = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None

    def summary(self) -> str:
        duration = 0.0
        if self.finished_at:
            duration = (self.finished_at - self.started_at).total_seconds()
        return (
            f"BatchFixer {self.status.value}: "
            f"{self.fixes_successful}/{self.fixes_attempted} fixes successful "
            f"across {self.batches_processed}/{self.batches_created} batches "
            f"({self.total_failures_found} failures found, {duration:.1f}s)"
        )


class FailureGrouper:
    """Groups test failures into batches by module path and category."""

    def group(
        self,
        failures: list[TestFailure],
        analyses: list[FailureAnalysis],
        max_batch_size: int = 5,
    ) -> list[FailureBatch]:
        """Group failures into batches.

        Groups by: (1) common test directory prefix, (2) same
        FailureCategory, (3) overlapping involved_files. Splits groups
        exceeding max_batch_size.

        Args:
            failures: All test failures from the run.
            analyses: Corresponding analyses for each failure.
            max_batch_size: Maximum failures per batch.

        Returns:
            List of FailureBatch ordered by priority (highest first).
        """
        if len(failures) != len(analyses):
            raise ValueError(f"Mismatch: {len(failures)} failures vs {len(analyses)} analyses")

        # Build groups keyed by (directory_prefix, category)
        groups: dict[tuple[str, FailureCategory], list[tuple[TestFailure, FailureAnalysis]]] = {}

        for failure, analysis in zip(failures, analyses):
            prefix = self._directory_prefix(failure.test_file)
            category = analysis.category
            key = (prefix, category)
            groups.setdefault(key, []).append((failure, analysis))

        # Convert to batches, splitting large groups
        batches: list[FailureBatch] = []
        for (prefix, category), items in groups.items():
            for chunk_start in range(0, len(items), max_batch_size):
                chunk = items[chunk_start : chunk_start + max_batch_size]
                batch_failures = [f for f, _ in chunk]
                batch_analyses = [a for _, a in chunk]
                severity = _CATEGORY_SEVERITY.get(category, 0)
                priority = severity * 100 + len(batch_failures)

                batches.append(
                    FailureBatch(
                        id=uuid.uuid4().hex[:12],
                        failures=batch_failures,
                        analyses=batch_analyses,
                        category=category,
                        module_path=prefix,
                        priority=priority,
                    )
                )

        # Sort by priority descending (highest first)
        batches.sort(key=lambda b: b.priority, reverse=True)
        return batches

    @staticmethod
    def _directory_prefix(test_file: str) -> str:
        """Extract directory prefix from a test file path."""
        parts = test_file.replace("\\", "/").split("/")
        # Keep up to 2 directory levels: tests/handlers/ etc.
        if len(parts) > 2:
            return "/".join(parts[:2])
        elif len(parts) > 1:
            return parts[0]
        return ""


class BatchOrchestrator:
    """Coordinates batch fix processing atop TestFixerOrchestrator.

    Collects all failures, groups them into batches, optionally
    selects agents per-batch via ELO, and applies post-fix validation
    (bug checking, impact analysis).
    """

    def __init__(
        self,
        orchestrator: TestFixerOrchestrator,
        config: FixLoopConfig,
    ) -> None:
        self._orch = orchestrator
        self._config = config
        self._grouper = FailureGrouper()

        # Optional components initialized lazily
        self._agent_selector: Any = None
        self._bug_checker: Any = None
        self._impact_analyzer: Any = None

    async def run(self, max_iterations: int | None = None) -> BatchFixResult:
        """Run the batch fix loop.

        Algorithm:
        1. Run tests to collect ALL failures
        2. If all pass -> SUCCESS
        3. Analyze all failures
        4. Group into batches
        5. For each batch: fix failures, validate, revert if needed
        6. After all batches: run full suite for final state
        7. If failures remain and iterations < max: loop

        Args:
            max_iterations: Maximum outer-loop iterations.

        Returns:
            BatchFixResult with complete history.
        """
        max_iter = max_iterations or self._config.max_iterations
        result = BatchFixResult(status=LoopStatus.RUNNING)

        logger.info(
            "batch.start run_id=%s max_iterations=%d",
            self._orch.run_id,
            max_iter,
        )
        await self._emit_event(
            StreamEventType.TESTFIXER_BATCH_STARTED,
            {"max_iterations": max_iter},
        )

        try:
            for iteration in range(1, max_iter + 1):
                logger.info("batch.iteration iteration=%d/%d", iteration, max_iter)

                # Step 1: Run full test suite
                test_result = await self._orch.runner.run_full(
                    override_command=getattr(self._config, "batch_test_command", None),
                )
                result.final_test_result = test_result

                # Step 2: Check for success
                if test_result.success:
                    result.status = LoopStatus.SUCCESS
                    logger.info("batch.all_pass iteration=%d", iteration)
                    break

                if not test_result.failures:
                    logger.warning("batch.no_failures_parsed iteration=%d", iteration)
                    result.status = LoopStatus.ERROR
                    break

                result.total_failures_found = len(test_result.failures)

                # Step 3: Analyze all failures
                analyses: list[FailureAnalysis] = []
                for failure in test_result.failures:
                    analysis = await self._orch.analyzer.analyze(failure)
                    analyses.append(analysis)

                # Step 4: Group into batches
                max_batch_size = getattr(self._config, "max_batch_size", 5)
                batches = self._grouper.group(
                    test_result.failures, analyses, max_batch_size=max_batch_size
                )
                result.batches_created = len(batches)
                logger.info(
                    "batch.grouped batches=%d failures=%d",
                    len(batches),
                    len(test_result.failures),
                )

                # Step 5: Collect baseline failure names for impact analysis
                baseline_failure_names = [f.test_name for f in test_result.failures]

                # Step 6: Process each batch
                any_progress = False
                for batch in batches:
                    batch_result = await self._process_batch(batch, baseline_failure_names)
                    result.batch_results.append(batch_result)
                    result.batches_processed += 1
                    result.fixes_attempted += batch_result.fixes_attempted
                    result.fixes_successful += batch_result.fixes_successful
                    result.fixes_reverted += batch_result.fixes_reverted

                    if not batch_result.bug_check_passed:
                        result.bug_checks_failed += 1
                    else:
                        result.bug_checks_passed += 1

                    if not batch_result.impact_check_passed:
                        result.regressions_detected += 1

                    if batch_result.fixes_successful > 0:
                        any_progress = True

                # Step 7: Check for stuck state
                if not any_progress:
                    logger.info("batch.no_progress iteration=%d", iteration)
                    result.status = LoopStatus.STUCK
                    break

                # Run final test to get updated state
                final_result = await self._orch.runner.run_full(
                    override_command=getattr(self._config, "batch_test_command", None),
                )
                result.final_test_result = final_result

                if final_result.success:
                    result.status = LoopStatus.SUCCESS
                    break
            else:
                result.status = LoopStatus.MAX_ITERATIONS

        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("batch.error error=%s", exc)
            result.status = LoopStatus.ERROR

        result.finished_at = datetime.now()
        logger.info("batch.finish status=%s summary=%s", result.status.value, result.summary())

        await self._emit_event(
            StreamEventType.TESTFIXER_BATCH_COMPLETE,
            {
                "status": result.status.value,
                "batches_processed": result.batches_processed,
                "fixes_successful": result.fixes_successful,
                "fixes_attempted": result.fixes_attempted,
                "regressions_detected": result.regressions_detected,
            },
        )

        return result

    async def _process_batch(
        self,
        batch: FailureBatch,
        baseline_failure_names: list[str],
    ) -> BatchResult:
        """Process a single batch of related failures."""
        batch_result = BatchResult(batch=batch, status=BatchStatus.IN_PROGRESS)

        logger.info(
            "batch.process id=%s category=%s module=%s size=%d priority=%d",
            batch.id,
            batch.category.value,
            batch.module_path,
            batch.size,
            batch.priority,
        )

        # ELO-based agent selection
        enable_elo = getattr(self._config, "enable_elo_selection", False)
        original_proposer_generators = None
        if enable_elo:
            selector = self._get_agent_selector()
            if selector is not None:
                generators = selector.select_agents_for_category(batch.category)
                if generators:
                    original_proposer_generators = self._orch.proposer.generators
                    self._orch.proposer.generators = generators
                    batch_result.notes.append(f"ELO-selected agents for {batch.category.value}")

        try:
            # Fix each failure in the batch
            for failure in batch.failures:
                try:
                    attempt = await self._orch.run_single_fix(failure)
                    batch_result.attempts.append(attempt)
                    batch_result.fixes_attempted += 1
                    if attempt.success:
                        batch_result.fixes_successful += 1
                    elif attempt.applied and not attempt.success:
                        batch_result.fixes_reverted += 1
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.warning(
                        "batch.fix_error batch=%s test=%s error=%s",
                        batch.id,
                        failure.test_name,
                        exc,
                    )
                    batch_result.notes.append(f"Error fixing {failure.test_name}: {exc}")

            # Post-fix bug check
            enable_bug_check = getattr(self._config, "enable_bug_check", False)
            if enable_bug_check and batch_result.fixes_successful > 0:
                bug_passed = await self._run_bug_check(batch_result)
                batch_result.bug_check_passed = bug_passed
                if not bug_passed:
                    # Revert all applied patches in this batch
                    self._revert_batch(batch_result)
                    batch_result.status = BatchStatus.REVERTED
                    await self._emit_event(
                        StreamEventType.TESTFIXER_BUG_CHECK,
                        {
                            "batch_id": batch.id,
                            "passed": False,
                            "action": "reverted",
                        },
                    )
                else:
                    await self._emit_event(
                        StreamEventType.TESTFIXER_BUG_CHECK,
                        {"batch_id": batch.id, "passed": True},
                    )

            # Impact analysis
            enable_impact = getattr(self._config, "enable_impact_analysis", False)
            if (
                enable_impact
                and batch_result.fixes_successful > 0
                and batch_result.status != BatchStatus.REVERTED
            ):
                impact_passed = await self._run_impact_check(batch_result, baseline_failure_names)
                batch_result.impact_check_passed = impact_passed
                if not impact_passed:
                    self._revert_batch(batch_result)
                    batch_result.status = BatchStatus.REVERTED
                    await self._emit_event(
                        StreamEventType.TESTFIXER_IMPACT_CHECK,
                        {
                            "batch_id": batch.id,
                            "passed": False,
                            "action": "reverted",
                        },
                    )
                else:
                    await self._emit_event(
                        StreamEventType.TESTFIXER_IMPACT_CHECK,
                        {"batch_id": batch.id, "passed": True},
                    )

            if batch_result.status != BatchStatus.REVERTED:
                batch_result.status = BatchStatus.COMPLETE

        finally:
            # Restore original generators if swapped
            if original_proposer_generators is not None:
                self._orch.proposer.generators = original_proposer_generators

        return batch_result

    async def _run_bug_check(self, batch_result: BatchResult) -> bool:
        """Run bug detector on patches from successful attempts."""
        checker = self._get_bug_checker()
        if checker is None:
            return True

        for attempt in batch_result.attempts:
            if attempt.success and attempt.applied:
                check = checker.check_patches(attempt.proposal)
                if not check.passes:
                    batch_result.notes.append(f"Bug check failed: {check.summary}")
                    logger.warning(
                        "batch.bug_check_failed batch=%s summary=%s",
                        batch_result.batch.id,
                        check.summary,
                    )
                    return False
        return True

    async def _run_impact_check(
        self,
        batch_result: BatchResult,
        baseline_failure_names: list[str],
    ) -> bool:
        """Run impact analysis to detect regressions."""
        analyzer = self._get_impact_analyzer()
        if analyzer is None:
            return True

        impact = await analyzer.check_impact(
            baseline_failure_names,
            override_command=getattr(self._config, "batch_test_command", None),
        )
        if impact.has_regressions:
            names = [f.test_name for f in impact.new_failures[:5]]
            batch_result.notes.append(f"Regressions detected: {names}")
            logger.warning(
                "batch.regressions batch=%s new_failures=%d",
                batch_result.batch.id,
                len(impact.new_failures),
            )
            return False
        return True

    def _revert_batch(self, batch_result: BatchResult) -> None:
        """Revert all applied patches in a batch."""
        for attempt in batch_result.attempts:
            if attempt.applied and attempt.success:
                try:
                    attempt.proposal.revert_all(self._orch.repo_path)
                    batch_result.fixes_reverted += 1
                    batch_result.fixes_successful -= 1
                    logger.info(
                        "batch.revert test=%s",
                        attempt.failure.test_name,
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.error(
                        "batch.revert_error test=%s error=%s",
                        attempt.failure.test_name,
                        exc,
                    )

    def _get_agent_selector(self) -> Any:
        if self._agent_selector is not None:
            return self._agent_selector
        try:
            from aragora.nomic.testfixer.agent_selector import AgentSelector

            elo_system = None
            try:
                from aragora.ranking.elo import get_elo_store

                elo_system = get_elo_store()
            except (ImportError, AttributeError, RuntimeError):
                pass

            fallback = getattr(self._config, "elo_fallback_agents", None)
            self._agent_selector = AgentSelector(
                elo_system=elo_system,
                fallback_agents=fallback,
            )
            return self._agent_selector
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("batch.agent_selector_unavailable error=%s", exc)
            return None

    def _get_bug_checker(self) -> Any:
        if self._bug_checker is not None:
            return self._bug_checker
        try:
            from aragora.nomic.testfixer.bug_check import PostFixBugChecker

            self._bug_checker = PostFixBugChecker(self._orch.repo_path)
            return self._bug_checker
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("batch.bug_checker_unavailable error=%s", exc)
            return None

    def _get_impact_analyzer(self) -> Any:
        if self._impact_analyzer is not None:
            return self._impact_analyzer
        try:
            from aragora.nomic.testfixer.impact import ImpactAnalyzer

            self._impact_analyzer = ImpactAnalyzer(self._orch.runner)
            return self._impact_analyzer
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("batch.impact_analyzer_unavailable error=%s", exc)
            return None

    async def _emit_event(self, event_type: StreamEventType, data: dict[str, Any]) -> None:
        """Emit event through the orchestrator's emitter."""
        await self._orch._emit_event(event_type, data)
