"""Tests for batch processing orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory
from aragora.nomic.testfixer.batch import (
    BatchFixResult,
    BatchOrchestrator,
    BatchResult,
    BatchStatus,
    FailureBatch,
    FailureGrouper,
)
from aragora.nomic.testfixer.orchestrator import (
    FixAttempt,
    FixLoopConfig,
    LoopStatus,
)


@dataclass
class FakeFailure:
    test_name: str
    test_file: str
    error_type: str = "AssertionError"
    error_message: str = "failed"
    stack_trace: str = ""
    involved_files: list = field(default_factory=list)
    involved_functions: list = field(default_factory=list)
    line_number: int | None = None
    relevant_code: str = ""
    duration_seconds: float = 0.0


@dataclass
class FakeTestResult:
    failures: list = field(default_factory=list)
    success: bool = False
    total_tests: int = 10
    passed: int = 8
    failed: int = 2
    skipped: int = 0
    errors: int = 0
    exit_code: int = 1
    command: str = "pytest"
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 1.0


@dataclass
class FakeProposal:
    id: str = "prop1"
    patches: list = field(default_factory=list)
    post_debate_confidence: float = 0.8
    description: str = "fix"

    def revert_all(self, repo_path):
        pass

    def as_diff(self):
        return ""


# ---------------------------------------------------------------------------
# FailureGrouper tests
# ---------------------------------------------------------------------------


class TestFailureGrouper:
    def test_groups_by_directory_and_category(self):
        f1 = FakeFailure("test_a", "tests/handlers/test_auth.py")
        f2 = FakeFailure("test_b", "tests/handlers/test_user.py")
        f3 = FakeFailure("test_c", "tests/models/test_db.py")

        a1 = FailureAnalysis(failure=f1, category=FailureCategory.TEST_ASSERTION)
        a2 = FailureAnalysis(failure=f2, category=FailureCategory.TEST_ASSERTION)
        a3 = FailureAnalysis(failure=f3, category=FailureCategory.IMPL_BUG)

        grouper = FailureGrouper()
        batches = grouper.group([f1, f2, f3], [a1, a2, a3])

        assert len(batches) == 2
        # IMPL_BUG has higher severity -> first
        assert batches[0].category == FailureCategory.IMPL_BUG
        assert batches[1].category == FailureCategory.TEST_ASSERTION
        assert batches[1].size == 2

    def test_splits_large_groups(self):
        failures = [FakeFailure(f"test_{i}", "tests/handlers/test_auth.py") for i in range(7)]
        analyses = [
            FailureAnalysis(failure=f, category=FailureCategory.TEST_ASSERTION) for f in failures
        ]

        grouper = FailureGrouper()
        batches = grouper.group(failures, analyses, max_batch_size=3)

        assert len(batches) == 3  # 3 + 3 + 1
        total = sum(b.size for b in batches)
        assert total == 7

    def test_raises_on_mismatch(self):
        grouper = FailureGrouper()
        with pytest.raises(ValueError, match="Mismatch"):
            grouper.group(
                [FakeFailure("t1", "tests/a.py")],
                [],
            )

    def test_empty_input(self):
        grouper = FailureGrouper()
        batches = grouper.group([], [])
        assert batches == []

    def test_directory_prefix_extraction(self):
        grouper = FailureGrouper()
        assert grouper._directory_prefix("tests/handlers/test_auth.py") == "tests/handlers"
        assert grouper._directory_prefix("tests/test_top.py") == "tests"
        assert grouper._directory_prefix("test_root.py") == ""

    def test_priority_ordering(self):
        f1 = FakeFailure("test_a", "tests/a/test_a.py")
        f2 = FakeFailure("test_b", "tests/b/test_b.py")
        a1 = FailureAnalysis(failure=f1, category=FailureCategory.FLAKY)
        a2 = FailureAnalysis(failure=f2, category=FailureCategory.IMPL_BUG)

        grouper = FailureGrouper()
        batches = grouper.group([f1, f2], [a1, a2])

        assert batches[0].category == FailureCategory.IMPL_BUG
        assert batches[0].priority > batches[1].priority


# ---------------------------------------------------------------------------
# BatchFixResult tests
# ---------------------------------------------------------------------------


class TestBatchFixResult:
    def test_summary(self):
        result = BatchFixResult(
            status=LoopStatus.SUCCESS,
            total_failures_found=5,
            batches_created=2,
            batches_processed=2,
            fixes_attempted=5,
            fixes_successful=4,
        )
        result.finished_at = result.started_at
        summary = result.summary()
        assert "4/5" in summary
        assert "2/2 batches" in summary


# ---------------------------------------------------------------------------
# BatchOrchestrator tests
# ---------------------------------------------------------------------------


class TestBatchOrchestrator:
    def _make_orchestrator(self, config=None):
        orch = MagicMock()
        orch.run_id = "test-run"
        orch.repo_path = "/tmp/repo"
        orch.runner = MagicMock()
        orch.analyzer = MagicMock()
        orch.proposer = MagicMock()
        orch._emit_event = AsyncMock()
        return orch

    @pytest.mark.asyncio
    async def test_success_on_all_passing(self):
        config = FixLoopConfig()
        orch = self._make_orchestrator()
        orch.runner.run_full = AsyncMock(return_value=FakeTestResult(success=True, failures=[]))

        batch_orch = BatchOrchestrator(orch, config)
        result = await batch_orch.run(max_iterations=1)

        assert result.status == LoopStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_processes_batches(self):
        config = FixLoopConfig()
        orch = self._make_orchestrator()

        f1 = FakeFailure("test_a", "tests/handlers/test_a.py")

        # First call: failures; second call (final check): success
        orch.runner.run_full = AsyncMock(
            side_effect=[
                FakeTestResult(failures=[f1]),
                FakeTestResult(success=True, failures=[]),
            ]
        )
        orch.analyzer.analyze = AsyncMock(
            return_value=FailureAnalysis(failure=f1, category=FailureCategory.TEST_ASSERTION)
        )
        orch.run_single_fix = AsyncMock(
            return_value=FixAttempt(
                iteration=0,
                failure=f1,
                analysis=FailureAnalysis(failure=f1),
                proposal=FakeProposal(),
                applied=True,
                test_result_after=FakeTestResult(success=True),
                success=True,
            )
        )

        batch_orch = BatchOrchestrator(orch, config)
        result = await batch_orch.run(max_iterations=1)

        assert result.status == LoopStatus.SUCCESS
        assert result.fixes_successful == 1
        assert result.batches_processed == 1

    @pytest.mark.asyncio
    async def test_stuck_when_no_progress(self):
        config = FixLoopConfig()
        orch = self._make_orchestrator()

        f1 = FakeFailure("test_a", "tests/a/test_a.py")
        orch.runner.run_full = AsyncMock(return_value=FakeTestResult(failures=[f1]))
        orch.analyzer.analyze = AsyncMock(
            return_value=FailureAnalysis(failure=f1, category=FailureCategory.UNKNOWN)
        )
        orch.run_single_fix = AsyncMock(
            return_value=FixAttempt(
                iteration=0,
                failure=f1,
                analysis=FailureAnalysis(failure=f1),
                proposal=FakeProposal(),
                applied=False,
                test_result_after=None,
                success=False,
            )
        )

        batch_orch = BatchOrchestrator(orch, config)
        result = await batch_orch.run(max_iterations=2)

        assert result.status == LoopStatus.STUCK

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        config = FixLoopConfig(max_iterations=1)
        orch = self._make_orchestrator()

        f1 = FakeFailure("test_a", "tests/a/test_a.py")
        # Keeps failing with at least one success to avoid STUCK
        orch.runner.run_full = AsyncMock(
            side_effect=[
                FakeTestResult(failures=[f1, FakeFailure("test_b", "tests/a/test_b.py")]),
                FakeTestResult(failures=[f1]),  # final check still fails
            ]
        )
        orch.analyzer.analyze = AsyncMock(
            return_value=FailureAnalysis(failure=f1, category=FailureCategory.TEST_ASSERTION)
        )
        # First fix succeeds, second fails
        orch.run_single_fix = AsyncMock(
            side_effect=[
                FixAttempt(
                    iteration=0,
                    failure=f1,
                    analysis=FailureAnalysis(failure=f1),
                    proposal=FakeProposal(),
                    applied=True,
                    test_result_after=FakeTestResult(success=True),
                    success=True,
                ),
                FixAttempt(
                    iteration=0,
                    failure=f1,
                    analysis=FailureAnalysis(failure=f1),
                    proposal=FakeProposal(),
                    applied=False,
                    test_result_after=None,
                    success=False,
                ),
            ]
        )

        batch_orch = BatchOrchestrator(orch, config)
        result = await batch_orch.run(max_iterations=1)

        assert result.status == LoopStatus.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_emits_events(self):
        config = FixLoopConfig()
        orch = self._make_orchestrator()
        orch.runner.run_full = AsyncMock(return_value=FakeTestResult(success=True, failures=[]))

        batch_orch = BatchOrchestrator(orch, config)
        await batch_orch.run(max_iterations=1)

        # Should emit batch_started and batch_complete
        event_types = [call.args[0] for call in orch._emit_event.call_args_list]
        from aragora.events.types import StreamEventType

        assert StreamEventType.TESTFIXER_BATCH_STARTED in event_types
        assert StreamEventType.TESTFIXER_BATCH_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_bug_check_reverts_on_failure(self):
        config = FixLoopConfig()
        config.enable_bug_check = True
        orch = self._make_orchestrator()

        f1 = FakeFailure("test_a", "tests/a/test_a.py")
        orch.runner.run_full = AsyncMock(
            side_effect=[
                FakeTestResult(failures=[f1]),
                FakeTestResult(failures=[f1]),  # still fails after revert
            ]
        )
        orch.analyzer.analyze = AsyncMock(
            return_value=FailureAnalysis(failure=f1, category=FailureCategory.TEST_ASSERTION)
        )

        proposal = FakeProposal()
        proposal.revert_all = MagicMock()
        orch.run_single_fix = AsyncMock(
            return_value=FixAttempt(
                iteration=0,
                failure=f1,
                analysis=FailureAnalysis(failure=f1),
                proposal=proposal,
                applied=True,
                test_result_after=FakeTestResult(success=True),
                success=True,
            )
        )

        batch_orch = BatchOrchestrator(orch, config)

        # Make bug checker fail
        from aragora.nomic.testfixer.bug_check import BugCheckResult

        mock_checker = MagicMock()
        mock_checker.check_patches.return_value = BugCheckResult(
            passes=False,
            summary="critical bug found",
            new_bugs=["bug1"],
        )
        batch_orch._bug_checker = mock_checker

        result = await batch_orch.run(max_iterations=1)

        assert result.bug_checks_failed >= 1

    @pytest.mark.asyncio
    async def test_error_on_no_failures_parsed(self):
        config = FixLoopConfig()
        orch = self._make_orchestrator()
        orch.runner.run_full = AsyncMock(return_value=FakeTestResult(success=False, failures=[]))

        batch_orch = BatchOrchestrator(orch, config)
        result = await batch_orch.run(max_iterations=1)

        assert result.status == LoopStatus.ERROR
