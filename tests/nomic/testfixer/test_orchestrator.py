"""Tests for TestFixerOrchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.testfixer.orchestrator import (
    FixAttempt,
    FixLoopConfig,
    FixLoopResult,
    LoopStatus,
    TestFixerOrchestrator,
)
from aragora.nomic.testfixer.runner import TestFailure, TestResult
from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory, FixTarget
from aragora.nomic.testfixer.proposer import FilePatch, PatchProposal, PatchStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    test_name: str = "test_example",
    test_file: str = "tests/test_foo.py",
    error_type: str = "AssertionError",
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        test_file=test_file,
        error_type=error_type,
        error_message="expected 1 got 2",
        stack_trace="Traceback ...",
    )


def _make_analysis(
    failure: TestFailure | None = None,
    fix_target: FixTarget = FixTarget.TEST_FILE,
) -> FailureAnalysis:
    return FailureAnalysis(
        failure=failure or _make_failure(),
        category=FailureCategory.TEST_ASSERTION,
        confidence=0.8,
        fix_target=fix_target,
        root_cause="Wrong value",
        root_cause_file="tests/test_foo.py",
    )


def _make_proposal(
    analysis: FailureAnalysis | None = None,
    confidence: float = 0.75,
    patches: list[FilePatch] | None = None,
) -> PatchProposal:
    a = analysis or _make_analysis()
    return PatchProposal(
        id="fix_1",
        analysis=a,
        status=PatchStatus.PROPOSED,
        description="Fix assertion",
        post_debate_confidence=confidence,
        patches=patches or [],
    )


def _passing_result() -> TestResult:
    return TestResult(
        command="pytest",
        exit_code=0,
        stdout="1 passed",
        stderr="",
        total_tests=1,
        passed=1,
        failed=0,
    )


def _failing_result(
    failure: TestFailure | None = None,
) -> TestResult:
    f = failure or _make_failure()
    return TestResult(
        command="pytest",
        exit_code=1,
        stdout="1 failed",
        stderr="",
        total_tests=1,
        passed=0,
        failed=1,
        failures=[f],
    )


# ---------------------------------------------------------------------------
# FixLoopResult dataclass tests
# ---------------------------------------------------------------------------


class TestFixLoopResultSummary:
    def test_summary_success(self) -> None:
        now = datetime.now()
        result = FixLoopResult(
            status=LoopStatus.SUCCESS,
            started_at=now,
            finished_at=now + timedelta(seconds=12.5),
            total_iterations=3,
            fixes_applied=2,
            fixes_successful=2,
        )
        s = result.summary()
        assert "success" in s
        assert "2/2" in s
        assert "3 iterations" in s
        assert "12.5s" in s

    def test_summary_failure(self) -> None:
        now = datetime.now()
        result = FixLoopResult(
            status=LoopStatus.MAX_ITERATIONS,
            started_at=now,
            finished_at=now + timedelta(seconds=5),
            total_iterations=10,
            fixes_applied=3,
            fixes_successful=1,
        )
        s = result.summary()
        assert "max_iterations" in s
        assert "1/3" in s
        assert "10 iterations" in s


class TestFixLoopResultToDict:
    def test_to_dict_includes_all_keys(self) -> None:
        failure = _make_failure()
        analysis = _make_analysis(failure)
        proposal = _make_proposal(analysis)
        attempt = FixAttempt(
            iteration=1,
            failure=failure,
            analysis=analysis,
            proposal=proposal,
            applied=True,
            test_result_after=None,
            success=True,
        )
        now = datetime.now()
        result = FixLoopResult(
            status=LoopStatus.SUCCESS,
            started_at=now,
            finished_at=now + timedelta(seconds=1),
            total_iterations=1,
            fixes_applied=1,
            fixes_successful=1,
            fixes_reverted=0,
            attempts=[attempt],
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert "started_at" in d
        assert "finished_at" in d
        assert d["total_iterations"] == 1
        assert d["fixes_applied"] == 1
        assert d["fixes_successful"] == 1
        assert d["fixes_reverted"] == 0
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["iteration"] == 1
        assert d["attempts"][0]["failure"] == "test_example"
        assert d["attempts"][0]["category"] == "test_assertion"
        assert d["attempts"][0]["applied"] is True
        assert d["attempts"][0]["success"] is True


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def _build_orchestrator(
    tmp_path: Path,
    config: FixLoopConfig | None = None,
) -> TestFixerOrchestrator:
    """Build an orchestrator with mocked internal components."""
    orch = TestFixerOrchestrator(
        repo_path=tmp_path,
        test_command="pytest tests/ -q",
        config=config or FixLoopConfig(max_iterations=5, max_same_failure=3),
    )
    orch.runner = AsyncMock()
    orch.analyzer = AsyncMock()
    orch.proposer = AsyncMock()
    return orch


class TestRunFixLoopTestsPassOnFirstRun:
    @pytest.mark.asyncio
    async def test_success_immediately(self, tmp_path: Path) -> None:
        orch = _build_orchestrator(tmp_path)
        orch.runner.run = AsyncMock(return_value=_passing_result())

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.SUCCESS
        assert result.total_iterations == 1
        assert result.fixes_applied == 0


class TestRunFixLoopMaxIterations:
    @pytest.mark.asyncio
    async def test_hits_max_iterations(self, tmp_path: Path) -> None:
        config = FixLoopConfig(max_iterations=2, max_same_failure=10)
        orch = _build_orchestrator(tmp_path, config)

        f1 = _make_failure(test_name="test_a")
        f2 = _make_failure(test_name="test_b")
        orch.runner.run = AsyncMock(
            side_effect=[_failing_result(f1), _failing_result(f2), _failing_result(f2)]
        )
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis())
        orch.proposer.propose_fix = AsyncMock(return_value=_make_proposal(confidence=0.3))

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.MAX_ITERATIONS
        assert result.total_iterations == 2


class TestRunFixLoopStuck:
    @pytest.mark.asyncio
    async def test_same_failure_repeats(self, tmp_path: Path) -> None:
        config = FixLoopConfig(max_iterations=10, max_same_failure=3)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        orch.runner.run = AsyncMock(return_value=_failing_result(failure))
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=_make_proposal(confidence=0.3))

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.STUCK


class TestRunFixLoopHumanRequired:
    @pytest.mark.asyncio
    async def test_human_required_from_analysis(self, tmp_path: Path) -> None:
        orch = _build_orchestrator(tmp_path)

        failure = _make_failure()
        orch.runner.run = AsyncMock(return_value=_failing_result(failure))
        analysis = _make_analysis(failure, fix_target=FixTarget.HUMAN)
        orch.analyzer.analyze = AsyncMock(return_value=analysis)

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.HUMAN_REQUIRED
        assert len(result.attempts) == 1
        assert result.attempts[0].applied is False


class TestRunFixLoopConfidenceBelowThreshold:
    @pytest.mark.asyncio
    async def test_skips_low_confidence(self, tmp_path: Path) -> None:
        config = FixLoopConfig(max_iterations=1, min_confidence_to_apply=0.8, max_same_failure=10)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        orch.runner.run = AsyncMock(return_value=_failing_result(failure))
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=_make_proposal(confidence=0.4))

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.MAX_ITERATIONS
        assert result.fixes_applied == 0
        assert len(result.attempts) == 1
        assert "Confidence below threshold" in result.attempts[0].notes


class TestRunFixLoopNoPatches:
    @pytest.mark.asyncio
    async def test_skips_when_no_patches(self, tmp_path: Path) -> None:
        config = FixLoopConfig(max_iterations=1, max_same_failure=10)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        orch.runner.run = AsyncMock(return_value=_failing_result(failure))
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(
            return_value=_make_proposal(confidence=0.9, patches=[])
        )

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.MAX_ITERATIONS
        assert result.fixes_applied == 0
        assert "No patches generated" in result.attempts[0].notes


class TestRunFixLoopAppliedAndWorks:
    @pytest.mark.asyncio
    async def test_fix_applied_and_works(self, tmp_path: Path) -> None:
        config = FixLoopConfig(max_iterations=5, max_same_failure=5)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        file_patch = FilePatch(
            file_path="tests/test_foo.py",
            original_content="old",
            patched_content="new",
        )
        proposal = _make_proposal(confidence=0.9, patches=[file_patch])
        proposal.apply_all = MagicMock(return_value=True)

        # 3 run() calls: initial test, retest after fix, next iteration (passes → SUCCESS)
        orch.runner.run = AsyncMock(
            side_effect=[_failing_result(failure), _passing_result(), _passing_result()]
        )
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=proposal)

        result = await orch.run_fix_loop()

        assert result.status == LoopStatus.SUCCESS
        assert result.fixes_applied == 1
        assert result.fixes_successful == 1


class TestRunFixLoopFixFailsAndReverts:
    @pytest.mark.asyncio
    async def test_fix_fails_and_reverts(self, tmp_path: Path) -> None:
        config = FixLoopConfig(
            max_iterations=1,
            max_same_failure=10,
            revert_on_failure=True,
        )
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        file_patch = FilePatch(
            file_path="tests/test_foo.py",
            original_content="old",
            patched_content="new",
        )
        proposal = _make_proposal(confidence=0.9, patches=[file_patch])
        proposal.apply_all = MagicMock(return_value=True)
        proposal.revert_all = MagicMock(return_value=True)

        # Both runs fail with the same failure
        orch.runner.run = AsyncMock(
            side_effect=[_failing_result(failure), _failing_result(failure)]
        )
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=proposal)

        result = await orch.run_fix_loop()

        assert result.fixes_applied == 1
        assert result.fixes_reverted == 1
        proposal.revert_all.assert_called_once()


# ---------------------------------------------------------------------------
# run_single_fix tests
# ---------------------------------------------------------------------------


class TestRunSingleFixConfidenceBelow:
    @pytest.mark.asyncio
    async def test_confidence_below_threshold(self, tmp_path: Path) -> None:
        config = FixLoopConfig(min_confidence_to_apply=0.8)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=_make_proposal(confidence=0.3))

        attempt = await orch.run_single_fix(failure)

        assert attempt.applied is False
        assert attempt.success is False
        assert "Confidence below threshold" in attempt.notes


class TestRunSingleFixSuccess:
    @pytest.mark.asyncio
    async def test_successful_single_fix(self, tmp_path: Path) -> None:
        config = FixLoopConfig(min_confidence_to_apply=0.5)
        orch = _build_orchestrator(tmp_path, config)

        failure = _make_failure()
        file_patch = FilePatch(
            file_path="tests/test_foo.py",
            original_content="old",
            patched_content="new",
        )
        proposal = _make_proposal(confidence=0.9, patches=[file_patch])
        proposal.apply_all = MagicMock(return_value=True)

        orch.analyzer.analyze = AsyncMock(return_value=_make_analysis(failure))
        orch.proposer.propose_fix = AsyncMock(return_value=proposal)
        orch.runner.run_single_test = AsyncMock(return_value=_passing_result())

        attempt = await orch.run_single_fix(failure)

        assert attempt.applied is True
        assert attempt.success is True
