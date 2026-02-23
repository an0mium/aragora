"""Tests for targeted re-test and PatternLearner integration in run_fix_loop()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.testfixer.orchestrator import TestFixerOrchestrator


def _make_failure(test_file="tests/test_foo.py", test_name="test_bar", error_type="AssertionError"):
    """Create a mock TestFailure."""
    f = MagicMock()
    f.test_file = test_file
    f.test_name = test_name
    f.error_type = error_type
    f.error_message = "assert 1 == 2"
    f.traceback = ""
    return f


def _make_analysis():
    """Create a mock FailureAnalysis."""
    a = MagicMock()
    a.category.value = "assertion"
    a.fix_target.value = "source"
    a.confidence = 0.8
    a.root_cause_file = "foo.py"
    a.suggested_approach = "Fix the assertion"
    a.analysis_notes = []
    return a


def _make_proposal(patches=True, confidence=0.9):
    """Create a mock PatchProposal."""
    p = MagicMock()
    p.id = "prop_1"
    p.post_debate_confidence = confidence
    p.patches = [MagicMock()] if patches else []
    p.apply_all.return_value = True
    p.description = "Fix assertion"
    p.as_diff.return_value = "diff content"
    p.status = MagicMock()
    return p


def _make_test_result(success=True, first_failure=None):
    """Create a mock TestResult."""
    r = MagicMock()
    r.success = success
    r.first_failure = first_failure
    r.summary.return_value = "1 passed"
    r.diagnostics = None
    return r


@pytest.fixture
def orchestrator(tmp_path):
    """Create a TestFixerOrchestrator with mocked dependencies."""
    config = MagicMock()
    config.max_iterations = 3
    config.max_same_failure = 5
    config.min_confidence_to_apply = 0.5
    config.min_confidence_for_auto = 0.7
    config.on_fix_proposed = None
    config.on_fix_applied = None
    config.on_iteration_complete = None
    config.revert_on_failure = False
    config.stop_on_first_success = True
    config.attempt_store = None
    config.enable_pattern_learning = True
    config.pattern_store_path = None

    orch = TestFixerOrchestrator.__new__(TestFixerOrchestrator)
    orch.repo_path = tmp_path
    orch.test_command = "pytest tests/ -q"
    orch.config = config
    orch.run_id = "test-run"
    orch.analyzer = AsyncMock()
    orch.proposer = AsyncMock()
    orch.runner = AsyncMock()
    orch.pattern_learner = MagicMock()
    orch._failure_history = []
    orch._applied_patches = []
    orch._event_emitter = None

    async def noop_emit(*args, **kwargs):
        pass

    orch._emit_event = noop_emit
    orch._save_attempt = AsyncMock()
    orch._attempt_log_dir = tmp_path / "attempts"
    orch._attempt_log_dir.mkdir(exist_ok=True)
    return orch


class TestPatternLearnerInFixLoop:
    """Tests that PatternLearner is called during run_fix_loop()."""

    @pytest.mark.asyncio
    async def test_pattern_learner_called_during_analysis(self, orchestrator):
        """PatternLearner.suggest_heuristic is called after analysis."""
        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)
        test_result = _make_test_result(success=True)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = test_result
        orchestrator.pattern_learner.suggest_heuristic.return_value = "Try patching the mock"

        result = await orchestrator.run_fix_loop(max_iterations=1)

        orchestrator.pattern_learner.suggest_heuristic.assert_called_once_with(analysis)
        assert "pattern_suggestion" in analysis.analysis_notes
        assert "Try patching the mock" in analysis.suggested_approach

    @pytest.mark.asyncio
    async def test_pattern_learner_error_does_not_crash(self, orchestrator):
        """PatternLearner errors are caught gracefully."""
        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)
        test_result = _make_test_result(success=True)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = test_result
        orchestrator.pattern_learner.suggest_heuristic.side_effect = RuntimeError("DB unavailable")

        result = await orchestrator.run_fix_loop(max_iterations=1)

        # Should not crash, and no pattern_suggestion note added
        assert "pattern_suggestion" not in analysis.analysis_notes

    @pytest.mark.asyncio
    async def test_pattern_learner_none_suggestion_ignored(self, orchestrator):
        """None suggestion from PatternLearner is ignored."""
        failure = _make_failure()
        analysis = _make_analysis()
        original_approach = analysis.suggested_approach
        proposal = _make_proposal(confidence=0.9)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = _make_test_result(success=True)
        orchestrator.pattern_learner.suggest_heuristic.return_value = None

        await orchestrator.run_fix_loop(max_iterations=1)

        assert "pattern_suggestion" not in analysis.analysis_notes

    @pytest.mark.asyncio
    async def test_no_pattern_learner_skips_gracefully(self, orchestrator):
        """When pattern_learner is None, loop runs without it."""
        orchestrator.pattern_learner = None

        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = _make_test_result(success=True)

        result = await orchestrator.run_fix_loop(max_iterations=1)
        # Should complete without error
        assert result is not None


class TestTargetedRetest:
    """Tests for targeted test re-run before full suite."""

    @pytest.mark.asyncio
    async def test_targeted_test_called_before_full_suite(self, orchestrator):
        """Targeted single test is run before the full suite."""
        failure = _make_failure(test_file="tests/test_foo.py", test_name="test_bar")
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = _make_test_result(success=True)
        orchestrator.pattern_learner.suggest_heuristic.return_value = None

        await orchestrator.run_fix_loop(max_iterations=1)

        orchestrator.runner.run_single_test.assert_called_with("tests/test_foo.py::test_bar")

    @pytest.mark.asyncio
    async def test_full_suite_skipped_when_targeted_fails(self, orchestrator):
        """Full suite is not run when the targeted test still fails."""
        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)

        targeted_fail = _make_test_result(success=False, first_failure=failure)
        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = targeted_fail
        orchestrator.pattern_learner.suggest_heuristic.return_value = None

        await orchestrator.run_fix_loop(max_iterations=1)

        # run() should only be called once (initial), not again after failed targeted test
        # The initial run() is called to get first_failure
        assert orchestrator.runner.run.call_count == 1

    @pytest.mark.asyncio
    async def test_full_suite_runs_after_targeted_success(self, orchestrator):
        """Full suite runs when targeted test passes (check for regressions)."""
        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)

        orchestrator.runner.run.return_value = _make_test_result(
            success=False, first_failure=failure
        )
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = _make_test_result(success=True)
        orchestrator.pattern_learner.suggest_heuristic.return_value = None

        # After targeted passes, full suite should be called
        # run() called: 1) initial, 2) after targeted success
        await orchestrator.run_fix_loop(max_iterations=1)

        assert orchestrator.runner.run.call_count == 2

    @pytest.mark.asyncio
    async def test_fix_success_detected_after_full_suite(self, orchestrator):
        """Fix is reported as success when both targeted and full suite pass."""
        failure = _make_failure()
        analysis = _make_analysis()
        proposal = _make_proposal(confidence=0.9)

        # Initial run fails, targeted passes, full suite passes
        orchestrator.runner.run.side_effect = [
            _make_test_result(success=False, first_failure=failure),
            _make_test_result(success=True),
        ]
        orchestrator.analyzer.analyze.return_value = analysis
        orchestrator.proposer.propose_fix.return_value = proposal
        orchestrator.runner.run_single_test.return_value = _make_test_result(success=True)
        orchestrator.pattern_learner.suggest_heuristic.return_value = None

        result = await orchestrator.run_fix_loop(max_iterations=1)

        assert result.fixes_successful >= 1
