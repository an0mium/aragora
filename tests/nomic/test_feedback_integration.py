"""Integration tests for TestFixer → FeedbackLoop wiring.

Tests the full pipeline:
    TestFailure → heuristic analysis → FeedbackLoop enrichment → rich hints

Uses real dataclass instances (not mocks) to verify the wiring between
testfixer components and the autonomous orchestrator's FeedbackLoop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.nomic.autonomous_orchestrator import FeedbackLoop
from aragora.nomic.testfixer.runner import TestFailure, TestResult, TestFramework


def _make_assignment(subtask_id: str = "test-1", agent_type: str = "claude"):
    """Create a minimal AgentAssignment for testing."""
    from aragora.nomic.autonomous_orchestrator import AgentAssignment, Track
    from aragora.nomic.task_decomposer import SubTask

    subtask = SubTask(
        id=subtask_id,
        title="Test subtask",
        description="A test subtask",
    )
    return AgentAssignment(
        subtask=subtask,
        track=Track.DEVELOPER,
        agent_type=agent_type,
    )


def _make_test_result(
    failures: list[TestFailure] | None = None,
    exit_code: int = 1,
) -> TestResult:
    """Create a TestResult with the given failures."""
    failures = failures or []
    return TestResult(
        command="pytest tests/ -x",
        exit_code=exit_code,
        stdout="FAILED tests/test_foo.py::test_bar",
        stderr="",
        total_tests=1,
        passed=0,
        failed=len(failures),
        failures=failures,
        framework=TestFramework.PYTEST,
    )


class TestRichHintExtraction:
    """Test that FeedbackLoop produces rich hints when TestResult is available."""

    def test_assertion_error_produces_rich_hints(self):
        """TestResult with an assertion failure → rich hints with category/target."""
        failure = TestFailure(
            test_name="test_bar",
            test_file="tests/test_foo.py",
            error_type="AssertionError",
            error_message="assert 1 == 2",
            stack_trace="tests/test_foo.py:10: AssertionError",
            line_number=10,
        )
        test_result = _make_test_result(failures=[failure])

        loop = FeedbackLoop(repo_path=Path("/tmp/fake-repo"))
        assignment = _make_assignment()

        result = loop.analyze_failure(
            assignment,
            {
                "type": "test_failure",
                "message": "Test failures",
                "test_result": test_result,
            },
        )

        assert result["action"] == "retry_implement"
        hints = result["hints"]

        # With testfixer available, hints should be a list of dicts
        assert isinstance(hints, list), f"Expected list of rich hints, got {type(hints)}"
        assert len(hints) == 1

        hint = hints[0]
        assert hint["test_name"] == "test_bar"
        assert hint["test_file"] == "tests/test_foo.py"
        assert hint["error_type"] == "AssertionError"
        assert "category" in hint
        assert "confidence" in hint
        assert "fix_target" in hint
        assert "suggested_approach" in hint

    def test_import_error_categorized_correctly(self):
        """ImportError failures should be categorized as IMPL_MISSING."""
        failure = TestFailure(
            test_name="test_import",
            test_file="tests/test_bar.py",
            error_type="ImportError",
            error_message="No module named 'missing_module'",
            stack_trace="ImportError: No module named 'missing_module'",
        )
        test_result = _make_test_result(failures=[failure])

        loop = FeedbackLoop(repo_path=Path("/tmp/fake-repo"))
        assignment = _make_assignment()

        result = loop.analyze_failure(
            assignment,
            {
                "type": "test_failure",
                "message": "Import failure",
                "test_result": test_result,
            },
        )

        hints = result["hints"]
        assert isinstance(hints, list)
        hint = hints[0]
        # ImportError should match impl_missing or env_dependency
        assert hint["category"] in ("impl_missing", "env_dependency")
        assert hint["confidence"] >= 0.8

    def test_multiple_failures_produces_multiple_hints(self):
        """Multiple failures should each get their own hint entry."""
        failures = [
            TestFailure(
                test_name=f"test_{i}",
                test_file=f"tests/test_{i}.py",
                error_type="AssertionError",
                error_message=f"assert {i} == {i + 1}",
                stack_trace=f"tests/test_{i}.py:{i}: AssertionError",
            )
            for i in range(3)
        ]
        test_result = _make_test_result(failures=failures)

        loop = FeedbackLoop(repo_path=Path("/tmp/fake-repo"))
        assignment = _make_assignment()

        result = loop.analyze_failure(
            assignment,
            {
                "type": "test_failure",
                "message": "Multiple failures",
                "test_result": test_result,
            },
        )

        hints = result["hints"]
        assert isinstance(hints, list)
        assert len(hints) == 3


class TestFallbackBehavior:
    """Test graceful degradation when testfixer is unavailable."""

    def test_fallback_without_test_result(self):
        """Without TestResult in error_info, falls back to basic string hints."""
        loop = FeedbackLoop()
        assignment = _make_assignment()

        result = loop.analyze_failure(
            assignment,
            {
                "type": "test_failure",
                "message": "AssertionError: expected True\nActual: False",
            },
        )

        assert result["action"] == "retry_implement"
        hints = result["hints"]
        # Without TestResult, hints should be a string from basic extraction
        assert isinstance(hints, str)
        assert "AssertionError" in hints or "Actual" in hints

    def test_fallback_with_testfixer_import_error(self):
        """When testfixer can't be imported, falls back gracefully."""
        failure = TestFailure(
            test_name="test_x",
            test_file="tests/test_x.py",
            error_type="AssertionError",
            error_message="assert False",
            stack_trace="",
        )
        test_result = _make_test_result(failures=[failure])

        loop = FeedbackLoop(repo_path=Path("/tmp/fake-repo"))
        assignment = _make_assignment()

        with patch.dict("sys.modules", {"aragora.nomic.testfixer.analyzer": None}):
            result = loop.analyze_failure(
                assignment,
                {
                    "type": "test_failure",
                    "message": "Test failed",
                    "test_result": test_result,
                },
            )

        assert result["action"] == "retry_implement"
        # Should still produce hints (either rich list or fallback string)
        assert result["hints"] is not None


class TestTargetedFixGuards:
    """Test guard conditions for _attempt_targeted_fix."""

    @pytest.mark.asyncio
    async def test_too_many_failures_skips_fix(self):
        """More than 5 failures should skip targeted fix."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/fake-repo"),
            require_human_approval=False,
        )

        failures = [
            TestFailure(
                test_name=f"test_{i}",
                test_file=f"tests/test_{i}.py",
                error_type="AssertionError",
                error_message=f"assert {i}",
                stack_trace="",
            )
            for i in range(6)
        ]
        test_result = _make_test_result(failures=failures)

        result = await orch._attempt_targeted_fix(
            _make_assignment(), test_result
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_low_confidence_skips_fix(self):
        """Failures with low heuristic confidence should skip fix."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/fake-repo"),
            require_human_approval=False,
        )

        # "unknown" error type won't match any high-confidence pattern
        failures = [
            TestFailure(
                test_name="test_weird",
                test_file="tests/test_weird.py",
                error_type="WeirdError",
                error_message="something unusual happened",
                stack_trace="",
            )
        ]
        test_result = _make_test_result(failures=failures)

        result = await orch._attempt_targeted_fix(
            _make_assignment(), test_result
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_no_failures_skips_fix(self):
        """Empty failure list should skip targeted fix."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/fake-repo"),
            require_human_approval=False,
        )

        test_result = _make_test_result(failures=[])
        result = await orch._attempt_targeted_fix(
            _make_assignment(), test_result
        )
        assert result is False


class TestTestPathsScoping:
    """Test that test_paths parameter limits pytest scope."""

    @pytest.fixture(autouse=True)
    def _fresh_event_loop(self):
        """Provide a fresh event loop to avoid pollution from prior async tests.

        Prior async tests may close the default event loop, causing
        asyncio.get_event_loop() to return a closed loop. Creating a
        dedicated loop per test avoids that problem.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    def test_verify_phase_passes_test_paths(self):
        """VerifyPhase._run_tests() should build command from test_paths."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(aragora_path=Path("/tmp/fake-repo"))

        # We can't easily test the full async run, but we can verify the
        # fallback path constructs the right command by checking _run_tests_raw
        # builds a proper cmd list with specific test paths

        async def check_raw():
            # _run_tests_raw will fail (no repo), but we verify it handles
            # test_paths parameter without crashing
            result = await phase._run_tests_raw(
                test_paths=["tests/specific/test_a.py", "tests/specific/test_b.py"]
            )
            # Should fail gracefully (no such directory)
            assert result["check"] == "tests"
            assert result["passed"] is False

        asyncio.run(check_raw())

    def test_feedback_loop_repo_path_propagation(self):
        """FeedbackLoop should accept and store repo_path."""
        repo = Path("/my/project")
        loop = FeedbackLoop(repo_path=repo)
        assert loop.repo_path == repo

    def test_feedback_loop_default_repo_path(self):
        """FeedbackLoop should default repo_path to None."""
        loop = FeedbackLoop()
        assert loop.repo_path is None
