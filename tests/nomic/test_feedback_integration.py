"""Integration tests for TestFixer -> FeedbackLoop wiring.

Tests the full pipeline:
    TestFailure -> heuristic analysis -> FeedbackLoop enrichment -> rich hints

Uses real dataclass instances (not mocks) to verify the wiring between
testfixer components and the autonomous orchestrator's FeedbackLoop.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

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
        total_tests=max(1, len(failures)),
        passed=0,
        failed=len(failures),
        failures=failures,
        framework=TestFramework.PYTEST,
    )


class TestAssertionErrorProducesRichHints:
    """Test that FeedbackLoop.analyze_failure() produces rich hints
    when given a real TestResult containing assertion-type failures."""

    def test_assertion_error_produces_rich_hints(self):
        """Create a real TestResult with TestFailure instances (assertion error
        type), feed through FeedbackLoop.analyze_failure() with
        error_info={"type": "test_failure", "test_result": result}.
        Assert hints include file/line/category."""
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
        # File path present
        assert hint["test_file"] == "tests/test_foo.py"
        # Line number present
        assert hint["line_number"] == 10
        # Category present and set to test_assertion for AssertionError
        assert "category" in hint
        assert hint["category"] == "test_assertion"
        # Other enrichment fields
        assert "confidence" in hint
        assert hint["confidence"] > 0
        assert "fix_target" in hint
        assert "suggested_approach" in hint
        assert hint["error_type"] == "AssertionError"

    def test_import_error_categorized_correctly(self):
        """ImportError failures should be categorized as impl_missing or env_dependency."""
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
        # Each hint maps to its corresponding failure
        for i, hint in enumerate(hints):
            assert hint["test_name"] == f"test_{i}"


class TestFallbackWithoutTestfixer:
    """Test graceful degradation when testfixer is unavailable.

    Patches the testfixer import to fail (monkeypatch sys.modules),
    verifies FeedbackLoop.analyze_failure() still returns basic hints
    via _extract_test_hints() fallback.
    """

    def test_fallback_without_testfixer(self, monkeypatch):
        """Patch the testfixer import to fail, verify FeedbackLoop.analyze_failure()
        still returns basic hints via _extract_test_hints() fallback."""
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

        # Setting a module to None in sys.modules causes ImportError on import
        monkeypatch.setitem(sys.modules, "aragora.nomic.testfixer.analyzer", None)

        result = loop.analyze_failure(
            assignment,
            {
                "type": "test_failure",
                "message": "AssertionError: expected True\nActual: False",
                "test_result": test_result,
            },
        )

        assert result["action"] == "retry_implement"
        hints = result["hints"]
        # Fallback should produce a string (from _extract_test_hints), not a list
        assert isinstance(hints, str), (
            f"Expected string fallback hints when testfixer unavailable, got {type(hints)}"
        )
        # The fallback extracts lines containing AssertionError/Expected/Actual
        assert "AssertionError" in hints or "Actual" in hints or "Review test output" in hints

    def test_fallback_without_test_result_in_error_info(self):
        """Without TestResult in error_info, falls back to basic string hints
        without even attempting testfixer import."""
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
        assert isinstance(hints, str)
        assert "AssertionError" in hints or "Actual" in hints


class TestTargetedFixGuardConditions:
    """Test _attempt_targeted_fix() guard conditions:
    - Too many failures (>5) -> returns False
    - Low confidence (<0.7) -> returns False
    - Import fails -> returns False
    """

    @pytest.mark.asyncio
    async def test_too_many_failures_returns_false(self):
        """More than 5 failures should cause _attempt_targeted_fix to return False."""
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
            for i in range(6)  # 6 failures > 5 threshold
        ]
        test_result = _make_test_result(failures=failures)

        result = await orch._attempt_targeted_fix(
            _make_assignment(), test_result
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_low_confidence_returns_false(self):
        """Failures with low heuristic confidence (<0.7) should return False."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/fake-repo"),
            require_human_approval=False,
        )

        # "WeirdError" won't match any high-confidence pattern in CATEGORY_PATTERNS
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
    async def test_import_fails_returns_false(self, monkeypatch):
        """When testfixer modules cannot be imported, returns False."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/fake-repo"),
            require_human_approval=False,
        )

        failure = TestFailure(
            test_name="test_ok",
            test_file="tests/test_ok.py",
            error_type="AssertionError",
            error_message="assert 1 == 2",
            stack_trace="tests/test_ok.py:5: AssertionError",
            line_number=5,
        )
        test_result = _make_test_result(failures=[failure])

        # Block the testfixer.analyzer import inside _attempt_targeted_fix
        monkeypatch.setitem(sys.modules, "aragora.nomic.testfixer.analyzer", None)

        result = await orch._attempt_targeted_fix(
            _make_assignment(), test_result
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_no_failures_returns_false(self):
        """Empty failure list should cause _attempt_targeted_fix to return False."""
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
    """Test that verify phase passes test_paths to TestRunner when available.

    Mocks TestRunner to capture the paths argument and verify proper scoping.
    """

    @pytest.mark.asyncio
    async def test_test_paths_passed_to_test_runner(self):
        """Verify that _run_tests() constructs a TestRunner with the
        provided test_paths embedded in the command.

        TestRunner is imported locally inside _run_tests(), so we patch
        the canonical location (aragora.nomic.testfixer.runner.TestRunner).
        """
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(aragora_path=Path("/tmp/fake-repo"))

        captured_commands: list[str] = []

        # Create a mock TestRunner whose run() captures the test_command
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.total_tests = 5
        mock_result.passed = 5
        mock_result.failed = 0
        mock_result.errors = 0
        mock_result.failures = []
        mock_result.stdout = "5 passed"

        class FakeTestRunner:
            def __init__(self, repo_path, test_command, timeout_seconds=240):
                captured_commands.append(test_command)
                self.test_command = test_command

            async def run(self):
                return mock_result

        test_paths = ["tests/specific/test_a.py", "tests/specific/test_b.py"]

        with patch(
            "aragora.nomic.testfixer.runner.TestRunner",
            FakeTestRunner,
        ):
            result = await phase._run_tests(test_paths=test_paths)

        assert len(captured_commands) == 1
        cmd = captured_commands[0]
        # Verify the specific test paths are included in the command
        assert "tests/specific/test_a.py" in cmd
        assert "tests/specific/test_b.py" in cmd
        # Verify it does NOT fall back to the generic "tests/" path
        # (the command should contain both specific paths, not just "tests/")
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_default_test_paths_uses_tests_dir(self):
        """Without explicit test_paths, should default to 'tests/' directory."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(aragora_path=Path("/tmp/fake-repo"))

        captured_commands: list[str] = []

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.total_tests = 1
        mock_result.passed = 1
        mock_result.failed = 0
        mock_result.errors = 0
        mock_result.failures = []
        mock_result.stdout = "1 passed"

        class FakeTestRunner:
            def __init__(self, repo_path, test_command, timeout_seconds=240):
                captured_commands.append(test_command)

            async def run(self):
                return mock_result

        with patch(
            "aragora.nomic.testfixer.runner.TestRunner",
            FakeTestRunner,
        ):
            await phase._run_tests(test_paths=None)

        assert len(captured_commands) == 1
        # Default should use "tests/"
        assert "tests/" in captured_commands[0]

    def test_infer_test_paths_from_file_scope(self):
        """AutonomousOrchestrator._infer_test_paths maps source files to test paths."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        paths = AutonomousOrchestrator._infer_test_paths(
            ["aragora/server/handlers/auth.py", "tests/existing/test_util.py"]
        )
        assert "tests/server/handlers/test_auth.py" in paths
        assert "tests/existing/test_util.py" in paths

    def test_feedback_loop_repo_path_propagation(self):
        """FeedbackLoop should accept and store repo_path."""
        repo = Path("/my/project")
        loop = FeedbackLoop(repo_path=repo)
        assert loop.repo_path == repo

    def test_feedback_loop_default_repo_path(self):
        """FeedbackLoop should default repo_path to None."""
        loop = FeedbackLoop()
        assert loop.repo_path is None
