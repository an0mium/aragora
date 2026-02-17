"""
Tests for harness integration in the implement executor.

Tests cover:
- HybridExecutor with use_harness=True delegates to ClaudeCodeHarness
- adapt_to_implement_result converts HarnessResult to TaskResult
- Harness errors produce failed TaskResults
- Fallback behavior: harness only on first attempt, agent on retry
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses.adapter import adapt_to_implement_result
from aragora.harnesses.base import AnalysisType, HarnessResult
from aragora.implement.types import ImplementTask, TaskResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task():
    """A standard implementation task."""
    return ImplementTask(
        id="task-001",
        description="Add logging to utils module",
        files=["aragora/utils.py"],
        complexity="simple",
    )


@pytest.fixture()
def successful_harness_result():
    """A successful HarnessResult."""
    return HarnessResult(
        harness="claude-code",
        analysis_type=AnalysisType.GENERAL,
        success=True,
        findings=[],
        duration_seconds=12.5,
        raw_output="Implementation complete",
    )


@pytest.fixture()
def failed_harness_result():
    """A failed HarnessResult."""
    return HarnessResult(
        harness="claude-code",
        analysis_type=AnalysisType.GENERAL,
        success=False,
        findings=[],
        duration_seconds=5.0,
        error_message="Timeout waiting for response",
    )


# ---------------------------------------------------------------------------
# adapt_to_implement_result
# ---------------------------------------------------------------------------


class TestAdaptToImplementResult:
    """Test the HarnessResult -> TaskResult adapter."""

    def test_successful_result(self, successful_harness_result):
        result = adapt_to_implement_result(
            successful_harness_result,
            task_id="task-001",
            diff="--- a/file.py\n+++ b/file.py",
        )

        assert isinstance(result, TaskResult)
        assert result.task_id == "task-001"
        assert result.success is True
        assert result.diff == "--- a/file.py\n+++ b/file.py"
        assert result.error is None
        assert result.model_used == "harness:claude-code"
        assert result.duration_seconds == 12.5

    def test_failed_result(self, failed_harness_result):
        result = adapt_to_implement_result(
            failed_harness_result,
            task_id="task-002",
        )

        assert result.task_id == "task-002"
        assert result.success is False
        assert result.error == "Timeout waiting for response"
        assert result.model_used == "harness:claude-code"
        assert result.duration_seconds == 5.0

    def test_failed_with_error_output_fallback(self):
        hr = HarnessResult(
            harness="claude-code",
            analysis_type=AnalysisType.GENERAL,
            success=False,
            findings=[],
            error_message="",
            error_output="stderr: process exited with code 1",
        )
        result = adapt_to_implement_result(hr, task_id="task-003")

        assert result.success is False
        assert result.error == "stderr: process exited with code 1"

    def test_failed_with_generic_fallback(self):
        hr = HarnessResult(
            harness="claude-code",
            analysis_type=AnalysisType.GENERAL,
            success=False,
            findings=[],
        )
        result = adapt_to_implement_result(hr, task_id="task-004")

        assert result.success is False
        assert result.error == "Harness execution failed"

    def test_empty_diff_defaults(self, successful_harness_result):
        result = adapt_to_implement_result(
            successful_harness_result,
            task_id="task-005",
        )

        assert result.diff == ""

    def test_model_used_includes_harness_name(self):
        hr = HarnessResult(
            harness="codex",
            analysis_type=AnalysisType.GENERAL,
            success=True,
            findings=[],
        )
        result = adapt_to_implement_result(hr, task_id="task-006")

        assert result.model_used == "harness:codex"


# ---------------------------------------------------------------------------
# HybridExecutor.use_harness flag
# ---------------------------------------------------------------------------


class TestHybridExecutorHarnessFlag:
    """Test that use_harness flag is stored correctly."""

    def test_default_false(self, tmp_path):
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path)
        assert executor.use_harness is False

    def test_set_true(self, tmp_path):
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)
        assert executor.use_harness is True


# ---------------------------------------------------------------------------
# HybridExecutor._execute_via_harness
# ---------------------------------------------------------------------------


class TestExecuteViaHarness:
    """Test the harness delegation path."""

    @pytest.mark.asyncio
    async def test_delegates_to_harness(self, tmp_path, task, successful_harness_result):
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        mock_harness = AsyncMock()
        mock_harness.initialize = AsyncMock(return_value=True)
        mock_harness.execute_implementation = AsyncMock(return_value=("implementation output", ""))

        with (
            patch(
                "aragora.harnesses.claude_code.ClaudeCodeHarness",
                return_value=mock_harness,
            ),
            patch(
                "aragora.harnesses.claude_code.ClaudeCodeConfig",
            ),
            patch.object(executor, "_get_git_diff", return_value="some diff"),
        ):
            result = await executor._execute_via_harness(task)

        assert result.success is True
        assert result.diff == "some diff"
        assert result.task_id == "task-001"
        mock_harness.initialize.assert_awaited_once()
        mock_harness.execute_implementation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_harness_error_returns_failure(self, tmp_path, task):
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        mock_harness = AsyncMock()
        mock_harness.initialize = AsyncMock(side_effect=RuntimeError("CLI not found"))

        with (
            patch(
                "aragora.harnesses.claude_code.ClaudeCodeHarness",
                return_value=mock_harness,
            ),
            patch(
                "aragora.harnesses.claude_code.ClaudeCodeConfig",
            ),
        ):
            result = await executor._execute_via_harness(task)

        assert result.success is False
        assert "Harness error" in result.error
        assert "CLI not found" in result.error
        assert result.model_used == "harness:claude-code"


# ---------------------------------------------------------------------------
# execute_task harness routing
# ---------------------------------------------------------------------------


class TestExecuteTaskHarnessRouting:
    """Test that execute_task routes to harness on first attempt."""

    @pytest.mark.asyncio
    async def test_first_attempt_uses_harness(self, tmp_path, task, successful_harness_result):
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        # Mock _execute_via_harness directly to test the routing logic
        mock_result = TaskResult(
            task_id="task-001",
            success=True,
            diff="diff",
            model_used="harness:claude-code",
            duration_seconds=12.5,
        )
        with patch.object(executor, "_execute_via_harness", new=AsyncMock(return_value=mock_result)):
            result = await executor.execute_task(task, attempt=1)

        assert result.success is True
        assert result.model_used == "harness:claude-code"

    @pytest.mark.asyncio
    async def test_retry_skips_harness(self, tmp_path, task):
        """On retry (attempt > 1), use normal agent path instead of harness."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "claude-implementer"
        mock_agent.timeout = 300

        with (
            patch.object(
                HybridExecutor,
                "claude",
                new_callable=lambda: property(lambda self: mock_agent),
            ),
            patch.object(executor, "_get_git_diff", return_value="retry diff"),
            patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
        ):
            result = await executor.execute_task(task, attempt=2)

        assert result.success is True
        assert result.model_used == "claude"

    @pytest.mark.asyncio
    async def test_fallback_skips_harness(self, tmp_path, task):
        """When use_fallback=True, skip harness even on first attempt."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "codex-specialist"
        mock_agent.timeout = 300

        with (
            patch.object(
                HybridExecutor,
                "codex",
                new_callable=lambda: property(lambda self: mock_agent),
            ),
            patch.object(executor, "_get_git_diff", return_value="fallback diff"),
            patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
        ):
            result = await executor.execute_task(task, attempt=1, use_fallback=True)

        assert result.success is True
        assert "codex" in result.model_used

    @pytest.mark.asyncio
    async def test_harness_disabled_uses_agent(self, tmp_path, task):
        """When use_harness=False, always use agent path."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=False, sandbox_mode=False)

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "claude-implementer"
        mock_agent.timeout = 300

        with (
            patch.object(
                HybridExecutor,
                "claude",
                new_callable=lambda: property(lambda self: mock_agent),
            ),
            patch.object(executor, "_get_git_diff", return_value="agent diff"),
            patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
        ):
            result = await executor.execute_task(task, attempt=1)

        assert result.success is True
        assert result.model_used == "claude"

    @pytest.mark.asyncio
    async def test_agent_override_skips_harness(self, tmp_path, task):
        """When agent_override is provided, skip harness."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=tmp_path, use_harness=True)

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "custom-agent"
        mock_agent.timeout = 300

        with (
            patch.object(executor, "_get_git_diff", return_value="override diff"),
            patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
        ):
            result = await executor.execute_task(
                task, attempt=1, agent_override=mock_agent, model_label="custom"
            )

        assert result.success is True
        assert result.model_used == "custom"
