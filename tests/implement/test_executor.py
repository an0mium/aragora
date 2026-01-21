"""
Tests for implement executor module.

Tests cover:
- HybridExecutor initialization
- Agent selection logic
- Timeout calculation
- Prompt building
- Task execution (mocked)
- Retry and fallback logic
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.implement.executor import HybridExecutor, TASK_PROMPT_TEMPLATE
from aragora.implement.types import ImplementTask, TaskResult


class TestHybridExecutorInit:
    """Tests for HybridExecutor initialization."""

    def test_basic_init(self, tmp_path):
        """Initializes with required parameters."""
        executor = HybridExecutor(repo_path=tmp_path)

        assert executor.repo_path == tmp_path
        assert executor.claude_timeout == 1200
        assert executor.codex_timeout == 1200
        assert executor.max_retries == 2

    def test_custom_timeouts(self, tmp_path):
        """Accepts custom timeout values."""
        executor = HybridExecutor(
            repo_path=tmp_path,
            claude_timeout=600,
            codex_timeout=900,
            max_retries=3,
        )

        assert executor.claude_timeout == 600
        assert executor.codex_timeout == 900
        assert executor.max_retries == 3

    def test_agents_lazy_initialized(self, tmp_path):
        """Agents are not created until accessed."""
        executor = HybridExecutor(repo_path=tmp_path)

        assert executor._claude is None
        assert executor._codex is None


class TestAgentSelection:
    """Tests for agent selection logic."""

    def test_always_selects_claude(self, tmp_path):
        """Always selects Claude for implementation."""
        executor = HybridExecutor(repo_path=tmp_path)

        # Mock claude property to avoid real agent creation
        with patch.object(HybridExecutor, 'claude', new_callable=lambda: property(lambda self: MagicMock())):
            for complexity in ["simple", "moderate", "complex"]:
                agent, name = executor._select_agent(complexity)
                assert name == "claude"


class TestTimeoutCalculation:
    """Tests for _get_task_timeout."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Executor for tests."""
        return HybridExecutor(repo_path=tmp_path)

    def test_simple_task_timeout(self, executor):
        """Simple tasks get 5 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Simple task",
            files=["a.py"],
            complexity="simple",
        )

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
            assert timeout == 300  # 5 min

    def test_moderate_task_timeout(self, executor):
        """Moderate tasks get 10 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Moderate task",
            files=["a.py"],
            complexity="moderate",
        )

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
            assert timeout == 600  # 10 min

    def test_complex_task_timeout(self, executor):
        """Complex tasks get 20 min base timeout."""
        task = ImplementTask(
            id="t1",
            description="Complex task",
            files=["a.py"],
            complexity="complex",
        )

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
            assert timeout == 1200  # 20 min

    def test_file_count_adds_time(self, executor):
        """Additional files add to timeout."""
        task = ImplementTask(
            id="t1",
            description="Multi-file task",
            files=["a.py", "b.py", "c.py"],
            complexity="simple",
        )

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
            # 300 base + 2 * 120 = 540
            assert timeout == 540

    def test_timeout_capped_at_30_min(self, executor):
        """Timeout is capped at 30 minutes."""
        task = ImplementTask(
            id="t1",
            description="Huge task",
            files=[f"file{i}.py" for i in range(20)],
            complexity="complex",
        )

        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
            assert timeout == 1800  # Max 30 min

    def test_disabled_complexity_timeout(self, executor):
        """Returns default timeout when feature disabled."""
        task = ImplementTask(
            id="t1",
            description="Task",
            files=["a.py"],
            complexity="simple",
        )

        # Patch the module-level constant directly
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", False):
            timeout = executor._get_task_timeout(task)
            assert timeout == executor.claude_timeout


class TestPromptBuilding:
    """Tests for _build_prompt."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Executor for tests."""
        return HybridExecutor(repo_path=tmp_path)

    def test_builds_prompt_with_files(self, executor):
        """Builds prompt with file list."""
        task = ImplementTask(
            id="t1",
            description="Add a function",
            files=["src/utils.py", "tests/test_utils.py"],
            complexity="moderate",
        )

        prompt = executor._build_prompt(task)

        assert "Add a function" in prompt
        assert "src/utils.py" in prompt
        assert "tests/test_utils.py" in prompt
        assert str(executor.repo_path) in prompt

    def test_builds_prompt_without_files(self, executor):
        """Builds prompt without specific files."""
        task = ImplementTask(
            id="t1",
            description="Implement feature",
            files=[],
            complexity="complex",
        )

        prompt = executor._build_prompt(task)

        assert "Implement feature" in prompt
        assert "determine from description" in prompt

    def test_prompt_includes_instructions(self, executor):
        """Prompt includes implementation instructions."""
        task = ImplementTask(
            id="t1",
            description="Task",
            files=["a.py"],
            complexity="simple",
        )

        prompt = executor._build_prompt(task)

        assert "type hints" in prompt.lower() or "docstrings" in prompt.lower()
        assert "existing" in prompt.lower()


class TestGitDiff:
    """Tests for _get_git_diff."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Executor for tests."""
        return HybridExecutor(repo_path=tmp_path)

    def test_returns_diff_output(self, executor):
        """Returns git diff output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=" 2 files changed")
            diff = executor._get_git_diff()
            assert "files changed" in diff

    def test_handles_timeout(self, executor):
        """Returns empty string on timeout."""
        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 180)
            diff = executor._get_git_diff()
            assert diff == ""

    def test_handles_error(self, executor):
        """Returns empty string on error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Git not found")
            diff = executor._get_git_diff()
            assert diff == ""


class TestTaskResultConstruction:
    """Tests for TaskResult construction in execute_task scenarios."""

    def test_success_result_structure(self):
        """TaskResult for success has correct structure."""
        result = TaskResult(
            task_id="t1",
            success=True,
            diff="+ changes",
            model_used="claude",
            duration_seconds=15.5,
        )

        assert result.task_id == "t1"
        assert result.success is True
        assert result.model_used == "claude"
        assert result.error is None

    def test_timeout_result_structure(self):
        """TaskResult for timeout has correct structure."""
        result = TaskResult(
            task_id="t1",
            success=False,
            error="Timeout: operation exceeded 600s",
            model_used="claude",
            duration_seconds=600.0,
        )

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_error_result_structure(self):
        """TaskResult for error has correct structure."""
        result = TaskResult(
            task_id="t1",
            success=False,
            error="API error: rate limited",
            model_used="codex",
            duration_seconds=5.0,
        )

        assert result.success is False
        assert result.error == "API error: rate limited"


class TestExecuteTaskWithRetry:
    """Tests for execute_task_with_retry method."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Executor for tests."""
        return HybridExecutor(repo_path=tmp_path, max_retries=3)

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self, executor):
        """Returns immediately on success."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )

        mock_execute = AsyncMock(return_value=TaskResult(task_id="t1", success=True))
        with patch.object(executor, 'execute_task', mock_execute):
            result = await executor.execute_task_with_retry(task)

        assert result.success is True
        assert mock_execute.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self, executor):
        """Retries on timeout error."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )

        call_count = 0

        async def mock_execute(task, attempt=1, use_fallback=False):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return TaskResult(task_id="t1", success=False, error="Timeout")
            return TaskResult(task_id="t1", success=True)

        with patch.object(executor, 'execute_task', side_effect=mock_execute):
            result = await executor.execute_task_with_retry(task)

        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_timeout(self, executor):
        """No retry on non-timeout errors."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py"],
            complexity="simple",
        )

        mock_execute = AsyncMock(return_value=TaskResult(
            task_id="t1",
            success=False,
            error="Invalid syntax"
        ))
        with patch.object(executor, 'execute_task', mock_execute):
            result = await executor.execute_task_with_retry(task)

        assert result.success is False
        # Only one call - no retry for non-timeout errors
        assert mock_execute.call_count == 1


class TestExecutePlan:
    """Tests for execute_plan method."""

    @pytest.fixture
    def executor(self, tmp_path):
        """Executor for tests."""
        return HybridExecutor(repo_path=tmp_path)

    @pytest.mark.asyncio
    async def test_skips_completed_tasks(self, executor):
        """Skips already completed tasks."""
        tasks = [
            ImplementTask(id="t1", description="Done", files=["a.py"], complexity="simple"),
            ImplementTask(id="t2", description="Todo", files=["b.py"], complexity="simple"),
        ]
        completed = {"t1"}

        mock_execute_retry = AsyncMock(return_value=TaskResult(task_id="t2", success=True))
        mock_execute = AsyncMock(return_value=TaskResult(task_id="t2", success=True))
        with patch.object(executor, 'execute_task_with_retry', mock_execute_retry):
            with patch.object(executor, 'execute_task', mock_execute):
                results = await executor.execute_plan(tasks, completed)

        # Only t2 should be executed
        assert mock_execute_retry.call_count == 1
        assert len(results) == 1
        assert results[0].task_id == "t2"

    @pytest.mark.asyncio
    async def test_calls_callback(self, executor):
        """Calls on_task_complete callback."""
        tasks = [
            ImplementTask(id="t1", description="Task", files=["a.py"], complexity="simple"),
        ]
        callback_calls = []

        def callback(task_id, result):
            callback_calls.append((task_id, result))

        mock_execute_retry = AsyncMock(return_value=TaskResult(task_id="t1", success=True))
        mock_execute = AsyncMock(return_value=TaskResult(task_id="t1", success=True))
        with patch.object(executor, 'execute_task_with_retry', mock_execute_retry):
            with patch.object(executor, 'execute_task', mock_execute):
                await executor.execute_plan(tasks, set(), on_task_complete=callback)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "t1"

    @pytest.mark.asyncio
    async def test_continues_after_failure_by_default(self, executor):
        """Continues execution after failure by default."""
        tasks = [
            ImplementTask(id="t1", description="Fail", files=["a.py"], complexity="simple"),
            ImplementTask(id="t2", description="Pass", files=["b.py"], complexity="simple"),
        ]

        retry_call_count = 0

        async def mock_execute_retry(task):
            nonlocal retry_call_count
            retry_call_count += 1
            if task.id == "t1":
                return TaskResult(task_id=task.id, success=False, error="Failed")
            return TaskResult(task_id=task.id, success=True)

        # Also mock execute_task to handle internal retry logic
        async def mock_execute(task, attempt=1, use_fallback=False):
            if task.id == "t1":
                return TaskResult(task_id=task.id, success=False, error="Failed")
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, 'execute_task_with_retry', side_effect=mock_execute_retry):
            with patch.object(executor, 'execute_task', side_effect=mock_execute):
                results = await executor.execute_plan(tasks, set())

        # Both tasks should be attempted in execute_task_with_retry
        assert retry_call_count == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stops_on_failure_when_requested(self, executor):
        """Stops on first failure when stop_on_failure=True."""
        tasks = [
            ImplementTask(id="t1", description="Fail", files=["a.py"], complexity="simple"),
            ImplementTask(id="t2", description="Skip", files=["b.py"], complexity="simple"),
        ]

        mock_execute_retry = AsyncMock(return_value=TaskResult(task_id="t1", success=False, error="Failed"))
        mock_execute = AsyncMock(return_value=TaskResult(task_id="t1", success=False, error="Failed"))
        with patch.object(executor, 'execute_task_with_retry', mock_execute_retry):
            with patch.object(executor, 'execute_task', mock_execute):
                results = await executor.execute_plan(tasks, set(), stop_on_failure=True)

        # Only first task should be attempted
        assert mock_execute_retry.call_count == 1
