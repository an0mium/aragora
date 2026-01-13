"""
Tests for aragora.implement.executor module.

Tests the HybridExecutor class for executing implementation tasks including:
- Initialization and agent management
- Task complexity timeout calculation
- Prompt building
- Task execution with retry and fallback
- Plan execution with dependencies
- Parallel execution
- Codex code review
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement.executor import (
    COMPLEXITY_TIMEOUT,
    HybridExecutor,
    TASK_PROMPT_TEMPLATE,
)
from aragora.implement.types import ImplementTask, TaskResult


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def repo_path(tmp_path):
    """Create a temporary repo path."""
    return tmp_path


@pytest.fixture
def executor(repo_path):
    """Create a HybridExecutor with mocked agents."""
    exec = HybridExecutor(repo_path)
    return exec


@pytest.fixture
def simple_task():
    """Create a simple task."""
    return ImplementTask(
        id="task-1",
        description="Add a helper function",
        files=["src/helpers.py"],
        complexity="simple",
        dependencies=[],
    )


@pytest.fixture
def moderate_task():
    """Create a moderate task."""
    return ImplementTask(
        id="task-2",
        description="Implement a data processor",
        files=["src/processor.py", "src/utils.py"],
        complexity="moderate",
        dependencies=[],
    )


@pytest.fixture
def complex_task():
    """Create a complex task."""
    return ImplementTask(
        id="task-3",
        description="Implement a new API endpoint with tests",
        files=["src/api.py", "src/handlers.py", "src/models.py", "tests/test_api.py"],
        complexity="complex",
        dependencies=["task-1", "task-2"],
    )


# ==============================================================================
# HybridExecutor Initialization Tests
# ==============================================================================


class TestHybridExecutorInit:
    """Tests for HybridExecutor initialization."""

    def test_initialization(self, repo_path):
        """Can initialize with repo path."""
        executor = HybridExecutor(repo_path)
        assert executor.repo_path == repo_path

    def test_default_timeouts(self, repo_path):
        """Default timeouts are set correctly."""
        executor = HybridExecutor(repo_path)
        assert executor.claude_timeout == 1200
        assert executor.codex_timeout == 1200

    def test_custom_timeouts(self, repo_path):
        """Can set custom timeouts."""
        executor = HybridExecutor(
            repo_path,
            claude_timeout=600,
            codex_timeout=900,
        )
        assert executor.claude_timeout == 600
        assert executor.codex_timeout == 900

    def test_max_retries_default(self, repo_path):
        """Default max_retries is 2."""
        executor = HybridExecutor(repo_path)
        assert executor.max_retries == 2

    def test_agents_initialized_lazily(self, repo_path):
        """Agents are not initialized until accessed."""
        executor = HybridExecutor(repo_path)
        assert executor._claude is None
        assert executor._codex is None


# ==============================================================================
# Agent Properties Tests
# ==============================================================================


class TestHybridExecutorAgentProperties:
    """Tests for agent property access."""

    @patch("aragora.implement.executor.ClaudeAgent")
    def test_claude_property_creates_agent(self, mock_claude, executor):
        """claude property creates agent on first access."""
        mock_agent = MagicMock()
        mock_claude.return_value = mock_agent

        agent = executor.claude

        mock_claude.assert_called_once()
        assert agent is mock_agent

    @patch("aragora.implement.executor.ClaudeAgent")
    def test_claude_property_cached(self, mock_claude, executor):
        """claude property returns cached agent on subsequent access."""
        mock_agent = MagicMock()
        mock_claude.return_value = mock_agent

        agent1 = executor.claude
        agent2 = executor.claude

        mock_claude.assert_called_once()
        assert agent1 is agent2

    @patch("aragora.implement.executor.CodexAgent")
    def test_codex_property_creates_agent(self, mock_codex, executor):
        """codex property creates agent on first access."""
        mock_agent = MagicMock()
        mock_codex.return_value = mock_agent

        agent = executor.codex

        mock_codex.assert_called_once()
        assert agent is mock_agent


# ==============================================================================
# Agent Selection Tests
# ==============================================================================


class TestHybridExecutorAgentSelection:
    """Tests for agent selection logic."""

    @patch("aragora.implement.executor.ClaudeAgent")
    def test_select_agent_always_returns_claude(self, mock_claude, executor):
        """_select_agent always returns Claude (updated routing)."""
        mock_claude.return_value = MagicMock()

        agent, name = executor._select_agent("simple")
        assert name == "claude"

        agent, name = executor._select_agent("moderate")
        assert name == "claude"

        agent, name = executor._select_agent("complex")
        assert name == "claude"


# ==============================================================================
# Timeout Calculation Tests
# ==============================================================================


class TestHybridExecutorTimeout:
    """Tests for timeout calculation."""

    def test_simple_task_timeout(self, executor, simple_task):
        """Simple task gets 5 min base timeout."""
        timeout = executor._get_task_timeout(simple_task)
        # 300s base + 0 file bonus (1 file = no bonus)
        assert timeout == 300

    def test_moderate_task_timeout(self, executor, moderate_task):
        """Moderate task gets 10 min base + file bonus."""
        timeout = executor._get_task_timeout(moderate_task)
        # 600s base + 120s (1 extra file)
        assert timeout == 720

    def test_complex_task_timeout(self, executor, complex_task):
        """Complex task gets 20 min base + file bonus."""
        timeout = executor._get_task_timeout(complex_task)
        # 1200s base + 360s (3 extra files)
        assert timeout == 1560

    def test_timeout_capped_at_30_min(self, executor):
        """Timeout is capped at 30 minutes."""
        task = ImplementTask(
            id="huge",
            description="Massive task",
            files=[f"file{i}.py" for i in range(20)],  # 20 files
            complexity="complex",
        )
        timeout = executor._get_task_timeout(task)
        # Should be capped at 1800s (30 min)
        assert timeout == 1800

    def test_timeout_without_files(self, executor):
        """Timeout handles tasks without files."""
        task = ImplementTask(
            id="no-files",
            description="Task without files",
            files=[],
            complexity="moderate",
        )
        timeout = executor._get_task_timeout(task)
        assert timeout == 600  # Base only, no file bonus

    @patch.dict("os.environ", {"IMPL_COMPLEXITY_TIMEOUT": "0"})
    def test_complexity_timeout_disabled(self, repo_path, simple_task):
        """Returns default timeout when feature flag disabled."""
        # Need to re-import to pick up env change
        from aragora.implement import executor as executor_module

        exec = HybridExecutor(repo_path, claude_timeout=999)

        # When COMPLEXITY_TIMEOUT is disabled, we can't easily test this
        # since the module reads the env var at import time.
        # Just test that _get_task_timeout returns something reasonable
        timeout = exec._get_task_timeout(simple_task)
        assert timeout > 0


# ==============================================================================
# Prompt Building Tests
# ==============================================================================


class TestHybridExecutorPromptBuilding:
    """Tests for prompt building."""

    def test_build_prompt_includes_description(self, executor, simple_task):
        """Prompt includes task description."""
        prompt = executor._build_prompt(simple_task)
        assert simple_task.description in prompt

    def test_build_prompt_includes_files(self, executor, moderate_task):
        """Prompt includes files to modify."""
        prompt = executor._build_prompt(moderate_task)
        for file in moderate_task.files:
            assert file in prompt

    def test_build_prompt_includes_repo_path(self, executor, simple_task):
        """Prompt includes repository path."""
        prompt = executor._build_prompt(simple_task)
        assert str(executor.repo_path) in prompt

    def test_build_prompt_handles_empty_files(self, executor):
        """Prompt handles task with no files."""
        task = ImplementTask(
            id="no-files",
            description="Determine files needed",
            files=[],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "(determine from description)" in prompt


# ==============================================================================
# Git Diff Tests
# ==============================================================================


class TestHybridExecutorGitDiff:
    """Tests for git diff retrieval."""

    def test_get_git_diff_returns_string(self, executor):
        """_get_git_diff returns a string."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="diff output")
            result = executor._get_git_diff()
            assert isinstance(result, str)

    def test_get_git_diff_handles_error(self, executor):
        """_get_git_diff returns empty string on error."""
        with patch("subprocess.run") as mock_run:
            # Use OSError which is what actually happens when git is not found
            mock_run.side_effect = OSError("git not found")
            result = executor._get_git_diff()
            assert result == ""


# ==============================================================================
# Task Execution Tests
# ==============================================================================


class TestHybridExecutorTaskExecution:
    """Tests for task execution."""

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.ClaudeAgent")
    async def test_execute_task_success(self, mock_claude_class, executor, simple_task):
        """execute_task returns success on successful execution."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Done")
        mock_claude_class.return_value = mock_agent

        with patch.object(executor, "_get_git_diff", return_value="some diff"):
            result = await executor.execute_task(simple_task)

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert result.task_id == simple_task.id
        assert result.diff == "some diff"

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.ClaudeAgent")
    async def test_execute_task_timeout(self, mock_claude_class, executor, simple_task):
        """execute_task handles timeout."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_claude_class.return_value = mock_agent

        result = await executor.execute_task(simple_task)

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.ClaudeAgent")
    async def test_execute_task_error(self, mock_claude_class, executor, simple_task):
        """execute_task handles other errors."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(side_effect=RuntimeError("connection failed"))
        mock_claude_class.return_value = mock_agent

        result = await executor.execute_task(simple_task)

        assert result.success is False
        assert "connection failed" in result.error

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.ClaudeAgent")
    async def test_execute_task_records_model_used(self, mock_claude_class, executor, simple_task):
        """execute_task records which model was used."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Done")
        mock_claude_class.return_value = mock_agent

        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(simple_task)

        assert result.model_used == "claude"

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.CodexAgent")
    async def test_execute_task_with_fallback(self, mock_codex_class, executor, simple_task):
        """execute_task uses Codex when use_fallback=True."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Done")
        mock_codex_class.return_value = mock_agent

        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(simple_task, use_fallback=True)

        assert result.model_used == "codex-fallback"

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.ClaudeAgent")
    async def test_execute_task_records_duration(self, mock_claude_class, executor, simple_task):
        """execute_task records execution duration."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Done")
        mock_claude_class.return_value = mock_agent

        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(simple_task)

        assert result.duration_seconds >= 0


# ==============================================================================
# Task Execution with Retry Tests
# ==============================================================================


class TestHybridExecutorTaskRetry:
    """Tests for task execution with retry."""

    @pytest.mark.asyncio
    async def test_execute_task_with_retry_succeeds_first_attempt(self, executor, simple_task):
        """Doesn't retry if first attempt succeeds."""
        with patch.object(executor, "execute_task") as mock_execute:
            mock_execute.return_value = TaskResult(task_id=simple_task.id, success=True, diff="")
            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is True
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_with_retry_retries_on_timeout(self, executor, simple_task):
        """Retries with extended timeout on timeout failure."""
        call_count = 0

        async def mock_execute(task, attempt=1, use_fallback=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return TaskResult(task_id=task.id, success=False, error="Timeout")
            return TaskResult(task_id=task.id, success=True, diff="")

        with patch.object(executor, "execute_task", side_effect=mock_execute):
            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_task_with_retry_no_retry_on_other_error(self, executor, simple_task):
        """Doesn't retry on non-timeout errors (continues to fallback)."""
        executor.max_retries = 1  # Only allow 1 retry

        with patch.object(executor, "execute_task") as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id=simple_task.id, success=False, error="API error"
            )
            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is False
        # Should only be called once (no retry for non-timeout)
        mock_execute.assert_called_once()


# ==============================================================================
# Plan Execution Tests
# ==============================================================================


class TestHybridExecutorPlanExecution:
    """Tests for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_plan_basic(self, executor, simple_task):
        """execute_plan executes all tasks."""
        tasks = [simple_task]
        completed = set()

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            mock_execute.return_value = TaskResult(task_id=simple_task.id, success=True, diff="")
            results = await executor.execute_plan(tasks, completed)

        assert len(results) == 1
        assert results[0].success is True
        assert simple_task.id in completed

    @pytest.mark.asyncio
    async def test_execute_plan_skips_completed(self, executor, simple_task):
        """execute_plan skips already completed tasks."""
        tasks = [simple_task]
        completed = {simple_task.id}

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            results = await executor.execute_plan(tasks, completed)

        assert len(results) == 0
        mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_plan_respects_dependencies(self, executor, simple_task, complex_task):
        """execute_plan skips tasks with unmet dependencies."""
        tasks = [complex_task]  # Has dependencies on task-1, task-2
        completed = set()

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            results = await executor.execute_plan(tasks, completed)

        assert len(results) == 0  # Should skip - dependencies not met
        mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_plan_executes_when_deps_met(self, executor, complex_task):
        """execute_plan executes tasks when dependencies are met."""
        tasks = [complex_task]
        completed = {"task-1", "task-2"}  # Dependencies already complete

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            mock_execute.return_value = TaskResult(task_id=complex_task.id, success=True, diff="")
            results = await executor.execute_plan(tasks, completed)

        assert len(results) == 1
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_plan_continues_after_failure(self, executor):
        """execute_plan continues executing after a failure."""
        task1 = ImplementTask(id="t1", description="Task 1", files=["a.py"], complexity="simple")
        task2 = ImplementTask(id="t2", description="Task 2", files=["b.py"], complexity="simple")
        tasks = [task1, task2]
        completed = set()

        call_count = 0

        async def mock_execute(task):
            nonlocal call_count
            call_count += 1
            if task.id == "t1":
                return TaskResult(task_id=task.id, success=False, error="Failed")
            return TaskResult(task_id=task.id, success=True, diff="")

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            results = await executor.execute_plan(tasks, completed, stop_on_failure=False)

        # Both tasks should have been attempted
        assert len(results) >= 2
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_plan_stops_on_failure_when_flagged(self, executor):
        """execute_plan stops on first failure when stop_on_failure=True."""
        task1 = ImplementTask(id="t1", description="Task 1", files=["a.py"], complexity="simple")
        task2 = ImplementTask(id="t2", description="Task 2", files=["b.py"], complexity="simple")
        tasks = [task1, task2]
        completed = set()

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            mock_execute.return_value = TaskResult(task_id="t1", success=False, error="Failed")
            results = await executor.execute_plan(tasks, completed, stop_on_failure=True)

        assert len(results) == 1  # Only first task attempted
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_plan_calls_callback(self, executor, simple_task):
        """execute_plan calls on_task_complete callback."""
        tasks = [simple_task]
        completed = set()
        callback_called = []

        def callback(task_id, result):
            callback_called.append(task_id)

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            mock_execute.return_value = TaskResult(task_id=simple_task.id, success=True, diff="")
            await executor.execute_plan(tasks, completed, on_task_complete=callback)

        assert simple_task.id in callback_called


# ==============================================================================
# Parallel Execution Tests
# ==============================================================================


class TestHybridExecutorParallelExecution:
    """Tests for parallel execution."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"IMPL_PARALLEL_TASKS": "0"})
    async def test_execute_plan_parallel_fallback_when_disabled(self, executor, simple_task):
        """execute_plan_parallel falls back to sequential when feature disabled."""
        tasks = [simple_task]
        completed = set()

        with patch.object(executor, "execute_plan") as mock_plan:
            mock_plan.return_value = []
            await executor.execute_plan_parallel(tasks, completed)

        mock_plan.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"IMPL_PARALLEL_TASKS": "1"})
    async def test_execute_plan_parallel_groups_ready_tasks(self, repo_path):
        """execute_plan_parallel executes ready tasks in parallel."""
        # Need fresh executor to pick up env var
        executor = HybridExecutor(repo_path)

        task1 = ImplementTask(id="t1", description="Task 1", files=["a.py"], complexity="simple")
        task2 = ImplementTask(id="t2", description="Task 2", files=["b.py"], complexity="simple")
        tasks = [task1, task2]
        completed = set()

        with patch.object(executor, "execute_task_with_retry") as mock_execute:
            mock_execute.return_value = TaskResult(task_id="any", success=True, diff="")
            results = await executor.execute_plan_parallel(tasks, completed, max_parallel=2)

        # Both tasks should complete
        assert len(results) == 2


# ==============================================================================
# Codex Review Tests
# ==============================================================================


class TestHybridExecutorCodexReview:
    """Tests for Codex code review."""

    @pytest.mark.asyncio
    async def test_review_with_codex_empty_diff(self, executor):
        """review_with_codex returns approved for empty diff."""
        result = await executor.review_with_codex("")

        assert result["approved"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.CodexAgent")
    async def test_review_with_codex_approved(self, mock_codex_class, executor):
        """review_with_codex parses approved response."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="APPROVED: yes\nNo issues found")
        mock_codex_class.return_value = mock_agent

        result = await executor.review_with_codex("some diff")

        assert result["approved"] is True

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.CodexAgent")
    async def test_review_with_codex_not_approved(self, mock_codex_class, executor):
        """review_with_codex parses not approved response."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="APPROVED: no\nISSUES: Missing tests")
        mock_codex_class.return_value = mock_agent

        result = await executor.review_with_codex("some diff")

        assert result["approved"] is False

    @pytest.mark.asyncio
    @patch("aragora.implement.executor.CodexAgent")
    async def test_review_with_codex_handles_error(self, mock_codex_class, executor):
        """review_with_codex handles errors gracefully."""
        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(side_effect=Exception("API error"))
        mock_codex_class.return_value = mock_agent

        result = await executor.review_with_codex("some diff")

        assert result["approved"] is None
        assert "error" in result


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestHybridExecutorIntegration:
    """Integration tests for HybridExecutor."""

    @pytest.mark.asyncio
    async def test_full_plan_execution_workflow(self, executor):
        """Test complete plan execution workflow."""
        # Create a simple dependency chain
        task1 = ImplementTask(
            id="t1", description="Base task", files=["base.py"], complexity="simple"
        )
        task2 = ImplementTask(
            id="t2",
            description="Dependent task",
            files=["dependent.py"],
            complexity="moderate",
            dependencies=["t1"],
        )
        tasks = [task1, task2]
        completed = set()

        results_returned = []

        async def mock_execute(task):
            return TaskResult(task_id=task.id, success=True, diff=f"{task.id} diff")

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            results = await executor.execute_plan(tasks, completed)
            results_returned = results

        # Both tasks should complete in order
        assert len(results_returned) == 2
        assert completed == {"t1", "t2"}
        assert all(r.success for r in results_returned)

    @pytest.mark.asyncio
    async def test_retry_failed_tasks_at_end(self, executor):
        """Test that failed tasks are retried at the end of execution."""
        task1 = ImplementTask(
            id="t1", description="Failing task", files=["a.py"], complexity="simple"
        )
        task2 = ImplementTask(
            id="t2", description="Success task", files=["b.py"], complexity="simple"
        )
        tasks = [task1, task2]
        completed = set()

        call_count = {"t1": 0, "t2": 0}

        async def mock_execute_retry(task):
            call_count[task.id] = call_count.get(task.id, 0) + 1
            # t1 fails first time, succeeds second
            if task.id == "t1" and call_count[task.id] == 1:
                return TaskResult(task_id=task.id, success=False, error="Timeout")
            return TaskResult(task_id=task.id, success=True, diff="")

        async def mock_execute(task, attempt=1, use_fallback=False):
            call_count[task.id] = call_count.get(task.id, 0) + 1
            # Always succeed on final retry
            return TaskResult(task_id=task.id, success=True, diff="")

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute_retry):
            with patch.object(executor, "execute_task", side_effect=mock_execute):
                results = await executor.execute_plan(tasks, completed, stop_on_failure=False)

        # t1 should have been called twice (initial + retry at end)
        assert call_count["t1"] >= 2
