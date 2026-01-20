"""
Tests for hybrid executor.

Tests cover:
- HybridExecutor initialization
- Agent selection logic
- Timeout calculation
- Prompt building
- Task execution flow (mocked)
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement.executor import HybridExecutor
from aragora.implement.types import ImplementTask, TaskResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def executor(temp_repo):
    """Create an executor with default settings."""
    return HybridExecutor(temp_repo)


@pytest.fixture
def simple_task():
    """Create a simple task."""
    return ImplementTask(
        id="task-simple",
        description="Add a helper function",
        files=["src/utils.py"],
        complexity="simple",
    )


@pytest.fixture
def moderate_task():
    """Create a moderate task."""
    return ImplementTask(
        id="task-moderate",
        description="Add feature with tests",
        files=["src/feature.py", "tests/test_feature.py"],
        complexity="moderate",
    )


@pytest.fixture
def complex_task():
    """Create a complex task."""
    return ImplementTask(
        id="task-complex",
        description="Refactor module architecture",
        files=["src/a.py", "src/b.py", "src/c.py", "src/d.py"],
        complexity="complex",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestHybridExecutorInit:
    """Tests for HybridExecutor initialization."""

    def test_basic_init(self, temp_repo):
        """Test basic initialization."""
        executor = HybridExecutor(temp_repo)

        assert executor.repo_path == temp_repo
        assert executor.claude_timeout == 1200
        assert executor.codex_timeout == 1200
        assert executor.max_retries == 2

    def test_custom_timeouts(self, temp_repo):
        """Test custom timeout configuration."""
        executor = HybridExecutor(
            temp_repo,
            claude_timeout=600,
            codex_timeout=900,
            max_retries=3,
        )

        assert executor.claude_timeout == 600
        assert executor.codex_timeout == 900
        assert executor.max_retries == 3

    def test_agents_lazily_initialized(self, temp_repo):
        """Test agents are not created until first access."""
        executor = HybridExecutor(temp_repo)

        assert executor._claude is None
        assert executor._codex is None

    def test_claude_property_creates_agent(self, temp_repo):
        """Test claude property creates agent on first access."""
        executor = HybridExecutor(temp_repo)

        # Access the property
        claude = executor.claude

        assert claude is not None
        assert executor._claude is not None

    def test_codex_property_creates_agent(self, temp_repo):
        """Test codex property creates agent on first access."""
        executor = HybridExecutor(temp_repo)

        # Access the property
        codex = executor.codex

        assert codex is not None
        assert executor._codex is not None


# ============================================================================
# Agent Selection Tests
# ============================================================================


class TestAgentSelection:
    """Tests for agent selection logic."""

    def test_always_selects_claude(self, executor):
        """Test Claude is always selected for implementation."""
        for complexity in ["simple", "moderate", "complex"]:
            agent, name = executor._select_agent(complexity)
            assert name == "claude"

    def test_returns_claude_agent(self, executor):
        """Test returned agent is the Claude agent."""
        agent, _ = executor._select_agent("simple")
        assert agent == executor.claude


# ============================================================================
# Timeout Calculation Tests
# ============================================================================


class TestTimeoutCalculation:
    """Tests for task timeout calculation."""

    def test_simple_task_timeout(self, executor, simple_task):
        """Test simple task gets shorter timeout."""
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", True):
            timeout = executor._get_task_timeout(simple_task)

        # 300 base for simple
        assert timeout == 300

    def test_moderate_task_timeout(self, executor, moderate_task):
        """Test moderate task gets medium timeout."""
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", True):
            timeout = executor._get_task_timeout(moderate_task)

        # 600 base + 120 for 1 extra file
        assert timeout == 720

    def test_complex_task_timeout(self, executor, complex_task):
        """Test complex task gets longer timeout."""
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", True):
            timeout = executor._get_task_timeout(complex_task)

        # 1200 base + 360 for 3 extra files = 1560
        assert timeout == 1560

    def test_timeout_caps_at_30_minutes(self, executor):
        """Test timeout is capped at 30 minutes."""
        task = ImplementTask(
            id="huge",
            description="Massive task",
            files=[f"file{i}.py" for i in range(20)],  # Many files
            complexity="complex",
        )

        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", True):
            timeout = executor._get_task_timeout(task)

        assert timeout == 1800  # 30 minutes max

    def test_no_files_uses_base_timeout(self, executor):
        """Test task with no files uses base timeout."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="moderate",
        )

        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", True):
            timeout = executor._get_task_timeout(task)

        assert timeout == 600  # Base moderate timeout

    def test_feature_flag_disabled(self, executor, simple_task):
        """Test returns default timeout when feature disabled."""
        executor.claude_timeout = 1234

        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", False):
            timeout = executor._get_task_timeout(simple_task)

        assert timeout == 1234


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestPromptBuilding:
    """Tests for prompt building."""

    def test_includes_task_description(self, executor, simple_task):
        """Test prompt includes task description."""
        prompt = executor._build_prompt(simple_task)
        assert simple_task.description in prompt

    def test_includes_file_list(self, executor, moderate_task):
        """Test prompt includes file list."""
        prompt = executor._build_prompt(moderate_task)
        assert "src/feature.py" in prompt
        assert "tests/test_feature.py" in prompt

    def test_includes_repo_path(self, executor, simple_task):
        """Test prompt includes repository path."""
        prompt = executor._build_prompt(simple_task)
        assert str(executor.repo_path) in prompt

    def test_handles_empty_files(self, executor):
        """Test prompt handles task with no files."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "determine from description" in prompt


# ============================================================================
# Git Diff Tests
# ============================================================================


class TestGitDiff:
    """Tests for git diff functionality."""

    def test_returns_empty_on_failure(self, executor):
        """Test returns empty string when git fails."""
        import subprocess
        with patch("aragora.implement.executor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.SubprocessError("git not found")
            diff = executor._get_git_diff()

        assert diff == ""

    def test_returns_diff_output(self, executor):
        """Test returns git diff output."""
        with patch("aragora.implement.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="file.py | 10 +")
            diff = executor._get_git_diff()

        assert diff == "file.py | 10 +"

    def test_handles_timeout(self, executor):
        """Test handles subprocess timeout."""
        import subprocess

        with patch("aragora.implement.executor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 30)
            diff = executor._get_git_diff()

        assert diff == ""


# ============================================================================
# Task Execution Tests (Mocked)
# ============================================================================


class TestTaskExecution:
    """Tests for task execution with mocked agents."""

    @pytest.mark.asyncio
    async def test_execute_task_success(self, executor, simple_task):
        """Test successful task execution."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="Done")
        mock_agent.timeout = 300
        mock_agent.name = "claude"

        with (
            patch.object(executor, "_select_agent", return_value=(mock_agent, "claude")),
            patch.object(executor, "_get_git_diff", return_value="+ new line"),
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            result = await executor.execute_task(simple_task)

        assert result.success is True
        assert result.task_id == "task-simple"
        assert result.diff == "+ new line"
        assert result.model_used == "claude"

    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, executor, simple_task):
        """Test task execution timeout."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_agent.timeout = 300
        mock_agent.name = "claude"

        with (
            patch.object(executor, "_select_agent", return_value=(mock_agent, "claude")),
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            result = await executor.execute_task(simple_task)

        assert result.success is False
        assert "Timeout" in result.error
        assert result.model_used == "claude"

    @pytest.mark.asyncio
    async def test_execute_task_error(self, executor, simple_task):
        """Test task execution error."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(side_effect=Exception("API error"))
        mock_agent.timeout = 300
        mock_agent.name = "claude"

        with (
            patch.object(executor, "_select_agent", return_value=(mock_agent, "claude")),
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            result = await executor.execute_task(simple_task)

        assert result.success is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, executor, simple_task):
        """Test task execution with fallback to codex."""
        # Mock codex agent for fallback
        mock_codex = MagicMock()
        mock_codex.generate = AsyncMock(return_value="Done")
        mock_codex.timeout = 300
        mock_codex.name = "codex"

        with (
            patch.object(executor, "_get_git_diff", return_value="diff"),
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            # Set executor._codex directly since property won't be called
            executor._codex = mock_codex
            result = await executor.execute_task(simple_task, use_fallback=True)

        assert result.success is True
        assert result.model_used == "codex-fallback"


# ============================================================================
# Task Retry Tests
# ============================================================================


class TestTaskRetry:
    """Tests for task retry logic."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self, executor, simple_task):
        """Test no retry when task succeeds."""
        with patch.object(executor, "execute_task") as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id="task-simple",
                success=True,
                diff="diff",
                model_used="claude",
            )

            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is True
        assert mock_execute.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, executor, simple_task):
        """Test retry on timeout error."""
        # First call fails with timeout, second succeeds
        call_count = [0]

        async def mock_execute(task, attempt=1, use_fallback=False):
            call_count[0] += 1
            if call_count[0] == 1:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error="Timeout: operation timed out",
                )
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, "execute_task", side_effect=mock_execute):
            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is True
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_timeout_error(self, executor, simple_task):
        """Test no retry on non-timeout errors."""
        with patch.object(executor, "execute_task") as mock_execute:
            mock_execute.return_value = TaskResult(
                task_id="task-simple",
                success=False,
                error="Syntax error in generated code",
            )

            result = await executor.execute_task_with_retry(simple_task)

        assert result.success is False
        # Should only be called once (no retry for non-timeout)
        assert mock_execute.call_count == 1


# ============================================================================
# Plan Execution Tests
# ============================================================================


class TestPlanExecution:
    """Tests for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_plan_respects_dependencies(self, executor):
        """Test plan execution respects task dependencies."""
        tasks = [
            ImplementTask("t1", "First", ["a.py"], "simple"),
            ImplementTask("t2", "Second", ["b.py"], "simple", ["t1"]),
        ]
        completed = set()
        executed_order = []

        async def mock_execute(task):
            executed_order.append(task.id)
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            await executor.execute_plan(tasks, completed)

        assert executed_order == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_execute_plan_skips_completed(self, executor):
        """Test plan execution skips already completed tasks."""
        tasks = [
            ImplementTask("t1", "First", ["a.py"], "simple"),
            ImplementTask("t2", "Second", ["b.py"], "simple"),
        ]
        completed = {"t1"}  # t1 already done
        executed = []

        async def mock_execute(task):
            executed.append(task.id)
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            await executor.execute_plan(tasks, completed)

        assert executed == ["t2"]

    @pytest.mark.asyncio
    async def test_execute_plan_continues_after_failure(self, executor):
        """Test plan execution continues after failure by default."""
        tasks = [
            ImplementTask("t1", "First", ["a.py"], "simple"),
            ImplementTask("t2", "Second", ["b.py"], "simple"),
        ]
        completed = set()
        call_count = [0]

        async def mock_execute(task):
            call_count[0] += 1
            if task.id == "t1":
                return TaskResult(task_id=task.id, success=False, error="Failed")
            return TaskResult(task_id=task.id, success=True)

        # Also mock execute_task to prevent retry logic from running real agents
        async def mock_execute_single(task, attempt=1, use_fallback=False):
            return TaskResult(task_id=task.id, success=False, error="Failed")

        with (
            patch.object(executor, "execute_task_with_retry", side_effect=mock_execute),
            patch.object(executor, "execute_task", side_effect=mock_execute_single),
        ):
            results = await executor.execute_plan(tasks, completed, stop_on_failure=False)

        # Both tasks should be in results (t1 failed, t2 succeeded, then retry for t1)
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_execute_plan_stops_on_failure(self, executor):
        """Test plan execution stops on failure when configured."""
        tasks = [
            ImplementTask("t1", "First", ["a.py"], "simple"),
            ImplementTask("t2", "Second", ["b.py"], "simple"),
        ]
        completed = set()

        async def mock_execute(task):
            if task.id == "t1":
                return TaskResult(task_id=task.id, success=False, error="Failed")
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            results = await executor.execute_plan(tasks, completed, stop_on_failure=True)

        # Should stop after t1 fails
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_execute_plan_callback(self, executor):
        """Test plan execution calls callback after each task."""
        tasks = [ImplementTask("t1", "First", ["a.py"], "simple")]
        completed = set()
        callback_calls = []

        def callback(task_id, result):
            callback_calls.append((task_id, result.success))

        async def mock_execute(task):
            return TaskResult(task_id=task.id, success=True)

        with patch.object(executor, "execute_task_with_retry", side_effect=mock_execute):
            await executor.execute_plan(tasks, completed, on_task_complete=callback)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("t1", True)


# ============================================================================
# Code Review Tests
# ============================================================================


class TestCodeReview:
    """Tests for code review functionality."""

    @pytest.mark.asyncio
    async def test_review_empty_diff(self, executor):
        """Test review of empty diff returns approved."""
        result = await executor.review_with_codex("")

        assert result["approved"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_review_approved(self, executor):
        """Test review returns approved when Codex approves."""
        with (
            patch("aragora.implement.executor.CodexAgent") as mock_agent_cls,
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(
                return_value="APPROVED: yes\nISSUES: None\nSUGGESTIONS: Add more tests"
            )
            mock_agent_cls.return_value = mock_agent

            result = await executor.review_with_codex("+ added code")

        assert result["approved"] is True
        assert "review" in result

    @pytest.mark.asyncio
    async def test_review_not_approved(self, executor):
        """Test review returns not approved when Codex rejects."""
        with (
            patch("aragora.implement.executor.CodexAgent") as mock_agent_cls,
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(
                return_value="APPROVED: no\nISSUES: Security vulnerability"
            )
            mock_agent_cls.return_value = mock_agent

            result = await executor.review_with_codex("+ unsafe code")

        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_review_error_handling(self, executor):
        """Test review handles errors gracefully."""
        with (
            patch("aragora.implement.executor.CodexAgent") as mock_agent_cls,
            patch("aragora.server.stream.arena_hooks.streaming_task_context", MagicMock()),
        ):
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(side_effect=Exception("API error"))
            mock_agent_cls.return_value = mock_agent

            result = await executor.review_with_codex("+ code")

        assert result["approved"] is None
        assert "error" in result
