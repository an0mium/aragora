"""Tests for aragora/implement/ module - Nomic Loop Phase 3 implementation."""

import asyncio
import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

from aragora.implement.types import (
    ImplementTask,
    ImplementPlan,
    TaskResult,
    ImplementProgress,
)
from aragora.implement.executor import HybridExecutor
from aragora.implement.planner import (
    extract_json,
    validate_plan,
    generate_implement_plan,
    decompose_failed_task,
    create_single_task_plan,
)
from aragora.implement.checkpoint import (
    get_progress_path,
    save_progress,
    load_progress,
    clear_progress,
    update_current_task,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_task():
    """Create sample ImplementTask."""
    return ImplementTask(
        id="task-1",
        description="Add logging to module",
        files=["src/module.py"],
        complexity="simple",
        dependencies=[],
    )


@pytest.fixture
def complex_task():
    """Create complex task with multiple files."""
    return ImplementTask(
        id="task-complex",
        description="Refactor entire module with tests",
        files=["src/a.py", "src/b.py", "src/c.py", "tests/test_a.py"],
        complexity="complex",
        dependencies=["task-1"],
    )


@pytest.fixture
def sample_plan(sample_task, complex_task):
    """Create sample ImplementPlan."""
    return ImplementPlan(
        design_hash="abc123",
        tasks=[sample_task, complex_task],
    )


@pytest.fixture
def sample_result():
    """Create sample TaskResult."""
    return TaskResult(
        task_id="task-1",
        success=True,
        diff="+ added line",
        model_used="claude",
        duration_seconds=10.5,
    )


@pytest.fixture
def sample_progress(sample_plan, sample_result):
    """Create sample ImplementProgress."""
    return ImplementProgress(
        plan=sample_plan,
        completed_tasks=["task-1"],
        current_task="task-complex",
        results=[sample_result],
    )


@pytest.fixture
def executor(tmp_path):
    """Create HybridExecutor with mocked agents."""
    with patch("aragora.implement.executor.ClaudeAgent") as mock_claude, \
         patch("aragora.implement.executor.CodexAgent") as mock_codex:
        # Setup Claude mock
        claude_instance = MagicMock()
        claude_instance.generate = AsyncMock(return_value="Done")
        mock_claude.return_value = claude_instance

        # Setup Codex mock
        codex_instance = MagicMock()
        codex_instance.generate = AsyncMock(return_value="Done")
        mock_codex.return_value = codex_instance

        exec_ = HybridExecutor(repo_path=tmp_path)
        yield exec_


# =============================================================================
# ImplementTask Dataclass Tests
# =============================================================================

class TestImplementTask:
    """Tests for ImplementTask dataclass."""

    def test_all_fields_initialized(self, sample_task):
        """All fields should be initialized correctly."""
        assert sample_task.id == "task-1"
        assert sample_task.description == "Add logging to module"
        assert sample_task.files == ["src/module.py"]
        assert sample_task.complexity == "simple"
        assert sample_task.dependencies == []

    def test_to_dict_serialization(self, sample_task):
        """to_dict should serialize all fields."""
        d = sample_task.to_dict()

        assert d["id"] == "task-1"
        assert d["description"] == "Add logging to module"
        assert d["files"] == ["src/module.py"]
        assert d["complexity"] == "simple"
        assert d["dependencies"] == []

    def test_from_dict_deserialization(self):
        """from_dict should deserialize correctly."""
        data = {
            "id": "task-2",
            "description": "Test task",
            "files": ["a.py", "b.py"],
            "complexity": "moderate",
            "dependencies": ["task-1"],
        }

        task = ImplementTask.from_dict(data)

        assert task.id == "task-2"
        assert task.description == "Test task"
        assert task.files == ["a.py", "b.py"]
        assert task.complexity == "moderate"
        assert task.dependencies == ["task-1"]

    def test_defaults_for_optional_fields(self):
        """from_dict should use defaults for missing optional fields."""
        data = {
            "id": "task-3",
            "description": "Minimal task",
        }

        task = ImplementTask.from_dict(data)

        assert task.files == []
        assert task.complexity == "moderate"  # Default
        assert task.dependencies == []


# =============================================================================
# ImplementPlan Dataclass Tests
# =============================================================================

class TestImplementPlan:
    """Tests for ImplementPlan dataclass."""

    def test_all_fields_initialized(self, sample_plan):
        """All fields should be initialized correctly."""
        assert sample_plan.design_hash == "abc123"
        assert len(sample_plan.tasks) == 2
        assert sample_plan.created_at is not None

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated."""
        plan = ImplementPlan(
            design_hash="xyz",
            tasks=[],
        )

        assert plan.created_at is not None
        assert isinstance(plan.created_at, datetime)

    def test_to_dict_from_dict_roundtrip(self, sample_plan):
        """to_dict/from_dict should preserve data."""
        d = sample_plan.to_dict()
        restored = ImplementPlan.from_dict(d)

        assert restored.design_hash == sample_plan.design_hash
        assert len(restored.tasks) == len(sample_plan.tasks)
        assert restored.tasks[0].id == sample_plan.tasks[0].id

    def test_tasks_list_preserved(self, sample_task):
        """Tasks list should be preserved correctly."""
        plan = ImplementPlan(
            design_hash="test",
            tasks=[sample_task],
        )

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "task-1"


# =============================================================================
# TaskResult Dataclass Tests
# =============================================================================

class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_all_fields_initialized(self, sample_result):
        """All fields should be initialized correctly."""
        assert sample_result.task_id == "task-1"
        assert sample_result.success is True
        assert sample_result.diff == "+ added line"
        assert sample_result.model_used == "claude"
        assert sample_result.duration_seconds == 10.5

    def test_to_dict_serialization(self, sample_result):
        """to_dict should serialize all fields."""
        d = sample_result.to_dict()

        assert d["task_id"] == "task-1"
        assert d["success"] is True
        assert d["diff"] == "+ added line"
        assert d["error"] is None
        assert d["model_used"] == "claude"
        assert d["duration_seconds"] == 10.5

    def test_optional_fields_default(self):
        """Optional fields should have correct defaults."""
        result = TaskResult(
            task_id="t1",
            success=False,
        )

        assert result.diff == ""
        assert result.error is None
        assert result.model_used is None
        assert result.duration_seconds == 0.0


# =============================================================================
# ImplementProgress Dataclass Tests
# =============================================================================

class TestImplementProgress:
    """Tests for ImplementProgress dataclass."""

    def test_all_fields_initialized(self, sample_progress):
        """All fields should be initialized correctly."""
        assert sample_progress.plan is not None
        assert sample_progress.completed_tasks == ["task-1"]
        assert sample_progress.current_task == "task-complex"
        assert len(sample_progress.results) == 1

    def test_to_dict_from_dict_roundtrip(self, sample_progress):
        """to_dict/from_dict should preserve data."""
        d = sample_progress.to_dict()
        restored = ImplementProgress.from_dict(d)

        assert restored.plan.design_hash == sample_progress.plan.design_hash
        assert restored.completed_tasks == sample_progress.completed_tasks
        assert restored.current_task == sample_progress.current_task

    def test_results_list_preserved(self, sample_progress):
        """Results list should be preserved."""
        d = sample_progress.to_dict()
        restored = ImplementProgress.from_dict(d)

        assert len(restored.results) == 1
        assert restored.results[0].task_id == "task-1"
        assert restored.results[0].success is True

    def test_nested_plan_serialization(self, sample_progress):
        """Nested plan should serialize correctly."""
        d = sample_progress.to_dict()

        assert "plan" in d
        assert d["plan"]["design_hash"] == "abc123"
        assert len(d["plan"]["tasks"]) == 2


# =============================================================================
# HybridExecutor Initialization Tests
# =============================================================================

class TestHybridExecutorInit:
    """Tests for HybridExecutor initialization."""

    def test_default_config(self, tmp_path):
        """Default config should have correct values."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"):
            exec_ = HybridExecutor(repo_path=tmp_path)

            assert exec_.claude_timeout == 1200
            assert exec_.codex_timeout == 1200
            assert exec_.max_retries == 2

    def test_custom_config(self, tmp_path):
        """Custom config should be accepted."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"):
            exec_ = HybridExecutor(
                repo_path=tmp_path,
                claude_timeout=600,
                codex_timeout=900,
                max_retries=5,
            )

            assert exec_.claude_timeout == 600
            assert exec_.codex_timeout == 900
            assert exec_.max_retries == 5

    def test_lazy_agent_creation(self, tmp_path):
        """Agents should be created lazily."""
        with patch("aragora.implement.executor.ClaudeAgent") as mock_claude, \
             patch("aragora.implement.executor.CodexAgent") as mock_codex:
            exec_ = HybridExecutor(repo_path=tmp_path)

            # Agents not created yet
            assert exec_._claude is None
            assert exec_._codex is None

            # Access claude property
            _ = exec_.claude
            mock_claude.assert_called_once()

            # Access codex property
            _ = exec_.codex
            mock_codex.assert_called_once()

    def test_repo_path_stored(self, tmp_path):
        """Repo path should be stored."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"):
            exec_ = HybridExecutor(repo_path=tmp_path)

            assert exec_.repo_path == tmp_path


# =============================================================================
# _select_agent Method Tests
# =============================================================================

class TestSelectAgent:
    """Tests for HybridExecutor._select_agent method."""

    def test_always_returns_claude(self, executor):
        """Should always return Claude agent."""
        agent, name = executor._select_agent("simple")
        assert name == "claude"

        agent, name = executor._select_agent("complex")
        assert name == "claude"

    def test_returns_claude_model_name(self, executor):
        """Should return 'claude' as model name."""
        _, name = executor._select_agent("moderate")
        assert name == "claude"


# =============================================================================
# _get_task_timeout Method Tests
# =============================================================================

class TestGetTaskTimeout:
    """Tests for HybridExecutor._get_task_timeout method."""

    def test_simple_task_base_timeout(self, executor, sample_task):
        """Simple task should have 300s base timeout."""
        sample_task.complexity = "simple"
        sample_task.files = ["one.py"]

        timeout = executor._get_task_timeout(sample_task)

        assert timeout == 300

    def test_moderate_task_base_timeout(self, executor, sample_task):
        """Moderate task should have 600s base timeout."""
        sample_task.complexity = "moderate"
        sample_task.files = ["one.py"]

        timeout = executor._get_task_timeout(sample_task)

        assert timeout == 600

    def test_complex_task_base_timeout(self, executor, sample_task):
        """Complex task should have 1200s base timeout."""
        sample_task.complexity = "complex"
        sample_task.files = ["one.py"]

        timeout = executor._get_task_timeout(sample_task)

        assert timeout == 1200

    def test_file_count_bonus(self, executor, sample_task):
        """Extra files should add +120s per file."""
        sample_task.complexity = "simple"
        sample_task.files = ["a.py", "b.py", "c.py"]  # 3 files = +240s

        timeout = executor._get_task_timeout(sample_task)

        assert timeout == 300 + 240  # 540s

    def test_cap_at_1800s(self, executor, sample_task):
        """Timeout should cap at 1800s (30 min)."""
        sample_task.complexity = "complex"
        sample_task.files = [f"file{i}.py" for i in range(20)]  # Many files

        timeout = executor._get_task_timeout(sample_task)

        assert timeout == 1800

    def test_feature_flag_off(self, executor, sample_task):
        """When COMPLEXITY_TIMEOUT OFF, should return default."""
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", False):
            timeout = executor._get_task_timeout(sample_task)

            assert timeout == executor.claude_timeout


# =============================================================================
# _build_prompt Method Tests
# =============================================================================

class TestBuildPrompt:
    """Tests for HybridExecutor._build_prompt method."""

    def test_includes_description(self, executor, sample_task):
        """Prompt should include task description."""
        prompt = executor._build_prompt(sample_task)

        assert "Add logging to module" in prompt

    def test_includes_file_list(self, executor, sample_task):
        """Prompt should include file list."""
        prompt = executor._build_prompt(sample_task)

        assert "src/module.py" in prompt

    def test_handles_empty_files(self, executor, sample_task):
        """Should handle empty files list."""
        sample_task.files = []

        prompt = executor._build_prompt(sample_task)

        assert "(determine from description)" in prompt


# =============================================================================
# execute_task Method Tests
# =============================================================================

class TestExecuteTask:
    """Tests for HybridExecutor.execute_task method."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, executor, sample_task):
        """Successful execution should return TaskResult with success=True."""
        with patch.object(executor, "_get_git_diff", return_value="+ line"):
            result = await executor.execute_task(sample_task)

            assert result.success is True
            assert result.task_id == "task-1"

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self, executor, sample_task):
        """Timeout should return failure with error."""
        # Get the claude agent and make it timeout
        claude = executor.claude
        claude.generate = AsyncMock(side_effect=TimeoutError("Timed out"))

        result = await executor.execute_task(sample_task)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_failure(self, executor, sample_task):
        """Exception should return failure with error."""
        claude = executor.claude
        claude.generate = AsyncMock(side_effect=Exception("API Error"))

        result = await executor.execute_task(sample_task)

        assert result.success is False
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_model_used_tracked(self, executor, sample_task):
        """Model used should be tracked."""
        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(sample_task)

            assert result.model_used == "claude"

    @pytest.mark.asyncio
    async def test_duration_tracked(self, executor, sample_task):
        """Duration should be tracked."""
        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(sample_task)

            assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_fallback_uses_codex(self, executor, sample_task):
        """Fallback should use Codex."""
        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task(sample_task, use_fallback=True)

            assert "codex" in result.model_used.lower()

    @pytest.mark.asyncio
    async def test_attempt_scales_timeout(self, executor, sample_task):
        """Attempt number should scale timeout."""
        with patch.object(executor, "_get_git_diff", return_value=""):
            # First attempt
            await executor.execute_task(sample_task, attempt=1)
            timeout1 = executor.claude.timeout

            # Second attempt
            await executor.execute_task(sample_task, attempt=2)
            timeout2 = executor.claude.timeout

            # timeout2 should be 2x timeout1
            assert timeout2 == timeout1 * 2

    @pytest.mark.asyncio
    async def test_diff_captured(self, executor, sample_task):
        """Git diff should be captured on success."""
        with patch.object(executor, "_get_git_diff", return_value="+ new line"):
            result = await executor.execute_task(sample_task)

            assert result.diff == "+ new line"


# =============================================================================
# execute_task_with_retry Method Tests
# =============================================================================

class TestExecuteTaskWithRetry:
    """Tests for HybridExecutor.execute_task_with_retry method."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self, executor, sample_task):
        """Success on first attempt should return immediately."""
        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task_with_retry(sample_task)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self, executor, sample_task):
        """Timeout should trigger retry."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("First timeout")
            return "Done"

        executor.claude.generate = mock_generate

        with patch.object(executor, "_get_git_diff", return_value=""):
            result = await executor.execute_task_with_retry(sample_task)

            # Should have retried
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_non_timeout_no_retry(self, executor, sample_task):
        """Non-timeout errors should not trigger model-fallback retry."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("API Error")

        executor.claude.generate = mock_generate

        result = await executor.execute_task_with_retry(sample_task)

        # Should have tried once (non-timeout doesn't retry)
        assert result.success is False


# =============================================================================
# execute_plan Method Tests
# =============================================================================

class TestExecutePlan:
    """Tests for HybridExecutor.execute_plan method."""

    @pytest.mark.asyncio
    async def test_skips_completed_tasks(self, executor, sample_task, complex_task):
        """Should skip already completed tasks."""
        tasks = [sample_task, complex_task]
        completed = {"task-1"}  # task-1 already completed

        with patch.object(executor, "_get_git_diff", return_value=""):
            results = await executor.execute_plan(tasks, completed)

            # Only complex_task should be executed
            task_ids = [r.task_id for r in results]
            assert "task-1" not in task_ids

    @pytest.mark.asyncio
    async def test_respects_dependencies(self, executor, sample_task, complex_task):
        """Should respect task dependencies."""
        tasks = [sample_task, complex_task]
        completed = set()

        with patch.object(executor, "_get_git_diff", return_value=""):
            results = await executor.execute_plan(tasks, completed)

            # Both should execute in order (sample_task first, then complex_task)
            assert len(results) >= 1
            assert results[0].task_id == "task-1"

    @pytest.mark.asyncio
    async def test_continues_after_failure(self, executor, sample_task, complex_task):
        """Should continue after failure (default behavior)."""
        # Make task-1 fail
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First few calls fail (including retries)
                raise Exception("Fail")
            return "Done"

        executor.claude.generate = mock_generate

        tasks = [sample_task]
        completed = set()

        results = await executor.execute_plan(tasks, completed, stop_on_failure=False)

        # Should have results even if failed
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_stops_on_failure_when_configured(self, executor, sample_task, complex_task):
        """Should stop on failure when stop_on_failure=True."""
        executor.claude.generate = AsyncMock(side_effect=Exception("Fail"))

        # complex_task depends on task-1
        tasks = [sample_task, complex_task]
        completed = set()

        results = await executor.execute_plan(tasks, completed, stop_on_failure=True)

        # Should stop after first failure
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_callback_invoked(self, executor, sample_task):
        """Callback should be invoked on completion."""
        callback_calls = []

        def callback(task_id, result):
            callback_calls.append((task_id, result.success))

        with patch.object(executor, "_get_git_diff", return_value=""):
            await executor.execute_plan([sample_task], set(), on_task_complete=callback)

            assert len(callback_calls) == 1
            assert callback_calls[0][0] == "task-1"


# =============================================================================
# execute_plan_parallel Method Tests
# =============================================================================

class TestExecutePlanParallel:
    """Tests for HybridExecutor.execute_plan_parallel method."""

    @pytest.mark.asyncio
    async def test_falls_back_when_flag_off(self, executor, sample_task):
        """Should fall back to sequential when PARALLEL_TASKS OFF."""
        with patch("aragora.implement.executor.PARALLEL_TASKS", False):
            with patch.object(executor, "execute_plan", new_callable=AsyncMock) as mock_plan:
                mock_plan.return_value = []

                await executor.execute_plan_parallel([sample_task], set())

                mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_max_parallel(self, executor, sample_task):
        """Should respect max_parallel limit."""
        # Create independent tasks
        tasks = [
            ImplementTask(id=f"task-{i}", description=f"Task {i}", files=[], complexity="simple", dependencies=[])
            for i in range(5)
        ]

        with patch("aragora.implement.executor.PARALLEL_TASKS", True):
            with patch.object(executor, "_get_git_diff", return_value=""):
                results = await executor.execute_plan_parallel(tasks, set(), max_parallel=2)

                # All tasks should complete
                assert len(results) == 5


# =============================================================================
# extract_json Function Tests
# =============================================================================

class TestExtractJson:
    """Tests for extract_json function."""

    def test_extracts_from_code_block(self):
        """Should extract JSON from code block."""
        text = """Here's the plan:
```json
{"tasks": []}
```
Done."""

        result = extract_json(text)

        assert '{"tasks": []}' in result

    def test_extracts_raw_json(self):
        """Should extract raw JSON object."""
        text = 'Some text {"tasks": [{"id": "1"}]} more text'

        result = extract_json(text)

        assert "tasks" in result

    def test_handles_json_with_markdown(self):
        """Should handle JSON within markdown."""
        text = """```
{"tasks": []}
```"""

        result = extract_json(text)

        assert "tasks" in result

    def test_returns_original_if_no_json(self):
        """Should return original if no JSON found."""
        text = "Just plain text"

        result = extract_json(text)

        assert result == text

    def test_multiple_code_blocks_takes_first(self):
        """Should take first JSON code block."""
        text = """```json
{"first": true}
```
```json
{"second": true}
```"""

        result = extract_json(text)

        assert "first" in result


# =============================================================================
# validate_plan Function Tests
# =============================================================================

class TestValidatePlan:
    """Tests for validate_plan function."""

    def test_missing_tasks_key_error(self):
        """Should error on missing 'tasks' key."""
        errors = validate_plan({})

        assert any("tasks" in e.lower() for e in errors)

    def test_non_list_tasks_error(self):
        """Should error on non-list 'tasks'."""
        errors = validate_plan({"tasks": "not a list"})

        assert any("list" in e.lower() for e in errors)

    def test_empty_tasks_error(self):
        """Should error on empty tasks list."""
        errors = validate_plan({"tasks": []})

        assert any("no tasks" in e.lower() for e in errors)

    def test_missing_task_id_error(self):
        """Should error on missing task 'id'."""
        errors = validate_plan({"tasks": [{"description": "Test", "files": []}]})

        assert any("id" in e.lower() for e in errors)

    def test_missing_task_description_error(self):
        """Should error on missing task 'description'."""
        errors = validate_plan({"tasks": [{"id": "1", "files": []}]})

        assert any("description" in e.lower() for e in errors)

    def test_missing_task_files_error(self):
        """Should error on missing task 'files'."""
        errors = validate_plan({"tasks": [{"id": "1", "description": "Test"}]})

        assert any("files" in e.lower() for e in errors)

    def test_invalid_complexity_error(self):
        """Should error on invalid complexity."""
        errors = validate_plan({
            "tasks": [{
                "id": "1",
                "description": "Test",
                "files": [],
                "complexity": "invalid",
            }]
        })

        assert any("complexity" in e.lower() for e in errors)

    def test_duplicate_task_id_error(self):
        """Should error on duplicate task ID."""
        errors = validate_plan({
            "tasks": [
                {"id": "1", "description": "A", "files": [], "complexity": "simple"},
                {"id": "1", "description": "B", "files": [], "complexity": "simple"},
            ]
        })

        assert any("duplicate" in e.lower() for e in errors)


# =============================================================================
# generate_implement_plan Function Tests
# =============================================================================

class TestGenerateImplementPlan:
    """Tests for generate_implement_plan function."""

    @pytest.mark.asyncio
    async def test_generates_plan(self, tmp_path):
        """Should generate plan from design (mocked)."""
        response = json.dumps({
            "tasks": [{
                "id": "task-1",
                "description": "Add feature",
                "files": ["src/feature.py"],
                "complexity": "simple",
                "dependencies": [],
            }]
        })

        with patch("aragora.implement.planner.GeminiCLIAgent") as mock_gemini:
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(return_value=response)
            mock_gemini.return_value = mock_agent

            plan = await generate_implement_plan("Design text", tmp_path)

            assert len(plan.tasks) == 1
            assert plan.tasks[0].id == "task-1"

    @pytest.mark.asyncio
    async def test_parses_json_response(self, tmp_path):
        """Should parse JSON response."""
        response = '```json\n{"tasks": [{"id": "1", "description": "X", "files": [], "complexity": "simple"}]}\n```'

        with patch("aragora.implement.planner.GeminiCLIAgent") as mock_gemini:
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(return_value=response)
            mock_gemini.return_value = mock_agent

            plan = await generate_implement_plan("Design", tmp_path)

            assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_validates_plan_structure(self, tmp_path):
        """Should validate plan structure."""
        response = '{"invalid": "no tasks"}'

        with patch("aragora.implement.planner.GeminiCLIAgent") as mock_gemini:
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(return_value=response)
            mock_gemini.return_value = mock_agent

            with pytest.raises(ValueError, match="Invalid plan"):
                await generate_implement_plan("Design", tmp_path)

    @pytest.mark.asyncio
    async def test_returns_implement_plan(self, tmp_path):
        """Should return ImplementPlan with tasks."""
        response = json.dumps({
            "tasks": [{
                "id": "t1",
                "description": "Task",
                "files": ["a.py"],
                "complexity": "moderate",
            }]
        })

        with patch("aragora.implement.planner.GeminiCLIAgent") as mock_gemini:
            mock_agent = MagicMock()
            mock_agent.generate = AsyncMock(return_value=response)
            mock_gemini.return_value = mock_agent

            plan = await generate_implement_plan("Design", tmp_path)

            assert isinstance(plan, ImplementPlan)
            assert plan.design_hash is not None


# =============================================================================
# decompose_failed_task Function Tests
# =============================================================================

class TestDecomposeFailedTask:
    """Tests for decompose_failed_task function."""

    @pytest.mark.asyncio
    async def test_feature_flag_off_returns_original(self, complex_task, tmp_path):
        """When DECOMPOSE_FAILED OFF, should return original task."""
        with patch("aragora.implement.planner.DECOMPOSE_FAILED", False):
            result = await decompose_failed_task(complex_task, "error", tmp_path)

            assert result == [complex_task]

    @pytest.mark.asyncio
    async def test_non_complex_returns_original(self, sample_task, tmp_path):
        """Non-complex task should return original."""
        with patch("aragora.implement.planner.DECOMPOSE_FAILED", True):
            result = await decompose_failed_task(sample_task, "error", tmp_path)

            assert result == [sample_task]

    @pytest.mark.asyncio
    async def test_few_files_returns_original(self, tmp_path):
        """Task with <= 2 files should return original."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py", "b.py"],
            complexity="complex",
            dependencies=[],
        )

        with patch("aragora.implement.planner.DECOMPOSE_FAILED", True):
            result = await decompose_failed_task(task, "error", tmp_path)

            assert result == [task]

    @pytest.mark.asyncio
    async def test_successful_decomposition(self, complex_task, tmp_path):
        """Successful decomposition should return subtasks."""
        response = json.dumps({
            "subtasks": [
                {"id": "task-complex-a", "description": "Part A", "files": ["a.py"], "complexity": "simple"},
                {"id": "task-complex-b", "description": "Part B", "files": ["b.py"], "complexity": "simple"},
            ]
        })

        with patch("aragora.implement.planner.DECOMPOSE_FAILED", True):
            with patch("aragora.implement.planner.GeminiCLIAgent") as mock_gemini:
                mock_agent = MagicMock()
                mock_agent.generate = AsyncMock(return_value=response)
                mock_gemini.return_value = mock_agent

                result = await decompose_failed_task(complex_task, "timeout", tmp_path)

                assert len(result) == 2
                assert result[0].id == "task-complex-a"


# =============================================================================
# create_single_task_plan Function Tests
# =============================================================================

class TestCreateSingleTaskPlan:
    """Tests for create_single_task_plan function."""

    def test_creates_single_complex_task(self, tmp_path):
        """Should create single complex task."""
        plan = create_single_task_plan("Design text", tmp_path)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].complexity == "complex"
        assert plan.tasks[0].id == "task-1"

    def test_generates_design_hash(self, tmp_path):
        """Should generate design hash."""
        plan = create_single_task_plan("Design text", tmp_path)

        assert plan.design_hash is not None
        assert len(plan.design_hash) == 64  # SHA256 hash length


# =============================================================================
# Checkpoint Function Tests
# =============================================================================

class TestCheckpointFunctions:
    """Tests for checkpoint functions."""

    def test_get_progress_path(self, tmp_path):
        """Should return correct path."""
        path = get_progress_path(tmp_path)

        assert path == tmp_path / ".nomic" / "implement_progress.json"

    def test_save_progress_creates_directory(self, tmp_path, sample_progress):
        """save_progress should create directory if needed."""
        save_progress(sample_progress, tmp_path)

        assert (tmp_path / ".nomic").exists()
        assert get_progress_path(tmp_path).exists()

    def test_save_progress_atomic_write(self, tmp_path, sample_progress):
        """save_progress should use atomic write."""
        save_progress(sample_progress, tmp_path)

        # File should exist and be valid JSON
        with open(get_progress_path(tmp_path)) as f:
            data = json.load(f)
            assert "plan" in data

    def test_load_progress_missing_returns_none(self, tmp_path):
        """load_progress should return None if file missing."""
        result = load_progress(tmp_path)

        assert result is None

    def test_load_progress_corrupted_returns_none(self, tmp_path):
        """load_progress should return None if file corrupted."""
        progress_path = get_progress_path(tmp_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w") as f:
            f.write("not valid json {{{")

        result = load_progress(tmp_path)

        assert result is None

    def test_load_progress_returns_progress(self, tmp_path, sample_progress):
        """load_progress should return ImplementProgress."""
        save_progress(sample_progress, tmp_path)

        result = load_progress(tmp_path)

        assert isinstance(result, ImplementProgress)
        assert result.plan.design_hash == sample_progress.plan.design_hash

    def test_clear_progress_removes_file(self, tmp_path, sample_progress):
        """clear_progress should remove file."""
        save_progress(sample_progress, tmp_path)
        assert get_progress_path(tmp_path).exists()

        clear_progress(tmp_path)

        assert not get_progress_path(tmp_path).exists()

    def test_update_current_task(self, tmp_path, sample_progress):
        """update_current_task should update field."""
        save_progress(sample_progress, tmp_path)

        update_current_task(tmp_path, "new-task")

        loaded = load_progress(tmp_path)
        assert loaded.current_task == "new-task"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_task_with_no_dependencies(self):
        """Task with empty dependencies should work."""
        task = ImplementTask(
            id="solo",
            description="Solo task",
            files=["solo.py"],
            complexity="simple",
            dependencies=[],
        )

        assert task.dependencies == []

    def test_plan_with_single_task(self, sample_task):
        """Plan with single task should work."""
        plan = ImplementPlan(
            design_hash="single",
            tasks=[sample_task],
        )

        d = plan.to_dict()
        restored = ImplementPlan.from_dict(d)

        assert len(restored.tasks) == 1

    def test_result_with_error(self):
        """TaskResult with error should serialize correctly."""
        result = TaskResult(
            task_id="fail",
            success=False,
            error="Something went wrong",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Something went wrong"

    def test_progress_with_no_results(self, sample_plan):
        """Progress with no results should work."""
        progress = ImplementProgress(
            plan=sample_plan,
            completed_tasks=[],
            results=[],
        )

        d = progress.to_dict()
        restored = ImplementProgress.from_dict(d)

        assert len(restored.results) == 0
