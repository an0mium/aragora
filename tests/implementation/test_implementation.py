"""
Tests for aragora.implement module.

Tests cover:
- types.py: ImplementTask, ImplementPlan, TaskResult, ImplementProgress
- checkpoint.py: save_progress, load_progress, clear_progress
- planner.py: extract_json, validate_plan, create_single_task_plan
- executor.py: HybridExecutor (with mocking)
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement import (
    HybridExecutor,
    ImplementPlan,
    ImplementProgress,
    ImplementTask,
    TaskResult,
    clear_progress,
    create_single_task_plan,
    load_progress,
    save_progress,
)
from aragora.implement.checkpoint import get_progress_path, update_current_task
from aragora.implement.planner import extract_json, validate_plan


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository directory."""
    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_task() -> ImplementTask:
    """Create a sample implementation task."""
    return ImplementTask(
        id="task-1",
        description="Add a rate limiter module",
        files=["src/rate_limiter.py", "tests/test_rate_limiter.py"],
        complexity="moderate",
        dependencies=[],
    )


@pytest.fixture
def sample_task_complex() -> ImplementTask:
    """Create a complex implementation task."""
    return ImplementTask(
        id="task-2",
        description="Refactor authentication system",
        files=[
            "src/auth/handler.py",
            "src/auth/tokens.py",
            "src/auth/middleware.py",
            "tests/auth/test_handler.py",
        ],
        complexity="complex",
        dependencies=["task-1"],
    )


@pytest.fixture
def sample_plan(sample_task: ImplementTask, sample_task_complex: ImplementTask) -> ImplementPlan:
    """Create a sample implementation plan."""
    return ImplementPlan(
        design_hash="abc123def456",
        tasks=[sample_task, sample_task_complex],
        created_at=datetime(2026, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_result() -> TaskResult:
    """Create a sample task result."""
    return TaskResult(
        task_id="task-1",
        success=True,
        diff="1 file changed, 50 insertions(+)",
        error=None,
        model_used="claude",
        duration_seconds=45.5,
    )


@pytest.fixture
def sample_progress(sample_plan: ImplementPlan, sample_result: TaskResult) -> ImplementProgress:
    """Create a sample progress checkpoint."""
    return ImplementProgress(
        plan=sample_plan,
        completed_tasks=["task-1"],
        current_task="task-2",
        git_stash_ref="stash@{0}",
        results=[sample_result],
    )


# =============================================================================
# Tests for types.py - ImplementTask
# =============================================================================


class TestImplementTask:
    """Tests for ImplementTask dataclass."""

    def test_task_creation(self, sample_task: ImplementTask):
        """Test task creation with all fields."""
        assert sample_task.id == "task-1"
        assert sample_task.description == "Add a rate limiter module"
        assert len(sample_task.files) == 2
        assert sample_task.complexity == "moderate"
        assert sample_task.dependencies == []

    def test_task_with_dependencies(self, sample_task_complex: ImplementTask):
        """Test task with dependencies."""
        assert sample_task_complex.dependencies == ["task-1"]

    def test_task_to_dict(self, sample_task: ImplementTask):
        """Test task serialization."""
        data = sample_task.to_dict()
        assert data["id"] == "task-1"
        assert data["description"] == "Add a rate limiter module"
        assert data["files"] == ["src/rate_limiter.py", "tests/test_rate_limiter.py"]
        assert data["complexity"] == "moderate"
        assert data["dependencies"] == []

    def test_task_from_dict(self):
        """Test task deserialization."""
        data = {
            "id": "task-new",
            "description": "New feature",
            "files": ["new.py"],
            "complexity": "simple",
            "dependencies": ["task-0"],
        }
        task = ImplementTask.from_dict(data)
        assert task.id == "task-new"
        assert task.description == "New feature"
        assert task.files == ["new.py"]
        assert task.complexity == "simple"
        assert task.dependencies == ["task-0"]

    def test_task_from_dict_defaults(self):
        """Test task deserialization with missing optional fields."""
        data = {
            "id": "task-min",
            "description": "Minimal task",
        }
        task = ImplementTask.from_dict(data)
        assert task.id == "task-min"
        assert task.files == []
        assert task.complexity == "moderate"
        assert task.dependencies == []

    def test_task_roundtrip(self, sample_task: ImplementTask):
        """Test task serialization roundtrip."""
        data = sample_task.to_dict()
        restored = ImplementTask.from_dict(data)
        assert restored.id == sample_task.id
        assert restored.description == sample_task.description
        assert restored.files == sample_task.files
        assert restored.complexity == sample_task.complexity
        assert restored.dependencies == sample_task.dependencies


# =============================================================================
# Tests for types.py - ImplementPlan
# =============================================================================


class TestImplementPlan:
    """Tests for ImplementPlan dataclass."""

    def test_plan_creation(self, sample_plan: ImplementPlan):
        """Test plan creation."""
        assert sample_plan.design_hash == "abc123def456"
        assert len(sample_plan.tasks) == 2
        assert sample_plan.created_at == datetime(2026, 1, 15, 10, 30, 0)

    def test_plan_to_dict(self, sample_plan: ImplementPlan):
        """Test plan serialization."""
        data = sample_plan.to_dict()
        assert data["design_hash"] == "abc123def456"
        assert len(data["tasks"]) == 2
        assert data["created_at"] == "2026-01-15T10:30:00"

    def test_plan_from_dict(self):
        """Test plan deserialization."""
        data = {
            "design_hash": "hash123",
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test task",
                    "files": ["test.py"],
                    "complexity": "simple",
                    "dependencies": [],
                }
            ],
            "created_at": "2026-01-01T12:00:00",
        }
        plan = ImplementPlan.from_dict(data)
        assert plan.design_hash == "hash123"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "task-1"
        assert plan.created_at == datetime(2026, 1, 1, 12, 0, 0)

    def test_plan_roundtrip(self, sample_plan: ImplementPlan):
        """Test plan serialization roundtrip."""
        data = sample_plan.to_dict()
        restored = ImplementPlan.from_dict(data)
        assert restored.design_hash == sample_plan.design_hash
        assert len(restored.tasks) == len(sample_plan.tasks)
        assert restored.created_at == sample_plan.created_at


# =============================================================================
# Tests for types.py - TaskResult
# =============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_result_success(self, sample_result: TaskResult):
        """Test successful result."""
        assert sample_result.task_id == "task-1"
        assert sample_result.success is True
        assert sample_result.diff == "1 file changed, 50 insertions(+)"
        assert sample_result.error is None
        assert sample_result.model_used == "claude"
        assert sample_result.duration_seconds == 45.5

    def test_result_failure(self):
        """Test failed result."""
        result = TaskResult(
            task_id="task-2",
            success=False,
            diff="",
            error="Timeout after 600s",
            model_used="claude",
            duration_seconds=600.0,
        )
        assert result.success is False
        assert result.error == "Timeout after 600s"

    def test_result_to_dict(self, sample_result: TaskResult):
        """Test result serialization."""
        data = sample_result.to_dict()
        assert data["task_id"] == "task-1"
        assert data["success"] is True
        assert data["diff"] == "1 file changed, 50 insertions(+)"
        assert data["error"] is None
        assert data["model_used"] == "claude"
        assert data["duration_seconds"] == 45.5


# =============================================================================
# Tests for types.py - ImplementProgress
# =============================================================================


class TestImplementProgress:
    """Tests for ImplementProgress dataclass."""

    def test_progress_creation(self, sample_progress: ImplementProgress):
        """Test progress creation."""
        assert sample_progress.completed_tasks == ["task-1"]
        assert sample_progress.current_task == "task-2"
        assert sample_progress.git_stash_ref == "stash@{0}"
        assert len(sample_progress.results) == 1

    def test_progress_to_dict(self, sample_progress: ImplementProgress):
        """Test progress serialization."""
        data = sample_progress.to_dict()
        assert data["completed_tasks"] == ["task-1"]
        assert data["current_task"] == "task-2"
        assert data["git_stash_ref"] == "stash@{0}"
        assert len(data["results"]) == 1
        assert data["plan"]["design_hash"] == "abc123def456"

    def test_progress_from_dict(self, sample_progress: ImplementProgress):
        """Test progress deserialization."""
        data = sample_progress.to_dict()
        restored = ImplementProgress.from_dict(data)
        assert restored.completed_tasks == sample_progress.completed_tasks
        assert restored.current_task == sample_progress.current_task
        assert restored.git_stash_ref == sample_progress.git_stash_ref
        assert len(restored.results) == len(sample_progress.results)

    def test_progress_roundtrip(self, sample_progress: ImplementProgress):
        """Test progress serialization roundtrip."""
        data = sample_progress.to_dict()
        restored = ImplementProgress.from_dict(data)
        assert restored.completed_tasks == sample_progress.completed_tasks
        assert restored.plan.design_hash == sample_progress.plan.design_hash


# =============================================================================
# Tests for checkpoint.py
# =============================================================================


class TestCheckpoint:
    """Tests for checkpoint save/load operations."""

    def test_get_progress_path(self, temp_repo: Path):
        """Test progress path construction."""
        path = get_progress_path(temp_repo)
        assert path == temp_repo / ".nomic" / "implement_progress.json"

    def test_save_progress(self, temp_repo: Path, sample_progress: ImplementProgress):
        """Test saving progress to disk."""
        save_progress(sample_progress, temp_repo)
        progress_path = get_progress_path(temp_repo)
        assert progress_path.exists()

        with open(progress_path) as f:
            data = json.load(f)
        assert data["completed_tasks"] == ["task-1"]

    def test_load_progress_success(self, temp_repo: Path, sample_progress: ImplementProgress):
        """Test loading progress from disk."""
        save_progress(sample_progress, temp_repo)
        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.completed_tasks == ["task-1"]
        assert loaded.current_task == "task-2"

    def test_load_progress_missing(self, temp_repo: Path):
        """Test loading when no progress file exists."""
        loaded = load_progress(temp_repo)
        assert loaded is None

    def test_load_progress_corrupted(self, temp_repo: Path):
        """Test loading corrupted progress file."""
        progress_path = get_progress_path(temp_repo)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w") as f:
            f.write("not valid json {{{")

        loaded = load_progress(temp_repo)
        assert loaded is None

    def test_clear_progress(self, temp_repo: Path, sample_progress: ImplementProgress):
        """Test clearing progress file."""
        save_progress(sample_progress, temp_repo)
        assert get_progress_path(temp_repo).exists()

        clear_progress(temp_repo)
        assert not get_progress_path(temp_repo).exists()

    def test_clear_progress_nonexistent(self, temp_repo: Path):
        """Test clearing when no file exists (should not error)."""
        clear_progress(temp_repo)  # Should not raise

    def test_update_current_task(self, temp_repo: Path, sample_progress: ImplementProgress):
        """Test updating just the current task."""
        save_progress(sample_progress, temp_repo)
        update_current_task(temp_repo, "task-3")

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.current_task == "task-3"

    def test_save_atomic_creates_parent_dir(
        self, tmp_path: Path, sample_progress: ImplementProgress
    ):
        """Test that save creates .nomic directory if missing."""
        repo = tmp_path / "new_repo"
        repo.mkdir()
        # No .nomic directory exists

        save_progress(sample_progress, repo)
        assert (repo / ".nomic" / "implement_progress.json").exists()


# =============================================================================
# Tests for planner.py - extract_json
# =============================================================================


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = """Here's the plan:

```json
{"tasks": [{"id": "task-1"}]}
```

Done!"""
        result = extract_json(text)
        assert '{"tasks"' in result

    def test_extract_from_bare_code_block(self):
        """Test extracting JSON from code block without language."""
        text = """
```
{"key": "value"}
```
"""
        result = extract_json(text)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_extract_raw_json(self):
        """Test extracting raw JSON object."""
        text = 'The plan is {"tasks": []} and more text'
        result = extract_json(text)
        data = json.loads(result)
        assert "tasks" in data

    def test_extract_multiline_json(self):
        """Test extracting multiline JSON."""
        text = """{
  "tasks": [
    {"id": "task-1"},
    {"id": "task-2"}
  ]
}"""
        result = extract_json(text)
        data = json.loads(result)
        assert len(data["tasks"]) == 2

    def test_extract_no_json(self):
        """Test when no JSON present."""
        text = "No JSON here"
        result = extract_json(text)
        assert result == text


# =============================================================================
# Tests for planner.py - validate_plan
# =============================================================================


class TestValidatePlan:
    """Tests for plan validation."""

    def test_validate_valid_plan(self):
        """Test validating a correct plan."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First task",
                    "files": ["file.py"],
                    "complexity": "simple",
                    "dependencies": [],
                }
            ]
        }
        errors = validate_plan(plan)
        assert errors == []

    def test_validate_missing_tasks(self):
        """Test validation when tasks key is missing."""
        plan: dict[str, Any] = {}
        errors = validate_plan(plan)
        assert "Missing 'tasks' key" in errors[0]

    def test_validate_tasks_not_list(self):
        """Test validation when tasks is not a list."""
        plan = {"tasks": "not a list"}
        errors = validate_plan(plan)
        assert "'tasks' must be a list" in errors[0]

    def test_validate_empty_tasks(self):
        """Test validation when tasks list is empty."""
        plan: dict[str, Any] = {"tasks": []}
        errors = validate_plan(plan)
        assert "Plan has no tasks" in errors[0]

    def test_validate_task_not_dict(self):
        """Test validation when task is not a dict."""
        plan = {"tasks": ["not a dict"]}
        errors = validate_plan(plan)
        assert "Task 0 is not a dict" in errors[0]

    def test_validate_missing_id(self):
        """Test validation when task is missing id."""
        plan = {"tasks": [{"description": "Task without id", "files": [], "complexity": "simple"}]}
        errors = validate_plan(plan)
        assert any("missing 'id'" in e for e in errors)

    def test_validate_missing_description(self):
        """Test validation when task is missing description."""
        plan = {"tasks": [{"id": "task-1", "files": [], "complexity": "simple"}]}
        errors = validate_plan(plan)
        assert any("missing 'description'" in e for e in errors)

    def test_validate_missing_files(self):
        """Test validation when task is missing files."""
        plan = {"tasks": [{"id": "task-1", "description": "Task", "complexity": "simple"}]}
        errors = validate_plan(plan)
        assert any("missing or invalid 'files'" in e for e in errors)

    def test_validate_invalid_complexity(self):
        """Test validation with invalid complexity value."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Task",
                    "files": [],
                    "complexity": "invalid",
                }
            ]
        }
        errors = validate_plan(plan)
        assert any("invalid complexity" in e for e in errors)

    def test_validate_duplicate_ids(self):
        """Test validation catches duplicate task IDs."""
        plan = {
            "tasks": [
                {
                    "id": "task-1",
                    "description": "First",
                    "files": [],
                    "complexity": "simple",
                },
                {
                    "id": "task-1",
                    "description": "Duplicate",
                    "files": [],
                    "complexity": "simple",
                },
            ]
        }
        errors = validate_plan(plan)
        assert any("Duplicate task id" in e for e in errors)


# =============================================================================
# Tests for planner.py - create_single_task_plan
# =============================================================================


class TestCreateSingleTaskPlan:
    """Tests for fallback single-task plan creation."""

    def test_single_task_plan(self, temp_repo: Path):
        """Test creating a single-task fallback plan."""
        design = "Implement a new feature with multiple components"
        plan = create_single_task_plan(design, temp_repo)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "task-1"
        assert plan.tasks[0].complexity == "complex"
        assert plan.tasks[0].files == []
        assert plan.design_hash is not None

    def test_single_task_plan_hash_deterministic(self, temp_repo: Path):
        """Test that the same design produces the same hash."""
        design = "Same design"
        plan1 = create_single_task_plan(design, temp_repo)
        plan2 = create_single_task_plan(design, temp_repo)
        assert plan1.design_hash == plan2.design_hash

    def test_single_task_plan_hash_varies(self, temp_repo: Path):
        """Test that different designs produce different hashes."""
        plan1 = create_single_task_plan("Design A", temp_repo)
        plan2 = create_single_task_plan("Design B", temp_repo)
        assert plan1.design_hash != plan2.design_hash


# =============================================================================
# Tests for executor.py - Timeout Calculation
# =============================================================================


class TestTimeoutCalculation:
    """Tests for complexity-based timeout calculation."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    def test_timeout_simple_single_file(self, executor: HybridExecutor):
        """Test timeout for simple single-file task."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=["file.py"],
            complexity="simple",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 300  # 5 minutes base

    def test_timeout_moderate_multiple_files(self, executor: HybridExecutor):
        """Test timeout for moderate task with multiple files."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=["a.py", "b.py", "c.py"],
            complexity="moderate",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        # 600 base + 2 * 120 = 840
        assert timeout == 840

    def test_timeout_complex_many_files(self, executor: HybridExecutor):
        """Test timeout for complex task with many files."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=["a.py", "b.py", "c.py", "d.py", "e.py"],
            complexity="complex",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        # 1200 base + 4 * 120 = 1680
        assert timeout == 1680

    def test_timeout_caps_at_30_minutes(self, executor: HybridExecutor):
        """Test timeout is capped at 30 minutes."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=[f"file{i}.py" for i in range(20)],  # Many files
            complexity="complex",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 1800  # Capped at 30 minutes

    def test_timeout_empty_files(self, executor: HybridExecutor):
        """Test timeout with no files uses default count of 1."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=[],
            complexity="simple",
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 300  # No file bonus added

    def test_timeout_unknown_complexity(self, executor: HybridExecutor):
        """Test timeout with unknown complexity uses moderate default."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=["file.py"],
            complexity="unknown",  # type: ignore
        )
        with patch.dict(os.environ, {"IMPL_COMPLEXITY_TIMEOUT": "1"}):
            timeout = executor._get_task_timeout(task)
        assert timeout == 600  # Moderate default

    def test_timeout_feature_flag_disabled(self, executor: HybridExecutor):
        """Test that disabling feature flag uses default timeout."""
        task = ImplementTask(
            id="test",
            description="Test",
            files=["a.py", "b.py", "c.py"],
            complexity="complex",
        )
        # Patch the module-level constant directly
        with patch("aragora.implement.executor.COMPLEXITY_TIMEOUT", False):
            timeout = executor._get_task_timeout(task)
        assert timeout == executor.claude_timeout


# =============================================================================
# Tests for executor.py - Prompt Building
# =============================================================================


class TestPromptBuilding:
    """Tests for task prompt construction."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    def test_build_prompt_with_files(self, executor: HybridExecutor, sample_task: ImplementTask):
        """Test prompt includes file list."""
        prompt = executor._build_prompt(sample_task)
        assert "src/rate_limiter.py" in prompt
        assert "tests/test_rate_limiter.py" in prompt
        assert "Add a rate limiter module" in prompt

    def test_build_prompt_no_files(self, executor: HybridExecutor):
        """Test prompt handles empty file list."""
        task = ImplementTask(
            id="test",
            description="Do something",
            files=[],
            complexity="simple",
        )
        prompt = executor._build_prompt(task)
        assert "(determine from description)" in prompt

    def test_build_prompt_includes_repo_path(
        self, executor: HybridExecutor, sample_task: ImplementTask
    ):
        """Test prompt includes repository path."""
        prompt = executor._build_prompt(sample_task)
        assert str(executor.repo_path) in prompt


# =============================================================================
# Tests for executor.py - Git Operations
# =============================================================================


class TestGitOperations:
    """Tests for git diff operations."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    def test_git_diff_success(self, executor: HybridExecutor):
        """Test successful git diff."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="1 file changed")
            diff = executor._get_git_diff()
        assert diff == "1 file changed"

    def test_git_diff_timeout(self, executor: HybridExecutor):
        """Test git diff timeout handling."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", 180)
            diff = executor._get_git_diff()
        assert diff == ""

    def test_git_diff_subprocess_error(self, executor: HybridExecutor):
        """Test git diff subprocess error handling."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.SubprocessError("git not found")
            diff = executor._get_git_diff()
        assert diff == ""

    def test_git_diff_os_error(self, executor: HybridExecutor):
        """Test git diff OS error handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("No such file")
            diff = executor._get_git_diff()
        assert diff == ""


# =============================================================================
# Tests for executor.py - Task Execution (with mocking)
# =============================================================================


class TestTaskExecution:
    """Tests for task execution with mocked agents."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    @pytest.mark.asyncio
    async def test_execute_task_success(self, executor: HybridExecutor, sample_task: ImplementTask):
        """Test successful task execution."""
        # Mock the claude agent
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(return_value="Code generated")
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        # Create a mock context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=None)
        mock_context.__exit__ = MagicMock(return_value=None)

        with patch.object(executor, "_get_git_diff", return_value="1 file changed"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.execute_task(sample_task)

        assert result.success is True
        assert result.task_id == "task-1"
        assert result.model_used == "claude"
        assert result.diff == "1 file changed"

    @pytest.mark.asyncio
    async def test_execute_task_timeout(self, executor: HybridExecutor, sample_task: ImplementTask):
        """Test task execution timeout."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(side_effect=TimeoutError("Timed out"))
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=None)
        mock_context.__exit__ = MagicMock(return_value=None)

        with patch(
            "aragora.server.stream.arena_hooks.streaming_task_context",
            return_value=mock_context,
        ):
            result = await executor.execute_task(sample_task)

        assert result.success is False
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_task_fallback(
        self, executor: HybridExecutor, sample_task: ImplementTask
    ):
        """Test fallback to Codex when use_fallback=True."""
        mock_codex = AsyncMock()
        mock_codex.generate = AsyncMock(return_value="Codex code")
        mock_codex.name = "codex-test"
        mock_codex.timeout = 600
        executor._codex = mock_codex

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=None)
        mock_context.__exit__ = MagicMock(return_value=None)

        with patch.object(executor, "_get_git_diff", return_value="changes"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.execute_task(sample_task, attempt=3, use_fallback=True)

        assert result.success is True
        assert result.model_used == "codex-fallback"


# =============================================================================
# Tests for executor.py - Plan Execution
# =============================================================================


class TestPlanExecution:
    """Tests for plan execution with mocked agents."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo, max_retries=1)

    @pytest.fixture
    def mock_context(self):
        """Create a mock context manager."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=None)
        return ctx

    @pytest.mark.asyncio
    async def test_execute_plan_success(
        self, executor: HybridExecutor, sample_task: ImplementTask, mock_context
    ):
        """Test successful plan execution."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(return_value="Done")
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        with patch.object(executor, "_get_git_diff", return_value="changes"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                results = await executor.execute_plan([sample_task], completed=set())

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_plan_skips_completed(
        self, executor: HybridExecutor, sample_task: ImplementTask
    ):
        """Test that already-completed tasks are skipped."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(return_value="Done")
        executor._claude = mock_claude

        results = await executor.execute_plan(
            [sample_task],
            completed={"task-1"},  # Already completed
        )

        assert len(results) == 0
        mock_claude.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_plan_respects_dependencies(
        self,
        executor: HybridExecutor,
        sample_task: ImplementTask,
        sample_task_complex: ImplementTask,
        mock_context,
    ):
        """Test that tasks with unmet dependencies are skipped."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(return_value="Done")
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        with patch.object(executor, "_get_git_diff", return_value="changes"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                # Execute with task-2 first (depends on task-1)
                results = await executor.execute_plan(
                    [sample_task_complex, sample_task], completed=set()
                )

        # task-2 should be skipped initially due to dependency
        # Only task-1 should execute first, then task-2 on dependency being met
        task_ids_executed = [r.task_id for r in results]
        assert "task-1" in task_ids_executed

    @pytest.mark.asyncio
    async def test_execute_plan_with_callback(
        self, executor: HybridExecutor, sample_task: ImplementTask, mock_context
    ):
        """Test callback is called on task completion."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(return_value="Done")
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        callback = MagicMock()

        with patch.object(executor, "_get_git_diff", return_value="changes"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                await executor.execute_plan(
                    [sample_task], completed=set(), on_task_complete=callback
                )

        callback.assert_called_once()


# =============================================================================
# Tests for executor.py - Agent Selection
# =============================================================================


class TestAgentSelection:
    """Tests for agent selection logic."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    def test_select_agent_always_claude(self, executor: HybridExecutor):
        """Test that Claude is always selected for implementation."""
        for complexity in ["simple", "moderate", "complex"]:
            agent, name = executor._select_agent(complexity)
            assert name == "claude"


# =============================================================================
# Tests for executor.py - Code Review
# =============================================================================


class TestCodeReview:
    """Tests for optional Codex code review."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor for testing."""
        return HybridExecutor(temp_repo)

    @pytest.fixture
    def mock_context(self):
        """Create a mock context manager."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=None)
        return ctx

    @pytest.mark.asyncio
    async def test_review_empty_diff(self, executor: HybridExecutor):
        """Test review with empty diff returns approved."""
        result = await executor.review_with_codex("")
        assert result["approved"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_review_with_diff(self, executor: HybridExecutor, mock_context):
        """Test review with actual diff."""
        mock_codex = AsyncMock()
        mock_codex.generate = AsyncMock(
            return_value="APPROVED: yes\nISSUES: None\nSUGGESTIONS: Add docstring"
        )
        mock_codex.name = "codex-reviewer"

        with patch("aragora.implement.executor.CodexAgent", return_value=mock_codex):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.review_with_codex("1 file changed")

        assert result["approved"] is True
        assert "review" in result

    @pytest.mark.asyncio
    async def test_review_not_approved(self, executor: HybridExecutor, mock_context):
        """Test review that is not approved."""
        mock_codex = AsyncMock()
        mock_codex.generate = AsyncMock(
            return_value="APPROVED: no\nISSUES: Security vulnerability found"
        )
        mock_codex.name = "codex-reviewer"

        with patch("aragora.implement.executor.CodexAgent", return_value=mock_codex):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.review_with_codex("1 file changed")

        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_review_error_handling(self, executor: HybridExecutor, mock_context):
        """Test review error handling."""
        mock_codex = AsyncMock()
        mock_codex.generate = AsyncMock(side_effect=Exception("API error"))
        mock_codex.name = "codex-reviewer"

        with patch("aragora.implement.executor.CodexAgent", return_value=mock_codex):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.review_with_codex("diff content")

        assert result["approved"] is None
        assert "error" in result


# =============================================================================
# Tests for executor.py - Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for retry and fallback logic."""

    @pytest.fixture
    def executor(self, temp_repo: Path) -> HybridExecutor:
        """Create executor with retry enabled."""
        return HybridExecutor(temp_repo, max_retries=3)

    @pytest.fixture
    def mock_context(self):
        """Create a mock context manager."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=None)
        return ctx

    @pytest.mark.asyncio
    async def test_retry_on_timeout(
        self, executor: HybridExecutor, sample_task: ImplementTask, mock_context
    ):
        """Test retry on timeout error."""
        # First call times out, second succeeds
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(side_effect=[TimeoutError("Timeout"), "Success"])
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        with patch.object(executor, "_get_git_diff", return_value="changes"):
            with patch(
                "aragora.server.stream.arena_hooks.streaming_task_context",
                return_value=mock_context,
            ):
                result = await executor.execute_task_with_retry(sample_task)

        assert result.success is True
        assert mock_claude.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_timeout(
        self, executor: HybridExecutor, sample_task: ImplementTask, mock_context
    ):
        """Test no retry on non-timeout error."""
        mock_claude = AsyncMock()
        mock_claude.generate = AsyncMock(side_effect=ValueError("Bad input"))
        mock_claude.name = "claude-test"
        executor._claude = mock_claude

        with patch(
            "aragora.server.stream.arena_hooks.streaming_task_context",
            return_value=mock_context,
        ):
            result = await executor.execute_task_with_retry(sample_task)

        assert result.success is False
        # Should not retry non-timeout errors with extended timeout
        assert mock_claude.generate.call_count == 1
