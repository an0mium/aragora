"""
Tests for implementation type definitions.

Tests cover:
- ImplementTask dataclass
- ImplementPlan dataclass
- TaskResult dataclass
- ImplementProgress dataclass
"""

from datetime import datetime

import pytest

from aragora.implement.types import (
    ImplementPlan,
    ImplementProgress,
    ImplementTask,
    TaskResult,
)


# ============================================================================
# ImplementTask Tests
# ============================================================================


class TestImplementTask:
    """Tests for ImplementTask dataclass."""

    def test_creation(self):
        """Test basic creation."""
        task = ImplementTask(
            id="task-1",
            description="Add new feature",
            files=["src/feature.py", "tests/test_feature.py"],
            complexity="moderate",
        )
        assert task.id == "task-1"
        assert task.description == "Add new feature"
        assert len(task.files) == 2
        assert task.complexity == "moderate"

    def test_default_dependencies(self):
        """Test default empty dependencies."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["file.py"],
            complexity="simple",
        )
        assert task.dependencies == []

    def test_with_dependencies(self):
        """Test task with dependencies."""
        task = ImplementTask(
            id="t2",
            description="Depends on t1",
            files=["file.py"],
            complexity="simple",
            dependencies=["t1"],
        )
        assert task.dependencies == ["t1"]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        task = ImplementTask(
            id="task-1",
            description="Test task",
            files=["a.py", "b.py"],
            complexity="complex",
            dependencies=["task-0"],
        )
        result = task.to_dict()

        assert result["id"] == "task-1"
        assert result["description"] == "Test task"
        assert result["files"] == ["a.py", "b.py"]
        assert result["complexity"] == "complex"
        assert result["dependencies"] == ["task-0"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "task-2",
            "description": "From dict",
            "files": ["x.py"],
            "complexity": "simple",
            "dependencies": ["task-1"],
        }
        task = ImplementTask.from_dict(data)

        assert task.id == "task-2"
        assert task.description == "From dict"
        assert task.files == ["x.py"]
        assert task.complexity == "simple"
        assert task.dependencies == ["task-1"]

    def test_from_dict_defaults(self):
        """Test deserialization with missing optional fields."""
        data = {
            "id": "task-3",
            "description": "Minimal",
        }
        task = ImplementTask.from_dict(data)

        assert task.id == "task-3"
        assert task.files == []
        assert task.complexity == "moderate"  # Default
        assert task.dependencies == []

    def test_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = ImplementTask(
            id="roundtrip",
            description="Test roundtrip",
            files=["file1.py", "file2.py"],
            complexity="moderate",
            dependencies=["dep1", "dep2"],
        )
        restored = ImplementTask.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.files == original.files
        assert restored.complexity == original.complexity
        assert restored.dependencies == original.dependencies


# ============================================================================
# ImplementPlan Tests
# ============================================================================


class TestImplementPlan:
    """Tests for ImplementPlan dataclass."""

    def test_creation(self):
        """Test basic creation."""
        task = ImplementTask("t1", "Test", ["f.py"], "simple")
        plan = ImplementPlan(
            design_hash="abc123",
            tasks=[task],
        )
        assert plan.design_hash == "abc123"
        assert len(plan.tasks) == 1
        assert isinstance(plan.created_at, datetime)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        task = ImplementTask("t1", "Test", ["f.py"], "simple")
        plan = ImplementPlan(
            design_hash="hash123",
            tasks=[task],
        )
        result = plan.to_dict()

        assert result["design_hash"] == "hash123"
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["id"] == "t1"
        assert "created_at" in result

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "design_hash": "hash456",
            "tasks": [
                {"id": "t1", "description": "Task 1", "files": [], "complexity": "simple"},
                {"id": "t2", "description": "Task 2", "files": ["a.py"], "complexity": "moderate"},
            ],
            "created_at": "2024-01-15T10:30:00",
        }
        plan = ImplementPlan.from_dict(data)

        assert plan.design_hash == "hash456"
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "t1"
        assert plan.tasks[1].id == "t2"
        assert plan.created_at.year == 2024

    def test_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        tasks = [
            ImplementTask("t1", "First", ["a.py"], "simple"),
            ImplementTask("t2", "Second", ["b.py"], "moderate", ["t1"]),
        ]
        original = ImplementPlan(design_hash="roundtrip", tasks=tasks)
        restored = ImplementPlan.from_dict(original.to_dict())

        assert restored.design_hash == original.design_hash
        assert len(restored.tasks) == len(original.tasks)
        assert restored.tasks[1].dependencies == ["t1"]


# ============================================================================
# TaskResult Tests
# ============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_creation_success(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="+ added line",
            model_used="claude",
            duration_seconds=45.5,
        )
        assert result.task_id == "task-1"
        assert result.success is True
        assert result.diff == "+ added line"
        assert result.error is None
        assert result.model_used == "claude"
        assert result.duration_seconds == 45.5

    def test_creation_failure(self):
        """Test failed task result."""
        result = TaskResult(
            task_id="task-2",
            success=False,
            error="Timeout: operation timed out",
            model_used="codex",
            duration_seconds=120.0,
        )
        assert result.success is False
        assert result.error == "Timeout: operation timed out"
        assert result.diff == ""

    def test_default_values(self):
        """Test default values."""
        result = TaskResult(task_id="t", success=True)
        assert result.diff == ""
        assert result.error is None
        assert result.model_used is None
        assert result.duration_seconds == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="some diff",
            model_used="claude",
            duration_seconds=30.0,
        )
        data = result.to_dict()

        assert data["task_id"] == "task-1"
        assert data["success"] is True
        assert data["diff"] == "some diff"
        assert data["error"] is None
        assert data["model_used"] == "claude"
        assert data["duration_seconds"] == 30.0


# ============================================================================
# ImplementProgress Tests
# ============================================================================


class TestImplementProgress:
    """Tests for ImplementProgress dataclass."""

    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        tasks = [
            ImplementTask("t1", "Task 1", ["a.py"], "simple"),
            ImplementTask("t2", "Task 2", ["b.py"], "moderate", ["t1"]),
        ]
        return ImplementPlan(design_hash="test_hash", tasks=tasks)

    def test_creation(self, sample_plan):
        """Test basic creation."""
        progress = ImplementProgress(plan=sample_plan)

        assert progress.plan == sample_plan
        assert progress.completed_tasks == []
        assert progress.current_task is None
        assert progress.git_stash_ref is None
        assert progress.results == []

    def test_with_progress(self, sample_plan):
        """Test with progress data."""
        result = TaskResult("t1", success=True, diff="diff", model_used="claude")
        progress = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["t1"],
            current_task="t2",
            git_stash_ref="stash@{0}",
            results=[result],
        )

        assert progress.completed_tasks == ["t1"]
        assert progress.current_task == "t2"
        assert progress.git_stash_ref == "stash@{0}"
        assert len(progress.results) == 1

    def test_to_dict(self, sample_plan):
        """Test serialization to dictionary."""
        result = TaskResult("t1", True, "diff", model_used="claude", duration_seconds=10.0)
        progress = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["t1"],
            current_task="t2",
            results=[result],
        )
        data = progress.to_dict()

        assert data["plan"]["design_hash"] == "test_hash"
        assert data["completed_tasks"] == ["t1"]
        assert data["current_task"] == "t2"
        assert len(data["results"]) == 1
        assert data["results"][0]["task_id"] == "t1"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "plan": {
                "design_hash": "restored_hash",
                "tasks": [
                    {"id": "t1", "description": "Restored", "files": [], "complexity": "simple"}
                ],
                "created_at": "2024-01-15T12:00:00",
            },
            "completed_tasks": ["t1"],
            "current_task": "t2",
            "git_stash_ref": "stash@{1}",
            "results": [
                {
                    "task_id": "t1",
                    "success": True,
                    "diff": "restored diff",
                    "model_used": "claude",
                    "duration_seconds": 25.0,
                }
            ],
        }
        progress = ImplementProgress.from_dict(data)

        assert progress.plan.design_hash == "restored_hash"
        assert progress.completed_tasks == ["t1"]
        assert progress.current_task == "t2"
        assert progress.git_stash_ref == "stash@{1}"
        assert len(progress.results) == 1
        assert progress.results[0].duration_seconds == 25.0

    def test_from_dict_defaults(self):
        """Test deserialization with missing optional fields."""
        data = {
            "plan": {
                "design_hash": "minimal",
                "tasks": [],
                "created_at": "2024-01-01T00:00:00",
            }
        }
        progress = ImplementProgress.from_dict(data)

        assert progress.completed_tasks == []
        assert progress.current_task is None
        assert progress.git_stash_ref is None
        assert progress.results == []

    def test_roundtrip(self, sample_plan):
        """Test serialization/deserialization roundtrip."""
        original = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["t1"],
            current_task="t2",
            git_stash_ref="stash@{2}",
            results=[
                TaskResult("t1", True, "diff1", model_used="claude", duration_seconds=15.0),
            ],
        )
        restored = ImplementProgress.from_dict(original.to_dict())

        assert restored.plan.design_hash == original.plan.design_hash
        assert restored.completed_tasks == original.completed_tasks
        assert restored.current_task == original.current_task
        assert restored.git_stash_ref == original.git_stash_ref
        assert len(restored.results) == len(original.results)
        assert restored.results[0].model_used == "claude"
