"""
Tests for implement module type definitions.

Tests cover:
- ImplementTask dataclass
- ImplementPlan dataclass
- TaskResult dataclass
- ImplementProgress dataclass
- Serialization/deserialization (to_dict/from_dict)
"""

import pytest
from datetime import datetime

from aragora.implement.types import (
    ImplementTask,
    ImplementPlan,
    TaskResult,
    ImplementProgress,
)


class TestImplementTask:
    """Tests for ImplementTask dataclass."""

    def test_create_basic_task(self):
        """Creates task with required fields."""
        task = ImplementTask(
            id="task-1",
            description="Add a new function",
            files=["src/utils.py"],
            complexity="simple",
        )

        assert task.id == "task-1"
        assert task.description == "Add a new function"
        assert task.files == ["src/utils.py"]
        assert task.complexity == "simple"
        assert task.dependencies == []

    def test_create_task_with_dependencies(self):
        """Creates task with dependencies."""
        task = ImplementTask(
            id="task-2",
            description="Add tests",
            files=["tests/test_utils.py"],
            complexity="moderate",
            dependencies=["task-1"],
        )

        assert task.dependencies == ["task-1"]

    def test_to_dict(self):
        """to_dict serializes all fields."""
        task = ImplementTask(
            id="task-1",
            description="Implement feature",
            files=["src/feature.py", "tests/test_feature.py"],
            complexity="moderate",
            dependencies=["task-0"],
        )

        data = task.to_dict()

        assert data["id"] == "task-1"
        assert data["description"] == "Implement feature"
        assert data["files"] == ["src/feature.py", "tests/test_feature.py"]
        assert data["complexity"] == "moderate"
        assert data["dependencies"] == ["task-0"]

    def test_from_dict(self):
        """from_dict deserializes all fields."""
        data = {
            "id": "task-3",
            "description": "Refactor module",
            "files": ["src/module.py"],
            "complexity": "complex",
            "dependencies": ["task-1", "task-2"],
        }

        task = ImplementTask.from_dict(data)

        assert task.id == "task-3"
        assert task.description == "Refactor module"
        assert task.files == ["src/module.py"]
        assert task.complexity == "complex"
        assert task.dependencies == ["task-1", "task-2"]

    def test_from_dict_with_defaults(self):
        """from_dict uses defaults for missing optional fields."""
        data = {
            "id": "task-1",
            "description": "Simple task",
        }

        task = ImplementTask.from_dict(data)

        assert task.files == []
        assert task.complexity == "moderate"
        assert task.dependencies == []

    def test_roundtrip_serialization(self):
        """to_dict and from_dict are inverse operations."""
        original = ImplementTask(
            id="roundtrip-task",
            description="Test roundtrip",
            files=["a.py", "b.py"],
            complexity="complex",
            dependencies=["dep-1"],
        )

        restored = ImplementTask.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.files == original.files
        assert restored.complexity == original.complexity
        assert restored.dependencies == original.dependencies


class TestImplementPlan:
    """Tests for ImplementPlan dataclass."""

    def test_create_plan(self):
        """Creates plan with tasks."""
        tasks = [
            ImplementTask(
                id="task-1",
                description="First task",
                files=["file1.py"],
                complexity="simple",
            ),
            ImplementTask(
                id="task-2",
                description="Second task",
                files=["file2.py"],
                complexity="moderate",
                dependencies=["task-1"],
            ),
        ]

        plan = ImplementPlan(design_hash="abc123", tasks=tasks)

        assert plan.design_hash == "abc123"
        assert len(plan.tasks) == 2
        assert isinstance(plan.created_at, datetime)

    def test_to_dict(self):
        """to_dict serializes plan with tasks."""
        tasks = [
            ImplementTask(
                id="task-1",
                description="Task",
                files=["f.py"],
                complexity="simple",
            ),
        ]
        plan = ImplementPlan(design_hash="hash123", tasks=tasks)

        data = plan.to_dict()

        assert data["design_hash"] == "hash123"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "task-1"
        assert "created_at" in data

    def test_from_dict(self):
        """from_dict deserializes plan with tasks."""
        data = {
            "design_hash": "hash456",
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test task",
                    "files": ["test.py"],
                    "complexity": "simple",
                    "dependencies": [],
                }
            ],
            "created_at": "2025-01-15T10:30:00",
        }

        plan = ImplementPlan.from_dict(data)

        assert plan.design_hash == "hash456"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "task-1"
        assert plan.created_at.year == 2025

    def test_roundtrip_serialization(self):
        """to_dict and from_dict preserve plan data."""
        tasks = [
            ImplementTask(
                id="t1",
                description="Desc 1",
                files=["a.py"],
                complexity="simple",
            ),
            ImplementTask(
                id="t2",
                description="Desc 2",
                files=["b.py"],
                complexity="complex",
                dependencies=["t1"],
            ),
        ]
        original = ImplementPlan(design_hash="original_hash", tasks=tasks)

        restored = ImplementPlan.from_dict(original.to_dict())

        assert restored.design_hash == original.design_hash
        assert len(restored.tasks) == len(original.tasks)
        assert restored.tasks[0].id == original.tasks[0].id
        assert restored.tasks[1].dependencies == original.tasks[1].dependencies


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_create_success_result(self):
        """Creates successful result."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="+ added line",
            model_used="claude",
            duration_seconds=45.2,
        )

        assert result.task_id == "task-1"
        assert result.success is True
        assert result.diff == "+ added line"
        assert result.error is None
        assert result.model_used == "claude"
        assert result.duration_seconds == 45.2

    def test_create_failure_result(self):
        """Creates failure result with error."""
        result = TaskResult(
            task_id="task-2",
            success=False,
            error="Timeout after 600s",
            model_used="codex",
            duration_seconds=600.0,
        )

        assert result.success is False
        assert result.error == "Timeout after 600s"
        assert result.diff == ""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="changes",
            error=None,
            model_used="claude",
            duration_seconds=30.5,
        )

        data = result.to_dict()

        assert data["task_id"] == "task-1"
        assert data["success"] is True
        assert data["diff"] == "changes"
        assert data["error"] is None
        assert data["model_used"] == "claude"
        assert data["duration_seconds"] == 30.5

    def test_default_values(self):
        """TaskResult has sensible defaults."""
        result = TaskResult(task_id="minimal", success=True)

        assert result.diff == ""
        assert result.error is None
        assert result.model_used is None
        assert result.duration_seconds == 0.0


class TestImplementProgress:
    """Tests for ImplementProgress dataclass."""

    @pytest.fixture
    def sample_plan(self):
        """Sample plan for tests."""
        tasks = [
            ImplementTask(
                id="task-1",
                description="First",
                files=["a.py"],
                complexity="simple",
            ),
            ImplementTask(
                id="task-2",
                description="Second",
                files=["b.py"],
                complexity="moderate",
                dependencies=["task-1"],
            ),
        ]
        return ImplementPlan(design_hash="test_hash", tasks=tasks)

    def test_create_progress(self, sample_plan):
        """Creates progress with plan."""
        progress = ImplementProgress(plan=sample_plan)

        assert progress.plan == sample_plan
        assert progress.completed_tasks == []
        assert progress.current_task is None
        assert progress.git_stash_ref is None
        assert progress.results == []

    def test_create_progress_with_state(self, sample_plan):
        """Creates progress with completed tasks."""
        result = TaskResult(task_id="task-1", success=True, diff="changes")

        progress = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["task-1"],
            current_task="task-2",
            git_stash_ref="stash@{0}",
            results=[result],
        )

        assert progress.completed_tasks == ["task-1"]
        assert progress.current_task == "task-2"
        assert progress.git_stash_ref == "stash@{0}"
        assert len(progress.results) == 1

    def test_to_dict(self, sample_plan):
        """to_dict serializes progress."""
        result = TaskResult(task_id="task-1", success=True)
        progress = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["task-1"],
            current_task="task-2",
            results=[result],
        )

        data = progress.to_dict()

        assert "plan" in data
        assert data["completed_tasks"] == ["task-1"]
        assert data["current_task"] == "task-2"
        assert len(data["results"]) == 1

    def test_from_dict(self, sample_plan):
        """from_dict deserializes progress."""
        data = {
            "plan": sample_plan.to_dict(),
            "completed_tasks": ["task-1"],
            "current_task": "task-2",
            "git_stash_ref": "stash@{1}",
            "results": [
                {
                    "task_id": "task-1",
                    "success": True,
                    "diff": "changes",
                    "error": None,
                    "model_used": "claude",
                    "duration_seconds": 25.0,
                }
            ],
        }

        progress = ImplementProgress.from_dict(data)

        assert progress.plan.design_hash == sample_plan.design_hash
        assert progress.completed_tasks == ["task-1"]
        assert progress.current_task == "task-2"
        assert progress.git_stash_ref == "stash@{1}"
        assert len(progress.results) == 1
        assert progress.results[0].success is True

    def test_roundtrip_serialization(self, sample_plan):
        """to_dict and from_dict preserve progress."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="diff",
            model_used="claude",
            duration_seconds=10.0,
        )
        original = ImplementProgress(
            plan=sample_plan,
            completed_tasks=["task-1"],
            current_task="task-2",
            git_stash_ref="ref",
            results=[result],
        )

        restored = ImplementProgress.from_dict(original.to_dict())

        assert restored.plan.design_hash == original.plan.design_hash
        assert restored.completed_tasks == original.completed_tasks
        assert restored.current_task == original.current_task
        assert restored.git_stash_ref == original.git_stash_ref
        assert len(restored.results) == len(original.results)
