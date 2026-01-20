"""Tests for implement module type definitions."""

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

    def test_create_task(self):
        """Basic task creation."""
        task = ImplementTask(
            id="task-1",
            description="Add error handling",
            files=["src/main.py", "src/utils.py"],
            complexity="moderate",
        )
        assert task.id == "task-1"
        assert task.description == "Add error handling"
        assert len(task.files) == 2
        assert task.complexity == "moderate"
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """Task with dependencies."""
        task = ImplementTask(
            id="task-2",
            description="Add tests",
            files=["tests/test_main.py"],
            complexity="simple",
            dependencies=["task-1"],
        )
        assert task.dependencies == ["task-1"]

    def test_task_to_dict(self):
        """Serialization to dictionary."""
        task = ImplementTask(
            id="task-1",
            description="Test",
            files=["file.py"],
            complexity="simple",
            dependencies=["dep-1"],
        )
        data = task.to_dict()
        assert data["id"] == "task-1"
        assert data["description"] == "Test"
        assert data["files"] == ["file.py"]
        assert data["complexity"] == "simple"
        assert data["dependencies"] == ["dep-1"]

    def test_task_from_dict(self):
        """Deserialization from dictionary."""
        data = {
            "id": "task-1",
            "description": "Test task",
            "files": ["a.py", "b.py"],
            "complexity": "complex",
            "dependencies": ["task-0"],
        }
        task = ImplementTask.from_dict(data)
        assert task.id == "task-1"
        assert task.description == "Test task"
        assert task.files == ["a.py", "b.py"]
        assert task.complexity == "complex"
        assert task.dependencies == ["task-0"]

    def test_task_from_dict_defaults(self):
        """Deserialization with missing optional fields."""
        data = {
            "id": "task-1",
            "description": "Minimal task",
        }
        task = ImplementTask.from_dict(data)
        assert task.files == []
        assert task.complexity == "moderate"
        assert task.dependencies == []


class TestImplementPlan:
    """Tests for ImplementPlan dataclass."""

    def test_create_plan(self):
        """Basic plan creation."""
        tasks = [
            ImplementTask(
                id="task-1",
                description="First task",
                files=["a.py"],
                complexity="simple",
            )
        ]
        plan = ImplementPlan(
            design_hash="abc123",
            tasks=tasks,
        )
        assert plan.design_hash == "abc123"
        assert len(plan.tasks) == 1
        assert isinstance(plan.created_at, datetime)

    def test_plan_to_dict(self):
        """Serialization to dictionary."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="simple",
        )
        plan = ImplementPlan(
            design_hash="hash123",
            tasks=[task],
        )
        data = plan.to_dict()
        assert data["design_hash"] == "hash123"
        assert len(data["tasks"]) == 1
        assert "created_at" in data

    def test_plan_from_dict(self):
        """Deserialization from dictionary."""
        data = {
            "design_hash": "hash456",
            "tasks": [
                {
                    "id": "t1",
                    "description": "Task 1",
                    "files": ["f.py"],
                    "complexity": "moderate",
                }
            ],
            "created_at": "2024-01-01T12:00:00",
        }
        plan = ImplementPlan.from_dict(data)
        assert plan.design_hash == "hash456"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "t1"

    def test_plan_roundtrip(self):
        """Serialization roundtrip preserves data."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=["a.py", "b.py"],
            complexity="complex",
            dependencies=["t0"],
        )
        original = ImplementPlan(
            design_hash="original_hash",
            tasks=[task],
        )
        data = original.to_dict()
        restored = ImplementPlan.from_dict(data)
        
        assert restored.design_hash == original.design_hash
        assert len(restored.tasks) == len(original.tasks)
        assert restored.tasks[0].id == original.tasks[0].id


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_create_success_result(self):
        """Successful task result."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            diff="2 files changed",
            model_used="claude",
            duration_seconds=45.5,
        )
        assert result.task_id == "task-1"
        assert result.success
        assert result.error is None

    def test_create_failure_result(self):
        """Failed task result."""
        result = TaskResult(
            task_id="task-1",
            success=False,
            error="Timeout occurred",
            model_used="claude",
            duration_seconds=120.0,
        )
        assert not result.success
        assert result.error == "Timeout occurred"

    def test_result_to_dict(self):
        """Serialization to dictionary."""
        result = TaskResult(
            task_id="t1",
            success=True,
            diff="changed",
            model_used="codex",
            duration_seconds=30.0,
        )
        data = result.to_dict()
        assert data["task_id"] == "t1"
        assert data["success"] is True
        assert data["diff"] == "changed"
        assert data["model_used"] == "codex"
        assert data["duration_seconds"] == 30.0


class TestImplementProgress:
    """Tests for ImplementProgress dataclass."""

    def test_create_progress(self):
        """Basic progress creation."""
        task = ImplementTask(
            id="t1",
            description="Test",
            files=[],
            complexity="simple",
        )
        plan = ImplementPlan(design_hash="hash", tasks=[task])
        progress = ImplementProgress(plan=plan)
        
        assert progress.plan == plan
        assert progress.completed_tasks == []
        assert progress.current_task is None
        assert progress.git_stash_ref is None
        assert progress.results == []

    def test_progress_with_state(self):
        """Progress with partial completion."""
        task1 = ImplementTask(id="t1", description="First", files=[], complexity="simple")
        task2 = ImplementTask(id="t2", description="Second", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task1, task2])
        
        result = TaskResult(task_id="t1", success=True, diff="done")
        
        progress = ImplementProgress(
            plan=plan,
            completed_tasks=["t1"],
            current_task="t2",
            git_stash_ref="stash@{0}",
            results=[result],
        )
        
        assert "t1" in progress.completed_tasks
        assert progress.current_task == "t2"
        assert progress.git_stash_ref == "stash@{0}"
        assert len(progress.results) == 1

    def test_progress_to_dict(self):
        """Serialization to dictionary."""
        task = ImplementTask(id="t1", description="Test", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task])
        result = TaskResult(task_id="t1", success=True)
        
        progress = ImplementProgress(
            plan=plan,
            completed_tasks=["t1"],
            current_task=None,
            results=[result],
        )
        
        data = progress.to_dict()
        assert "plan" in data
        assert data["completed_tasks"] == ["t1"]
        assert len(data["results"]) == 1

    def test_progress_from_dict(self):
        """Deserialization from dictionary."""
        data = {
            "plan": {
                "design_hash": "hash123",
                "tasks": [
                    {"id": "t1", "description": "Task", "files": [], "complexity": "simple"}
                ],
                "created_at": "2024-01-01T12:00:00",
            },
            "completed_tasks": ["t1"],
            "current_task": None,
            "git_stash_ref": "ref123",
            "results": [
                {"task_id": "t1", "success": True, "diff": "changed", "error": None}
            ],
        }
        
        progress = ImplementProgress.from_dict(data)
        assert progress.plan.design_hash == "hash123"
        assert progress.completed_tasks == ["t1"]
        assert progress.git_stash_ref == "ref123"
        assert len(progress.results) == 1

    def test_progress_roundtrip(self):
        """Serialization roundtrip preserves data."""
        task = ImplementTask(id="t1", description="Test", files=["f.py"], complexity="moderate")
        plan = ImplementPlan(design_hash="test_hash", tasks=[task])
        result = TaskResult(task_id="t1", success=True, diff="2 files", duration_seconds=30.0)
        
        original = ImplementProgress(
            plan=plan,
            completed_tasks=["t1"],
            current_task=None,
            git_stash_ref="stash@{0}",
            results=[result],
        )
        
        data = original.to_dict()
        restored = ImplementProgress.from_dict(data)
        
        assert restored.plan.design_hash == original.plan.design_hash
        assert restored.completed_tasks == original.completed_tasks
        assert restored.git_stash_ref == original.git_stash_ref
        assert len(restored.results) == len(original.results)
