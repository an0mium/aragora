"""Tests for checkpoint module crash recovery."""

import json
import os
import pytest
from pathlib import Path

from aragora.implement.checkpoint import (
    PROGRESS_FILENAME,
    get_progress_path,
    save_progress,
    load_progress,
    clear_progress,
    update_current_task,
)
from aragora.implement.types import (
    ImplementTask,
    ImplementPlan,
    TaskResult,
    ImplementProgress,
)


class TestGetProgressPath:
    """Tests for progress path resolution."""

    def test_returns_correct_path(self, tmp_path):
        """Should return .nomic/implement_progress.json."""
        result = get_progress_path(tmp_path)
        expected = tmp_path / ".nomic" / PROGRESS_FILENAME
        assert result == expected

    def test_path_is_pathlib(self, tmp_path):
        """Result should be a Path object."""
        result = get_progress_path(tmp_path)
        assert isinstance(result, Path)


class TestSaveProgress:
    """Tests for atomic progress saving."""

    @pytest.fixture
    def sample_progress(self):
        """Create a sample progress object."""
        task = ImplementTask(
            id="task-1",
            description="Test task",
            files=["test.py"],
            complexity="simple",
        )
        plan = ImplementPlan(design_hash="abc123", tasks=[task])
        return ImplementProgress(
            plan=plan,
            completed_tasks=[],
            current_task="task-1",
        )

    def test_creates_nomic_directory(self, tmp_path, sample_progress):
        """Should create .nomic directory if it doesn't exist."""
        nomic_dir = tmp_path / ".nomic"
        assert not nomic_dir.exists()

        save_progress(sample_progress, tmp_path)

        assert nomic_dir.exists()
        assert nomic_dir.is_dir()

    def test_saves_valid_json(self, tmp_path, sample_progress):
        """Should save valid JSON file."""
        save_progress(sample_progress, tmp_path)

        progress_path = get_progress_path(tmp_path)
        with open(progress_path) as f:
            data = json.load(f)

        assert data["plan"]["design_hash"] == "abc123"
        assert data["current_task"] == "task-1"

    def test_overwrites_existing(self, tmp_path, sample_progress):
        """Should overwrite existing progress file."""
        save_progress(sample_progress, tmp_path)

        # Modify and save again
        sample_progress.current_task = "task-2"
        save_progress(sample_progress, tmp_path)

        progress_path = get_progress_path(tmp_path)
        with open(progress_path) as f:
            data = json.load(f)

        assert data["current_task"] == "task-2"

    def test_atomic_no_temp_files_left(self, tmp_path, sample_progress):
        """Should not leave temp files after save."""
        save_progress(sample_progress, tmp_path)

        nomic_dir = tmp_path / ".nomic"
        files = list(nomic_dir.iterdir())
        temp_files = [f for f in files if f.name.startswith(".progress_")]

        assert len(temp_files) == 0

    def test_file_has_correct_name(self, tmp_path, sample_progress):
        """Progress file should have standard name."""
        save_progress(sample_progress, tmp_path)

        progress_path = get_progress_path(tmp_path)
        assert progress_path.exists()
        assert progress_path.name == PROGRESS_FILENAME


class TestLoadProgress:
    """Tests for progress loading."""

    def test_returns_none_if_missing(self, tmp_path):
        """Should return None if no progress file exists."""
        result = load_progress(tmp_path)
        assert result is None

    def test_loads_saved_progress(self, tmp_path):
        """Should load previously saved progress."""
        task = ImplementTask(
            id="task-1",
            description="Test",
            files=["a.py"],
            complexity="moderate",
        )
        plan = ImplementPlan(design_hash="xyz789", tasks=[task])
        original = ImplementProgress(
            plan=plan,
            completed_tasks=["task-0"],
            current_task="task-1",
            git_stash_ref="stash@{0}",
        )

        save_progress(original, tmp_path)
        loaded = load_progress(tmp_path)

        assert loaded is not None
        assert loaded.plan.design_hash == "xyz789"
        assert loaded.completed_tasks == ["task-0"]
        assert loaded.current_task == "task-1"
        assert loaded.git_stash_ref == "stash@{0}"

    def test_returns_none_for_corrupted_json(self, tmp_path):
        """Should return None for corrupted JSON."""
        progress_path = get_progress_path(tmp_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w") as f:
            f.write("not valid json {{{")

        result = load_progress(tmp_path)
        assert result is None

    def test_returns_none_for_invalid_data(self, tmp_path):
        """Should return None for valid JSON with missing required fields."""
        progress_path = get_progress_path(tmp_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w") as f:
            json.dump({"invalid": "data"}, f)

        result = load_progress(tmp_path)
        assert result is None

    def test_returns_progress_instance(self, tmp_path):
        """Loaded result should be ImplementProgress instance."""
        task = ImplementTask(id="t", description="d", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task])
        original = ImplementProgress(plan=plan)

        save_progress(original, tmp_path)
        loaded = load_progress(tmp_path)

        assert isinstance(loaded, ImplementProgress)


class TestClearProgress:
    """Tests for progress clearing."""

    def test_clears_existing_file(self, tmp_path):
        """Should delete existing progress file."""
        task = ImplementTask(id="t", description="d", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task])
        progress = ImplementProgress(plan=plan)

        save_progress(progress, tmp_path)
        progress_path = get_progress_path(tmp_path)
        assert progress_path.exists()

        clear_progress(tmp_path)
        assert not progress_path.exists()

    def test_handles_nonexistent_file(self, tmp_path):
        """Should not raise if file doesn't exist."""
        clear_progress(tmp_path)  # Should not raise

    def test_preserves_nomic_directory(self, tmp_path):
        """Should not delete .nomic directory."""
        task = ImplementTask(id="t", description="d", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task])
        progress = ImplementProgress(plan=plan)

        save_progress(progress, tmp_path)
        clear_progress(tmp_path)

        nomic_dir = tmp_path / ".nomic"
        assert nomic_dir.exists()


class TestUpdateCurrentTask:
    """Tests for lightweight task updates."""

    def test_updates_current_task(self, tmp_path):
        """Should update current task in existing progress."""
        task1 = ImplementTask(id="t1", description="First", files=[], complexity="simple")
        task2 = ImplementTask(id="t2", description="Second", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="hash", tasks=[task1, task2])
        progress = ImplementProgress(plan=plan, current_task="t1")

        save_progress(progress, tmp_path)
        update_current_task(tmp_path, "t2")

        loaded = load_progress(tmp_path)
        assert loaded.current_task == "t2"

    def test_no_op_if_no_progress(self, tmp_path):
        """Should do nothing if no progress exists."""
        update_current_task(tmp_path, "any-task")

        # Should not create file
        progress_path = get_progress_path(tmp_path)
        assert not progress_path.exists()

    def test_preserves_other_fields(self, tmp_path):
        """Should preserve all other progress fields."""
        task = ImplementTask(id="t1", description="Test", files=["a.py"], complexity="moderate")
        plan = ImplementPlan(design_hash="preserved_hash", tasks=[task])
        result = TaskResult(task_id="t0", success=True, diff="changes")
        progress = ImplementProgress(
            plan=plan,
            completed_tasks=["t0"],
            current_task="t1",
            git_stash_ref="stash@{1}",
            results=[result],
        )

        save_progress(progress, tmp_path)
        update_current_task(tmp_path, "t2")

        loaded = load_progress(tmp_path)
        assert loaded.plan.design_hash == "preserved_hash"
        assert loaded.completed_tasks == ["t0"]
        assert loaded.git_stash_ref == "stash@{1}"
        assert len(loaded.results) == 1


class TestRoundtrip:
    """Integration tests for save/load cycles."""

    def test_full_roundtrip_with_results(self, tmp_path):
        """Complete save/load cycle should preserve all data."""
        tasks = [
            ImplementTask(
                id="t1", description="First task", files=["a.py", "b.py"], complexity="simple"
            ),
            ImplementTask(
                id="t2",
                description="Second task",
                files=["c.py"],
                complexity="complex",
                dependencies=["t1"],
            ),
        ]
        plan = ImplementPlan(design_hash="full_roundtrip_hash", tasks=tasks)
        results = [
            TaskResult(
                task_id="t1",
                success=True,
                diff="Added feature",
                model_used="claude",
                duration_seconds=45.0,
            ),
        ]
        original = ImplementProgress(
            plan=plan,
            completed_tasks=["t1"],
            current_task="t2",
            git_stash_ref="stash@{2}",
            results=results,
        )

        save_progress(original, tmp_path)
        loaded = load_progress(tmp_path)

        assert loaded.plan.design_hash == original.plan.design_hash
        assert len(loaded.plan.tasks) == 2
        assert loaded.plan.tasks[1].dependencies == ["t1"]
        assert loaded.completed_tasks == ["t1"]
        assert loaded.current_task == "t2"
        assert loaded.git_stash_ref == "stash@{2}"
        assert len(loaded.results) == 1
        assert loaded.results[0].success is True

    def test_multiple_save_load_cycles(self, tmp_path):
        """Multiple cycles should work correctly."""
        task = ImplementTask(id="t1", description="Test", files=[], complexity="simple")
        plan = ImplementPlan(design_hash="cycle_test", tasks=[task])
        progress = ImplementProgress(plan=plan, current_task="t1")

        for i in range(5):
            progress.current_task = f"task-{i}"
            save_progress(progress, tmp_path)
            loaded = load_progress(tmp_path)
            assert loaded.current_task == f"task-{i}"
