"""
Tests for checkpoint/progress persistence.

Tests cover:
- get_progress_path function
- save_progress function
- load_progress function
- clear_progress function
- update_current_task function
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from aragora.implement.checkpoint import (
    clear_progress,
    get_progress_path,
    load_progress,
    save_progress,
    update_current_task,
)
from aragora.implement.types import (
    ImplementPlan,
    ImplementProgress,
    ImplementTask,
    TaskResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_repo():
    """Create a temporary repository directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_plan():
    """Create a sample implementation plan."""
    tasks = [
        ImplementTask("t1", "Task 1", ["a.py"], "simple"),
        ImplementTask("t2", "Task 2", ["b.py"], "moderate", ["t1"]),
    ]
    return ImplementPlan(design_hash="test_hash_123", tasks=tasks)


@pytest.fixture
def sample_progress(sample_plan):
    """Create a sample progress object."""
    return ImplementProgress(
        plan=sample_plan,
        completed_tasks=["t1"],
        current_task="t2",
        git_stash_ref="stash@{0}",
        results=[
            TaskResult("t1", True, "diff content", model_used="claude", duration_seconds=30.0),
        ],
    )


# ============================================================================
# get_progress_path Tests
# ============================================================================


class TestGetProgressPath:
    """Tests for get_progress_path function."""

    def test_returns_path_in_nomic_dir(self, temp_repo):
        """Test progress path is in .nomic directory."""
        path = get_progress_path(temp_repo)

        assert path.parent.name == ".nomic"
        assert path.name == "implement_progress.json"
        assert str(temp_repo) in str(path)

    def test_path_is_absolute(self, temp_repo):
        """Test returned path is absolute."""
        path = get_progress_path(temp_repo)
        assert path.is_absolute()


# ============================================================================
# save_progress Tests
# ============================================================================


class TestSaveProgress:
    """Tests for save_progress function."""

    def test_creates_nomic_directory(self, temp_repo, sample_progress):
        """Test .nomic directory is created if needed."""
        nomic_dir = temp_repo / ".nomic"
        assert not nomic_dir.exists()

        save_progress(sample_progress, temp_repo)

        assert nomic_dir.exists()
        assert nomic_dir.is_dir()

    def test_creates_progress_file(self, temp_repo, sample_progress):
        """Test progress file is created."""
        save_progress(sample_progress, temp_repo)

        progress_path = get_progress_path(temp_repo)
        assert progress_path.exists()

    def test_file_contains_valid_json(self, temp_repo, sample_progress):
        """Test progress file contains valid JSON."""
        save_progress(sample_progress, temp_repo)

        progress_path = get_progress_path(temp_repo)
        with open(progress_path) as f:
            data = json.load(f)

        assert "plan" in data
        assert "completed_tasks" in data
        assert "current_task" in data

    def test_saves_all_fields(self, temp_repo, sample_progress):
        """Test all fields are saved correctly."""
        save_progress(sample_progress, temp_repo)

        progress_path = get_progress_path(temp_repo)
        with open(progress_path) as f:
            data = json.load(f)

        assert data["plan"]["design_hash"] == "test_hash_123"
        assert data["completed_tasks"] == ["t1"]
        assert data["current_task"] == "t2"
        assert data["git_stash_ref"] == "stash@{0}"
        assert len(data["results"]) == 1

    def test_overwrites_existing_file(self, temp_repo, sample_plan):
        """Test saving overwrites existing progress."""
        # Save initial progress
        progress1 = ImplementProgress(plan=sample_plan, completed_tasks=["t1"])
        save_progress(progress1, temp_repo)

        # Save updated progress
        progress2 = ImplementProgress(plan=sample_plan, completed_tasks=["t1", "t2"])
        save_progress(progress2, temp_repo)

        # Verify only the updated progress exists
        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.completed_tasks == ["t1", "t2"]

    def test_atomic_write_no_temp_files(self, temp_repo, sample_progress):
        """Test no temp files are left after successful save."""
        save_progress(sample_progress, temp_repo)

        nomic_dir = temp_repo / ".nomic"
        files = list(nomic_dir.iterdir())

        # Should only have the progress file
        assert len(files) == 1
        assert files[0].name == "implement_progress.json"


# ============================================================================
# load_progress Tests
# ============================================================================


class TestLoadProgress:
    """Tests for load_progress function."""

    def test_returns_none_if_no_file(self, temp_repo):
        """Test returns None when no progress file exists."""
        result = load_progress(temp_repo)
        assert result is None

    def test_loads_saved_progress(self, temp_repo, sample_progress):
        """Test loads previously saved progress."""
        save_progress(sample_progress, temp_repo)
        loaded = load_progress(temp_repo)

        assert loaded is not None
        assert loaded.plan.design_hash == "test_hash_123"
        assert loaded.completed_tasks == ["t1"]
        assert loaded.current_task == "t2"

    def test_loads_results(self, temp_repo, sample_progress):
        """Test loads task results correctly."""
        save_progress(sample_progress, temp_repo)
        loaded = load_progress(temp_repo)

        assert loaded is not None
        assert len(loaded.results) == 1
        assert loaded.results[0].task_id == "t1"
        assert loaded.results[0].success is True
        assert loaded.results[0].model_used == "claude"

    def test_returns_none_for_corrupted_json(self, temp_repo):
        """Test returns None for corrupted JSON file."""
        progress_path = get_progress_path(temp_repo)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w") as f:
            f.write("{ invalid json }")

        result = load_progress(temp_repo)
        assert result is None

    def test_returns_none_for_invalid_structure(self, temp_repo):
        """Test returns None for valid JSON with invalid structure."""
        progress_path = get_progress_path(temp_repo)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_path, "w") as f:
            json.dump({"wrong": "structure"}, f)

        result = load_progress(temp_repo)
        assert result is None


# ============================================================================
# clear_progress Tests
# ============================================================================


class TestClearProgress:
    """Tests for clear_progress function."""

    def test_removes_progress_file(self, temp_repo, sample_progress):
        """Test progress file is removed."""
        save_progress(sample_progress, temp_repo)
        progress_path = get_progress_path(temp_repo)
        assert progress_path.exists()

        clear_progress(temp_repo)

        assert not progress_path.exists()

    def test_no_error_if_no_file(self, temp_repo):
        """Test no error when clearing non-existent progress."""
        # Should not raise
        clear_progress(temp_repo)

    def test_load_returns_none_after_clear(self, temp_repo, sample_progress):
        """Test load returns None after clear."""
        save_progress(sample_progress, temp_repo)
        clear_progress(temp_repo)

        result = load_progress(temp_repo)
        assert result is None


# ============================================================================
# update_current_task Tests
# ============================================================================


class TestUpdateCurrentTask:
    """Tests for update_current_task function."""

    def test_updates_current_task(self, temp_repo, sample_progress):
        """Test current task is updated."""
        save_progress(sample_progress, temp_repo)

        update_current_task(temp_repo, "t3")

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.current_task == "t3"

    def test_preserves_other_fields(self, temp_repo, sample_progress):
        """Test other fields are preserved."""
        save_progress(sample_progress, temp_repo)

        update_current_task(temp_repo, "t3")

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.completed_tasks == ["t1"]
        assert loaded.git_stash_ref == "stash@{0}"
        assert len(loaded.results) == 1

    def test_no_op_if_no_progress(self, temp_repo):
        """Test no error when no progress exists."""
        # Should not raise
        update_current_task(temp_repo, "t1")

        # Should still be None
        result = load_progress(temp_repo)
        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestCheckpointIntegration:
    """Integration tests for checkpoint operations."""

    def test_full_workflow(self, temp_repo, sample_plan):
        """Test complete save/load/update/clear workflow."""
        # 1. Start fresh - no progress
        assert load_progress(temp_repo) is None

        # 2. Save initial progress
        progress = ImplementProgress(plan=sample_plan)
        save_progress(progress, temp_repo)

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.completed_tasks == []

        # 3. Update with task completion
        progress.completed_tasks.append("t1")
        progress.current_task = "t2"
        progress.results.append(TaskResult("t1", True, "diff", model_used="claude"))
        save_progress(progress, temp_repo)

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.completed_tasks == ["t1"]
        assert loaded.current_task == "t2"

        # 4. Update current task
        update_current_task(temp_repo, "t3")

        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.current_task == "t3"
        assert loaded.completed_tasks == ["t1"]  # Preserved

        # 5. Clear on completion
        clear_progress(temp_repo)
        assert load_progress(temp_repo) is None

    def test_concurrent_saves(self, temp_repo, sample_plan):
        """Test multiple rapid saves don't corrupt data."""
        for i in range(10):
            progress = ImplementProgress(
                plan=sample_plan,
                completed_tasks=[f"t{j}" for j in range(i)],
                current_task=f"t{i}",
            )
            save_progress(progress, temp_repo)

        # Final state should be intact
        loaded = load_progress(temp_repo)
        assert loaded is not None
        assert loaded.current_task == "t9"
        assert len(loaded.completed_tasks) == 9
