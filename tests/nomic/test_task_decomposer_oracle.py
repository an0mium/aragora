"""Tests for oracle-driven task decomposition validation.

Tests file-independence validation, oracle checks (syntax, existence),
and decomposition quality scoring added in Tier 3.
"""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest

from aragora.nomic.task_decomposer import (
    DecompositionQuality,
    FileConflict,
    OracleResult,
    SubTask,
    TaskDecomposer,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def decomposer() -> TaskDecomposer:
    return TaskDecomposer()


def _make_subtask(
    id: str,
    file_scope: list[str] | None = None,
    title: str = "",
    description: str = "",
) -> SubTask:
    """Helper to create a SubTask with minimal boilerplate."""
    return SubTask(
        id=id,
        title=title or f"Task {id}",
        description=description or f"Description for {id}",
        file_scope=file_scope or [],
    )


# =========================================================================
# validate_file_independence tests
# =========================================================================


class TestFileIndependence:
    """Tests for TaskDecomposer.validate_file_independence."""

    def test_no_overlaps_returns_empty(self, decomposer: TaskDecomposer) -> None:
        """Non-overlapping file scopes produce zero conflicts."""
        subtasks = [
            _make_subtask("s1", file_scope=["aragora/debate/orchestrator.py"]),
            _make_subtask("s2", file_scope=["aragora/agents/cli_agents.py"]),
            _make_subtask("s3", file_scope=["aragora/memory/consensus.py"]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert conflicts == []

    def test_single_overlap_detected(self, decomposer: TaskDecomposer) -> None:
        """Two subtasks sharing one file produces one conflict."""
        subtasks = [
            _make_subtask("s1", file_scope=["aragora/server/unified_server.py"]),
            _make_subtask("s2", file_scope=["aragora/server/unified_server.py"]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert len(conflicts) == 1
        assert conflicts[0].file_path == "aragora/server/unified_server.py"
        assert set(conflicts[0].subtask_ids) == {"s1", "s2"}

    def test_multiple_overlaps_detected(self, decomposer: TaskDecomposer) -> None:
        """Multiple shared files each produce a separate conflict."""
        subtasks = [
            _make_subtask("s1", file_scope=["a.py", "b.py", "c.py"]),
            _make_subtask("s2", file_scope=["b.py", "d.py"]),
            _make_subtask("s3", file_scope=["c.py", "e.py"]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        conflict_files = {c.file_path for c in conflicts}
        assert "b.py" in conflict_files
        assert "c.py" in conflict_files
        assert len(conflicts) == 2

    def test_three_way_overlap(self, decomposer: TaskDecomposer) -> None:
        """Three subtasks sharing the same file lists all three."""
        subtasks = [
            _make_subtask("s1", file_scope=["shared.py"]),
            _make_subtask("s2", file_scope=["shared.py"]),
            _make_subtask("s3", file_scope=["shared.py"]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert len(conflicts) == 1
        assert set(conflicts[0].subtask_ids) == {"s1", "s2", "s3"}

    def test_empty_file_scopes_no_conflicts(self, decomposer: TaskDecomposer) -> None:
        """Subtasks with empty file_scope never conflict."""
        subtasks = [
            _make_subtask("s1", file_scope=[]),
            _make_subtask("s2", file_scope=[]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert conflicts == []

    def test_trailing_slash_normalized(self, decomposer: TaskDecomposer) -> None:
        """Trailing slashes are stripped before comparison."""
        subtasks = [
            _make_subtask("s1", file_scope=["aragora/debate/"]),
            _make_subtask("s2", file_scope=["aragora/debate"]),
        ]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert len(conflicts) == 1

    def test_single_subtask_no_conflicts(self, decomposer: TaskDecomposer) -> None:
        """A single subtask can never have conflicts."""
        subtasks = [_make_subtask("s1", file_scope=["a.py", "b.py"])]
        conflicts = decomposer.validate_file_independence(subtasks)
        assert conflicts == []


# =========================================================================
# validate_with_oracle tests
# =========================================================================


class TestOracleValidation:
    """Tests for TaskDecomposer.validate_with_oracle."""

    def test_valid_python_file(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """A valid Python file passes oracle validation."""
        py_file = tmp_path / "valid.py"
        py_file.write_text("x = 1\n")
        subtask = _make_subtask("s1", file_scope=["valid.py"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is True
        assert result.errors == []
        assert "valid.py" in result.checked_files

    def test_invalid_python_syntax(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """A Python file with syntax errors fails oracle validation."""
        py_file = tmp_path / "bad.py"
        py_file.write_text("def foo(\n")
        subtask = _make_subtask("s1", file_scope=["bad.py"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is False
        assert any("Syntax error" in e or "syntax" in e.lower() for e in result.errors)

    def test_missing_file(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """A file that doesn't exist produces an error."""
        subtask = _make_subtask("s1", file_scope=["nonexistent.py"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_directory_scope_skipped(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """Directory-only scopes (trailing slash) are skipped, not checked."""
        subtask = _make_subtask("s1", file_scope=["aragora/debate/"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        # Directories are skipped, so no files checked and result is valid
        assert result.valid is True
        assert result.checked_files == []

    def test_non_python_file_existence_only(
        self, decomposer: TaskDecomposer, tmp_path
    ) -> None:
        """Non-Python files are checked for existence only, not syntax."""
        ts_file = tmp_path / "component.tsx"
        ts_file.write_text("invalid python {{ but valid for existence check")
        subtask = _make_subtask("s1", file_scope=["component.tsx"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is True
        assert "component.tsx" in result.checked_files

    def test_mixed_valid_and_invalid(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """Mix of valid and invalid files: result is invalid, errors list all issues."""
        good = tmp_path / "good.py"
        good.write_text("x = 42\n")
        # bad.py does not exist
        subtask = _make_subtask("s1", file_scope=["good.py", "bad.py"])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is False
        assert len(result.errors) == 1  # Only bad.py has an error
        assert len(result.checked_files) == 2

    def test_empty_file_scope(self, decomposer: TaskDecomposer, tmp_path) -> None:
        """Subtask with no files passes trivially."""
        subtask = _make_subtask("s1", file_scope=[])
        result = decomposer.validate_with_oracle(subtask, worktree_path=str(tmp_path))
        assert result.valid is True
        assert result.checked_files == []
        assert result.errors == []


# =========================================================================
# score_decomposition tests
# =========================================================================


class TestDecompositionQuality:
    """Tests for TaskDecomposer.score_decomposition."""

    def test_perfect_decomposition(self, decomposer: TaskDecomposer) -> None:
        """Non-overlapping subtasks with 1-5 files each score high."""
        subtasks = [
            _make_subtask("s1", file_scope=["a.py", "b.py"]),
            _make_subtask("s2", file_scope=["c.py", "d.py"]),
            _make_subtask("s3", file_scope=["e.py"]),
        ]
        quality = decomposer.score_decomposition(subtasks)
        assert quality.file_conflicts == 0
        assert quality.score >= 0.8
        assert 1 <= quality.avg_scope_size <= 5

    def test_conflicting_decomposition_scores_lower(
        self, decomposer: TaskDecomposer
    ) -> None:
        """Overlapping file scopes reduce the score."""
        subtasks = [
            _make_subtask("s1", file_scope=["shared.py", "a.py"]),
            _make_subtask("s2", file_scope=["shared.py", "b.py"]),
        ]
        quality = decomposer.score_decomposition(subtasks)
        assert quality.file_conflicts == 1
        # With 1 conflict, independence score drops, so overall score should
        # be lower than a conflict-free decomposition with the same shape
        perfect_subtasks = [
            _make_subtask("p1", file_scope=["x.py", "y.py"]),
            _make_subtask("p2", file_scope=["z.py", "w.py"]),
        ]
        perfect_quality = decomposer.score_decomposition(perfect_subtasks)
        assert quality.score < perfect_quality.score

    def test_empty_subtasks_score_zero(self, decomposer: TaskDecomposer) -> None:
        """Empty subtask list scores zero."""
        quality = decomposer.score_decomposition([])
        assert quality.score == 0.0
        assert quality.file_conflicts == 0

    def test_coverage_with_original_scope(self, decomposer: TaskDecomposer) -> None:
        """Coverage ratio reflects how much of the original scope is covered."""
        subtasks = [
            _make_subtask("s1", file_scope=["a.py"]),
            _make_subtask("s2", file_scope=["b.py"]),
        ]
        quality = decomposer.score_decomposition(
            subtasks, original_file_scope=["a.py", "b.py", "c.py"]
        )
        # Covers 2/3 of original scope
        assert 0.6 <= quality.coverage_ratio <= 0.7

    def test_full_coverage(self, decomposer: TaskDecomposer) -> None:
        """Subtasks covering all original files get coverage_ratio 1.0."""
        subtasks = [
            _make_subtask("s1", file_scope=["a.py", "b.py"]),
            _make_subtask("s2", file_scope=["c.py"]),
        ]
        quality = decomposer.score_decomposition(
            subtasks, original_file_scope=["a.py", "b.py", "c.py"]
        )
        assert quality.coverage_ratio == 1.0

    def test_oversized_scope_penalized(self, decomposer: TaskDecomposer) -> None:
        """Subtasks with >5 files per scope get a lower granularity score."""
        subtasks = [
            _make_subtask(
                "s1",
                file_scope=[f"file_{i}.py" for i in range(10)],
            ),
        ]
        quality = decomposer.score_decomposition(subtasks)
        assert quality.avg_scope_size == 10.0
        # Score should be lower due to granularity penalty
        assert quality.score < 0.8

    def test_no_file_scope_neutral(self, decomposer: TaskDecomposer) -> None:
        """Subtasks with empty file_scope get neutral granularity score."""
        subtasks = [
            _make_subtask("s1", file_scope=[]),
            _make_subtask("s2", file_scope=[]),
        ]
        quality = decomposer.score_decomposition(subtasks)
        assert quality.avg_scope_size == 0.0
        # Should not crash, and should give a moderate score
        assert 0.0 < quality.score < 1.0


# =========================================================================
# Dataclass tests
# =========================================================================


class TestDataclasses:
    """Tests for the new dataclass types."""

    def test_file_conflict_repr(self) -> None:
        conflict = FileConflict(file_path="a.py", subtask_ids=["s1", "s2"])
        assert "a.py" in repr(conflict)
        assert "s1" in repr(conflict)

    def test_oracle_result_defaults(self) -> None:
        result = OracleResult(valid=True)
        assert result.errors == []
        assert result.checked_files == []

    def test_decomposition_quality_fields(self) -> None:
        quality = DecompositionQuality(
            score=0.85, file_conflicts=1, avg_scope_size=2.5, coverage_ratio=0.9
        )
        assert quality.score == 0.85
        assert quality.file_conflicts == 1
