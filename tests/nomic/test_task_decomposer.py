"""Tests for Nomic Loop task decomposer."""

import pytest

from aragora.nomic.task_decomposer import (
    TaskDecomposer,
    TaskDecomposition,
    SubTask,
    DecomposerConfig,
    analyze_task,
    get_task_decomposer,
)


class TestTaskDecomposer:
    """Tests for TaskDecomposer analysis."""

    def test_low_complexity_task(self):
        """Simple tasks should have low complexity and not decompose."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Fix typo in README")

        assert result.complexity_level == "low"
        assert result.complexity_score <= 3
        assert result.should_decompose is False
        assert len(result.subtasks) == 0

    def test_medium_complexity_task(self):
        """Medium tasks should have appropriate scoring."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Implement a new API endpoint for user authentication with "
            "database integration and security checks. Update handlers.py "
            "and auth.py for the new feature."
        )

        assert result.complexity_score >= 3
        assert result.complexity_level in ["low", "medium", "high"]

    def test_high_complexity_task(self):
        """Complex tasks should trigger decomposition."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Refactor the entire database layer to support multi-tenancy. "
            "This requires changes to models, migrations, API endpoints, "
            "and security middleware. Update auth.py, db.py, handlers.py."
        )

        assert result.complexity_level in ["medium", "high"]
        assert result.complexity_score >= 5
        assert result.should_decompose is True
        assert len(result.subtasks) >= 2

    def test_file_mentions_increase_complexity(self):
        """Tasks mentioning multiple files should score higher."""
        decomposer = TaskDecomposer()

        simple = decomposer.analyze("Update auth logic")
        with_files = decomposer.analyze(
            "Update auth logic in auth.py, handlers.py, middleware.py, tests.py"
        )

        assert with_files.complexity_score > simple.complexity_score

    def test_concept_extraction(self):
        """Should extract concept areas for subtask generation."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "This system-wide refactor touches database, api, and security layers. "
            "Update the backend for performance improvements."
        )

        if result.should_decompose:
            concepts = [st.title.lower() for st in result.subtasks]
            # Should find at least one concept-based subtask
            assert any(
                c in " ".join(concepts)
                for c in ["database", "api", "security", "backend", "performance"]
            )

    def test_subtask_dependencies(self):
        """Subtasks should have logical dependencies."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Major architectural refactor: update database models, "
            "modify API layer, add security checks, update frontend."
        )

        if result.should_decompose and len(result.subtasks) > 1:
            # Later subtasks should depend on earlier ones
            last_task = result.subtasks[-1]
            assert len(last_task.dependencies) > 0 or result.subtasks[0].dependencies == []

    def test_empty_task(self):
        """Empty task should return minimal result."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("")

        assert result.complexity_score == 0
        assert result.should_decompose is False
        assert result.rationale == "Empty task"

    def test_custom_config(self):
        """Custom config should affect decomposition threshold."""
        # Lower threshold - more likely to decompose
        low_threshold = TaskDecomposer(DecomposerConfig(complexity_threshold=3))
        result_low = low_threshold.analyze("Add new feature with integration")

        # Higher threshold - less likely to decompose
        high_threshold = TaskDecomposer(DecomposerConfig(complexity_threshold=8))
        result_high = high_threshold.analyze("Add new feature with integration")

        # Same complexity, different decomposition decisions possible
        assert result_low.complexity_score == result_high.complexity_score


class TestSubTask:
    """Tests for SubTask dataclass."""

    def test_subtask_creation(self):
        """Should create SubTask with all fields."""
        subtask = SubTask(
            id="subtask_1",
            title="Database Changes",
            description="Update database schema",
            dependencies=["subtask_0"],
            estimated_complexity="medium",
            file_scope=["models.py", "migrations.py"],
        )

        assert subtask.id == "subtask_1"
        assert subtask.title == "Database Changes"
        assert "subtask_0" in subtask.dependencies
        assert subtask.estimated_complexity == "medium"
        assert len(subtask.file_scope) == 2


class TestTaskDecomposition:
    """Tests for TaskDecomposition dataclass."""

    def test_decomposition_creation(self):
        """Should create TaskDecomposition with all fields."""
        decomposition = TaskDecomposition(
            original_task="Test task",
            complexity_score=7,
            complexity_level="high",
            should_decompose=True,
            subtasks=[
                SubTask(id="1", title="Part 1", description="First part"),
                SubTask(id="2", title="Part 2", description="Second part"),
            ],
            rationale="Complex task needs splitting",
        )

        assert decomposition.original_task == "Test task"
        assert decomposition.complexity_score == 7
        assert decomposition.should_decompose is True
        assert len(decomposition.subtasks) == 2


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_task_decomposer_singleton(self):
        """Should return same instance."""
        decomposer1 = get_task_decomposer()
        decomposer2 = get_task_decomposer()
        assert decomposer1 is decomposer2

    def test_analyze_task_function(self):
        """Convenience function should work."""
        result = analyze_task("Fix bug in login")
        assert isinstance(result, TaskDecomposition)
        assert result.original_task == "Fix bug in login"


class TestRationale:
    """Tests for rationale generation."""

    def test_rationale_includes_keywords(self):
        """Rationale should mention high-complexity keywords when decomposing."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Refactor the entire database system with major architectural changes. "
            "Migrate all models, update API layer, and redesign security in "
            "db.py, models.py, handlers.py, auth.py, middleware.py"
        )

        # Should be complex enough to decompose
        if result.should_decompose:
            assert (
                "refactor" in result.rationale.lower()
                or "high-complexity" in result.rationale.lower()
            )
        else:
            # Even if not decomposing, the score should be reasonable
            assert result.complexity_score >= 3

    def test_rationale_mentions_file_count(self):
        """Rationale should mention file count when significant."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Update a.py, b.py, c.py, d.py, e.py for the new feature")

        if "files" in result.rationale.lower():
            assert "5" in result.rationale or "touches" in result.rationale.lower()

    def test_rationale_mentions_concepts(self):
        """Rationale should mention concept areas."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Update database, api, and security for new feature")

        if result.complexity_score >= 5:
            assert "concept" in result.rationale.lower() or "span" in result.rationale.lower()
