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
        """Rationale should mention concept areas or expansion sources."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Update database, api, and security for new feature")

        if result.complexity_score >= 5:
            rationale_lower = result.rationale.lower()
            assert (
                "concept" in rationale_lower
                or "span" in rationale_lower
                or "expanded" in rationale_lower
            )


class TestDepthLimits:
    """Tests for recursive decomposition depth limits."""

    def test_depth_limit_prevents_decomposition(self):
        """analyze() at max depth should return should_decompose=False."""
        config = DecomposerConfig(max_depth=2, complexity_threshold=3)
        decomposer = TaskDecomposer(config=config)

        # This task would normally trigger decomposition
        result = decomposer.analyze(
            "Refactor the entire database system with major architectural changes. "
            "Migrate all models and redesign security in db.py, models.py",
            depth=2,  # At max depth
        )

        assert result.should_decompose is False
        assert "depth" in result.rationale.lower()
        assert len(result.subtasks) == 0

    def test_depth_below_limit_allows_decomposition(self):
        """analyze() below max depth should still allow decomposition."""
        config = DecomposerConfig(max_depth=3, complexity_threshold=3)
        decomposer = TaskDecomposer(config=config)

        result = decomposer.analyze(
            "Refactor the entire database system with major architectural changes. "
            "Migrate all models, update API layer, and redesign security in "
            "db.py, models.py, handlers.py, auth.py, middleware.py",
            depth=1,
        )

        # Should still be allowed to decompose at depth=1 (below max_depth=3)
        if result.complexity_score >= config.complexity_threshold:
            assert result.should_decompose is True

    def test_default_max_depth_is_three(self):
        """Default max_depth should be 3."""
        config = DecomposerConfig()
        assert config.max_depth == 3

    def test_depth_zero_is_default(self):
        """analyze() without depth parameter should default to 0."""
        decomposer = TaskDecomposer()
        # Should work normally (depth=0 is well below max_depth=3)
        result = decomposer.analyze(
            "Refactor the entire system-wide database and redesign security. "
            "Update db.py, models.py, handlers.py, auth.py",
        )
        # Should compute complexity normally
        assert result.complexity_score > 0

    @pytest.mark.parametrize("max_depth", [1, 2, 5])
    def test_configurable_max_depth(self, max_depth):
        """max_depth should be configurable."""
        config = DecomposerConfig(max_depth=max_depth, complexity_threshold=1)
        decomposer = TaskDecomposer(config=config)

        # At exactly max_depth -> blocked
        result = decomposer.analyze("Refactor everything in db.py", depth=max_depth)
        assert result.should_decompose is False

        # One below max_depth -> allowed
        result = decomposer.analyze("Refactor everything in db.py", depth=max_depth - 1)
        # Should compute normally (may or may not decompose based on complexity)


class TestAbstractGoalDetection:
    """Tests for abstract/meta goal detection and scoring improvements."""

    def test_find_highest_impact_bug_scores_high(self):
        """'Find and fix the highest-impact bug' should score >= 7."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Find and fix the highest-impact bug in the codebase")

        assert result.complexity_score >= 7
        assert result.complexity_level == "high"
        assert result.should_decompose is True

    def test_improve_test_coverage_scores_medium_high(self):
        """'Improve test coverage' should score >= 6."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Improve test coverage")

        assert result.complexity_score >= 6
        assert result.should_decompose is True

    def test_question_form_scores_higher(self):
        """Goals phrased as questions should get a complexity boost."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("What is the most fragile part of the system?")

        assert result.complexity_score >= 6
        assert result.should_decompose is True

    def test_superlative_broad_scope_scores_high(self):
        """Goals with superlatives and broad scope words score high."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Identify the most critical security vulnerabilities across the system"
        )

        assert result.complexity_score >= 7
        assert result.should_decompose is True

    def test_abstract_goal_triggers_debate_recommendation(self):
        """Abstract goals (high score, no file hints) should recommend debate."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Find and fix the highest-impact bug in the codebase")

        assert result.recommend_debate is True

    def test_concrete_goal_does_not_trigger_debate(self):
        """Concrete goals with file paths should not recommend debate."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("Fix the bug in auth.py")

        assert result.recommend_debate is False

    def test_auto_debate_abstract_config_flag(self):
        """auto_debate_abstract=False should suppress debate recommendation."""
        config = DecomposerConfig(auto_debate_abstract=False)
        decomposer = TaskDecomposer(config=config)
        result = decomposer.analyze("Find and fix the highest-impact bug in the codebase")

        # Should still score high but not recommend debate
        assert result.complexity_score >= 7
        assert result.recommend_debate is False

    def test_simple_goals_unaffected(self):
        """Simple concrete goals should still score low and not decompose."""
        decomposer = TaskDecomposer()

        result = decomposer.analyze("Fix typo in README")
        assert result.complexity_score <= 3
        assert result.should_decompose is False
        assert result.recommend_debate is False

    def test_file_paths_discount_abstract_bonus(self):
        """Goals with file paths should have reduced abstract bonus."""
        decomposer = TaskDecomposer()

        # Abstract version (no files)
        abstract = decomposer.analyze("Find the best way to optimize performance")
        # Concrete version (with file)
        concrete = decomposer.analyze("Find the best way to optimize performance in server.py")

        assert abstract.complexity_score > concrete.complexity_score

    def test_strategic_verb_with_concept_boosts_score(self):
        """Strategic verbs + domain concepts without files should boost score."""
        decomposer = TaskDecomposer()

        # "optimize" + "performance" concept, no files
        result = decomposer.analyze("Optimize the codebase for performance")
        assert result.complexity_score >= 6

    def test_has_file_hints_method(self):
        """_has_file_hints should detect file and path references."""
        decomposer = TaskDecomposer()

        assert decomposer._has_file_hints("Fix bug in auth.py") is True
        assert decomposer._has_file_hints("Update aragora/server/handlers") is True
        assert decomposer._has_file_hints("Fix tests/nomic/test_decomposer.py") is True
        assert decomposer._has_file_hints("Improve test coverage") is False
        assert decomposer._has_file_hints("Find the highest-impact bug") is False

    def test_recommend_debate_field_default(self):
        """TaskDecomposition.recommend_debate should default to False."""
        result = TaskDecomposition(
            original_task="test",
            complexity_score=5,
            complexity_level="medium",
            should_decompose=True,
        )
        assert result.recommend_debate is False
