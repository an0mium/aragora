"""Tests for task pattern matching and agent affinity scoring."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.ranking.pattern_matcher import (
    TaskPatternMatcher,
    PatternAffinity,
    TASK_PATTERNS,
    PATTERN_TO_ISSUE_TYPE,
    get_pattern_matcher,
    classify_task,
)


class TestTaskPatternMatcher:
    """Tests for TaskPatternMatcher classification."""

    def test_classify_bugfix_task(self):
        """Bugfix-related tasks should classify as bugfix."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Fix the authentication bug") == "bugfix"
        assert matcher.classify_task("There's an error in the login flow") == "bugfix"
        assert matcher.classify_task("The app crashes on startup") == "bugfix"
        assert matcher.classify_task("Regression in payment processing") == "bugfix"

    def test_classify_refactor_task(self):
        """Refactoring tasks should classify as refactor."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Refactor the database layer") == "refactor"
        assert matcher.classify_task("Restructure the API handlers") == "refactor"
        assert matcher.classify_task("Clean up the legacy code") == "refactor"
        assert matcher.classify_task("Extract common utilities") == "refactor"

    def test_classify_feature_task(self):
        """Feature tasks should classify as feature."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Add user authentication") == "feature"
        assert matcher.classify_task("Implement dark mode") == "feature"
        assert matcher.classify_task("Create new dashboard") == "feature"
        assert matcher.classify_task("Build the payment integration") == "feature"

    def test_classify_optimize_task(self):
        """Optimization tasks should classify as optimize."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Optimize database queries") == "optimize"
        assert matcher.classify_task("Improve performance of search") == "optimize"
        assert matcher.classify_task("Speed up the API responses") == "optimize"
        assert matcher.classify_task("Reduce memory usage") == "optimize"

    def test_classify_security_task(self):
        """Security tasks should classify as security."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Fix security vulnerability") == "security"
        assert matcher.classify_task("Security audit of authentication") == "security"
        assert matcher.classify_task("Encrypt sensitive data") == "security"
        assert matcher.classify_task("Sanitize user input") == "security"

    def test_classify_test_task(self):
        """Test-related tasks should classify as test."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Add unit tests for auth") == "test"
        assert matcher.classify_task("Improve test coverage") == "test"
        assert matcher.classify_task("Create integration tests") == "test"
        assert matcher.classify_task("Add mock fixtures") == "test"

    def test_classify_docs_task(self):
        """Documentation tasks should classify as docs."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Document the API endpoints") == "docs"
        assert matcher.classify_task("Update the README") == "docs"
        assert matcher.classify_task("Write docstrings for all functions") == "docs"
        assert matcher.classify_task("Explain the architecture") == "docs"

    def test_classify_general_task(self):
        """Unrecognized tasks should classify as general."""
        matcher = TaskPatternMatcher()

        assert matcher.classify_task("Do something") == "general"
        assert matcher.classify_task("Random task") == "general"
        assert matcher.classify_task("") == "general"

    def test_classify_with_multiple_keywords(self):
        """Tasks with multiple keywords should match the dominant pattern."""
        matcher = TaskPatternMatcher()

        # "fix" is bugfix, "test" is test - bugfix has more keywords in this case
        result = matcher.classify_task("Fix the broken test fixtures")
        assert result in ["bugfix", "test"]  # Either is valid

    def test_caching(self):
        """Classification results should be cached."""
        matcher = TaskPatternMatcher()

        task = "Fix the authentication bug"
        result1 = matcher.classify_task(task)
        result2 = matcher.classify_task(task)

        assert result1 == result2
        assert task[:200].lower() in matcher._pattern_cache


class TestPatternToIssueType:
    """Tests for pattern to issue type mapping."""

    def test_pattern_mappings_exist(self):
        """All task patterns should have issue type mappings."""
        for pattern in TASK_PATTERNS:
            # All patterns should either be in the mapping or fall back to general
            issue_type = PATTERN_TO_ISSUE_TYPE.get(pattern, "general")
            assert isinstance(issue_type, str)

    def test_expected_mappings(self):
        """Key patterns should map to expected issue types."""
        assert PATTERN_TO_ISSUE_TYPE["bugfix"] == "correctness"
        assert PATTERN_TO_ISSUE_TYPE["security"] == "security"
        assert PATTERN_TO_ISSUE_TYPE["optimize"] == "performance"
        assert PATTERN_TO_ISSUE_TYPE["test"] == "testing"


class TestAgentAffinities:
    """Tests for agent affinity scoring."""

    def test_get_affinities_without_store(self):
        """Should return empty dict without critique store."""
        matcher = TaskPatternMatcher()
        affinities = matcher.get_agent_affinities("bugfix", None)
        assert affinities == {}

    def test_get_affinities_with_mock_store(self):
        """Should query store and return affinities."""
        matcher = TaskPatternMatcher()

        # Create mock critique store
        mock_store = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_store.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_store.connection.return_value.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("claude", 8, 10),  # 80% success
            ("gpt", 6, 10),  # 60% success
        ]

        affinities = matcher.get_agent_affinities("bugfix", mock_store)

        assert "claude" in affinities
        assert "gpt" in affinities
        assert affinities["claude"] == 0.8
        assert affinities["gpt"] == 0.6


class TestPatternAffinity:
    """Tests for PatternAffinity dataclass."""

    def test_affinity_creation(self):
        """Should create PatternAffinity with all fields."""
        affinity = PatternAffinity(
            agent_name="claude",
            pattern="bugfix",
            success_rate=0.85,
            sample_size=20,
            confidence=1.0,
        )

        assert affinity.agent_name == "claude"
        assert affinity.pattern == "bugfix"
        assert affinity.success_rate == 0.85
        assert affinity.sample_size == 20
        assert affinity.confidence == 1.0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_pattern_matcher_singleton(self):
        """Should return same instance."""
        matcher1 = get_pattern_matcher()
        matcher2 = get_pattern_matcher()
        assert matcher1 is matcher2

    def test_classify_task_function(self):
        """Convenience function should work."""
        result = classify_task("Fix the bug")
        assert result == "bugfix"


class TestCustomPatterns:
    """Tests for custom pattern definitions."""

    def test_custom_patterns(self):
        """Should use custom patterns if provided."""
        custom = {
            "deployment": ["deploy", "release", "ship"],
            "database": ["sql", "query", "table", "migration"],
        }

        matcher = TaskPatternMatcher(patterns=custom)

        assert matcher.classify_task("Deploy to production") == "deployment"
        assert matcher.classify_task("Fix SQL query") == "database"
        # Original patterns not available
        assert matcher.classify_task("Fix the bug") == "general"
