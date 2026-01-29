"""
Tests for aragora.observability.n1_detector module.

Covers:
- N1QueryError exception
- QueryRecord dataclass
- N1Detection dataclass and is_likely_n1 property
- N1QueryDetector: record_query, normalize_query, analyze, get_violations
- Context manager behavior (__enter__/__exit__)
- Context variable tracking (get_current_detector, record_query global)
- detect_n1 decorator (sync and async)
- n1_detection_scope context manager
- Detection modes (off, warn, error)
"""

from __future__ import annotations

import asyncio

import pytest

from aragora.observability.n1_detector import (
    N1Detection,
    N1QueryDetector,
    N1QueryError,
    QueryRecord,
    detect_n1,
    get_current_detector,
    n1_detection_scope,
    record_query,
)


# =============================================================================
# TestN1QueryError
# =============================================================================


class TestN1QueryError:
    """Tests for N1QueryError exception."""

    def test_creation(self):
        """Should create with table, count, threshold."""
        err = N1QueryError("users", 10, 5)
        assert err.table == "users"
        assert err.count == 10
        assert err.threshold == 5

    def test_message(self):
        """Should have descriptive message."""
        err = N1QueryError("users", 10, 5)
        assert "users" in str(err)
        assert "10" in str(err)
        assert "5" in str(err)


# =============================================================================
# TestQueryRecord
# =============================================================================


class TestQueryRecord:
    """Tests for QueryRecord dataclass."""

    def test_creation(self):
        """Should create with required fields."""
        record = QueryRecord(table="users", query_pattern="SELECT ? FROM users", timestamp=1.0)
        assert record.table == "users"
        assert record.duration_ms == 0.0  # default


# =============================================================================
# TestN1Detection
# =============================================================================


class TestN1Detection:
    """Tests for N1Detection dataclass."""

    def test_is_likely_n1_true(self):
        """Should detect likely N+1 pattern."""
        detection = N1Detection(
            table="users", query_count=10, unique_patterns=1, total_duration_ms=50.0
        )
        assert detection.is_likely_n1 is True

    def test_is_likely_n1_false_many_patterns(self):
        """Should not flag as N+1 if many unique patterns."""
        detection = N1Detection(
            table="users", query_count=10, unique_patterns=8, total_duration_ms=50.0
        )
        assert detection.is_likely_n1 is False

    def test_is_likely_n1_false_single_query(self):
        """Should not flag single query."""
        detection = N1Detection(
            table="users", query_count=1, unique_patterns=1, total_duration_ms=5.0
        )
        assert detection.is_likely_n1 is False


# =============================================================================
# TestN1QueryDetector
# =============================================================================


class TestN1QueryDetector:
    """Tests for N1QueryDetector."""

    def test_default_init(self):
        """Should initialize with defaults."""
        detector = N1QueryDetector()
        assert detector.queries == []
        assert detector.name == "unnamed"

    def test_custom_init(self):
        """Should accept custom threshold and mode."""
        detector = N1QueryDetector(threshold=3, mode="warn", name="test")
        assert detector.threshold == 3
        assert detector.mode == "warn"
        assert detector.name == "test"

    def test_record_query(self):
        """Should record queries."""
        detector = N1QueryDetector(mode="warn")
        detector.record_query("users", "SELECT * FROM users WHERE id = 1")
        assert len(detector.queries) == 1
        assert detector.queries[0].table == "users"

    def test_record_query_off_mode(self):
        """Should not record in off mode."""
        detector = N1QueryDetector(mode="off")
        detector.record_query("users", "SELECT * FROM users WHERE id = 1")
        assert len(detector.queries) == 0

    def test_normalize_query_replaces_strings(self):
        """Should replace string literals."""
        detector = N1QueryDetector()
        result = detector._normalize_query("SELECT * FROM users WHERE name = 'John'")
        assert "'?'" in result
        assert "John" not in result

    def test_normalize_query_replaces_numbers(self):
        """Should replace numeric literals."""
        detector = N1QueryDetector()
        result = detector._normalize_query("SELECT * FROM users WHERE id = 42")
        assert "?" in result

    def test_normalize_query_replaces_uuids(self):
        """Should replace UUIDs."""
        detector = N1QueryDetector()
        result = detector._normalize_query(
            "SELECT * FROM users WHERE id = '550e8400-e29b-41d4-a716-446655440000'"
        )
        assert "550e8400" not in result

    def test_normalize_query_replaces_in_lists(self):
        """Should replace IN clause contents."""
        detector = N1QueryDetector()
        result = detector._normalize_query("SELECT * FROM users WHERE id IN (1, 2, 3)")
        assert "IN (?)" in result

    def test_analyze_empty(self):
        """Should return empty dict with no queries."""
        detector = N1QueryDetector()
        assert detector.analyze() == {}

    def test_analyze_groups_by_table(self):
        """Should group queries by table."""
        detector = N1QueryDetector(mode="warn")
        detector.record_query("users", "SELECT * FROM users WHERE id = 1")
        detector.record_query("users", "SELECT * FROM users WHERE id = 2")
        detector.record_query("posts", "SELECT * FROM posts WHERE user_id = 1")

        results = detector.analyze()
        assert "users" in results
        assert results["users"].query_count == 2
        assert "posts" in results
        assert results["posts"].query_count == 1

    def test_get_violations_empty(self):
        """Should return no violations for few queries."""
        detector = N1QueryDetector(threshold=5, mode="warn")
        detector.record_query("users", "SELECT * FROM users WHERE id = 1")
        assert detector.get_violations() == []

    def test_get_violations_threshold_exceeded(self):
        """Should return violations when threshold exceeded."""
        detector = N1QueryDetector(threshold=3, mode="warn")
        for i in range(5):
            detector.record_query("users", f"SELECT * FROM users WHERE id = {i}")

        violations = detector.get_violations()
        assert len(violations) == 1
        assert violations[0].table == "users"
        assert violations[0].query_count == 5


# =============================================================================
# TestContextManager
# =============================================================================


class TestContextManager:
    """Tests for detector as context manager."""

    def test_enter_sets_detector(self):
        """__enter__ should set context variable."""
        with N1QueryDetector(mode="warn") as detector:
            current = get_current_detector()
            assert current is detector

    def test_exit_resets_detector(self):
        """__exit__ should reset context variable."""
        with N1QueryDetector(mode="warn"):
            pass
        assert get_current_detector() is None

    def test_warn_mode_logs_violations(self):
        """Warn mode should log but not raise."""
        with N1QueryDetector(threshold=2, mode="warn") as detector:
            detector.record_query("users", "SELECT * FROM users WHERE id = 1")
            detector.record_query("users", "SELECT * FROM users WHERE id = 2")
            detector.record_query("users", "SELECT * FROM users WHERE id = 3")
        # Should not raise

    def test_error_mode_raises(self):
        """Error mode should raise N1QueryError."""
        with pytest.raises(N1QueryError) as exc_info:
            with N1QueryDetector(threshold=2, mode="error") as detector:
                detector.record_query("users", "SELECT * FROM users WHERE id = 1")
                detector.record_query("users", "SELECT * FROM users WHERE id = 2")

        assert exc_info.value.table == "users"

    def test_off_mode_no_action(self):
        """Off mode should not analyze."""
        with N1QueryDetector(threshold=1, mode="off") as detector:
            detector.record_query("users", "SELECT * FROM users WHERE id = 1")
            detector.record_query("users", "SELECT * FROM users WHERE id = 2")
        # Should not raise even though threshold exceeded


# =============================================================================
# TestGlobalRecordQuery
# =============================================================================


class TestGlobalRecordQuery:
    """Tests for global record_query function."""

    def test_records_when_detector_active(self):
        """Should record to active detector."""
        with N1QueryDetector(mode="warn") as detector:
            record_query("users", "SELECT * FROM users WHERE id = 1", 5.0)
        assert len(detector.queries) == 1

    def test_no_op_without_detector(self):
        """Should do nothing without active detector."""
        record_query("users", "SELECT * FROM users WHERE id = 1")
        # Should not raise


# =============================================================================
# TestDetectN1Decorator
# =============================================================================


class TestDetectN1Decorator:
    """Tests for detect_n1 decorator."""

    def test_sync_function(self):
        """Should wrap sync function."""

        @detect_n1(threshold=10, mode="warn")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Should wrap async function."""

        @detect_n1(threshold=10, mode="warn")
        async def my_func():
            return 42

        result = await my_func()
        assert result == 42

    def test_preserves_function_name(self):
        """Should preserve wrapped function name."""

        @detect_n1(threshold=10)
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"


# =============================================================================
# TestN1DetectionScope
# =============================================================================


class TestN1DetectionScope:
    """Tests for n1_detection_scope context manager."""

    def test_basic_scope(self):
        """Should work as context manager."""
        with n1_detection_scope("test", threshold=10) as detector:
            assert isinstance(detector, N1QueryDetector)

    def test_scope_name(self):
        """Should pass name to detector."""
        with n1_detection_scope("my_scope") as detector:
            assert detector.name == "my_scope"
