"""
Tests for database query profiling utilities.

Tests cover:
- QueryRecord and normalization
- QueryProfile statistics
- QueryProfiler context manager
- N+1 pattern detection
- SQLite connection instrumentation
- Recommended indexes
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.db.profiling import (
    N_PLUS_ONE_THRESHOLD,
    QueryProfile,
    QueryProfiler,
    QueryRecord,
    RECOMMENDED_INDEXES,
    SLOW_QUERY_THRESHOLD_MS,
    apply_recommended_indexes,
    get_index_recommendations,
    instrument_sqlite_connection,
    profile_function,
    profile_queries,
)


# ============================================================================
# QueryRecord Tests
# ============================================================================


class TestQueryRecord:
    """Tests for QueryRecord dataclass."""

    def test_create_query_record(self):
        """Test creating a query record."""
        record = QueryRecord(
            query="SELECT * FROM users WHERE id = ?",
            params=(1,),
            duration_ms=10.5,
            timestamp=time.time(),
            rows_affected=1,
        )

        assert record.query == "SELECT * FROM users WHERE id = ?"
        assert record.params == (1,)
        assert record.duration_ms == 10.5
        assert record.rows_affected == 1
        assert record.is_slow is False

    def test_query_record_slow_flag(self):
        """Test slow query flag based on threshold."""
        # Under threshold
        fast_record = QueryRecord(
            query="SELECT 1",
            params=(),
            duration_ms=50.0,
            timestamp=time.time(),
            is_slow=False,
        )
        assert fast_record.is_slow is False

        # Over threshold
        slow_record = QueryRecord(
            query="SELECT 1",
            params=(),
            duration_ms=150.0,
            timestamp=time.time(),
            is_slow=True,
        )
        assert slow_record.is_slow is True

    def test_normalized_query_removes_strings(self):
        """Test query normalization removes string literals."""
        record = QueryRecord(
            query="SELECT * FROM users WHERE name = 'John Doe'",
            params=(),
            duration_ms=1.0,
            timestamp=time.time(),
        )

        normalized = record.normalized_query()

        assert "'John Doe'" not in normalized
        assert "'?'" in normalized

    def test_normalized_query_removes_numbers(self):
        """Test query normalization removes numeric literals."""
        record = QueryRecord(
            query="SELECT * FROM items WHERE price > 100 AND quantity < 50",
            params=(),
            duration_ms=1.0,
            timestamp=time.time(),
        )

        normalized = record.normalized_query()

        assert "100" not in normalized
        assert "50" not in normalized
        assert "?" in normalized

    def test_normalized_query_preserves_structure(self):
        """Test that normalization preserves query structure."""
        record = QueryRecord(
            query="SELECT id, name FROM users WHERE status = 'active'",
            params=(),
            duration_ms=1.0,
            timestamp=time.time(),
        )

        normalized = record.normalized_query()

        assert "SELECT" in normalized
        assert "FROM users" in normalized
        assert "WHERE" in normalized

    def test_normalized_query_normalizes_whitespace(self):
        """Test that normalization collapses whitespace."""
        record = QueryRecord(
            query="SELECT   *   FROM   users   WHERE   id = 1",
            params=(),
            duration_ms=1.0,
            timestamp=time.time(),
        )

        normalized = record.normalized_query()

        # No double spaces
        assert "  " not in normalized


# ============================================================================
# QueryProfile Tests
# ============================================================================


class TestQueryProfile:
    """Tests for QueryProfile class."""

    def test_empty_profile(self):
        """Test empty profile statistics."""
        profile = QueryProfile()

        assert profile.total_queries == 0
        assert profile.total_duration_ms == 0.0
        assert profile.slow_queries == []
        assert profile.potential_n_plus_one == []

    def test_add_query(self):
        """Test adding queries to profile."""
        profile = QueryProfile()

        profile.add_query("SELECT 1", (), 10.0, 1)
        profile.add_query("SELECT 2", (), 20.0, 1)

        assert profile.total_queries == 2
        assert profile.total_duration_ms == 30.0

    def test_slow_query_detection(self):
        """Test slow query detection."""
        profile = QueryProfile()

        profile.add_query("SELECT fast", (), 50.0)
        profile.add_query("SELECT slow", (), 150.0)  # Over threshold

        assert len(profile.slow_queries) == 1
        assert profile.slow_queries[0].query == "SELECT slow"

    def test_query_patterns_grouping(self):
        """Test grouping queries by normalized pattern."""
        profile = QueryProfile()

        # Same pattern with different values
        profile.add_query("SELECT * FROM users WHERE id = 1", (), 10.0)
        profile.add_query("SELECT * FROM users WHERE id = 2", (), 10.0)
        profile.add_query("SELECT * FROM users WHERE id = 3", (), 10.0)

        patterns = profile.query_patterns

        # Should be grouped into single pattern
        assert len(patterns) == 1
        pattern_key = list(patterns.keys())[0]
        assert len(patterns[pattern_key]) == 3

    def test_n_plus_one_detection(self):
        """Test N+1 pattern detection."""
        profile = QueryProfile()

        # Create N+1 pattern (same query > threshold times)
        for i in range(10):  # Over N_PLUS_ONE_THRESHOLD
            profile.add_query(f"SELECT * FROM items WHERE user_id = {i}", (), 5.0)

        n_plus_one = profile.potential_n_plus_one

        assert len(n_plus_one) > 0
        pattern, count = n_plus_one[0]
        assert count == 10

    def test_no_n_plus_one_under_threshold(self):
        """Test no N+1 detected under threshold."""
        profile = QueryProfile()

        # Add queries under threshold
        for i in range(3):  # Under N_PLUS_ONE_THRESHOLD
            profile.add_query(f"SELECT * FROM items WHERE user_id = {i}", (), 5.0)

        assert profile.potential_n_plus_one == []

    def test_profile_finish(self):
        """Test finishing a profile sets end time."""
        profile = QueryProfile()
        assert profile.end_time is None

        profile.finish()

        assert profile.end_time is not None

    def test_report_generation(self):
        """Test report generation."""
        profile = QueryProfile()
        profile.add_query("SELECT 1", (), 10.0)
        profile.add_query("SELECT 2", (), 200.0)  # Slow
        profile.finish()

        report = profile.report()

        assert "DATABASE QUERY PROFILE" in report
        assert "Total queries: 2" in report
        assert "Slow queries: 1" in report
        assert "QUERY TYPES:" in report
        assert "SELECT: 2" in report

    def test_report_verbose(self):
        """Test verbose report includes all queries."""
        profile = QueryProfile()
        profile.add_query("SELECT 1", (), 10.0)
        profile.finish()

        report = profile.report(verbose=True)

        assert "ALL QUERIES:" in report

    def test_to_dict(self):
        """Test converting profile to dictionary."""
        profile = QueryProfile()
        profile.add_query("SELECT 1", (), 10.0)
        profile.finish()

        result = profile.to_dict()

        assert result["total_queries"] == 1
        assert "total_duration_ms" in result
        assert "slow_queries" in result
        assert "potential_n_plus_one" in result


# ============================================================================
# QueryProfiler Tests
# ============================================================================


class TestQueryProfiler:
    """Tests for QueryProfiler context manager."""

    def test_profiler_context_manager(self):
        """Test profiler as context manager."""
        with QueryProfiler() as profiler:
            profiler.record("SELECT 1", (), 10.0)

        assert profiler.profile.total_queries == 1
        assert profiler.profile.end_time is not None

    def test_profiler_nesting(self):
        """Test nested profilers."""
        with QueryProfiler() as outer:
            outer.record("OUTER", (), 10.0)

            with QueryProfiler() as inner:
                inner.record("INNER", (), 10.0)

            outer.record("OUTER 2", (), 10.0)

        assert outer.profile.total_queries == 2
        assert inner.profile.total_queries == 1

    def test_current_profiler(self):
        """Test QueryProfiler.current() returns active profiler."""
        assert QueryProfiler.current() is None

        with QueryProfiler() as profiler:
            assert QueryProfiler.current() is profiler

        assert QueryProfiler.current() is None

    def test_profiler_report_method(self):
        """Test profiler report method."""
        profiler = QueryProfiler()
        profiler.record("SELECT 1", (), 10.0)

        report = profiler.report()

        assert "Total queries: 1" in report


# ============================================================================
# profile_queries Context Manager Tests
# ============================================================================


class TestProfileQueries:
    """Tests for profile_queries context manager."""

    def test_profile_queries_basic(self):
        """Test basic profile_queries usage."""
        with profile_queries() as profiler:
            profiler.record("SELECT 1", (), 10.0)

        assert profiler.profile.total_queries == 1


# ============================================================================
# profile_function Decorator Tests
# ============================================================================


class TestProfileFunction:
    """Tests for profile_function decorator."""

    def test_profile_function_sync(self):
        """Test profiling sync function."""
        @profile_function
        def test_func():
            profiler = QueryProfiler.current()
            if profiler:
                profiler.record("SELECT 1", (), 10.0)
            return "result"

        result = test_func()

        assert result == "result"

    @pytest.mark.asyncio
    async def test_profile_function_async(self):
        """Test profiling async function."""
        @profile_function
        async def async_func():
            profiler = QueryProfiler.current()
            if profiler:
                profiler.record("SELECT 1", (), 10.0)
            return "async_result"

        result = await async_func()

        assert result == "async_result"


# ============================================================================
# SQLite Instrumentation Tests
# ============================================================================


class TestSQLiteInstrumentation:
    """Tests for SQLite connection instrumentation.

    Note: Native sqlite3.Connection.execute is read-only in Python.
    The instrumentation function works with connections that allow attribute
    assignment (e.g., custom connection wrappers used in the codebase).
    These tests use MagicMock to verify the instrumentation logic.
    """

    def test_instrument_connection_calls_are_wrapped(self):
        """Test that instrumentation wraps execute/executemany methods."""
        # Create a mock connection that allows attribute assignment
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=MagicMock(rowcount=1))
        mock_conn.executemany = MagicMock(return_value=MagicMock(rowcount=3))

        result = instrument_sqlite_connection(mock_conn)

        assert result is mock_conn
        # The execute/executemany should now be wrapped functions
        assert callable(mock_conn.execute)
        assert callable(mock_conn.executemany)

    def test_instrumented_execute_records_to_profiler(self):
        """Test that wrapped execute records queries to active profiler."""
        # Setup mock connection
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        original_execute = MagicMock(return_value=mock_cursor)
        mock_conn.execute = original_execute

        # Instrument it
        instrument_sqlite_connection(mock_conn)

        # Get the new wrapped execute
        wrapped_execute = mock_conn.execute

        # Execute with profiler active
        with QueryProfiler() as profiler:
            wrapped_execute("SELECT * FROM test WHERE id = ?", (1,))

        # Verify query was recorded
        assert profiler.profile.total_queries == 1
        assert "SELECT * FROM test" in profiler.profile.queries[0].query

    def test_instrumented_executemany_records_batch_size(self):
        """Test that wrapped executemany records batch size."""
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_conn = MagicMock()
        mock_conn.executemany = MagicMock(return_value=mock_cursor)

        instrument_sqlite_connection(mock_conn)
        wrapped_executemany = mock_conn.executemany

        with QueryProfiler() as profiler:
            wrapped_executemany("INSERT INTO test VALUES (?)", [(1,), (2,), (3,)])

        assert profiler.profile.total_queries == 1
        assert "x3" in profiler.profile.queries[0].query

    def test_instrumented_works_without_active_profiler(self):
        """Test that wrapped methods work without active profiler."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        instrument_sqlite_connection(mock_conn)

        # Should not raise when no profiler is active
        result = mock_conn.execute("SELECT 1")
        assert result is mock_cursor

    def test_instrumented_preserves_exceptions(self):
        """Test that exceptions are propagated from wrapped methods."""
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(side_effect=sqlite3.OperationalError("Test error"))

        instrument_sqlite_connection(mock_conn)

        with pytest.raises(sqlite3.OperationalError):
            mock_conn.execute("SELECT 1")


# ============================================================================
# Index Recommendation Tests
# ============================================================================


class TestIndexRecommendations:
    """Tests for index recommendation functions."""

    def test_get_index_recommendations_returns_sql(self):
        """Test that recommendations return SQL statements."""
        sql = get_index_recommendations()

        assert "CREATE INDEX" in sql
        assert "IF NOT EXISTS" in sql

    def test_recommended_indexes_constant(self):
        """Test RECOMMENDED_INDEXES constant."""
        assert "idx_debates_created_at" in RECOMMENDED_INDEXES
        assert "idx_messages_debate_id" in RECOMMENDED_INDEXES
        assert "idx_votes_debate_id" in RECOMMENDED_INDEXES

    def test_apply_recommended_indexes(self):
        """Test applying recommended indexes to database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))

            # Create required tables
            conn.execute("CREATE TABLE debates (id TEXT, created_at TEXT, status TEXT, task_hash TEXT)")
            conn.execute("CREATE TABLE messages (id TEXT, debate_id TEXT, agent_id TEXT, round_num INT)")
            conn.execute("CREATE TABLE votes (id TEXT, debate_id TEXT, agent_id TEXT)")
            conn.commit()

            # Apply indexes
            apply_recommended_indexes(conn)

            # Verify indexes exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            index_names = [row[0] for row in cursor.fetchall()]

            assert "idx_debates_created_at" in index_names
            assert "idx_messages_debate_id" in index_names

            conn.close()

    def test_apply_indexes_handles_missing_tables(self):
        """Test that applying indexes handles missing tables gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "empty.db"
            conn = sqlite3.connect(str(db_path))

            # Should not raise even with missing tables
            apply_recommended_indexes(conn)

            conn.close()


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for profiling constants."""

    def test_slow_query_threshold(self):
        """Test slow query threshold is reasonable."""
        assert SLOW_QUERY_THRESHOLD_MS > 0
        assert SLOW_QUERY_THRESHOLD_MS <= 1000  # Should be <= 1 second

    def test_n_plus_one_threshold(self):
        """Test N+1 threshold is reasonable."""
        assert N_PLUS_ONE_THRESHOLD > 0
        assert N_PLUS_ONE_THRESHOLD <= 20


# ============================================================================
# Integration Tests
# ============================================================================


class TestProfilingIntegration:
    """Integration tests for profiling system."""

    def test_full_profiling_workflow(self):
        """Test complete profiling workflow with manual recording."""
        # Use manual recording since native sqlite3.Connection can't be instrumented
        with profile_queries() as profiler:
            # Simulate typical operations
            profiler.record("INSERT INTO users VALUES (1, 'Alice')", (), 5.0, 1)
            profiler.record("INSERT INTO users VALUES (2, 'Bob')", (), 5.0, 1)
            profiler.record("SELECT * FROM users", (), 10.0, 2)

        # Verify profiling results
        assert profiler.profile.total_queries == 3
        report = profiler.report()
        assert "INSERT: 2" in report
        assert "SELECT: 1" in report

    def test_n_plus_one_detection_simulated(self):
        """Test N+1 detection with simulated queries."""
        with profile_queries() as profiler:
            # Simulate N+1 pattern
            for user_id in range(10):
                profiler.record(
                    f"SELECT * FROM items WHERE user_id = {user_id}",
                    (),
                    5.0,
                    1,
                )

        n_plus_one = profiler.profile.potential_n_plus_one

        assert len(n_plus_one) > 0
        pattern, count = n_plus_one[0]
        assert count == 10

    def test_profiling_with_mocked_connection(self):
        """Test profiling with a mock connection that allows instrumentation."""
        # Create mock
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        # Instrument
        instrument_sqlite_connection(mock_conn)

        # Profile operations
        with profile_queries() as profiler:
            mock_conn.execute("SELECT 1")
            mock_conn.execute("SELECT 2")

        assert profiler.profile.total_queries == 2
