"""
Tests for aragora.observability.query_analyzer module.

Covers:
- QueryPlanIssue dataclass and serialization
- SlowQuery dataclass and serialization
- QueryPlanAnalyzer pattern detection (SELECT *, LIKE, missing WHERE, ORDER BY, IN)
- Table extraction from queries
- Slow query tracking
- Global analyzer and convenience functions
- Query analysis stats
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.observability.query_analyzer import (
    QueryPlanAnalyzer,
    QueryPlanIssue,
    SlowQuery,
    clear_slow_queries,
    get_analyzer,
    get_query_analysis_stats,
    get_slow_queries,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear slow query state between tests."""
    clear_slow_queries()
    yield
    clear_slow_queries()


# =============================================================================
# TestQueryPlanIssue
# =============================================================================


class TestQueryPlanIssue:
    """Tests for QueryPlanIssue dataclass."""

    def test_creation(self):
        """Should create with required fields."""
        issue = QueryPlanIssue(
            query="SELECT * FROM users",
            duration_ms=150.0,
            issue="select_star_large_table",
            suggestion="Avoid SELECT *",
        )
        assert issue.query == "SELECT * FROM users"
        assert issue.duration_ms == 150.0
        assert issue.table is None

    def test_to_dict(self):
        """Should convert to dictionary."""
        issue = QueryPlanIssue(
            query="SELECT * FROM users",
            duration_ms=150.5,
            issue="slow",
            suggestion="Add index",
            table="users",
        )
        d = issue.to_dict()
        assert d["query"] == "SELECT * FROM users"
        assert d["duration_ms"] == 150.5
        assert d["issue"] == "slow"
        assert d["suggestion"] == "Add index"
        assert d["table"] == "users"
        assert "timestamp" in d

    def test_to_dict_truncates_long_query(self):
        """Should truncate queries longer than 200 chars."""
        long_query = "SELECT " + "x" * 250
        issue = QueryPlanIssue(query=long_query, duration_ms=100, issue="slow", suggestion="Fix")
        d = issue.to_dict()
        assert len(d["query"]) <= 204  # 200 + "..."
        assert d["query"].endswith("...")


# =============================================================================
# TestSlowQuery
# =============================================================================


class TestSlowQuery:
    """Tests for SlowQuery dataclass."""

    def test_creation(self):
        """Should create with required fields."""
        sq = SlowQuery(query="SELECT 1", duration_ms=500.0)
        assert sq.query == "SELECT 1"
        assert sq.table is None
        assert sq.context is None

    def test_to_dict(self):
        """Should serialize to dictionary."""
        sq = SlowQuery(query="SELECT 1", duration_ms=200.0, table="users", context="handler")
        d = sq.to_dict()
        assert d["duration_ms"] == 200.0
        assert d["table"] == "users"
        assert d["context"] == "handler"


# =============================================================================
# TestQueryPlanAnalyzer
# =============================================================================


class TestQueryPlanAnalyzer:
    """Tests for QueryPlanAnalyzer."""

    def test_init_defaults(self):
        """Should initialize with default values."""
        analyzer = QueryPlanAnalyzer()
        assert analyzer.mode == "off"  # default from env

    def test_init_custom_threshold(self):
        """Should accept custom threshold."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50.0)
        assert analyzer.slow_threshold_ms == 50.0

    @pytest.mark.asyncio
    async def test_analyze_off_mode(self):
        """Should return None when mode is off."""
        analyzer = QueryPlanAnalyzer(mode="off")
        result = await analyzer.analyze("SELECT * FROM users", 200.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_fast_query(self):
        """Should return None for fast queries."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=100, mode="warn")
        result = await analyzer.analyze("SELECT id FROM users WHERE id = 1", 10.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_detect_select_star_large_table(self):
        """Should detect SELECT * on known large tables."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        result = await analyzer.analyze(
            "SELECT * FROM knowledge_nodes WHERE id = 1",
            150.0,
        )
        assert result is not None
        assert result.issue == "select_star_large_table"
        assert result.table == "knowledge_nodes"

    @pytest.mark.asyncio
    async def test_detect_leading_wildcard_like(self):
        """Should detect LIKE with leading wildcard."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        result = await analyzer.analyze(
            "SELECT id FROM users WHERE name LIKE '%john%'",
            150.0,
        )
        assert result is not None
        assert result.issue == "leading_wildcard_like"

    @pytest.mark.asyncio
    async def test_detect_missing_where_clause(self):
        """Should detect missing WHERE on large tables."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        result = await analyzer.analyze(
            "SELECT id FROM debate_messages",
            150.0,
        )
        assert result is not None
        assert result.issue == "missing_where_clause"

    @pytest.mark.asyncio
    async def test_detect_slow_order_by(self):
        """Should detect slow ORDER BY on large tables."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        result = await analyzer.analyze(
            "SELECT id FROM knowledge_nodes ORDER BY created_at",
            250.0,  # > 2x threshold
        )
        assert result is not None
        assert result.issue == "slow_order_by"

    @pytest.mark.asyncio
    async def test_detect_large_in_clause(self):
        """Should detect IN clause with >100 values."""
        values = ", ".join(str(i) for i in range(150))
        query = f"SELECT id FROM users WHERE id IN ({values})"
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        result = await analyzer.analyze(query, 200.0)
        assert result is not None
        assert result.issue == "large_in_clause"


# =============================================================================
# TestTableExtraction
# =============================================================================


class TestTableExtraction:
    """Tests for table name extraction."""

    def test_extract_from_select(self):
        """Should extract table from SELECT ... FROM."""
        analyzer = QueryPlanAnalyzer()
        table = analyzer._extract_table("SELECT id FROM users WHERE id = 1")
        assert table == "users"

    def test_extract_from_insert(self):
        """Should extract table from INSERT INTO."""
        analyzer = QueryPlanAnalyzer()
        table = analyzer._extract_table("INSERT INTO events (name) VALUES ('test')")
        assert table == "events"

    def test_extract_from_update(self):
        """Should extract table from UPDATE."""
        analyzer = QueryPlanAnalyzer()
        table = analyzer._extract_table("UPDATE users SET name = 'test' WHERE id = 1")
        assert table == "users"

    def test_no_table_found(self):
        """Should return None for queries without recognized table pattern."""
        analyzer = QueryPlanAnalyzer()
        table = analyzer._extract_table("TRUNCATE foo")
        assert table is None


# =============================================================================
# TestSlowQueryTracking
# =============================================================================


class TestSlowQueryTracking:
    """Tests for slow query global tracking."""

    def test_get_slow_queries_empty(self):
        """Should return empty list when no slow queries."""
        result = get_slow_queries()
        assert result == []

    @pytest.mark.asyncio
    async def test_slow_query_recorded(self):
        """Slow queries should be recorded in global store."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        await analyzer.analyze("SELECT id FROM users WHERE id = 1", 200.0)

        queries = get_slow_queries()
        assert len(queries) >= 1
        assert queries[0]["duration_ms"] == 200.0

    def test_get_slow_queries_with_threshold(self):
        """Should filter by threshold_ms."""
        from aragora.observability.query_analyzer import _slow_queries, SlowQuery

        _slow_queries.append(SlowQuery(query="q1", duration_ms=100))
        _slow_queries.append(SlowQuery(query="q2", duration_ms=500))

        result = get_slow_queries(threshold_ms=200)
        assert len(result) == 1
        assert result[0]["duration_ms"] == 500.0

    def test_get_slow_queries_sorted(self):
        """Should return sorted by duration descending."""
        from aragora.observability.query_analyzer import _slow_queries, SlowQuery

        _slow_queries.append(SlowQuery(query="fast", duration_ms=100))
        _slow_queries.append(SlowQuery(query="slow", duration_ms=500))
        _slow_queries.append(SlowQuery(query="medium", duration_ms=250))

        result = get_slow_queries()
        durations = [q["duration_ms"] for q in result]
        assert durations == sorted(durations, reverse=True)

    def test_clear_slow_queries(self):
        """clear_slow_queries should empty the store."""
        from aragora.observability.query_analyzer import _slow_queries, SlowQuery

        _slow_queries.append(SlowQuery(query="q1", duration_ms=100))
        clear_slow_queries()
        assert len(get_slow_queries()) == 0


# =============================================================================
# TestGlobalAnalyzer
# =============================================================================


class TestGlobalAnalyzer:
    """Tests for global analyzer instance."""

    def test_get_analyzer_returns_instance(self):
        """get_analyzer should return a QueryPlanAnalyzer."""
        analyzer = get_analyzer()
        assert isinstance(analyzer, QueryPlanAnalyzer)

    def test_get_analyzer_singleton(self):
        """get_analyzer should return same instance."""
        a1 = get_analyzer()
        a2 = get_analyzer()
        assert a1 is a2

    def test_get_query_analysis_stats(self):
        """Should return stats dict."""
        stats = get_query_analysis_stats()
        assert "mode" in stats
        assert "slow_threshold_ms" in stats
        assert "slow_query_count" in stats
        assert "issue_count" in stats
        assert "issues_by_type" in stats


# =============================================================================
# TestAnalyzerIssueRecording
# =============================================================================


class TestAnalyzerIssueRecording:
    """Tests for issue recording and retrieval."""

    def test_get_issues_empty(self):
        """Should return empty list initially."""
        analyzer = QueryPlanAnalyzer()
        assert analyzer.get_issues() == []

    @pytest.mark.asyncio
    async def test_issues_recorded(self):
        """Detected issues should be recorded."""
        analyzer = QueryPlanAnalyzer(slow_threshold_ms=50, mode="warn")
        await analyzer.analyze("SELECT * FROM knowledge_nodes", 200.0)
        issues = analyzer.get_issues()
        assert len(issues) >= 1

    def test_clear_issues(self):
        """clear_issues should empty the list."""
        analyzer = QueryPlanAnalyzer()
        analyzer._issues.append(
            QueryPlanIssue(query="q", duration_ms=1, issue="test", suggestion="fix")
        )
        analyzer.clear_issues()
        assert analyzer.get_issues() == []
