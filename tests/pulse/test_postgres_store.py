"""Comprehensive tests for the Pulse PostgreSQL scheduled debate store.

Tests cover:
- ScheduledDebateRecord dataclass
- PostgresScheduledDebateStore async methods
- Sync wrapper methods
- Query filtering, pagination, and SQL parameterization
- Connection management
- Error handling and edge cases
- Analytics aggregation
- Cleanup operations
- Topic hashing and deduplication
"""

from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pulse.postgres_store import (
    PostgresScheduledDebateStore,
    ScheduledDebateRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides: Any) -> ScheduledDebateRecord:
    """Create a ScheduledDebateRecord with sensible defaults."""
    defaults = dict(
        id="rec-001",
        topic_hash="abc123",
        topic_text="Test topic about AI",
        platform="hackernews",
        category="tech",
        volume=150,
        debate_id="debate-001",
        created_at=time.time(),
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        scheduler_run_id="run-001",
    )
    defaults.update(overrides)
    return ScheduledDebateRecord(**defaults)


def _make_row(**overrides: Any) -> dict[str, Any]:
    """Create a dict mimicking an asyncpg.Record for _row_to_record."""
    defaults = dict(
        id="rec-001",
        topic_hash="abc123",
        topic_text="Test topic about AI",
        platform="hackernews",
        category="tech",
        volume=150,
        debate_id="debate-001",
        created_at=time.time(),
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        scheduler_run_id="run-001",
    )
    defaults.update(overrides)
    return defaults


def _mock_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock()
    return pool


def _mock_conn() -> AsyncMock:
    """Create a mock asyncpg connection with all needed async methods."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@asynccontextmanager
async def _fake_connection(conn: AsyncMock):
    """Async context manager that yields a mock connection."""
    yield conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    """Provide a fresh mock connection."""
    return _mock_conn()


@pytest.fixture
def mock_pool():
    """Provide a fresh mock pool."""
    return _mock_pool()


@pytest.fixture
def store(mock_pool, mock_conn):
    """Create a PostgresScheduledDebateStore with mocked pool and connection.

    Patches:
    - The parent __init__ to avoid asyncpg availability checks
    - The connection() context manager to yield our mock connection
    """
    with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
        s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)
        # Set attributes that __init__ would normally set
        s._pool = mock_pool
        s._initialized = False
        s._use_resilient = False
        s._acquire_timeout = 10.0
        s._acquire_retries = 3

    # Patch connection() to yield our mock connection
    with patch.object(
        PostgresScheduledDebateStore,
        "connection",
        return_value=_fake_connection(mock_conn),
    ):
        yield s


# ===========================================================================
# ScheduledDebateRecord dataclass tests
# ===========================================================================


class TestScheduledDebateRecord:
    """Tests for the ScheduledDebateRecord dataclass."""

    def test_basic_creation(self):
        """A record stores all fields correctly."""
        now = time.time()
        rec = ScheduledDebateRecord(
            id="r1",
            topic_hash="h1",
            topic_text="Test topic",
            platform="reddit",
            category="science",
            volume=500,
            debate_id="d1",
            created_at=now,
            consensus_reached=True,
            confidence=0.9,
            rounds_used=4,
            scheduler_run_id="s1",
        )
        assert rec.id == "r1"
        assert rec.topic_hash == "h1"
        assert rec.topic_text == "Test topic"
        assert rec.platform == "reddit"
        assert rec.category == "science"
        assert rec.volume == 500
        assert rec.debate_id == "d1"
        assert rec.created_at == now
        assert rec.consensus_reached is True
        assert rec.confidence == 0.9
        assert rec.rounds_used == 4
        assert rec.scheduler_run_id == "s1"

    def test_optional_fields_none(self):
        """Fields that may be None are handled correctly."""
        rec = _make_record(debate_id=None, consensus_reached=None, confidence=None)
        assert rec.debate_id is None
        assert rec.consensus_reached is None
        assert rec.confidence is None

    def test_hours_ago_recent(self):
        """hours_ago returns a small value for recently created records."""
        rec = _make_record(created_at=time.time() - 60)
        assert rec.hours_ago < 0.1  # about 0.0167 hours

    def test_hours_ago_old(self):
        """hours_ago returns correct value for an older record."""
        rec = _make_record(created_at=time.time() - 3600 * 12)
        assert abs(rec.hours_ago - 12.0) < 0.1

    def test_hours_ago_exactly_one_hour(self):
        """hours_ago returns ~1.0 for a record created one hour ago."""
        rec = _make_record(created_at=time.time() - 3600)
        assert abs(rec.hours_ago - 1.0) < 0.01

    def test_hours_ago_future_record(self):
        """hours_ago returns a negative value for a future timestamp."""
        rec = _make_record(created_at=time.time() + 7200)
        assert rec.hours_ago < 0

    def test_zero_volume(self):
        """Volume can be zero."""
        rec = _make_record(volume=0)
        assert rec.volume == 0

    def test_zero_rounds(self):
        """Rounds used can be zero."""
        rec = _make_record(rounds_used=0)
        assert rec.rounds_used == 0


# ===========================================================================
# hash_topic static method tests
# ===========================================================================


class TestHashTopic:
    """Tests for the static hash_topic method."""

    def test_basic_hash(self):
        """hash_topic returns a 16-char hex string."""
        h = PostgresScheduledDebateStore.hash_topic("Hello World")
        assert isinstance(h, str)
        assert len(h) == 16
        # Verify it's hex
        int(h, 16)

    def test_deterministic(self):
        """Same input always produces the same hash."""
        h1 = PostgresScheduledDebateStore.hash_topic("AI Regulation")
        h2 = PostgresScheduledDebateStore.hash_topic("AI Regulation")
        assert h1 == h2

    def test_case_insensitive(self):
        """Hashing is case-insensitive (normalizes to lower)."""
        h1 = PostgresScheduledDebateStore.hash_topic("AI Regulation")
        h2 = PostgresScheduledDebateStore.hash_topic("ai regulation")
        assert h1 == h2

    def test_whitespace_normalized(self):
        """Extra whitespace is collapsed to single spaces."""
        h1 = PostgresScheduledDebateStore.hash_topic("AI  Regulation")
        h2 = PostgresScheduledDebateStore.hash_topic("AI Regulation")
        assert h1 == h2

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        h1 = PostgresScheduledDebateStore.hash_topic("  AI Regulation  ")
        h2 = PostgresScheduledDebateStore.hash_topic("AI Regulation")
        assert h1 == h2

    def test_different_topics_different_hashes(self):
        """Different topics produce different hashes."""
        h1 = PostgresScheduledDebateStore.hash_topic("Topic Alpha")
        h2 = PostgresScheduledDebateStore.hash_topic("Topic Beta")
        assert h1 != h2

    def test_matches_manual_sha256(self):
        """Hash matches manual sha256 computation on normalized text."""
        topic = "AI in Healthcare"
        normalized = "ai in healthcare"
        expected = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        assert PostgresScheduledDebateStore.hash_topic(topic) == expected

    def test_empty_string(self):
        """Empty string produces a valid hash."""
        h = PostgresScheduledDebateStore.hash_topic("")
        assert len(h) == 16

    def test_unicode_topic(self):
        """Unicode characters in topics are handled correctly."""
        h = PostgresScheduledDebateStore.hash_topic("Tema sobre IA y regulacion")
        assert len(h) == 16

    def test_tab_and_newline_normalized(self):
        """Tabs and newlines are treated as whitespace and collapsed."""
        h1 = PostgresScheduledDebateStore.hash_topic("AI\tRegulation\nToday")
        h2 = PostgresScheduledDebateStore.hash_topic("AI Regulation Today")
        assert h1 == h2


# ===========================================================================
# _row_to_record tests
# ===========================================================================


class TestRowToRecord:
    """Tests for the _row_to_record helper method."""

    def test_all_fields_present(self, store):
        """All fields map correctly from row to record."""
        now = time.time()
        row = _make_row(
            id="r99",
            topic_hash="xyz",
            topic_text="Test",
            platform="twitter",
            category="politics",
            volume=1000,
            debate_id="d99",
            created_at=now,
            consensus_reached=False,
            confidence=0.5,
            rounds_used=2,
            scheduler_run_id="s99",
        )
        rec = store._row_to_record(row)
        assert rec.id == "r99"
        assert rec.topic_hash == "xyz"
        assert rec.topic_text == "Test"
        assert rec.platform == "twitter"
        assert rec.category == "politics"
        assert rec.volume == 1000
        assert rec.debate_id == "d99"
        assert rec.created_at == now
        assert rec.consensus_reached is False
        assert rec.confidence == 0.5
        assert rec.rounds_used == 2
        assert rec.scheduler_run_id == "s99"

    def test_none_category_becomes_empty_string(self, store):
        """None category becomes empty string."""
        row = _make_row(category=None)
        rec = store._row_to_record(row)
        assert rec.category == ""

    def test_none_volume_becomes_zero(self, store):
        """None volume becomes 0."""
        row = _make_row(volume=None)
        rec = store._row_to_record(row)
        assert rec.volume == 0

    def test_none_rounds_used_becomes_zero(self, store):
        """None rounds_used becomes 0."""
        row = _make_row(rounds_used=None)
        rec = store._row_to_record(row)
        assert rec.rounds_used == 0

    def test_none_scheduler_run_id_becomes_empty(self, store):
        """None scheduler_run_id becomes empty string."""
        row = _make_row(scheduler_run_id=None)
        rec = store._row_to_record(row)
        assert rec.scheduler_run_id == ""

    def test_none_debate_id_stays_none(self, store):
        """None debate_id stays None."""
        row = _make_row(debate_id=None)
        rec = store._row_to_record(row)
        assert rec.debate_id is None

    def test_none_consensus_stays_none(self, store):
        """None consensus_reached stays None."""
        row = _make_row(consensus_reached=None)
        rec = store._row_to_record(row)
        assert rec.consensus_reached is None

    def test_none_confidence_stays_none(self, store):
        """None confidence stays None."""
        row = _make_row(confidence=None)
        rec = store._row_to_record(row)
        assert rec.confidence is None


# ===========================================================================
# Async method tests
# ===========================================================================


class TestRecordScheduledDebateAsync:
    """Tests for record_scheduled_debate_async."""

    @pytest.mark.asyncio
    async def test_inserts_record(self, store, mock_conn):
        """record_scheduled_debate_async executes INSERT with correct params."""
        rec = _make_record()
        await store.record_scheduled_debate_async(rec)

        mock_conn.execute.assert_awaited_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO scheduled_debates" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql
        # All 12 positional args should be passed
        assert call_args[0][1] == rec.id
        assert call_args[0][2] == rec.topic_hash
        assert call_args[0][3] == rec.topic_text
        assert call_args[0][4] == rec.platform
        assert call_args[0][5] == rec.category
        assert call_args[0][6] == rec.volume
        assert call_args[0][7] == rec.debate_id
        assert call_args[0][8] == rec.created_at
        assert call_args[0][9] == rec.consensus_reached
        assert call_args[0][10] == rec.confidence
        assert call_args[0][11] == rec.rounds_used
        assert call_args[0][12] == rec.scheduler_run_id

    @pytest.mark.asyncio
    async def test_upsert_on_conflict(self, store, mock_conn):
        """SQL includes ON CONFLICT clause for upsert behavior."""
        rec = _make_record()
        await store.record_scheduled_debate_async(rec)
        sql = mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT (id) DO UPDATE SET" in sql

    @pytest.mark.asyncio
    async def test_record_with_none_fields(self, store, mock_conn):
        """Records with None optional fields are inserted correctly."""
        rec = _make_record(debate_id=None, consensus_reached=None, confidence=None)
        await store.record_scheduled_debate_async(rec)
        call_args = mock_conn.execute.call_args
        assert call_args[0][7] is None  # debate_id
        assert call_args[0][9] is None  # consensus_reached
        assert call_args[0][10] is None  # confidence


class TestGetRecentTopicsAsync:
    """Tests for get_recent_topics_async."""

    @pytest.mark.asyncio
    async def test_returns_records(self, store, mock_conn):
        """get_recent_topics_async returns a list of ScheduledDebateRecord."""
        now = time.time()
        mock_conn.fetch.return_value = [
            _make_row(id="r1", created_at=now),
            _make_row(id="r2", created_at=now - 100),
        ]
        results = await store.get_recent_topics_async(hours=24)
        assert len(results) == 2
        assert results[0].id == "r1"
        assert results[1].id == "r2"

    @pytest.mark.asyncio
    async def test_empty_result(self, store, mock_conn):
        """Returns empty list when no recent topics."""
        mock_conn.fetch.return_value = []
        results = await store.get_recent_topics_async(hours=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_cutoff_calculation(self, store, mock_conn):
        """Cutoff is computed as current time minus hours * 3600."""
        mock_conn.fetch.return_value = []
        before = time.time()
        await store.get_recent_topics_async(hours=12)
        after = time.time()

        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        cutoff = call_args[0][1]
        assert "WHERE created_at >= $1" in sql
        # Cutoff should be roughly now - 12*3600
        expected_cutoff_low = before - 12 * 3600
        expected_cutoff_high = after - 12 * 3600
        assert expected_cutoff_low <= cutoff <= expected_cutoff_high

    @pytest.mark.asyncio
    async def test_ordered_by_created_at_desc(self, store, mock_conn):
        """Query orders results by created_at DESC."""
        mock_conn.fetch.return_value = []
        await store.get_recent_topics_async()
        sql = mock_conn.fetch.call_args[0][0]
        assert "ORDER BY created_at DESC" in sql

    @pytest.mark.asyncio
    async def test_default_hours_is_24(self, store, mock_conn):
        """Default hours parameter is 24."""
        mock_conn.fetch.return_value = []
        before = time.time()
        await store.get_recent_topics_async()
        cutoff = mock_conn.fetch.call_args[0][1]
        # cutoff should be ~24 hours ago
        assert cutoff < before
        assert cutoff > before - 25 * 3600


class TestIsDuplicateAsync:
    """Tests for is_duplicate_async."""

    @pytest.mark.asyncio
    async def test_returns_true_when_found(self, store, mock_conn):
        """is_duplicate_async returns True when a matching row exists."""
        mock_conn.fetchrow.return_value = {"?column?": 1}
        result = await store.is_duplicate_async("Some topic")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, store, mock_conn):
        """is_duplicate_async returns False when no matching row."""
        mock_conn.fetchrow.return_value = None
        result = await store.is_duplicate_async("Unique topic")
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_hash_for_lookup(self, store, mock_conn):
        """The query uses topic_hash, not raw topic text."""
        mock_conn.fetchrow.return_value = None
        await store.is_duplicate_async("Check this topic")
        call_args = mock_conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "topic_hash = $1" in sql
        passed_hash = call_args[0][1]
        expected_hash = PostgresScheduledDebateStore.hash_topic("Check this topic")
        assert passed_hash == expected_hash

    @pytest.mark.asyncio
    async def test_custom_hours(self, store, mock_conn):
        """Custom hours parameter adjusts the cutoff."""
        mock_conn.fetchrow.return_value = None
        before = time.time()
        await store.is_duplicate_async("Topic", hours=48)
        cutoff = mock_conn.fetchrow.call_args[0][2]
        expected_cutoff = before - 48 * 3600
        assert abs(cutoff - expected_cutoff) < 2

    @pytest.mark.asyncio
    async def test_limit_1_in_query(self, store, mock_conn):
        """Query uses LIMIT 1 for efficiency."""
        mock_conn.fetchrow.return_value = None
        await store.is_duplicate_async("Topic")
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "LIMIT 1" in sql


class TestGetHistoryAsync:
    """Tests for get_history_async."""

    @pytest.mark.asyncio
    async def test_basic_query_no_filters(self, store, mock_conn):
        """Basic history query without platform or category filters."""
        mock_conn.fetch.return_value = [_make_row(id="r1")]
        results = await store.get_history_async(limit=10, offset=0)
        assert len(results) == 1
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "WHERE 1=1" in sql
        assert "ORDER BY created_at DESC" in sql
        # limit and offset should be passed
        params = call_args[0][1:]
        assert 10 in params
        assert 0 in params

    @pytest.mark.asyncio
    async def test_platform_filter(self, store, mock_conn):
        """Platform filter is applied correctly."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(platform="reddit")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "platform = $1" in sql
        assert call_args[0][1] == "reddit"

    @pytest.mark.asyncio
    async def test_category_filter(self, store, mock_conn):
        """Category filter is applied correctly."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(category="tech")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "category = $1" in sql
        assert call_args[0][1] == "tech"

    @pytest.mark.asyncio
    async def test_both_filters(self, store, mock_conn):
        """Both platform and category filters are applied together."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(platform="hackernews", category="science")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "platform = $1" in sql
        assert "category = $2" in sql
        # params: platform, category, limit, offset
        assert call_args[0][1] == "hackernews"
        assert call_args[0][2] == "science"

    @pytest.mark.asyncio
    async def test_default_limit_and_offset(self, store, mock_conn):
        """Default limit=50, offset=0."""
        mock_conn.fetch.return_value = []
        await store.get_history_async()
        call_args = mock_conn.fetch.call_args
        params = call_args[0][1:]
        assert 50 in params
        assert 0 in params

    @pytest.mark.asyncio
    async def test_custom_limit_offset(self, store, mock_conn):
        """Custom limit and offset are passed to query."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(limit=20, offset=40)
        call_args = mock_conn.fetch.call_args
        params = call_args[0][1:]
        assert 20 in params
        assert 40 in params

    @pytest.mark.asyncio
    async def test_empty_history(self, store, mock_conn):
        """Empty result returns empty list."""
        mock_conn.fetch.return_value = []
        results = await store.get_history_async()
        assert results == []

    @pytest.mark.asyncio
    async def test_parameterized_sql_numbering_with_platform(self, store, mock_conn):
        """Parameter numbering is correct when platform filter is used."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(limit=5, offset=10, platform="reddit")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        # platform=$1, LIMIT $2, OFFSET $3
        assert "platform = $1" in sql
        assert "LIMIT $2" in sql
        assert "OFFSET $3" in sql

    @pytest.mark.asyncio
    async def test_parameterized_sql_numbering_with_category(self, store, mock_conn):
        """Parameter numbering is correct when category filter is used."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(limit=5, offset=10, category="tech")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        # category=$1, LIMIT $2, OFFSET $3
        assert "category = $1" in sql
        assert "LIMIT $2" in sql
        assert "OFFSET $3" in sql

    @pytest.mark.asyncio
    async def test_parameterized_sql_numbering_with_both(self, store, mock_conn):
        """Parameter numbering is correct when both filters are used."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(limit=5, offset=10, platform="reddit", category="tech")
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        # platform=$1, category=$2, LIMIT $3, OFFSET $4
        assert "platform = $1" in sql
        assert "category = $2" in sql
        assert "LIMIT $3" in sql
        assert "OFFSET $4" in sql


class TestCountTotalAsync:
    """Tests for count_total_async."""

    @pytest.mark.asyncio
    async def test_returns_count(self, store, mock_conn):
        """count_total_async returns the count from the query."""
        mock_conn.fetchrow.return_value = {"count": 42}
        result = await store.count_total_async()
        assert result == 42

    @pytest.mark.asyncio
    async def test_zero_count(self, store, mock_conn):
        """count_total_async returns 0 for empty table."""
        mock_conn.fetchrow.return_value = {"count": 0}
        result = await store.count_total_async()
        assert result == 0

    @pytest.mark.asyncio
    async def test_none_row_returns_zero(self, store, mock_conn):
        """count_total_async returns 0 if fetchrow returns None."""
        mock_conn.fetchrow.return_value = None
        result = await store.count_total_async()
        assert result == 0

    @pytest.mark.asyncio
    async def test_query_structure(self, store, mock_conn):
        """Query selects COUNT(*) from scheduled_debates."""
        mock_conn.fetchrow.return_value = {"count": 5}
        await store.count_total_async()
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "COUNT(*)" in sql
        assert "scheduled_debates" in sql


class TestGetAnalyticsAsync:
    """Tests for get_analytics_async."""

    @pytest.mark.asyncio
    async def test_basic_analytics(self, store, mock_conn):
        """get_analytics_async returns a dict with expected keys."""
        # Set up mock responses for multiple fetchrow/fetch calls in order
        mock_conn.fetchrow.side_effect = [
            # Total count
            {"count": 100},
            # Consensus stats
            {
                "total": 80,
                "consensus_count": 60,
                "avg_confidence": 0.85,
                "avg_rounds": 3.2,
            },
        ]
        mock_conn.fetch.side_effect = [
            # By platform
            [
                {"platform": "reddit", "count": 50, "consensus_count": 40},
                {"platform": "hackernews", "count": 30, "consensus_count": 20},
            ],
            # By category
            [
                {"category": "tech", "count": 40, "consensus_count": 30},
                {"category": "science", "count": 20, "consensus_count": 15},
            ],
            # Daily counts
            [
                {"day": "2024-01-15", "count": 10},
                {"day": "2024-01-14", "count": 8},
            ],
        ]

        analytics = await store.get_analytics_async()

        assert analytics["total"] == 100
        assert analytics["completed_debates"] == 80
        assert analytics["consensus_rate"] == 60 / 80
        assert analytics["avg_confidence"] == 0.85
        assert analytics["avg_rounds"] == 3.2
        assert "reddit" in analytics["by_platform"]
        assert analytics["by_platform"]["reddit"]["total"] == 50
        assert "tech" in analytics["by_category"]
        assert analytics["by_category"]["tech"]["total"] == 40
        assert "2024-01-15" in analytics["daily_counts"]

    @pytest.mark.asyncio
    async def test_analytics_empty_database(self, store, mock_conn):
        """Analytics on empty database returns zero counts."""
        mock_conn.fetchrow.side_effect = [
            {"count": 0},  # Total count
            {"total": 0, "consensus_count": 0, "avg_confidence": None, "avg_rounds": None},
        ]
        mock_conn.fetch.side_effect = [
            [],  # By platform
            [],  # By category
            [],  # Daily counts
        ]

        analytics = await store.get_analytics_async()
        assert analytics["total"] == 0
        # completed_debates won't be set because total == 0
        assert "completed_debates" not in analytics
        assert analytics["by_platform"] == {}
        assert analytics["by_category"] == {}
        assert analytics["daily_counts"] == {}

    @pytest.mark.asyncio
    async def test_analytics_no_total_row(self, store, mock_conn):
        """Analytics handles None from fetchrow for total count."""
        mock_conn.fetchrow.side_effect = [
            None,  # Total count
            None,  # Consensus stats
        ]
        mock_conn.fetch.side_effect = [
            [],  # By platform
            [],  # By category
            [],  # Daily counts
        ]
        analytics = await store.get_analytics_async()
        assert analytics["total"] == 0

    @pytest.mark.asyncio
    async def test_analytics_consensus_with_null_avg(self, store, mock_conn):
        """Analytics handles NULL avg_confidence and avg_rounds."""
        mock_conn.fetchrow.side_effect = [
            {"count": 10},
            {"total": 5, "consensus_count": 3, "avg_confidence": None, "avg_rounds": None},
        ]
        mock_conn.fetch.side_effect = [[], [], []]

        analytics = await store.get_analytics_async()
        assert analytics["completed_debates"] == 5
        assert analytics["consensus_rate"] == 3 / 5
        assert analytics["avg_confidence"] == 0.0
        assert analytics["avg_rounds"] == 0.0


class TestFinalizeDebateOutcomeAsync:
    """Tests for finalize_debate_outcome_async."""

    @pytest.mark.asyncio
    async def test_successful_update(self, store, mock_conn):
        """Returns True when update matches a row."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.finalize_debate_outcome_async(
            debate_id="d1",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=3,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_no_matching_row(self, store, mock_conn):
        """Returns False when no row is updated (UPDATE 0)."""
        mock_conn.execute.return_value = "UPDATE 0"
        result = await store.finalize_debate_outcome_async(
            debate_id="nonexistent",
            consensus_reached=False,
            confidence=0.1,
            rounds_used=1,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_sql_params(self, store, mock_conn):
        """SQL passes correct parameters for the update."""
        mock_conn.execute.return_value = "UPDATE 1"
        await store.finalize_debate_outcome_async(
            debate_id="d42",
            consensus_reached=True,
            confidence=0.75,
            rounds_used=5,
        )
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "UPDATE scheduled_debates" in sql
        assert "SET consensus_reached = $1" in sql
        assert "confidence = $2" in sql
        assert "rounds_used = $3" in sql
        assert "WHERE debate_id = $4" in sql
        assert call_args[0][1] is True  # consensus_reached
        assert call_args[0][2] == 0.75  # confidence
        assert call_args[0][3] == 5  # rounds_used
        assert call_args[0][4] == "d42"  # debate_id

    @pytest.mark.asyncio
    async def test_multiple_rows_updated(self, store, mock_conn):
        """Returns True when multiple rows match (UPDATE 3)."""
        mock_conn.execute.return_value = "UPDATE 3"
        result = await store.finalize_debate_outcome_async(
            debate_id="d1",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=3,
        )
        assert result is True


class TestGetPendingOutcomesAsync:
    """Tests for get_pending_outcomes_async."""

    @pytest.mark.asyncio
    async def test_returns_pending_records(self, store, mock_conn):
        """Returns records with debate_id but no outcome."""
        mock_conn.fetch.return_value = [
            _make_row(id="r1", debate_id="d1", consensus_reached=None),
            _make_row(id="r2", debate_id="d2", consensus_reached=None),
        ]
        results = await store.get_pending_outcomes_async()
        assert len(results) == 2
        assert results[0].consensus_reached is None
        assert results[1].consensus_reached is None

    @pytest.mark.asyncio
    async def test_empty_pending(self, store, mock_conn):
        """Returns empty list when no pending outcomes."""
        mock_conn.fetch.return_value = []
        results = await store.get_pending_outcomes_async()
        assert results == []

    @pytest.mark.asyncio
    async def test_custom_limit(self, store, mock_conn):
        """Custom limit is passed to the query."""
        mock_conn.fetch.return_value = []
        await store.get_pending_outcomes_async(limit=25)
        call_args = mock_conn.fetch.call_args
        assert call_args[0][1] == 25

    @pytest.mark.asyncio
    async def test_query_filters_pending(self, store, mock_conn):
        """Query filters for debate_id IS NOT NULL AND consensus_reached IS NULL."""
        mock_conn.fetch.return_value = []
        await store.get_pending_outcomes_async()
        sql = mock_conn.fetch.call_args[0][0]
        assert "debate_id IS NOT NULL" in sql
        assert "consensus_reached IS NULL" in sql

    @pytest.mark.asyncio
    async def test_default_limit_is_100(self, store, mock_conn):
        """Default limit parameter is 100."""
        mock_conn.fetch.return_value = []
        await store.get_pending_outcomes_async()
        call_args = mock_conn.fetch.call_args
        assert call_args[0][1] == 100

    @pytest.mark.asyncio
    async def test_ordered_desc(self, store, mock_conn):
        """Results ordered by created_at DESC."""
        mock_conn.fetch.return_value = []
        await store.get_pending_outcomes_async()
        sql = mock_conn.fetch.call_args[0][0]
        assert "ORDER BY created_at DESC" in sql


class TestCleanupOldAsync:
    """Tests for cleanup_old_async."""

    @pytest.mark.asyncio
    async def test_deletes_and_returns_count(self, store, mock_conn):
        """cleanup_old_async returns the number of deleted records."""
        mock_conn.execute.return_value = "DELETE 15"
        result = await store.cleanup_old_async(days=30)
        assert result == 15

    @pytest.mark.asyncio
    async def test_zero_deleted(self, store, mock_conn):
        """Returns 0 when no records match."""
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.cleanup_old_async(days=7)
        assert result == 0

    @pytest.mark.asyncio
    async def test_cutoff_calculation(self, store, mock_conn):
        """Cutoff is days * 24 * 3600 seconds ago."""
        mock_conn.execute.return_value = "DELETE 0"
        before = time.time()
        await store.cleanup_old_async(days=14)
        after = time.time()

        cutoff = mock_conn.execute.call_args[0][1]
        expected_low = before - 14 * 24 * 3600
        expected_high = after - 14 * 24 * 3600
        assert expected_low <= cutoff <= expected_high

    @pytest.mark.asyncio
    async def test_delete_sql(self, store, mock_conn):
        """SQL deletes from scheduled_debates where created_at < cutoff."""
        mock_conn.execute.return_value = "DELETE 0"
        await store.cleanup_old_async(days=30)
        sql = mock_conn.execute.call_args[0][0]
        assert "DELETE FROM scheduled_debates" in sql
        assert "created_at < $1" in sql

    @pytest.mark.asyncio
    async def test_default_days_is_30(self, store, mock_conn):
        """Default days parameter is 30."""
        mock_conn.execute.return_value = "DELETE 0"
        before = time.time()
        await store.cleanup_old_async()
        cutoff = mock_conn.execute.call_args[0][1]
        expected = before - 30 * 24 * 3600
        assert abs(cutoff - expected) < 2

    @pytest.mark.asyncio
    async def test_malformed_result_returns_zero(self, store, mock_conn):
        """Handles malformed execute result gracefully."""
        mock_conn.execute.return_value = "UNEXPECTED"
        result = await store.cleanup_old_async()
        # "UNEXPECTED".split()[-1] is "UNEXPECTED" which can't be int
        assert result == 0

    @pytest.mark.asyncio
    async def test_empty_result_returns_zero(self, store, mock_conn):
        """Handles empty string result."""
        mock_conn.execute.return_value = ""
        result = await store.cleanup_old_async()
        assert result == 0


# ===========================================================================
# Sync wrapper tests
# ===========================================================================


class TestSyncWrappers:
    """Tests for sync wrapper methods that delegate to async methods."""

    def test_record_scheduled_debate_sync(self, store):
        """record_scheduled_debate calls run_async with async method."""
        rec = _make_record()
        with patch("aragora.pulse.postgres_store.run_async") as mock_run_async:
            store.record_scheduled_debate(rec)
            mock_run_async.assert_called_once()

    def test_get_recent_topics_sync(self, store):
        """get_recent_topics calls run_async with correct hours."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=[]) as mock_run_async:
            result = store.get_recent_topics(hours=12)
            mock_run_async.assert_called_once()
            assert result == []

    def test_is_duplicate_sync(self, store):
        """is_duplicate calls run_async."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=True) as mock_run_async:
            result = store.is_duplicate("Test topic", hours=48)
            mock_run_async.assert_called_once()
            assert result is True

    def test_get_history_sync(self, store):
        """get_history calls run_async with correct params."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=[]) as mock_run_async:
            result = store.get_history(limit=20, offset=5, platform="reddit", category="tech")
            mock_run_async.assert_called_once()
            assert result == []

    def test_count_total_sync(self, store):
        """count_total calls run_async."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=42) as mock_run_async:
            result = store.count_total()
            mock_run_async.assert_called_once()
            assert result == 42

    def test_get_analytics_sync(self, store):
        """get_analytics calls run_async."""
        analytics = {"total": 10}
        with patch(
            "aragora.pulse.postgres_store.run_async", return_value=analytics
        ) as mock_run_async:
            result = store.get_analytics()
            mock_run_async.assert_called_once()
            assert result == analytics

    def test_finalize_debate_outcome_sync(self, store):
        """finalize_debate_outcome calls run_async."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=True) as mock_run_async:
            result = store.finalize_debate_outcome(
                debate_id="d1",
                consensus_reached=True,
                confidence=0.8,
                rounds_used=3,
            )
            mock_run_async.assert_called_once()
            assert result is True

    def test_get_pending_outcomes_sync(self, store):
        """get_pending_outcomes calls run_async."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=[]) as mock_run_async:
            result = store.get_pending_outcomes(limit=50)
            mock_run_async.assert_called_once()
            assert result == []

    def test_cleanup_old_sync(self, store):
        """cleanup_old calls run_async."""
        with patch("aragora.pulse.postgres_store.run_async", return_value=7) as mock_run_async:
            result = store.cleanup_old(days=14)
            mock_run_async.assert_called_once()
            assert result == 7


# ===========================================================================
# Schema and class attribute tests
# ===========================================================================


class TestSchemaAttributes:
    """Tests for class-level schema attributes."""

    def test_schema_name(self):
        """SCHEMA_NAME is set correctly."""
        assert PostgresScheduledDebateStore.SCHEMA_NAME == "scheduled_debates"

    def test_schema_version(self):
        """SCHEMA_VERSION is set correctly."""
        assert PostgresScheduledDebateStore.SCHEMA_VERSION == 1

    def test_initial_schema_contains_table(self):
        """INITIAL_SCHEMA creates the scheduled_debates table."""
        schema = PostgresScheduledDebateStore.INITIAL_SCHEMA
        assert "CREATE TABLE IF NOT EXISTS scheduled_debates" in schema

    def test_initial_schema_contains_indexes(self):
        """INITIAL_SCHEMA creates required indexes."""
        schema = PostgresScheduledDebateStore.INITIAL_SCHEMA
        assert "idx_scheduled_topic_hash" in schema
        assert "idx_scheduled_created_at" in schema
        assert "idx_scheduled_platform" in schema
        assert "idx_scheduled_category" in schema
        assert "idx_scheduled_debate_id" in schema
        assert "idx_scheduled_pending" in schema

    def test_initial_schema_has_primary_key(self):
        """INITIAL_SCHEMA defines id as PRIMARY KEY."""
        schema = PostgresScheduledDebateStore.INITIAL_SCHEMA
        assert "id TEXT PRIMARY KEY" in schema

    def test_initial_schema_partial_index(self):
        """INITIAL_SCHEMA includes a partial index for pending debates."""
        schema = PostgresScheduledDebateStore.INITIAL_SCHEMA
        assert "WHERE debate_id IS NOT NULL AND consensus_reached IS NULL" in schema


# ===========================================================================
# close() method tests
# ===========================================================================


class TestClose:
    """Tests for the close() method."""

    def test_close_is_noop(self, store):
        """close() is a no-op and doesn't raise."""
        store.close()
        # No exception means success

    def test_close_can_be_called_multiple_times(self, store):
        """close() can be called multiple times without error."""
        store.close()
        store.close()
        store.close()


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Tests for module-level __all__ exports."""

    def test_exports(self):
        """__all__ includes expected names."""
        from aragora.pulse import postgres_store

        assert "PostgresScheduledDebateStore" in postgres_store.__all__
        assert "ScheduledDebateRecord" in postgres_store.__all__


# ===========================================================================
# Connection error handling
# ===========================================================================


class TestConnectionErrorHandling:
    """Tests for error handling during database operations."""

    @pytest.mark.asyncio
    async def test_record_debate_propagates_connection_error(self, mock_conn):
        """Connection errors propagate from record_scheduled_debate_async."""
        mock_conn.execute.side_effect = OSError("Connection refused")

        with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
            s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)

        with patch.object(
            PostgresScheduledDebateStore,
            "connection",
            return_value=_fake_connection(mock_conn),
        ):
            with pytest.raises(OSError, match="Connection refused"):
                await s.record_scheduled_debate_async(_make_record())

    @pytest.mark.asyncio
    async def test_fetch_propagates_connection_error(self, mock_conn):
        """Connection errors propagate from get_recent_topics_async."""
        mock_conn.fetch.side_effect = OSError("Connection lost")

        with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
            s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)

        with patch.object(
            PostgresScheduledDebateStore,
            "connection",
            return_value=_fake_connection(mock_conn),
        ):
            with pytest.raises(OSError, match="Connection lost"):
                await s.get_recent_topics_async()

    @pytest.mark.asyncio
    async def test_cleanup_propagates_error(self, mock_conn):
        """Connection errors propagate from cleanup_old_async."""
        mock_conn.execute.side_effect = RuntimeError("Pool exhausted")

        with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
            s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)

        with patch.object(
            PostgresScheduledDebateStore,
            "connection",
            return_value=_fake_connection(mock_conn),
        ):
            with pytest.raises(RuntimeError, match="Pool exhausted"):
                await s.cleanup_old_async()

    @pytest.mark.asyncio
    async def test_is_duplicate_propagates_error(self, mock_conn):
        """Connection errors propagate from is_duplicate_async."""
        mock_conn.fetchrow.side_effect = ConnectionError("Connection reset")

        with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
            s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)

        with patch.object(
            PostgresScheduledDebateStore,
            "connection",
            return_value=_fake_connection(mock_conn),
        ):
            with pytest.raises(ConnectionError, match="Connection reset"):
                await s.is_duplicate_async("test topic")

    @pytest.mark.asyncio
    async def test_analytics_partial_failure(self, mock_conn):
        """Analytics fails cleanly if a query raises mid-stream."""
        mock_conn.fetchrow.side_effect = [
            {"count": 10},  # Total count succeeds
            OSError("Network error"),  # Consensus stats fails
        ]

        with patch.object(PostgresScheduledDebateStore, "__init__", lambda self, *a, **kw: None):
            s = PostgresScheduledDebateStore.__new__(PostgresScheduledDebateStore)

        with patch.object(
            PostgresScheduledDebateStore,
            "connection",
            return_value=_fake_connection(mock_conn),
        ):
            with pytest.raises(OSError, match="Network error"):
                await s.get_analytics_async()


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_hash_very_long_topic(self):
        """hash_topic handles very long topic strings."""
        long_topic = "A" * 10000
        h = PostgresScheduledDebateStore.hash_topic(long_topic)
        assert len(h) == 16

    def test_hash_special_characters(self):
        """hash_topic handles special characters."""
        h = PostgresScheduledDebateStore.hash_topic("SQL injection'; DROP TABLE--")
        assert len(h) == 16

    def test_record_equality(self):
        """Two records with same fields are equal (dataclass default)."""
        now = time.time()
        r1 = _make_record(created_at=now)
        r2 = _make_record(created_at=now)
        assert r1 == r2

    def test_record_inequality(self):
        """Two records with different IDs are not equal."""
        r1 = _make_record(id="a")
        r2 = _make_record(id="b")
        assert r1 != r2

    @pytest.mark.asyncio
    async def test_get_history_no_filters_sql_has_where_1_equals_1(self, store, mock_conn):
        """When no filters, WHERE 1=1 is still present in SQL."""
        mock_conn.fetch.return_value = []
        await store.get_history_async()
        sql = mock_conn.fetch.call_args[0][0]
        assert "WHERE 1=1" in sql

    @pytest.mark.asyncio
    async def test_finalize_outcome_false_consensus(self, store, mock_conn):
        """Finalize with consensus_reached=False is valid."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.finalize_debate_outcome_async(
            debate_id="d1",
            consensus_reached=False,
            confidence=0.2,
            rounds_used=5,
        )
        assert result is True
        assert mock_conn.execute.call_args[0][1] is False

    @pytest.mark.asyncio
    async def test_finalize_outcome_zero_confidence(self, store, mock_conn):
        """Finalize with zero confidence is valid."""
        mock_conn.execute.return_value = "UPDATE 1"
        result = await store.finalize_debate_outcome_async(
            debate_id="d1",
            consensus_reached=True,
            confidence=0.0,
            rounds_used=1,
        )
        assert result is True
        assert mock_conn.execute.call_args[0][2] == 0.0

    @pytest.mark.asyncio
    async def test_cleanup_large_count(self, store, mock_conn):
        """cleanup_old_async handles large delete counts."""
        mock_conn.execute.return_value = "DELETE 999999"
        result = await store.cleanup_old_async()
        assert result == 999999

    @pytest.mark.asyncio
    async def test_multiple_rows_to_records(self, store, mock_conn):
        """Multiple rows are correctly converted to records."""
        rows = [_make_row(id=f"r{i}", platform=f"platform_{i}") for i in range(5)]
        mock_conn.fetch.return_value = rows
        results = await store.get_recent_topics_async()
        assert len(results) == 5
        for i, rec in enumerate(results):
            assert rec.id == f"r{i}"
            assert rec.platform == f"platform_{i}"

    @pytest.mark.asyncio
    async def test_get_history_limit_zero(self, store, mock_conn):
        """limit=0 is passed through to the query."""
        mock_conn.fetch.return_value = []
        await store.get_history_async(limit=0)
        params = mock_conn.fetch.call_args[0][1:]
        assert 0 in params

    @pytest.mark.asyncio
    async def test_cleanup_one_day(self, store, mock_conn):
        """cleanup_old_async works with days=1."""
        mock_conn.execute.return_value = "DELETE 3"
        before = time.time()
        result = await store.cleanup_old_async(days=1)
        assert result == 3
        cutoff = mock_conn.execute.call_args[0][1]
        expected = before - 1 * 24 * 3600
        assert abs(cutoff - expected) < 2
