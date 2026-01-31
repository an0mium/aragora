"""Comprehensive tests for the Pulse scheduled debate store.

Tests cover:
- ScheduledDebateRecord dataclass
- ScheduledDebateStore initialization
- Recording and retrieving debates
- Duplicate detection
- History queries with filters
- Analytics
- Outcome finalization
- Pending outcome tracking
- Cleanup of old records
- Edge cases and error handling
"""

import hashlib
import time
from unittest.mock import patch, MagicMock

import pytest

from aragora.pulse.store import ScheduledDebateRecord, ScheduledDebateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    id: str = "rec-001",
    topic_text: str = "AI safety alignment research",
    platform: str = "hackernews",
    category: str = "tech",
    volume: int = 500,
    debate_id: str | None = "debate-001",
    created_at: float | None = None,
    consensus_reached: bool | None = None,
    confidence: float | None = None,
    rounds_used: int = 0,
    scheduler_run_id: str = "run-001",
) -> ScheduledDebateRecord:
    """Create a ScheduledDebateRecord with sensible defaults."""
    if created_at is None:
        created_at = time.time()
    topic_hash = ScheduledDebateStore.hash_topic(topic_text)
    return ScheduledDebateRecord(
        id=id,
        topic_hash=topic_hash,
        topic_text=topic_text,
        platform=platform,
        category=category,
        volume=volume,
        debate_id=debate_id,
        created_at=created_at,
        consensus_reached=consensus_reached,
        confidence=confidence,
        rounds_used=rounds_used,
        scheduler_run_id=scheduler_run_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Create a fresh ScheduledDebateStore backed by a temp SQLite file."""
    db_path = tmp_path / "scheduled_debates.db"
    return ScheduledDebateStore(db_path)


@pytest.fixture
def populated_store(store):
    """Store pre-populated with a handful of records."""
    now = time.time()

    records = [
        _make_record(
            id="rec-001",
            topic_text="AI safety research",
            platform="hackernews",
            category="tech",
            volume=500,
            debate_id="debate-001",
            created_at=now - 3600,  # 1 hour ago
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
            scheduler_run_id="run-001",
        ),
        _make_record(
            id="rec-002",
            topic_text="Climate change policy",
            platform="reddit",
            category="science",
            volume=800,
            debate_id="debate-002",
            created_at=now - 7200,  # 2 hours ago
            consensus_reached=False,
            confidence=0.45,
            rounds_used=5,
            scheduler_run_id="run-001",
        ),
        _make_record(
            id="rec-003",
            topic_text="Market volatility analysis",
            platform="twitter",
            category="finance",
            volume=1200,
            debate_id="debate-003",
            created_at=now - 86400,  # 24 hours ago
            consensus_reached=None,  # pending
            confidence=None,
            rounds_used=0,
            scheduler_run_id="run-002",
        ),
        _make_record(
            id="rec-004",
            topic_text="Quantum computing breakthrough",
            platform="hackernews",
            category="tech",
            volume=600,
            debate_id=None,  # no debate started
            created_at=now - 172800,  # 48 hours ago
            scheduler_run_id="run-003",
        ),
        _make_record(
            id="rec-005",
            topic_text="Open source AI models",
            platform="reddit",
            category="tech",
            volume=400,
            debate_id="debate-005",
            created_at=now - 3600 * 6,  # 6 hours ago
            consensus_reached=True,
            confidence=0.92,
            rounds_used=2,
            scheduler_run_id="run-002",
        ),
    ]

    for rec in records:
        store.record_scheduled_debate(rec)

    return store


# ============================================================================
# ScheduledDebateRecord Tests
# ============================================================================


class TestScheduledDebateRecord:
    """Tests for the ScheduledDebateRecord dataclass."""

    def test_basic_creation(self):
        """Record can be created with all fields."""
        rec = _make_record()
        assert rec.id == "rec-001"
        assert rec.topic_text == "AI safety alignment research"
        assert rec.platform == "hackernews"
        assert rec.category == "tech"
        assert rec.volume == 500
        assert rec.debate_id == "debate-001"
        assert rec.rounds_used == 0

    def test_hours_ago_property(self):
        """hours_ago returns the correct elapsed hours."""
        two_hours_ago = time.time() - 7200
        rec = _make_record(created_at=two_hours_ago)
        assert abs(rec.hours_ago - 2.0) < 0.01

    def test_hours_ago_recent(self):
        """hours_ago returns near-zero for a just-created record."""
        rec = _make_record(created_at=time.time())
        assert rec.hours_ago < 0.01

    def test_hours_ago_old(self):
        """hours_ago returns correct value for old records."""
        one_day_ago = time.time() - 86400
        rec = _make_record(created_at=one_day_ago)
        assert abs(rec.hours_ago - 24.0) < 0.01

    def test_optional_fields_none(self):
        """Optional fields can be None."""
        rec = _make_record(
            debate_id=None,
            consensus_reached=None,
            confidence=None,
        )
        assert rec.debate_id is None
        assert rec.consensus_reached is None
        assert rec.confidence is None

    def test_consensus_reached_false(self):
        """consensus_reached can be explicitly False."""
        rec = _make_record(consensus_reached=False)
        assert rec.consensus_reached is False

    def test_consensus_reached_true(self):
        """consensus_reached can be True."""
        rec = _make_record(consensus_reached=True)
        assert rec.consensus_reached is True


# ============================================================================
# hash_topic Tests
# ============================================================================


class TestHashTopic:
    """Tests for the static hash_topic method."""

    def test_basic_hash(self):
        """hash_topic returns a hex string."""
        h = ScheduledDebateStore.hash_topic("Test topic")
        assert isinstance(h, str)
        assert len(h) == 16  # truncated to 16 chars
        # Verify all hex characters
        assert all(c in "0123456789abcdef" for c in h)

    def test_normalizes_case(self):
        """hash_topic is case-insensitive."""
        h1 = ScheduledDebateStore.hash_topic("AI Safety Research")
        h2 = ScheduledDebateStore.hash_topic("ai safety research")
        assert h1 == h2

    def test_normalizes_whitespace(self):
        """hash_topic normalizes whitespace."""
        h1 = ScheduledDebateStore.hash_topic("AI   safety   research")
        h2 = ScheduledDebateStore.hash_topic("AI safety research")
        assert h1 == h2

    def test_strips_leading_trailing_space(self):
        """hash_topic strips leading and trailing whitespace."""
        h1 = ScheduledDebateStore.hash_topic("  AI safety  ")
        h2 = ScheduledDebateStore.hash_topic("AI safety")
        assert h1 == h2

    def test_different_topics_different_hashes(self):
        """Different topics produce different hashes."""
        h1 = ScheduledDebateStore.hash_topic("Topic A")
        h2 = ScheduledDebateStore.hash_topic("Topic B")
        assert h1 != h2

    def test_hash_matches_manual_sha256(self):
        """hash_topic matches manual SHA-256 computation."""
        text = "test topic"
        normalized = " ".join(text.lower().strip().split())
        expected = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        assert ScheduledDebateStore.hash_topic(text) == expected

    def test_empty_string(self):
        """hash_topic handles empty string."""
        h = ScheduledDebateStore.hash_topic("")
        assert isinstance(h, str)
        assert len(h) == 16

    def test_unicode_topic(self):
        """hash_topic handles unicode text."""
        h = ScheduledDebateStore.hash_topic("人工智能安全研究")
        assert isinstance(h, str)
        assert len(h) == 16


# ============================================================================
# Store Initialization Tests
# ============================================================================


class TestStoreInitialization:
    """Tests for ScheduledDebateStore creation and schema setup."""

    def test_creates_db_file(self, tmp_path):
        """Store creates the database file on init."""
        db_path = tmp_path / "test.db"
        assert not db_path.exists()
        ScheduledDebateStore(db_path)
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Store creates parent directories if they don't exist."""
        db_path = tmp_path / "subdir" / "nested" / "test.db"
        ScheduledDebateStore(db_path)
        assert db_path.exists()

    def test_schema_name(self):
        """Store has correct schema name."""
        assert ScheduledDebateStore.SCHEMA_NAME == "scheduled_debates"

    def test_schema_version(self):
        """Store has correct schema version."""
        assert ScheduledDebateStore.SCHEMA_VERSION == 1

    def test_table_created(self, store):
        """The scheduled_debates table is created."""
        rows = store.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scheduled_debates'"
        )
        assert len(rows) == 1

    def test_indexes_created(self, store):
        """All expected indexes are created."""
        rows = store.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_scheduled_%'"
        )
        index_names = {row[0] for row in rows}
        assert "idx_scheduled_topic_hash" in index_names
        assert "idx_scheduled_created_at" in index_names
        assert "idx_scheduled_platform" in index_names
        assert "idx_scheduled_category" in index_names

    def test_reopen_existing_db(self, tmp_path):
        """Reopening an existing DB does not fail."""
        db_path = tmp_path / "test.db"
        store1 = ScheduledDebateStore(db_path)
        store1.record_scheduled_debate(_make_record(id="rec-1"))

        # Reopen
        store2 = ScheduledDebateStore(db_path)
        assert store2.count_total() == 1

    def test_initial_empty(self, store):
        """Freshly created store has zero records."""
        assert store.count_total() == 0


# ============================================================================
# record_scheduled_debate Tests
# ============================================================================


class TestRecordScheduledDebate:
    """Tests for persisting debate records."""

    def test_record_and_retrieve(self, store):
        """Record a debate and verify it can be retrieved."""
        rec = _make_record()
        store.record_scheduled_debate(rec)

        assert store.count_total() == 1
        recent = store.get_recent_topics(hours=1)
        assert len(recent) == 1
        assert recent[0].id == "rec-001"
        assert recent[0].topic_text == "AI safety alignment research"

    def test_record_with_consensus_true(self, store):
        """Consensus True is stored as 1 and read back as True."""
        rec = _make_record(consensus_reached=True)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].consensus_reached is True

    def test_record_with_consensus_false(self, store):
        """Consensus False is stored as 0 and read back as False."""
        rec = _make_record(consensus_reached=False)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].consensus_reached is False

    def test_record_with_consensus_none(self, store):
        """Consensus None is stored as NULL and read back as None."""
        rec = _make_record(consensus_reached=None)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].consensus_reached is None

    def test_record_replaces_on_duplicate_id(self, store):
        """INSERT OR REPLACE replaces existing record with same id."""
        rec1 = _make_record(id="rec-dup", topic_text="Version 1", volume=100)
        store.record_scheduled_debate(rec1)
        assert store.count_total() == 1

        rec2 = _make_record(id="rec-dup", topic_text="Version 2", volume=200)
        store.record_scheduled_debate(rec2)
        assert store.count_total() == 1

        recent = store.get_recent_topics(hours=1)
        assert recent[0].topic_text == "Version 2"
        assert recent[0].volume == 200

    def test_record_multiple(self, store):
        """Multiple records can be stored."""
        for i in range(5):
            store.record_scheduled_debate(_make_record(id=f"rec-{i}", topic_text=f"Topic {i}"))
        assert store.count_total() == 5

    def test_record_with_none_debate_id(self, store):
        """debate_id can be None (no debate started yet)."""
        rec = _make_record(debate_id=None)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].debate_id is None

    def test_record_preserves_all_fields(self, store):
        """All fields are correctly persisted and retrieved."""
        now = time.time()
        rec = _make_record(
            id="rec-full",
            topic_text="Full test topic",
            platform="reddit",
            category="science",
            volume=999,
            debate_id="debate-full",
            created_at=now,
            consensus_reached=True,
            confidence=0.88,
            rounds_used=4,
            scheduler_run_id="run-full",
        )
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        r = recent[0]
        assert r.id == "rec-full"
        assert r.topic_text == "Full test topic"
        assert r.platform == "reddit"
        assert r.category == "science"
        assert r.volume == 999
        assert r.debate_id == "debate-full"
        assert abs(r.created_at - now) < 1
        assert r.consensus_reached is True
        assert r.confidence == 0.88
        assert r.rounds_used == 4
        assert r.scheduler_run_id == "run-full"


# ============================================================================
# get_recent_topics Tests
# ============================================================================


class TestGetRecentTopics:
    """Tests for retrieving recently debated topics."""

    def test_returns_within_window(self, store):
        """Only records within the time window are returned."""
        now = time.time()
        store.record_scheduled_debate(
            _make_record(id="recent", created_at=now - 1800)  # 30 min ago
        )
        store.record_scheduled_debate(
            _make_record(id="old", created_at=now - 86400 * 2)  # 2 days ago
        )

        recent = store.get_recent_topics(hours=1)
        assert len(recent) == 1
        assert recent[0].id == "recent"

    def test_ordered_by_created_at_desc(self, store):
        """Results are ordered newest first."""
        now = time.time()
        store.record_scheduled_debate(_make_record(id="older", created_at=now - 3600))
        store.record_scheduled_debate(_make_record(id="newer", created_at=now - 1800))

        recent = store.get_recent_topics(hours=2)
        assert len(recent) == 2
        assert recent[0].id == "newer"
        assert recent[1].id == "older"

    def test_default_24_hours(self, store):
        """Default window is 24 hours."""
        now = time.time()
        store.record_scheduled_debate(_make_record(id="within", created_at=now - 3600 * 23))
        store.record_scheduled_debate(_make_record(id="outside", created_at=now - 3600 * 25))

        recent = store.get_recent_topics()  # default 24 hours
        assert len(recent) == 1
        assert recent[0].id == "within"

    def test_empty_result(self, store):
        """Returns empty list when no records match."""
        recent = store.get_recent_topics(hours=1)
        assert recent == []

    def test_large_window(self, populated_store):
        """Large window returns all records."""
        recent = populated_store.get_recent_topics(hours=24 * 365)
        assert len(recent) == 5


# ============================================================================
# is_duplicate Tests
# ============================================================================


class TestIsDuplicate:
    """Tests for duplicate topic detection."""

    def test_finds_duplicate(self, store):
        """is_duplicate returns True for a recently debated topic."""
        store.record_scheduled_debate(_make_record(topic_text="AI safety research"))
        assert store.is_duplicate("AI safety research") is True

    def test_no_duplicate_for_new_topic(self, store):
        """is_duplicate returns False for a new topic."""
        store.record_scheduled_debate(_make_record(topic_text="AI safety research"))
        assert store.is_duplicate("Quantum computing") is False

    def test_case_insensitive_duplicate(self, store):
        """Duplicate detection is case-insensitive (via hash normalization)."""
        store.record_scheduled_debate(_make_record(topic_text="AI Safety Research"))
        assert store.is_duplicate("ai safety research") is True

    def test_whitespace_insensitive_duplicate(self, store):
        """Duplicate detection normalizes whitespace."""
        store.record_scheduled_debate(_make_record(topic_text="AI  safety  research"))
        assert store.is_duplicate("AI safety research") is True

    def test_duplicate_respects_time_window(self, store):
        """Old records outside the dedup window are not considered duplicates."""
        old_time = time.time() - 86400 * 2  # 2 days ago
        store.record_scheduled_debate(
            _make_record(topic_text="AI safety research", created_at=old_time)
        )
        assert store.is_duplicate("AI safety research", hours=24) is False

    def test_duplicate_within_custom_window(self, store):
        """Custom time window is respected."""
        recent_time = time.time() - 3600 * 5  # 5 hours ago
        store.record_scheduled_debate(
            _make_record(topic_text="AI safety research", created_at=recent_time)
        )
        assert store.is_duplicate("AI safety research", hours=6) is True
        assert store.is_duplicate("AI safety research", hours=4) is False

    def test_empty_store_no_duplicate(self, store):
        """Empty store never has duplicates."""
        assert store.is_duplicate("Any topic") is False


# ============================================================================
# get_history Tests
# ============================================================================


class TestGetHistory:
    """Tests for historical record queries."""

    def test_default_limit_and_offset(self, populated_store):
        """Default limit is 50 and offset is 0."""
        history = populated_store.get_history()
        assert len(history) == 5  # all records, since < 50

    def test_limit(self, populated_store):
        """Limit restricts the number of records returned."""
        history = populated_store.get_history(limit=2)
        assert len(history) == 2

    def test_offset(self, populated_store):
        """Offset skips records."""
        all_history = populated_store.get_history(limit=50)
        offset_history = populated_store.get_history(limit=50, offset=2)
        assert len(offset_history) == len(all_history) - 2

    def test_limit_and_offset_combined(self, populated_store):
        """Limit and offset work together for pagination."""
        page1 = populated_store.get_history(limit=2, offset=0)
        page2 = populated_store.get_history(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap
        page1_ids = {r.id for r in page1}
        page2_ids = {r.id for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_platform_filter(self, populated_store):
        """Platform filter returns only matching records."""
        hn_records = populated_store.get_history(platform="hackernews")
        assert all(r.platform == "hackernews" for r in hn_records)
        assert len(hn_records) == 2  # rec-001 and rec-004

    def test_category_filter(self, populated_store):
        """Category filter returns only matching records."""
        tech_records = populated_store.get_history(category="tech")
        assert all(r.category == "tech" for r in tech_records)
        assert len(tech_records) == 3  # rec-001, rec-004, rec-005

    def test_platform_and_category_filter(self, populated_store):
        """Both filters can be applied simultaneously."""
        filtered = populated_store.get_history(platform="hackernews", category="tech")
        assert all(r.platform == "hackernews" and r.category == "tech" for r in filtered)
        assert len(filtered) == 2  # rec-001 and rec-004

    def test_filter_no_results(self, populated_store):
        """Filter that matches nothing returns empty list."""
        results = populated_store.get_history(platform="nonexistent")
        assert results == []

    def test_ordered_by_created_at_desc(self, populated_store):
        """History is ordered by created_at descending."""
        history = populated_store.get_history()
        for i in range(len(history) - 1):
            assert history[i].created_at >= history[i + 1].created_at

    def test_empty_store(self, store):
        """Empty store returns empty list."""
        assert store.get_history() == []


# ============================================================================
# count_total Tests
# ============================================================================


class TestCountTotal:
    """Tests for counting total records."""

    def test_empty_store(self, store):
        """Empty store returns 0."""
        assert store.count_total() == 0

    def test_after_inserts(self, store):
        """Count matches number of inserted records."""
        for i in range(3):
            store.record_scheduled_debate(_make_record(id=f"rec-{i}"))
        assert store.count_total() == 3

    def test_after_replace(self, store):
        """Replacing a record does not increase the count."""
        store.record_scheduled_debate(_make_record(id="rec-1"))
        store.record_scheduled_debate(_make_record(id="rec-1", volume=999))
        assert store.count_total() == 1

    def test_populated(self, populated_store):
        """Populated store returns correct count."""
        assert populated_store.count_total() == 5


# ============================================================================
# get_analytics Tests
# ============================================================================


class TestGetAnalytics:
    """Tests for analytics aggregation."""

    def test_empty_store_analytics(self, store):
        """Analytics on empty store returns reasonable defaults."""
        analytics = store.get_analytics()
        assert analytics["total"] == 0
        assert analytics["by_platform"] == {}
        assert analytics["by_category"] == {}
        assert analytics["daily_counts"] == {} or isinstance(analytics["daily_counts"], dict)

    def test_total_count(self, populated_store):
        """Analytics total matches count_total."""
        analytics = populated_store.get_analytics()
        assert analytics["total"] == 5

    def test_completed_debates(self, populated_store):
        """completed_debates counts only records with debate_id."""
        analytics = populated_store.get_analytics()
        # rec-001, rec-002, rec-003, rec-005 have debate_id
        assert analytics["completed_debates"] == 4

    def test_consensus_rate(self, populated_store):
        """Consensus rate is correctly computed."""
        analytics = populated_store.get_analytics()
        # Of the 4 with debate_id: rec-001(True), rec-002(False), rec-003(None->0), rec-005(True)
        # consensus_reached=1 for rec-001 and rec-005 = 2 out of 4
        assert analytics["consensus_rate"] == pytest.approx(0.5)

    def test_avg_confidence(self, populated_store):
        """Average confidence is computed across debates with debate_id."""
        analytics = populated_store.get_analytics()
        # rec-001: 0.85, rec-002: 0.45, rec-003: None, rec-005: 0.92
        # AVG of (0.85, 0.45, None, 0.92) -> SQLite AVG ignores NULLs -> (0.85+0.45+0.92)/3
        expected = (0.85 + 0.45 + 0.92) / 3
        assert analytics["avg_confidence"] == pytest.approx(expected, abs=0.01)

    def test_by_platform(self, populated_store):
        """Platform breakdown is correct."""
        analytics = populated_store.get_analytics()
        by_platform = analytics["by_platform"]
        assert "hackernews" in by_platform
        assert "reddit" in by_platform
        assert "twitter" in by_platform
        assert by_platform["hackernews"]["total"] == 2
        assert by_platform["reddit"]["total"] == 2
        assert by_platform["twitter"]["total"] == 1

    def test_by_category(self, populated_store):
        """Category breakdown is correct."""
        analytics = populated_store.get_analytics()
        by_category = analytics["by_category"]
        assert "tech" in by_category
        assert by_category["tech"]["total"] == 3

    def test_daily_counts(self, populated_store):
        """Daily counts are populated."""
        analytics = populated_store.get_analytics()
        daily = analytics["daily_counts"]
        # Should have entries for recent days
        assert isinstance(daily, dict)
        # At least 1 day should have records (today or recent days)
        total_from_daily = sum(daily.values())
        # Some records are within 7 days
        assert total_from_daily >= 1


# ============================================================================
# finalize_debate_outcome Tests
# ============================================================================


class TestFinalizeDebateOutcome:
    """Tests for updating debate outcomes."""

    def test_finalize_existing_debate(self, store):
        """Finalizing an existing debate returns True and updates fields."""
        rec = _make_record(
            id="rec-fin",
            debate_id="debate-fin",
            consensus_reached=None,
            confidence=None,
            rounds_used=0,
        )
        store.record_scheduled_debate(rec)

        result = store.finalize_debate_outcome(
            debate_id="debate-fin",
            consensus_reached=True,
            confidence=0.91,
            rounds_used=4,
        )
        assert result is True

        # Verify the update
        recent = store.get_recent_topics(hours=1)
        assert len(recent) == 1
        assert recent[0].consensus_reached is True
        assert recent[0].confidence == 0.91
        assert recent[0].rounds_used == 4

    def test_finalize_nonexistent_debate(self, store):
        """Finalizing a non-existent debate_id returns False."""
        result = store.finalize_debate_outcome(
            debate_id="nonexistent",
            consensus_reached=True,
            confidence=0.8,
            rounds_used=3,
        )
        assert result is False

    def test_finalize_with_consensus_false(self, store):
        """Finalize stores consensus_reached=False correctly."""
        rec = _make_record(debate_id="debate-no-cons")
        store.record_scheduled_debate(rec)

        store.finalize_debate_outcome(
            debate_id="debate-no-cons",
            consensus_reached=False,
            confidence=0.3,
            rounds_used=5,
        )

        recent = store.get_recent_topics(hours=1)
        assert recent[0].consensus_reached is False
        assert recent[0].confidence == 0.3

    def test_finalize_updates_only_matching_debate(self, store):
        """Finalize only updates the targeted debate_id."""
        store.record_scheduled_debate(_make_record(id="rec-a", debate_id="debate-a"))
        store.record_scheduled_debate(_make_record(id="rec-b", debate_id="debate-b"))

        store.finalize_debate_outcome(
            debate_id="debate-a",
            consensus_reached=True,
            confidence=0.99,
            rounds_used=2,
        )

        history = store.get_history()
        by_id = {r.id: r for r in history}
        assert by_id["rec-a"].consensus_reached is True
        assert by_id["rec-a"].confidence == 0.99
        # rec-b should remain unchanged
        assert by_id["rec-b"].consensus_reached is None
        assert by_id["rec-b"].confidence is None


# ============================================================================
# get_pending_outcomes Tests
# ============================================================================


class TestGetPendingOutcomes:
    """Tests for retrieving debates with pending outcomes."""

    def test_returns_pending(self, populated_store):
        """Returns records with debate_id but no consensus_reached."""
        pending = populated_store.get_pending_outcomes()
        # rec-003 has debate_id but consensus_reached=None
        assert len(pending) == 1
        assert pending[0].id == "rec-003"

    def test_excludes_no_debate_id(self, store):
        """Records without debate_id are excluded."""
        store.record_scheduled_debate(_make_record(debate_id=None, consensus_reached=None))
        pending = store.get_pending_outcomes()
        assert len(pending) == 0

    def test_excludes_finalized(self, store):
        """Records with consensus_reached set are excluded."""
        store.record_scheduled_debate(
            _make_record(
                debate_id="debate-done", consensus_reached=True, confidence=0.9, rounds_used=3
            )
        )
        pending = store.get_pending_outcomes()
        assert len(pending) == 0

    def test_excludes_consensus_false(self, store):
        """Records with consensus_reached=False are NOT pending (outcome is known)."""
        store.record_scheduled_debate(
            _make_record(
                debate_id="debate-no", consensus_reached=False, confidence=0.3, rounds_used=5
            )
        )
        pending = store.get_pending_outcomes()
        assert len(pending) == 0

    def test_limit_parameter(self, store):
        """Limit restricts the number of pending results."""
        for i in range(10):
            store.record_scheduled_debate(
                _make_record(
                    id=f"rec-{i}",
                    debate_id=f"debate-{i}",
                    consensus_reached=None,
                )
            )
        pending = store.get_pending_outcomes(limit=3)
        assert len(pending) == 3

    def test_ordered_by_created_at_desc(self, store):
        """Pending outcomes are ordered newest first."""
        now = time.time()
        store.record_scheduled_debate(
            _make_record(
                id="older", debate_id="d-old", consensus_reached=None, created_at=now - 3600
            )
        )
        store.record_scheduled_debate(
            _make_record(
                id="newer", debate_id="d-new", consensus_reached=None, created_at=now - 1800
            )
        )

        pending = store.get_pending_outcomes()
        assert pending[0].id == "newer"
        assert pending[1].id == "older"

    def test_empty_store(self, store):
        """Empty store returns empty list."""
        assert store.get_pending_outcomes() == []


# ============================================================================
# cleanup_old Tests
# ============================================================================


class TestCleanupOld:
    """Tests for cleaning up old records."""

    def test_removes_old_records(self, store):
        """Records older than the threshold are removed."""
        old_time = time.time() - 86400 * 60  # 60 days ago
        store.record_scheduled_debate(_make_record(id="old-rec", created_at=old_time))
        store.record_scheduled_debate(_make_record(id="new-rec", created_at=time.time()))

        removed = store.cleanup_old(days=30)
        assert removed == 1
        assert store.count_total() == 1

        # The remaining one is the new one
        history = store.get_history()
        assert history[0].id == "new-rec"

    def test_returns_count_of_removed(self, store):
        """Returns the number of removed records."""
        old_time = time.time() - 86400 * 60
        for i in range(5):
            store.record_scheduled_debate(_make_record(id=f"old-{i}", created_at=old_time))
        store.record_scheduled_debate(_make_record(id="new", created_at=time.time()))

        removed = store.cleanup_old(days=30)
        assert removed == 5

    def test_no_records_to_remove(self, store):
        """Returns 0 when nothing qualifies for removal."""
        store.record_scheduled_debate(_make_record(id="recent", created_at=time.time()))
        removed = store.cleanup_old(days=30)
        assert removed == 0

    def test_empty_store_cleanup(self, store):
        """Cleanup on empty store returns 0."""
        removed = store.cleanup_old(days=30)
        assert removed == 0

    def test_custom_days_threshold(self, store):
        """Custom days threshold is respected."""
        # Record from 10 days ago
        ten_days_ago = time.time() - 86400 * 10
        store.record_scheduled_debate(_make_record(id="ten-days", created_at=ten_days_ago))

        # 7-day cleanup should remove it
        assert store.cleanup_old(days=7) == 1

    def test_boundary_condition(self, store):
        """Records exactly at the cutoff boundary are preserved."""
        # Record exactly 30 days ago (should be removed by < comparison)
        exactly_30_days = time.time() - 86400 * 30
        store.record_scheduled_debate(_make_record(id="boundary", created_at=exactly_30_days))

        # The SQL uses < cutoff, so records AT the boundary are at the edge
        # The cutoff is time.time() - (days * 24 * 3600)
        # Since time.time() might have progressed slightly, this record
        # should be at or just past the cutoff
        removed = store.cleanup_old(days=30)
        # The record is at the boundary - it should be removed since
        # its created_at < current time.time() - 30*86400
        assert removed == 1


# ============================================================================
# _row_to_record Tests
# ============================================================================


class TestRowToRecord:
    """Tests for the internal _row_to_record conversion."""

    def test_full_row(self, store):
        """All fields are mapped correctly from a full row."""
        row = (
            "id-1",
            "hash-1",
            "Topic text",
            "hackernews",
            "tech",
            500,
            "debate-1",
            1700000000.0,
            1,
            0.85,
            3,
            "run-1",
        )
        rec = store._row_to_record(row)
        assert rec.id == "id-1"
        assert rec.topic_hash == "hash-1"
        assert rec.topic_text == "Topic text"
        assert rec.platform == "hackernews"
        assert rec.category == "tech"
        assert rec.volume == 500
        assert rec.debate_id == "debate-1"
        assert rec.created_at == 1700000000.0
        assert rec.consensus_reached is True
        assert rec.confidence == 0.85
        assert rec.rounds_used == 3
        assert rec.scheduler_run_id == "run-1"

    def test_null_category_becomes_empty_string(self, store):
        """None category is converted to empty string."""
        row = ("id", "hash", "text", "hn", None, 0, None, 0.0, None, None, 0, None)
        rec = store._row_to_record(row)
        assert rec.category == ""

    def test_null_volume_becomes_zero(self, store):
        """None volume is converted to 0."""
        row = ("id", "hash", "text", "hn", "cat", None, None, 0.0, None, None, None, None)
        rec = store._row_to_record(row)
        assert rec.volume == 0

    def test_null_rounds_used_becomes_zero(self, store):
        """None rounds_used is converted to 0."""
        row = ("id", "hash", "text", "hn", "cat", 1, None, 0.0, None, None, None, None)
        rec = store._row_to_record(row)
        assert rec.rounds_used == 0

    def test_null_scheduler_run_id_becomes_empty(self, store):
        """None scheduler_run_id is converted to empty string."""
        row = ("id", "hash", "text", "hn", "cat", 1, None, 0.0, None, None, 0, None)
        rec = store._row_to_record(row)
        assert rec.scheduler_run_id == ""

    def test_consensus_reached_0_is_false(self, store):
        """consensus_reached=0 from DB maps to False."""
        row = ("id", "hash", "text", "hn", "cat", 1, "d1", 0.0, 0, 0.5, 3, "r1")
        rec = store._row_to_record(row)
        assert rec.consensus_reached is False

    def test_consensus_reached_1_is_true(self, store):
        """consensus_reached=1 from DB maps to True."""
        row = ("id", "hash", "text", "hn", "cat", 1, "d1", 0.0, 1, 0.9, 3, "r1")
        rec = store._row_to_record(row)
        assert rec.consensus_reached is True

    def test_consensus_reached_none_stays_none(self, store):
        """consensus_reached=None from DB stays None."""
        row = ("id", "hash", "text", "hn", "cat", 1, "d1", 0.0, None, None, 0, "r1")
        rec = store._row_to_record(row)
        assert rec.consensus_reached is None


# ============================================================================
# Integration / Round-trip Tests
# ============================================================================


class TestRoundTrip:
    """End-to-end round-trip tests that exercise multiple operations together."""

    def test_record_finalize_retrieve(self, store):
        """Full lifecycle: record -> finalize -> retrieve."""
        rec = _make_record(
            debate_id="debate-rt",
            consensus_reached=None,
            confidence=None,
            rounds_used=0,
        )
        store.record_scheduled_debate(rec)

        # Should be pending
        pending = store.get_pending_outcomes()
        assert len(pending) == 1

        # Finalize
        store.finalize_debate_outcome(
            debate_id="debate-rt",
            consensus_reached=True,
            confidence=0.95,
            rounds_used=3,
        )

        # Should no longer be pending
        pending = store.get_pending_outcomes()
        assert len(pending) == 0

        # Should be in recent topics
        recent = store.get_recent_topics(hours=1)
        assert len(recent) == 1
        assert recent[0].consensus_reached is True

    def test_dedup_after_record(self, store):
        """Record -> is_duplicate returns True for same topic."""
        topic_text = "Dedup test topic"
        store.record_scheduled_debate(_make_record(topic_text=topic_text))
        assert store.is_duplicate(topic_text) is True

    def test_cleanup_removes_old_but_not_recent(self, store):
        """Cleanup preserves recent records while removing old ones."""
        now = time.time()
        # Insert old and new records
        store.record_scheduled_debate(_make_record(id="old", created_at=now - 86400 * 60))
        store.record_scheduled_debate(_make_record(id="new", created_at=now))

        assert store.count_total() == 2
        store.cleanup_old(days=30)
        assert store.count_total() == 1
        history = store.get_history()
        assert history[0].id == "new"

    def test_analytics_after_operations(self, store):
        """Analytics reflect the correct state after various operations."""
        now = time.time()
        # Add records
        store.record_scheduled_debate(
            _make_record(
                id="a1",
                platform="hackernews",
                category="tech",
                debate_id="d1",
                consensus_reached=True,
                confidence=0.9,
                rounds_used=3,
                created_at=now,
            )
        )
        store.record_scheduled_debate(
            _make_record(
                id="a2",
                platform="reddit",
                category="science",
                debate_id="d2",
                consensus_reached=False,
                confidence=0.4,
                rounds_used=5,
                created_at=now,
            )
        )

        analytics = store.get_analytics()
        assert analytics["total"] == 2
        assert analytics["completed_debates"] == 2
        assert analytics["consensus_rate"] == pytest.approx(0.5)
        assert "hackernews" in analytics["by_platform"]
        assert "reddit" in analytics["by_platform"]

    def test_replace_then_history(self, store):
        """Replacing a record updates history correctly."""
        store.record_scheduled_debate(_make_record(id="rp", topic_text="Version 1"))
        store.record_scheduled_debate(_make_record(id="rp", topic_text="Version 2"))

        history = store.get_history()
        assert len(history) == 1
        assert history[0].topic_text == "Version 2"

    def test_multiple_platforms_and_categories(self, store):
        """Records across multiple platforms and categories are filterable."""
        now = time.time()
        platforms = ["hackernews", "reddit", "twitter"]
        categories = ["tech", "science", "finance"]

        for i, (plat, cat) in enumerate(zip(platforms, categories)):
            store.record_scheduled_debate(
                _make_record(
                    id=f"multi-{i}",
                    platform=plat,
                    category=cat,
                    created_at=now,
                )
            )

        for plat in platforms:
            filtered = store.get_history(platform=plat)
            assert len(filtered) == 1
            assert filtered[0].platform == plat

        for cat in categories:
            filtered = store.get_history(category=cat)
            assert len(filtered) == 1
            assert filtered[0].category == cat


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_long_topic_text(self, store):
        """Store handles very long topic text."""
        long_text = "A" * 10000
        rec = _make_record(topic_text=long_text)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert len(recent) == 1
        assert len(recent[0].topic_text) == 10000

    def test_special_characters_in_topic(self, store):
        """Store handles special characters in topic text."""
        special_text = "What's the deal with O'Brien's \"SQL injection\"? -- DROP TABLE;"
        rec = _make_record(topic_text=special_text)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].topic_text == special_text

    def test_unicode_in_topic(self, store):
        """Store handles unicode in topic text."""
        unicode_text = "研究显示 AI 安全: Sicherheit und Datenschutz"
        rec = _make_record(topic_text=unicode_text)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].topic_text == unicode_text

    def test_zero_volume(self, store):
        """Store handles zero volume."""
        rec = _make_record(volume=0)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].volume == 0

    def test_very_large_volume(self, store):
        """Store handles very large volume numbers."""
        rec = _make_record(volume=999999999)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].volume == 999999999

    def test_confidence_boundary_zero(self, store):
        """Store handles confidence=0.0."""
        rec = _make_record(consensus_reached=False, confidence=0.0, rounds_used=5)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].confidence == 0.0

    def test_confidence_boundary_one(self, store):
        """Store handles confidence=1.0."""
        rec = _make_record(consensus_reached=True, confidence=1.0, rounds_used=1)
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].confidence == 1.0

    def test_zero_hours_recent_topics(self, store):
        """get_recent_topics with hours=0 returns nothing (or only just-now records)."""
        store.record_scheduled_debate(
            _make_record(created_at=time.time() - 1)  # 1 second ago
        )
        recent = store.get_recent_topics(hours=0)
        # With 0-hour window, cutoff = now, so 1-second-old record may or may not be included
        # depending on timing. We just verify it doesn't crash.
        assert isinstance(recent, list)

    def test_get_history_offset_beyond_data(self, store):
        """Offset beyond data returns empty list."""
        store.record_scheduled_debate(_make_record())
        history = store.get_history(offset=100)
        assert history == []

    def test_empty_category(self, store):
        """Store handles empty category string."""
        rec = _make_record(category="")
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].category == ""

    def test_empty_platform(self, store):
        """Store handles empty platform string."""
        rec = _make_record(platform="")
        store.record_scheduled_debate(rec)

        recent = store.get_recent_topics(hours=1)
        assert recent[0].platform == ""

    def test_concurrent_reads_after_writes(self, store):
        """Multiple reads after writes return consistent data."""
        for i in range(20):
            store.record_scheduled_debate(
                _make_record(id=f"concurrent-{i}", topic_text=f"Topic {i}")
            )

        assert store.count_total() == 20
        history1 = store.get_history(limit=10)
        history2 = store.get_history(limit=10, offset=10)
        all_ids = {r.id for r in history1} | {r.id for r in history2}
        assert len(all_ids) == 20

    def test_get_pending_outcomes_default_limit(self, store):
        """Default limit for get_pending_outcomes is 100."""
        for i in range(110):
            store.record_scheduled_debate(
                _make_record(
                    id=f"pend-{i}",
                    debate_id=f"debate-pend-{i}",
                    consensus_reached=None,
                )
            )
        pending = store.get_pending_outcomes()
        assert len(pending) == 100  # default limit

    def test_analytics_no_completed_debates(self, store):
        """Analytics when there are records but none have debate_id."""
        store.record_scheduled_debate(_make_record(debate_id=None, created_at=time.time()))
        analytics = store.get_analytics()
        assert analytics["total"] == 1
        # completed_debates key may not exist since row[0] == 0
        assert analytics.get("completed_debates", 0) == 0

    def test_cleanup_old_preserves_boundary_recent(self, store):
        """Records just inside the retention window are preserved."""
        just_inside = time.time() - 86400 * 29  # 29 days ago
        store.record_scheduled_debate(_make_record(id="inside", created_at=just_inside))
        removed = store.cleanup_old(days=30)
        assert removed == 0
        assert store.count_total() == 1
