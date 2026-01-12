"""
Unit tests for ScheduledDebateStore.

Tests SQLite persistence for scheduled debate outcomes.
"""

import pytest
import time
from pathlib import Path

from aragora.pulse.store import ScheduledDebateRecord, ScheduledDebateStore


@pytest.fixture
def store(tmp_path):
    """Create a ScheduledDebateStore with temp database."""
    db_path = tmp_path / "test_scheduled_debates.db"
    return ScheduledDebateStore(db_path)


@pytest.fixture
def sample_record(store):
    """Create a sample ScheduledDebateRecord with correctly computed hash."""
    topic_text = "Should AI be regulated?"
    return ScheduledDebateRecord(
        id="sched-123",
        topic_hash=store.hash_topic(topic_text),  # Use actual computed hash
        topic_text=topic_text,
        platform="hackernews",
        category="tech",
        volume=500,
        debate_id="debate-456",
        created_at=time.time(),
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        scheduler_run_id="run-789",
    )


class TestScheduledDebateRecord:
    """Tests for ScheduledDebateRecord dataclass."""

    def test_hours_ago_calculation(self, sample_record):
        """Test hours_ago property."""
        # Record created just now
        assert sample_record.hours_ago < 0.01

        # Record from 2 hours ago
        old_record = ScheduledDebateRecord(
            id="old",
            topic_hash="hash",
            topic_text="topic",
            platform="test",
            category="test",
            volume=0,
            debate_id=None,
            created_at=time.time() - 7200,  # 2 hours ago
            consensus_reached=None,
            confidence=None,
            rounds_used=0,
            scheduler_run_id="",
        )
        assert 1.9 < old_record.hours_ago < 2.1


class TestScheduledDebateStoreBasics:
    """Basic CRUD tests for ScheduledDebateStore."""

    def test_store_initialization(self, store):
        """Test that store initializes correctly."""
        assert store is not None
        assert store.count_total() == 0

    def test_record_scheduled_debate(self, store, sample_record):
        """Test recording a scheduled debate."""
        store.record_scheduled_debate(sample_record)

        assert store.count_total() == 1

    def test_record_updates_existing(self, store, sample_record):
        """Test that recording with same ID updates."""
        store.record_scheduled_debate(sample_record)

        # Update the record
        updated = ScheduledDebateRecord(
            id=sample_record.id,
            topic_hash=sample_record.topic_hash,
            topic_text="Updated topic",
            platform=sample_record.platform,
            category=sample_record.category,
            volume=1000,  # Changed
            debate_id=sample_record.debate_id,
            created_at=sample_record.created_at,
            consensus_reached=sample_record.consensus_reached,
            confidence=sample_record.confidence,
            rounds_used=sample_record.rounds_used,
            scheduler_run_id=sample_record.scheduler_run_id,
        )
        store.record_scheduled_debate(updated)

        # Should still be only 1 record
        assert store.count_total() == 1

        # Should have updated values
        records = store.get_recent_topics(hours=1)
        assert records[0].topic_text == "Updated topic"
        assert records[0].volume == 1000


class TestTopicHashing:
    """Tests for topic hashing functionality."""

    def test_hash_topic_consistent(self, store):
        """Test that hashing is consistent."""
        topic = "Should AI be regulated?"
        hash1 = store.hash_topic(topic)
        hash2 = store.hash_topic(topic)

        assert hash1 == hash2

    def test_hash_topic_normalizes_whitespace(self, store):
        """Test that extra whitespace is normalized."""
        hash1 = store.hash_topic("Should AI be regulated?")
        hash2 = store.hash_topic("  Should   AI   be   regulated?  ")

        assert hash1 == hash2

    def test_hash_topic_case_insensitive(self, store):
        """Test that hashing is case insensitive."""
        hash1 = store.hash_topic("Should AI be regulated?")
        hash2 = store.hash_topic("SHOULD AI BE REGULATED?")

        assert hash1 == hash2

    def test_hash_topic_different_topics(self, store):
        """Test that different topics have different hashes."""
        hash1 = store.hash_topic("Should AI be regulated?")
        hash2 = store.hash_topic("Is machine learning useful?")

        assert hash1 != hash2

    def test_hash_topic_length(self, store):
        """Test that hash is 16 characters."""
        topic_hash = store.hash_topic("Any topic")
        assert len(topic_hash) == 16


class TestDeduplication:
    """Tests for topic deduplication."""

    def test_is_duplicate_returns_false_for_new(self, store):
        """Test that new topics are not duplicates."""
        assert not store.is_duplicate("New topic")

    def test_is_duplicate_returns_true_for_recent(self, store, sample_record):
        """Test that recently debated topics are duplicates."""
        store.record_scheduled_debate(sample_record)

        assert store.is_duplicate(sample_record.topic_text)

    def test_is_duplicate_respects_time_window(self, store):
        """Test that old topics are not duplicates."""
        old_record = ScheduledDebateRecord(
            id="old",
            topic_hash=store.hash_topic("Old topic"),
            topic_text="Old topic",
            platform="test",
            category="test",
            volume=100,
            debate_id="debate",
            created_at=time.time() - 48 * 3600,  # 48 hours ago
            consensus_reached=True,
            confidence=0.8,
            rounds_used=2,
            scheduler_run_id="run",
        )
        store.record_scheduled_debate(old_record)

        # Should not be duplicate within 24 hour window
        assert not store.is_duplicate("Old topic", hours=24)

        # But should be duplicate with 72 hour window
        assert store.is_duplicate("Old topic", hours=72)


class TestGetRecentTopics:
    """Tests for retrieving recent topics."""

    def test_get_recent_topics_empty(self, store):
        """Test getting recent topics when empty."""
        topics = store.get_recent_topics()
        assert topics == []

    def test_get_recent_topics_returns_records(self, store, sample_record):
        """Test getting recent topics."""
        store.record_scheduled_debate(sample_record)

        topics = store.get_recent_topics(hours=1)

        assert len(topics) == 1
        assert topics[0].topic_text == sample_record.topic_text

    def test_get_recent_topics_respects_hours(self, store):
        """Test that hours parameter filters correctly."""
        # Recent record
        recent = ScheduledDebateRecord(
            id="recent",
            topic_hash="recent123",
            topic_text="Recent topic",
            platform="test",
            category="test",
            volume=100,
            debate_id="debate",
            created_at=time.time() - 100,  # 100 seconds ago
            consensus_reached=True,
            confidence=0.8,
            rounds_used=2,
            scheduler_run_id="run",
        )
        store.record_scheduled_debate(recent)

        # Old record
        old = ScheduledDebateRecord(
            id="old",
            topic_hash="old123",
            topic_text="Old topic",
            platform="test",
            category="test",
            volume=100,
            debate_id="debate",
            created_at=time.time() - 48 * 3600,  # 48 hours ago
            consensus_reached=True,
            confidence=0.8,
            rounds_used=2,
            scheduler_run_id="run",
        )
        store.record_scheduled_debate(old)

        # Get topics from last hour
        recent_topics = store.get_recent_topics(hours=1)
        assert len(recent_topics) == 1
        assert recent_topics[0].topic_text == "Recent topic"

        # Get topics from last 72 hours
        all_topics = store.get_recent_topics(hours=72)
        assert len(all_topics) == 2

    def test_get_recent_topics_sorted_by_created_at(self, store):
        """Test that topics are sorted by created_at descending."""
        for i in range(5):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="test",
                category="test",
                volume=100,
                debate_id="debate",
                created_at=time.time() - i * 100,  # Older records have larger i
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        topics = store.get_recent_topics(hours=24)

        # Most recent first
        assert topics[0].topic_text == "Topic 0"
        assert topics[-1].topic_text == "Topic 4"


class TestGetHistory:
    """Tests for retrieving historical records."""

    def test_get_history_with_limit(self, store):
        """Test history with limit."""
        for i in range(10):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="hackernews",
                category="tech",
                volume=100,
                debate_id="debate",
                created_at=time.time() - i * 100,
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        history = store.get_history(limit=5)
        assert len(history) == 5

    def test_get_history_with_offset(self, store):
        """Test history with offset for pagination."""
        for i in range(10):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="hackernews",
                category="tech",
                volume=100,
                debate_id="debate",
                created_at=time.time() - i * 100,
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        page1 = store.get_history(limit=5, offset=0)
        page2 = store.get_history(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].id != page2[0].id

    def test_get_history_with_platform_filter(self, store):
        """Test history filtered by platform."""
        for platform in ["hackernews", "reddit", "hackernews"]:
            record = ScheduledDebateRecord(
                id=f"rec-{platform}-{time.time()}",
                topic_hash=f"hash{time.time()}",
                topic_text=f"Topic on {platform}",
                platform=platform,
                category="tech",
                volume=100,
                debate_id="debate",
                created_at=time.time(),
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        hn_history = store.get_history(platform="hackernews")
        assert len(hn_history) == 2

        reddit_history = store.get_history(platform="reddit")
        assert len(reddit_history) == 1

    def test_get_history_with_category_filter(self, store):
        """Test history filtered by category."""
        for category in ["tech", "ai", "tech"]:
            record = ScheduledDebateRecord(
                id=f"rec-{category}-{time.time()}",
                topic_hash=f"hash{time.time()}",
                topic_text=f"Topic in {category}",
                platform="hackernews",
                category=category,
                volume=100,
                debate_id="debate",
                created_at=time.time(),
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        tech_history = store.get_history(category="tech")
        assert len(tech_history) == 2


class TestAnalytics:
    """Tests for analytics functionality."""

    def test_get_analytics_empty(self, store):
        """Test analytics with no data."""
        analytics = store.get_analytics()

        assert analytics["total"] == 0

    def test_get_analytics_with_data(self, store):
        """Test analytics with data."""
        # Create various records
        for i in range(5):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="hackernews" if i < 3 else "reddit",
                category="tech" if i < 2 else "ai",
                volume=100,
                debate_id="debate",
                created_at=time.time() - i * 100,
                consensus_reached=i % 2 == 0,  # Alternating
                confidence=0.7 + i * 0.05,
                rounds_used=2 + i,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        analytics = store.get_analytics()

        assert analytics["total"] == 5
        assert "by_platform" in analytics
        assert "hackernews" in analytics["by_platform"]
        assert analytics["by_platform"]["hackernews"]["total"] == 3
        assert "by_category" in analytics

    def test_get_analytics_consensus_rate(self, store):
        """Test consensus rate calculation."""
        # 3 with consensus, 1 without
        for i in range(4):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="hackernews",
                category="tech",
                volume=100,
                debate_id="debate",
                created_at=time.time(),
                consensus_reached=i < 3,  # First 3 have consensus
                confidence=0.8,
                rounds_used=3,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        analytics = store.get_analytics()

        assert analytics["consensus_rate"] == 0.75  # 3/4


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_old_removes_old_records(self, store):
        """Test that cleanup removes old records."""
        # Create old record
        old = ScheduledDebateRecord(
            id="old",
            topic_hash="oldhash",
            topic_text="Old topic",
            platform="test",
            category="test",
            volume=100,
            debate_id="debate",
            created_at=time.time() - 40 * 24 * 3600,  # 40 days ago
            consensus_reached=True,
            confidence=0.8,
            rounds_used=2,
            scheduler_run_id="run",
        )
        store.record_scheduled_debate(old)

        # Create recent record
        recent = ScheduledDebateRecord(
            id="recent",
            topic_hash="recenthash",
            topic_text="Recent topic",
            platform="test",
            category="test",
            volume=100,
            debate_id="debate",
            created_at=time.time(),
            consensus_reached=True,
            confidence=0.8,
            rounds_used=2,
            scheduler_run_id="run",
        )
        store.record_scheduled_debate(recent)

        assert store.count_total() == 2

        # Cleanup records older than 30 days
        removed = store.cleanup_old(days=30)

        assert removed == 1
        assert store.count_total() == 1

    def test_cleanup_old_returns_count(self, store):
        """Test that cleanup returns number of removed records."""
        for i in range(5):
            record = ScheduledDebateRecord(
                id=f"rec-{i}",
                topic_hash=f"hash{i}",
                topic_text=f"Topic {i}",
                platform="test",
                category="test",
                volume=100,
                debate_id="debate",
                created_at=time.time() - 100 * 24 * 3600,  # 100 days ago
                consensus_reached=True,
                confidence=0.8,
                rounds_used=2,
                scheduler_run_id="run",
            )
            store.record_scheduled_debate(record)

        removed = store.cleanup_old(days=30)

        assert removed == 5


class TestRowConversion:
    """Tests for database row to record conversion."""

    def test_handles_null_values(self, store):
        """Test that null values in database are handled."""
        record = ScheduledDebateRecord(
            id="nullable",
            topic_hash="hash",
            topic_text="Topic",
            platform="test",
            category="",  # Empty
            volume=0,
            debate_id=None,  # Null
            created_at=time.time(),
            consensus_reached=None,  # Null
            confidence=None,  # Null
            rounds_used=0,
            scheduler_run_id="",
        )
        store.record_scheduled_debate(record)

        records = store.get_recent_topics(hours=1)
        assert len(records) == 1
        assert records[0].debate_id is None
        assert records[0].consensus_reached is None
        assert records[0].confidence is None
