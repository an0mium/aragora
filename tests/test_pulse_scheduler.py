"""
Unit tests for PulseDebateScheduler.

Tests scheduler lifecycle, rate limiting, topic selection, and debate creation.
"""

import asyncio
import pytest
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.pulse.scheduler import (
    PulseDebateScheduler,
    SchedulerConfig,
    SchedulerState,
    SchedulerMetrics,
    TopicSelector,
    TopicScore,
)
from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.store import ScheduledDebateRecord, ScheduledDebateStore


# Test fixtures


@pytest.fixture
def mock_pulse_manager():
    """Create a mock PulseManager."""
    manager = MagicMock()
    manager.get_trending_topics = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_store(tmp_path):
    """Create a real ScheduledDebateStore with temp database."""
    db_path = tmp_path / "test_scheduled_debates.db"
    return ScheduledDebateStore(db_path)


@pytest.fixture
def scheduler_config():
    """Create a test configuration with short intervals."""
    return SchedulerConfig(
        poll_interval_seconds=1,  # Fast for testing
        max_debates_per_hour=3,
        min_interval_between_debates=0,  # No minimum for testing
        min_volume_threshold=50,
        min_controversy_score=0.2,
        dedup_window_hours=24,
        platforms=["test"],
    )


@pytest.fixture
def scheduler(mock_pulse_manager, mock_store, scheduler_config):
    """Create a PulseDebateScheduler instance."""
    return PulseDebateScheduler(
        pulse_manager=mock_pulse_manager,
        store=mock_store,
        config=scheduler_config,
    )


def make_topic(
    topic: str = "Test topic",
    platform: str = "hackernews",
    category: str = "tech",
    volume: int = 100,
) -> TrendingTopic:
    """Helper to create a TrendingTopic."""
    return TrendingTopic(
        topic=topic,
        platform=platform,
        category=category,
        volume=volume,
    )


# TopicSelector Tests


class TestTopicSelector:
    """Tests for TopicSelector scoring logic."""

    def test_score_topic_allowed_category(self, scheduler_config):
        """Test that allowed categories get positive scores."""
        selector = TopicSelector(scheduler_config)
        topic = make_topic(category="tech", volume=200)

        score = selector.score_topic(topic)

        assert score.is_viable
        assert score.score > 0
        assert "category 'tech' is allowed" in score.reasons

    def test_score_topic_blocked_category(self, scheduler_config):
        """Test that blocked categories are rejected."""
        selector = TopicSelector(scheduler_config)
        topic = make_topic(category="politics", volume=200)

        score = selector.score_topic(topic)

        assert not score.is_viable
        assert score.score < 0
        assert "blocked" in score.reasons[0]

    def test_score_topic_below_volume_threshold(self, scheduler_config):
        """Test that low volume topics are rejected."""
        selector = TopicSelector(scheduler_config)
        topic = make_topic(volume=10)  # Below threshold of 50

        score = selector.score_topic(topic)

        assert not score.is_viable
        assert "below threshold" in score.reasons[0]

    def test_score_topic_controversy_keywords(self, scheduler_config):
        """Test that controversy keywords boost score."""
        selector = TopicSelector(scheduler_config)

        # Topic with controversy keywords
        controversial = make_topic(
            topic="Should AI vs humans debate the future?",
            volume=200,
        )
        normal = make_topic(topic="New software released", volume=200)

        controversial_score = selector.score_topic(controversial)
        normal_score = selector.score_topic(normal)

        assert controversial_score.score > normal_score.score

    def test_score_topic_boost_keywords(self, scheduler_config):
        """Test that boost keywords increase score."""
        selector = TopicSelector(scheduler_config)

        boosted = make_topic(
            topic="AI breakthrough in machine learning ethics",
            volume=200,
        )
        normal = make_topic(topic="Software update available", volume=200)

        boosted_score = selector.score_topic(boosted)
        normal_score = selector.score_topic(normal)

        assert boosted_score.score > normal_score.score
        assert "keyword boost" in str(boosted_score.reasons)

    def test_select_best_topics_returns_sorted(self, scheduler_config):
        """Test that select_best_topics returns topics sorted by score."""
        selector = TopicSelector(scheduler_config)

        topics = [
            make_topic(topic="Low interest topic", volume=60),
            make_topic(topic="AI vs humans debate controversy", volume=500),
            make_topic(topic="New tech breakthrough", volume=200),
        ]

        best = selector.select_best_topics(topics, limit=3)

        assert len(best) > 0
        # Should be sorted descending by score
        scores = [b.score for b in best]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_topics_filters_nonviable(self, scheduler_config):
        """Test that select_best_topics filters out non-viable topics."""
        selector = TopicSelector(scheduler_config)

        topics = [
            make_topic(category="politics", volume=500),  # Blocked
            make_topic(volume=5),  # Below threshold
            make_topic(topic="Good tech topic", volume=200),  # Viable
        ]

        best = selector.select_best_topics(topics, limit=5)

        # Only the viable topic should remain
        assert len(best) == 1
        assert "Good tech topic" in best[0].topic.topic

    def test_select_best_topics_respects_limit(self, scheduler_config):
        """Test that select_best_topics respects the limit parameter."""
        selector = TopicSelector(scheduler_config)

        topics = [make_topic(topic=f"Topic {i}", volume=100 + i * 10) for i in range(10)]

        best = selector.select_best_topics(topics, limit=3)

        assert len(best) <= 3


# SchedulerConfig Tests


class TestSchedulerConfig:
    """Tests for SchedulerConfig serialization."""

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = SchedulerConfig(
            poll_interval_seconds=60,
            max_debates_per_hour=5,
        )

        data = config.to_dict()

        assert data["poll_interval_seconds"] == 60
        assert data["max_debates_per_hour"] == 5
        assert "platforms" in data
        assert "allowed_categories" in data

    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        data = {
            "poll_interval_seconds": 120,
            "max_debates_per_hour": 10,
            "platforms": ["twitter"],
        }

        config = SchedulerConfig.from_dict(data)

        assert config.poll_interval_seconds == 120
        assert config.max_debates_per_hour == 10
        assert config.platforms == ["twitter"]

    def test_config_from_dict_defaults(self):
        """Test that missing fields use defaults."""
        config = SchedulerConfig.from_dict({})

        assert config.poll_interval_seconds == 300
        assert config.max_debates_per_hour == 6
        assert "hackernews" in config.platforms


# SchedulerMetrics Tests


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics tracking."""

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = SchedulerMetrics(
            polls_completed=5,
            debates_created=2,
            start_time=time.time() - 100,
        )

        data = metrics.to_dict()

        assert data["polls_completed"] == 5
        assert data["debates_created"] == 2
        assert data["uptime_seconds"] is not None
        assert data["uptime_seconds"] >= 100

    def test_metrics_uptime_none_when_not_started(self):
        """Test that uptime is None when scheduler hasn't started."""
        metrics = SchedulerMetrics()

        data = metrics.to_dict()

        assert data["uptime_seconds"] is None


# PulseDebateScheduler State Machine Tests


class TestSchedulerStateMachine:
    """Tests for scheduler state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_stopped(self, scheduler):
        """Test that scheduler starts in STOPPED state."""
        assert scheduler.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self, scheduler):
        """Test that start() transitions to RUNNING."""
        # Set up debate creator
        scheduler.set_debate_creator(AsyncMock(return_value={"debate_id": "test"}))

        await scheduler.start()
        try:
            assert scheduler.state == SchedulerState.RUNNING
        finally:
            await scheduler.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_start_without_creator_raises(self, scheduler):
        """Test that start() without debate creator raises."""
        with pytest.raises(RuntimeError, match="No debate creator"):
            await scheduler.start()

    @pytest.mark.asyncio
    async def test_stop_transitions_to_stopped(self, scheduler):
        """Test that stop() transitions to STOPPED."""
        scheduler.set_debate_creator(AsyncMock(return_value={"debate_id": "test"}))
        await scheduler.start()

        await scheduler.stop()

        assert scheduler.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_transitions_to_paused(self, scheduler):
        """Test that pause() transitions to PAUSED."""
        scheduler.set_debate_creator(AsyncMock(return_value={"debate_id": "test"}))
        await scheduler.start()

        try:
            await scheduler.pause()
            assert scheduler.state == SchedulerState.PAUSED
        finally:
            await scheduler.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_resume_transitions_to_running(self, scheduler):
        """Test that resume() transitions back to RUNNING."""
        scheduler.set_debate_creator(AsyncMock(return_value={"debate_id": "test"}))
        await scheduler.start()
        await scheduler.pause()

        try:
            await scheduler.resume()
            assert scheduler.state == SchedulerState.RUNNING
        finally:
            await scheduler.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_pause_when_not_running_is_noop(self, scheduler):
        """Test that pause() when not running does nothing."""
        await scheduler.pause()
        assert scheduler.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_resume_when_not_paused_is_noop(self, scheduler):
        """Test that resume() when not paused does nothing."""
        await scheduler.resume()
        assert scheduler.state == SchedulerState.STOPPED


# Rate Limiting Tests


class TestSchedulerRateLimiting:
    """Tests for scheduler rate limiting."""

    def test_can_create_debate_within_limit(self, scheduler):
        """Test that debates can be created within hourly limit."""
        assert scheduler._can_create_debate()

    def test_can_create_debate_at_limit(self, scheduler):
        """Test that debates are blocked at hourly limit."""
        # Fill up the hour's quota
        now = time.time()
        scheduler._debates_this_hour = [now, now, now]  # 3 debates (at limit)

        assert not scheduler._can_create_debate()

    def test_can_create_debate_clears_old_entries(self, scheduler):
        """Test that old debate times are cleared."""
        old_time = time.time() - 4000  # More than an hour ago
        scheduler._debates_this_hour = [old_time, old_time, old_time]

        assert scheduler._can_create_debate()
        assert len(scheduler._debates_this_hour) == 0

    def test_can_create_debate_respects_interval(self, scheduler):
        """Test minimum interval between debates."""
        scheduler.config.min_interval_between_debates = 600  # 10 minutes
        scheduler._debates_this_hour = [time.time()]  # Just created one

        assert not scheduler._can_create_debate()


# Config Update Tests


class TestSchedulerConfigUpdate:
    """Tests for runtime config updates."""

    def test_update_config_changes_values(self, scheduler):
        """Test that update_config modifies configuration."""
        scheduler.update_config({"max_debates_per_hour": 10})

        assert scheduler.config.max_debates_per_hour == 10

    def test_update_config_preserves_other_values(self, scheduler):
        """Test that update_config preserves unchanged values."""
        original_interval = scheduler.config.poll_interval_seconds
        scheduler.update_config({"max_debates_per_hour": 10})

        assert scheduler.config.poll_interval_seconds == original_interval

    def test_update_config_recreates_topic_selector(self, scheduler):
        """Test that topic selector is recreated on config update."""
        original_selector = scheduler._topic_selector
        scheduler.update_config({"min_controversy_score": 0.5})

        assert scheduler._topic_selector is not original_selector


# Status Tests


class TestSchedulerStatus:
    """Tests for scheduler status reporting."""

    def test_get_status_includes_state(self, scheduler):
        """Test that get_status includes current state."""
        status = scheduler.get_status()

        assert "state" in status
        assert status["state"] == "stopped"

    def test_get_status_includes_config(self, scheduler):
        """Test that get_status includes config."""
        status = scheduler.get_status()

        assert "config" in status
        assert status["config"]["poll_interval_seconds"] == 1

    def test_get_status_includes_metrics(self, scheduler):
        """Test that get_status includes metrics."""
        status = scheduler.get_status()

        assert "metrics" in status
        assert "polls_completed" in status["metrics"]


# Debate Creation Tests


class TestSchedulerDebateCreation:
    """Tests for debate creation flow."""

    @pytest.mark.asyncio
    async def test_poll_creates_debate_for_good_topic(
        self, scheduler, mock_pulse_manager
    ):
        """Test that polling creates debates for suitable topics."""
        # Set up trending topics
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(topic="AI debate topic should we discuss", volume=200),
        ]

        # Set up debate creator
        creator = AsyncMock(
            return_value={
                "debate_id": "test-123",
                "consensus_reached": True,
                "confidence": 0.85,
                "rounds_used": 3,
            }
        )
        scheduler.set_debate_creator(creator)

        # Run a single poll
        await scheduler._poll_and_create()

        # Verify debate was created
        assert scheduler.metrics.debates_created == 1
        creator.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_skips_duplicate_topics(self, scheduler, mock_pulse_manager, mock_store):
        """Test that duplicate topics are skipped."""
        topic_text = "AI debate topic"

        # Record a previous debate with this topic
        record = ScheduledDebateRecord(
            id="old-123",
            topic_hash=mock_store.hash_topic(topic_text),
            topic_text=topic_text,
            platform="hackernews",
            category="tech",
            volume=100,
            debate_id="old-debate",
            created_at=time.time() - 100,  # Recent
            consensus_reached=True,
            confidence=0.8,
            rounds_used=3,
            scheduler_run_id="old-run",
        )
        mock_store.record_scheduled_debate(record)

        # Set up trending topics with the same topic
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(topic=topic_text, volume=200),
        ]

        creator = AsyncMock(return_value={"debate_id": "new"})
        scheduler.set_debate_creator(creator)

        await scheduler._poll_and_create()

        # Debate should not be created
        assert scheduler.metrics.duplicates_skipped == 1
        creator.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_handles_no_topics(self, scheduler, mock_pulse_manager):
        """Test that polling handles empty topic list."""
        mock_pulse_manager.get_trending_topics.return_value = []

        creator = AsyncMock()
        scheduler.set_debate_creator(creator)

        await scheduler._poll_and_create()

        assert scheduler.metrics.polls_completed == 1
        creator.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_handles_creator_failure(self, scheduler, mock_pulse_manager):
        """Test that debate creation failures are recorded."""
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(topic="Good topic for debate", volume=200),
        ]

        creator = AsyncMock(side_effect=Exception("Creator failed"))
        scheduler.set_debate_creator(creator)

        await scheduler._poll_and_create()

        assert scheduler.metrics.debates_failed == 1

    @pytest.mark.asyncio
    async def test_poll_handles_creator_returning_none(
        self, scheduler, mock_pulse_manager
    ):
        """Test handling when creator returns None."""
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(topic="Good topic for debate", volume=200),
        ]

        creator = AsyncMock(return_value=None)
        scheduler.set_debate_creator(creator)

        await scheduler._poll_and_create()

        assert scheduler.metrics.debates_failed == 1


# Integration Tests


class TestSchedulerIntegration:
    """Integration tests for scheduler lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, scheduler, mock_pulse_manager):
        """Test complete scheduler lifecycle: start -> poll -> stop."""
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(topic="AI ethics debate should we", volume=200),
        ]

        creator = AsyncMock(
            return_value={
                "debate_id": "test-123",
                "consensus_reached": True,
                "confidence": 0.9,
                "rounds_used": 2,
            }
        )
        scheduler.set_debate_creator(creator)

        # Start scheduler
        await scheduler.start()
        assert scheduler.state == SchedulerState.RUNNING

        # Wait for at least one poll
        await asyncio.sleep(1.5)

        # Stop scheduler
        await scheduler.stop()
        assert scheduler.state == SchedulerState.STOPPED

        # Verify polls occurred
        assert scheduler.metrics.polls_completed >= 1

    @pytest.mark.asyncio
    async def test_graceful_stop_timeout(self, scheduler, mock_pulse_manager):
        """Test graceful stop with timeout."""
        # Create a slow debate creator
        async def slow_creator(*args):
            await asyncio.sleep(10)
            return {"debate_id": "slow"}

        scheduler.set_debate_creator(slow_creator)
        mock_pulse_manager.get_trending_topics.return_value = [
            make_topic(volume=200),
        ]

        await scheduler.start()
        await asyncio.sleep(0.1)

        # Graceful stop should timeout and cancel
        start = time.time()
        await scheduler.stop(graceful=True)
        elapsed = time.time() - start

        # Should complete in reasonable time (timeout is 30s but we expect faster cancel)
        assert elapsed < 35
        assert scheduler.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_forced_stop(self, scheduler, mock_pulse_manager):
        """Test forced (non-graceful) stop."""
        scheduler.set_debate_creator(AsyncMock())

        await scheduler.start()

        # Force stop
        await scheduler.stop(graceful=False)

        assert scheduler.state == SchedulerState.STOPPED
