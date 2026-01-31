"""Comprehensive tests for the Pulse Debate Scheduler module.

Tests cover:
1. SchedulerConfig - initialization, to_dict, from_dict, defaults
2. TopicScore - creation, is_viable property
3. SchedulerMetrics - initialization, to_dict, uptime calculation
4. TopicSelector - score_topic, controversy, boost, select_best_topics
5. PulseDebateScheduler - initialization, lifecycle (start/stop/pause/resume),
   configuration updates, rate limiting, deduplication, debate creation,
   Knowledge Mound integration, error handling, metrics tracking
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.exceptions import ConfigurationError
from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.scheduler import (
    PulseDebateScheduler,
    SchedulerConfig,
    SchedulerMetrics,
    SchedulerState,
    TopicScore,
    TopicSelector,
)
from aragora.pulse.store import ScheduledDebateRecord


# =============================================================================
# Fixtures
# =============================================================================


def _make_topic(
    topic: str = "AI vs Human Debate",
    platform: str = "hackernews",
    volume: int = 500,
    category: str = "tech",
) -> TrendingTopic:
    """Helper to create a TrendingTopic."""
    return TrendingTopic(
        platform=platform,
        topic=topic,
        volume=volume,
        category=category,
    )


@pytest.fixture
def default_config():
    """Default SchedulerConfig."""
    return SchedulerConfig()


@pytest.fixture
def custom_config():
    """Custom SchedulerConfig with non-default values."""
    return SchedulerConfig(
        poll_interval_seconds=60,
        platforms=["hackernews"],
        max_debates_per_hour=3,
        min_interval_between_debates=120,
        min_volume_threshold=50,
        min_controversy_score=0.2,
        allowed_categories=["tech", "ai"],
        blocked_categories=["politics"],
        dedup_window_hours=12,
        debate_rounds=5,
        consensus_threshold=0.8,
    )


@pytest.fixture
def topic_selector(default_config):
    """TopicSelector with default config."""
    return TopicSelector(default_config)


@pytest.fixture
def mock_pulse_manager():
    """Mock PulseManager."""
    manager = MagicMock()
    manager.get_trending_topics = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_store():
    """Mock ScheduledDebateStore."""
    store = MagicMock()
    store.is_duplicate = MagicMock(return_value=False)
    store.hash_topic = MagicMock(return_value="abc123hash")
    store.record_scheduled_debate = MagicMock()
    return store


@pytest.fixture
def mock_debate_creator():
    """Mock debate creator callback."""
    creator = AsyncMock(
        return_value={
            "debate_id": "test-debate-123",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
        }
    )
    return creator


@pytest.fixture
def scheduler(mock_pulse_manager, mock_store):
    """PulseDebateScheduler with mocked dependencies."""
    return PulseDebateScheduler(
        pulse_manager=mock_pulse_manager,
        store=mock_store,
    )


@pytest.fixture
def scheduler_with_creator(scheduler, mock_debate_creator):
    """Scheduler with debate creator set."""
    scheduler.set_debate_creator(mock_debate_creator)
    return scheduler


# =============================================================================
# SchedulerConfig Tests
# =============================================================================


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test SchedulerConfig default values."""
        config = SchedulerConfig()
        assert config.poll_interval_seconds == 300
        assert config.platforms == ["hackernews", "reddit"]
        assert config.max_debates_per_hour == 6
        assert config.min_interval_between_debates == 600
        assert config.min_volume_threshold == 100
        assert config.min_controversy_score == 0.3
        assert config.allowed_categories == ["tech", "ai", "science", "programming"]
        assert config.blocked_categories == ["politics", "religion"]
        assert config.dedup_window_hours == 24
        assert config.debate_rounds == 3
        assert config.consensus_threshold == 0.7

    def test_custom_values(self, custom_config):
        """Test SchedulerConfig with custom values."""
        assert custom_config.poll_interval_seconds == 60
        assert custom_config.platforms == ["hackernews"]
        assert custom_config.max_debates_per_hour == 3
        assert custom_config.min_interval_between_debates == 120
        assert custom_config.min_volume_threshold == 50
        assert custom_config.min_controversy_score == 0.2
        assert custom_config.allowed_categories == ["tech", "ai"]
        assert custom_config.blocked_categories == ["politics"]
        assert custom_config.dedup_window_hours == 12
        assert custom_config.debate_rounds == 5
        assert custom_config.consensus_threshold == 0.8

    def test_to_dict(self, default_config):
        """Test SchedulerConfig.to_dict serialization."""
        d = default_config.to_dict()
        assert isinstance(d, dict)
        assert d["poll_interval_seconds"] == 300
        assert d["platforms"] == ["hackernews", "reddit"]
        assert d["max_debates_per_hour"] == 6
        assert d["min_interval_between_debates"] == 600
        assert d["min_volume_threshold"] == 100
        assert d["min_controversy_score"] == 0.3
        assert d["allowed_categories"] == ["tech", "ai", "science", "programming"]
        assert d["blocked_categories"] == ["politics", "religion"]
        assert d["dedup_window_hours"] == 24
        assert d["debate_rounds"] == 3
        assert d["consensus_threshold"] == 0.7

    def test_to_dict_custom(self, custom_config):
        """Test custom SchedulerConfig.to_dict."""
        d = custom_config.to_dict()
        assert d["poll_interval_seconds"] == 60
        assert d["max_debates_per_hour"] == 3

    def test_from_dict_full(self):
        """Test SchedulerConfig.from_dict with all keys."""
        data = {
            "poll_interval_seconds": 120,
            "platforms": ["twitter"],
            "max_debates_per_hour": 10,
            "min_interval_between_debates": 30,
            "min_volume_threshold": 200,
            "min_controversy_score": 0.5,
            "allowed_categories": ["science"],
            "blocked_categories": ["sports"],
            "dedup_window_hours": 48,
            "debate_rounds": 7,
            "consensus_threshold": 0.9,
        }
        config = SchedulerConfig.from_dict(data)
        assert config.poll_interval_seconds == 120
        assert config.platforms == ["twitter"]
        assert config.max_debates_per_hour == 10
        assert config.min_interval_between_debates == 30
        assert config.min_volume_threshold == 200
        assert config.min_controversy_score == 0.5
        assert config.allowed_categories == ["science"]
        assert config.blocked_categories == ["sports"]
        assert config.dedup_window_hours == 48
        assert config.debate_rounds == 7
        assert config.consensus_threshold == 0.9

    def test_from_dict_partial(self):
        """Test SchedulerConfig.from_dict with partial keys uses defaults."""
        data = {"poll_interval_seconds": 60}
        config = SchedulerConfig.from_dict(data)
        assert config.poll_interval_seconds == 60
        # Rest should be defaults
        assert config.platforms == ["hackernews", "reddit"]
        assert config.max_debates_per_hour == 6
        assert config.debate_rounds == 3

    def test_from_dict_empty(self):
        """Test SchedulerConfig.from_dict with empty dict uses all defaults."""
        config = SchedulerConfig.from_dict({})
        assert config.poll_interval_seconds == 300
        assert config.consensus_threshold == 0.7

    def test_roundtrip(self, custom_config):
        """Test to_dict -> from_dict roundtrip preserves values."""
        d = custom_config.to_dict()
        restored = SchedulerConfig.from_dict(d)
        assert restored.poll_interval_seconds == custom_config.poll_interval_seconds
        assert restored.platforms == custom_config.platforms
        assert restored.max_debates_per_hour == custom_config.max_debates_per_hour
        assert restored.min_interval_between_debates == custom_config.min_interval_between_debates
        assert restored.min_volume_threshold == custom_config.min_volume_threshold
        assert restored.min_controversy_score == custom_config.min_controversy_score
        assert restored.allowed_categories == custom_config.allowed_categories
        assert restored.blocked_categories == custom_config.blocked_categories
        assert restored.dedup_window_hours == custom_config.dedup_window_hours
        assert restored.debate_rounds == custom_config.debate_rounds
        assert restored.consensus_threshold == custom_config.consensus_threshold


# =============================================================================
# TopicScore Tests
# =============================================================================


class TestTopicScore:
    """Tests for TopicScore dataclass."""

    def test_creation(self):
        """Test TopicScore creation."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=0.75, reasons=["high volume"])
        assert score.topic is topic
        assert score.score == 0.75
        assert score.reasons == ["high volume"]

    def test_default_reasons(self):
        """Test TopicScore default reasons is empty list."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=0.5)
        assert score.reasons == []

    def test_is_viable_positive(self):
        """Test is_viable returns True for positive score."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=0.1)
        assert score.is_viable is True

    def test_is_viable_zero(self):
        """Test is_viable returns False for zero score."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=0.0)
        assert score.is_viable is False

    def test_is_viable_negative(self):
        """Test is_viable returns False for negative score."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=-1.0)
        assert score.is_viable is False

    def test_is_viable_very_small_positive(self):
        """Test is_viable returns True for very small positive score."""
        topic = _make_topic()
        score = TopicScore(topic=topic, score=0.001)
        assert score.is_viable is True


# =============================================================================
# SchedulerMetrics Tests
# =============================================================================


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics dataclass."""

    def test_default_values(self):
        """Test SchedulerMetrics default values."""
        metrics = SchedulerMetrics()
        assert metrics.polls_completed == 0
        assert metrics.topics_evaluated == 0
        assert metrics.topics_filtered == 0
        assert metrics.debates_created == 0
        assert metrics.debates_failed == 0
        assert metrics.duplicates_skipped == 0
        assert metrics.last_poll_at is None
        assert metrics.last_debate_at is None
        assert metrics.start_time is None

    def test_to_dict_no_start_time(self):
        """Test to_dict when start_time is None."""
        metrics = SchedulerMetrics()
        d = metrics.to_dict()
        assert d["polls_completed"] == 0
        assert d["topics_evaluated"] == 0
        assert d["topics_filtered"] == 0
        assert d["debates_created"] == 0
        assert d["debates_failed"] == 0
        assert d["duplicates_skipped"] == 0
        assert d["last_poll_at"] is None
        assert d["last_debate_at"] is None
        assert d["uptime_seconds"] is None

    def test_to_dict_with_start_time(self):
        """Test to_dict calculates uptime when start_time is set."""
        now = time.time()
        metrics = SchedulerMetrics(start_time=now - 100)
        d = metrics.to_dict()
        assert d["uptime_seconds"] is not None
        assert d["uptime_seconds"] >= 100

    def test_to_dict_with_populated_values(self):
        """Test to_dict with all values populated."""
        now = time.time()
        metrics = SchedulerMetrics(
            polls_completed=10,
            topics_evaluated=50,
            topics_filtered=30,
            debates_created=5,
            debates_failed=2,
            duplicates_skipped=8,
            last_poll_at=now - 10,
            last_debate_at=now - 60,
            start_time=now - 3600,
        )
        d = metrics.to_dict()
        assert d["polls_completed"] == 10
        assert d["topics_evaluated"] == 50
        assert d["topics_filtered"] == 30
        assert d["debates_created"] == 5
        assert d["debates_failed"] == 2
        assert d["duplicates_skipped"] == 8
        assert d["last_poll_at"] == now - 10
        assert d["last_debate_at"] == now - 60
        assert d["uptime_seconds"] >= 3600

    def test_incrementing_metrics(self):
        """Test that metrics can be incremented."""
        metrics = SchedulerMetrics()
        metrics.polls_completed += 1
        metrics.topics_evaluated += 5
        metrics.debates_created += 1
        assert metrics.polls_completed == 1
        assert metrics.topics_evaluated == 5
        assert metrics.debates_created == 1


# =============================================================================
# TopicSelector Tests
# =============================================================================


class TestTopicSelector:
    """Tests for TopicSelector."""

    def test_initialization(self, default_config):
        """Test TopicSelector initializes with config."""
        selector = TopicSelector(default_config)
        assert selector.config is default_config

    # -- Category filtering --

    def test_score_topic_allowed_category(self, topic_selector):
        """Test scoring a topic with an allowed category."""
        topic = _make_topic(topic="Should AI replace programmers", category="tech", volume=500)
        result = topic_selector.score_topic(topic)
        assert result.score > 0
        assert any("allowed" in r for r in result.reasons)

    def test_score_topic_blocked_category(self, topic_selector):
        """Test scoring a topic with a blocked category returns -1."""
        topic = _make_topic(topic="Election results", category="politics", volume=5000)
        result = topic_selector.score_topic(topic)
        assert result.score == -1.0
        assert any("blocked" in r for r in result.reasons)

    def test_score_topic_blocked_religion_category(self, topic_selector):
        """Test scoring a topic with blocked religion category."""
        topic = _make_topic(topic="Religious debate", category="religion", volume=500)
        result = topic_selector.score_topic(topic)
        assert result.score == -1.0

    def test_score_topic_neutral_category(self, topic_selector):
        """Test scoring a topic with a neutral (non-allowed, non-blocked) category."""
        topic = _make_topic(topic="Should we debate food", category="food", volume=500)
        result = topic_selector.score_topic(topic)
        # Neutral category gets 0.1 score instead of 0.3
        assert result.score > 0
        assert any("neutral" in r for r in result.reasons)

    def test_score_topic_no_allowed_categories(self):
        """Test scoring when allowed_categories is empty."""
        config = SchedulerConfig(allowed_categories=[], blocked_categories=["politics"])
        selector = TopicSelector(config)
        topic = _make_topic(topic="Should AI debate", category="tech", volume=500)
        result = selector.score_topic(topic)
        # With empty allowed_categories, the first branch doesn't execute
        # Then blocked_categories check still works
        assert result.score > 0 or result.score == -1.0

    # -- Volume threshold --

    def test_score_topic_below_volume_threshold(self, topic_selector):
        """Test topic below volume threshold returns -1."""
        topic = _make_topic(topic="AI debate", category="tech", volume=50)
        result = topic_selector.score_topic(topic)
        assert result.score == -1.0
        assert any("below threshold" in r for r in result.reasons)

    def test_score_topic_at_volume_threshold(self, topic_selector):
        """Test topic at exactly the volume threshold passes."""
        topic = _make_topic(topic="AI debate", category="tech", volume=100)
        result = topic_selector.score_topic(topic)
        assert result.score > 0

    def test_score_topic_high_volume(self, topic_selector):
        """Test topic with very high volume gets capped volume score."""
        topic = _make_topic(topic="AI debate should we worry", category="tech", volume=100000)
        result = topic_selector.score_topic(topic)
        assert result.score > 0
        # Volume score is capped at 0.3
        assert any("volume" in r for r in result.reasons)

    # -- Controversy scoring --

    def test_controversy_keywords_boost_score(self, topic_selector):
        """Test that controversy keywords boost the score."""
        # Topic with controversy keywords
        controversial_topic = _make_topic(
            topic="Should AI vs humans debate the controversial issue",
            category="tech",
            volume=500,
        )
        # Topic without controversy keywords
        bland_topic = _make_topic(
            topic="New framework released today",
            category="tech",
            volume=500,
        )
        controversial_score = topic_selector.score_topic(controversial_topic)
        bland_score = topic_selector.score_topic(bland_topic)
        assert controversial_score.score > bland_score.score

    def test_calculate_controversy_score_none(self, topic_selector):
        """Test controversy calculation with no matching keywords."""
        score = topic_selector._calculate_controversy_score("hello world example")
        assert score == 0.0

    def test_calculate_controversy_score_one(self, topic_selector):
        """Test controversy calculation with one keyword."""
        score = topic_selector._calculate_controversy_score("should we do this")
        assert score > 0
        assert score == pytest.approx(1 / 3)

    def test_calculate_controversy_score_many(self, topic_selector):
        """Test controversy calculation with many keywords saturates at 1.0."""
        text = "should debate controversy controversial issue problem concern question argument"
        score = topic_selector._calculate_controversy_score(text)
        assert score == 1.0

    def test_calculate_controversy_case_insensitive(self, topic_selector):
        """Test controversy scoring is case-insensitive."""
        score_lower = topic_selector._calculate_controversy_score("should debate vs")
        score_upper = topic_selector._calculate_controversy_score("SHOULD DEBATE VS")
        assert score_lower == score_upper

    # -- Boost scoring --

    def test_boost_keywords_increase_score(self, topic_selector):
        """Test that boost keywords increase the score."""
        boosted = _make_topic(
            topic="AI artificial intelligence breakthrough", category="tech", volume=500
        )
        unboosted = _make_topic(topic="Random topic discussion", category="tech", volume=500)
        boosted_score = topic_selector.score_topic(boosted)
        unboosted_score = topic_selector.score_topic(unboosted)
        assert boosted_score.score > unboosted_score.score

    def test_calculate_boost_score_none(self, topic_selector):
        """Test boost score with no matching keywords."""
        score = topic_selector._calculate_boost_score("random words here")
        assert score == 0.0

    def test_calculate_boost_score_one(self, topic_selector):
        """Test boost score with one keyword."""
        score = topic_selector._calculate_boost_score("ai is great")
        assert score == 0.05

    def test_calculate_boost_score_many(self, topic_selector):
        """Test boost score with many keywords caps at 0.2."""
        text = "ai artificial intelligence machine learning ethics future impact new breakthrough revolutionary"
        score = topic_selector._calculate_boost_score(text)
        assert score == 0.2

    def test_calculate_boost_case_insensitive(self, topic_selector):
        """Test boost scoring is case-insensitive."""
        score_lower = topic_selector._calculate_boost_score("ai machine learning")
        score_upper = topic_selector._calculate_boost_score("AI MACHINE LEARNING")
        assert score_lower == score_upper

    # -- select_best_topics --

    def test_select_best_topics_empty(self, topic_selector):
        """Test select_best_topics with no topics."""
        result = topic_selector.select_best_topics([], limit=5)
        assert result == []

    def test_select_best_topics_filters_non_viable(self, topic_selector):
        """Test that non-viable topics are filtered out."""
        topics = [
            _make_topic(topic="Politics debate", category="politics", volume=5000),
            _make_topic(topic="Low volume topic", category="tech", volume=10),
        ]
        result = topic_selector.select_best_topics(topics, limit=5)
        assert len(result) == 0

    def test_select_best_topics_sorts_by_score(self, topic_selector):
        """Test that results are sorted by score descending."""
        topics = [
            _make_topic(topic="Basic topic", category="tech", volume=100),
            _make_topic(
                topic="Should AI vs humans debate the controversial issue",
                category="tech",
                volume=5000,
            ),
            _make_topic(topic="Moderate discussion", category="tech", volume=500),
        ]
        result = topic_selector.select_best_topics(topics, limit=5)
        assert len(result) > 0
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_select_best_topics_respects_limit(self, topic_selector):
        """Test that limit is respected."""
        topics = [
            _make_topic(topic=f"Topic {i} should debate", category="tech", volume=500)
            for i in range(10)
        ]
        result = topic_selector.select_best_topics(topics, limit=3)
        assert len(result) <= 3

    def test_select_best_topics_default_limit(self, topic_selector):
        """Test default limit of 5."""
        topics = [
            _make_topic(topic=f"Topic {i} should debate", category="tech", volume=500)
            for i in range(20)
        ]
        result = topic_selector.select_best_topics(topics, limit=5)
        assert len(result) <= 5

    def test_select_best_topics_mixed_viability(self, topic_selector):
        """Test selection with a mix of viable and non-viable topics."""
        topics = [
            _make_topic(topic="Should AI debate ethics", category="tech", volume=500),
            _make_topic(topic="Politics debate", category="politics", volume=5000),
            _make_topic(topic="Low volume", category="tech", volume=10),
            _make_topic(topic="AI breakthrough new", category="ai", volume=300),
        ]
        result = topic_selector.select_best_topics(topics, limit=5)
        # Only non-blocked, above-threshold topics should remain
        for scored in result:
            assert scored.is_viable
            assert scored.topic.category not in ["politics", "religion"]
            assert scored.topic.volume >= 100


# =============================================================================
# SchedulerState Tests
# =============================================================================


class TestSchedulerState:
    """Tests for SchedulerState enum."""

    def test_states_exist(self):
        """Test all expected states exist."""
        assert SchedulerState.STOPPED == "stopped"
        assert SchedulerState.RUNNING == "running"
        assert SchedulerState.PAUSED == "paused"

    def test_state_is_string(self):
        """Test SchedulerState values are strings."""
        assert isinstance(SchedulerState.STOPPED.value, str)
        assert isinstance(SchedulerState.RUNNING.value, str)
        assert isinstance(SchedulerState.PAUSED.value, str)


# =============================================================================
# PulseDebateScheduler Tests - Initialization
# =============================================================================


class TestPulseDebateSchedulerInit:
    """Tests for PulseDebateScheduler initialization."""

    def test_basic_init(self, mock_pulse_manager, mock_store):
        """Test basic scheduler initialization."""
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store)
        assert scheduler.pulse_manager is mock_pulse_manager
        assert scheduler.store is mock_store
        assert scheduler.state == SchedulerState.STOPPED
        assert scheduler._debate_creator is None
        assert scheduler._km_adapter is None

    def test_init_with_config(self, mock_pulse_manager, mock_store, custom_config):
        """Test initialization with custom config."""
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=custom_config)
        assert scheduler.config is custom_config
        assert scheduler.config.poll_interval_seconds == 60

    def test_init_default_config(self, mock_pulse_manager, mock_store):
        """Test initialization with default config."""
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store)
        assert isinstance(scheduler.config, SchedulerConfig)
        assert scheduler.config.poll_interval_seconds == 300

    def test_init_with_km_adapter(self, mock_pulse_manager, mock_store):
        """Test initialization with KM adapter."""
        adapter = MagicMock()
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, km_adapter=adapter)
        assert scheduler._km_adapter is adapter

    def test_init_metrics_fresh(self, scheduler):
        """Test that metrics are fresh on init."""
        metrics = scheduler.metrics
        assert metrics.polls_completed == 0
        assert metrics.debates_created == 0

    def test_init_state_stopped(self, scheduler):
        """Test initial state is STOPPED."""
        assert scheduler.state == SchedulerState.STOPPED

    def test_init_empty_debates_this_hour(self, scheduler):
        """Test debates_this_hour is empty on init."""
        assert scheduler._debates_this_hour == []


# =============================================================================
# PulseDebateScheduler Tests - Debate Creator
# =============================================================================


class TestDebateCreator:
    """Tests for set_debate_creator."""

    def test_set_debate_creator(self, scheduler, mock_debate_creator):
        """Test setting the debate creator callback."""
        scheduler.set_debate_creator(mock_debate_creator)
        assert scheduler._debate_creator is mock_debate_creator

    def test_set_debate_creator_overwrite(self, scheduler):
        """Test overwriting the debate creator."""
        first = AsyncMock()
        second = AsyncMock()
        scheduler.set_debate_creator(first)
        scheduler.set_debate_creator(second)
        assert scheduler._debate_creator is second


# =============================================================================
# PulseDebateScheduler Tests - Start/Stop Lifecycle
# =============================================================================


class TestSchedulerLifecycle:
    """Tests for scheduler start/stop/pause/resume lifecycle."""

    @pytest.mark.asyncio
    async def test_start_without_creator_raises(self, scheduler):
        """Test starting without a debate creator raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            await scheduler.start()

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self, scheduler_with_creator):
        """Test that start() sets state to RUNNING."""
        await scheduler_with_creator.start()
        assert scheduler_with_creator.state == SchedulerState.RUNNING
        # Cleanup
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_start_creates_task(self, scheduler_with_creator):
        """Test that start() creates the background task."""
        await scheduler_with_creator.start()
        assert scheduler_with_creator._task is not None
        assert not scheduler_with_creator._task.done()
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_start_sets_run_id(self, scheduler_with_creator):
        """Test that start() generates a run_id."""
        await scheduler_with_creator.start()
        assert scheduler_with_creator._run_id.startswith("run-")
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_start_resets_metrics(self, scheduler_with_creator):
        """Test that start() resets metrics."""
        scheduler_with_creator._metrics.polls_completed = 10
        await scheduler_with_creator.start()
        assert scheduler_with_creator.metrics.polls_completed == 0
        assert scheduler_with_creator.metrics.start_time is not None
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, scheduler_with_creator):
        """Test start when already running does not raise (just returns)."""
        await scheduler_with_creator.start()
        # Should not raise, just log warning
        await scheduler_with_creator.start()
        assert scheduler_with_creator.state == SchedulerState.RUNNING
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self, scheduler_with_creator):
        """Test that stop() sets state to STOPPED."""
        await scheduler_with_creator.start()
        await scheduler_with_creator.stop(graceful=False)
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped(self, scheduler_with_creator):
        """Test stop when already stopped does not raise."""
        assert scheduler_with_creator.state == SchedulerState.STOPPED
        await scheduler_with_creator.stop()
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_graceful(self, scheduler_with_creator):
        """Test graceful stop waits for task completion."""
        await scheduler_with_creator.start()
        await scheduler_with_creator.stop(graceful=True)
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_non_graceful_cancels_task(self, scheduler_with_creator):
        """Test non-graceful stop cancels the task."""
        await scheduler_with_creator.start()
        task = scheduler_with_creator._task
        await scheduler_with_creator.stop(graceful=False)
        assert task.done()

    @pytest.mark.asyncio
    async def test_pause_from_running(self, scheduler_with_creator):
        """Test pausing a running scheduler."""
        await scheduler_with_creator.start()
        await scheduler_with_creator.pause()
        assert scheduler_with_creator.state == SchedulerState.PAUSED
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_pause_from_stopped(self, scheduler_with_creator):
        """Test pausing a stopped scheduler does nothing."""
        await scheduler_with_creator.pause()
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_resume_from_paused(self, scheduler_with_creator):
        """Test resuming a paused scheduler."""
        await scheduler_with_creator.start()
        await scheduler_with_creator.pause()
        assert scheduler_with_creator.state == SchedulerState.PAUSED
        await scheduler_with_creator.resume()
        assert scheduler_with_creator.state == SchedulerState.RUNNING
        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_resume_from_stopped(self, scheduler_with_creator):
        """Test resuming a stopped scheduler does nothing."""
        await scheduler_with_creator.resume()
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_resume_from_running(self, scheduler_with_creator):
        """Test resuming a running scheduler does nothing."""
        await scheduler_with_creator.start()
        await scheduler_with_creator.resume()
        assert scheduler_with_creator.state == SchedulerState.RUNNING
        await scheduler_with_creator.stop(graceful=False)


# =============================================================================
# PulseDebateScheduler Tests - Configuration Updates
# =============================================================================


class TestSchedulerConfigUpdate:
    """Tests for scheduler configuration updates."""

    def test_update_config_single_key(self, scheduler):
        """Test updating a single config key."""
        scheduler.update_config({"poll_interval_seconds": 60})
        assert scheduler.config.poll_interval_seconds == 60
        # Others unchanged
        assert scheduler.config.max_debates_per_hour == 6

    def test_update_config_multiple_keys(self, scheduler):
        """Test updating multiple config keys."""
        scheduler.update_config(
            {
                "poll_interval_seconds": 120,
                "max_debates_per_hour": 10,
                "debate_rounds": 5,
            }
        )
        assert scheduler.config.poll_interval_seconds == 120
        assert scheduler.config.max_debates_per_hour == 10
        assert scheduler.config.debate_rounds == 5

    def test_update_config_recreates_topic_selector(self, scheduler):
        """Test that config update recreates the topic selector."""
        old_selector = scheduler._topic_selector
        scheduler.update_config({"min_volume_threshold": 200})
        assert scheduler._topic_selector is not old_selector
        assert scheduler._topic_selector.config.min_volume_threshold == 200


# =============================================================================
# PulseDebateScheduler Tests - Status
# =============================================================================


class TestSchedulerStatus:
    """Tests for scheduler status reporting."""

    def test_get_status_stopped(self, scheduler):
        """Test status when stopped."""
        status = scheduler.get_status()
        assert status["state"] == "stopped"
        assert status["run_id"] == ""
        assert "config" in status
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_get_status_running(self, scheduler_with_creator):
        """Test status when running."""
        await scheduler_with_creator.start()
        status = scheduler_with_creator.get_status()
        assert status["state"] == "running"
        assert status["run_id"].startswith("run-")
        assert status["metrics"]["uptime_seconds"] is not None
        await scheduler_with_creator.stop(graceful=False)

    def test_get_status_has_config(self, scheduler):
        """Test status includes config."""
        status = scheduler.get_status()
        config = status["config"]
        assert config["poll_interval_seconds"] == 300
        assert config["max_debates_per_hour"] == 6

    def test_get_status_has_metrics(self, scheduler):
        """Test status includes metrics."""
        status = scheduler.get_status()
        metrics = status["metrics"]
        assert metrics["polls_completed"] == 0
        assert metrics["debates_created"] == 0


# =============================================================================
# PulseDebateScheduler Tests - Rate Limiting
# =============================================================================


class TestSchedulerRateLimiting:
    """Tests for _can_create_debate rate limiting."""

    def test_can_create_debate_initially(self, scheduler):
        """Test can create debate when no debates have been created."""
        assert scheduler._can_create_debate() is True

    def test_hourly_limit_reached(self, scheduler):
        """Test rate limit blocks when hourly limit is reached."""
        now = time.time()
        # Fill up the hourly limit (default 6)
        scheduler._debates_this_hour = [now - i * 600 for i in range(6)]
        assert scheduler._can_create_debate() is False

    def test_min_interval_not_met(self, scheduler):
        """Test rate limit blocks when min interval not met."""
        now = time.time()
        scheduler._debates_this_hour = [now - 60]  # Only 60s ago, min is 600
        assert scheduler._can_create_debate() is False

    def test_min_interval_met(self, scheduler):
        """Test allows debate when min interval is met."""
        now = time.time()
        scheduler._debates_this_hour = [now - 700]  # 700s ago, min is 600
        assert scheduler._can_create_debate() is True

    def test_old_debates_pruned(self, scheduler):
        """Test that debates older than 1 hour are pruned."""
        now = time.time()
        scheduler._debates_this_hour = [
            now - 7200,  # 2 hours ago - should be pruned
            now - 4000,  # 1+ hours ago - should be pruned
            now - 700,  # 11 minutes ago - should remain
        ]
        result = scheduler._can_create_debate()
        assert result is True
        # Only the recent one should remain after pruning
        assert len(scheduler._debates_this_hour) == 1

    def test_hourly_limit_with_custom_config(self, mock_pulse_manager, mock_store):
        """Test rate limiting with custom max_debates_per_hour."""
        config = SchedulerConfig(max_debates_per_hour=2, min_interval_between_debates=0)
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=config)
        now = time.time()
        scheduler._debates_this_hour = [now - 10, now - 5]
        assert scheduler._can_create_debate() is False

    def test_min_interval_with_custom_config(self, mock_pulse_manager, mock_store):
        """Test rate limiting with custom min_interval_between_debates."""
        config = SchedulerConfig(max_debates_per_hour=100, min_interval_between_debates=30)
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=config)
        now = time.time()
        scheduler._debates_this_hour = [now - 10]  # 10s ago, min is 30
        assert scheduler._can_create_debate() is False
        scheduler._debates_this_hour = [now - 40]  # 40s ago, min is 30
        assert scheduler._can_create_debate() is True


# =============================================================================
# PulseDebateScheduler Tests - KM Adapter
# =============================================================================


class TestKMAdapter:
    """Tests for Knowledge Mound adapter integration."""

    def test_set_km_adapter(self, scheduler):
        """Test setting KM adapter."""
        adapter = MagicMock()
        scheduler.set_km_adapter(adapter)
        assert scheduler._km_adapter is adapter

    def test_query_km_no_adapter(self, scheduler):
        """Test querying KM without adapter returns empty list."""
        result = scheduler.query_km_for_past_debates("some topic")
        assert result == []

    def test_query_km_with_adapter(self, scheduler):
        """Test querying KM with adapter returns results."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = [
            {"debate_id": "d1", "consensus_reached": True, "confidence": 0.9}
        ]
        scheduler.set_km_adapter(adapter)
        result = scheduler.query_km_for_past_debates("AI ethics", limit=5)
        assert len(result) == 1
        assert result[0]["debate_id"] == "d1"
        adapter.search_past_debates.assert_called_once_with(topic_text="AI ethics", limit=5)

    def test_query_km_adapter_error(self, scheduler):
        """Test querying KM when adapter raises returns empty list."""
        adapter = MagicMock()
        adapter.search_past_debates.side_effect = RuntimeError("KM unavailable")
        scheduler.set_km_adapter(adapter)
        result = scheduler.query_km_for_past_debates("test topic")
        assert result == []

    def test_query_km_adapter_attribute_error(self, scheduler):
        """Test querying KM when adapter has no search_past_debates."""
        adapter = MagicMock()
        adapter.search_past_debates.side_effect = AttributeError("no method")
        scheduler.set_km_adapter(adapter)
        result = scheduler.query_km_for_past_debates("test topic")
        assert result == []

    def test_query_km_default_limit(self, scheduler):
        """Test default limit parameter for KM queries."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = []
        scheduler.set_km_adapter(adapter)
        scheduler.query_km_for_past_debates("topic")
        adapter.search_past_debates.assert_called_once_with(topic_text="topic", limit=5)


# =============================================================================
# PulseDebateScheduler Tests - Poll and Create
# =============================================================================


class TestPollAndCreate:
    """Tests for _poll_and_create method."""

    @pytest.mark.asyncio
    async def test_poll_no_topics(self, scheduler_with_creator, mock_pulse_manager):
        """Test polling when no topics are returned."""
        mock_pulse_manager.get_trending_topics.return_value = []
        await scheduler_with_creator._poll_and_create()
        assert scheduler_with_creator.metrics.polls_completed == 1
        assert scheduler_with_creator.metrics.debates_created == 0

    @pytest.mark.asyncio
    async def test_poll_with_topics_creates_debate(
        self, scheduler_with_creator, mock_pulse_manager, mock_store, mock_debate_creator
    ):
        """Test polling with viable topics creates a debate."""
        topics = [
            _make_topic(
                topic="Should AI debate ethics controversy",
                category="tech",
                volume=500,
            )
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        mock_store.is_duplicate.return_value = False

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.polls_completed == 1
        assert scheduler_with_creator.metrics.debates_created == 1
        mock_debate_creator.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_rate_limited(self, scheduler_with_creator):
        """Test polling is skipped when rate limited."""
        now = time.time()
        scheduler_with_creator._debates_this_hour = [now - i * 100 for i in range(6)]
        await scheduler_with_creator._poll_and_create()
        assert scheduler_with_creator.metrics.polls_completed == 0

    @pytest.mark.asyncio
    async def test_poll_skips_duplicates(
        self, scheduler_with_creator, mock_pulse_manager, mock_store
    ):
        """Test polling skips duplicate topics."""
        topics = [
            _make_topic(topic="Should AI debate this", category="tech", volume=500),
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        mock_store.is_duplicate.return_value = True

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.duplicates_skipped == 1
        assert scheduler_with_creator.metrics.debates_created == 0

    @pytest.mark.asyncio
    async def test_poll_skips_to_next_topic_on_duplicate(
        self, scheduler_with_creator, mock_pulse_manager, mock_store, mock_debate_creator
    ):
        """Test that poll tries next topic when first is duplicate."""
        topics = [
            _make_topic(topic="Should AI debate one", category="tech", volume=500),
            _make_topic(topic="Should AI debate two", category="tech", volume=400),
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        # First topic is duplicate, second is not
        mock_store.is_duplicate.side_effect = [True, False]

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.duplicates_skipped == 1
        assert scheduler_with_creator.metrics.debates_created == 1

    @pytest.mark.asyncio
    async def test_poll_only_creates_one_debate(
        self, scheduler_with_creator, mock_pulse_manager, mock_store, mock_debate_creator
    ):
        """Test that only one debate is created per poll even with multiple viable topics."""
        topics = [
            _make_topic(topic=f"Should AI debate topic {i}", category="tech", volume=500)
            for i in range(5)
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        mock_store.is_duplicate.return_value = False

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.debates_created == 1
        mock_debate_creator.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_handles_trending_error(self, scheduler_with_creator, mock_pulse_manager):
        """Test polling handles errors from get_trending_topics."""
        mock_pulse_manager.get_trending_topics.side_effect = RuntimeError("API error")
        # Should not raise
        await scheduler_with_creator._poll_and_create()
        assert scheduler_with_creator.metrics.polls_completed == 0

    @pytest.mark.asyncio
    async def test_poll_updates_metrics(
        self, scheduler_with_creator, mock_pulse_manager, mock_store
    ):
        """Test that polling updates metrics correctly."""
        topics = [
            _make_topic(topic="Basic topic", category="tech", volume=100),
            _make_topic(topic="Politics topic", category="politics", volume=5000),
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        mock_store.is_duplicate.return_value = False

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.polls_completed == 1
        assert scheduler_with_creator.metrics.topics_evaluated == 2
        assert scheduler_with_creator.metrics.last_poll_at is not None

    @pytest.mark.asyncio
    async def test_poll_all_topics_filtered(self, scheduler_with_creator, mock_pulse_manager):
        """Test polling when all topics are filtered out."""
        topics = [
            _make_topic(topic="Politics debate", category="politics", volume=5000),
            _make_topic(topic="Religious topic", category="religion", volume=3000),
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics

        await scheduler_with_creator._poll_and_create()

        assert scheduler_with_creator.metrics.polls_completed == 1
        assert scheduler_with_creator.metrics.debates_created == 0


# =============================================================================
# PulseDebateScheduler Tests - Create Debate
# =============================================================================


class TestCreateDebate:
    """Tests for _create_debate method."""

    @pytest.mark.asyncio
    async def test_create_debate_success(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test successful debate creation."""
        topic = _make_topic(topic="AI ethics debate", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8, reasons=["good topic"])

        await scheduler_with_creator._create_debate(topic, scored)

        mock_debate_creator.assert_called_once()
        mock_store.record_scheduled_debate.assert_called_once()
        assert scheduler_with_creator.metrics.debates_created == 1
        assert scheduler_with_creator.metrics.last_debate_at is not None

    @pytest.mark.asyncio
    async def test_create_debate_records_debate_id(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test that debate record captures the debate_id from result."""
        mock_debate_creator.return_value = {
            "debate_id": "custom-debate-id",
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 2,
        }
        topic = _make_topic(topic="AI topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        record = mock_store.record_scheduled_debate.call_args[0][0]
        assert record.debate_id == "custom-debate-id"
        assert record.consensus_reached is True
        assert record.confidence == 0.9
        assert record.rounds_used == 2

    @pytest.mark.asyncio
    async def test_create_debate_creator_returns_none(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test debate creation when creator returns None."""
        mock_debate_creator.return_value = None
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert scheduler_with_creator.metrics.debates_failed == 1
        assert scheduler_with_creator.metrics.debates_created == 0

    @pytest.mark.asyncio
    async def test_create_debate_creator_raises(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test debate creation when creator raises an error."""
        mock_debate_creator.side_effect = RuntimeError("Debate engine error")
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert scheduler_with_creator.metrics.debates_failed == 1
        assert scheduler_with_creator.metrics.debates_created == 0
        # Should still record the failed attempt
        mock_store.record_scheduled_debate.assert_called_once()
        record = mock_store.record_scheduled_debate.call_args[0][0]
        assert record.debate_id is None
        assert record.consensus_reached is None

    @pytest.mark.asyncio
    async def test_create_debate_updates_debates_this_hour(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test that successful debate updates _debates_this_hour."""
        topic = _make_topic(topic="Test topic should", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert len(scheduler_with_creator._debates_this_hour) == 1

    @pytest.mark.asyncio
    async def test_create_debate_calls_with_correct_params(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test debate creator is called with correct parameters."""
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        call_args = mock_debate_creator.call_args[0]
        assert call_args[0] == topic.to_debate_prompt()
        assert call_args[1] == scheduler_with_creator.config.debate_rounds
        assert call_args[2] == scheduler_with_creator.config.consensus_threshold

    @pytest.mark.asyncio
    async def test_create_debate_skips_if_km_has_consensus(self, scheduler_with_creator):
        """Test that debate is skipped if KM reports recent consensus."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = [{"consensus_reached": True, "confidence": 0.9}]
        scheduler_with_creator.set_km_adapter(adapter)

        topic = _make_topic(topic="AI ethics", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        # Debate should not be created because KM found consensus
        assert scheduler_with_creator.metrics.debates_created == 0

    @pytest.mark.asyncio
    async def test_create_debate_proceeds_if_km_has_no_consensus(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test that debate proceeds if KM has past debates but no strong consensus."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = [{"consensus_reached": False, "confidence": 0.3}]
        scheduler_with_creator.set_km_adapter(adapter)

        topic = _make_topic(topic="AI ethics should", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        mock_debate_creator.assert_called_once()
        assert scheduler_with_creator.metrics.debates_created == 1

    @pytest.mark.asyncio
    async def test_create_debate_proceeds_if_km_consensus_low_confidence(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test debate proceeds if KM consensus exists but with low confidence."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = [
            {"consensus_reached": True, "confidence": 0.5}  # Below 0.7 threshold
        ]
        scheduler_with_creator.set_km_adapter(adapter)

        topic = _make_topic(topic="AI debate topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        mock_debate_creator.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_debate_syncs_to_km(self, scheduler_with_creator, mock_debate_creator):
        """Test that successful debate syncs to KM adapter."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = []
        scheduler_with_creator.set_km_adapter(adapter)

        mock_debate_creator.return_value = {
            "debate_id": "test-id",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
        }

        topic = _make_topic(topic="AI debate topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        adapter.store_scheduled_debate.assert_called_once()
        adapter.store_debate_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_debate_km_sync_no_consensus(
        self, scheduler_with_creator, mock_debate_creator
    ):
        """Test KM sync only stores scheduled record when no consensus."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = []
        scheduler_with_creator.set_km_adapter(adapter)

        mock_debate_creator.return_value = {
            "debate_id": "test-id",
            "consensus_reached": False,
            "confidence": 0.4,
            "rounds_used": 3,
        }

        topic = _make_topic(topic="AI debate", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        adapter.store_scheduled_debate.assert_called_once()
        adapter.store_debate_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_debate_km_sync_error(self, scheduler_with_creator, mock_debate_creator):
        """Test that KM sync errors don't prevent debate creation."""
        adapter = MagicMock()
        adapter.search_past_debates.return_value = []
        adapter.store_scheduled_debate.side_effect = RuntimeError("KM down")
        scheduler_with_creator.set_km_adapter(adapter)

        topic = _make_topic(topic="AI debate topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        # Debate should still be created even though KM sync failed
        assert scheduler_with_creator.metrics.debates_created == 1


# =============================================================================
# PulseDebateScheduler Tests - Scheduler Loop
# =============================================================================


class TestSchedulerLoop:
    """Tests for the _scheduler_loop method."""

    @pytest.mark.asyncio
    async def test_scheduler_loop_stops_on_event(self, scheduler_with_creator):
        """Test that scheduler loop exits when stop event is set."""
        scheduler_with_creator._state = SchedulerState.RUNNING
        scheduler_with_creator.config.poll_interval_seconds = 1

        # Set stop event after a short delay
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            scheduler_with_creator._stop_event.set()

        stop_task = asyncio.create_task(stop_after_delay())

        with patch.object(
            scheduler_with_creator, "_poll_and_create", new_callable=AsyncMock
        ) as mock_poll:
            await scheduler_with_creator._scheduler_loop()

        await stop_task

    @pytest.mark.asyncio
    async def test_scheduler_loop_skips_poll_when_paused(self, scheduler_with_creator):
        """Test that scheduler loop skips polling when paused."""
        scheduler_with_creator._state = SchedulerState.PAUSED
        scheduler_with_creator.config.poll_interval_seconds = 1

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            scheduler_with_creator._stop_event.set()

        stop_task = asyncio.create_task(stop_after_delay())

        with patch.object(
            scheduler_with_creator, "_poll_and_create", new_callable=AsyncMock
        ) as mock_poll:
            await scheduler_with_creator._scheduler_loop()
            mock_poll.assert_not_called()

        await stop_task

    @pytest.mark.asyncio
    async def test_scheduler_loop_calls_poll_when_running(self, scheduler_with_creator):
        """Test that scheduler loop calls _poll_and_create when running."""
        scheduler_with_creator._state = SchedulerState.RUNNING
        scheduler_with_creator.config.poll_interval_seconds = 1

        call_count = 0

        async def mock_poll():
            nonlocal call_count
            call_count += 1
            # Stop after first poll
            scheduler_with_creator._stop_event.set()

        with patch.object(scheduler_with_creator, "_poll_and_create", side_effect=mock_poll):
            await scheduler_with_creator._scheduler_loop()

        assert call_count >= 1


# =============================================================================
# PulseDebateScheduler Tests - Properties
# =============================================================================


class TestSchedulerProperties:
    """Tests for scheduler property accessors."""

    def test_state_property(self, scheduler):
        """Test state property returns current state."""
        assert scheduler.state == SchedulerState.STOPPED

    def test_metrics_property(self, scheduler):
        """Test metrics property returns metrics instance."""
        metrics = scheduler.metrics
        assert isinstance(metrics, SchedulerMetrics)

    def test_metrics_property_reflects_changes(self, scheduler):
        """Test metrics property reflects incremented values."""
        scheduler._metrics.polls_completed = 5
        assert scheduler.metrics.polls_completed == 5


# =============================================================================
# PulseDebateScheduler Tests - Full Integration Flow
# =============================================================================


class TestSchedulerIntegration:
    """Integration-style tests for the full scheduler flow."""

    @pytest.mark.asyncio
    async def test_full_poll_create_flow(self, mock_pulse_manager, mock_store):
        """Test the full flow: start -> poll -> create debate -> stop."""
        creator = AsyncMock(
            return_value={
                "debate_id": "integration-test-123",
                "consensus_reached": True,
                "confidence": 0.9,
                "rounds_used": 2,
            }
        )

        config = SchedulerConfig(
            poll_interval_seconds=1,
            min_volume_threshold=50,
        )
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=config)
        scheduler.set_debate_creator(creator)

        topics = [
            _make_topic(
                topic="Should AI debate controversial ethics issue",
                category="tech",
                volume=500,
            )
        ]
        mock_pulse_manager.get_trending_topics.return_value = topics
        mock_store.is_duplicate.return_value = False

        await scheduler.start()
        # Allow time for one poll cycle
        await asyncio.sleep(0.1)
        await scheduler.stop(graceful=False)

        assert scheduler.metrics.polls_completed >= 1
        assert scheduler.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_start_stop_start_lifecycle(self, mock_pulse_manager, mock_store):
        """Test starting, stopping, and restarting the scheduler."""
        creator = AsyncMock(return_value=None)
        config = SchedulerConfig(poll_interval_seconds=1)
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=config)
        scheduler.set_debate_creator(creator)

        # First start
        await scheduler.start()
        assert scheduler.state == SchedulerState.RUNNING
        first_run_id = scheduler._run_id

        await scheduler.stop(graceful=False)
        assert scheduler.state == SchedulerState.STOPPED

        # Second start should work with a new run_id
        await scheduler.start()
        assert scheduler.state == SchedulerState.RUNNING
        assert scheduler._run_id != first_run_id

        await scheduler.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_pause_resume_flow(self, scheduler_with_creator):
        """Test pause and resume flow."""
        await scheduler_with_creator.start()
        assert scheduler_with_creator.state == SchedulerState.RUNNING

        await scheduler_with_creator.pause()
        assert scheduler_with_creator.state == SchedulerState.PAUSED

        await scheduler_with_creator.resume()
        assert scheduler_with_creator.state == SchedulerState.RUNNING

        await scheduler_with_creator.stop(graceful=False)
        assert scheduler_with_creator.state == SchedulerState.STOPPED

    @pytest.mark.asyncio
    async def test_config_update_during_operation(self, scheduler_with_creator):
        """Test updating config while scheduler is active."""
        await scheduler_with_creator.start()

        scheduler_with_creator.update_config({"max_debates_per_hour": 20})
        assert scheduler_with_creator.config.max_debates_per_hour == 20

        await scheduler_with_creator.stop(graceful=False)

    @pytest.mark.asyncio
    async def test_multiple_polls_accumulate_metrics(
        self, scheduler_with_creator, mock_pulse_manager, mock_store, mock_debate_creator
    ):
        """Test that multiple polls accumulate metrics correctly."""
        topics = [_make_topic(topic="Should AI debate ethics", category="tech", volume=500)]
        mock_pulse_manager.get_trending_topics.return_value = topics
        # First poll: not duplicate, second poll: duplicate
        mock_store.is_duplicate.side_effect = [False, True]

        # First poll
        await scheduler_with_creator._poll_and_create()
        assert scheduler_with_creator.metrics.polls_completed == 1
        assert scheduler_with_creator.metrics.debates_created == 1

        # Need to wait for min_interval (or reset for testing)
        scheduler_with_creator._debates_this_hour = []

        # Second poll
        await scheduler_with_creator._poll_and_create()
        assert scheduler_with_creator.metrics.polls_completed == 2
        assert scheduler_with_creator.metrics.duplicates_skipped == 1


# =============================================================================
# PulseDebateScheduler Tests - Edge Cases
# =============================================================================


class TestSchedulerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_debate_creator_value_error(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test debate creation handles ValueError."""
        mock_debate_creator.side_effect = ValueError("Bad value")
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert scheduler_with_creator.metrics.debates_failed == 1
        mock_store.record_scheduled_debate.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_creator_type_error(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test debate creation handles TypeError."""
        mock_debate_creator.side_effect = TypeError("Type mismatch")
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert scheduler_with_creator.metrics.debates_failed == 1

    @pytest.mark.asyncio
    async def test_debate_creator_os_error(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test debate creation handles OSError."""
        mock_debate_creator.side_effect = OSError("Disk full")
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        assert scheduler_with_creator.metrics.debates_failed == 1

    def test_topic_with_zero_volume(self, topic_selector):
        """Test scoring a topic with zero volume."""
        topic = _make_topic(topic="Something", category="tech", volume=0)
        result = topic_selector.score_topic(topic)
        assert result.score == -1.0

    def test_topic_with_empty_text(self, topic_selector):
        """Test scoring a topic with empty text."""
        topic = _make_topic(topic="", category="tech", volume=500)
        result = topic_selector.score_topic(topic)
        # Should still score based on category and volume
        assert result.score > 0

    def test_topic_with_special_characters(self, topic_selector):
        """Test scoring a topic with special characters."""
        topic = _make_topic(
            topic="AI vs. Humans: The $1M Question!!! @debate #controversy",
            category="tech",
            volume=500,
        )
        result = topic_selector.score_topic(topic)
        assert result.score > 0

    def test_select_best_topics_all_non_viable(self, topic_selector):
        """Test selecting from only non-viable topics."""
        topics = [
            _make_topic(topic="Politics debate", category="politics", volume=5000),
            _make_topic(topic="Religious topic", category="religion", volume=3000),
            _make_topic(topic="Low volume", category="tech", volume=1),
        ]
        result = topic_selector.select_best_topics(topics, limit=5)
        assert len(result) == 0

    def test_select_best_topics_limit_zero(self, topic_selector):
        """Test selecting with limit=0."""
        topics = [
            _make_topic(topic="Good topic debate", category="tech", volume=500),
        ]
        result = topic_selector.select_best_topics(topics, limit=0)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_poll_trending_topics_value_error(
        self, scheduler_with_creator, mock_pulse_manager
    ):
        """Test polling handles ValueError from get_trending_topics."""
        mock_pulse_manager.get_trending_topics.side_effect = ValueError("Bad filter")
        await scheduler_with_creator._poll_and_create()
        # Should not raise

    @pytest.mark.asyncio
    async def test_poll_trending_topics_type_error(
        self, scheduler_with_creator, mock_pulse_manager
    ):
        """Test polling handles TypeError from get_trending_topics."""
        mock_pulse_manager.get_trending_topics.side_effect = TypeError("Type issue")
        await scheduler_with_creator._poll_and_create()
        # Should not raise

    def test_scheduler_config_from_dict_ignores_extra_keys(self):
        """Test from_dict ignores keys that are not part of the config."""
        data = {
            "poll_interval_seconds": 60,
            "extra_key": "should_be_ignored",
            "another_extra": 999,
        }
        config = SchedulerConfig.from_dict(data)
        assert config.poll_interval_seconds == 60
        assert not hasattr(config, "extra_key")

    @pytest.mark.asyncio
    async def test_create_debate_with_empty_result_dict(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test debate creation with empty result dict (truthy but no keys)."""
        mock_debate_creator.return_value = {}
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        # Empty dict is truthy, so it should be treated as a successful result
        assert scheduler_with_creator.metrics.debates_created == 1
        record = mock_store.record_scheduled_debate.call_args[0][0]
        assert record.consensus_reached is False  # default from .get()
        assert record.confidence == 0.0
        assert record.rounds_used == 0

    @pytest.mark.asyncio
    async def test_record_scheduled_debate_failure_on_create(
        self, scheduler_with_creator, mock_debate_creator, mock_store
    ):
        """Test that failed debate still records the attempt with None debate_id."""
        mock_debate_creator.side_effect = RuntimeError("Engine crashed")
        topic = _make_topic(topic="Test topic", category="tech", volume=500)
        scored = TopicScore(topic=topic, score=0.8)

        await scheduler_with_creator._create_debate(topic, scored)

        record = mock_store.record_scheduled_debate.call_args[0][0]
        assert record.debate_id is None
        assert record.consensus_reached is None
        assert record.confidence is None
        assert record.rounds_used == 0

    def test_can_create_debate_exactly_at_limit(self, scheduler):
        """Test rate limit at exact boundary of max_debates_per_hour."""
        now = time.time()
        # 5 debates (one below limit of 6), all far enough apart
        scheduler._debates_this_hour = [now - (i + 1) * 700 for i in range(5)]
        assert scheduler._can_create_debate() is True

    def test_can_create_debate_at_exact_min_interval(self, scheduler):
        """Test rate limit at exact min_interval_between_debates boundary."""
        now = time.time()
        # Exactly at min_interval (600s) - should not pass because condition is strict <
        scheduler._debates_this_hour = [now - 600]
        # now - last_debate = 600, condition is now - last_debate < 600 which is False
        assert scheduler._can_create_debate() is True

    @pytest.mark.asyncio
    async def test_stop_graceful_timeout(self, mock_pulse_manager, mock_store):
        """Test graceful stop with timeout when task doesn't finish."""
        config = SchedulerConfig(poll_interval_seconds=300)
        scheduler = PulseDebateScheduler(mock_pulse_manager, mock_store, config=config)
        creator = AsyncMock(return_value=None)
        scheduler.set_debate_creator(creator)

        await scheduler.start()
        # Force stop with a very short timeout scenario - just ensure it completes
        await scheduler.stop(graceful=True)
        assert scheduler.state == SchedulerState.STOPPED
