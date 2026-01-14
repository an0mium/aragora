"""Tests for Pulse freshness decay system."""

import time
import pytest

from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.freshness import (
    DEFAULT_PLATFORM_HALF_LIVES,
    DecayModel,
    FreshnessCalculator,
    FreshnessConfig,
    FreshnessScore,
)


class TestDecayModel:
    """Tests for DecayModel enum."""

    def test_decay_models_exist(self):
        """Test all decay models are defined."""
        assert DecayModel.EXPONENTIAL.value == "exponential"
        assert DecayModel.LINEAR.value == "linear"
        assert DecayModel.STEP.value == "step"
        assert DecayModel.LOGARITHMIC.value == "logarithmic"


class TestFreshnessConfig:
    """Tests for FreshnessConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FreshnessConfig()

        assert config.decay_model == DecayModel.EXPONENTIAL
        assert config.half_life_hours == 6.0
        assert config.max_age_hours == 48.0
        assert config.min_freshness == 0.05

    def test_custom_config(self):
        """Test custom configuration."""
        config = FreshnessConfig(
            decay_model=DecayModel.LINEAR,
            half_life_hours=12.0,
            max_age_hours=72.0,
            min_freshness=0.1,
        )

        assert config.decay_model == DecayModel.LINEAR
        assert config.half_life_hours == 12.0


class TestFreshnessScore:
    """Tests for FreshnessScore dataclass."""

    def test_freshness_score_is_stale(self):
        """Test is_stale property."""
        topic = TrendingTopic("hackernews", "Test", 100, "tech")

        fresh = FreshnessScore(
            topic=topic,
            freshness=0.8,
            age_hours=1.0,
            decay_model=DecayModel.EXPONENTIAL,
            half_life_hours=6.0,
        )

        stale = FreshnessScore(
            topic=topic,
            freshness=0.05,
            age_hours=30.0,
            decay_model=DecayModel.EXPONENTIAL,
            half_life_hours=6.0,
        )

        assert not fresh.is_stale
        assert stale.is_stale


class TestPlatformHalfLives:
    """Tests for default platform half-lives."""

    def test_platform_half_lives_defined(self):
        """Test default half-lives for platforms."""
        assert "twitter" in DEFAULT_PLATFORM_HALF_LIVES
        assert "hackernews" in DEFAULT_PLATFORM_HALF_LIVES
        assert "reddit" in DEFAULT_PLATFORM_HALF_LIVES
        assert "github" in DEFAULT_PLATFORM_HALF_LIVES

    def test_twitter_decays_fastest(self):
        """Test Twitter has shortest half-life."""
        twitter_hl = DEFAULT_PLATFORM_HALF_LIVES["twitter"]
        github_hl = DEFAULT_PLATFORM_HALF_LIVES["github"]

        assert twitter_hl < github_hl


class TestFreshnessCalculator:
    """Tests for FreshnessCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a freshness calculator for testing."""
        return FreshnessCalculator()

    def test_brand_new_topic(self, calculator):
        """Test freshness of a brand new topic."""
        topic = TrendingTopic("hackernews", "Just posted", 100, "tech")
        now = time.time()

        score = calculator.calculate_freshness(topic, created_at=now, reference_time=now)

        assert score.freshness > 0.99
        assert score.age_hours < 0.01

    def test_old_topic(self, calculator):
        """Test freshness of an old topic."""
        topic = TrendingTopic("hackernews", "Old news", 100, "tech")
        now = time.time()
        created = now - (24 * 3600)  # 24 hours ago

        score = calculator.calculate_freshness(topic, created_at=created, reference_time=now)

        assert score.freshness < 0.3
        assert 23.9 < score.age_hours < 24.1

    def test_expired_topic(self, calculator):
        """Test topic past max age."""
        topic = TrendingTopic("hackernews", "Ancient", 100, "tech")
        now = time.time()
        created = now - (72 * 3600)  # 72 hours ago (past 48h max)

        score = calculator.calculate_freshness(topic, created_at=created, reference_time=now)

        assert score.freshness == 0.0

    def test_platform_specific_decay(self, calculator):
        """Test different platforms decay at different rates."""
        now = time.time()
        created = now - (6 * 3600)  # 6 hours ago

        twitter_topic = TrendingTopic("twitter", "Tweet", 1000, "general")
        github_topic = TrendingTopic("github", "Repo", 500, "tech")

        twitter_score = calculator.calculate_freshness(
            twitter_topic, created_at=created, reference_time=now
        )
        github_score = calculator.calculate_freshness(
            github_topic, created_at=created, reference_time=now
        )

        # Twitter should decay faster than GitHub
        assert twitter_score.freshness < github_score.freshness

    def test_exponential_decay(self):
        """Test exponential decay model."""
        config = FreshnessConfig(
            decay_model=DecayModel.EXPONENTIAL,
            half_life_hours=6.0,
            platform_half_lives={"hackernews": 6.0},  # Override platform default
        )
        calculator = FreshnessCalculator(config)
        topic = TrendingTopic("hackernews", "Test", 100, "tech")
        now = time.time()

        # At half-life, should be ~0.5
        score = calculator.calculate_freshness(
            topic, created_at=now - (6 * 3600), reference_time=now
        )
        assert 0.45 < score.freshness < 0.55

    def test_linear_decay(self):
        """Test linear decay model."""
        config = FreshnessConfig(
            decay_model=DecayModel.LINEAR,
            max_age_hours=48.0,
        )
        calculator = FreshnessCalculator(config)
        topic = TrendingTopic("hackernews", "Test", 100, "tech")
        now = time.time()

        # At 24h (half of max), should be ~0.5
        score = calculator.calculate_freshness(
            topic, created_at=now - (24 * 3600), reference_time=now
        )
        assert 0.45 < score.freshness < 0.55

    def test_step_decay(self):
        """Test step decay model."""
        config = FreshnessConfig(
            decay_model=DecayModel.STEP,
            half_life_hours=6.0,
            platform_half_lives={"hackernews": 6.0},  # Override platform default
        )
        calculator = FreshnessCalculator(config)
        topic = TrendingTopic("hackernews", "Test", 100, "tech")
        now = time.time()

        # Before threshold = 1.0
        fresh_score = calculator.calculate_freshness(
            topic, created_at=now - (5 * 3600), reference_time=now
        )
        # After threshold = 0.0 (or min_freshness)
        stale_score = calculator.calculate_freshness(
            topic, created_at=now - (7 * 3600), reference_time=now
        )

        assert fresh_score.freshness == 1.0
        assert stale_score.freshness <= 0.05  # min_freshness

    def test_score_topics_batch(self, calculator):
        """Test scoring multiple topics."""
        now = time.time()
        topics = [
            TrendingTopic("hackernews", "Topic 1", 100, "tech"),
            TrendingTopic("reddit", "Topic 2", 200, "tech"),
            TrendingTopic("twitter", "Topic 3", 300, "tech"),
        ]
        timestamps = {
            "Topic 1": now - (2 * 3600),
            "Topic 2": now - (12 * 3600),
            "Topic 3": now - (1 * 3600),
        }

        scores = calculator.score_topics(topics, timestamps, now)

        assert len(scores) == 3
        # Newest should have highest freshness
        twitter_score = next(s for s in scores if s.topic.platform == "twitter")
        reddit_score = next(s for s in scores if s.topic.platform == "reddit")
        assert twitter_score.freshness > reddit_score.freshness

    def test_filter_stale(self, calculator):
        """Test filtering out stale topics."""
        now = time.time()
        topics = [
            TrendingTopic("hackernews", "Fresh", 100, "tech"),
            TrendingTopic("hackernews", "Stale", 100, "tech"),
        ]
        timestamps = {
            "Fresh": now - (1 * 3600),  # 1 hour old
            "Stale": now - (40 * 3600),  # 40 hours old
        }

        fresh_only = calculator.filter_stale(topics, timestamps, min_freshness=0.2)

        assert len(fresh_only) == 1
        assert fresh_only[0].topic.topic == "Fresh"

    def test_set_platform_half_life(self, calculator):
        """Test setting custom platform half-life."""
        calculator.set_platform_half_life("hackernews", 2.0)

        now = time.time()
        topic = TrendingTopic("hackernews", "Test", 100, "tech")
        created = now - (2 * 3600)  # 2 hours ago

        score = calculator.calculate_freshness(topic, created_at=created, reference_time=now)

        # With 2h half-life, should be ~0.5 after 2h
        assert score.half_life_hours == 2.0
        assert 0.45 < score.freshness < 0.55

    def test_get_decay_curve(self, calculator):
        """Test getting decay curve data."""
        curve = calculator.get_decay_curve("hackernews", max_hours=24, points=10)

        assert len(curve) == 11  # points + 1
        assert curve[0]["hours"] == 0.0
        assert curve[0]["freshness"] > 0.99
        assert curve[-1]["hours"] == 24.0
        assert curve[-1]["freshness"] < curve[0]["freshness"]

    def test_get_stats(self, calculator):
        """Test getting calculator stats."""
        stats = calculator.get_stats()

        assert "decay_model" in stats
        assert "half_life_hours" in stats
        assert "platform_half_lives" in stats


class TestFreshnessEdgeCases:
    """Edge case tests for freshness calculator."""

    def test_future_timestamp(self):
        """Test handling future creation time."""
        calculator = FreshnessCalculator()
        topic = TrendingTopic("hackernews", "Future", 100, "tech")
        now = time.time()
        future = now + (1 * 3600)  # 1 hour in future

        score = calculator.calculate_freshness(topic, created_at=future, reference_time=now)

        # Future timestamp should result in max freshness (age = 0)
        assert score.freshness > 0.99
        assert score.age_hours == 0.0

    def test_no_timestamp_defaults_to_fresh(self):
        """Test handling missing timestamp."""
        calculator = FreshnessCalculator()
        topic = TrendingTopic("hackernews", "No time", 100, "tech")

        score = calculator.calculate_freshness(topic)

        # Should default to fresh (current time)
        assert score.freshness > 0.99

    def test_timestamp_from_raw_data(self):
        """Test extracting timestamp from raw_data via score_topics."""
        calculator = FreshnessCalculator()
        now = time.time()
        topic = TrendingTopic(
            platform="hackernews",
            topic="Has timestamp",
            volume=100,
            category="tech",
            raw_data={"created_at": now - (3 * 3600)},  # 3 hours ago
        )

        # score_topics extracts timestamps from raw_data
        scores = calculator.score_topics([topic], reference_time=now)
        score = scores[0]

        assert 2.9 < score.age_hours < 3.1

    def test_min_freshness_floor(self):
        """Test minimum freshness floor is applied."""
        config = FreshnessConfig(min_freshness=0.1)
        calculator = FreshnessCalculator(config)
        topic = TrendingTopic("hackernews", "Old", 100, "tech")
        now = time.time()
        created = now - (45 * 3600)  # 45 hours (close to max)

        score = calculator.calculate_freshness(topic, created_at=created, reference_time=now)

        # Should be at least min_freshness
        assert score.freshness >= 0.1
