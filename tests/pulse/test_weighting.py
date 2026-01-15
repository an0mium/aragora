"""Tests for Pulse source weighting system."""

import math
import pytest

from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.weighting import (
    DEFAULT_SOURCE_WEIGHTS,
    SourceWeight,
    SourceWeightingSystem,
    WeightedTopic,
)


class TestSourceWeight:
    """Tests for SourceWeight dataclass."""

    def test_source_weight_creation(self):
        """Test basic SourceWeight creation."""
        weight = SourceWeight(
            platform="test",
            base_credibility=0.75,
            authority_score=0.80,
            volume_multiplier=1.5,
            freshness_weight=1.2,
            description="Test source",
        )
        assert weight.platform == "test"
        assert weight.base_credibility == 0.75
        assert weight.authority_score == 0.80
        assert weight.volume_multiplier == 1.5
        assert weight.freshness_weight == 1.2

    def test_default_source_weights_present(self):
        """Test that default weights include major platforms."""
        assert "hackernews" in DEFAULT_SOURCE_WEIGHTS
        assert "reddit" in DEFAULT_SOURCE_WEIGHTS
        assert "twitter" in DEFAULT_SOURCE_WEIGHTS
        assert "github" in DEFAULT_SOURCE_WEIGHTS

    def test_github_highest_credibility(self):
        """Test GitHub has highest default credibility."""
        github = DEFAULT_SOURCE_WEIGHTS["github"]
        twitter = DEFAULT_SOURCE_WEIGHTS["twitter"]
        assert github.base_credibility > twitter.base_credibility
        assert github.authority_score > twitter.authority_score


class TestWeightedTopic:
    """Tests for WeightedTopic dataclass."""

    def test_weighted_topic_properties(self):
        """Test WeightedTopic property access."""
        topic = TrendingTopic(
            platform="hackernews",
            topic="AI Breakthrough",
            volume=500,
            category="tech",
        )
        weight = DEFAULT_SOURCE_WEIGHTS["hackernews"]

        weighted = WeightedTopic(
            topic=topic,
            source_weight=weight,
            weighted_score=0.75,
            components={"credibility": 0.34},
        )

        assert weighted.platform == "hackernews"
        assert weighted.credibility == weight.base_credibility
        assert weighted.authority == weight.authority_score


class TestSourceWeightingSystem:
    """Tests for SourceWeightingSystem."""

    @pytest.fixture
    def system(self):
        """Create a weighting system for testing."""
        return SourceWeightingSystem()

    def test_get_known_source_weight(self, system):
        """Test getting weight for known platform."""
        weight = system.get_source_weight("hackernews")
        assert weight.platform == "hackernews"
        assert weight.base_credibility == DEFAULT_SOURCE_WEIGHTS["hackernews"].base_credibility

    def test_get_unknown_source_weight(self, system):
        """Test getting weight for unknown platform."""
        weight = system.get_source_weight("unknown_platform")
        assert weight.platform == "unknown_platform"
        assert weight.base_credibility == 0.50  # Default

    def test_weight_topic_basic(self, system):
        """Test basic topic weighting."""
        topic = TrendingTopic(
            platform="hackernews",
            topic="Show HN: New AI Tool",
            volume=342,
            category="tech",
        )

        weighted = system.weight_topic(topic)

        assert weighted.topic == topic
        assert 0.0 <= weighted.weighted_score <= 1.0
        assert "credibility" in weighted.components
        assert "volume" in weighted.components
        assert "authority" in weighted.components

    def test_weight_topic_high_volume(self, system):
        """Test that high volume increases score."""
        low_volume = TrendingTopic(platform="hackernews", topic="Test", volume=10, category="tech")
        high_volume = TrendingTopic(
            platform="hackernews", topic="Test", volume=100000, category="tech"
        )

        low_weighted = system.weight_topic(low_volume)
        high_weighted = system.weight_topic(high_volume)

        assert high_weighted.weighted_score > low_weighted.weighted_score

    def test_weight_topics_batch(self, system):
        """Test weighting multiple topics."""
        topics = [
            TrendingTopic("hackernews", "Topic 1", 100, "tech"),
            TrendingTopic("reddit", "Topic 2", 5000, "tech"),
            TrendingTopic("twitter", "Topic 3", 50000, "tech"),
        ]

        weighted = system.weight_topics(topics)

        assert len(weighted) == 3
        assert all(isinstance(w, WeightedTopic) for w in weighted)

    def test_rank_by_weighted_score(self, system):
        """Test ranking topics by score."""
        topics = [
            TrendingTopic("twitter", "Low credibility", 1000, "general"),
            TrendingTopic("github", "High credibility", 1000, "tech"),
            TrendingTopic("hackernews", "Medium credibility", 1000, "tech"),
        ]

        weighted = system.weight_topics(topics)
        ranked = system.rank_by_weighted_score(weighted)

        # GitHub should rank highest due to credibility
        assert ranked[0].platform == "github"

    def test_rank_with_min_credibility_filter(self, system):
        """Test filtering by minimum credibility."""
        topics = [
            TrendingTopic("github", "High", 100, "tech"),
            TrendingTopic("twitter", "Low", 100, "general"),
        ]

        weighted = system.weight_topics(topics)
        ranked = system.rank_by_weighted_score(weighted, min_credibility=0.80)

        # Only GitHub should pass high credibility filter
        assert len(ranked) == 1
        assert ranked[0].platform == "github"

    def test_update_source_weight(self, system):
        """Test updating source weights."""
        original = system.get_source_weight("hackernews")
        original_cred = original.base_credibility

        updated = system.update_source_weight("hackernews", credibility=0.95)

        assert updated.base_credibility == 0.95
        assert updated.base_credibility != original_cred

    def test_record_topic_performance(self, system):
        """Test recording and tracking performance."""
        # Record some performances
        system.record_topic_performance("hackernews", 0.8)
        system.record_topic_performance("hackernews", 0.9)
        system.record_topic_performance("hackernews", 0.7)

        # Check adaptive credibility changes
        adaptive = system.get_adaptive_credibility("hackernews")

        # Should be blend of base (0.85) and average performance (0.8)
        # 0.85 * 0.7 + 0.8 * 0.3 = 0.595 + 0.24 = 0.835
        expected = 0.85 * 0.7 + 0.8 * 0.3
        assert abs(adaptive - expected) < 0.01

    def test_get_source_stats(self, system):
        """Test getting source statistics."""
        system.record_topic_performance("hackernews", 0.75)

        stats = system.get_source_stats()

        assert "hackernews" in stats
        assert "base_credibility" in stats["hackernews"]
        assert "debate_count" in stats["hackernews"]
        assert stats["hackernews"]["debate_count"] == 1


class TestWeightingEdgeCases:
    """Edge case tests for weighting system."""

    def test_zero_volume_topic(self):
        """Test handling topics with zero volume."""
        system = SourceWeightingSystem()
        topic = TrendingTopic("hackernews", "Zero volume", 0, "tech")

        weighted = system.weight_topic(topic)

        assert weighted.weighted_score > 0  # Should still have credibility score
        assert weighted.components["volume"] >= 0

    def test_very_high_volume(self):
        """Test handling extremely high volume."""
        system = SourceWeightingSystem()
        topic = TrendingTopic("twitter", "Viral", 10_000_000, "viral")

        weighted = system.weight_topic(topic)

        # Volume component should be capped
        assert weighted.components["volume"] <= 0.30

    def test_case_insensitive_platform(self):
        """Test platform names are case-insensitive."""
        system = SourceWeightingSystem()

        weight1 = system.get_source_weight("HackerNews")
        weight2 = system.get_source_weight("hackernews")
        weight3 = system.get_source_weight("HACKERNEWS")

        assert weight1.base_credibility == weight2.base_credibility
        assert weight2.base_credibility == weight3.base_credibility
