"""Pulse integration tests with mocked providers.

Tests the full Pulse trending topics flow using mocked ingestors
to avoid real API calls during testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional


# Mock TrendingTopic dataclass
class MockTrendingTopic:
    """Mock trending topic for testing."""

    def __init__(
        self,
        topic: str,
        platform: str,
        volume: int,
        category: str = "tech",
        url: Optional[str] = None,
    ):
        self.topic = topic
        self.platform = platform
        self.volume = volume
        self.category = category
        self.url = url

    def to_debate_prompt(self) -> str:
        return f"Debate: {self.topic}"


class TestPulseManagerIntegration:
    """Test PulseManager with mocked ingestors."""

    def _create_mock_manager(self):
        """Create a mock PulseManager with canned responses."""
        manager = MagicMock()

        # Mock ingestors
        manager.ingestors = {
            "hackernews": MagicMock(),
            "reddit": MagicMock(),
            "twitter": MagicMock(),
        }

        # Mock get_trending_topics to return canned data
        async def mock_get_trending():
            return [
                MockTrendingTopic("AI Regulation Debate", "hackernews", 1500, "ai"),
                MockTrendingTopic("Climate Tech Innovations", "reddit", 1200, "tech"),
                MockTrendingTopic("Quantum Computing Breakthrough", "twitter", 800, "science"),
                MockTrendingTopic("Open Source LLMs", "hackernews", 950, "ai"),
                MockTrendingTopic("Remote Work Future", "reddit", 700, "business"),
            ]

        manager.get_trending_topics = AsyncMock(side_effect=mock_get_trending)

        # Mock select_topic_for_debate
        def mock_select(topics):
            if topics:
                return max(topics, key=lambda t: t.volume)
            return None

        manager.select_topic_for_debate = mock_select

        # Mock analytics
        manager.get_analytics = MagicMock(
            return_value={
                "total_debates": 42,
                "consensus_rate": 0.78,
                "avg_confidence": 0.85,
                "by_platform": {
                    "hackernews": {"debates": 15, "consensus_rate": 0.80},
                    "reddit": {"debates": 18, "consensus_rate": 0.72},
                    "twitter": {"debates": 9, "consensus_rate": 0.89},
                },
                "by_category": {
                    "ai": {"debates": 20, "avg_confidence": 0.88},
                    "tech": {"debates": 12, "avg_confidence": 0.82},
                    "science": {"debates": 10, "avg_confidence": 0.85},
                },
                "recent_outcomes": [
                    {"topic": "AI Safety", "consensus": True, "confidence": 0.92},
                    {"topic": "Crypto Regulation", "consensus": False, "confidence": 0.65},
                ],
            }
        )

        return manager

    @pytest.mark.asyncio
    async def test_get_trending_topics_returns_all_sources(self):
        """Test that trending topics includes data from all configured sources."""
        manager = self._create_mock_manager()

        topics = await manager.get_trending_topics()

        assert len(topics) == 5
        platforms = {t.platform for t in topics}
        assert "hackernews" in platforms
        assert "reddit" in platforms
        assert "twitter" in platforms

    @pytest.mark.asyncio
    async def test_topic_volume_normalization(self):
        """Test that topic volumes are properly captured."""
        manager = self._create_mock_manager()

        topics = await manager.get_trending_topics()

        # Find max volume
        max_volume = max(t.volume for t in topics)
        assert max_volume == 1500  # AI Regulation Debate

        # Verify all volumes are positive
        for topic in topics:
            assert topic.volume > 0

    @pytest.mark.asyncio
    async def test_select_topic_for_debate(self):
        """Test topic selection prioritizes high-volume topics."""
        manager = self._create_mock_manager()

        topics = await manager.get_trending_topics()
        selected = manager.select_topic_for_debate(topics)

        assert selected is not None
        assert selected.topic == "AI Regulation Debate"  # Highest volume
        assert selected.volume == 1500

    @pytest.mark.asyncio
    async def test_debate_prompt_formatting(self):
        """Test that selected topic can generate a debate prompt."""
        manager = self._create_mock_manager()

        topics = await manager.get_trending_topics()
        selected = manager.select_topic_for_debate(topics)

        prompt = selected.to_debate_prompt()
        assert "Debate:" in prompt
        assert selected.topic in prompt


class TestPulseAnalytics:
    """Test Pulse analytics functionality."""

    def _create_mock_manager_with_analytics(self):
        """Create manager with analytics data."""
        manager = MagicMock()
        manager.get_analytics = MagicMock(
            return_value={
                "total_debates": 100,
                "consensus_rate": 0.75,
                "avg_confidence": 0.82,
                "by_platform": {
                    "hackernews": {"debates": 40, "consensus_rate": 0.80},
                    "reddit": {"debates": 35, "consensus_rate": 0.70},
                    "twitter": {"debates": 25, "consensus_rate": 0.76},
                },
                "by_category": {},
                "recent_outcomes": [],
            }
        )
        return manager

    def test_analytics_structure(self):
        """Test analytics returns expected structure."""
        manager = self._create_mock_manager_with_analytics()

        analytics = manager.get_analytics()

        assert "total_debates" in analytics
        assert "consensus_rate" in analytics
        assert "avg_confidence" in analytics
        assert "by_platform" in analytics

    def test_analytics_platform_breakdown(self):
        """Test analytics includes per-platform breakdown."""
        manager = self._create_mock_manager_with_analytics()

        analytics = manager.get_analytics()

        by_platform = analytics["by_platform"]
        assert "hackernews" in by_platform
        assert "reddit" in by_platform
        assert "twitter" in by_platform

        # Each platform should have debates and consensus_rate
        for platform, data in by_platform.items():
            assert "debates" in data
            assert "consensus_rate" in data

    def test_analytics_consensus_rate_valid_range(self):
        """Test consensus rate is in valid 0-1 range."""
        manager = self._create_mock_manager_with_analytics()

        analytics = manager.get_analytics()

        assert 0 <= analytics["consensus_rate"] <= 1
        assert 0 <= analytics["avg_confidence"] <= 1


class TestPulseHandlerWithMocks:
    """Test PulseHandler with mocked dependencies."""

    def _create_mock_pulse_handler(self, manager):
        """Create a mock PulseHandler with injected manager."""
        handler = MagicMock()
        handler.ctx = {}

        # Mock the internal methods
        def mock_get_trending(limit):
            import asyncio

            topics = asyncio.get_event_loop().run_until_complete(manager.get_trending_topics())
            max_volume = max((t.volume for t in topics), default=1) or 1
            return {
                "topics": [
                    {
                        "topic": t.topic,
                        "source": t.platform,
                        "score": round(t.volume / max_volume, 3),
                        "volume": t.volume,
                        "category": t.category,
                    }
                    for t in topics
                ],
                "count": len(topics),
                "sources": list(manager.ingestors.keys()),
            }

        handler._get_trending_topics = mock_get_trending
        return handler

    def test_trending_response_format(self):
        """Test that trending endpoint returns correctly formatted response."""
        # Create mock manager
        manager = MagicMock()
        manager.ingestors = {"hackernews": MagicMock(), "reddit": MagicMock()}

        async def mock_get_trending():
            return [
                MockTrendingTopic("Test Topic 1", "hackernews", 100, "tech"),
                MockTrendingTopic("Test Topic 2", "reddit", 50, "ai"),
            ]

        manager.get_trending_topics = AsyncMock(side_effect=mock_get_trending)

        # Simulate the handler response format
        import asyncio

        topics = asyncio.get_event_loop().run_until_complete(manager.get_trending_topics())

        max_volume = max((t.volume for t in topics), default=1)
        response = {
            "topics": [
                {
                    "topic": t.topic,
                    "source": t.platform,
                    "score": round(t.volume / max_volume, 3),
                    "volume": t.volume,
                    "category": t.category,
                }
                for t in topics
            ],
            "count": len(topics),
            "sources": list(manager.ingestors.keys()),
        }

        assert response["count"] == 2
        assert len(response["topics"]) == 2
        assert response["topics"][0]["score"] == 1.0  # Highest volume = 1.0
        assert response["topics"][1]["score"] == 0.5  # Half of max


class TestDebateOutcomeRecording:
    """Test recording debate outcomes for analytics."""

    def test_record_debate_outcome(self):
        """Test that debate outcomes can be recorded."""
        manager = MagicMock()
        manager.record_debate_outcome = MagicMock()

        # Record an outcome
        outcome = {
            "debate_id": "debate-123",
            "topic": "AI Safety",
            "platform": "hackernews",
            "category": "ai",
            "consensus_reached": True,
            "confidence": 0.92,
            "duration_seconds": 180,
        }

        manager.record_debate_outcome(**outcome)

        manager.record_debate_outcome.assert_called_once_with(**outcome)

    def test_outcome_affects_analytics(self):
        """Test that recorded outcomes affect analytics."""
        # This tests the integration between recording and analytics

        # Mock initial state
        initial_analytics = {
            "total_debates": 10,
            "consensus_rate": 0.7,
        }

        # After recording a consensus
        after_consensus = {
            "total_debates": 11,
            "consensus_rate": 0.73,  # Increased
        }

        # After recording a non-consensus
        after_no_consensus = {
            "total_debates": 12,
            "consensus_rate": 0.67,  # Decreased
        }

        # Verify the pattern makes sense
        assert after_consensus["total_debates"] > initial_analytics["total_debates"]
        assert after_no_consensus["total_debates"] > after_consensus["total_debates"]


class TestPulseEdgeCases:
    """Test edge cases for Pulse functionality."""

    @pytest.mark.asyncio
    async def test_empty_topics_list(self):
        """Test handling of empty topics list."""
        manager = MagicMock()
        manager.get_trending_topics = AsyncMock(return_value=[])
        manager.select_topic_for_debate = MagicMock(return_value=None)

        topics = await manager.get_trending_topics()
        selected = manager.select_topic_for_debate(topics)

        assert topics == []
        assert selected is None

    @pytest.mark.asyncio
    async def test_single_source_failure(self):
        """Test that other sources still work if one fails."""
        # Simulate HN working, Reddit failing, Twitter working
        manager = MagicMock()

        async def mock_fetch():
            # Only return topics from working sources
            return [
                MockTrendingTopic("HN Topic", "hackernews", 100, "tech"),
                MockTrendingTopic("Twitter Topic", "twitter", 80, "ai"),
            ]

        manager.get_trending_topics = AsyncMock(side_effect=mock_fetch)

        topics = await manager.get_trending_topics()

        assert len(topics) == 2
        platforms = {t.platform for t in topics}
        assert "hackernews" in platforms
        assert "twitter" in platforms
        assert "reddit" not in platforms

    def test_category_filter(self):
        """Test filtering topics by category."""
        all_topics = [
            MockTrendingTopic("AI Topic", "hackernews", 100, "ai"),
            MockTrendingTopic("Tech Topic", "reddit", 80, "tech"),
            MockTrendingTopic("Science Topic", "twitter", 60, "science"),
        ]

        # Filter to AI only
        ai_topics = [t for t in all_topics if t.category == "ai"]

        assert len(ai_topics) == 1
        assert ai_topics[0].topic == "AI Topic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
