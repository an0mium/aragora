"""Tests for MCP trending tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.trending import list_trending_topics_tool

pytest.importorskip("mcp")


class TestListTrendingTopicsTool:
    """Tests for list_trending_topics_tool."""

    @pytest.mark.asyncio
    async def test_list_success(self):
        """Test successful trending topics listing."""
        mock_topic = MagicMock()
        mock_topic.topic = "AI Safety Research"
        mock_topic.platform = "hackernews"
        mock_topic.category = "technology"
        mock_topic.volume = 150

        mock_score = MagicMock()
        mock_score.score = 0.85

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic]

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool()

        assert result["count"] == 1
        assert len(result["topics"]) == 1
        assert result["topics"][0]["topic"] == "AI Safety Research"
        assert result["topics"][0]["score"] == 0.85
        assert result["topics"][0]["debate_potential"] == "high"

    @pytest.mark.asyncio
    async def test_list_with_platform_filter(self):
        """Test trending topics with platform filter."""
        mock_topic_hn = MagicMock()
        mock_topic_hn.topic = "HN Topic"
        mock_topic_hn.platform = "hackernews"
        mock_topic_hn.category = ""
        mock_topic_hn.volume = 100

        mock_topic_reddit = MagicMock()
        mock_topic_reddit.topic = "Reddit Topic"
        mock_topic_reddit.platform = "reddit"
        mock_topic_reddit.category = ""
        mock_topic_reddit.volume = 50

        mock_score = MagicMock()
        mock_score.score = 0.7

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic_hn, mock_topic_reddit]

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool(platform="hackernews")

        assert result["platform"] == "hackernews"
        # Only HN topics should be returned
        assert all(t["platform"] == "hackernews" for t in result["topics"])

    @pytest.mark.asyncio
    async def test_list_with_min_score_filter(self):
        """Test trending topics filtered by minimum score."""
        mock_topic = MagicMock()
        mock_topic.topic = "Low Score Topic"
        mock_topic.platform = "hackernews"
        mock_topic.category = ""
        mock_topic.volume = 10

        mock_score = MagicMock()
        mock_score.score = 0.3  # Below threshold

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic]

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool(min_score=0.5)

        # Topic should be filtered out due to low score
        assert result["count"] == 0
        assert result["min_score"] == 0.5

    @pytest.mark.asyncio
    async def test_list_with_category_filter(self):
        """Test trending topics with category filter."""
        mock_topic_tech = MagicMock()
        mock_topic_tech.topic = "Tech Topic"
        mock_topic_tech.platform = "hackernews"
        mock_topic_tech.category = "technology"
        mock_topic_tech.volume = 100

        mock_topic_science = MagicMock()
        mock_topic_science.topic = "Science Topic"
        mock_topic_science.platform = "hackernews"
        mock_topic_science.category = "science"
        mock_topic_science.volume = 80

        mock_score = MagicMock()
        mock_score.score = 0.7

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic_tech, mock_topic_science]

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool(category="technology")

        assert result["category"] == "technology"
        assert all(t["category"] == "technology" for t in result["topics"])

    @pytest.mark.asyncio
    async def test_list_debate_potential_medium(self):
        """Test debate potential is medium for lower scores."""
        mock_topic = MagicMock()
        mock_topic.topic = "Medium Topic"
        mock_topic.platform = "reddit"
        mock_topic.category = ""
        mock_topic.volume = 50

        mock_score = MagicMock()
        mock_score.score = 0.6  # Between 0.5 and 0.7

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic]

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool()

        assert result["topics"][0]["debate_potential"] == "medium"

    @pytest.mark.asyncio
    async def test_list_respects_limit(self):
        """Test that results are limited."""
        mock_topics = []
        for i in range(20):
            topic = MagicMock()
            topic.topic = f"Topic {i}"
            topic.platform = "hackernews"
            topic.category = ""
            topic.volume = 100 - i
            mock_topics.append(topic)

        mock_score = MagicMock()
        mock_score.score = 0.8

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = mock_topics

        mock_selector = MagicMock()
        mock_selector.score_topic.return_value = mock_score

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool(limit=5)

        assert len(result["topics"]) == 5

    @pytest.mark.asyncio
    async def test_list_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        mock_topic1 = MagicMock()
        mock_topic1.topic = "Topic 1"
        mock_topic1.platform = "hackernews"
        mock_topic1.category = ""
        mock_topic1.volume = 100

        mock_topic2 = MagicMock()
        mock_topic2.topic = "Topic 2"
        mock_topic2.platform = "hackernews"
        mock_topic2.category = ""
        mock_topic2.volume = 80

        scores = [0.6, 0.9]  # Topic 2 has higher score

        mock_pulse = AsyncMock()
        mock_pulse.get_trending_topics.return_value = [mock_topic1, mock_topic2]

        mock_selector = MagicMock()
        mock_selector.score_topic.side_effect = [MagicMock(score=s) for s in scores]

        with (
            patch(
                "aragora.pulse.PulseManager",
                return_value=mock_pulse,
            ),
            patch(
                "aragora.pulse.SchedulerConfig",
            ),
            patch(
                "aragora.pulse.TopicSelector",
                return_value=mock_selector,
            ),
        ):
            result = await list_trending_topics_tool()

        # Higher score should be first
        assert result["topics"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_list_import_error(self):
        """Test graceful handling when pulse not available."""
        with patch(
            "aragora.pulse.PulseManager",
            side_effect=ImportError("Not installed"),
        ):
            result = await list_trending_topics_tool()

        assert result["count"] == 0
        assert result["topics"] == []
