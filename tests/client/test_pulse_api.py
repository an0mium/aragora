"""
Tests for Pulse API resource.

Tests cover:
- TrendingTopic dataclass and from_dict factory
- DebateSuggestion dataclass and from_dict factory
- PulseAnalytics dataclass and from_dict factory
- PulseAPI.trending() sync and async
- PulseAPI.suggest() sync and async
- PulseAPI.get_analytics() sync and async
- PulseAPI.refresh() sync and async
- Parameter filtering and default values
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.pulse import (
    DebateSuggestion,
    PulseAnalytics,
    PulseAPI,
    TrendingTopic,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def pulse_api(mock_client: AragoraClient) -> PulseAPI:
    """Create a PulseAPI with mock client."""
    return PulseAPI(mock_client)


# ============================================================================
# TrendingTopic Dataclass Tests
# ============================================================================


class TestTrendingTopicDataclass:
    """Tests for TrendingTopic dataclass."""

    def test_trending_topic_minimal(self):
        """Test TrendingTopic with required fields only."""
        topic = TrendingTopic(
            title="AI Regulation",
            source="hackernews",
            score=0.95,
        )
        assert topic.title == "AI Regulation"
        assert topic.source == "hackernews"
        assert topic.score == 0.95
        assert topic.category == "general"
        assert topic.url is None
        assert topic.summary is None
        assert topic.suggested_agents == []

    def test_trending_topic_full(self):
        """Test TrendingTopic with all fields."""
        topic = TrendingTopic(
            title="Quantum Computing Advances",
            source="arxiv",
            score=0.88,
            category="science",
            url="https://arxiv.org/abs/1234",
            summary="New quantum error correction",
            suggested_agents=["claude", "gemini"],
        )
        assert topic.category == "science"
        assert topic.url == "https://arxiv.org/abs/1234"
        assert topic.summary == "New quantum error correction"
        assert topic.suggested_agents == ["claude", "gemini"]

    def test_trending_topic_from_dict_full(self):
        """Test TrendingTopic.from_dict() with all fields."""
        data = {
            "title": "Rust in Linux",
            "source": "reddit",
            "score": 0.92,
            "category": "technology",
            "url": "https://example.com",
            "summary": "Rust modules accepted upstream",
            "suggested_agents": ["codex", "claude"],
        }
        topic = TrendingTopic.from_dict(data)
        assert topic.title == "Rust in Linux"
        assert topic.source == "reddit"
        assert topic.score == 0.92
        assert topic.category == "technology"
        assert topic.url == "https://example.com"
        assert topic.suggested_agents == ["codex", "claude"]

    def test_trending_topic_from_dict_defaults(self):
        """Test TrendingTopic.from_dict() fills defaults for missing keys."""
        data = {}
        topic = TrendingTopic.from_dict(data)
        assert topic.title == ""
        assert topic.source == "unknown"
        assert topic.score == 0.0
        assert topic.category == "general"
        assert topic.url is None
        assert topic.summary is None
        assert topic.suggested_agents == []

    def test_trending_topic_post_init_none_agents(self):
        """Test that __post_init__ converts None suggested_agents to empty list."""
        topic = TrendingTopic(
            title="Test",
            source="test",
            score=0.5,
            suggested_agents=None,
        )
        assert topic.suggested_agents == []


# ============================================================================
# DebateSuggestion Dataclass Tests
# ============================================================================


class TestDebateSuggestionDataclass:
    """Tests for DebateSuggestion dataclass."""

    def test_debate_suggestion_minimal(self):
        """Test DebateSuggestion with required fields only."""
        suggestion = DebateSuggestion(
            topic="Should AI be regulated?",
            rationale="Growing societal impact",
        )
        assert suggestion.topic == "Should AI be regulated?"
        assert suggestion.rationale == "Growing societal impact"
        assert suggestion.difficulty == "medium"
        assert suggestion.estimated_rounds == 3
        assert suggestion.suggested_agents == []
        assert suggestion.related_topics == []

    def test_debate_suggestion_full(self):
        """Test DebateSuggestion with all fields."""
        suggestion = DebateSuggestion(
            topic="Microservices vs Monolith",
            rationale="Common architecture debate",
            difficulty="hard",
            estimated_rounds=5,
            suggested_agents=["claude", "grok"],
            related_topics=["scalability", "deployment"],
        )
        assert suggestion.difficulty == "hard"
        assert suggestion.estimated_rounds == 5
        assert suggestion.suggested_agents == ["claude", "grok"]
        assert suggestion.related_topics == ["scalability", "deployment"]

    def test_debate_suggestion_from_dict_full(self):
        """Test DebateSuggestion.from_dict() with all fields."""
        data = {
            "topic": "GraphQL vs REST",
            "rationale": "API design trends",
            "difficulty": "easy",
            "estimated_rounds": 2,
            "suggested_agents": ["gemini"],
            "related_topics": ["API design", "performance"],
        }
        suggestion = DebateSuggestion.from_dict(data)
        assert suggestion.topic == "GraphQL vs REST"
        assert suggestion.difficulty == "easy"
        assert suggestion.estimated_rounds == 2

    def test_debate_suggestion_from_dict_defaults(self):
        """Test DebateSuggestion.from_dict() fills defaults for missing keys."""
        data = {}
        suggestion = DebateSuggestion.from_dict(data)
        assert suggestion.topic == ""
        assert suggestion.rationale == ""
        assert suggestion.difficulty == "medium"
        assert suggestion.estimated_rounds == 3
        assert suggestion.suggested_agents == []
        assert suggestion.related_topics == []

    def test_debate_suggestion_post_init_none_lists(self):
        """Test that __post_init__ converts None lists to empty lists."""
        suggestion = DebateSuggestion(
            topic="Test",
            rationale="Test",
            suggested_agents=None,
            related_topics=None,
        )
        assert suggestion.suggested_agents == []
        assert suggestion.related_topics == []


# ============================================================================
# PulseAnalytics Dataclass Tests
# ============================================================================


class TestPulseAnalyticsDataclass:
    """Tests for PulseAnalytics dataclass."""

    def test_pulse_analytics(self):
        """Test PulseAnalytics with all fields."""
        analytics = PulseAnalytics(
            total_topics=150,
            by_source={"hackernews": 60, "reddit": 50, "arxiv": 40},
            by_category={"technology": 80, "science": 40, "business": 30},
            top_categories=["technology", "science", "business"],
            freshness_hours=12.0,
        )
        assert analytics.total_topics == 150
        assert analytics.by_source["hackernews"] == 60
        assert analytics.by_category["technology"] == 80
        assert analytics.top_categories[0] == "technology"
        assert analytics.freshness_hours == 12.0

    def test_pulse_analytics_from_dict_full(self):
        """Test PulseAnalytics.from_dict() with all fields."""
        data = {
            "total_topics": 200,
            "by_source": {"hackernews": 100, "reddit": 100},
            "by_category": {"tech": 120, "sci": 80},
            "top_categories": ["tech", "sci"],
            "freshness_hours": 6.0,
        }
        analytics = PulseAnalytics.from_dict(data)
        assert analytics.total_topics == 200
        assert analytics.freshness_hours == 6.0

    def test_pulse_analytics_from_dict_defaults(self):
        """Test PulseAnalytics.from_dict() fills defaults for missing keys."""
        data = {}
        analytics = PulseAnalytics.from_dict(data)
        assert analytics.total_topics == 0
        assert analytics.by_source == {}
        assert analytics.by_category == {}
        assert analytics.top_categories == []
        assert analytics.freshness_hours == 24.0


# ============================================================================
# PulseAPI.trending() Tests
# ============================================================================


class TestPulseAPITrending:
    """Tests for PulseAPI.trending() method."""

    def test_trending_basic(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() returns a list of TrendingTopic objects."""
        mock_client._get.return_value = {
            "topics": [
                {
                    "title": "AI Safety",
                    "source": "hackernews",
                    "score": 0.95,
                    "category": "technology",
                },
                {
                    "title": "Climate Change",
                    "source": "reddit",
                    "score": 0.85,
                    "category": "science",
                },
            ]
        }

        results = pulse_api.trending()

        assert len(results) == 2
        assert isinstance(results[0], TrendingTopic)
        assert results[0].title == "AI Safety"
        assert results[0].score == 0.95
        assert results[1].title == "Climate Change"
        mock_client._get.assert_called_once_with(
            "/api/pulse/trending", params={"limit": 20}
        )

    def test_trending_with_category(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() passes category filter."""
        mock_client._get.return_value = {"topics": []}

        pulse_api.trending(category="technology")

        mock_client._get.assert_called_once_with(
            "/api/pulse/trending",
            params={"limit": 20, "category": "technology"},
        )

    def test_trending_with_source(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() passes source filter."""
        mock_client._get.return_value = {"topics": []}

        pulse_api.trending(source="arxiv")

        mock_client._get.assert_called_once_with(
            "/api/pulse/trending",
            params={"limit": 20, "source": "arxiv"},
        )

    def test_trending_with_all_params(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() passes all parameters."""
        mock_client._get.return_value = {"topics": []}

        pulse_api.trending(category="science", source="reddit", limit=10)

        mock_client._get.assert_called_once_with(
            "/api/pulse/trending",
            params={"limit": 10, "category": "science", "source": "reddit"},
        )

    def test_trending_empty_response(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() handles empty topic list."""
        mock_client._get.return_value = {"topics": []}

        results = pulse_api.trending()

        assert results == []

    def test_trending_missing_topics_key(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() handles missing 'topics' key gracefully."""
        mock_client._get.return_value = {}

        results = pulse_api.trending()

        assert results == []


class TestPulseAPITrendingAsync:
    """Tests for PulseAPI.trending_async() method."""

    @pytest.mark.asyncio
    async def test_trending_async_basic(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending_async() returns TrendingTopic list."""
        mock_client._get_async = AsyncMock(
            return_value={
                "topics": [
                    {
                        "title": "Async Topic",
                        "source": "hackernews",
                        "score": 0.9,
                        "category": "technology",
                    }
                ]
            }
        )

        results = await pulse_api.trending_async()

        assert len(results) == 1
        assert isinstance(results[0], TrendingTopic)
        assert results[0].title == "Async Topic"

    @pytest.mark.asyncio
    async def test_trending_async_with_params(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending_async() passes parameters correctly."""
        mock_client._get_async = AsyncMock(return_value={"topics": []})

        await pulse_api.trending_async(category="business", source="hackernews", limit=5)

        mock_client._get_async.assert_called_once_with(
            "/api/pulse/trending",
            params={"limit": 5, "category": "business", "source": "hackernews"},
        )


# ============================================================================
# PulseAPI.suggest() Tests
# ============================================================================


class TestPulseAPISuggest:
    """Tests for PulseAPI.suggest() method."""

    def test_suggest_basic(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() returns DebateSuggestion list."""
        mock_client._get.return_value = {
            "suggestions": [
                {
                    "topic": "Should we use Kubernetes?",
                    "rationale": "Common infrastructure decision",
                    "difficulty": "hard",
                    "estimated_rounds": 4,
                    "suggested_agents": ["claude", "codex"],
                    "related_topics": ["containers", "orchestration"],
                },
            ]
        }

        results = pulse_api.suggest()

        assert len(results) == 1
        assert isinstance(results[0], DebateSuggestion)
        assert results[0].topic == "Should we use Kubernetes?"
        assert results[0].difficulty == "hard"
        assert results[0].estimated_rounds == 4
        assert results[0].suggested_agents == ["claude", "codex"]
        mock_client._get.assert_called_once_with(
            "/api/pulse/suggest", params={"count": 5}
        )

    def test_suggest_with_domain(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() passes domain filter."""
        mock_client._get.return_value = {"suggestions": []}

        pulse_api.suggest(domain="security")

        mock_client._get.assert_called_once_with(
            "/api/pulse/suggest",
            params={"count": 5, "domain": "security"},
        )

    def test_suggest_with_difficulty(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() passes difficulty filter."""
        mock_client._get.return_value = {"suggestions": []}

        pulse_api.suggest(difficulty="easy")

        mock_client._get.assert_called_once_with(
            "/api/pulse/suggest",
            params={"count": 5, "difficulty": "easy"},
        )

    def test_suggest_with_all_params(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() passes all parameters."""
        mock_client._get.return_value = {"suggestions": []}

        pulse_api.suggest(domain="architecture", difficulty="medium", count=10)

        mock_client._get.assert_called_once_with(
            "/api/pulse/suggest",
            params={"count": 10, "domain": "architecture", "difficulty": "medium"},
        )

    def test_suggest_empty_response(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() handles empty suggestions."""
        mock_client._get.return_value = {"suggestions": []}

        results = pulse_api.suggest()

        assert results == []

    def test_suggest_missing_key(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest() handles missing 'suggestions' key gracefully."""
        mock_client._get.return_value = {}

        results = pulse_api.suggest()

        assert results == []


class TestPulseAPISuggestAsync:
    """Tests for PulseAPI.suggest_async() method."""

    @pytest.mark.asyncio
    async def test_suggest_async_basic(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest_async() returns DebateSuggestion list."""
        mock_client._get_async = AsyncMock(
            return_value={
                "suggestions": [
                    {
                        "topic": "Async Suggestion",
                        "rationale": "Testing async path",
                        "difficulty": "easy",
                    }
                ]
            }
        )

        results = await pulse_api.suggest_async()

        assert len(results) == 1
        assert isinstance(results[0], DebateSuggestion)
        assert results[0].topic == "Async Suggestion"

    @pytest.mark.asyncio
    async def test_suggest_async_with_params(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test suggest_async() passes parameters correctly."""
        mock_client._get_async = AsyncMock(return_value={"suggestions": []})

        await pulse_api.suggest_async(domain="security", difficulty="hard", count=3)

        mock_client._get_async.assert_called_once_with(
            "/api/pulse/suggest",
            params={"count": 3, "domain": "security", "difficulty": "hard"},
        )


# ============================================================================
# PulseAPI.get_analytics() Tests
# ============================================================================


class TestPulseAPIGetAnalytics:
    """Tests for PulseAPI.get_analytics() method."""

    def test_get_analytics(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test get_analytics() returns PulseAnalytics."""
        mock_client._get.return_value = {
            "total_topics": 300,
            "by_source": {"hackernews": 120, "reddit": 100, "arxiv": 80},
            "by_category": {"technology": 150, "science": 100, "business": 50},
            "top_categories": ["technology", "science", "business"],
            "freshness_hours": 8.5,
        }

        result = pulse_api.get_analytics()

        assert isinstance(result, PulseAnalytics)
        assert result.total_topics == 300
        assert result.by_source["hackernews"] == 120
        assert result.top_categories == ["technology", "science", "business"]
        assert result.freshness_hours == 8.5
        mock_client._get.assert_called_once_with("/api/pulse/analytics")

    def test_get_analytics_sparse_response(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test get_analytics() handles sparse response with defaults."""
        mock_client._get.return_value = {}

        result = pulse_api.get_analytics()

        assert isinstance(result, PulseAnalytics)
        assert result.total_topics == 0
        assert result.by_source == {}
        assert result.by_category == {}
        assert result.top_categories == []
        assert result.freshness_hours == 24.0


class TestPulseAPIGetAnalyticsAsync:
    """Tests for PulseAPI.get_analytics_async() method."""

    @pytest.mark.asyncio
    async def test_get_analytics_async(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test get_analytics_async() returns PulseAnalytics."""
        mock_client._get_async = AsyncMock(
            return_value={
                "total_topics": 50,
                "by_source": {"hackernews": 50},
                "by_category": {"tech": 50},
                "top_categories": ["tech"],
                "freshness_hours": 4.0,
            }
        )

        result = await pulse_api.get_analytics_async()

        assert isinstance(result, PulseAnalytics)
        assert result.total_topics == 50
        assert result.freshness_hours == 4.0
        mock_client._get_async.assert_called_once_with("/api/pulse/analytics")


# ============================================================================
# PulseAPI.refresh() Tests
# ============================================================================


class TestPulseAPIRefresh:
    """Tests for PulseAPI.refresh() method."""

    def test_refresh_no_sources(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh() without specific sources."""
        mock_client._post.return_value = {"refreshed": True}

        result = pulse_api.refresh()

        assert result is True
        mock_client._post.assert_called_once_with("/api/pulse/refresh", {})

    def test_refresh_with_sources(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh() with specific sources."""
        mock_client._post.return_value = {"refreshed": True}

        result = pulse_api.refresh(sources=["hackernews", "reddit"])

        assert result is True
        mock_client._post.assert_called_once_with(
            "/api/pulse/refresh",
            {"sources": ["hackernews", "reddit"]},
        )

    def test_refresh_returns_false(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh() returns False when not refreshed."""
        mock_client._post.return_value = {"refreshed": False}

        result = pulse_api.refresh()

        assert result is False

    def test_refresh_missing_key(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh() returns False when key is missing."""
        mock_client._post.return_value = {}

        result = pulse_api.refresh()

        assert result is False


class TestPulseAPIRefreshAsync:
    """Tests for PulseAPI.refresh_async() method."""

    @pytest.mark.asyncio
    async def test_refresh_async_no_sources(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh_async() without sources."""
        mock_client._post_async = AsyncMock(return_value={"refreshed": True})

        result = await pulse_api.refresh_async()

        assert result is True
        mock_client._post_async.assert_called_once_with("/api/pulse/refresh", {})

    @pytest.mark.asyncio
    async def test_refresh_async_with_sources(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh_async() with specific sources."""
        mock_client._post_async = AsyncMock(return_value={"refreshed": True})

        result = await pulse_api.refresh_async(sources=["arxiv"])

        assert result is True
        mock_client._post_async.assert_called_once_with(
            "/api/pulse/refresh",
            {"sources": ["arxiv"]},
        )

    @pytest.mark.asyncio
    async def test_refresh_async_returns_false(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test refresh_async() returns False when not refreshed."""
        mock_client._post_async = AsyncMock(return_value={"refreshed": False})

        result = await pulse_api.refresh_async()

        assert result is False


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestPulseAPIIntegration:
    """Integration-like tests for PulseAPI."""

    def test_trending_to_suggestion_workflow(
        self, pulse_api: PulseAPI, mock_client: MagicMock
    ):
        """Test workflow: get trending -> get suggestions for related domain."""
        # Step 1: Get trending topics
        mock_client._get.return_value = {
            "topics": [
                {
                    "title": "Zero Trust Architecture",
                    "source": "hackernews",
                    "score": 0.97,
                    "category": "security",
                },
            ]
        }
        trending = pulse_api.trending(category="security")
        assert len(trending) == 1
        assert trending[0].category == "security"

        # Step 2: Get suggestions for the same domain
        mock_client._get.return_value = {
            "suggestions": [
                {
                    "topic": "Is Zero Trust practical for SMEs?",
                    "rationale": "Trending topic in security",
                    "difficulty": "medium",
                    "related_topics": ["Zero Trust Architecture"],
                },
            ]
        }
        suggestions = pulse_api.suggest(domain="security")
        assert len(suggestions) == 1
        assert "Zero Trust" in suggestions[0].topic

    def test_refresh_then_analytics_workflow(
        self, pulse_api: PulseAPI, mock_client: MagicMock
    ):
        """Test workflow: refresh sources -> check analytics."""
        # Step 1: Refresh
        mock_client._post.return_value = {"refreshed": True}
        refreshed = pulse_api.refresh(sources=["hackernews", "reddit"])
        assert refreshed is True

        # Step 2: Check analytics
        mock_client._get.return_value = {
            "total_topics": 250,
            "by_source": {"hackernews": 130, "reddit": 120},
            "by_category": {"technology": 200, "science": 50},
            "top_categories": ["technology", "science"],
            "freshness_hours": 1.0,
        }
        analytics = pulse_api.get_analytics()
        assert analytics.total_topics == 250
        assert analytics.freshness_hours == 1.0

    def test_multiple_trending_results(self, pulse_api: PulseAPI, mock_client: MagicMock):
        """Test trending() correctly deserializes multiple topics."""
        mock_client._get.return_value = {
            "topics": [
                {"title": f"Topic {i}", "source": "test", "score": 1.0 - i * 0.1}
                for i in range(5)
            ]
        }

        results = pulse_api.trending(limit=5)

        assert len(results) == 5
        for i, topic in enumerate(results):
            assert isinstance(topic, TrendingTopic)
            assert topic.title == f"Topic {i}"
            assert topic.score == pytest.approx(1.0 - i * 0.1)
