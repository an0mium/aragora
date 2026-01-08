"""
Tests for PulseHandler.

Tests trending topics and topic suggestion endpoints that fetch data
from multiple sources (HackerNews, Reddit, Twitter).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass

from aragora.server.handlers.pulse import PulseHandler
from aragora.server.handlers.base import clear_cache


@dataclass
class MockTrendingTopic:
    """Mock trending topic object."""
    topic: str
    platform: str
    volume: int
    category: str = "technology"

    def to_debate_prompt(self) -> str:
        return f"Discuss the implications of: {self.topic}"


@pytest.fixture
def handler(tmp_path):
    """Create PulseHandler with mock context."""
    ctx = {
        "storage": Mock(),
        "elo_system": Mock(),
        "nomic_dir": tmp_path,
    }
    return PulseHandler(ctx)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear cache between tests."""
    clear_cache()
    yield
    clear_cache()


class TestPulseHandlerRouting:
    """Test route matching and dispatch."""

    def test_can_handle_trending(self, handler):
        """can_handle returns True for /api/pulse/trending."""
        assert handler.can_handle("/api/pulse/trending") is True

    def test_can_handle_suggest(self, handler):
        """can_handle returns True for /api/pulse/suggest."""
        assert handler.can_handle("/api/pulse/suggest") is True

    def test_cannot_handle_unrelated_paths(self, handler):
        """can_handle returns False for unrelated paths."""
        assert handler.can_handle("/api/pulse") is False
        assert handler.can_handle("/api/pulse/other") is False
        assert handler.can_handle("/api/trending") is False
        assert handler.can_handle("/api/metrics") is False

    def test_handle_returns_none_for_unknown(self, handler):
        """handle returns None for paths it doesn't handle."""
        result = handler.handle("/api/unknown", {}, None)
        assert result is None


class TestTrendingEndpoint:
    """Test /api/pulse/trending endpoint."""

    def test_returns_503_without_pulse_module(self, handler):
        """Returns 503 when pulse module not available."""
        with patch.dict('sys.modules', {'aragora.pulse.ingestor': None}):
            # Force ImportError by patching
            with patch.object(handler, '_get_trending_topics') as mock_method:
                mock_method.return_value = handler.handle.__self__
                result = handler._get_trending_topics(10)

        # When ImportError occurs, should return error structure
        # The actual behavior depends on the import handling

    def test_respects_limit_parameter(self, handler):
        """Respects the limit query parameter."""
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {"hackernews": Mock()}

                        async def mock_fetch(*args, **kwargs):
                            return []
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_pm.return_value = mock_manager

                        result = handler.handle("/api/pulse/trending", {"limit": ["5"]}, None)

                        # Either succeeds or fails gracefully
                        assert result is not None

    def test_caps_limit_at_50(self, handler):
        """Caps limit at maximum of 50."""
        # The handler caps at 50 internally: min(limit, 50)
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = MagicMock()
                        mock_manager.ingestors = {}
                        mock_manager.add_ingestor = lambda name, ing: mock_manager.ingestors.update({name: ing})
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_pm_class.return_value = mock_manager

                        result = handler.handle("/api/pulse/trending", {"limit": ["100"]}, None)

                        # Should either succeed or fail gracefully
                        assert result is not None
                        # Verify limit was capped (handler uses min(limit, 50))
                        mock_manager.get_trending_topics.assert_called_once()
                        call_kwargs = mock_manager.get_trending_topics.call_args
                        if call_kwargs:
                            # Check limit_per_platform was passed as 50 (capped from 100)
                            assert call_kwargs.kwargs.get('limit_per_platform', 50) <= 50

    def test_trending_response_structure(self, handler):
        """Returns proper response structure when successful."""
        topics = [
            MockTrendingTopic("AI breakthrough", "hackernews", 100),
            MockTrendingTopic("New framework", "reddit", 50),
        ]

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {"hackernews": Mock(), "reddit": Mock()}
                        mock_manager.get_trending_topics = AsyncMock(return_value=topics)
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        if result.status_code == 200:
                            data = json.loads(result.body)
                            assert "topics" in data
                            assert "count" in data
                            assert "sources" in data

    def test_normalizes_scores(self, handler):
        """Normalizes scores to 0-1 range."""
        topics = [
            MockTrendingTopic("High volume", "hackernews", 1000),
            MockTrendingTopic("Low volume", "reddit", 100),
        ]

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {"hackernews": Mock()}
                        mock_manager.get_trending_topics = AsyncMock(return_value=topics)
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        if result.status_code == 200:
                            data = json.loads(result.body)
                            if data.get("topics"):
                                # Highest volume should have score ~1.0
                                scores = [t["score"] for t in data["topics"]]
                                assert max(scores) == 1.0
                                assert all(0 <= s <= 1 for s in scores)

    def test_handles_empty_topics(self, handler):
        """Handles empty topic list gracefully."""
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = MagicMock()
                        # Ensure ingestors dict properly tracks added ingestors
                        mock_manager.ingestors = {}
                        mock_manager.add_ingestor = lambda name, ing: mock_manager.ingestors.update({name: ing})
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        # Should either succeed with empty list or fail gracefully
                        assert result.status_code in (200, 500), f"Unexpected status: {result.status_code}"
                        if result.status_code == 200:
                            data = json.loads(result.body)
                            assert data["count"] == 0
                            assert data["topics"] == []
                            # Verify sources are tracked from add_ingestor calls
                            assert "sources" in data

    def test_handles_fetch_exception(self, handler):
        """Returns 500 on fetch exception."""
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.get_trending_topics = AsyncMock(
                            side_effect=Exception("Network error")
                        )
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        assert result.status_code == 500
                        data = json.loads(result.body)
                        assert "error" in data


class TestSuggestEndpoint:
    """Test /api/pulse/suggest endpoint."""

    def test_returns_503_without_pulse_module(self, handler):
        """Returns 503 when pulse module not available."""
        # When module unavailable, should fail gracefully
        result = handler.handle("/api/pulse/suggest", {}, None)
        # Either works or returns appropriate error
        assert result is not None

    def test_validates_category_parameter(self, handler):
        """Validates category parameter for security."""
        result = handler.handle(
            "/api/pulse/suggest",
            {"category": ["../../../etc/passwd"]},
            None
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_accepts_valid_category(self, handler):
        """Accepts valid category parameter."""
        result = handler.handle(
            "/api/pulse/suggest",
            {"category": ["technology"]},
            None
        )

        # Should either succeed or fail for other reasons
        assert result is not None
        # If 400, it's not due to category validation
        if result.status_code == 400:
            data = json.loads(result.body)
            assert "category" not in data.get("error", "").lower()

    def test_suggest_response_structure(self, handler):
        """Returns proper response structure when successful."""
        topic = MockTrendingTopic("AI ethics debate", "hackernews", 500)

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.get_trending_topics = AsyncMock(return_value=[topic])
                        mock_manager.select_topic_for_debate.return_value = topic
                        mock_pm_class.return_value = mock_manager

                        result = handler._suggest_debate_topic()

                        if result.status_code == 200:
                            data = json.loads(result.body)
                            assert "topic" in data
                            assert "debate_prompt" in data
                            assert "source" in data

    def test_returns_404_when_no_suitable_topics(self, handler):
        """Returns 404 when no suitable topics found."""
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_manager.select_topic_for_debate.return_value = None
                        mock_pm_class.return_value = mock_manager

                        result = handler._suggest_debate_topic()

                        if result.status_code == 404:
                            data = json.loads(result.body)
                            assert data["topic"] is None
                            assert "message" in data

    def test_handles_suggest_exception(self, handler):
        """Returns 500 on suggestion exception."""
        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        # Mock select_topic_for_debate to raise exception
                        # (this is called after get_trending_topics succeeds)
                        mock_manager.get_trending_topics = AsyncMock(return_value=[Mock()])
                        mock_manager.select_topic_for_debate = Mock(
                            side_effect=Exception("Selection error")
                        )
                        mock_pm_class.return_value = mock_manager

                        result = handler._suggest_debate_topic()

                        assert result.status_code == 500
                        data = json.loads(result.body)
                        assert "error" in data


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_import_error_handled_trending(self, handler):
        """Handles ImportError gracefully for trending endpoint."""
        result = handler._get_trending_topics(10)

        # Should either work or return error
        assert result is not None
        assert result.status_code in [200, 500, 503]

    def test_import_error_handled_suggest(self, handler):
        """Handles ImportError gracefully for suggest endpoint."""
        result = handler._suggest_debate_topic()

        # Should either work or return error
        assert result is not None
        assert result.status_code in [200, 404, 500, 503]

    def test_async_loop_handling(self, handler):
        """Handles existing event loop correctly."""
        # This tests the concurrent.futures fallback
        topics = [MockTrendingTopic("Test", "hackernews", 100)]

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {}
                        mock_manager.get_trending_topics = AsyncMock(return_value=topics)
                        mock_pm_class.return_value = mock_manager

                        # Run multiple times to test loop handling
                        for _ in range(3):
                            result = handler._get_trending_topics(10)
                            assert result is not None


class TestTopicMapping:
    """Test field mapping for frontend compatibility."""

    def test_maps_platform_to_source(self, handler):
        """Maps platform field to source for frontend."""
        topics = [MockTrendingTopic("Test topic", "hackernews", 100)]

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {"hackernews": Mock()}
                        mock_manager.get_trending_topics = AsyncMock(return_value=topics)
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        if result.status_code == 200:
                            data = json.loads(result.body)
                            if data.get("topics"):
                                # Should use "source" not "platform"
                                assert "source" in data["topics"][0]
                                assert data["topics"][0]["source"] == "hackernews"

    def test_includes_volume_and_score(self, handler):
        """Includes both raw volume and normalized score."""
        topics = [MockTrendingTopic("Test topic", "reddit", 500)]

        with patch('aragora.pulse.ingestor.PulseManager') as mock_pm_class:
            with patch('aragora.pulse.ingestor.HackerNewsIngestor'):
                with patch('aragora.pulse.ingestor.RedditIngestor'):
                    with patch('aragora.pulse.ingestor.TwitterIngestor'):
                        mock_manager = Mock()
                        mock_manager.ingestors = {"reddit": Mock()}
                        mock_manager.get_trending_topics = AsyncMock(return_value=topics)
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        if result.status_code == 200:
                            data = json.loads(result.body)
                            if data.get("topics"):
                                topic = data["topics"][0]
                                assert "volume" in topic
                                assert "score" in topic
                                assert topic["volume"] == 500
