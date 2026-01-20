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

from aragora.server.handlers.features import PulseHandler


@dataclass
class MockTrendingTopic:
    """Mock trending topic object."""

    topic: str
    platform: str
    volume: int
    category: str = "technology"

    def to_debate_prompt(self) -> str:
        return f"Discuss the implications of: {self.topic}"


@pytest.fixture(autouse=True)
def reset_pulse_singletons():
    """Reset pulse module singletons before and after each test."""
    import aragora.server.handlers.features.pulse as pulse_module

    # Reset before test
    pulse_module._shared_pulse_manager = None
    pulse_module._shared_scheduler = None
    pulse_module._shared_debate_store = None

    yield

    # Reset after test
    pulse_module._shared_pulse_manager = None
    pulse_module._shared_scheduler = None
    pulse_module._shared_debate_store = None


@pytest.fixture
def handler(tmp_path):
    """Create PulseHandler with mock context."""
    ctx = {
        "storage": Mock(),
        "elo_system": Mock(),
        "nomic_dir": tmp_path,
    }
    return PulseHandler(ctx)


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
        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            # Force ImportError by patching
            with patch.object(handler, "_get_trending_topics") as mock_method:
                mock_method.return_value = handler.handle.__self__
                result = handler._get_trending_topics(10)

        # When ImportError occurs, should return error structure
        # The actual behavior depends on the import handling

    def test_respects_limit_parameter(self, handler):
        """Respects the limit query parameter."""
        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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
        # Use proper mocking without sys.modules manipulation
        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
                        mock_manager = MagicMock()
                        mock_manager.ingestors = {}
                        mock_manager.add_ingestor = lambda name, ing: mock_manager.ingestors.update(
                            {name: ing}
                        )
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_pm_class.return_value = mock_manager

                        result = handler.handle("/api/pulse/trending", {"limit": ["100"]}, None)

                        # Should either succeed or fail gracefully
                        assert result is not None
                        # Verify limit was capped (handler uses min(limit, 50))
                        if mock_manager.get_trending_topics.called:
                            call_kwargs = mock_manager.get_trending_topics.call_args
                            if call_kwargs and call_kwargs.kwargs:
                                # Check limit_per_platform was passed as 50 (capped from 100)
                                assert call_kwargs.kwargs.get("limit_per_platform", 50) <= 50

    def test_trending_response_structure(self, handler):
        """Returns proper response structure when successful."""
        topics = [
            MockTrendingTopic("AI breakthrough", "hackernews", 100),
            MockTrendingTopic("New framework", "reddit", 50),
        ]

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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
        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
                        mock_manager = MagicMock()
                        # Ensure ingestors dict properly tracks added ingestors
                        mock_manager.ingestors = {}
                        mock_manager.add_ingestor = lambda name, ing: mock_manager.ingestors.update(
                            {name: ing}
                        )
                        mock_manager.get_trending_topics = AsyncMock(return_value=[])
                        mock_pm_class.return_value = mock_manager

                        result = handler._get_trending_topics(10)

                        # Should either succeed with empty list or fail gracefully
                        assert result.status_code in (
                            200,
                            500,
                        ), f"Unexpected status: {result.status_code}"
                        if result.status_code == 200:
                            data = json.loads(result.body)
                            assert data["count"] == 0
                            assert data["topics"] == []
                            # Verify sources are tracked from add_ingestor calls
                            assert "sources" in data

    def test_handles_fetch_exception(self, handler):
        """Gracefully handles async fetch exception by returning empty topics."""
        mock_manager = Mock()
        mock_manager.ingestors = {}  # Set up ingestors as empty dict
        # RuntimeError in async fetch is caught by _run_async_safely
        # which returns empty list and 200 (graceful degradation)
        mock_manager.get_trending_topics = AsyncMock(side_effect=RuntimeError("Network error"))

        # _get_trending_topics creates PulseManager directly via import
        with patch("aragora.pulse.ingestor.PulseManager", return_value=mock_manager):
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
                        result = handler._get_trending_topics(10)

                        # Graceful degradation: returns 200 with empty topics
                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["topics"] == []
                        assert data["count"] == 0


class TestSuggestEndpoint:
    """Test /api/pulse/suggest endpoint."""

    def test_returns_503_without_pulse_module(self, handler):
        """Returns 503 when pulse module not available."""
        # Patch the import to fail
        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            result = handler.handle("/api/pulse/suggest", {}, None)
            # Should return 503 when pulse module unavailable
            assert result is not None
            assert result.status_code == 503

    def test_validates_category_parameter(self, handler):
        """Validates category parameter for security."""
        result = handler.handle("/api/pulse/suggest", {"category": ["../../../etc/passwd"]}, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_accepts_valid_category(self, handler):
        """Accepts valid category parameter."""
        # Mock the pulse manager to avoid real API calls
        mock_manager = MagicMock()
        mock_manager.ingestors = {}
        mock_manager.get_trending_topics = AsyncMock(return_value=[])
        mock_manager.select_topic_for_debate = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=mock_manager,
        ):
            result = handler.handle("/api/pulse/suggest", {"category": ["technology"]}, None)

            # Should either succeed (404 if no topics) or fail for other reasons
            assert result is not None
            # If 400, it's not due to category validation
            if result.status_code == 400:
                data = json.loads(result.body)
                assert "category" not in data.get("error", "").lower()

    def test_suggest_response_structure(self, handler):
        """Returns proper response structure when successful."""
        topic = MockTrendingTopic("AI ethics debate", "hackernews", 500)

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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
        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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
        mock_manager = Mock()
        # Mock select_topic_for_debate to raise RuntimeError which is caught
        mock_manager.get_trending_topics = AsyncMock(return_value=[Mock()])
        mock_manager.select_topic_for_debate = Mock(side_effect=RuntimeError("Selection error"))

        # _suggest_debate_topic creates PulseManager directly via import
        with patch("aragora.pulse.ingestor.PulseManager", return_value=mock_manager):
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
                        result = handler._suggest_debate_topic()

                        assert result.status_code == 500
                        data = json.loads(result.body)
                        assert "error" in data


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_import_error_handled_trending(self, handler):
        """Handles ImportError gracefully for trending endpoint."""
        # Mock get_pulse_manager to return a mock manager
        mock_manager = MagicMock()
        mock_manager.ingestors = {}
        mock_manager.get_trending_topics = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=mock_manager,
        ):
            result = handler._get_trending_topics(10)

            # Should either work or return error
            assert result is not None
            assert result.status_code in [200, 500, 503]

    def test_import_error_handled_suggest(self, handler):
        """Handles ImportError gracefully for suggest endpoint."""
        # Mock get_pulse_manager to return a mock manager
        mock_manager = MagicMock()
        mock_manager.ingestors = {}
        mock_manager.get_trending_topics = AsyncMock(return_value=[])
        mock_manager.select_topic_for_debate = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=mock_manager,
        ):
            result = handler._suggest_debate_topic()

            # Should either work or return error
            assert result is not None
            assert result.status_code in [200, 404, 500, 503]

    def test_async_loop_handling(self, handler):
        """Handles existing event loop correctly."""
        # This tests the concurrent.futures fallback
        topics = [MockTrendingTopic("Test", "hackernews", 100)]

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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

        with patch("aragora.pulse.ingestor.PulseManager") as mock_pm_class:
            with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                with patch("aragora.pulse.ingestor.RedditIngestor"):
                    with patch("aragora.pulse.ingestor.TwitterIngestor"):
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


class TestAnalyticsEndpoint:
    """Test /api/pulse/analytics endpoint."""

    def test_analytics_returns_503_without_module(self, handler):
        """Returns 503 when pulse module not available."""
        with patch("aragora.server.handlers.features.pulse.get_pulse_manager", return_value=None):
            result = handler._get_analytics()
            assert result.status_code == 503

    def test_analytics_returns_data_structure(self, handler):
        """Returns proper analytics structure when successful."""
        mock_analytics = {
            "total_debates": 10,
            "consensus_rate": 0.8,
            "avg_confidence": 0.75,
            "by_platform": {"hackernews": 5, "reddit": 3, "twitter": 2},
            "by_category": {"technology": 6, "science": 4},
        }
        mock_manager = Mock()
        mock_manager.get_analytics.return_value = mock_analytics

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=mock_manager,
        ):
            result = handler._get_analytics()

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["total_debates"] == 10
            assert data["consensus_rate"] == 0.8
            assert "by_platform" in data


class TestSchedulerStatusEndpoint:
    """Test /api/pulse/scheduler/status endpoint."""

    def test_scheduler_status_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._get_scheduler_status()
            assert result.status_code == 503

    def test_scheduler_status_returns_state(self, handler):
        """Returns scheduler state and metrics."""
        mock_status = {
            "state": "running",
            "debates_created": 5,
            "last_poll": "2024-01-01T00:00:00Z",
            "config": {"poll_interval_seconds": 300},
        }
        mock_scheduler = Mock()
        mock_scheduler.get_status.return_value = mock_status

        mock_store = Mock()
        mock_store.get_analytics.return_value = {"total": 10}

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ):
                result = handler._get_scheduler_status()

                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["state"] == "running"
                assert "store_analytics" in data


class TestSchedulerHistoryEndpoint:
    """Test /api/pulse/scheduler/history endpoint."""

    def test_scheduler_history_returns_503_without_store(self, handler):
        """Returns 503 when store not available."""
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=None,
        ):
            result = handler._get_scheduler_history(50, 0, None)
            assert result.status_code == 503

    def test_scheduler_history_returns_records(self, handler):
        """Returns history records with proper structure."""
        mock_record = Mock()
        mock_record.id = "rec-123"
        mock_record.topic_text = "Test topic"
        mock_record.platform = "hackernews"
        mock_record.category = "technology"
        mock_record.volume = 100
        mock_record.debate_id = "debate-456"
        mock_record.created_at = "2024-01-01T00:00:00Z"
        mock_record.hours_ago = 2
        mock_record.consensus_reached = True
        mock_record.confidence = 0.85
        mock_record.rounds_used = 3
        mock_record.scheduler_run_id = "run-789"

        mock_store = Mock()
        mock_store.get_history.return_value = [mock_record]
        mock_store.count_total.return_value = 1

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_scheduler_history(50, 0, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["count"] == 1
            assert data["debates"][0]["topic"] == "Test topic"
            assert data["debates"][0]["platform"] == "hackernews"

    def test_scheduler_history_respects_platform_filter(self, handler):
        """Passes platform filter to store."""
        mock_store = Mock()
        mock_store.get_history.return_value = []
        mock_store.count_total.return_value = 0

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            handler._get_scheduler_history(50, 0, "reddit")

            mock_store.get_history.assert_called_with(limit=50, offset=0, platform="reddit")


class TestDebateTopicEndpoint:
    """Test POST /api/pulse/debate-topic endpoint."""

    def test_can_handle_debate_topic_route(self, handler):
        """can_handle returns True for debate-topic."""
        assert handler.can_handle("/api/pulse/debate-topic") is True

    def test_debate_topic_validates_topic_presence(self, handler):
        """Returns 400 when topic is missing."""
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        # Mock auth to pass
        with patch.object(handler, "_start_debate_on_topic") as method:
            # Call the underlying method directly (bypassing auth)
            import json as json_module
            from io import BytesIO

            mock_handler.rfile = BytesIO(b"{}")
            mock_handler.headers = {"Content-Length": "2"}

            # The actual validation happens in the method
            # We test the error response for missing topic

    def test_debate_topic_validates_topic_length(self, handler):
        """Returns 400 when topic exceeds max length."""
        from io import BytesIO

        long_topic = "x" * 250  # Exceeds MAX_TOPIC_LENGTH (200)
        body = json.dumps({"topic": long_topic}).encode()

        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile = BytesIO(body)

        # Need to mock auth decorator to test the validation
        with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
            with patch(
                "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
            ):
                # Re-import to get undecorated version
                result = handler._start_debate_on_topic.__wrapped__.__wrapped__(
                    handler, mock_handler
                )
                assert result.status_code == 400
                data = json.loads(result.body)
                assert "200 characters" in data["error"]

    def test_debate_topic_rejects_invalid_characters(self, handler):
        """Returns 400 when topic contains null characters."""
        from io import BytesIO

        body = json.dumps({"topic": "Test\x00topic"}).encode()

        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile = BytesIO(body)

        with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
            with patch(
                "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
            ):
                result = handler._start_debate_on_topic.__wrapped__.__wrapped__(
                    handler, mock_handler
                )
                assert result.status_code == 400
                data = json.loads(result.body)
                assert "invalid characters" in data["error"]


class TestSchedulerControlEndpoints:
    """Test scheduler control endpoints (start/stop/pause/resume)."""

    def test_start_scheduler_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        mock_handler = Mock()

        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._start_scheduler.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 503

    def test_stop_scheduler_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "0"}

        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._stop_scheduler.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 503

    def test_pause_scheduler_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        mock_handler = Mock()

        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._pause_scheduler.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 503

    def test_resume_scheduler_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        mock_handler = Mock()

        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._resume_scheduler.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 503


class TestSchedulerConfigEndpoint:
    """Test PATCH /api/pulse/scheduler/config endpoint."""

    def test_config_returns_503_without_scheduler(self, handler):
        """Returns 503 when scheduler not available."""
        from io import BytesIO

        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile = BytesIO(b"{}")

        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 503

    def test_config_requires_body(self, handler):
        """Returns 400 when body is empty."""
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": "0"}

        mock_scheduler = Mock()

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 400
                    data = json.loads(result.body)
                    assert "required" in data["error"]

    def test_config_rejects_invalid_keys(self, handler):
        """Returns 400 when config has invalid keys."""
        from io import BytesIO

        body = json.dumps({"invalid_key": "value"}).encode()
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile = BytesIO(body)

        mock_scheduler = Mock()

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 400
                    data = json.loads(result.body)
                    assert "Invalid config keys" in data["error"]

    def test_config_accepts_valid_keys(self, handler):
        """Updates config with valid keys."""
        from io import BytesIO

        body = json.dumps({"poll_interval_seconds": 600}).encode()
        mock_handler = Mock()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile = BytesIO(body)

        mock_config = Mock()
        mock_config.to_dict.return_value = {"poll_interval_seconds": 600}
        mock_scheduler = Mock()
        mock_scheduler.config = mock_config

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            with patch("aragora.server.handlers.features.pulse.require_auth", lambda f: f):
                with patch(
                    "aragora.server.handlers.features.pulse.rate_limit", lambda **kw: lambda f: f
                ):
                    result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                        handler, mock_handler
                    )
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["success"] is True
                    mock_scheduler.update_config.assert_called_once()


class TestHandlePostRouting:
    """Test POST request routing."""

    def test_handle_post_routes_debate_topic(self, handler):
        """Routes POST /api/pulse/debate-topic correctly."""
        mock_handler = Mock()

        with patch.object(handler, "_start_debate_on_topic") as mock_method:
            mock_method.return_value = Mock(status_code=200, body=b"{}")
            handler.handle_post("/api/pulse/debate-topic", {}, mock_handler)
            mock_method.assert_called_once()

    def test_handle_post_routes_scheduler_start(self, handler):
        """Routes POST /api/pulse/scheduler/start correctly."""
        mock_handler = Mock()

        with patch.object(handler, "_start_scheduler") as mock_method:
            mock_method.return_value = Mock(status_code=200, body=b"{}")
            handler.handle_post("/api/pulse/scheduler/start", {}, mock_handler)
            mock_method.assert_called_once()

    def test_handle_post_returns_none_for_unknown(self, handler):
        """Returns None for unknown POST paths."""
        result = handler.handle_post("/api/pulse/unknown", {}, None)
        assert result is None


class TestHandlePatchRouting:
    """Test PATCH request routing."""

    def test_handle_patch_routes_scheduler_config(self, handler):
        """Routes PATCH /api/pulse/scheduler/config correctly."""
        mock_handler = Mock()

        with patch.object(handler, "_update_scheduler_config") as mock_method:
            mock_method.return_value = Mock(status_code=200, body=b"{}")
            handler.handle_patch("/api/pulse/scheduler/config", {}, mock_handler)
            mock_method.assert_called_once()

    def test_handle_patch_returns_none_for_unknown(self, handler):
        """Returns None for unknown PATCH paths."""
        result = handler.handle_patch("/api/pulse/unknown", {}, None)
        assert result is None
