"""
Tests for Pulse/Trending Topics Handler.

Tests cover:
- Trending topics endpoint
- Source management (HackerNews, Reddit, Twitter)
- Filtering and scoring
- Suggest debate topic
- Analytics
- Debate topic creation
- Scheduler endpoints (status, start, stop, pause, resume, config, history)
- Error handling
"""

import json
from dataclasses import dataclass, field
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.pulse import (
    MAX_TOPIC_LENGTH,
    PulseHandler,
    get_pulse_manager,
    get_pulse_scheduler,
    get_scheduled_debate_store,
    _pulse_lock,
    _shared_pulse_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result):
    """Parse a HandlerResult dataclass into (body_dict, status_code)."""
    body = json.loads(result.body) if result.body else {}
    return body, result.status_code


def _make_handler(
    *,
    headers: dict[str, str] | None = None,
    rfile: BytesIO | None = None,
    content_length: int = 0,
):
    """Build a fake HTTP handler object."""
    handler = SimpleNamespace()
    handler.headers = headers or {}
    # Add default auth header for tests
    if "Authorization" not in handler.headers:
        handler.headers["Authorization"] = "Bearer test-token"
    if content_length > 0:
        handler.headers["Content-Length"] = str(content_length)
    handler.rfile = rfile or BytesIO(b"")
    return handler


def _make_trending_topic(
    topic: str = "Test Topic",
    platform: str = "hackernews",
    volume: int = 100,
    category: str = "tech",
):
    """Create a mock TrendingTopic object."""
    return SimpleNamespace(
        topic=topic,
        platform=platform,
        volume=volume,
        category=category,
        to_debate_prompt=lambda: f"Debate: {topic}",
    )


def _make_debate_record(
    id: str = "rec1",
    topic_text: str = "Test Topic",
    platform: str = "hackernews",
    category: str = "tech",
    volume: int = 100,
    debate_id: str = "debate-123",
    created_at: str = "2025-01-01T00:00:00Z",
    hours_ago: float = 1.0,
    consensus_reached: bool = True,
    confidence: float = 0.85,
    rounds_used: int = 3,
    scheduler_run_id: str = "run-1",
):
    """Create a mock scheduled debate record."""
    return SimpleNamespace(
        id=id,
        topic_text=topic_text,
        platform=platform,
        category=category,
        volume=volume,
        debate_id=debate_id,
        created_at=created_at,
        hours_ago=hours_ago,
        consensus_reached=consensus_reached,
        confidence=confidence,
        rounds_used=rounds_used,
        scheduler_run_id=scheduler_run_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_auth():
    """Mock authentication for all tests."""
    from aragora.server.auth import auth_config

    # Directly patch the singleton's attributes
    original_token = auth_config.api_token
    original_validate = auth_config.validate_token

    auth_config.api_token = "test-token"
    auth_config.validate_token = lambda t: True

    yield

    # Restore
    auth_config.api_token = original_token
    auth_config.validate_token = original_validate


@pytest.fixture
def pulse_handler():
    """Create a PulseHandler instance."""
    return PulseHandler(server_context={})


@pytest.fixture
def mock_pulse_manager():
    """Create a mock PulseManager."""
    manager = MagicMock()
    manager.ingestors = {"hackernews": MagicMock(), "reddit": MagicMock(), "twitter": MagicMock()}
    return manager


@pytest.fixture
def mock_scheduler():
    """Create a mock PulseDebateScheduler."""
    scheduler = MagicMock()
    scheduler.state = SimpleNamespace(value="stopped")
    scheduler._debate_creator = None
    scheduler.get_status = MagicMock(return_value={"state": "stopped", "metrics": {}})
    scheduler.config = MagicMock()
    scheduler.config.to_dict = MagicMock(return_value={"poll_interval_seconds": 300})
    return scheduler


@pytest.fixture
def mock_store():
    """Create a mock ScheduledDebateStore."""
    store = MagicMock()
    store.get_analytics = MagicMock(return_value={"total_debates": 10})
    store.get_history = MagicMock(return_value=[])
    store.count_total = MagicMock(return_value=0)
    return store


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestPulseConstants:
    """Tests for pulse module constants."""

    def test_max_topic_length(self):
        """Test that max topic length is reasonable."""
        assert MAX_TOPIC_LENGTH > 0
        assert MAX_TOPIC_LENGTH == 200


# ---------------------------------------------------------------------------
# Tests: Handler Setup
# ---------------------------------------------------------------------------


class TestPulseHandler:
    """Tests for PulseHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = PulseHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(PulseHandler, "ROUTES")
        routes = PulseHandler.ROUTES

        # Core pulse routes
        assert "/api/v1/pulse/trending" in routes
        assert "/api/v1/pulse/suggest" in routes
        assert "/api/v1/pulse/analytics" in routes
        assert "/api/v1/pulse/debate-topic" in routes

        # Scheduler routes
        assert "/api/v1/pulse/scheduler/status" in routes
        assert "/api/v1/pulse/scheduler/start" in routes
        assert "/api/v1/pulse/scheduler/stop" in routes
        assert "/api/v1/pulse/scheduler/pause" in routes
        assert "/api/v1/pulse/scheduler/resume" in routes
        assert "/api/v1/pulse/scheduler/config" in routes
        assert "/api/v1/pulse/scheduler/history" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = PulseHandler(server_context={})

        assert handler.can_handle("/api/v1/pulse/trending") is True
        assert handler.can_handle("/api/v1/pulse/suggest") is True
        assert handler.can_handle("/api/v1/pulse/scheduler/status") is True

        # Invalid routes
        assert handler.can_handle("/api/v1/invalid/route") is False
        assert handler.can_handle("/api/v1/pulse/unknown") is False

    def test_handler_has_post_method(self):
        """Test that handler has handle_post method."""
        handler = PulseHandler(server_context={})
        assert hasattr(handler, "handle_post")
        assert callable(handler.handle_post)

    def test_handler_has_patch_method(self):
        """Test that handler has handle_patch method."""
        handler = PulseHandler(server_context={})
        assert hasattr(handler, "handle_patch")
        assert callable(handler.handle_patch)


# ---------------------------------------------------------------------------
# Tests: Trending Topics
# ---------------------------------------------------------------------------


class TestGetTrendingTopics:
    """Tests for GET /api/v1/pulse/trending endpoint."""

    def test_trending_topics_success(self, pulse_handler):
        """Test successful trending topics retrieval."""
        topics = [
            _make_trending_topic("AI Advances", "hackernews", 500, "tech"),
            _make_trending_topic("Climate Change", "reddit", 300, "science"),
        ]

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {"hackernews": MagicMock()}
            mock_manager_cls.return_value = mock_manager

            pulse_handler._run_async_safely = MagicMock(return_value=topics)

            http_handler = _make_handler()
            result = pulse_handler._get_trending_topics(10)
            body, status = _parse_result(result)

            assert status == 200
            assert "topics" in body
            assert body["count"] == 2
            assert body["topics"][0]["topic"] == "AI Advances"
            assert body["topics"][0]["source"] == "hackernews"
            assert "score" in body["topics"][0]

    def test_trending_topics_with_limit(self, pulse_handler):
        """Test trending topics with custom limit."""
        http_handler = _make_handler()

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {"hackernews": MagicMock()}
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=[])

            result = pulse_handler.handle("/api/v1/pulse/trending", {"limit": ["5"]}, http_handler)
            body, status = _parse_result(result)

            assert status == 200

    def test_trending_topics_limit_capped_at_50(self, pulse_handler):
        """Test that limit is capped at 50."""
        http_handler = _make_handler()

        # The method should cap limit at 50
        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=[])

            # Request 100, should be capped
            result = pulse_handler.handle(
                "/api/v1/pulse/trending", {"limit": ["100"]}, http_handler
            )
            body, status = _parse_result(result)
            assert status == 200

    def test_trending_topics_feature_unavailable(self, pulse_handler):
        """Test response when pulse module is not available."""
        with patch.dict(
            "sys.modules",
            {"aragora.pulse.ingestor": None},
        ):
            # Simulate ImportError
            with patch(
                "aragora.pulse.ingestor.PulseManager",
                side_effect=ImportError("Module not found"),
            ):
                result = pulse_handler._get_trending_topics(10)
                body, status = _parse_result(result)
                assert status == 503

    def test_trending_topics_normalizes_scores(self, pulse_handler):
        """Test that scores are normalized to 0-1 range."""
        topics = [
            _make_trending_topic("Topic A", "hackernews", 1000, "tech"),
            _make_trending_topic("Topic B", "reddit", 500, "science"),
        ]

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {"hackernews": MagicMock()}
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=topics)

            result = pulse_handler._get_trending_topics(10)
            body, status = _parse_result(result)

            assert status == 200
            # First topic should have score 1.0 (max volume)
            assert body["topics"][0]["score"] == 1.0
            # Second topic should have score 0.5 (half volume)
            assert body["topics"][1]["score"] == 0.5


# ---------------------------------------------------------------------------
# Tests: Suggest Debate Topic
# ---------------------------------------------------------------------------


class TestSuggestDebateTopic:
    """Tests for GET /api/v1/pulse/suggest endpoint."""

    def test_suggest_topic_success(self, pulse_handler):
        """Test successful topic suggestion."""
        selected_topic = _make_trending_topic("AI Safety", "hackernews", 1000, "tech")

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager.select_topic_for_debate = MagicMock(return_value=selected_topic)
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=[selected_topic])

            result = pulse_handler._suggest_debate_topic()
            body, status = _parse_result(result)

            assert status == 200
            assert body["topic"] == "AI Safety"
            assert "debate_prompt" in body
            assert body["source"] == "hackernews"

    def test_suggest_topic_with_category_filter(self, pulse_handler):
        """Test topic suggestion with category filter."""
        selected_topic = _make_trending_topic("Climate", "reddit", 500, "science")

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager.select_topic_for_debate = MagicMock(return_value=selected_topic)
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=[selected_topic])

            result = pulse_handler._suggest_debate_topic(category="science")
            body, status = _parse_result(result)

            assert status == 200
            assert body["category"] == "science"

    def test_suggest_topic_no_suitable_topic(self, pulse_handler):
        """Test response when no suitable topic is found."""
        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager.select_topic_for_debate = MagicMock(return_value=None)
            mock_manager_cls.return_value = mock_manager
            pulse_handler._run_async_safely = MagicMock(return_value=[])

            result = pulse_handler._suggest_debate_topic()
            body, status = _parse_result(result)

            assert status == 404
            assert body["topic"] is None

    def test_suggest_topic_invalid_category(self, pulse_handler):
        """Test validation of category parameter."""
        http_handler = _make_handler()
        # Invalid category with special characters
        result = pulse_handler.handle(
            "/api/v1/pulse/suggest",
            {"category": ["<script>alert(1)</script>"]},
            http_handler,
        )
        body, status = _parse_result(result)
        assert status == 400


# ---------------------------------------------------------------------------
# Tests: Analytics
# ---------------------------------------------------------------------------


class TestPulseAnalytics:
    """Tests for GET /api/v1/pulse/analytics endpoint."""

    def test_analytics_success(self, pulse_handler):
        """Test successful analytics retrieval."""
        analytics_data = {
            "total_debates": 50,
            "consensus_rate": 0.72,
            "avg_confidence": 0.85,
            "by_platform": {"hackernews": {"total": 30}},
            "by_category": {"tech": {"total": 25}},
            "recent_outcomes": [],
        }

        with patch("aragora.server.handlers.features.pulse.get_pulse_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_analytics = MagicMock(return_value=analytics_data)
            mock_get_manager.return_value = mock_manager

            result = pulse_handler._get_analytics()
            body, status = _parse_result(result)

            assert status == 200
            assert body["total_debates"] == 50
            assert body["consensus_rate"] == 0.72

    def test_analytics_feature_unavailable(self, pulse_handler):
        """Test response when pulse manager is not available."""
        with patch("aragora.server.handlers.features.pulse.get_pulse_manager") as mock_get_manager:
            mock_get_manager.return_value = None

            result = pulse_handler._get_analytics()
            body, status = _parse_result(result)

            assert status == 503


# ---------------------------------------------------------------------------
# Tests: Start Debate on Topic
# ---------------------------------------------------------------------------


class TestStartDebateOnTopic:
    """Tests for POST /api/v1/pulse/debate-topic endpoint."""

    def test_start_debate_missing_body(self, pulse_handler):
        """Test error when request body is empty."""
        http_handler = _make_handler(content_length=0)

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "body" in body["error"].lower() or "required" in body["error"].lower()

    def test_start_debate_missing_topic(self, pulse_handler):
        """Test error when topic is missing."""
        body_data = json.dumps({"agents": ["anthropic-api"]})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "topic" in body["error"].lower()

    def test_start_debate_empty_topic(self, pulse_handler):
        """Test error when topic is empty string."""
        body_data = json.dumps({"topic": "   "})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "topic" in body["error"].lower()

    def test_start_debate_topic_too_long(self, pulse_handler):
        """Test error when topic exceeds max length."""
        long_topic = "x" * (MAX_TOPIC_LENGTH + 1)
        body_data = json.dumps({"topic": long_topic})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "200" in body["error"] or "characters" in body["error"].lower()

    def test_start_debate_topic_invalid_characters(self, pulse_handler):
        """Test error when topic contains invalid characters."""
        body_data = json.dumps({"topic": "Topic with\x00null"})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "invalid" in body["error"].lower()

    def test_start_debate_invalid_json(self, pulse_handler):
        """Test error when body contains invalid JSON."""
        body_data = "not valid json {"
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert body["error"]  # Sanitized error message present

    def test_start_debate_invalid_consensus(self, pulse_handler):
        """Test error when consensus type is invalid."""
        body_data = json.dumps({"topic": "AI Ethics", "consensus": "invalid_consensus_type"})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        # Validation now happens before feature import check
        result = pulse_handler._start_debate_on_topic(http_handler)
        body, status = _parse_result(result)

        assert status == 400
        assert "consensus" in body["error"].lower()


# ---------------------------------------------------------------------------
# Tests: Scheduler Status
# ---------------------------------------------------------------------------


class TestSchedulerStatus:
    """Tests for GET /api/v1/pulse/scheduler/status endpoint."""

    def test_scheduler_status_success(self, pulse_handler, mock_scheduler, mock_store):
        """Test successful scheduler status retrieval."""
        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler"
            ) as mock_get_scheduler,
            patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
            ) as mock_get_store,
        ):
            mock_get_scheduler.return_value = mock_scheduler
            mock_get_store.return_value = mock_store

            result = pulse_handler._get_scheduler_status()
            body, status = _parse_result(result)

            assert status == 200
            assert "state" in body
            assert "store_analytics" in body

    def test_scheduler_status_unavailable(self, pulse_handler):
        """Test response when scheduler is not available."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = None

            result = pulse_handler._get_scheduler_status()
            body, status = _parse_result(result)

            assert status == 503


# ---------------------------------------------------------------------------
# Tests: Scheduler Control (Start/Stop/Pause/Resume)
# ---------------------------------------------------------------------------


class TestSchedulerControl:
    """Tests for scheduler control endpoints."""

    def test_start_scheduler_success(self, pulse_handler, mock_scheduler):
        """Test starting the scheduler."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            http_handler = _make_handler()
            result = pulse_handler._start_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert "started" in body["message"].lower()

    def test_start_scheduler_unavailable(self, pulse_handler):
        """Test starting scheduler when unavailable."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = None

            http_handler = _make_handler()
            result = pulse_handler._start_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 503

    def test_stop_scheduler_success(self, pulse_handler, mock_scheduler):
        """Test stopping the scheduler."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            body_data = json.dumps({"graceful": True})
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )
            result = pulse_handler._stop_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert "stopped" in body["message"].lower()

    def test_stop_scheduler_non_graceful(self, pulse_handler, mock_scheduler):
        """Test stopping scheduler non-gracefully."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            body_data = json.dumps({"graceful": False})
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )
            result = pulse_handler._stop_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert "graceful=False" in body["message"]

    def test_pause_scheduler_success(self, pulse_handler, mock_scheduler):
        """Test pausing the scheduler."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            http_handler = _make_handler()
            result = pulse_handler._pause_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert "paused" in body["message"].lower()

    def test_resume_scheduler_success(self, pulse_handler, mock_scheduler):
        """Test resuming the scheduler."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            http_handler = _make_handler()
            result = pulse_handler._resume_scheduler(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert "resumed" in body["message"].lower()


# ---------------------------------------------------------------------------
# Tests: Scheduler Config
# ---------------------------------------------------------------------------


class TestSchedulerConfig:
    """Tests for PATCH /api/v1/pulse/scheduler/config endpoint."""

    def test_update_config_success(self, pulse_handler, mock_scheduler):
        """Test successful config update."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler

            body_data = json.dumps({"poll_interval_seconds": 600, "max_debates_per_hour": 10})
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )

            result = pulse_handler._update_scheduler_config(http_handler)
            body, status = _parse_result(result)

            assert status == 200
            assert body["success"] is True
            mock_scheduler.update_config.assert_called_once()

    def test_update_config_empty_body(self, pulse_handler, mock_scheduler):
        """Test error when body is empty."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler

            http_handler = _make_handler(content_length=0)

            result = pulse_handler._update_scheduler_config(http_handler)
            body, status = _parse_result(result)

            assert status == 400

    def test_update_config_invalid_keys(self, pulse_handler, mock_scheduler):
        """Test error when invalid config keys are provided."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler

            body_data = json.dumps({"invalid_key": "value"})
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )

            result = pulse_handler._update_scheduler_config(http_handler)
            body, status = _parse_result(result)

            assert status == 400
            assert "invalid" in body["error"].lower()

    def test_update_config_not_dict(self, pulse_handler, mock_scheduler):
        """Test error when body is not a dict."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler

            body_data = json.dumps(["not", "a", "dict"])
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )

            result = pulse_handler._update_scheduler_config(http_handler)
            body, status = _parse_result(result)

            assert status == 400
            assert "object" in body["error"].lower()


# ---------------------------------------------------------------------------
# Tests: Scheduler History
# ---------------------------------------------------------------------------


class TestSchedulerHistory:
    """Tests for GET /api/v1/pulse/scheduler/history endpoint."""

    def test_history_success(self, pulse_handler, mock_store):
        """Test successful history retrieval."""
        records = [
            _make_debate_record(id="rec1"),
            _make_debate_record(id="rec2", topic_text="Another Topic"),
        ]
        mock_store.get_history = MagicMock(return_value=records)
        mock_store.count_total = MagicMock(return_value=2)

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
        ) as mock_get_store:
            mock_get_store.return_value = mock_store

            result = pulse_handler._get_scheduler_history(50, 0, None)
            body, status = _parse_result(result)

            assert status == 200
            assert body["count"] == 2
            assert body["total"] == 2
            assert len(body["debates"]) == 2

    def test_history_with_pagination(self, pulse_handler, mock_store):
        """Test history with pagination parameters."""
        mock_store.get_history = MagicMock(return_value=[])
        mock_store.count_total = MagicMock(return_value=100)

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
        ) as mock_get_store:
            mock_get_store.return_value = mock_store

            result = pulse_handler._get_scheduler_history(10, 20, None)
            body, status = _parse_result(result)

            assert status == 200
            assert body["limit"] == 10
            assert body["offset"] == 20
            mock_store.get_history.assert_called_once_with(limit=10, offset=20, platform=None)

    def test_history_with_platform_filter(self, pulse_handler, mock_store):
        """Test history with platform filter."""
        mock_store.get_history = MagicMock(return_value=[])
        mock_store.count_total = MagicMock(return_value=50)

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
        ) as mock_get_store:
            mock_get_store.return_value = mock_store

            result = pulse_handler._get_scheduler_history(50, 0, "hackernews")
            body, status = _parse_result(result)

            assert status == 200
            mock_store.get_history.assert_called_once_with(
                limit=50, offset=0, platform="hackernews"
            )

    def test_history_unavailable(self, pulse_handler):
        """Test response when store is not available."""
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
        ) as mock_get_store:
            mock_get_store.return_value = None

            result = pulse_handler._get_scheduler_history(50, 0, None)
            body, status = _parse_result(result)

            assert status == 503


# ---------------------------------------------------------------------------
# Tests: POST Routing
# ---------------------------------------------------------------------------


class TestPostRouting:
    """Tests for POST request routing."""

    def test_handle_post_debate_topic(self, pulse_handler):
        """Test POST routing to debate-topic endpoint."""
        body_data = json.dumps({"topic": "Test"})
        http_handler = _make_handler(
            rfile=BytesIO(body_data.encode()),
            content_length=len(body_data),
        )

        # Should route to _start_debate_on_topic
        result = pulse_handler.handle_post("/api/v1/pulse/debate-topic", {}, http_handler)
        # We expect an error since topic is too short or other validation
        # The point is that routing worked
        assert result is not None

    def test_handle_post_scheduler_start(self, pulse_handler, mock_scheduler):
        """Test POST routing to scheduler/start."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            pulse_handler._run_async_safely = MagicMock(return_value=None)

            http_handler = _make_handler()
            result = pulse_handler.handle_post("/api/v1/pulse/scheduler/start", {}, http_handler)
            body, status = _parse_result(result)

            assert status == 200

    def test_handle_post_unknown_path(self, pulse_handler):
        """Test POST returns None for unknown path."""
        http_handler = _make_handler()
        result = pulse_handler.handle_post("/api/v1/pulse/unknown", {}, http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: PATCH Routing
# ---------------------------------------------------------------------------


class TestPatchRouting:
    """Tests for PATCH request routing."""

    def test_handle_patch_scheduler_config(self, pulse_handler, mock_scheduler):
        """Test PATCH routing to scheduler/config."""
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler"
        ) as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler

            body_data = json.dumps({"poll_interval_seconds": 300})
            http_handler = _make_handler(
                rfile=BytesIO(body_data.encode()),
                content_length=len(body_data),
            )

            result = pulse_handler.handle_patch("/api/v1/pulse/scheduler/config", {}, http_handler)
            body, status = _parse_result(result)

            assert status == 200

    def test_handle_patch_unknown_path(self, pulse_handler):
        """Test PATCH returns None for unknown path."""
        http_handler = _make_handler()
        result = pulse_handler.handle_patch("/api/v1/pulse/unknown", {}, http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_async_timeout_error(self, pulse_handler):
        """Test handling of async timeout errors."""
        import asyncio

        pulse_handler._run_async_safely = MagicMock(side_effect=asyncio.TimeoutError("Timeout"))

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager_cls.return_value = mock_manager

            result = pulse_handler._get_trending_topics(10)
            body, status = _parse_result(result)

            assert status == 500

    def test_runtime_error(self, pulse_handler):
        """Test handling of runtime errors."""
        pulse_handler._run_async_safely = MagicMock(
            side_effect=RuntimeError("Something went wrong")
        )

        with (
            patch("aragora.pulse.ingestor.PulseManager") as mock_manager_cls,
            patch("aragora.pulse.ingestor.HackerNewsIngestor"),
            patch("aragora.pulse.ingestor.RedditIngestor"),
            patch("aragora.pulse.ingestor.TwitterIngestor"),
        ):
            mock_manager = MagicMock()
            mock_manager.ingestors = {}
            mock_manager_cls.return_value = mock_manager

            result = pulse_handler._get_trending_topics(10)
            body, status = _parse_result(result)

            assert status == 500


# ---------------------------------------------------------------------------
# Tests: Singleton Functions
# ---------------------------------------------------------------------------


class TestSingletonFunctions:
    """Tests for singleton getter functions."""

    def test_get_pulse_manager_import_error(self):
        """Test get_pulse_manager handles import errors gracefully."""
        import aragora.server.handlers.features.pulse as pulse_module

        # Reset shared state
        pulse_module._shared_pulse_manager = None

        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            result = get_pulse_manager()
            # Should return None on import error
            assert result is None

    def test_get_scheduled_debate_store_error(self):
        """Test get_scheduled_debate_store handles errors gracefully."""
        import aragora.server.handlers.features.pulse as pulse_module

        # Reset shared state
        pulse_module._shared_debate_store = None

        with patch(
            "aragora.pulse.store.ScheduledDebateStore",
            side_effect=ImportError("Not found"),
        ):
            result = get_scheduled_debate_store()
            assert result is None
