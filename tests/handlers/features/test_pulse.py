"""Tests for pulse/trending topics handler.

Tests the pulse API endpoints including:
- GET /api/v1/pulse/trending - Get trending topics
- GET /api/v1/pulse/suggest - Suggest a trending topic for debate
- GET /api/v1/pulse/analytics - Get analytics on debate outcomes
- POST /api/v1/pulse/debate-topic - Start a debate on a trending topic
- GET /api/v1/pulse/topics/{topic_id}/outcomes - Get debate outcomes for a topic
- GET /api/v1/pulse/scheduler/status - Scheduler status
- GET /api/v1/pulse/scheduler/analytics - Scheduler analytics
- POST /api/v1/pulse/scheduler/start - Start scheduler
- POST /api/v1/pulse/scheduler/stop - Stop scheduler
- POST /api/v1/pulse/scheduler/pause - Pause scheduler
- POST /api/v1/pulse/scheduler/resume - Resume scheduler
- PATCH /api/v1/pulse/scheduler/config - Update scheduler config
- GET /api/v1/pulse/scheduler/history - Get scheduler history
"""

import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.pulse import PulseHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: dict[str, Any] | None = None, token: str = "test-valid-token"):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        else:
            self.rfile.read.return_value = b""
            self.headers = {
                "Content-Length": "0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        self.client_address = ("127.0.0.1", 12345)


# Mock trending topic dataclass
@dataclass
class FakeTrendingTopic:
    topic: str
    platform: str
    volume: int
    category: str

    def to_debate_prompt(self) -> str:
        return f"Debate: {self.topic}"


class FakeSchedulerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pulse_singletons(monkeypatch):
    """Reset pulse module singletons between tests."""
    import aragora.server.handlers.features.pulse as pulse_mod

    monkeypatch.setattr(pulse_mod, "_shared_pulse_manager", None)
    monkeypatch.setattr(pulse_mod, "_shared_scheduler", None)
    monkeypatch.setattr(pulse_mod, "_shared_debate_store", None)


@pytest.fixture(autouse=True)
def _patch_require_auth(monkeypatch):
    """Bypass the @require_auth token check for tests."""
    try:
        from aragora.server import auth as server_auth

        mock_auth_config = MagicMock()
        mock_auth_config.api_token = "test-valid-token"
        mock_auth_config.validate_token.return_value = True
        mock_auth_config.enabled = True
        monkeypatch.setattr(server_auth, "auth_config", mock_auth_config)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _clear_ttl_cache():
    """Clear handler caches between tests."""
    try:
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def handler():
    """Create a PulseHandler instance."""
    return PulseHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler factory."""

    def _make(body=None, token="test-valid-token"):
        return MockHTTPHandler(body=body, token=token)

    return _make


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_mock_manager(topics=None, analytics=None, outcomes=None):
    """Create a mock PulseManager."""
    mgr = MagicMock()
    mgr.ingestors = {"hackernews": MagicMock(), "reddit": MagicMock(), "twitter": MagicMock()}
    if topics is not None:
        mgr.get_trending_topics = MagicMock(return_value=topics)
    if analytics is not None:
        mgr.get_analytics.return_value = analytics
    if outcomes is not None:
        mgr._outcomes = outcomes
    else:
        mgr._outcomes = []
    mgr.select_topic_for_debate.return_value = topics[0] if topics else None
    return mgr


def _make_mock_scheduler(state="idle"):
    """Create a mock PulseDebateScheduler."""
    sched = MagicMock()
    sched.state = FakeSchedulerState(state)
    sched.get_status.return_value = {"state": state, "polls": 0}
    sched.metrics.to_dict.return_value = {"total_polls": 10, "debates_created": 5}
    sched._debate_creator = None
    config_mock = MagicMock()
    config_mock.to_dict.return_value = {"poll_interval_seconds": 300}
    sched.config = config_mock
    return sched


def _make_mock_store(records=None, analytics=None, rows=None, total=0):
    """Create a mock ScheduledDebateStore."""
    store = MagicMock()
    store.get_history.return_value = records or []
    store.get_analytics.return_value = analytics or {"total": 0}
    store.count_total.return_value = total
    store.fetch_all.return_value = rows or []
    return store


def _mock_pulse_ingestor_module():
    """Create a mock module for aragora.pulse.ingestor imports."""
    mock_mod = MagicMock()
    mock_mod.PulseManager = MagicMock
    mock_mod.HackerNewsIngestor = MagicMock
    mock_mod.RedditIngestor = MagicMock
    mock_mod.TwitterIngestor = MagicMock
    mock_mod.TrendingTopic = FakeTrendingTopic
    return mock_mod


# ============================================================================
# can_handle tests
# ============================================================================


class TestCanHandle:
    """Tests for PulseHandler.can_handle()."""

    def test_trending_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/trending") is True

    def test_suggest_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/suggest") is True

    def test_analytics_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/analytics") is True

    def test_debate_topic_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/debate-topic") is True

    def test_scheduler_status_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/status") is True

    def test_scheduler_start_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/start") is True

    def test_scheduler_stop_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/stop") is True

    def test_scheduler_pause_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/pause") is True

    def test_scheduler_resume_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/resume") is True

    def test_scheduler_config_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/config") is True

    def test_scheduler_history_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/history") is True

    def test_scheduler_analytics_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/analytics") is True

    def test_topic_outcomes_dynamic_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/topics/topic123/outcomes") is True

    def test_topic_outcomes_with_dashes(self, handler):
        assert handler.can_handle("/api/v1/pulse/topics/topic-abc-123/outcomes") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/unknown") is False

    def test_partial_topics_route(self, handler):
        assert handler.can_handle("/api/v1/pulse/topics/") is False

    def test_non_pulse_route(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_topics_without_outcomes_suffix(self, handler):
        assert handler.can_handle("/api/v1/pulse/topics/abc123/details") is False


# ============================================================================
# GET /api/v1/pulse/trending
# ============================================================================


class TestGetTrending:
    """Tests for trending topics endpoint."""

    def test_trending_returns_topics(self, handler, mock_http):
        topics = [
            FakeTrendingTopic("AI breakthrough", "hackernews", 100, "tech"),
            FakeTrendingTopic("Climate change", "reddit", 50, "science"),
        ]
        mock_pm = MagicMock()
        mock_pm.return_value = MagicMock(
            ingestors={"hackernews": MagicMock(), "reddit": MagicMock(), "twitter": MagicMock()},
        )
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=topics,
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._get_trending_topics(10)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["topics"]) == 2

    def test_trending_with_limit(self, handler, mock_http):
        """Limit param is clamped to max 50."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._get_trending_topics"
        ) as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, body=json.dumps({"topics": [], "count": 0}).encode()
            )
            handler.handle("/api/v1/pulse/trending", {"limit": "100"}, mock_http())
            mock_get.assert_called_once_with(50)

    def test_trending_default_limit(self, handler, mock_http):
        """Default limit is 10."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._get_trending_topics"
        ) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/trending", {}, mock_http())
            mock_get.assert_called_once_with(10)

    def test_trending_small_limit(self, handler, mock_http):
        """Small limit is passed through."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._get_trending_topics"
        ) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/trending", {"limit": "5"}, mock_http())
            mock_get.assert_called_once_with(5)

    def test_trending_score_normalization(self, handler, mock_http):
        """Scores should be normalized to 0-1 range."""
        topics = [
            FakeTrendingTopic("Top story", "hackernews", 200, "tech"),
            FakeTrendingTopic("Mid story", "reddit", 100, "science"),
            FakeTrendingTopic("Low story", "twitter", 50, "tech"),
        ]
        mock_pm = MagicMock()
        mock_pm.return_value = MagicMock(
            ingestors={"hackernews": MagicMock(), "reddit": MagicMock(), "twitter": MagicMock()},
        )
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=topics,
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._get_trending_topics(10)

        body = _body(result)
        assert body["topics"][0]["score"] == 1.0  # 200/200
        assert body["topics"][1]["score"] == 0.5  # 100/200
        assert body["topics"][2]["score"] == 0.25  # 50/200

    def test_trending_empty_topics(self, handler, mock_http):
        """Empty topic list returns count=0."""
        mock_pm = MagicMock()
        mock_pm.return_value = MagicMock(
            ingestors={"hackernews": MagicMock()},
        )
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=[],
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._get_trending_topics(10)
        body = _body(result)
        assert body["count"] == 0
        assert body["topics"] == []

    def test_trending_runtime_error(self, handler, mock_http):
        """RuntimeError during fetch returns 500."""
        mock_pm = MagicMock(side_effect=RuntimeError("boom"))
        with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
            with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                        result = handler._get_trending_topics(10)
        assert _status(result) == 500

    def test_trending_topic_fields(self, handler, mock_http):
        """Verify all topic fields are present in response."""
        topics = [FakeTrendingTopic("Test topic", "hackernews", 50, "tech")]
        mock_pm = MagicMock()
        mock_pm.return_value = MagicMock(
            ingestors={"hackernews": MagicMock()},
        )
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=topics,
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._get_trending_topics(10)
        body = _body(result)
        topic = body["topics"][0]
        assert "topic" in topic
        assert "source" in topic
        assert "score" in topic
        assert "volume" in topic
        assert "category" in topic

    def test_trending_sources_in_response(self, handler, mock_http):
        """Response includes list of sources."""
        mock_pm = MagicMock()
        mock_pm.return_value = MagicMock(
            ingestors={"hackernews": MagicMock(), "reddit": MagicMock()},
        )
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=[],
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._get_trending_topics(10)
        body = _body(result)
        assert "sources" in body


# ============================================================================
# GET /api/v1/pulse/suggest
# ============================================================================


class TestSuggestDebateTopic:
    """Tests for suggest debate topic endpoint."""

    def test_suggest_with_no_category(self, handler, mock_http):
        """Suggest without category filter."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._suggest_debate_topic"
        ) as mock_sug:
            mock_sug.return_value = MagicMock(
                status_code=200,
                body=json.dumps(
                    {
                        "topic": "AI Ethics",
                        "debate_prompt": "Debate: AI Ethics",
                        "source": "hackernews",
                        "category": "tech",
                        "volume": 100,
                    }
                ).encode(),
            )
            result = handler.handle("/api/v1/pulse/suggest", {}, mock_http())
        assert _status(result) == 200

    def test_suggest_with_valid_category(self, handler, mock_http):
        """Suggest with a valid category filter."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._suggest_debate_topic"
        ) as mock_sug:
            mock_sug.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/suggest", {"category": "tech"}, mock_http())
            mock_sug.assert_called_once_with("tech")

    def test_suggest_with_invalid_category(self, handler, mock_http):
        """Invalid category returns 400."""
        result = handler.handle("/api/v1/pulse/suggest", {"category": "../etc/passwd"}, mock_http())
        assert _status(result) == 400

    def test_suggest_no_topics_found(self, handler, mock_http):
        """When no suitable topic found, returns 404."""
        mock_pm_inst = MagicMock()
        mock_pm_inst.ingestors = {"hn": MagicMock()}
        mock_pm_inst.select_topic_for_debate.return_value = None
        mock_pm = MagicMock(return_value=mock_pm_inst)
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=[],
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._suggest_debate_topic(None)
        body = _body(result)
        assert _status(result) == 404
        assert body["topic"] is None

    def test_suggest_returns_topic(self, handler, mock_http):
        """Returns selected topic when found."""
        topic = FakeTrendingTopic("Quantum computing", "reddit", 80, "science")
        mock_pm_inst = MagicMock()
        mock_pm_inst.ingestors = {"hackernews": MagicMock()}
        mock_pm_inst.select_topic_for_debate.return_value = topic
        mock_pm = MagicMock(return_value=mock_pm_inst)
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=[topic],
        ):
            with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                        with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                            result = handler._suggest_debate_topic("science")
        body = _body(result)
        assert _status(result) == 200
        assert body["topic"] == "Quantum computing"
        assert body["source"] == "reddit"
        assert body["category"] == "science"
        assert body["volume"] == 80
        assert body["debate_prompt"] == "Debate: Quantum computing"

    def test_suggest_error_handling(self, handler, mock_http):
        """RuntimeError during suggest returns 500."""
        mock_pm = MagicMock(side_effect=RuntimeError("fail"))
        with patch("aragora.pulse.ingestor.PulseManager", mock_pm):
            with patch("aragora.pulse.ingestor.HackerNewsIngestor", MagicMock):
                with patch("aragora.pulse.ingestor.RedditIngestor", MagicMock):
                    with patch("aragora.pulse.ingestor.TwitterIngestor", MagicMock):
                        result = handler._suggest_debate_topic(None)
        assert _status(result) == 500

    def test_suggest_category_with_special_chars_rejected(self, handler, mock_http):
        """Categories with special characters are rejected."""
        result = handler.handle("/api/v1/pulse/suggest", {"category": "tech<script>"}, mock_http())
        assert _status(result) == 400

    def test_suggest_none_category_passed_through(self, handler, mock_http):
        """None category is passed through when not provided."""
        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._suggest_debate_topic"
        ) as mock_sug:
            mock_sug.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/suggest", {}, mock_http())
            mock_sug.assert_called_once_with(None)


# ============================================================================
# GET /api/v1/pulse/analytics
# ============================================================================


class TestGetAnalytics:
    """Tests for pulse analytics endpoint."""

    def test_analytics_with_manager(self, handler, mock_http):
        """Returns analytics when manager is available."""
        mock_mgr = _make_mock_manager(analytics={"total_debates": 5, "consensus_rate": 0.8})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
        ):
            result = handler._get_analytics()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_debates"] == 5
        assert body["consensus_rate"] == 0.8

    def test_analytics_without_manager(self, handler, mock_http):
        """Returns feature_unavailable when manager is None."""
        with patch("aragora.server.handlers.features.pulse.get_pulse_manager", return_value=None):
            result = handler._get_analytics()
        assert _status(result) == 503

    def test_analytics_route_dispatch(self, handler, mock_http):
        """Verify handle() dispatches /analytics to _get_analytics."""
        with patch.object(handler, "_get_analytics") as mock_an:
            mock_an.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/analytics", {}, mock_http())
            mock_an.assert_called_once()

    def test_analytics_manager_returns_empty(self, handler, mock_http):
        """Analytics with empty data returns 200."""
        mock_mgr = _make_mock_manager(analytics={})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
        ):
            result = handler._get_analytics()
        assert _status(result) == 200


# ============================================================================
# GET /api/v1/pulse/topics/{topic_id}/outcomes
# ============================================================================


class TestTopicOutcomes:
    """Tests for topic outcomes endpoint."""

    def test_outcomes_from_store(self, handler, mock_http):
        """Returns outcomes from scheduled debate store."""
        mock_record = MagicMock()
        mock_record.id = "rec1"
        mock_record.topic_text = "AI Ethics"
        mock_record.platform = "hackernews"
        mock_record.category = "tech"
        mock_record.debate_id = "d123"
        mock_record.consensus_reached = True
        mock_record.confidence = 0.9
        mock_record.rounds_used = 3
        mock_record.created_at = "2026-01-01T00:00:00"
        mock_record.hours_ago = 24.0

        mock_store = MagicMock()
        rows = [("row_data",)]
        mock_store.fetch_all.return_value = rows
        mock_store._row_to_record.return_value = mock_record

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=None
            ):
                result = handler._get_topic_outcomes("topic123")

        body = _body(result)
        assert _status(result) == 200
        assert body["topic_id"] == "topic123"
        assert body["count"] == 1
        assert body["outcomes"][0]["topic"] == "AI Ethics"
        assert body["outcomes"][0]["confidence"] == 0.9

    def test_outcomes_fallback_to_manager(self, handler, mock_http):
        """Falls back to in-memory outcomes when store has nothing."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []

        mock_outcome = MagicMock()
        mock_outcome.topic = "Topic X"
        mock_outcome.platform = "reddit"
        mock_outcome.debate_id = "d456"
        mock_outcome.consensus_reached = False
        mock_outcome.confidence = 0.5
        mock_outcome.rounds_used = 2
        mock_outcome.category = "science"
        mock_outcome.timestamp = "2026-01-01T00:00:00"

        mock_mgr = MagicMock()
        mock_mgr._outcomes = [mock_outcome]

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
            ):
                result = handler._get_topic_outcomes("d456")

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1

    def test_outcomes_not_found(self, handler, mock_http):
        """Returns 404 when no outcomes found anywhere."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []
        mock_mgr = MagicMock()
        mock_mgr._outcomes = []

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
            ):
                result = handler._get_topic_outcomes("nonexistent")
        body = _body(result)
        assert _status(result) == 404
        assert body["count"] == 0

    def test_outcomes_no_store_no_manager(self, handler, mock_http):
        """When both store and manager are unavailable, returns 404."""
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store", return_value=None
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=None
            ):
                result = handler._get_topic_outcomes("topic123")
        body = _body(result)
        assert _status(result) == 404
        assert body["outcomes"] == []

    def test_outcomes_store_error_falls_through(self, handler, mock_http):
        """SQLite error in store lookup falls through to manager."""
        import sqlite3

        mock_store = MagicMock()
        mock_store.fetch_all.side_effect = sqlite3.Error("db locked")
        mock_mgr = MagicMock()
        mock_mgr._outcomes = []

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
            ):
                result = handler._get_topic_outcomes("topic123")
        body = _body(result)
        assert _status(result) == 404

    def test_outcomes_route_dispatch(self, handler, mock_http):
        """The handle() method dispatches topic outcomes paths correctly."""
        with patch.object(handler, "_get_topic_outcomes") as mock_out:
            mock_out.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/topics/abc123/outcomes", {}, mock_http())
            mock_out.assert_called_once_with("abc123")

    def test_outcomes_invalid_topic_id(self, handler, mock_http):
        """Invalid topic_id (path traversal) returns None (wrong segment count)."""
        result = handler.handle("/api/v1/pulse/topics/../etc/outcomes", {}, mock_http())
        assert result is None

    def test_outcomes_route_wrong_segment_count(self, handler, mock_http):
        """Path with wrong number of segments returns None."""
        result = handler.handle("/api/v1/pulse/topics/outcomes", {}, mock_http())
        assert result is None

    def test_outcomes_invalid_topic_id_chars(self, handler, mock_http):
        """Topic ID with invalid chars returns 400."""
        result = handler.handle("/api/v1/pulse/topics/<script>/outcomes", {}, mock_http())
        assert result is not None
        assert _status(result) == 400

    def test_outcomes_manager_without_outcomes_attr(self, handler, mock_http):
        """Manager without _outcomes attribute returns 404."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []
        mock_mgr = MagicMock(spec=[])  # no _outcomes attribute

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
            ):
                result = handler._get_topic_outcomes("topic123")
        assert _status(result) == 404

    def test_outcomes_store_attribute_error(self, handler, mock_http):
        """AttributeError in store lookup falls through gracefully."""
        mock_store = MagicMock()
        mock_store.fetch_all.side_effect = AttributeError("no attr")
        mock_mgr = MagicMock()
        mock_mgr._outcomes = []

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_manager", return_value=mock_mgr
            ):
                result = handler._get_topic_outcomes("topic123")
        assert _status(result) == 404

    def test_outcomes_multiple_store_rows(self, handler, mock_http):
        """Multiple rows from store are all returned."""
        rec1 = MagicMock()
        rec1.id = "r1"
        rec1.topic_text = "Topic A"
        rec1.platform = "reddit"
        rec1.category = "tech"
        rec1.debate_id = "d1"
        rec1.consensus_reached = True
        rec1.confidence = 0.9
        rec1.rounds_used = 3
        rec1.created_at = "2026-01-01"
        rec1.hours_ago = 24.0

        rec2 = MagicMock()
        rec2.id = "r2"
        rec2.topic_text = "Topic A v2"
        rec2.platform = "reddit"
        rec2.category = "tech"
        rec2.debate_id = "d2"
        rec2.consensus_reached = False
        rec2.confidence = 0.5
        rec2.rounds_used = 2
        rec2.created_at = "2026-01-02"
        rec2.hours_ago = 12.0

        mock_store = MagicMock()
        mock_store.fetch_all.return_value = [("row1",), ("row2",)]
        mock_store._row_to_record.side_effect = [rec1, rec2]

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_topic_outcomes("topicABC")
        body = _body(result)
        assert body["count"] == 2


# ============================================================================
# POST /api/v1/pulse/debate-topic
# ============================================================================


class TestStartDebateOnTopic:
    """Tests for starting a debate on a trending topic."""

    def test_post_routes_to_debate_topic(self, handler, mock_http):
        """POST /debate-topic dispatches to _start_debate_on_topic."""
        with patch.object(handler, "_start_debate_on_topic") as mock_start:
            mock_start.return_value = MagicMock(status_code=200, body=b"{}")
            h = mock_http(body={"topic": "test"})
            handler.handle_post("/api/v1/pulse/debate-topic", {}, h)
            mock_start.assert_called_once_with(h)

    def test_missing_body(self, handler, mock_http):
        """Empty body returns 400."""
        h = MockHTTPHandler(body=None)
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_empty_topic(self, handler, mock_http):
        """Empty string topic returns 400."""
        h = mock_http(body={"topic": ""})
        result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "required" in body.get("error", "").lower()

    def test_whitespace_only_topic(self, handler, mock_http):
        """Whitespace-only topic returns 400."""
        h = mock_http(body={"topic": "   "})
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_topic_too_long(self, handler, mock_http):
        """Topic exceeding max length returns 400."""
        h = mock_http(body={"topic": "x" * 201})
        result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "200" in body.get("error", "")

    def test_topic_with_null_byte(self, handler, mock_http):
        """Topic with null byte returns 400."""
        h = mock_http(body={"topic": "test\x00topic"})
        result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "invalid" in body.get("error", "").lower()

    def test_topic_with_newline(self, handler, mock_http):
        """Topic with newline returns 400."""
        h = mock_http(body={"topic": "test\ntopic"})
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_topic_with_carriage_return(self, handler, mock_http):
        """Topic with carriage return returns 400."""
        h = mock_http(body={"topic": "test\rtopic"})
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_topic_not_string(self, handler, mock_http):
        """Non-string topic returns 400."""
        h = mock_http(body={"topic": 12345})
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_agents_not_list_or_string(self, handler, mock_http):
        """Non-list/string agents returns 400."""
        h = mock_http(body={"topic": "Test topic", "agents": 12345})
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_invalid_consensus(self, handler, mock_http):
        """Invalid consensus mode returns 400."""
        h = mock_http(body={"topic": "Test", "consensus": "invalid_mode"})
        result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "consensus" in body.get("error", "").lower()

    def test_valid_consensus_majority(self, handler, mock_http):
        """Majority consensus is accepted."""
        h = mock_http(body={"topic": "Test", "consensus": "majority"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        # Fails at "no agents", not consensus validation
        assert "consensus" not in body.get("error", "").lower()

    def test_valid_consensus_unanimous(self, handler, mock_http):
        h = mock_http(body={"topic": "Test", "consensus": "unanimous"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert "consensus" not in body.get("error", "").lower()

    def test_valid_consensus_judge(self, handler, mock_http):
        h = mock_http(body={"topic": "Test", "consensus": "judge"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert "consensus" not in body.get("error", "").lower()

    def test_valid_consensus_none(self, handler, mock_http):
        h = mock_http(body={"topic": "Test", "consensus": "none"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert "consensus" not in body.get("error", "").lower()

    def test_rounds_clamped_low(self, handler, mock_http):
        """Rounds below 1 are clamped to 1."""
        h = mock_http(body={"topic": "Test", "rounds": -5})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        assert result is not None

    def test_rounds_clamped_high(self, handler, mock_http):
        """Rounds above 10 are clamped to 10."""
        h = mock_http(body={"topic": "Test", "rounds": 100})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        assert result is not None

    def test_rounds_invalid_type_uses_default(self, handler, mock_http):
        """Non-numeric rounds fall back to default."""
        h = mock_http(body={"topic": "Test", "rounds": "not_a_number"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        assert result is not None

    def test_no_valid_agents(self, handler, mock_http):
        """When get_agents_by_names returns empty, returns 400."""
        h = mock_http(body={"topic": "Test topic"})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "agent" in body.get("error", "").lower()

    def test_successful_debate(self, handler, mock_http):
        """Full successful debate returns 200 with result."""
        h = mock_http(body={"topic": "AI Safety"})
        mock_result = MagicMock()
        mock_result.id = "debate-123"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.95
        mock_result.final_answer = "AI safety is important"
        mock_result.rounds_used = 3

        mock_agent = MagicMock()
        mock_agent.name = "anthropic-api"

        mock_arena_cls = MagicMock()
        mock_arena_cls.from_env.return_value = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=mock_result,
        ):
            with patch("aragora.agents.get_agents_by_names", return_value=[mock_agent]):
                with patch("aragora.debate.orchestrator.Arena", mock_arena_cls):
                    with patch("aragora.Arena", mock_arena_cls):
                        with patch(
                            "aragora.server.handlers.features.pulse.get_pulse_manager",
                            return_value=None,
                        ):
                            result = handler._start_debate_on_topic(h)

        body = _body(result)
        assert _status(result) == 200
        assert body["debate_id"] == "debate-123"
        assert body["status"] == "completed"
        assert body["consensus_reached"] is True
        assert body["confidence"] == 0.95
        assert body["topic"] == "AI Safety"
        assert body["rounds_used"] == 3

    def test_debate_none_result(self, handler, mock_http):
        """When debate returns None (from _run_async_safely returning []), returns 500."""
        h = mock_http(body={"topic": "Test"})
        mock_agent = MagicMock()
        mock_agent.name = "test"

        mock_arena_cls = MagicMock()
        mock_arena_cls.from_env.return_value = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=None,
        ):
            with patch("aragora.agents.get_agents_by_names", return_value=[mock_agent]):
                with patch("aragora.debate.orchestrator.Arena", mock_arena_cls):
                    with patch("aragora.Arena", mock_arena_cls):
                        result = handler._start_debate_on_topic(h)
        assert _status(result) == 500

    def test_invalid_json_body(self, handler, mock_http):
        """Malformed JSON body returns 400."""
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers = {"Content-Length": "8", "Authorization": "Bearer test-valid-token"}
        result = handler._start_debate_on_topic(h)
        assert _status(result) == 400

    def test_agents_max_5(self, handler, mock_http):
        """Only the first 5 agents are used."""
        h = mock_http(
            body={
                "topic": "Test",
                "agents": ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
            }
        )
        mock_gan = MagicMock(return_value=[])
        with patch("aragora.agents.get_agents_by_names", mock_gan):
            handler._start_debate_on_topic(h)
        call_args = mock_gan.call_args[0][0]
        assert len(call_args) == 5

    def test_debate_records_outcome(self, handler, mock_http):
        """Successful debate records outcome with pulse manager."""
        h = mock_http(body={"topic": "Test"})
        mock_result = MagicMock()
        mock_result.id = "d1"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.8
        mock_result.final_answer = "answer"
        mock_result.rounds_used = 2

        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_mgr = MagicMock()

        mock_arena_cls = MagicMock()
        mock_arena_cls.from_env.return_value = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=mock_result,
        ):
            with patch("aragora.agents.get_agents_by_names", return_value=[mock_agent]):
                with patch("aragora.debate.orchestrator.Arena", mock_arena_cls):
                    with patch("aragora.Arena", mock_arena_cls):
                        with patch(
                            "aragora.server.handlers.features.pulse.get_pulse_manager",
                            return_value=mock_mgr,
                        ):
                            with patch("aragora.pulse.ingestor.TrendingTopic", FakeTrendingTopic):
                                result = handler._start_debate_on_topic(h)
        assert _status(result) == 200
        mock_mgr.record_debate_outcome.assert_called_once()

    def test_debate_import_error(self, handler, mock_http):
        """ImportError for debate modules returns feature_unavailable (503)."""
        h = mock_http(body={"topic": "Test topic"})
        # Intercept the `from aragora import Arena, ...` inside the handler
        import builtins

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "aragora" and fromlist and "Arena" in fromlist:
                raise ImportError("debate not available")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = handler._start_debate_on_topic(h)
        assert _status(result) == 503

    def test_agents_as_comma_string(self, handler, mock_http):
        """Agents provided as comma-separated string are split into a list."""
        h = mock_http(body={"topic": "Test topic", "agents": "agent1, agent2"})
        mock_gan = MagicMock(return_value=[])
        with patch("aragora.agents.get_agents_by_names", mock_gan):
            result = handler._start_debate_on_topic(h)
        call_args = mock_gan.call_args[0][0]
        assert call_args == ["agent1", "agent2"]

    def test_final_answer_truncated(self, handler, mock_http):
        """Final answer is truncated to 500 chars in response."""
        h = mock_http(body={"topic": "Test"})
        mock_result = MagicMock()
        mock_result.id = "d1"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.8
        mock_result.final_answer = "a" * 1000
        mock_result.rounds_used = 2
        mock_agent = MagicMock()
        mock_agent.name = "test"

        mock_arena_cls = MagicMock()
        mock_arena_cls.from_env.return_value = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=mock_result,
        ):
            with patch("aragora.agents.get_agents_by_names", return_value=[mock_agent]):
                with patch("aragora.debate.orchestrator.Arena", mock_arena_cls):
                    with patch("aragora.Arena", mock_arena_cls):
                        with patch(
                            "aragora.server.handlers.features.pulse.get_pulse_manager",
                            return_value=None,
                        ):
                            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert len(body["final_answer"]) == 500

    def test_final_answer_none(self, handler, mock_http):
        """None final_answer is returned as None."""
        h = mock_http(body={"topic": "Test"})
        mock_result = MagicMock()
        mock_result.id = "d1"
        mock_result.consensus_reached = False
        mock_result.confidence = 0.3
        mock_result.final_answer = None
        mock_result.rounds_used = 1
        mock_agent = MagicMock()
        mock_agent.name = "test"

        mock_arena_cls = MagicMock()
        mock_arena_cls.from_env.return_value = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.PulseHandler._run_async_safely",
            return_value=mock_result,
        ):
            with patch("aragora.agents.get_agents_by_names", return_value=[mock_agent]):
                with patch("aragora.debate.orchestrator.Arena", mock_arena_cls):
                    with patch("aragora.Arena", mock_arena_cls):
                        with patch(
                            "aragora.server.handlers.features.pulse.get_pulse_manager",
                            return_value=None,
                        ):
                            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert body["final_answer"] is None


# ============================================================================
# POST handler dispatch
# ============================================================================


class TestHandlePost:
    """Tests for handle_post routing."""

    def test_post_scheduler_start(self, handler, mock_http):
        with patch.object(handler, "_start_scheduler") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle_post("/api/v1/pulse/scheduler/start", {}, mock_http())
            mock_s.assert_called_once()

    def test_post_scheduler_stop(self, handler, mock_http):
        with patch.object(handler, "_stop_scheduler") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle_post("/api/v1/pulse/scheduler/stop", {}, mock_http())
            mock_s.assert_called_once()

    def test_post_scheduler_pause(self, handler, mock_http):
        with patch.object(handler, "_pause_scheduler") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle_post("/api/v1/pulse/scheduler/pause", {}, mock_http())
            mock_s.assert_called_once()

    def test_post_scheduler_resume(self, handler, mock_http):
        with patch.object(handler, "_resume_scheduler") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle_post("/api/v1/pulse/scheduler/resume", {}, mock_http())
            mock_s.assert_called_once()

    def test_post_unknown_path(self, handler, mock_http):
        result = handler.handle_post("/api/v1/pulse/unknown", {}, mock_http())
        assert result is None

    def test_post_debate_topic_routes(self, handler, mock_http):
        with patch.object(handler, "_start_debate_on_topic") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            h = mock_http(body={"topic": "test"})
            handler.handle_post("/api/v1/pulse/debate-topic", {}, h)
            mock_s.assert_called_once_with(h)


# ============================================================================
# PATCH handler dispatch
# ============================================================================


class TestHandlePatch:
    """Tests for handle_patch routing."""

    def test_patch_scheduler_config(self, handler, mock_http):
        with patch.object(handler, "_update_scheduler_config") as mock_u:
            mock_u.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle_patch("/api/v1/pulse/scheduler/config", {}, mock_http())
            mock_u.assert_called_once()

    def test_patch_unknown_path(self, handler, mock_http):
        result = handler.handle_patch("/api/v1/pulse/unknown", {}, mock_http())
        assert result is None

    def test_patch_non_config_path(self, handler, mock_http):
        result = handler.handle_patch("/api/v1/pulse/scheduler/status", {}, mock_http())
        assert result is None


# ============================================================================
# Scheduler Status
# ============================================================================


class TestSchedulerStatus:
    """Tests for scheduler status endpoint."""

    def test_status_with_scheduler(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_store = _make_mock_store(analytics={"total": 10})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ):
                result = handler._get_scheduler_status()
        body = _body(result)
        assert _status(result) == 200
        assert "state" in body

    def test_status_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._get_scheduler_status()
        assert _status(result) == 503

    def test_status_no_store(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=None,
            ):
                result = handler._get_scheduler_status()
        body = _body(result)
        assert _status(result) == 200
        assert "store_analytics" not in body

    def test_status_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_status") as mock_s:
            mock_s.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/status", {}, mock_http())
            mock_s.assert_called_once()

    def test_status_includes_store_analytics(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_store = _make_mock_store(analytics={"by_platform": {"reddit": 5}})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ):
                result = handler._get_scheduler_status()
        body = _body(result)
        assert body["store_analytics"]["by_platform"]["reddit"] == 5


# ============================================================================
# Scheduler Analytics
# ============================================================================


class TestSchedulerAnalytics:
    """Tests for scheduler analytics endpoint."""

    def test_analytics_with_scheduler(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_store = _make_mock_store(analytics={"by_platform": {}})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ):
                result = handler._get_scheduler_analytics()
        body = _body(result)
        assert _status(result) == 200
        assert "scheduler_metrics" in body
        assert "store_analytics" in body

    def test_analytics_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._get_scheduler_analytics()
        assert _status(result) == 503

    def test_analytics_scheduler_no_metrics(self, handler, mock_http):
        mock_sched = MagicMock(spec=[])  # No metrics attribute
        mock_store = _make_mock_store()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ):
                result = handler._get_scheduler_analytics()
        body = _body(result)
        assert _status(result) == 200
        assert body["scheduler_metrics"] == {}

    def test_analytics_route_dispatch(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_analytics") as mock_a:
            mock_a.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/analytics", {}, mock_http())
            mock_a.assert_called_once()

    def test_analytics_no_store(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=None,
            ):
                result = handler._get_scheduler_analytics()
        body = _body(result)
        assert _status(result) == 200
        assert body["store_analytics"] == {}


# ============================================================================
# Scheduler Start
# ============================================================================


class TestSchedulerStart:
    """Tests for starting the scheduler."""

    def test_start_scheduler(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._start_scheduler(mock_http())
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["message"] == "Scheduler started"

    def test_start_sets_debate_creator(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched._debate_creator = None
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                handler._start_scheduler(mock_http())
        mock_sched.set_debate_creator.assert_called_once()

    def test_start_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._start_scheduler(mock_http())
        assert _status(result) == 503

    def test_start_skips_creator_if_set(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        mock_sched._debate_creator = MagicMock()  # Already set
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                handler._start_scheduler(mock_http())
        mock_sched.set_debate_creator.assert_not_called()

    def test_start_returns_state(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(state="running")
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._start_scheduler(mock_http())
        body = _body(result)
        assert body["state"] == "running"


# ============================================================================
# Scheduler Stop
# ============================================================================


class TestSchedulerStop:
    """Tests for stopping the scheduler."""

    def test_stop_scheduler_graceful(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._stop_scheduler(mock_http(body={"graceful": True}))
        body = _body(result)
        assert _status(result) == 200
        assert "graceful=True" in body["message"]

    def test_stop_scheduler_force(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._stop_scheduler(mock_http(body={"graceful": False}))
        body = _body(result)
        assert _status(result) == 200
        assert "graceful=False" in body["message"]

    def test_stop_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._stop_scheduler(mock_http())
        assert _status(result) == 503

    def test_stop_default_graceful(self, handler, mock_http):
        """Default graceful is True when no body provided."""
        mock_sched = _make_mock_scheduler()
        h = MockHTTPHandler(body=None)
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._stop_scheduler(h)
        body = _body(result)
        assert "graceful=True" in body["message"]

    def test_stop_invalid_body_uses_default(self, handler, mock_http):
        """Invalid JSON body falls back to graceful=True."""
        mock_sched = _make_mock_scheduler()
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers = {"Content-Length": "8", "Authorization": "Bearer test-valid-token"}
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._stop_scheduler(h)
        body = _body(result)
        assert "graceful=True" in body["message"]

    def test_stop_returns_state(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(state="stopped")
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._stop_scheduler(mock_http(body={}))
        body = _body(result)
        assert body["state"] == "stopped"


# ============================================================================
# Scheduler Pause
# ============================================================================


class TestSchedulerPause:
    """Tests for pausing the scheduler."""

    def test_pause_scheduler(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._pause_scheduler(mock_http())
        body = _body(result)
        assert _status(result) == 200
        assert body["message"] == "Scheduler paused"

    def test_pause_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._pause_scheduler(mock_http())
        assert _status(result) == 503

    def test_pause_returns_state(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(state="paused")
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._pause_scheduler(mock_http())
        body = _body(result)
        assert body["state"] == "paused"


# ============================================================================
# Scheduler Resume
# ============================================================================


class TestSchedulerResume:
    """Tests for resuming the scheduler."""

    def test_resume_scheduler(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._resume_scheduler(mock_http())
        body = _body(result)
        assert _status(result) == 200
        assert body["message"] == "Scheduler resumed"

    def test_resume_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._resume_scheduler(mock_http())
        assert _status(result) == 503

    def test_resume_returns_state(self, handler, mock_http):
        mock_sched = _make_mock_scheduler(state="running")
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            with patch("aragora.server.handlers.features.pulse.PulseHandler._run_async_safely"):
                result = handler._resume_scheduler(mock_http())
        body = _body(result)
        assert body["state"] == "running"


# ============================================================================
# Scheduler Config
# ============================================================================


class TestUpdateSchedulerConfig:
    """Tests for updating scheduler configuration."""

    def test_update_valid_config(self, handler, mock_http):
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"poll_interval_seconds": 600})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        mock_sched.update_config.assert_called_once_with({"poll_interval_seconds": 600})

    def test_update_no_scheduler(self, handler, mock_http):
        with patch("aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=None):
            result = handler._update_scheduler_config(mock_http(body={}))
        assert _status(result) == 503

    def test_update_empty_body(self, handler, mock_http):
        h = MockHTTPHandler(body=None)
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        assert _status(result) == 400

    def test_update_invalid_json(self, handler, mock_http):
        h = MockHTTPHandler()
        h.rfile.read.return_value = b"not json"
        h.headers = {"Content-Length": "8", "Authorization": "Bearer test-valid-token"}
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        assert _status(result) == 400

    def test_update_body_not_dict(self, handler, mock_http):
        body_bytes = json.dumps([1, 2, 3]).encode()
        h = MockHTTPHandler()
        h.rfile.read.return_value = body_bytes
        h.headers = {
            "Content-Length": str(len(body_bytes)),
            "Authorization": "Bearer test-valid-token",
        }
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        body = _body(result)
        assert _status(result) == 400
        assert "object" in body.get("error", "").lower()

    def test_update_invalid_keys(self, handler, mock_http):
        h = mock_http(body={"bad_key": "value", "another_bad": 1})
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        body = _body(result)
        assert _status(result) == 400
        assert "invalid" in body.get("error", "").lower() or "Invalid" in body.get("error", "")

    def test_update_all_valid_keys(self, handler, mock_http):
        """All valid config keys are accepted."""
        valid_config = {
            "poll_interval_seconds": 300,
            "platforms": ["hackernews"],
            "max_debates_per_hour": 6,
            "min_interval_between_debates": 60,
            "min_volume_threshold": 100,
            "min_controversy_score": 0.5,
            "allowed_categories": ["tech"],
            "blocked_categories": ["spam"],
            "dedup_window_hours": 24,
            "debate_rounds": 3,
            "consensus_threshold": 0.7,
        }
        h = mock_http(body=valid_config)
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        assert _status(result) == 200

    def test_update_mixed_valid_invalid_keys(self, handler, mock_http):
        """Mix of valid and invalid keys returns 400."""
        h = mock_http(body={"poll_interval_seconds": 300, "bad_key": "x"})
        mock_sched = _make_mock_scheduler()
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        assert _status(result) == 400

    def test_update_returns_config(self, handler, mock_http):
        """Response includes updated config."""
        mock_sched = _make_mock_scheduler()
        h = mock_http(body={"debate_rounds": 5})
        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler", return_value=mock_sched
        ):
            result = handler._update_scheduler_config(h)
        body = _body(result)
        assert "config" in body
        assert body["message"] == "Updated config keys: ['debate_rounds']"


# ============================================================================
# Scheduler History
# ============================================================================


class TestSchedulerHistory:
    """Tests for scheduler history endpoint."""

    def test_history_with_records(self, handler, mock_http):
        record = MagicMock()
        record.id = "r1"
        record.topic_text = "Topic 1"
        record.platform = "hackernews"
        record.category = "tech"
        record.volume = 100
        record.debate_id = "d1"
        record.created_at = "2026-01-01T00:00:00"
        record.hours_ago = 48.0
        record.consensus_reached = True
        record.confidence = 0.9
        record.rounds_used = 3
        record.scheduler_run_id = "run1"

        mock_store = _make_mock_store(records=[record], total=1)
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_scheduler_history(50, 0, None)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        assert body["total"] == 1
        assert body["debates"][0]["topic"] == "Topic 1"
        assert body["debates"][0]["scheduler_run_id"] == "run1"

    def test_history_no_store(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store", return_value=None
        ):
            result = handler._get_scheduler_history(50, 0, None)
        assert _status(result) == 503

    def test_history_with_platform_filter(self, handler, mock_http):
        mock_store = _make_mock_store(records=[], total=0)
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            handler._get_scheduler_history(50, 0, "hackernews")
        mock_store.get_history.assert_called_once_with(limit=50, offset=0, platform="hackernews")

    def test_history_limit_clamped(self, handler, mock_http):
        """Limit is clamped to max 100 in handle()."""
        with patch.object(handler, "_get_scheduler_history") as mock_h:
            mock_h.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/history", {"limit": "200"}, mock_http())
            mock_h.assert_called_once_with(100, 0, None)

    def test_history_default_params(self, handler, mock_http):
        """Default limit=50, offset=0, platform=None."""
        with patch.object(handler, "_get_scheduler_history") as mock_h:
            mock_h.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/history", {}, mock_http())
            mock_h.assert_called_once_with(50, 0, None)

    def test_history_with_offset(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_history") as mock_h:
            mock_h.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/history", {"offset": "10"}, mock_http())
            mock_h.assert_called_once_with(50, 10, None)

    def test_history_empty(self, handler, mock_http):
        mock_store = _make_mock_store(records=[], total=0)
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_scheduler_history(50, 0, None)
        body = _body(result)
        assert body["count"] == 0
        assert body["debates"] == []

    def test_history_pagination_params(self, handler, mock_http):
        mock_store = _make_mock_store(records=[], total=100)
        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            result = handler._get_scheduler_history(25, 50, None)
        body = _body(result)
        assert body["limit"] == 25
        assert body["offset"] == 50
        assert body["total"] == 100

    def test_history_with_platform_param(self, handler, mock_http):
        with patch.object(handler, "_get_scheduler_history") as mock_h:
            mock_h.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/pulse/scheduler/history", {"platform": "reddit"}, mock_http())
            mock_h.assert_called_once_with(50, 0, "reddit")


# ============================================================================
# _run_async_safely
# ============================================================================


class TestRunAsyncSafely:
    """Tests for the async runner helper."""

    def test_successful_run(self, handler):
        async def coro():
            return [1, 2, 3]

        with patch("aragora.server.handlers.features.pulse.run_async", return_value=[1, 2, 3]):
            result = handler._run_async_safely(coro)
        assert result == [1, 2, 3]

    def test_timeout_returns_empty_list(self, handler):
        import asyncio

        async def coro():
            raise asyncio.TimeoutError()

        with patch(
            "aragora.server.handlers.features.pulse.run_async", side_effect=asyncio.TimeoutError()
        ):
            result = handler._run_async_safely(coro)
        assert result == []

    def test_runtime_error_returns_empty_list(self, handler):
        with patch(
            "aragora.server.handlers.features.pulse.run_async",
            side_effect=RuntimeError("loop closed"),
        ):

            async def coro():
                pass

            result = handler._run_async_safely(coro)
        assert result == []

    def test_os_error_returns_empty_list(self, handler):
        with patch(
            "aragora.server.handlers.features.pulse.run_async", side_effect=OSError("network")
        ):

            async def coro():
                pass

            result = handler._run_async_safely(coro)
        assert result == []


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for PulseHandler initialization."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        h = PulseHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx(self):
        ctx = {"key": "value"}
        h = PulseHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none(self):
        h = PulseHandler()
        assert h.ctx == {}

    def test_init_server_context_overrides_ctx(self):
        h = PulseHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}

    def test_init_both_none(self):
        h = PulseHandler(ctx=None, server_context=None)
        assert h.ctx == {}


# ============================================================================
# Handle returns None for non-matching paths
# ============================================================================


class TestHandleRouteReturnsNone:
    """Tests that handle() returns None for unrecognized paths."""

    def test_handle_unrecognized_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/pulse/nope", {}, mock_http())
        assert result is None

    def test_handle_empty_path(self, handler, mock_http):
        result = handler.handle("", {}, mock_http())
        assert result is None

    def test_handle_partial_path(self, handler, mock_http):
        result = handler.handle("/api/v1/pulse", {}, mock_http())
        assert result is None

    def test_handle_wrong_version(self, handler, mock_http):
        result = handler.handle("/api/v2/pulse/trending", {}, mock_http())
        assert result is None


# ============================================================================
# Module-level singleton functions
# ============================================================================


class TestGetPulseManager:
    """Tests for get_pulse_manager singleton."""

    def test_returns_none_on_import_error(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            pulse_mod._shared_pulse_manager = None
            result = pulse_mod.get_pulse_manager()
            assert result is None

    def test_returns_cached_instance(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        mock_mgr = MagicMock()
        pulse_mod._shared_pulse_manager = mock_mgr
        result = pulse_mod.get_pulse_manager()
        assert result is mock_mgr


class TestGetPulseScheduler:
    """Tests for get_pulse_scheduler singleton."""

    def test_returns_none_when_manager_unavailable(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        with patch.object(pulse_mod, "get_pulse_manager", return_value=None):
            result = pulse_mod.get_pulse_scheduler()
            assert result is None

    def test_returns_none_when_store_unavailable(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        with patch.object(pulse_mod, "get_pulse_manager", return_value=MagicMock()):
            with patch.object(pulse_mod, "get_scheduled_debate_store", return_value=None):
                result = pulse_mod.get_pulse_scheduler()
                assert result is None

    def test_returns_cached_scheduler(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        mock_sched = MagicMock()
        pulse_mod._shared_scheduler = mock_sched
        result = pulse_mod.get_pulse_scheduler()
        assert result is mock_sched


class TestGetScheduledDebateStore:
    """Tests for get_scheduled_debate_store singleton."""

    def test_creates_lazy_store(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        mock_lazy = MagicMock()
        mock_lazy.get.return_value = MagicMock()
        with patch.object(pulse_mod, "_create_lazy_debate_store", return_value=mock_lazy):
            result = pulse_mod.get_scheduled_debate_store()
        assert result is not None

    def test_returns_cached_store(self):
        import aragora.server.handlers.features.pulse as pulse_mod

        mock_lazy = MagicMock()
        mock_lazy.get.return_value = "cached_store"
        pulse_mod._shared_debate_store = mock_lazy
        result = pulse_mod.get_scheduled_debate_store()
        assert result == "cached_store"


# ============================================================================
# ROUTES constant
# ============================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_list_contains_all_endpoints(self, handler):
        expected = [
            "/api/v1/pulse/trending",
            "/api/v1/pulse/suggest",
            "/api/v1/pulse/analytics",
            "/api/v1/pulse/debate-topic",
            "/api/v1/pulse/scheduler/status",
            "/api/v1/pulse/scheduler/start",
            "/api/v1/pulse/scheduler/stop",
            "/api/v1/pulse/scheduler/pause",
            "/api/v1/pulse/scheduler/resume",
            "/api/v1/pulse/scheduler/config",
            "/api/v1/pulse/scheduler/history",
            "/api/v1/pulse/scheduler/analytics",
        ]
        for route in expected:
            assert route in handler.ROUTES, f"Missing route: {route}"

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 12


# ============================================================================
# MAX_TOPIC_LENGTH
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_topic_length(self):
        from aragora.server.handlers.features.pulse import MAX_TOPIC_LENGTH

        assert MAX_TOPIC_LENGTH == 200

    def test_topic_at_exactly_max_length(self, handler, mock_http):
        """Topic at exactly max length is accepted (fails for no agents, not length)."""
        h = mock_http(body={"topic": "x" * 200})
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert "200" not in body.get("error", "")
        assert "exceeds" not in body.get("error", "").lower()

    def test_topic_at_max_plus_one(self, handler, mock_http):
        """Topic at 201 chars fails."""
        h = mock_http(body={"topic": "x" * 201})
        result = handler._start_debate_on_topic(h)
        body = _body(result)
        assert _status(result) == 400
        assert "200" in body.get("error", "")
