"""
Tests for aragora.server.handlers.features.pulse module.

Tests Pulse (trending topics) handler including:
- Trending topics retrieval
- Topic suggestions for debates
- Analytics endpoints
- Scheduler lifecycle (start/stop/pause/resume)
- Scheduler configuration
- Scheduler history
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Mock TrendingTopic for testing
@dataclass
class MockTrendingTopic:
    """Mock trending topic for testing."""

    topic: str
    platform: str
    volume: int
    category: str = "tech"

    def to_debate_prompt(self) -> str:
        return f"Debate: {self.topic}"


class MockSchedulerState(str, Enum):
    """Mock scheduler state."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class MockSchedulerConfig:
    """Mock scheduler config."""

    poll_interval_seconds: int = 300
    platforms: list = field(default_factory=lambda: ["hackernews", "reddit"])
    max_debates_per_hour: int = 6

    def to_dict(self) -> dict:
        return {
            "poll_interval_seconds": self.poll_interval_seconds,
            "platforms": self.platforms,
            "max_debates_per_hour": self.max_debates_per_hour,
        }


@dataclass
class MockScheduledDebateRecord:
    """Mock scheduled debate record."""

    id: str = "rec_001"
    topic_text: str = "AI Ethics"
    platform: str = "hackernews"
    category: str = "tech"
    volume: int = 500
    debate_id: str = "debate_001"
    created_at: str = "2026-01-15T10:00:00Z"
    hours_ago: float = 2.5
    consensus_reached: bool = True
    confidence: float = 0.85
    rounds_used: int = 3
    scheduler_run_id: str = "run_001"


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, headers: dict = None, body: bytes = b""):
        self.headers = headers or {}
        self.rfile = BytesIO(body)
        self.client_address = ("127.0.0.1", 12345)


class TestPulseHandlerRouting:
    """Test PulseHandler route detection."""

    def test_can_handle_trending(self):
        """Test handler recognizes trending endpoint."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        assert handler.can_handle("/api/pulse/trending") is True

    def test_can_handle_suggest(self):
        """Test handler recognizes suggest endpoint."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        assert handler.can_handle("/api/pulse/suggest") is True

    def test_can_handle_analytics(self):
        """Test handler recognizes analytics endpoint."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        assert handler.can_handle("/api/pulse/analytics") is True

    def test_can_handle_scheduler_endpoints(self):
        """Test handler recognizes all scheduler endpoints."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        scheduler_endpoints = [
            "/api/pulse/scheduler/status",
            "/api/pulse/scheduler/start",
            "/api/pulse/scheduler/stop",
            "/api/pulse/scheduler/pause",
            "/api/pulse/scheduler/resume",
            "/api/pulse/scheduler/config",
            "/api/pulse/scheduler/history",
        ]
        for endpoint in scheduler_endpoints:
            assert handler.can_handle(endpoint) is True, f"Should handle {endpoint}"

    def test_cannot_handle_unknown(self):
        """Test handler rejects unknown paths."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        assert handler.can_handle("/api/other/endpoint") is False

    def test_routes_list(self):
        """Test ROUTES list contains expected endpoints."""
        from aragora.server.handlers.features.pulse import PulseHandler

        expected_routes = [
            "/api/pulse/trending",
            "/api/pulse/suggest",
            "/api/pulse/analytics",
            "/api/pulse/debate-topic",
            "/api/pulse/scheduler/status",
            "/api/pulse/scheduler/start",
            "/api/pulse/scheduler/stop",
            "/api/pulse/scheduler/pause",
            "/api/pulse/scheduler/resume",
            "/api/pulse/scheduler/config",
            "/api/pulse/scheduler/history",
        ]
        for route in expected_routes:
            assert route in PulseHandler.ROUTES


class TestPulseAnalytics:
    """Test analytics endpoint."""

    def test_analytics_returns_data(self):
        """Test successful analytics retrieval."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_analytics = {
            "total_debates": 50,
            "consensus_rate": 0.72,
            "avg_confidence": 0.68,
            "by_platform": {"hackernews": 30, "reddit": 20},
            "by_category": {"tech": 25, "ai": 25},
        }

        mock_manager = MagicMock()
        mock_manager.get_analytics.return_value = mock_analytics

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=mock_manager,
        ):
            handler = PulseHandler({})
            # Access underlying method, bypassing cache and auto_error_response decorators
            result = handler._get_analytics.__wrapped__.__wrapped__(handler)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["total_debates"] == 50
            assert body["consensus_rate"] == 0.72

    def test_analytics_feature_unavailable(self):
        """Test analytics when pulse not available."""
        from aragora.server.handlers.features.pulse import PulseHandler

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_manager",
            return_value=None,
        ):
            handler = PulseHandler({})
            # Access underlying method, bypassing cache and auto_error_response decorators
            result = handler._get_analytics.__wrapped__.__wrapped__(handler)

            assert result.status_code == 503


class TestSchedulerStatus:
    """Test scheduler status endpoint."""

    def test_status_returns_info(self):
        """Test successful scheduler status retrieval."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.get_status.return_value = {
            "state": "running",
            "config": {"poll_interval_seconds": 300},
            "metrics": {"debates_created": 10},
        }

        mock_store = MagicMock()
        mock_store.get_analytics.return_value = {"total_records": 100}

        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ),
            patch(
                "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
                return_value=mock_store,
            ),
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_status()

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["state"] == "running"
            assert "store_analytics" in body

    def test_status_scheduler_unavailable(self):
        """Test status when scheduler not available."""
        from aragora.server.handlers.features.pulse import PulseHandler

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=None,
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_status()

            assert result.status_code == 503


class TestSchedulerLifecycle:
    """Test scheduler start/stop/pause/resume."""

    def test_start_scheduler(self):
        """Test starting the scheduler."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.state = MockSchedulerState.RUNNING
        mock_scheduler._debate_creator = None

        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ),
            patch.object(
                PulseHandler,
                "_run_async_safely",
                return_value=None,
            ),
        ):
            handler = PulseHandler({})
            mock_http = MockHandler(headers={"Authorization": "Bearer test"})

            # Access underlying method directly (skip decorators)
            result = handler._start_scheduler.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["state"] == "running"

    def test_stop_scheduler_graceful(self):
        """Test gracefully stopping the scheduler."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.state = MockSchedulerState.STOPPED

        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ),
            patch.object(
                PulseHandler,
                "_run_async_safely",
                return_value=None,
            ),
        ):
            handler = PulseHandler({})
            body_bytes = json.dumps({"graceful": True}).encode()
            mock_http = MockHandler(
                headers={"Content-Length": str(len(body_bytes))},
                body=body_bytes,
            )

            result = handler._stop_scheduler.__wrapped__.__wrapped__.__wrapped__(handler, mock_http)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert "graceful=True" in body["message"]

    def test_pause_scheduler(self):
        """Test pausing the scheduler."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.state = MockSchedulerState.PAUSED

        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ),
            patch.object(
                PulseHandler,
                "_run_async_safely",
                return_value=None,
            ),
        ):
            handler = PulseHandler({})
            mock_http = MockHandler()

            result = handler._pause_scheduler.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["state"] == "paused"

    def test_resume_scheduler(self):
        """Test resuming the scheduler."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.state = MockSchedulerState.RUNNING

        with (
            patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ),
            patch.object(
                PulseHandler,
                "_run_async_safely",
                return_value=None,
            ),
        ):
            handler = PulseHandler({})
            mock_http = MockHandler()

            result = handler._resume_scheduler.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            assert body["state"] == "running"

    def test_scheduler_lifecycle_unavailable(self):
        """Test scheduler controls when unavailable."""
        from aragora.server.handlers.features.pulse import PulseHandler

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=None,
        ):
            handler = PulseHandler({})
            mock_http = MockHandler()

            # Test all lifecycle methods return 503 when scheduler unavailable
            for method_name in [
                "_start_scheduler",
                "_stop_scheduler",
                "_pause_scheduler",
                "_resume_scheduler",
            ]:
                method = getattr(handler, method_name)
                # Access wrapped method
                result = method.__wrapped__.__wrapped__.__wrapped__(handler, mock_http)
                assert result.status_code == 503


class TestSchedulerConfig:
    """Test scheduler configuration updates."""

    def test_update_config_success(self):
        """Test successful config update."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.config = MockSchedulerConfig()

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            handler = PulseHandler({})
            body_bytes = json.dumps({"poll_interval_seconds": 600}).encode()
            mock_http = MockHandler(
                headers={"Content-Length": str(len(body_bytes))},
                body=body_bytes,
            )

            result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["success"] is True
            mock_scheduler.update_config.assert_called_once()

    def test_update_config_invalid_keys(self):
        """Test config update with invalid keys."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            handler = PulseHandler({})
            body_bytes = json.dumps({"invalid_key": "value"}).encode()
            mock_http = MockHandler(
                headers={"Content-Length": str(len(body_bytes))},
                body=body_bytes,
            )

            result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 400
            body = json.loads(result.body)
            assert "invalid_key" in body["error"]

    def test_update_config_empty_body(self):
        """Test config update with empty body."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()

        with patch(
            "aragora.server.handlers.features.pulse.get_pulse_scheduler",
            return_value=mock_scheduler,
        ):
            handler = PulseHandler({})
            mock_http = MockHandler(headers={"Content-Length": "0"})

            result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                handler, mock_http
            )

            assert result.status_code == 400

    def test_update_config_valid_keys(self):
        """Test config update with various valid keys."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_scheduler = MagicMock()
        mock_scheduler.config = MockSchedulerConfig()

        valid_configs = [
            {"poll_interval_seconds": 600},
            {"platforms": ["hackernews", "twitter"]},
            {"max_debates_per_hour": 10},
            {"min_volume_threshold": 200},
            {"allowed_categories": ["ai", "science"]},
            {"blocked_categories": ["politics"]},
            {"debate_rounds": 5},
        ]

        for config in valid_configs:
            with patch(
                "aragora.server.handlers.features.pulse.get_pulse_scheduler",
                return_value=mock_scheduler,
            ):
                handler = PulseHandler({})
                body_bytes = json.dumps(config).encode()
                mock_http = MockHandler(
                    headers={"Content-Length": str(len(body_bytes))},
                    body=body_bytes,
                )

                result = handler._update_scheduler_config.__wrapped__.__wrapped__.__wrapped__(
                    handler, mock_http
                )

                assert result.status_code == 200, f"Failed for config: {config}"


class TestSchedulerHistory:
    """Test scheduler history endpoint."""

    def test_history_returns_records(self):
        """Test successful history retrieval."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_records = [
            MockScheduledDebateRecord(),
            MockScheduledDebateRecord(
                id="rec_002",
                topic_text="Quantum Computing",
                debate_id="debate_002",
            ),
        ]

        mock_store = MagicMock()
        mock_store.get_history.return_value = mock_records
        mock_store.count_total.return_value = 100

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_history(50, 0, None)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["count"] == 2
            assert body["total"] == 100
            assert body["debates"][0]["topic"] == "AI Ethics"

    def test_history_with_platform_filter(self):
        """Test history with platform filter."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_store = MagicMock()
        mock_store.get_history.return_value = []
        mock_store.count_total.return_value = 0

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_history(50, 0, "hackernews")

            mock_store.get_history.assert_called_once_with(
                limit=50, offset=0, platform="hackernews"
            )

    def test_history_pagination(self):
        """Test history pagination."""
        from aragora.server.handlers.features.pulse import PulseHandler

        mock_store = MagicMock()
        mock_store.get_history.return_value = []
        mock_store.count_total.return_value = 200

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=mock_store,
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_history(25, 50, None)

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["limit"] == 25
            assert body["offset"] == 50
            assert body["total"] == 200

    def test_history_store_unavailable(self):
        """Test history when store not available."""
        from aragora.server.handlers.features.pulse import PulseHandler

        with patch(
            "aragora.server.handlers.features.pulse.get_scheduled_debate_store",
            return_value=None,
        ):
            handler = PulseHandler({})

            result = handler._get_scheduler_history(50, 0, None)

            assert result.status_code == 503


class TestStartDebateOnTopic:
    """Test starting a debate on a trending topic."""

    def test_start_debate_missing_topic(self):
        """Test debate start without topic."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        body_bytes = json.dumps({"agents": ["anthropic-api"]}).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body_bytes))},
            body=body_bytes,
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "topic" in body["error"]

    def test_start_debate_empty_topic(self):
        """Test debate start with empty topic."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        body_bytes = json.dumps({"topic": "   "}).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body_bytes))},
            body=body_bytes,
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "topic" in body["error"]

    def test_start_debate_topic_too_long(self):
        """Test debate start with topic exceeding max length."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        long_topic = "A" * 250  # Exceeds MAX_TOPIC_LENGTH (200)
        body_bytes = json.dumps({"topic": long_topic}).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body_bytes))},
            body=body_bytes,
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "200" in body["error"]

    def test_start_debate_topic_invalid_chars(self):
        """Test debate start with invalid characters in topic."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        body_bytes = json.dumps({"topic": "Test\x00topic"}).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body_bytes))},
            body=body_bytes,
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid" in body["error"].lower()

    def test_start_debate_invalid_consensus(self):
        """Test debate start with invalid consensus mode."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        body_bytes = json.dumps(
            {
                "topic": "Test topic",
                "consensus": "invalid_mode",
            }
        ).encode()
        mock_http = MockHandler(
            headers={"Content-Length": str(len(body_bytes))},
            body=body_bytes,
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consensus" in body["error"]

    def test_start_debate_no_body(self):
        """Test debate start without request body."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler(headers={"Content-Length": "0"})

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400

    def test_start_debate_invalid_json(self):
        """Test debate start with invalid JSON body."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler(
            headers={"Content-Length": "10"},
            body=b"not json!",
        )

        result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "JSON" in body["error"]

    def test_start_debate_valid_consensus_modes(self):
        """Test that all valid consensus modes are accepted."""
        from aragora.server.handlers.features.pulse import PulseHandler

        valid_modes = ["majority", "unanimous", "judge", "none"]

        for mode in valid_modes:
            handler = PulseHandler({})
            body_bytes = json.dumps(
                {
                    "topic": "Test topic",
                    "consensus": mode,
                }
            ).encode()
            mock_http = MockHandler(
                headers={"Content-Length": str(len(body_bytes))},
                body=body_bytes,
            )

            # Patch the imports that happen inside the function
            mock_arena = MagicMock()
            mock_env = MagicMock()
            mock_protocol = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "aragora": MagicMock(
                        Arena=mock_arena,
                        Environment=mock_env,
                        DebateProtocol=mock_protocol,
                    ),
                    "aragora.agents": MagicMock(get_agents_by_names=MagicMock(return_value=[])),
                },
            ):
                result = handler._start_debate_on_topic.__wrapped__.__wrapped__(handler, mock_http)
                # 400 for no agents is expected, not 400 for invalid consensus
                if result.status_code == 400:
                    body = json.loads(result.body)
                    assert "consensus" not in body["error"].lower(), f"Mode {mode} should be valid"


class TestPulseHandlerHandleMethods:
    """Test handle() and handle_post() routing."""

    def test_handle_routes_to_trending(self):
        """Test handle() routes trending requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_trending_topics",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/trending", {}, mock_http)

            mock_method.assert_called_once_with(10)  # Default limit

    def test_handle_routes_with_limit(self):
        """Test handle() routes trending with custom limit."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_trending_topics",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/trending", {"limit": ["25"]}, mock_http)

            mock_method.assert_called_once_with(25)

    def test_handle_limits_max(self):
        """Test handle() caps trending limit at 50."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_trending_topics",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/trending", {"limit": ["100"]}, mock_http)

            mock_method.assert_called_once_with(50)  # Capped at 50

    def test_handle_routes_to_suggest(self):
        """Test handle() routes suggest requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_suggest_debate_topic",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle(
                "/api/pulse/suggest",
                {"category": ["ai"]},
                mock_http,
            )

            mock_method.assert_called_once_with("ai")

    def test_handle_routes_to_analytics(self):
        """Test handle() routes analytics requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_analytics",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/analytics", {}, mock_http)

            mock_method.assert_called_once()

    def test_handle_routes_to_scheduler_status(self):
        """Test handle() routes scheduler status requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_scheduler_status",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/scheduler/status", {}, mock_http)

            mock_method.assert_called_once()

    def test_handle_routes_to_scheduler_history(self):
        """Test handle() routes scheduler history requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_get_scheduler_history",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle("/api/pulse/scheduler/history", {}, mock_http)

            mock_method.assert_called_once_with(50, 0, None)

    def test_handle_unknown_returns_none(self):
        """Test handle() returns None for unknown paths."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        result = handler.handle("/api/unknown", {}, mock_http)

        assert result is None

    def test_handle_post_routes_to_start_debate(self):
        """Test handle_post() routes debate start requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_start_debate_on_topic",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle_post("/api/pulse/debate-topic", {}, mock_http)

            mock_method.assert_called_once_with(mock_http)

    def test_handle_post_routes_scheduler_controls(self):
        """Test handle_post() routes scheduler control requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        controls = [
            ("/api/pulse/scheduler/start", "_start_scheduler"),
            ("/api/pulse/scheduler/stop", "_stop_scheduler"),
            ("/api/pulse/scheduler/pause", "_pause_scheduler"),
            ("/api/pulse/scheduler/resume", "_resume_scheduler"),
        ]

        for path, method_name in controls:
            with patch.object(
                handler,
                method_name,
                return_value={"status": 200, "body": "{}"},
            ) as mock_method:
                result = handler.handle_post(path, {}, mock_http)
                mock_method.assert_called_once()

    def test_handle_post_unknown_returns_none(self):
        """Test handle_post() returns None for unknown paths."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        result = handler.handle_post("/api/unknown", {}, mock_http)

        assert result is None

    def test_handle_patch_routes_config(self):
        """Test handle_patch() routes config requests."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "_update_scheduler_config",
            return_value={"status": 200, "body": "{}"},
        ) as mock_method:
            result = handler.handle_patch(
                "/api/pulse/scheduler/config",
                {},
                mock_http,
            )

            mock_method.assert_called_once_with(mock_http)

    def test_handle_patch_unknown_returns_none(self):
        """Test handle_patch() returns None for unknown paths."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        result = handler.handle_patch("/api/unknown", {}, mock_http)

        assert result is None


class TestCategoryValidation:
    """Test category validation in suggest endpoint."""

    def test_suggest_rejects_invalid_category(self):
        """Test suggest rejects invalid category format."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        # Invalid category with special characters
        result = handler.handle(
            "/api/pulse/suggest",
            {"category": ["<script>alert(1)</script>"]},
            mock_http,
        )

        assert result.status_code == 400

    def test_suggest_accepts_valid_category(self):
        """Test suggest accepts valid category format."""
        from aragora.server.handlers.features.pulse import PulseHandler

        handler = PulseHandler({})
        mock_http = MockHandler()

        valid_categories = ["tech", "ai", "science", "programming", "tech_news"]

        for category in valid_categories:
            with patch.object(
                handler,
                "_suggest_debate_topic",
                return_value={"status": 200, "body": "{}"},
            ):
                result = handler.handle(
                    "/api/pulse/suggest",
                    {"category": [category]},
                    mock_http,
                )
                # Should not be rejected for invalid category
                assert result is not None


class TestMaxTopicLength:
    """Test MAX_TOPIC_LENGTH constant."""

    def test_max_topic_length_defined(self):
        """Test MAX_TOPIC_LENGTH is defined and reasonable."""
        from aragora.server.handlers.features.pulse import MAX_TOPIC_LENGTH

        assert MAX_TOPIC_LENGTH == 200
        assert isinstance(MAX_TOPIC_LENGTH, int)
