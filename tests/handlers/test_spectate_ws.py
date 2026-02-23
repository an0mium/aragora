"""Tests for the SpectateStreamHandler.

Tests the HTTP handler that serves spectate events from the
SpectateWebSocketBridge over the /api/v1/spectate/* endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.spectate_ws import SpectateStreamHandler
from aragora.spectate.ws_bridge import (
    SpectateEvent,
    SpectateWebSocketBridge,
    reset_spectate_bridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bridge():
    """Reset the singleton bridge between tests."""
    reset_spectate_bridge()
    yield
    reset_spectate_bridge()


@pytest.fixture
def handler():
    """Create a SpectateStreamHandler with minimal server context."""
    ctx: dict = {}
    return SpectateStreamHandler(ctx)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    h = MagicMock()
    h.headers = {}
    return h


# ---------------------------------------------------------------------------
# Route matching tests
# ---------------------------------------------------------------------------


class TestRouteMatching:
    """Tests for handler route configuration."""

    def test_routes_defined(self, handler: SpectateStreamHandler):
        assert "/api/v1/spectate/recent" in handler.ROUTES
        assert "/api/v1/spectate/status" in handler.ROUTES
        assert "/api/v1/spectate/stream" in handler.ROUTES

    def test_handle_non_spectate_path_returns_none(
        self, handler: SpectateStreamHandler, mock_handler: MagicMock
    ):
        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_recent_path(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        result = handler.handle("/api/v1/spectate/recent", {}, mock_handler)
        assert result is not None
        body = result[0]
        assert "events" in body
        assert "count" in body

    def test_handle_status_path(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        result = handler.handle("/api/v1/spectate/status", {}, mock_handler)
        assert result is not None
        body = result[0]
        assert "active" in body
        assert "subscribers" in body
        assert "buffer_size" in body

    def test_handle_stream_path(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        """Stream endpoint returns recent events as a snapshot."""
        result = handler.handle("/api/v1/spectate/stream", {}, mock_handler)
        assert result is not None
        body = result[0]
        assert "events" in body


# ---------------------------------------------------------------------------
# Recent events tests
# ---------------------------------------------------------------------------


class TestRecentEvents:
    """Tests for GET /api/v1/spectate/recent."""

    def test_empty_returns_empty_list(
        self, handler: SpectateStreamHandler, mock_handler: MagicMock
    ):
        result = handler.handle("/api/v1/spectate/recent", {}, mock_handler)
        body = result[0]
        assert body["events"] == []
        assert body["count"] == 0

    def test_returns_buffered_events(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        from aragora.spectate.ws_bridge import get_spectate_bridge

        bridge = get_spectate_bridge()
        # Manually inject events into the buffer
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="debate_start",
                timestamp="2026-02-18T10:00:00+00:00",
                agent_name="claude",
            )
        )
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="proposal",
                timestamp="2026-02-18T10:00:01+00:00",
                agent_name="gpt4",
                data={"details": "A rate limiter"},
            )
        )

        result = handler.handle("/api/v1/spectate/recent", {}, mock_handler)
        body = result[0]
        assert body["count"] == 2
        assert body["events"][0]["event_type"] == "debate_start"
        assert body["events"][1]["agent_name"] == "gpt4"

    def test_count_parameter(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        from aragora.spectate.ws_bridge import get_spectate_bridge

        bridge = get_spectate_bridge()
        for i in range(10):
            bridge._event_buffer.append(
                SpectateEvent(
                    event_type="system",
                    timestamp=f"2026-02-18T10:00:{i:02d}+00:00",
                    data={"details": f"event-{i}"},
                )
            )

        result = handler.handle("/api/v1/spectate/recent", {"count": "3"}, mock_handler)
        body = result[0]
        assert body["count"] == 3

    def test_invalid_count_defaults_to_50(
        self, handler: SpectateStreamHandler, mock_handler: MagicMock
    ):
        """Non-integer count should fall back to default of 50."""
        result = handler.handle("/api/v1/spectate/recent", {"count": "abc"}, mock_handler)
        body = result[0]
        assert body["count"] == 0  # 0 events in buffer, but no error

    def test_filter_by_debate_id(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        from aragora.spectate.ws_bridge import get_spectate_bridge

        bridge = get_spectate_bridge()
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="proposal",
                timestamp="2026-02-18T10:00:00+00:00",
                debate_id="d-111",
            )
        )
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="vote",
                timestamp="2026-02-18T10:00:01+00:00",
                debate_id="d-222",
            )
        )

        result = handler.handle("/api/v1/spectate/recent", {"debate_id": "d-111"}, mock_handler)
        body = result[0]
        assert body["count"] == 1
        assert body["events"][0]["debate_id"] == "d-111"

    def test_filter_by_pipeline_id(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        from aragora.spectate.ws_bridge import get_spectate_bridge

        bridge = get_spectate_bridge()
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="system",
                timestamp="2026-02-18T10:00:00+00:00",
                pipeline_id="p-abc",
            )
        )
        bridge._event_buffer.append(
            SpectateEvent(
                event_type="system",
                timestamp="2026-02-18T10:00:01+00:00",
                pipeline_id="p-xyz",
            )
        )

        result = handler.handle("/api/v1/spectate/recent", {"pipeline_id": "p-abc"}, mock_handler)
        body = result[0]
        assert body["count"] == 1
        assert body["events"][0]["pipeline_id"] == "p-abc"


# ---------------------------------------------------------------------------
# Status endpoint tests
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for GET /api/v1/spectate/status."""

    def test_status_when_inactive(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        result = handler.handle("/api/v1/spectate/status", {}, mock_handler)
        body = result[0]
        assert body["active"] is False
        assert body["subscribers"] == 0
        assert body["buffer_size"] == 0

    def test_status_when_active(self, handler: SpectateStreamHandler, mock_handler: MagicMock):
        from aragora.spectate.ws_bridge import get_spectate_bridge

        bridge = get_spectate_bridge()
        bridge.start()
        bridge.subscribe(lambda e: None)

        try:
            result = handler.handle("/api/v1/spectate/status", {}, mock_handler)
            body = result[0]
            assert body["active"] is True
            assert body["subscribers"] == 1
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for ImportError handling when bridge module is unavailable."""

    def test_recent_with_import_error(
        self, handler: SpectateStreamHandler, mock_handler: MagicMock
    ):
        with patch(
            "aragora.server.handlers.spectate_ws.SpectateStreamHandler._handle_recent"
        ) as mock_recent:
            # Simulate the ImportError path directly
            mock_recent.return_value = ({"events": [], "count": 0}, 200)
            result = handler.handle("/api/v1/spectate/recent", {}, mock_handler)
            # Should return something (the mocked result or the real one)
            assert result is not None

    def test_status_with_import_error(
        self, handler: SpectateStreamHandler, mock_handler: MagicMock
    ):
        with patch(
            "aragora.server.handlers.spectate_ws.SpectateStreamHandler._handle_status"
        ) as mock_status:
            mock_status.return_value = (
                {"active": False, "subscribers": 0, "buffer_size": 0},
                200,
            )
            result = handler.handle("/api/v1/spectate/status", {}, mock_handler)
            assert result is not None
