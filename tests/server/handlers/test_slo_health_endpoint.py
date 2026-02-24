"""Tests for the /api/health/slos debate SLO health endpoint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aragora.observability.debate_slos import (
    reset_debate_slo_tracker,
)


@pytest.fixture(autouse=True)
def _reset_slo_tracker():
    """Reset global debate SLO tracker between tests."""
    reset_debate_slo_tracker()
    yield
    reset_debate_slo_tracker()


@pytest.fixture()
def handler():
    """Create an SLO handler instance."""
    from aragora.server.handlers.slo import SLOHandler

    return SLOHandler()


@pytest.fixture()
def mock_http_handler():
    """Create a mock HTTP handler for rate limiting."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {}
    return h


def _parse_body(result) -> dict:
    """Parse HandlerResult body to dict."""
    body_bytes = result.body
    if isinstance(body_bytes, bytes):
        return json.loads(body_bytes)
    if isinstance(body_bytes, str):
        return json.loads(body_bytes)
    return body_bytes


class TestCanHandle:
    def test_handles_health_slos(self, handler):
        assert handler.can_handle("/api/health/slos") is True

    def test_handles_versioned_health_slos(self, handler):
        assert handler.can_handle("/api/v1/health/slos") is True


class TestHandleDebateSLOHealth:
    def test_returns_status_default_window(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.5)
        tracker.record_debate_completion(30.0)
        tracker.record_consensus_latency(0.2)
        tracker.record_dispatch_concurrency(0.9)

        result = handler._handle_debate_slo_health({}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

        body = _parse_body(result)
        data = body.get("data", body)
        assert "slos" in data
        assert "overall_healthy" in data
        assert "timestamp" in data

    def test_returns_specific_window(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)

        result = handler._handle_debate_slo_health({"window": "24h"}, mock_http_handler)
        assert result.status_code == 200

        body = _parse_body(result)
        data = body.get("data", body)
        assert "slos" in data

    def test_invalid_window_returns_400(self, handler, mock_http_handler):
        result = handler._handle_debate_slo_health(
            {"window": "invalid"}, mock_http_handler
        )
        assert result.status_code == 400

    def test_all_windows_returns_multi(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)

        result = handler._handle_debate_slo_health(
            {"all_windows": "true"}, mock_http_handler
        )
        assert result.status_code == 200

        body = _parse_body(result)
        data = body.get("data", body)
        assert "1h" in data
        assert "24h" in data
        assert "7d" in data

    def test_five_slos_present(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)
        tracker.record_debate_completion(30.0)
        for _ in range(10):
            tracker.record_websocket_reconnection(success=True)
        tracker.record_consensus_latency(0.2)
        tracker.record_dispatch_concurrency(0.9)

        result = handler._handle_debate_slo_health({}, mock_http_handler)
        body = _parse_body(result)
        data = body.get("data", body)
        slos = data.get("slos", {})
        assert "time_to_first_token" in slos
        assert "debate_completion" in slos
        assert "websocket_reconnection" in slos
        assert "consensus_detection" in slos
        assert "agent_dispatch_concurrency" in slos

    def test_each_slo_has_level(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)

        result = handler._handle_debate_slo_health({}, mock_http_handler)
        body = _parse_body(result)
        data = body.get("data", body)
        slos = data.get("slos", {})
        for slo_data in slos.values():
            assert "level" in slo_data
            assert slo_data["level"] in ("green", "yellow", "red")

    def test_compliant_fields(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        tracker.record_first_token_latency(1.0)
        tracker.record_debate_completion(20.0)
        for _ in range(100):
            tracker.record_websocket_reconnection(success=True)
        tracker.record_consensus_latency(0.1)
        tracker.record_dispatch_concurrency(0.95)

        result = handler._handle_debate_slo_health({}, mock_http_handler)
        body = _parse_body(result)
        data = body.get("data", body)
        assert data["overall_healthy"] is True
        assert data["overall_level"] == "green"

        for slo_data in data["slos"].values():
            assert slo_data["compliant"] is True
            assert "target" in slo_data
            assert "current" in slo_data
            assert "sample_count" in slo_data

    def test_degraded_slo_shows_yellow_or_red(self, handler, mock_http_handler):
        from aragora.observability.debate_slos import get_debate_slo_tracker

        tracker = get_debate_slo_tracker()
        # Record a latency above target but below critical
        tracker.record_first_token_latency(4.0)  # target is 3.0, warning at 4.5

        result = handler._handle_debate_slo_health({}, mock_http_handler)
        body = _parse_body(result)
        data = body.get("data", body)
        ttft = data["slos"]["time_to_first_token"]
        assert ttft["level"] in ("yellow", "red")
        assert ttft["compliant"] is False
