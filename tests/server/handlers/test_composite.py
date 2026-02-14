"""
Tests for CompositeHandler - composite API endpoints aggregating multiple subsystems.

Tests cover:
- CompositeCircuitBreaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Route matching (can_handle)
- Full context aggregation for debates
- Reliability metrics for agents
- Compression analysis endpoints
- Circuit breaker protected data fetching
- Rate limiting
- Error handling
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.composite import (
    CompositeCircuitBreaker,
    CompositeHandler,
    _clear_cached_components,
    get_circuit_breaker_status,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_state():
    """Clear cached components and circuit breakers between tests."""
    _clear_cached_components()
    yield
    _clear_cached_components()


@pytest.fixture
def handler():
    """Create a CompositeHandler instance with empty context."""
    return CompositeHandler({})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler for rate limit extraction."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {}
    return h


# ===========================================================================
# CompositeCircuitBreaker Tests
# ===========================================================================


class TestCompositeCircuitBreaker:
    """Tests for CompositeCircuitBreaker state machine."""

    def test_initial_state_is_closed(self):
        cb = CompositeCircuitBreaker()
        assert cb.state == "closed"

    def test_can_proceed_when_closed(self):
        cb = CompositeCircuitBreaker()
        assert cb.can_proceed() is True

    def test_opens_after_failure_threshold(self):
        cb = CompositeCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_cannot_proceed_when_open(self):
        cb = CompositeCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CompositeCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"

    def test_half_open_allows_limited_calls(self):
        cb = CompositeCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        assert cb.can_proceed() is True  # First call
        assert cb.can_proceed() is True  # Second call
        assert cb.can_proceed() is False  # Exceeds max

    def test_closes_after_successful_recovery(self):
        cb = CompositeCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()
        cb.record_success()
        cb.can_proceed()
        cb.record_success()
        assert cb.state == "closed"

    def test_reopens_on_failure_in_half_open(self):
        cb = CompositeCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()
        cb.record_failure()
        assert cb.state == "open"

    def test_success_in_closed_resets_failure_count(self):
        cb = CompositeCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Failure count reset, so one more failure should not open
        cb.record_failure()
        assert cb.state == "closed"

    def test_get_status_returns_dict(self):
        cb = CompositeCircuitBreaker()
        status = cb.get_status()
        assert isinstance(status, dict)
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert status["state"] == "closed"

    def test_reset_returns_to_closed(self):
        cb = CompositeCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"


# ===========================================================================
# Module-level Functions
# ===========================================================================


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_get_circuit_breaker_status_empty(self):
        status = get_circuit_breaker_status()
        assert isinstance(status, dict)
        assert len(status) == 0

    def test_clear_cached_components(self):
        _clear_cached_components()
        status = get_circuit_breaker_status()
        assert len(status) == 0


# ===========================================================================
# Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for CompositeHandler.can_handle route matching."""

    def test_handles_full_context(self, handler):
        assert handler.can_handle("/api/v1/debates/abc123/full-context") is True

    def test_handles_reliability(self, handler):
        assert handler.can_handle("/api/v1/agents/agent-1/reliability") is True

    def test_handles_compression_analysis(self, handler):
        assert handler.can_handle("/api/v1/debates/debate-1/compression-analysis") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/unknown/path") is False

    def test_rejects_unrelated_debate_suffix(self, handler):
        assert handler.can_handle("/api/v1/debates/abc/unknown-suffix") is False

    def test_routes_list(self, handler):
        assert len(handler.ROUTES) == 3


# ===========================================================================
# ID Extraction
# ===========================================================================


class TestExtractId:
    """Tests for _extract_id helper."""

    def test_extracts_debate_id(self, handler):
        result = handler._extract_id(
            "/api/v1/debates/test-123/full-context",
            "/api/v1/debates/",
            "/full-context",
        )
        assert result == "test-123"

    def test_extracts_agent_id(self, handler):
        result = handler._extract_id(
            "/api/v1/agents/agent-42/reliability",
            "/api/v1/agents/",
            "/reliability",
        )
        assert result == "agent-42"


# ===========================================================================
# Handle Method (full-context)
# ===========================================================================


class TestHandleFullContext:
    """Tests for full-context endpoint."""

    @patch("aragora.server.handlers.composite._composite_limiter")
    @patch("aragora.server.handlers.composite.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.server.handlers.composite.validate_path_segment", return_value=(True, None))
    def test_full_context_returns_200(
        self, mock_validate, mock_ip, mock_limiter, handler, mock_handler
    ):
        mock_limiter.is_allowed.return_value = True
        result = handler.handle(
            "/api/v1/debates/abc/full-context",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "abc"
        assert "memory" in body
        assert "knowledge" in body
        assert "belief" in body

    @patch("aragora.server.handlers.composite._composite_limiter")
    @patch("aragora.server.handlers.composite.get_client_ip", return_value="127.0.0.1")
    def test_rate_limit_returns_429(self, mock_ip, mock_limiter, handler, mock_handler):
        mock_limiter.is_allowed.return_value = False
        result = handler.handle(
            "/api/v1/debates/abc/full-context",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 429


# ===========================================================================
# Handle Method (reliability)
# ===========================================================================


class TestHandleReliability:
    """Tests for reliability endpoint."""

    @patch("aragora.server.handlers.composite._composite_limiter")
    @patch("aragora.server.handlers.composite.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.server.handlers.composite.validate_path_segment", return_value=(True, None))
    def test_reliability_returns_200(
        self, mock_validate, mock_ip, mock_limiter, handler, mock_handler
    ):
        mock_limiter.is_allowed.return_value = True
        result = handler.handle(
            "/api/v1/agents/agent-1/reliability",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent_id"] == "agent-1"
        assert "circuit_breaker" in body
        assert "airlock" in body
        assert "overall_score" in body

    @patch("aragora.server.handlers.composite._composite_limiter")
    @patch("aragora.server.handlers.composite.get_client_ip", return_value="127.0.0.1")
    @patch(
        "aragora.server.handlers.composite.validate_path_segment",
        return_value=(False, "Invalid agent_id"),
    )
    def test_invalid_id_returns_400(
        self, mock_validate, mock_ip, mock_limiter, handler, mock_handler
    ):
        mock_limiter.is_allowed.return_value = True
        result = handler.handle(
            "/api/v1/agents/<script>/reliability",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Handle Method (compression-analysis)
# ===========================================================================


class TestHandleCompressionAnalysis:
    """Tests for compression-analysis endpoint."""

    @patch("aragora.server.handlers.composite._composite_limiter")
    @patch("aragora.server.handlers.composite.get_client_ip", return_value="127.0.0.1")
    @patch("aragora.server.handlers.composite.validate_path_segment", return_value=(True, None))
    def test_compression_analysis_returns_200(
        self, mock_validate, mock_ip, mock_limiter, handler, mock_handler
    ):
        mock_limiter.is_allowed.return_value = True
        result = handler.handle(
            "/api/v1/debates/d-1/compression-analysis",
            {},
            mock_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "d-1"
        assert "compression" in body
        assert "quality" in body
        assert "recommendations" in body


# ===========================================================================
# Circuit Breaker Protected Fetching
# ===========================================================================


class TestFetchWithCircuitBreaker:
    """Tests for _fetch_with_circuit_breaker."""

    def test_returns_result_on_success(self, handler):
        result = handler._fetch_with_circuit_breaker(
            "test_subsystem",
            lambda: {"data": "value"},
            {"available": False},
        )
        assert result == {"data": "value"}

    def test_returns_fallback_on_failure(self, handler):
        def failing_func():
            raise ValueError("boom")

        result = handler._fetch_with_circuit_breaker(
            "test_subsystem",
            failing_func,
            {"available": False, "error": "fallback"},
        )
        assert result["available"] is False
        assert "error" in result

    def test_returns_none_fallback_on_failure(self, handler):
        def failing_func():
            raise RuntimeError("crash")

        result = handler._fetch_with_circuit_breaker(
            "test_subsystem",
            failing_func,
            None,
        )
        assert result is None


# ===========================================================================
# Reliability Score Calculation
# ===========================================================================


class TestReliabilityScore:
    """Tests for _calculate_reliability_score."""

    def test_perfect_score(self, handler):
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.0},
        }
        assert handler._calculate_reliability_score(metrics) == 1.0

    def test_open_circuit_breaker_penalizes(self, handler):
        metrics = {
            "circuit_breaker": {"state": "open"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score < 0.5

    def test_high_error_rate_penalizes(self, handler):
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.5},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score < 1.0


# ===========================================================================
# Compression Recommendations
# ===========================================================================


class TestCompressionRecommendations:
    """Tests for _generate_compression_recommendations."""

    def test_recommends_enabling_compression(self, handler):
        analysis = {"compression": {"enabled": False, "ratio": 0.0}, "quality": {}}
        recs = handler._generate_compression_recommendations(analysis)
        assert any("Enable" in r for r in recs)

    def test_recommends_increasing_compression(self, handler):
        analysis = {"compression": {"enabled": True, "ratio": 0.1}, "quality": {}}
        recs = handler._generate_compression_recommendations(analysis)
        assert any("increasing" in r.lower() for r in recs)

    def test_recommends_reducing_compression_when_quality_low(self, handler):
        analysis = {
            "compression": {"enabled": True, "ratio": 0.8},
            "quality": {"information_retained": 0.5},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("Reduce" in r for r in recs)

    def test_no_recommendations_when_optimal(self, handler):
        analysis = {
            "compression": {"enabled": True, "ratio": 0.5},
            "quality": {"information_retained": 0.95},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert len(recs) == 0
