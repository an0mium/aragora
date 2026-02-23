"""Tests for composite handler (aragora/server/handlers/composite.py).

Covers all routes and behavior of the CompositeHandler class:
- can_handle() routing for all ROUTES and non-matching paths
- GET /api/v1/debates/{id}/full-context - Memory + Knowledge + Belief context
- GET /api/v1/agents/{id}/reliability - Circuit breaker + Airlock metrics
- GET /api/v1/debates/{id}/compression-analysis - RLM compression metrics
- Rate limiting (429 responses)
- Input validation (invalid IDs)
- Circuit breaker protection for subsystem access
- Error isolation (subsystem failures return fallback values)
- Module-level helpers (_get_circuit_breaker, get_circuit_breaker_status, _clear_cached_components)
- Reliability score calculation
- Compression recommendations generation
- RBAC permission checks (via conftest auto-bypass)
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
    _get_circuit_breaker,
    get_circuit_breaker_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw)
        return raw
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return 0


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler and rate limiting."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_circuit_breakers():
    """Clear circuit breakers before/after each test."""
    _clear_cached_components()
    yield
    _clear_cached_components()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter state between tests."""
    with patch(
        "aragora.server.handlers.composite._composite_limiter"
    ) as mock_limiter:
        mock_limiter.is_allowed.return_value = True
        yield mock_limiter


@pytest.fixture
def handler():
    """Create a CompositeHandler instance with empty server context."""
    return CompositeHandler(server_context={})


@pytest.fixture
def http():
    """Create a default mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def handler_with_memory():
    """Handler with a mock continuum memory in context."""
    mock_memory = MagicMock()
    mock_memory.recall.return_value = []
    ctx = {"continuum_memory": mock_memory}
    return CompositeHandler(server_context=ctx), mock_memory


@pytest.fixture
def handler_with_knowledge():
    """Handler with a mock knowledge mound in context."""
    mock_km = MagicMock()
    mock_km.query.return_value = ["fact-1", "fact-2"]
    ctx = {"knowledge_mound": mock_km}
    return CompositeHandler(server_context=ctx), mock_km


@pytest.fixture
def handler_with_belief():
    """Handler with a mock dissent retriever in context."""
    mock_dissent = MagicMock()
    mock_dissent.get_cruxes.return_value = ["crux-1"]
    ctx = {"dissent_retriever": mock_dissent}
    return CompositeHandler(server_context=ctx), mock_dissent


@pytest.fixture
def handler_with_airlock():
    """Handler with a mock airlock registry in context."""
    mock_metrics = MagicMock()
    mock_metrics.total_calls = 100
    mock_metrics.fallback_responses = 5
    mock_metrics.avg_latency_ms = 120.5
    mock_metrics.success_rate = 95.0
    mock_proxy = MagicMock()
    mock_proxy.metrics = mock_metrics
    ctx = {"airlock_registry": {"agent-1": mock_proxy}}
    return CompositeHandler(server_context=ctx)


@pytest.fixture
def handler_with_rlm():
    """Handler with a mock RLM handler in context."""
    mock_rlm = MagicMock()
    mock_rlm.get_compression_stats.return_value = {
        "compression": {
            "rounds_compressed": 3,
            "original_tokens": 5000,
            "compressed_tokens": 2000,
            "ratio": 0.4,
            "savings_percent": 60.0,
        },
        "quality": {
            "information_retained": 0.95,
            "coherence_score": 0.88,
        },
    }
    ctx = {"rlm_handler": mock_rlm}
    return CompositeHandler(server_context=ctx), mock_rlm


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_full_context(self, handler):
        assert handler.can_handle("/api/v1/debates/debate-123/full-context") is True

    def test_handles_reliability(self, handler):
        assert handler.can_handle("/api/v1/agents/agent-1/reliability") is True

    def test_handles_compression_analysis(self, handler):
        assert handler.can_handle("/api/v1/debates/debate-456/compression-analysis") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/debates/abc/other") is False

    def test_rejects_agents_without_reliability(self, handler):
        assert handler.can_handle("/api/v1/agents/agent-1/status") is False

    def test_rejects_debates_without_valid_suffix(self, handler):
        assert handler.can_handle("/api/v1/debates/abc/context") is False

    def test_handles_full_context_with_uuid_id(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-def-123/full-context") is True

    def test_handles_reliability_with_complex_id(self, handler):
        assert handler.can_handle("/api/v1/agents/my_agent-v2/reliability") is True


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutesAttribute:
    """Verify the ROUTES class attribute lists all expected patterns."""

    def test_routes_contains_full_context(self):
        assert "/api/v1/debates/*/full-context" in CompositeHandler.ROUTES

    def test_routes_contains_reliability(self):
        assert "/api/v1/agents/*/reliability" in CompositeHandler.ROUTES

    def test_routes_contains_compression_analysis(self):
        assert "/api/v1/debates/*/compression-analysis" in CompositeHandler.ROUTES

    def test_routes_count(self):
        assert len(CompositeHandler.ROUTES) == 3


# ===========================================================================
# GET /api/v1/debates/{id}/full-context
# ===========================================================================


class TestFullContext:
    """Test GET /api/v1/debates/{id}/full-context."""

    def test_full_context_basic_response(self, handler, http):
        result = handler.handle("/api/v1/debates/debate-1/full-context", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-1"
        assert "timestamp" in body
        assert "memory" in body
        assert "knowledge" in body
        assert "belief" in body

    def test_full_context_memory_unavailable(self, handler, http):
        """When no continuum memory in context, memory section shows unavailable."""
        result = handler.handle("/api/v1/debates/debate-1/full-context", {}, http)
        body = _body(result)
        assert body["memory"]["available"] is False

    def test_full_context_knowledge_unavailable(self, handler, http):
        """When no knowledge mound in context, knowledge section shows unavailable."""
        result = handler.handle("/api/v1/debates/debate-1/full-context", {}, http)
        body = _body(result)
        assert body["knowledge"]["available"] is False

    def test_full_context_belief_unavailable(self, handler, http):
        """When no dissent retriever in context, belief section shows unavailable."""
        result = handler.handle("/api/v1/debates/debate-1/full-context", {}, http)
        body = _body(result)
        assert body["belief"]["available"] is False

    def test_full_context_with_memory(self, handler_with_memory, http):
        """When continuum memory is available, memory data is returned."""
        handler, mock_mem = handler_with_memory
        # Need to patch ContinuumMemory import for isinstance check
        with patch(
            "aragora.server.handlers.composite.CompositeHandler._get_memory_context"
        ) as mock_get_mem:
            mock_get_mem.return_value = {
                "available": True,
                "outcomes": [{"id": "o1"}],
                "patterns": [],
                "related_debates": [],
            }
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        assert body["memory"]["available"] is True
        assert body["memory"]["outcomes"] == [{"id": "o1"}]

    def test_full_context_with_knowledge(self, handler_with_knowledge, http):
        """When knowledge mound is available, knowledge data is returned."""
        handler, mock_km = handler_with_knowledge
        with patch(
            "aragora.server.handlers.composite.CompositeHandler._get_knowledge_context"
        ) as mock_get_km:
            mock_get_km.return_value = {
                "available": True,
                "facts": ["fact-1", "fact-2"],
                "concepts": [],
                "sources": [],
            }
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        assert body["knowledge"]["available"] is True
        assert len(body["knowledge"]["facts"]) == 2

    def test_full_context_with_belief(self, handler_with_belief, http):
        """When dissent retriever is available, belief data is returned."""
        handler, mock_dissent = handler_with_belief
        with patch(
            "aragora.server.handlers.composite.CompositeHandler._get_belief_context"
        ) as mock_get_belief:
            mock_get_belief.return_value = {
                "available": True,
                "cruxes": ["crux-1"],
                "positions": [],
                "confidence_distribution": {},
            }
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        assert body["belief"]["available"] is True
        assert body["belief"]["cruxes"] == ["crux-1"]

    def test_full_context_invalid_id_path_traversal(self, handler, http):
        """IDs with path traversal characters are rejected."""
        result = handler.handle(
            "/api/v1/debates/../etc/passwd/full-context", {}, http
        )
        assert _status(result) == 400

    def test_full_context_invalid_id_special_chars(self, handler, http):
        """IDs with special characters are rejected."""
        result = handler.handle(
            "/api/v1/debates/<script>alert(1)</script>/full-context", {}, http
        )
        assert _status(result) == 400

    def test_full_context_empty_id(self, handler, http):
        """Empty debate ID is rejected."""
        result = handler.handle("/api/v1/debates//full-context", {}, http)
        assert _status(result) == 400

    def test_full_context_json_content_type(self, handler, http):
        result = handler.handle("/api/v1/debates/debate-1/full-context", {}, http)
        assert result.content_type == "application/json"


# ===========================================================================
# GET /api/v1/agents/{id}/reliability
# ===========================================================================


class TestReliability:
    """Test GET /api/v1/agents/{id}/reliability."""

    def test_reliability_basic_response(self, handler, http):
        result = handler.handle("/api/v1/agents/agent-1/reliability", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent-1"
        assert "timestamp" in body
        assert "circuit_breaker" in body
        assert "airlock" in body
        assert "availability" in body
        assert "overall_score" in body

    def test_reliability_default_availability(self, handler, http):
        """Default availability values when no data is available."""
        result = handler.handle("/api/v1/agents/agent-1/reliability", {}, http)
        body = _body(result)
        avail = body["availability"]
        assert avail["available"] is True
        assert avail["uptime_percent"] == 99.9

    def test_reliability_circuit_breaker_unknown(self, handler, http):
        """Circuit breaker shows unknown when not found."""
        with patch(
            "aragora.server.handlers.composite.CompositeHandler._get_circuit_breaker_state"
        ) as mock_cb:
            mock_cb.return_value = {"available": False, "state": "unknown"}
            result = handler.handle(
                "/api/v1/agents/agent-1/reliability", {}, http
            )
        body = _body(result)
        assert body["circuit_breaker"]["state"] == "unknown"

    def test_reliability_with_airlock_metrics(self, handler_with_airlock, http):
        """When airlock registry has metrics, they are returned."""
        with patch(
            "aragora.server.handlers.composite.CompositeHandler._get_airlock_metrics"
        ) as mock_airlock:
            mock_airlock.return_value = {
                "available": True,
                "requests_total": 100,
                "requests_blocked": 5,
                "latency_avg_ms": 120.5,
                "error_rate": 0.05,
            }
            result = handler_with_airlock.handle(
                "/api/v1/agents/agent-1/reliability", {}, http
            )
        body = _body(result)
        assert body["airlock"]["available"] is True
        assert body["airlock"]["requests_total"] == 100

    def test_reliability_no_airlock(self, handler, http):
        """When no airlock registry, airlock shows unavailable."""
        result = handler.handle("/api/v1/agents/agent-1/reliability", {}, http)
        body = _body(result)
        assert body["airlock"]["available"] is False

    def test_reliability_invalid_agent_id(self, handler, http):
        """Agent IDs with invalid characters are rejected."""
        result = handler.handle(
            "/api/v1/agents/agent@evil.com/reliability", {}, http
        )
        assert _status(result) == 400

    def test_reliability_empty_agent_id(self, handler, http):
        """Empty agent ID is rejected."""
        result = handler.handle("/api/v1/agents//reliability", {}, http)
        assert _status(result) == 400

    def test_reliability_json_content_type(self, handler, http):
        result = handler.handle("/api/v1/agents/agent-1/reliability", {}, http)
        assert result.content_type == "application/json"

    def test_reliability_overall_score_range(self, handler, http):
        """Overall score should be between 0 and 1."""
        result = handler.handle("/api/v1/agents/agent-1/reliability", {}, http)
        body = _body(result)
        assert 0.0 <= body["overall_score"] <= 1.0


# ===========================================================================
# GET /api/v1/debates/{id}/compression-analysis
# ===========================================================================


class TestCompressionAnalysis:
    """Test GET /api/v1/debates/{id}/compression-analysis."""

    def test_compression_analysis_basic_response(self, handler, http):
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-1"
        assert "timestamp" in body
        assert "compression" in body
        assert "quality" in body
        assert "recommendations" in body

    def test_compression_analysis_no_rlm(self, handler, http):
        """When no RLM handler, compression is disabled."""
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        body = _body(result)
        assert body["compression"]["enabled"] is False
        assert body["compression"]["rounds_compressed"] == 0

    def test_compression_analysis_with_rlm(self, handler_with_rlm, http):
        """When RLM handler is available, compression data is populated."""
        handler, mock_rlm = handler_with_rlm
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        body = _body(result)
        assert body["compression"]["enabled"] is True
        assert body["compression"]["rounds_compressed"] == 3
        assert body["compression"]["original_tokens"] == 5000
        assert body["compression"]["compressed_tokens"] == 2000
        assert body["compression"]["savings_percent"] == 60.0

    def test_compression_analysis_with_rlm_quality(self, handler_with_rlm, http):
        """Quality metrics from RLM are included."""
        handler, _ = handler_with_rlm
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        body = _body(result)
        assert body["quality"]["information_retained"] == 0.95
        assert body["quality"]["coherence_score"] == 0.88

    def test_compression_analysis_invalid_id(self, handler, http):
        """Invalid debate ID is rejected."""
        result = handler.handle(
            "/api/v1/debates/../../etc/compression-analysis", {}, http
        )
        assert _status(result) == 400

    def test_compression_analysis_json_content_type(self, handler, http):
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        assert result.content_type == "application/json"

    def test_compression_analysis_empty_id(self, handler, http):
        """Empty debate ID is rejected."""
        result = handler.handle(
            "/api/v1/debates//compression-analysis", {}, http
        )
        assert _status(result) == 400


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on composite endpoints."""

    def test_rate_limit_exceeded_full_context(self, handler, http, _reset_rate_limiter):
        """Rate limit exceeded returns 429."""
        _reset_rate_limiter.is_allowed.return_value = False
        result = handler.handle(
            "/api/v1/debates/debate-1/full-context", {}, http
        )
        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in body["error"].lower()

    def test_rate_limit_exceeded_reliability(self, handler, http, _reset_rate_limiter):
        """Rate limit exceeded returns 429 for reliability endpoint."""
        _reset_rate_limiter.is_allowed.return_value = False
        result = handler.handle(
            "/api/v1/agents/agent-1/reliability", {}, http
        )
        assert _status(result) == 429

    def test_rate_limit_exceeded_compression(self, handler, http, _reset_rate_limiter):
        """Rate limit exceeded returns 429 for compression endpoint."""
        _reset_rate_limiter.is_allowed.return_value = False
        result = handler.handle(
            "/api/v1/debates/debate-1/compression-analysis", {}, http
        )
        assert _status(result) == 429


# ===========================================================================
# Circuit breaker protection
# ===========================================================================


class TestCircuitBreakerProtection:
    """Test circuit breaker protection for subsystem access."""

    def test_circuit_breaker_open_returns_fallback(self, handler, http):
        """When circuit breaker is open, fallback value is returned."""
        # Trip the circuit breaker for 'memory'
        cb = _get_circuit_breaker("memory")
        # Exceed failure threshold (default is 3 for composite)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"

        result = handler.handle(
            "/api/v1/debates/debate-1/full-context", {}, http
        )
        body = _body(result)
        # Memory subsystem should return the fallback
        assert body["memory"]["available"] is False
        assert "unavailable" in body["memory"].get("error", "").lower()

    def test_circuit_breaker_records_success(self, handler, http):
        """Successful fetch records success on circuit breaker."""
        result = handler.handle(
            "/api/v1/debates/debate-1/full-context", {}, http
        )
        assert _status(result) == 200
        # The memory CB should exist and be closed
        cb = _get_circuit_breaker("memory")
        assert cb.state == "closed"

    def test_circuit_breaker_fetch_failure_returns_fallback(self, handler, http):
        """When a subsystem fetch raises, fallback is returned."""
        with patch.object(
            handler, "_get_memory_context", side_effect=KeyError("boom")
        ):
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        # Should get a fallback with error info
        assert body["memory"]["available"] is False

    def test_circuit_breaker_runtime_error_returns_fallback(self, handler, http):
        """RuntimeError from a subsystem returns fallback."""
        with patch.object(
            handler, "_get_memory_context", side_effect=RuntimeError("connection lost")
        ):
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        assert body["memory"]["available"] is False

    def test_multiple_subsystem_failures_isolated(self, handler, http):
        """Failures in one subsystem don't affect others."""
        with patch.object(
            handler, "_get_memory_context", side_effect=ValueError("mem error")
        ), patch.object(
            handler, "_get_knowledge_context",
            return_value={"available": True, "facts": [], "concepts": [], "sources": []},
        ):
            result = handler.handle(
                "/api/v1/debates/debate-1/full-context", {}, http
            )
        body = _body(result)
        # Memory failed, knowledge succeeded
        assert body["memory"]["available"] is False
        assert body["knowledge"]["available"] is True


# ===========================================================================
# Module-level helpers
# ===========================================================================


class TestModuleLevelHelpers:
    """Test module-level circuit breaker helper functions."""

    def test_get_circuit_breaker_creates_new(self):
        """_get_circuit_breaker creates a new breaker for unknown subsystems."""
        cb = _get_circuit_breaker("test_subsystem")
        assert cb is not None
        assert isinstance(cb, CompositeCircuitBreaker)
        assert cb.name == "test_subsystem"

    def test_get_circuit_breaker_returns_same_instance(self):
        """Same subsystem returns the same circuit breaker instance."""
        cb1 = _get_circuit_breaker("same_sub")
        cb2 = _get_circuit_breaker("same_sub")
        assert cb1 is cb2

    def test_get_circuit_breaker_different_subsystems(self):
        """Different subsystems get different circuit breakers."""
        cb1 = _get_circuit_breaker("sub_a")
        cb2 = _get_circuit_breaker("sub_b")
        assert cb1 is not cb2

    def test_get_circuit_breaker_status_empty(self):
        """When no breakers exist, status is empty dict."""
        status = get_circuit_breaker_status()
        assert status == {}

    def test_get_circuit_breaker_status_with_breakers(self):
        """Status returns info about all registered breakers."""
        _get_circuit_breaker("alpha")
        _get_circuit_breaker("beta")
        status = get_circuit_breaker_status()
        assert "alpha" in status
        assert "beta" in status
        assert status["alpha"]["state"] == "closed"
        assert status["beta"]["state"] == "closed"

    def test_clear_cached_components(self):
        """_clear_cached_components removes all circuit breakers."""
        _get_circuit_breaker("temp_sub")
        assert get_circuit_breaker_status() != {}
        _clear_cached_components()
        assert get_circuit_breaker_status() == {}

    def test_circuit_breaker_default_thresholds(self):
        """Circuit breakers are created with correct default thresholds."""
        cb = _get_circuit_breaker("defaults_test")
        assert cb.failure_threshold == 3
        assert cb.half_open_max_calls == 2


# ===========================================================================
# Reliability score calculation
# ===========================================================================


class TestReliabilityScore:
    """Test the _calculate_reliability_score method."""

    def test_perfect_score(self, handler):
        """All healthy metrics yield score of 1.0."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 1.0

    def test_open_circuit_breaker_penalty(self, handler):
        """Open circuit breaker penalizes score by 0.3."""
        metrics = {
            "circuit_breaker": {"state": "open"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.3

    def test_half_open_circuit_breaker_penalty(self, handler):
        """Half-open circuit breaker penalizes score by 0.7."""
        metrics = {
            "circuit_breaker": {"state": "half-open"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.7

    def test_high_error_rate_penalty(self, handler):
        """High error rate reduces score proportionally."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.2},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.8

    def test_error_rate_capped_at_half(self, handler):
        """Error rate penalty is capped at 0.5."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.9},
        }
        score = handler._calculate_reliability_score(metrics)
        # 1.0 * (1.0 - 0.5) = 0.5
        assert score == 0.5

    def test_combined_penalties(self, handler):
        """Open breaker + high error rate compound."""
        metrics = {
            "circuit_breaker": {"state": "open"},
            "airlock": {"error_rate": 0.2},
        }
        score = handler._calculate_reliability_score(metrics)
        # 1.0 * 0.3 * (1.0 - 0.2) = 0.24
        assert score == 0.24

    def test_missing_circuit_breaker_info(self, handler):
        """Missing circuit breaker info doesn't crash."""
        metrics = {"airlock": {"error_rate": 0.0}}
        score = handler._calculate_reliability_score(metrics)
        assert score == 1.0

    def test_missing_airlock_info(self, handler):
        """Missing airlock info uses 0.0 error rate."""
        metrics = {"circuit_breaker": {"state": "closed"}}
        score = handler._calculate_reliability_score(metrics)
        assert score == 1.0

    def test_empty_metrics(self, handler):
        """Empty metrics yield perfect score."""
        score = handler._calculate_reliability_score({})
        assert score == 1.0

    def test_score_rounding(self, handler):
        """Score is rounded to 3 decimal places."""
        metrics = {
            "circuit_breaker": {"state": "half-open"},
            "airlock": {"error_rate": 0.15},
        }
        score = handler._calculate_reliability_score(metrics)
        # 1.0 * 0.7 * (1.0 - 0.15) = 0.595
        assert score == 0.595
        assert isinstance(score, float)


# ===========================================================================
# Compression recommendations
# ===========================================================================


class TestCompressionRecommendations:
    """Test _generate_compression_recommendations method."""

    def test_recommendations_when_disabled(self, handler):
        """Recommends enabling compression when disabled."""
        analysis = {
            "compression": {"enabled": False, "ratio": 0.0},
            "quality": {"information_retained": 1.0},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("enable" in r.lower() for r in recs)

    def test_recommendations_low_ratio(self, handler):
        """Recommends increasing compression for low ratio."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.2},
            "quality": {"information_retained": 1.0},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("increasing" in r.lower() or "compression" in r.lower() for r in recs)

    def test_recommendations_low_retention(self, handler):
        """Recommends reducing compression when information retention is low."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.5},
            "quality": {"information_retained": 0.6},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("reduce" in r.lower() for r in recs)

    def test_recommendations_optimal_settings(self, handler):
        """No recommendations when settings are optimal."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.5},
            "quality": {"information_retained": 0.95},
        }
        recs = handler._generate_compression_recommendations(analysis)
        # Should not recommend enabling or reducing
        assert not any("enable" in r.lower() for r in recs)
        assert not any("reduce" in r.lower() for r in recs)

    def test_recommendations_all_issues(self, handler):
        """Multiple recommendations when multiple issues exist."""
        analysis = {
            "compression": {"enabled": False, "ratio": 0.1},
            "quality": {"information_retained": 0.5},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert len(recs) >= 2  # At least enable + reduce


# ===========================================================================
# Data fetching helpers
# ===========================================================================


class TestDataFetchingHelpers:
    """Test individual data fetching helper methods."""

    def test_get_memory_context_no_continuum(self, handler):
        """Returns unavailable when no continuum memory."""
        result = handler._get_memory_context("debate-1")
        assert result["available"] is False
        assert result["outcomes"] == []

    def test_get_knowledge_context_no_mound(self, handler):
        """Returns unavailable when no knowledge mound."""
        result = handler._get_knowledge_context("debate-1")
        assert result["available"] is False
        assert result["facts"] == []

    def test_get_knowledge_context_with_mound(self, handler_with_knowledge):
        """Returns facts when knowledge mound is available."""
        handler, mock_km = handler_with_knowledge
        result = handler._get_knowledge_context("debate-1")
        assert result["available"] is True
        assert result["facts"] == ["fact-1", "fact-2"]
        mock_km.query.assert_called_once_with("debate-1", limit=10)

    def test_get_knowledge_context_error(self, handler_with_knowledge):
        """Returns unavailable when knowledge mound raises."""
        handler, mock_km = handler_with_knowledge
        mock_km.query.side_effect = ValueError("query failed")
        result = handler._get_knowledge_context("debate-1")
        assert result["available"] is False

    def test_get_belief_context_no_dissent(self, handler):
        """Returns unavailable when no dissent retriever."""
        result = handler._get_belief_context("debate-1")
        assert result["available"] is False
        assert result["cruxes"] == []

    def test_get_belief_context_with_dissent(self, handler_with_belief):
        """Returns cruxes when dissent retriever is available."""
        handler, mock_dissent = handler_with_belief
        result = handler._get_belief_context("debate-1")
        assert result["available"] is True
        assert result["cruxes"] == ["crux-1"]
        mock_dissent.get_cruxes.assert_called_once_with("debate-1", limit=5)

    def test_get_belief_context_error(self, handler_with_belief):
        """Returns unavailable when dissent retriever raises."""
        handler, mock_dissent = handler_with_belief
        mock_dissent.get_cruxes.side_effect = AttributeError("no get_cruxes")
        result = handler._get_belief_context("debate-1")
        assert result["available"] is False

    def test_get_airlock_metrics_no_registry(self, handler):
        """Returns unavailable when no airlock registry."""
        result = handler._get_airlock_metrics("agent-1")
        assert result["available"] is False

    def test_get_airlock_metrics_agent_not_found(self, handler_with_airlock):
        """Returns unavailable when agent not in registry."""
        result = handler_with_airlock._get_airlock_metrics("unknown-agent")
        assert result["available"] is False

    def test_get_airlock_metrics_found(self, handler_with_airlock):
        """Returns metrics when agent proxy is found."""
        result = handler_with_airlock._get_airlock_metrics("agent-1")
        assert result["available"] is True
        assert result["requests_total"] == 100
        assert result["requests_blocked"] == 5
        assert result["latency_avg_ms"] == 120.5
        # success_rate is 95.0, so error_rate = 1.0 - (95.0/100.0) = 0.05
        assert abs(result["error_rate"] - 0.05) < 1e-9

    def test_get_circuit_breaker_state_no_v2_module(self, handler):
        """Returns unknown when circuit_breaker_v2 import fails."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "circuit_breaker_v2" in name:
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = handler._get_circuit_breaker_state("agent-1")
        assert result["available"] is False
        assert result["state"] == "unknown"

    def test_calculate_availability_defaults(self, handler):
        """Default availability returns reasonable values."""
        result = handler._calculate_availability("agent-1")
        assert result["available"] is True
        assert result["uptime_percent"] == 99.9
        assert result["last_24h_errors"] == 0
        assert result["mean_response_time_ms"] == 500

    def test_get_rlm_metrics_no_handler(self, handler):
        """Returns None when no RLM handler in context."""
        result = handler._get_rlm_metrics("debate-1")
        assert result is None

    def test_get_rlm_metrics_with_handler(self, handler_with_rlm):
        """Returns compression stats when RLM handler is available."""
        handler, mock_rlm = handler_with_rlm
        result = handler._get_rlm_metrics("debate-1")
        assert result is not None
        assert result["compression"]["rounds_compressed"] == 3

    def test_get_rlm_metrics_error(self, handler_with_rlm):
        """Returns None when RLM handler raises."""
        handler, mock_rlm = handler_with_rlm
        mock_rlm.get_compression_stats.side_effect = ValueError("bad debate")
        result = handler._get_rlm_metrics("debate-1")
        assert result is None


# ===========================================================================
# _extract_id
# ===========================================================================


class TestExtractId:
    """Test ID extraction from path."""

    def test_extract_debate_id_full_context(self, handler):
        result = handler._extract_id(
            "/api/v1/debates/my-debate-123/full-context",
            "/api/v1/debates/",
            "/full-context",
        )
        assert result == "my-debate-123"

    def test_extract_agent_id_reliability(self, handler):
        result = handler._extract_id(
            "/api/v1/agents/agent-v2/reliability",
            "/api/v1/agents/",
            "/reliability",
        )
        assert result == "agent-v2"

    def test_extract_debate_id_compression(self, handler):
        result = handler._extract_id(
            "/api/v1/debates/test_debate/compression-analysis",
            "/api/v1/debates/",
            "/compression-analysis",
        )
        assert result == "test_debate"


# ===========================================================================
# _error_response
# ===========================================================================


class TestErrorResponse:
    """Test the _error_response helper."""

    def test_error_response_400(self, handler):
        result = handler._error_response("Bad request", 400)
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "Bad request"

    def test_error_response_429(self, handler):
        result = handler._error_response("Rate limited", 429)
        assert _status(result) == 429
        body = _body(result)
        assert body["error"] == "Rate limited"

    def test_error_response_500(self, handler):
        result = handler._error_response("Internal error", 500)
        assert _status(result) == 500
        body = _body(result)
        assert body["error"] == "Internal error"

    def test_error_response_json_content_type(self, handler):
        result = handler._error_response("Error", 400)
        assert result.content_type == "application/json"


# ===========================================================================
# Unknown route returns None
# ===========================================================================


class TestUnknownRoutes:
    """Test that unknown routes return None."""

    def test_handle_unknown_path(self, handler, http):
        result = handler.handle("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_debates_without_suffix(self, handler, http):
        result = handler.handle("/api/v1/debates/debate-1/other-thing", {}, http)
        assert result is None

    def test_handle_agents_without_reliability(self, handler, http):
        result = handler.handle("/api/v1/agents/agent-1/status", {}, http)
        assert result is None


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import composite

        assert "CompositeHandler" in composite.__all__
        assert "CompositeCircuitBreaker" in composite.__all__
        assert "get_circuit_breaker_status" in composite.__all__
        assert "_clear_cached_components" in composite.__all__


# ===========================================================================
# _fetch_with_circuit_breaker edge cases
# ===========================================================================


class TestFetchWithCircuitBreaker:
    """Test _fetch_with_circuit_breaker method edge cases."""

    def test_successful_fetch(self, handler):
        """Successful fetch returns result and records success."""
        result = handler._fetch_with_circuit_breaker(
            "test_sub",
            lambda: {"data": "value"},
            {"available": False},
        )
        assert result == {"data": "value"}

    def test_none_fallback_on_failure(self, handler):
        """When fallback is None, None is returned on failure."""
        result = handler._fetch_with_circuit_breaker(
            "test_none",
            lambda: (_ for _ in ()).throw(ValueError("fail")),
            None,
        )
        assert result is None

    def test_dict_fallback_copied(self, handler):
        """Dict fallback is copied, not modified in place."""
        fallback = {"available": False, "error": "original"}
        handler._fetch_with_circuit_breaker(
            "test_copy",
            lambda: (_ for _ in ()).throw(KeyError("fail")),
            fallback,
        )
        # Original fallback should NOT be mutated
        assert fallback["error"] == "original"

    def test_non_dict_fallback_returned(self, handler):
        """Non-dict fallback values are returned as-is."""
        result = handler._fetch_with_circuit_breaker(
            "test_non_dict",
            lambda: (_ for _ in ()).throw(ValueError("fail")),
            "fallback_string",
        )
        assert result == "fallback_string"

    def test_circuit_breaker_open_returns_fallback_directly(self, handler):
        """When circuit breaker is open, fallback is returned immediately."""
        cb = _get_circuit_breaker("test_open_fb")
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"

        call_count = 0

        def should_not_be_called():
            nonlocal call_count
            call_count += 1
            return {"data": "value"}

        result = handler._fetch_with_circuit_breaker(
            "test_open_fb",
            should_not_be_called,
            {"available": False},
        )
        assert result == {"available": False}
        assert call_count == 0  # Function was never called


# ===========================================================================
# ID validation patterns
# ===========================================================================


class TestIDValidation:
    """Test ID validation for all endpoints."""

    def test_valid_alphanumeric_id(self, handler, http):
        result = handler.handle(
            "/api/v1/debates/abc123/full-context", {}, http
        )
        assert _status(result) == 200

    def test_valid_id_with_hyphens(self, handler, http):
        result = handler.handle(
            "/api/v1/debates/my-debate-id/full-context", {}, http
        )
        assert _status(result) == 200

    def test_valid_id_with_underscores(self, handler, http):
        result = handler.handle(
            "/api/v1/debates/my_debate_id/full-context", {}, http
        )
        assert _status(result) == 200

    def test_too_long_id(self, handler, http):
        """IDs longer than 64 chars are rejected."""
        long_id = "a" * 65
        result = handler.handle(
            f"/api/v1/debates/{long_id}/full-context", {}, http
        )
        assert _status(result) == 400

    def test_max_length_id(self, handler, http):
        """IDs of exactly 64 chars are accepted."""
        max_id = "a" * 64
        result = handler.handle(
            f"/api/v1/debates/{max_id}/full-context", {}, http
        )
        assert _status(result) == 200

    def test_spaces_in_id(self, handler, http):
        """IDs with spaces are rejected."""
        result = handler.handle(
            "/api/v1/debates/my debate/full-context", {}, http
        )
        assert _status(result) == 400

    def test_dots_in_id(self, handler, http):
        """IDs with dots are rejected by SAFE_ID_PATTERN."""
        result = handler.handle(
            "/api/v1/debates/my.debate/full-context", {}, http
        )
        assert _status(result) == 400
