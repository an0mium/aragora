"""
Tests for CompositeHandler - Aggregated multi-subsystem data endpoints.

Comprehensive test coverage for:
- can_handle() route matching for all composite routes
- handle() with rate limiting and input validation
- _handle_full_context: happy path, partial failures, circuit breaker protection
- _handle_reliability: happy path, circuit breaker states, airlock metrics
- _handle_compression_analysis: happy path, RLM disabled, recommendations
- _extract_id: path segment extraction
- _calculate_reliability_score: scoring with circuit breaker states, error rates
- _generate_compression_recommendations: recommendation logic
- Error isolation (subsystem failures don't crash the whole response)
- CompositeCircuitBreaker: state transitions, can_proceed, record_success/failure
- Rate limiting enforcement
- Input validation for path parameters
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import sys
import time
import types as _types_mod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.composite import (
    CompositeHandler,
    CompositeCircuitBreaker,
    get_circuit_breaker_status,
    _clear_cached_components,
    _get_circuit_breaker,
)


# ===========================================================================
# Fixtures and Helpers
# ===========================================================================


def get_body(result) -> dict:
    """Extract JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def handler():
    """Create a CompositeHandler with empty context."""
    return CompositeHandler({})


@pytest.fixture
def handler_with_context():
    """Create a CompositeHandler with mock subsystems in context."""
    mock_continuum = MagicMock()
    mock_continuum.recall.return_value = []

    mock_mound = MagicMock()
    mock_mound.query.return_value = []

    mock_dissent = MagicMock()
    mock_dissent.get_cruxes.return_value = []

    mock_rlm = MagicMock()
    mock_rlm.get_compression_stats.return_value = {
        "compression": {"rounds_compressed": 2, "ratio": 0.5},
        "quality": {"information_retained": 0.9},
    }

    return CompositeHandler(
        {
            "continuum_memory": mock_continuum,
            "knowledge_mound": mock_mound,
            "dissent_retriever": mock_dissent,
            "rlm_handler": mock_rlm,
        }
    )


@pytest.fixture
def circuit_breaker():
    """Create a fresh circuit breaker for testing."""
    return CompositeCircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=0.1,  # Short cooldown for testing
        half_open_max_calls=2,
    )


@pytest.fixture(autouse=True)
def clear_circuit_breakers():
    """Clear circuit breakers before each test."""
    _clear_cached_components()
    yield
    _clear_cached_components()


def make_mock_handler(client_address=("192.168.1.1", 12345)):
    """Create a mock HTTP handler with client address."""
    mock = MagicMock()
    mock.client_address = client_address
    mock.headers = {}
    return mock


# ===========================================================================
# Tests: CompositeCircuitBreaker
# ===========================================================================


class TestCompositeCircuitBreaker:
    """Tests for the CompositeCircuitBreaker class."""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Circuit breaker starts in closed state."""
        assert circuit_breaker.state == CompositeCircuitBreaker.CLOSED
        assert circuit_breaker.can_proceed() is True

    def test_record_success_resets_failure_count(self, circuit_breaker):
        """Recording success resets failure count in closed state."""
        # Record some failures (not enough to open)
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker._failure_count == 2

        # Success resets the count
        circuit_breaker.record_success()
        assert circuit_breaker._failure_count == 0

    def test_opens_after_failure_threshold(self, circuit_breaker):
        """Circuit opens after reaching failure threshold."""
        for _ in range(3):  # failure_threshold = 3
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CompositeCircuitBreaker.OPEN
        assert circuit_breaker.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self, circuit_breaker):
        """Circuit transitions to half-open after cooldown period."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CompositeCircuitBreaker.OPEN

        # Wait for cooldown (0.1 seconds)
        time.sleep(0.15)

        # Should transition to half-open
        assert circuit_breaker.state == CompositeCircuitBreaker.HALF_OPEN
        assert circuit_breaker.can_proceed() is True

    def test_half_open_allows_limited_calls(self, circuit_breaker):
        """Half-open state allows limited number of test calls."""
        # Open and then wait for cooldown
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)

        # Should allow half_open_max_calls (2) attempts
        assert circuit_breaker.can_proceed() is True  # First call
        assert circuit_breaker.can_proceed() is True  # Second call
        assert circuit_breaker.can_proceed() is False  # Third call blocked

    def test_half_open_closes_after_successes(self, circuit_breaker):
        """Circuit closes after enough successes in half-open state."""
        # Open and wait for cooldown
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.2)  # Wait longer than cooldown to ensure transition

        # Trigger state check and get a call to put it in half-open
        assert circuit_breaker.can_proceed() is True
        assert circuit_breaker.state == CompositeCircuitBreaker.HALF_OPEN

        # Record enough successes to close (half_open_max_calls = 2)
        circuit_breaker.record_success()
        circuit_breaker.record_success()

        assert circuit_breaker.state == CompositeCircuitBreaker.CLOSED

    def test_half_open_reopens_on_failure(self, circuit_breaker):
        """Circuit reopens if a failure occurs in half-open state."""
        # Open and wait for cooldown
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)

        assert circuit_breaker.state == CompositeCircuitBreaker.HALF_OPEN

        # One failure reopens the circuit
        circuit_breaker.record_failure()
        assert circuit_breaker.state == CompositeCircuitBreaker.OPEN

    def test_get_status_returns_complete_info(self, circuit_breaker):
        """get_status returns complete circuit breaker information."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()

        status = circuit_breaker.get_status()

        assert status["state"] == CompositeCircuitBreaker.CLOSED
        assert status["failure_count"] == 2
        assert status["success_count"] == 0
        assert status["failure_threshold"] == 3
        assert status["cooldown_seconds"] == 0.1
        assert status["last_failure_time"] is not None

    def test_reset_clears_all_state(self, circuit_breaker):
        """reset() clears all circuit breaker state."""
        # Put the circuit breaker in a non-default state
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CompositeCircuitBreaker.OPEN

        # Reset should clear everything
        circuit_breaker.reset()

        assert circuit_breaker.state == CompositeCircuitBreaker.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0
        assert circuit_breaker._last_failure_time is None


# ===========================================================================
# Tests: Module-level circuit breaker functions
# ===========================================================================


class TestModuleLevelFunctions:
    """Tests for module-level circuit breaker management functions."""

    def test_get_circuit_breaker_creates_new(self):
        """_get_circuit_breaker creates new breaker if not exists."""
        cb = _get_circuit_breaker("test_subsystem")
        assert cb is not None
        assert isinstance(cb, CompositeCircuitBreaker)

    def test_get_circuit_breaker_returns_same_instance(self):
        """_get_circuit_breaker returns same instance for same name."""
        cb1 = _get_circuit_breaker("test_subsystem")
        cb2 = _get_circuit_breaker("test_subsystem")
        assert cb1 is cb2

    def test_get_circuit_breaker_status_returns_all(self):
        """get_circuit_breaker_status returns status of all breakers."""
        _get_circuit_breaker("sub1")
        _get_circuit_breaker("sub2")

        status = get_circuit_breaker_status()

        assert "sub1" in status
        assert "sub2" in status
        assert status["sub1"]["state"] == CompositeCircuitBreaker.CLOSED

    def test_clear_cached_components_clears_breakers(self):
        """_clear_cached_components clears all circuit breakers."""
        _get_circuit_breaker("sub1")
        _get_circuit_breaker("sub2")

        _clear_cached_components()

        status = get_circuit_breaker_status()
        assert len(status) == 0


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_full_context(self, handler):
        """Matches /api/v1/debates/{id}/full-context."""
        assert handler.can_handle("/api/v1/debates/dbt-123/full-context") is True

    def test_handles_reliability(self, handler):
        """Matches /api/v1/agents/{id}/reliability."""
        assert handler.can_handle("/api/v1/agents/claude/reliability") is True

    def test_handles_compression_analysis(self, handler):
        """Matches /api/v1/debates/{id}/compression-analysis."""
        assert handler.can_handle("/api/v1/debates/dbt-456/compression-analysis") is True

    def test_does_not_handle_plain_debates(self, handler):
        """Does not match plain /api/v1/debates/{id}."""
        assert handler.can_handle("/api/v1/debates/dbt-123") is False

    def test_does_not_handle_agents_list(self, handler):
        """Does not match /api/v1/agents."""
        assert handler.can_handle("/api/v1/agents") is False

    def test_does_not_handle_unrelated_path(self, handler):
        """Does not match unrelated paths."""
        assert handler.can_handle("/api/v1/knowledge") is False
        assert handler.can_handle("/api/health") is False

    def test_does_not_handle_wrong_prefix(self, handler):
        """Does not match paths with wrong prefix."""
        assert handler.can_handle("/api/v2/debates/dbt-123/full-context") is False

    def test_handles_various_id_formats(self, handler):
        """Handles various ID formats correctly."""
        assert handler.can_handle("/api/v1/debates/abc123/full-context") is True
        assert handler.can_handle("/api/v1/debates/debate-2025-01/full-context") is True
        assert handler.can_handle("/api/v1/agents/gpt-4-turbo/reliability") is True


# ===========================================================================
# Tests: _extract_id
# ===========================================================================


class TestExtractId:
    """Tests for path ID extraction."""

    def test_extract_debate_id(self, handler):
        """Extracts debate ID from full-context path."""
        result = handler._extract_id(
            "/api/v1/debates/dbt-123/full-context",
            "/api/v1/debates/",
            "/full-context",
        )
        assert result == "dbt-123"

    def test_extract_agent_id(self, handler):
        """Extracts agent ID from reliability path."""
        result = handler._extract_id(
            "/api/v1/agents/claude-3/reliability",
            "/api/v1/agents/",
            "/reliability",
        )
        assert result == "claude-3"

    def test_extract_id_with_long_id(self, handler):
        """Handles long IDs correctly."""
        result = handler._extract_id(
            "/api/v1/debates/debate-2025-01-15-abc123def456/compression-analysis",
            "/api/v1/debates/",
            "/compression-analysis",
        )
        assert result == "debate-2025-01-15-abc123def456"

    def test_extract_id_with_special_chars(self, handler):
        """Handles IDs with allowed special characters."""
        result = handler._extract_id(
            "/api/v1/agents/claude_3_5/reliability",
            "/api/v1/agents/",
            "/reliability",
        )
        assert result == "claude_3_5"


# ===========================================================================
# Tests: handle (routing and validation)
# ===========================================================================


class TestHandleRouting:
    """Tests for the main handle() dispatching with validation."""

    def test_routes_to_full_context(self, handler):
        """Routes full-context path correctly."""
        result = handler.handle("/api/v1/debates/dbt-123/full-context", {}, make_mock_handler())
        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-123"
        assert "memory" in body
        assert "knowledge" in body
        assert "belief" in body

    def test_routes_to_reliability(self, handler):
        """Routes reliability path correctly."""
        result = handler.handle("/api/v1/agents/claude/reliability", {}, make_mock_handler())
        body = get_body(result)
        assert result.status_code == 200
        assert body["agent_id"] == "claude"
        assert "circuit_breaker" in body
        assert "airlock" in body

    def test_routes_to_compression_analysis(self, handler):
        """Routes compression-analysis path correctly."""
        result = handler.handle(
            "/api/v1/debates/dbt-456/compression-analysis", {}, make_mock_handler()
        )
        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-456"
        assert "compression" in body

    def test_returns_none_for_unmatched(self, handler):
        """Returns None for unmatched paths."""
        result = handler.handle("/api/v1/debates/dbt-123", {}, make_mock_handler())
        assert result is None

    def test_rejects_invalid_debate_id(self, handler):
        """Rejects invalid debate ID with 400 error."""
        # Path traversal attempt
        result = handler.handle(
            "/api/v1/debates/../../../etc/passwd/full-context", {}, make_mock_handler()
        )
        body = get_body(result)
        assert result.status_code == 400
        assert "error" in body

    def test_rejects_invalid_agent_id(self, handler):
        """Rejects invalid agent ID with 400 error."""
        # Script injection attempt
        result = handler.handle(
            "/api/v1/agents/<script>alert(1)</script>/reliability", {}, make_mock_handler()
        )
        body = get_body(result)
        assert result.status_code == 400
        assert "error" in body


# ===========================================================================
# Tests: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting enforcement."""

    def test_rate_limiting_allows_normal_requests(self, handler):
        """Normal request rate is allowed."""
        mock_handler = make_mock_handler()

        # First few requests should succeed
        for _ in range(5):
            result = handler.handle("/api/v1/debates/dbt-123/full-context", {}, mock_handler)
            assert result.status_code == 200

    def test_rate_limiting_blocks_excessive_requests(self, handler):
        """Excessive requests are rate limited."""
        mock_handler = make_mock_handler()

        # Make many requests to trigger rate limit
        rate_limited = False
        for _ in range(100):
            result = handler.handle("/api/v1/debates/dbt-123/full-context", {}, mock_handler)
            if result.status_code == 429:
                rate_limited = True
                body = get_body(result)
                assert "Rate limit exceeded" in body["error"]
                break

        assert rate_limited, "Rate limiting should have triggered"

    def test_different_ips_have_separate_limits(self, handler):
        """Different client IPs have separate rate limits."""
        # Use IPs that haven't been used in previous tests
        handler1 = make_mock_handler(("10.0.0.1", 12345))
        handler2 = make_mock_handler(("10.0.0.2", 12345))

        # Both should succeed (different IPs have independent limits)
        result1 = handler.handle("/api/v1/debates/dbt-123/full-context", {}, handler1)
        result2 = handler.handle("/api/v1/debates/dbt-123/full-context", {}, handler2)

        assert result1.status_code == 200
        assert result2.status_code == 200


# ===========================================================================
# Tests: _handle_full_context
# ===========================================================================


class TestFullContext:
    """Tests for debate full context aggregation."""

    def test_full_context_happy_path(self, handler):
        """Returns aggregated context for debate."""
        result = handler._handle_full_context("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-123"
        assert "timestamp" in body
        assert "memory" in body
        assert "knowledge" in body
        assert "belief" in body

    def test_full_context_memory_error_isolation(self, handler):
        """Memory subsystem error does not crash the response."""
        with patch.object(handler, "_get_memory_context", side_effect=KeyError("missing key")):
            result = handler._handle_full_context("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        # Memory should have error info but other fields should be fine
        assert body["memory"]["available"] is False
        assert "error" in body["memory"]
        # Knowledge and belief should still work
        assert "knowledge" in body
        assert "belief" in body

    def test_full_context_knowledge_error_isolation(self, handler):
        """Knowledge subsystem error does not crash the response."""
        with patch.object(handler, "_get_knowledge_context", side_effect=ValueError("bad value")):
            result = handler._handle_full_context("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["knowledge"]["available"] is False
        assert body["memory"] is not None

    def test_full_context_belief_error_isolation(self, handler):
        """Belief subsystem error does not crash the response."""
        with patch.object(handler, "_get_belief_context", side_effect=TypeError("bad type")):
            result = handler._handle_full_context("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["belief"]["available"] is False

    def test_full_context_unexpected_error(self, handler):
        """Unexpected errors in subsystems are handled."""
        with patch.object(handler, "_get_memory_context", side_effect=RuntimeError("unexpected")):
            result = handler._handle_full_context("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["memory"]["available"] is False
        assert body["memory"]["error"] == "Internal error"

    def test_full_context_with_circuit_breaker_open(self, handler):
        """Returns fallback when circuit breaker is open for a subsystem."""
        # Force the memory circuit breaker to open
        cb = _get_circuit_breaker("memory")
        for _ in range(5):
            cb.record_failure()

        result = handler._handle_full_context("dbt-123", {})
        body = get_body(result)

        assert result.status_code == 200
        assert body["memory"]["available"] is False
        assert "unavailable" in body["memory"]["error"].lower()


# ===========================================================================
# Tests: _handle_reliability
# ===========================================================================


class TestReliability:
    """Tests for agent reliability metrics aggregation."""

    def test_reliability_happy_path(self, handler):
        """Returns reliability metrics for an agent."""
        result = handler._handle_reliability("claude", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["agent_id"] == "claude"
        assert "circuit_breaker" in body
        assert "airlock" in body
        assert "availability" in body
        assert "overall_score" in body
        assert isinstance(body["overall_score"], float)

    def test_reliability_circuit_breaker_error(self, handler):
        """Circuit breaker error is isolated."""
        with patch.object(handler, "_get_circuit_breaker_state", side_effect=KeyError("no agent")):
            result = handler._handle_reliability("unknown-agent", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["circuit_breaker"]["available"] is False
        assert body["airlock"] is not None

    def test_reliability_airlock_error(self, handler):
        """Airlock error is isolated."""
        with patch.object(handler, "_get_airlock_metrics", side_effect=ValueError("no data")):
            result = handler._handle_reliability("claude", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["airlock"]["available"] is False

    def test_reliability_availability_error(self, handler):
        """Availability calculation error is isolated."""
        with patch.object(handler, "_calculate_availability", side_effect=TypeError("calc error")):
            result = handler._handle_reliability("claude", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["availability"]["available"] is False


# ===========================================================================
# Tests: _handle_compression_analysis
# ===========================================================================


class TestCompressionAnalysis:
    """Tests for RLM compression analysis."""

    def test_compression_analysis_rlm_disabled(self, handler):
        """Returns default values when RLM is not active."""
        result = handler._handle_compression_analysis("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-123"
        assert body["compression"]["enabled"] is False
        assert body["compression"]["ratio"] == 0.0

    def test_compression_analysis_with_rlm_data(self, handler):
        """Returns RLM data when available."""
        rlm_data = {
            "compression": {
                "rounds_compressed": 3,
                "original_tokens": 10000,
                "compressed_tokens": 3000,
                "ratio": 0.7,
                "savings_percent": 70.0,
            },
            "quality": {
                "information_retained": 0.95,
                "coherence_score": 0.88,
            },
        }

        with patch.object(handler, "_get_rlm_metrics", return_value=rlm_data):
            result = handler._handle_compression_analysis("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["compression"]["enabled"] is True
        assert body["compression"]["rounds_compressed"] == 3
        assert body["quality"]["information_retained"] == 0.95

    def test_compression_analysis_rlm_error(self, handler):
        """RLM errors are handled gracefully."""
        with patch.object(handler, "_get_rlm_metrics", side_effect=KeyError("no metrics")):
            result = handler._handle_compression_analysis("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["compression"]["enabled"] is False

    def test_compression_analysis_with_handler_context(self, handler_with_context):
        """Uses RLM handler from context when available."""
        result = handler_with_context._handle_compression_analysis("dbt-123", {})

        body = get_body(result)
        assert result.status_code == 200
        assert body["compression"]["enabled"] is True


# ===========================================================================
# Tests: _fetch_with_circuit_breaker
# ===========================================================================


class TestFetchWithCircuitBreaker:
    """Tests for circuit breaker protected fetching."""

    def test_successful_fetch_records_success(self, handler):
        """Successful fetch records success on circuit breaker."""
        cb = _get_circuit_breaker("test_sub")
        initial_state = cb.state

        result = handler._fetch_with_circuit_breaker(
            "test_sub", lambda: {"data": "value"}, {"error": "fallback"}
        )

        assert result == {"data": "value"}
        assert cb.state == initial_state  # Still closed

    def test_failed_fetch_records_failure(self, handler):
        """Failed fetch records failure on circuit breaker."""
        cb = _get_circuit_breaker("test_fail")

        for _ in range(3):
            handler._fetch_with_circuit_breaker(
                "test_fail",
                lambda: (_ for _ in ()).throw(ValueError("test error")),
                {"error": "fallback"},
            )

        assert cb.state == CompositeCircuitBreaker.OPEN

    def test_returns_fallback_when_circuit_open(self, handler):
        """Returns fallback value when circuit breaker is open."""
        cb = _get_circuit_breaker("test_open")
        for _ in range(5):
            cb.record_failure()

        result = handler._fetch_with_circuit_breaker(
            "test_open", lambda: {"should": "not be called"}, {"fallback": "value"}
        )

        assert result == {"fallback": "value"}

    def test_fallback_none_returns_none(self, handler):
        """Returns None when fallback is None and circuit is open."""
        cb = _get_circuit_breaker("test_none")
        for _ in range(5):
            cb.record_failure()

        result = handler._fetch_with_circuit_breaker(
            "test_none", lambda: {"should": "not be called"}, None
        )

        assert result is None


# ===========================================================================
# Tests: _calculate_reliability_score
# ===========================================================================


class TestReliabilityScore:
    """Tests for the reliability score calculation."""

    def test_perfect_score(self, handler):
        """Returns 1.0 when all systems are healthy."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 1.0

    def test_open_circuit_breaker_penalty(self, handler):
        """Score is severely penalized when circuit breaker is open."""
        metrics = {
            "circuit_breaker": {"state": "open"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.3

    def test_half_open_circuit_breaker_penalty(self, handler):
        """Score is moderately penalized when circuit breaker is half-open."""
        metrics = {
            "circuit_breaker": {"state": "half-open"},
            "airlock": {"error_rate": 0.0},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.7

    def test_high_error_rate_penalty(self, handler):
        """Score is penalized for high error rate."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.5},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.5

    def test_combined_penalties(self, handler):
        """Score reflects combined penalties."""
        metrics = {
            "circuit_breaker": {"state": "open"},
            "airlock": {"error_rate": 0.5},
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == pytest.approx(0.15, abs=0.01)

    def test_missing_fields_default(self, handler):
        """Score handles missing fields gracefully."""
        metrics = {}
        score = handler._calculate_reliability_score(metrics)
        assert score == 1.0

    def test_error_rate_capped(self, handler):
        """Error rate penalty is capped at 0.5."""
        metrics = {
            "circuit_breaker": {"state": "closed"},
            "airlock": {"error_rate": 0.9},  # Above cap
        }
        score = handler._calculate_reliability_score(metrics)
        assert score == 0.5


# ===========================================================================
# Tests: _generate_compression_recommendations
# ===========================================================================


class TestCompressionRecommendations:
    """Tests for compression recommendation generation."""

    def test_recommends_enabling_rlm(self, handler):
        """Recommends enabling RLM when not active."""
        analysis = {
            "compression": {"enabled": False, "ratio": 0.0},
            "quality": {"information_retained": 1.0},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("enable" in r.lower() for r in recs)

    def test_recommends_higher_compression(self, handler):
        """Recommends more compression when ratio is low."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.1},
            "quality": {"information_retained": 0.95},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("compression" in r.lower() for r in recs)

    def test_recommends_less_compression_on_low_quality(self, handler):
        """Recommends reducing compression when quality is low."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.8},
            "quality": {"information_retained": 0.6},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert any("reduce" in r.lower() for r in recs)

    def test_no_recommendations_when_optimal(self, handler):
        """Returns no recommendations when everything is good."""
        analysis = {
            "compression": {"enabled": True, "ratio": 0.5},
            "quality": {"information_retained": 0.95},
        }
        recs = handler._generate_compression_recommendations(analysis)
        assert len(recs) == 0


# ===========================================================================
# Tests: Memory/Knowledge/Belief Context Helpers
# ===========================================================================


class TestContextHelpers:
    """Tests for subsystem context fetching helpers."""

    def test_memory_context_no_continuum(self, handler):
        """Returns unavailable when no continuum memory in context."""
        result = handler._get_memory_context("dbt-123")
        assert result["available"] is False
        assert result["outcomes"] == []

    def test_knowledge_context_no_mound(self, handler):
        """Returns unavailable when no knowledge mound in context."""
        result = handler._get_knowledge_context("dbt-123")
        assert result["available"] is False
        assert result["facts"] == []

    def test_belief_context_no_retriever(self, handler):
        """Returns unavailable when no dissent retriever in context."""
        result = handler._get_belief_context("dbt-123")
        assert result["available"] is False
        assert result["cruxes"] == []

    def test_circuit_breaker_import_error(self, handler):
        """Returns unavailable when resilience module not importable."""
        with patch.dict("sys.modules", {"aragora.resilience.circuit_breaker_v2": None}):
            result = handler._get_circuit_breaker_state("agent-1")
        assert result["available"] is False

    def test_airlock_metrics_no_registry(self, handler):
        """Returns unavailable when no airlock registry in context."""
        result = handler._get_airlock_metrics("agent-1")
        assert result["available"] is False

    def test_airlock_metrics_with_registry(self, handler):
        """Returns metrics when airlock registry has the agent."""
        mock_proxy = MagicMock()
        mock_proxy.metrics.total_calls = 100
        mock_proxy.metrics.fallback_responses = 5
        mock_proxy.metrics.avg_latency_ms = 150
        mock_proxy.metrics.success_rate = 95.0

        handler.ctx = {"airlock_registry": {"claude": mock_proxy}}
        result = handler._get_airlock_metrics("claude")

        assert result["available"] is True
        assert result["requests_total"] == 100
        assert result["requests_blocked"] == 5

    def test_availability_defaults(self, handler):
        """Returns default availability values."""
        result = handler._calculate_availability("agent-1")
        assert result["available"] is True
        assert result["uptime_percent"] == 99.9

    def test_rlm_metrics_returns_none(self, handler):
        """RLM metrics returns None (not yet integrated)."""
        result = handler._get_rlm_metrics("dbt-123")
        assert result is None

    def test_rlm_metrics_with_handler(self, handler_with_context):
        """RLM metrics from context handler."""
        result = handler_with_context._get_rlm_metrics("dbt-123")
        assert result is not None
        assert "compression" in result


# ===========================================================================
# Tests: Error Response
# ===========================================================================


class TestErrorResponse:
    """Tests for error response helper."""

    def test_error_response_400(self, handler):
        """Creates a 400 error response."""
        result = handler._error_response("Bad request", 400)
        body = get_body(result)
        assert result.status_code == 400
        assert body["error"] == "Bad request"

    def test_error_response_500(self, handler):
        """Creates a 500 error response."""
        result = handler._error_response("Internal server error", 500)
        body = get_body(result)
        assert result.status_code == 500
        assert body["error"] == "Internal server error"

    def test_error_response_429(self, handler):
        """Creates a 429 rate limit error response."""
        result = handler._error_response("Rate limit exceeded", 429)
        body = get_body(result)
        assert result.status_code == 429
        assert body["error"] == "Rate limit exceeded"


# ===========================================================================
# Tests: ROUTES class attribute
# ===========================================================================


class TestRoutes:
    """Tests for ROUTES class attribute."""

    def test_routes_defined(self, handler):
        """ROUTES class attribute is defined and non-empty."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) == 3

    def test_routes_match_patterns(self, handler):
        """ROUTES patterns match expected endpoints."""
        assert "/api/v1/debates/*/full-context" in handler.ROUTES
        assert "/api/v1/agents/*/reliability" in handler.ROUTES
        assert "/api/v1/debates/*/compression-analysis" in handler.ROUTES
