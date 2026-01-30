"""
Tests for CompositeHandler - Aggregated multi-subsystem data endpoints.

Tests cover:
- can_handle() route matching for all composite routes
- _handle_full_context: happy path, partial failures (memory/knowledge/belief),
  complete failure, error response
- _handle_reliability: happy path, circuit breaker states,
  airlock metrics unavailable, error handling
- _handle_compression_analysis: happy path, RLM disabled, recommendations
- _extract_id: path segment extraction
- _calculate_reliability_score: scoring with circuit breaker states, error rates
- _generate_compression_recommendations: recommendation logic
- Error isolation (subsystem failures don't crash the whole response)
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import sys
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


from aragora.server.handlers.composite import CompositeHandler


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


# ===========================================================================
# Tests: handle (routing)
# ===========================================================================


class TestHandleRouting:
    """Tests for the main handle() dispatching."""

    def test_routes_to_full_context(self, handler):
        """Routes full-context path correctly."""
        result = handler.handle("/api/v1/debates/dbt-123/full-context", {}, MagicMock())
        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-123"
        assert "memory" in body
        assert "knowledge" in body
        assert "belief" in body

    def test_routes_to_reliability(self, handler):
        """Routes reliability path correctly."""
        result = handler.handle("/api/v1/agents/claude/reliability", {}, MagicMock())
        body = get_body(result)
        assert result.status_code == 200
        assert body["agent_id"] == "claude"
        assert "circuit_breaker" in body
        assert "airlock" in body

    def test_routes_to_compression_analysis(self, handler):
        """Routes compression-analysis path correctly."""
        result = handler.handle("/api/v1/debates/dbt-456/compression-analysis", {}, MagicMock())
        body = get_body(result)
        assert result.status_code == 200
        assert body["debate_id"] == "dbt-456"
        assert "compression" in body

    def test_returns_none_for_unmatched(self, handler):
        """Returns None for unmatched paths."""
        result = handler.handle("/api/v1/debates/dbt-123", {}, MagicMock())
        assert result is None


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
        with patch.dict("sys.modules", {"aragora.resilience": None}):
            # The import will raise ImportError inside the method
            result = handler._get_circuit_breaker_state("agent-1")
        assert result["available"] is False

    def test_airlock_import_error(self, handler):
        """Returns unavailable when airlock module not importable."""
        with patch.dict("sys.modules", {"aragora.agents.airlock": None}):
            result = handler._get_airlock_metrics("agent-1")
        assert result["available"] is False

    def test_availability_defaults(self, handler):
        """Returns default availability values."""
        result = handler._calculate_availability("agent-1")
        assert result["available"] is True
        assert result["uptime_percent"] == 99.9

    def test_rlm_metrics_returns_none(self, handler):
        """RLM metrics returns None (not yet integrated)."""
        result = handler._get_rlm_metrics("dbt-123")
        assert result is None


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
