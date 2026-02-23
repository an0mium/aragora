"""Tests for the HybridDebateHandler REST endpoints.

Covers all 3 endpoints:
- POST /api/v1/debates/hybrid       - Start a hybrid debate
- GET  /api/v1/debates/hybrid       - List hybrid debates
- GET  /api/v1/debates/hybrid/{id}  - Get a specific hybrid debate result

Also covers:
- can_handle() path matching
- ROUTES class attribute
- Circuit breaker integration
- Input validation (task, external_agent, consensus_threshold, max_rounds, etc.)
- HYBRID_AVAILABLE guard
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.hybrid_debate_handler import (
    HybridDebateHandler,
    get_hybrid_debate_circuit_breaker,
    reset_hybrid_debate_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _make_mock_handler(body_dict: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    handler = MagicMock()
    if body_dict is not None:
        body_bytes = json.dumps(body_dict).encode()
        handler.headers = {"Content-Length": str(len(body_bytes))}
        handler.rfile.read.return_value = body_bytes
    else:
        handler.headers = {"Content-Length": "0"}
        handler.rfile.read.return_value = b""
    return handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker state between tests."""
    reset_hybrid_debate_circuit_breaker()
    yield
    reset_hybrid_debate_circuit_breaker()


@pytest.fixture
def handler():
    """Create a HybridDebateHandler with an external_agents entry in context."""
    ctx = {
        "external_agents": {
            "crewai-agent": {"type": "crewai", "url": "http://localhost:9000"},
            "langgraph-agent": {"type": "langgraph", "url": "http://localhost:9001"},
        }
    }
    return HybridDebateHandler(server_context=ctx)


@pytest.fixture
def handler_no_agents():
    """Create a HybridDebateHandler with no external agents registered."""
    return HybridDebateHandler(server_context={"external_agents": {}})


# ---------------------------------------------------------------------------
# ROUTES and can_handle
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_routes_defined(self):
        assert HybridDebateHandler.ROUTES == [
            "/api/v1/debates/hybrid",
            "/api/v1/debates/hybrid/*",
        ]

    def test_routes_length(self):
        assert len(HybridDebateHandler.ROUTES) == 2


class TestCanHandle:
    def test_exact_path(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid")

    def test_with_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid/")

    def test_with_id(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid/abc123")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_match_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates/standard")

    def test_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/gateway/agents")


# ---------------------------------------------------------------------------
# GET /api/v1/debates/hybrid - List debates
# ---------------------------------------------------------------------------


class TestListDebates:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_empty_list(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_handler)
        body = _body(result)
        assert body["debates"] == []
        assert body["total"] == 0

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_returns_debates(self, handler):
        # Seed internal debates
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "Test task 1",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.9,
            "started_at": "2026-01-01T00:00:00Z",
        }
        handler._debates["d2"] = {
            "debate_id": "d2",
            "task": "Test task 2",
            "status": "pending",
            "consensus_reached": False,
            "confidence": 0.0,
            "started_at": "2026-01-02T00:00:00Z",
        }
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_handler)
        body = _body(result)
        assert body["total"] == 2
        assert len(body["debates"]) == 2
        # Verify summary fields
        debate_ids = {d["debate_id"] for d in body["debates"]}
        assert "d1" in debate_ids
        assert "d2" in debate_ids

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_status_filter(self, handler):
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "Task 1",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.9,
            "started_at": "2026-01-01T00:00:00Z",
        }
        handler._debates["d2"] = {
            "debate_id": "d2",
            "task": "Task 2",
            "status": "pending",
            "consensus_reached": False,
            "confidence": 0.0,
            "started_at": "2026-01-02T00:00:00Z",
        }
        mock_handler = _make_mock_handler()
        result = handler.handle(
            "/api/v1/debates/hybrid", {"status": "completed"}, mock_handler
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["debates"][0]["debate_id"] == "d1"

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_limit_param(self, handler):
        for i in range(5):
            handler._debates[f"d{i}"] = {
                "debate_id": f"d{i}",
                "task": f"Task {i}",
                "status": "completed",
                "consensus_reached": True,
                "confidence": 0.8,
                "started_at": "2026-01-01T00:00:00Z",
            }
        mock_handler = _make_mock_handler()
        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "2"}, mock_handler
        )
        body = _body(result)
        assert body["total"] == 2

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_limit_clamped_to_max_100(self, handler):
        mock_handler = _make_mock_handler()
        # Should not crash with limit=999; clamped to 100
        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "999"}, mock_handler
        )
        body = _body(result)
        assert "debates" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_limit_invalid_string(self, handler):
        mock_handler = _make_mock_handler()
        # Invalid limit should fall back to default 20
        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "not_a_number"}, mock_handler
        )
        body = _body(result)
        assert "debates" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_summary_fields(self, handler):
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "Summarize report",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.92,
            "started_at": "2026-01-01T00:00:00Z",
            "extra_field": "should not appear in summary",
        }
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_handler)
        body = _body(result)
        summary = body["debates"][0]
        assert set(summary.keys()) == {
            "debate_id",
            "task",
            "status",
            "consensus_reached",
            "confidence",
            "started_at",
        }


# ---------------------------------------------------------------------------
# GET /api/v1/debates/hybrid/{id} - Get specific debate
# ---------------------------------------------------------------------------


class TestGetDebate:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_not_found(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle(
            "/api/v1/debates/hybrid/nonexistent", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert result.status_code == 404

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_found(self, handler):
        handler._debates["hybrid_abc123"] = {
            "debate_id": "hybrid_abc123",
            "task": "Evaluate architecture",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.85,
        }
        mock_handler = _make_mock_handler()
        result = handler.handle(
            "/api/v1/debates/hybrid/hybrid_abc123", {}, mock_handler
        )
        body = _body(result)
        assert body["debate_id"] == "hybrid_abc123"
        assert body["task"] == "Evaluate architecture"
        assert result.status_code == 200

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_trailing_slash_on_id(self, handler):
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "Test",
            "status": "pending",
        }
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/hybrid/d1/", {}, mock_handler)
        body = _body(result)
        assert body["debate_id"] == "d1"


# ---------------------------------------------------------------------------
# GET - HYBRID_AVAILABLE guard
# ---------------------------------------------------------------------------


class TestGetHybridUnavailable:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False)
    def test_list_returns_503(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_handler)
        body = _body(result)
        assert "error" in body
        assert result.status_code == 503

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False)
    def test_get_by_id_returns_503(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle(
            "/api/v1/debates/hybrid/some_id", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# GET - returns None for unmatched paths
# ---------------------------------------------------------------------------


class TestGetReturnsNone:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_unrelated_path(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/debates/hybrid - Create debate
# ---------------------------------------------------------------------------


class TestCreateDebate:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_successful_creation(self, handler):
        body_data = {
            "task": "Evaluate the new API design",
            "external_agent": "crewai-agent",
            "verification_agents": ["claude", "gpt-4"],
            "consensus_threshold": 0.8,
            "max_rounds": 5,
            "domain": "engineering",
            "config": {"verbose": True},
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert result.status_code == 201
        assert body["debate_id"].startswith("hybrid_")
        assert body["task"] == "Evaluate the new API design"
        assert body["external_agent"] == "crewai-agent"
        assert body["status"] == "completed"
        assert body["consensus_reached"] is True
        assert body["confidence"] == 0.85
        assert body["verification_agents"] == ["claude", "gpt-4"]
        assert body["consensus_threshold"] == 0.8
        assert body["max_rounds"] == 5
        assert body["domain"] == "engineering"
        assert body["config"] == {"verbose": True}
        assert body["started_at"] is not None
        assert body["completed_at"] is not None

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_minimal_creation(self, handler):
        body_data = {
            "task": "Simple task",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert result.status_code == 201
        assert body["consensus_threshold"] == 0.7  # default
        assert body["max_rounds"] == 3  # default max_rounds
        assert body["rounds"] == 2  # min(3, 2) from _run_debate
        assert body["domain"] == "general"  # default
        assert body["config"] == {}  # default
        assert body["verification_agents"] == []  # default

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_debate_stored_internally(self, handler):
        body_data = {
            "task": "Test storage",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        debate_id = body["debate_id"]
        # Verify it's stored in the handler's internal dict
        assert debate_id in handler._debates
        assert handler._debates[debate_id]["task"] == "Test storage"


# ---------------------------------------------------------------------------
# POST - Validation errors
# ---------------------------------------------------------------------------


class TestCreateDebateValidation:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_missing_task(self, handler):
        body_data = {"external_agent": "crewai-agent"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "task" in body["error"].lower()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_empty_task(self, handler):
        body_data = {"task": "   ", "external_agent": "crewai-agent"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "task" in body["error"].lower()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_task_not_string(self, handler):
        body_data = {"task": 12345, "external_agent": "crewai-agent"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_task_too_long(self, handler):
        body_data = {"task": "x" * 5001, "external_agent": "crewai-agent"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "5000" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_missing_external_agent(self, handler):
        body_data = {"task": "My task"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "external_agent" in body["error"].lower()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_empty_external_agent(self, handler):
        body_data = {"task": "My task", "external_agent": "  "}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_unregistered_external_agent(self, handler):
        body_data = {"task": "My task", "external_agent": "unknown-agent"}
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "not found" in body["error"].lower()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_invalid_consensus_threshold_string(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "consensus_threshold": "not_a_number",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "consensus_threshold" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_consensus_threshold_too_high(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "consensus_threshold": 1.5,
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "consensus_threshold" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_consensus_threshold_negative(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "consensus_threshold": -0.1,
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_invalid_max_rounds_string(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "max_rounds": "abc",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "max_rounds" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_max_rounds_too_high(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "max_rounds": 11,
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "max_rounds" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_max_rounds_zero(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "max_rounds": 0,
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_verification_agents_not_list(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "verification_agents": "single-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert "verification_agents" in body["error"]

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_invalid_json_body(self, handler):
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "5"}
        mock_handler.rfile.read.return_value = b"notjson"
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_domain_non_string_defaults_to_general(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "domain": 12345,
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert result.status_code == 201
        assert body["domain"] == "general"

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_config_non_dict_defaults_to_empty(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
            "config": "not a dict",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert result.status_code == 201
        assert body["config"] == {}


# ---------------------------------------------------------------------------
# POST - HYBRID_AVAILABLE guard
# ---------------------------------------------------------------------------


class TestPostHybridUnavailable:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False)
    def test_returns_503(self, handler):
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# POST - returns None for unmatched paths
# ---------------------------------------------------------------------------


class TestPostReturnsNone:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_unrelated_path(self, handler):
        mock_handler = _make_mock_handler({"task": "test"})
        result = handler.handle_post("/api/v1/debates", {}, mock_handler)
        assert result is None

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_get_path_on_post(self, handler):
        """POST to a sub-path should return None (only exact match creates)."""
        mock_handler = _make_mock_handler({"task": "test"})
        result = handler.handle_post(
            "/api/v1/debates/hybrid/some-id", {}, mock_handler
        )
        assert result is None


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_circuit_breaker_open_returns_503(self, handler):
        cb = get_hybrid_debate_circuit_breaker()
        # Trip the circuit breaker by recording failures
        for _ in range(10):
            cb.record_failure()

        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        body = _body(result)
        assert "error" in body
        assert result.status_code == 503
        assert "temporarily unavailable" in body["error"].lower()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_successful_creation_records_success(self, handler):
        cb = get_hybrid_debate_circuit_breaker()
        body_data = {
            "task": "My task",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        assert result.status_code == 201
        # Circuit breaker should still be closed
        assert cb.can_proceed()

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_debate_failure_records_circuit_breaker_failure(self, handler):
        cb = get_hybrid_debate_circuit_breaker()

        with patch.object(
            handler, "_run_debate", side_effect=RuntimeError("debate failed")
        ):
            body_data = {
                "task": "My task",
                "external_agent": "crewai-agent",
            }
            mock_handler = _make_mock_handler(body_data)
            result = handler.handle_post(
                "/api/v1/debates/hybrid", {}, mock_handler
            )
            body = _body(result)
            # @handle_errors catches the RuntimeError
            assert "error" in body

    def test_reset_circuit_breaker(self):
        cb = get_hybrid_debate_circuit_breaker()
        for _ in range(10):
            cb.record_failure()
        reset_hybrid_debate_circuit_breaker()
        assert cb.can_proceed()

    def test_get_circuit_breaker_returns_same_instance(self):
        cb1 = get_hybrid_debate_circuit_breaker()
        cb2 = get_hybrid_debate_circuit_breaker()
        assert cb1 is cb2


# ---------------------------------------------------------------------------
# _run_debate
# ---------------------------------------------------------------------------


class TestRunDebate:
    def test_returns_completed_result(self, handler):
        record = {
            "task": "Evaluate architecture",
            "max_rounds": 3,
        }
        result = handler._run_debate(record)
        assert result["status"] == "completed"
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.85
        assert "Evaluate architecture" in result["final_answer"]
        assert result["rounds"] == 2  # min(3, 2)
        assert result["completed_at"] is not None

    def test_rounds_capped_by_max_rounds(self, handler):
        record = {"task": "Test", "max_rounds": 1}
        result = handler._run_debate(record)
        assert result["rounds"] == 1  # min(1, 2) = 1

    def test_rounds_with_high_max_rounds(self, handler):
        record = {"task": "Test", "max_rounds": 10}
        result = handler._run_debate(record)
        assert result["rounds"] == 2  # min(10, 2) = 2

    def test_task_truncated_in_final_answer(self, handler):
        long_task = "x" * 200
        record = {"task": long_task, "max_rounds": 3}
        result = handler._run_debate(record)
        # final_answer contains task[:100]
        assert len(result["final_answer"]) < len(long_task) + 50


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_context(self):
        h = HybridDebateHandler(server_context={})
        assert h.ctx == {}

    def test_custom_context(self):
        h = HybridDebateHandler(server_context={"key": "value"})
        assert h.ctx["key"] == "value"

    def test_debates_dict_initialized(self):
        h = HybridDebateHandler(server_context={})
        assert h._debates == {}

    def test_debates_isolated_between_instances(self):
        h1 = HybridDebateHandler(server_context={})
        h2 = HybridDebateHandler(server_context={})
        h1._debates["d1"] = {"debate_id": "d1"}
        assert "d1" not in h2._debates


# ---------------------------------------------------------------------------
# End-to-end: Create then retrieve
# ---------------------------------------------------------------------------


class TestE2E:
    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_create_then_list(self, handler):
        # Create a debate
        body_data = {
            "task": "Evaluate caching strategy",
            "external_agent": "crewai-agent",
        }
        mock_handler = _make_mock_handler(body_data)
        create_result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        create_body = _body(create_result)
        assert create_result.status_code == 201
        debate_id = create_body["debate_id"]

        # List debates
        list_handler = _make_mock_handler()
        list_result = handler.handle("/api/v1/debates/hybrid", {}, list_handler)
        list_body = _body(list_result)
        assert list_body["total"] == 1
        assert list_body["debates"][0]["debate_id"] == debate_id

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_create_then_get_by_id(self, handler):
        # Create a debate
        body_data = {
            "task": "Review PR strategy",
            "external_agent": "langgraph-agent",
            "domain": "code-review",
        }
        mock_handler = _make_mock_handler(body_data)
        create_result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, mock_handler
        )
        create_body = _body(create_result)
        debate_id = create_body["debate_id"]

        # Get by ID
        get_handler = _make_mock_handler()
        get_result = handler.handle(
            f"/api/v1/debates/hybrid/{debate_id}", {}, get_handler
        )
        get_body = _body(get_result)
        assert get_body["debate_id"] == debate_id
        assert get_body["task"] == "Review PR strategy"
        assert get_body["domain"] == "code-review"
        assert get_body["external_agent"] == "langgraph-agent"

    @patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True)
    def test_create_multiple_then_filter(self, handler):
        # Create two debates
        for task in ["Task A", "Task B"]:
            body_data = {"task": task, "external_agent": "crewai-agent"}
            mock_handler = _make_mock_handler(body_data)
            handler.handle_post("/api/v1/debates/hybrid", {}, mock_handler)

        # Both should be completed (from _run_debate)
        list_handler = _make_mock_handler()
        list_result = handler.handle(
            "/api/v1/debates/hybrid",
            {"status": "completed"},
            list_handler,
        )
        list_body = _body(list_result)
        assert list_body["total"] == 2

        # Filter by a status that doesn't exist
        list_handler2 = _make_mock_handler()
        list_result2 = handler.handle(
            "/api/v1/debates/hybrid",
            {"status": "pending"},
            list_handler2,
        )
        list_body2 = _body(list_result2)
        assert list_body2["total"] == 0
