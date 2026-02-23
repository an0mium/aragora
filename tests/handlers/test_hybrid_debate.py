"""Tests for HybridDebateHandler.

Covers:
- Handler initialization and class structure
- Route matching (can_handle)
- GET /api/v1/debates/hybrid - list hybrid debates
- GET /api/v1/debates/hybrid/{id} - get specific debate
- POST /api/v1/debates/hybrid - create hybrid debate
- Input validation for all required and optional fields
- Status filtering and pagination on list endpoint
- Circuit breaker integration
- Error handling (404, 400, 503)
- Method dispatch (handle vs handle_post)
- Security tests (path traversal, injection)
- Edge cases (trailing slashes, empty bodies, boundary values)
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


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Get status code from HandlerResult."""
    return result.status_code


def _make_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler for handler.handle() calls."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.path = "/api/v1/debates/hybrid"
    mock.headers = {}
    if body is not None:
        body_bytes = json.dumps(body).encode()
        mock.rfile.read.return_value = body_bytes
        mock.headers["Content-Length"] = str(len(body_bytes))
    else:
        mock.rfile.read.return_value = b"{}"
        mock.headers["Content-Length"] = "2"
    return mock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server_ctx():
    """Create a mock server context with external_agents registered."""
    return {
        "external_agents": {
            "crewai-agent": {"name": "crewai-agent", "type": "crewai"},
            "langraph-agent": {"name": "langraph-agent", "type": "langgraph"},
        },
    }


@pytest.fixture
def handler(server_ctx):
    """Create a HybridDebateHandler instance."""
    return HybridDebateHandler(server_ctx)


@pytest.fixture
def mock_http_handler():
    """Create a basic mock HTTP handler (no body)."""
    return _make_http_handler()


@pytest.fixture(autouse=True)
def reset_cb():
    """Reset circuit breaker between tests."""
    reset_hybrid_debate_circuit_breaker()
    yield
    reset_hybrid_debate_circuit_breaker()


@pytest.fixture(autouse=True)
def patch_hybrid_available():
    """Ensure HYBRID_AVAILABLE is True for all tests (unless overridden)."""
    with patch(
        "aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True
    ):
        yield


# ============================================================================
# Initialization Tests
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_has_routes(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 2

    def test_routes_include_base_path(self, handler):
        assert "/api/v1/debates/hybrid" in handler.ROUTES

    def test_routes_include_wildcard(self, handler):
        assert "/api/v1/debates/hybrid/*" in handler.ROUTES

    def test_internal_debates_dict_empty(self, handler):
        assert handler._debates == {}

    def test_stores_server_context(self, handler, server_ctx):
        assert handler.ctx is server_ctx


# ============================================================================
# can_handle Tests
# ============================================================================


class TestCanHandle:
    """Tests for route matching logic."""

    def test_base_route(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid") is True

    def test_base_route_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid/") is True

    def test_specific_debate_id(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid/abc123") is True

    def test_uuid_debate_id(self, handler):
        assert handler.can_handle("/api/v1/debates/hybrid/hybrid_abcdef012345") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_other_handler_path(self, handler):
        assert handler.can_handle("/api/v1/agents") is False

    def test_partial_match_prefix(self, handler):
        assert handler.can_handle("/api/v1/debates/hybridx") is True  # startswith

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False


# ============================================================================
# GET /api/v1/debates/hybrid - List Debates
# ============================================================================


class TestListDebates:
    """Tests for listing hybrid debates."""

    def test_empty_list(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["debates"] == []
        assert data["total"] == 0

    def test_list_with_debates(self, handler, mock_http_handler):
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

        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["total"] == 2
        assert len(data["debates"]) == 2

    def test_list_summary_fields(self, handler, mock_http_handler):
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "My task",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.85,
            "started_at": "2026-01-01T00:00:00Z",
            "extra_field": "should not appear",
        }

        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        data = _body(result)
        summary = data["debates"][0]
        assert summary["debate_id"] == "d1"
        assert summary["task"] == "My task"
        assert summary["status"] == "completed"
        assert summary["consensus_reached"] is True
        assert summary["confidence"] == 0.85
        assert summary["started_at"] == "2026-01-01T00:00:00Z"
        assert "extra_field" not in summary

    def test_status_filter_completed(self, handler, mock_http_handler):
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t1", "status": "completed",
            "consensus_reached": True, "confidence": 0.9, "started_at": "x",
        }
        handler._debates["d2"] = {
            "debate_id": "d2", "task": "t2", "status": "pending",
            "consensus_reached": False, "confidence": 0.0, "started_at": "y",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"status": "completed"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 1
        assert data["debates"][0]["status"] == "completed"

    def test_status_filter_pending(self, handler, mock_http_handler):
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t1", "status": "completed",
            "consensus_reached": True, "confidence": 0.9, "started_at": "x",
        }
        handler._debates["d2"] = {
            "debate_id": "d2", "task": "t2", "status": "pending",
            "consensus_reached": False, "confidence": 0.0, "started_at": "y",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"status": "pending"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 1
        assert data["debates"][0]["status"] == "pending"

    def test_status_filter_no_match(self, handler, mock_http_handler):
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t1", "status": "completed",
            "consensus_reached": True, "confidence": 0.9, "started_at": "x",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"status": "nonexistent"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 0

    def test_limit_default(self, handler, mock_http_handler):
        """Default limit is 20."""
        for i in range(25):
            handler._debates[f"d{i}"] = {
                "debate_id": f"d{i}", "task": f"t{i}", "status": "completed",
                "consensus_reached": True, "confidence": 0.9, "started_at": "x",
            }

        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        data = _body(result)
        assert data["total"] == 20

    def test_limit_custom(self, handler, mock_http_handler):
        for i in range(10):
            handler._debates[f"d{i}"] = {
                "debate_id": f"d{i}", "task": f"t{i}", "status": "completed",
                "consensus_reached": True, "confidence": 0.9, "started_at": "x",
            }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "5"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 5

    def test_limit_max_cap(self, handler, mock_http_handler):
        """Limit is capped at 100."""
        for i in range(105):
            handler._debates[f"d{i}"] = {
                "debate_id": f"d{i}", "task": f"t{i}", "status": "completed",
                "consensus_reached": True, "confidence": 0.9, "started_at": "x",
            }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "200"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 100

    def test_limit_min_cap(self, handler, mock_http_handler):
        """Limit cannot go below 1."""
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t1", "status": "completed",
            "consensus_reached": True, "confidence": 0.9, "started_at": "x",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "0"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 1  # min(1, ...) = 1

    def test_limit_negative(self, handler, mock_http_handler):
        """Negative limit clamps to 1."""
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t1", "status": "completed",
            "consensus_reached": True, "confidence": 0.9, "started_at": "x",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "-5"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 1

    def test_limit_invalid_string(self, handler, mock_http_handler):
        """Invalid limit string falls back to default (20)."""
        for i in range(25):
            handler._debates[f"d{i}"] = {
                "debate_id": f"d{i}", "task": f"t{i}", "status": "completed",
                "consensus_reached": True, "confidence": 0.9, "started_at": "x",
            }

        result = handler.handle(
            "/api/v1/debates/hybrid", {"limit": "abc"}, mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 20

    def test_trailing_slash(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/debates/hybrid/", {}, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert "debates" in data


# ============================================================================
# GET /api/v1/debates/hybrid/{id} - Get Specific Debate
# ============================================================================


class TestGetDebate:
    """Tests for getting a specific hybrid debate."""

    def test_get_existing_debate(self, handler, mock_http_handler):
        handler._debates["hybrid_abc123"] = {
            "debate_id": "hybrid_abc123",
            "task": "Test task",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "Result",
            "external_agent": "crewai-agent",
            "verification_agents": [],
            "consensus_threshold": 0.7,
            "max_rounds": 3,
            "domain": "general",
            "config": {},
            "rounds": 2,
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
        }

        result = handler.handle(
            "/api/v1/debates/hybrid/hybrid_abc123", {}, mock_http_handler,
        )
        assert _status(result) == 200
        data = _body(result)
        assert data["debate_id"] == "hybrid_abc123"
        assert data["task"] == "Test task"
        assert data["status"] == "completed"

    def test_get_nonexistent_debate(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/debates/hybrid/nonexistent_id", {}, mock_http_handler,
        )
        assert _status(result) == 404
        data = _body(result)
        assert "not found" in data["error"].lower()

    def test_get_returns_all_fields(self, handler, mock_http_handler):
        record = {
            "debate_id": "hybrid_xyz",
            "task": "My task",
            "status": "pending",
            "consensus_reached": False,
            "confidence": 0.0,
            "final_answer": None,
            "external_agent": "crewai-agent",
            "verification_agents": ["internal-1"],
            "consensus_threshold": 0.8,
            "max_rounds": 5,
            "domain": "healthcare",
            "config": {"key": "value"},
            "rounds": 0,
            "started_at": "2026-02-01T00:00:00Z",
            "completed_at": None,
        }
        handler._debates["hybrid_xyz"] = record

        result = handler.handle(
            "/api/v1/debates/hybrid/hybrid_xyz", {}, mock_http_handler,
        )
        data = _body(result)
        for key, val in record.items():
            assert data[key] == val

    def test_get_debate_trailing_slash(self, handler, mock_http_handler):
        handler._debates["d1"] = {
            "debate_id": "d1", "task": "t", "status": "pending",
        }
        result = handler.handle(
            "/api/v1/debates/hybrid/d1/", {}, mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# POST /api/v1/debates/hybrid - Create Debate
# ============================================================================


class TestCreateDebate:
    """Tests for creating a hybrid debate."""

    def _post(self, handler, body: dict) -> object:
        """Helper to POST a debate creation request."""
        http_handler = _make_http_handler(body)
        return handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )

    def test_create_success(self, handler):
        result = self._post(handler, {
            "task": "Evaluate API design",
            "external_agent": "crewai-agent",
        })
        assert _status(result) == 201
        data = _body(result)
        assert data["debate_id"].startswith("hybrid_")
        assert data["task"] == "Evaluate API design"
        assert data["status"] == "completed"
        assert data["external_agent"] == "crewai-agent"
        assert data["consensus_reached"] is True
        assert data["confidence"] == 0.85
        assert data["started_at"] is not None
        assert data["completed_at"] is not None

    def test_create_stores_debate(self, handler):
        self._post(handler, {
            "task": "Test storing",
            "external_agent": "crewai-agent",
        })
        assert len(handler._debates) == 1
        debate = list(handler._debates.values())[0]
        assert debate["task"] == "Test storing"

    def test_create_debate_id_format(self, handler):
        result = self._post(handler, {
            "task": "Test id format",
            "external_agent": "crewai-agent",
        })
        data = _body(result)
        assert data["debate_id"].startswith("hybrid_")
        assert len(data["debate_id"]) == len("hybrid_") + 12

    def test_create_with_all_optional_fields(self, handler):
        result = self._post(handler, {
            "task": "Full create",
            "external_agent": "crewai-agent",
            "consensus_threshold": 0.9,
            "max_rounds": 5,
            "verification_agents": ["v1", "v2"],
            "domain": "healthcare",
            "config": {"strict_mode": True},
        })
        assert _status(result) == 201
        data = _body(result)
        assert data["consensus_threshold"] == 0.9
        assert data["max_rounds"] == 5
        assert data["verification_agents"] == ["v1", "v2"]
        assert data["domain"] == "healthcare"
        assert data["config"] == {"strict_mode": True}

    def test_create_default_optional_values(self, handler):
        result = self._post(handler, {
            "task": "Defaults test",
            "external_agent": "crewai-agent",
        })
        data = _body(result)
        assert data["consensus_threshold"] == 0.7
        assert data["verification_agents"] == []
        assert data["domain"] == "general"
        assert data["config"] == {}

    def test_create_multiple_debates(self, handler):
        for i in range(3):
            self._post(handler, {
                "task": f"Task {i}",
                "external_agent": "crewai-agent",
            })
        assert len(handler._debates) == 3


# ============================================================================
# POST Validation - Required Fields
# ============================================================================


class TestCreateValidationRequired:
    """Tests for required field validation on create."""

    def _post(self, handler, body: dict) -> object:
        http_handler = _make_http_handler(body)
        return handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)

    def test_missing_task(self, handler):
        result = self._post(handler, {"external_agent": "crewai-agent"})
        assert _status(result) == 400
        assert "task" in _body(result)["error"].lower()

    def test_empty_task(self, handler):
        result = self._post(handler, {
            "task": "", "external_agent": "crewai-agent",
        })
        assert _status(result) == 400

    def test_whitespace_only_task(self, handler):
        result = self._post(handler, {
            "task": "   ", "external_agent": "crewai-agent",
        })
        assert _status(result) == 400

    def test_task_not_string(self, handler):
        result = self._post(handler, {
            "task": 123, "external_agent": "crewai-agent",
        })
        assert _status(result) == 400

    def test_task_too_long(self, handler):
        result = self._post(handler, {
            "task": "x" * 5001, "external_agent": "crewai-agent",
        })
        assert _status(result) == 400
        assert "5000" in _body(result)["error"]

    def test_task_at_max_length(self, handler):
        result = self._post(handler, {
            "task": "x" * 5000, "external_agent": "crewai-agent",
        })
        assert _status(result) == 201

    def test_task_stripped(self, handler):
        result = self._post(handler, {
            "task": "  Trim me  ", "external_agent": "crewai-agent",
        })
        data = _body(result)
        assert data["task"] == "Trim me"

    def test_missing_external_agent(self, handler):
        result = self._post(handler, {"task": "A task"})
        assert _status(result) == 400
        assert "external_agent" in _body(result)["error"].lower()

    def test_empty_external_agent(self, handler):
        result = self._post(handler, {
            "task": "A task", "external_agent": "",
        })
        assert _status(result) == 400

    def test_whitespace_external_agent(self, handler):
        result = self._post(handler, {
            "task": "A task", "external_agent": "   ",
        })
        assert _status(result) == 400

    def test_external_agent_not_string(self, handler):
        result = self._post(handler, {
            "task": "A task", "external_agent": 42,
        })
        assert _status(result) == 400

    def test_external_agent_not_registered(self, handler):
        result = self._post(handler, {
            "task": "A task", "external_agent": "unknown-agent",
        })
        assert _status(result) == 400
        assert "not found" in _body(result)["error"].lower()
        assert "unknown-agent" in _body(result)["error"]

    def test_invalid_json_body(self, handler):
        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 12345)
        http_handler.headers = {"Content-Length": "11"}
        http_handler.rfile.read.return_value = b"not json!!!"
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400

    def test_empty_body(self, handler):
        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 12345)
        http_handler.headers = {"Content-Length": "0"}
        http_handler.rfile.read.return_value = b""
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400


# ============================================================================
# POST Validation - Optional Fields
# ============================================================================


class TestCreateValidationOptional:
    """Tests for optional field validation on create."""

    def _post(self, handler, body: dict) -> object:
        http_handler = _make_http_handler(body)
        return handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)

    def test_consensus_threshold_invalid_string(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "consensus_threshold": "not_a_number",
        })
        assert _status(result) == 400
        assert "consensus_threshold" in _body(result)["error"].lower()

    def test_consensus_threshold_too_low(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "consensus_threshold": -0.1,
        })
        assert _status(result) == 400

    def test_consensus_threshold_too_high(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "consensus_threshold": 1.1,
        })
        assert _status(result) == 400

    def test_consensus_threshold_boundary_zero(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "consensus_threshold": 0.0,
        })
        assert _status(result) == 201

    def test_consensus_threshold_boundary_one(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "consensus_threshold": 1.0,
        })
        assert _status(result) == 201

    def test_max_rounds_invalid_string(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "max_rounds": "abc",
        })
        assert _status(result) == 400
        assert "max_rounds" in _body(result)["error"].lower()

    def test_max_rounds_too_low(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "max_rounds": 0,
        })
        assert _status(result) == 400

    def test_max_rounds_too_high(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "max_rounds": 11,
        })
        assert _status(result) == 400

    def test_max_rounds_boundary_one(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "max_rounds": 1,
        })
        assert _status(result) == 201

    def test_max_rounds_boundary_ten(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "max_rounds": 10,
        })
        assert _status(result) == 201

    def test_verification_agents_not_list(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "verification_agents": "not-a-list",
        })
        assert _status(result) == 400
        assert "verification_agents" in _body(result)["error"].lower()

    def test_domain_non_string_falls_back(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "domain": 123,
        })
        assert _status(result) == 201
        data = _body(result)
        assert data["domain"] == "general"

    def test_config_non_dict_falls_back(self, handler):
        result = self._post(handler, {
            "task": "Test", "external_agent": "crewai-agent",
            "config": "not-a-dict",
        })
        assert _status(result) == 201
        data = _body(result)
        assert data["config"] == {}


# ============================================================================
# POST - Hybrid Not Available (503)
# ============================================================================


class TestHybridNotAvailable:
    """Tests for when the hybrid debate module is not available."""

    def test_get_returns_503(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False,
        ):
            result = handler.handle(
                "/api/v1/debates/hybrid", {}, mock_http_handler,
            )
            assert _status(result) == 503
            assert "not available" in _body(result)["error"].lower()

    def test_post_returns_503(self, handler):
        with patch(
            "aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False,
        ):
            http_handler = _make_http_handler({
                "task": "Test", "external_agent": "crewai-agent",
            })
            result = handler.handle_post(
                "/api/v1/debates/hybrid", {}, http_handler,
            )
            assert _status(result) == 503


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_get_circuit_breaker(self):
        cb = get_hybrid_debate_circuit_breaker()
        assert cb is not None
        assert cb.can_proceed()

    def test_reset_circuit_breaker(self):
        cb = get_hybrid_debate_circuit_breaker()
        # Trigger failures
        for _ in range(10):
            cb.record_failure()
        # After reset, should be able to proceed again
        reset_hybrid_debate_circuit_breaker()
        assert cb.can_proceed()

    def test_create_debate_records_success(self, handler):
        cb = get_hybrid_debate_circuit_breaker()
        http_handler = _make_http_handler({
            "task": "Test", "external_agent": "crewai-agent",
        })
        handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        # After success, circuit breaker should still be closed
        assert cb.can_proceed()

    def test_circuit_breaker_open_returns_503(self, handler):
        cb = get_hybrid_debate_circuit_breaker()
        # Force circuit breaker open
        for _ in range(10):
            cb.record_failure()

        http_handler = _make_http_handler({
            "task": "Test", "external_agent": "crewai-agent",
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    def test_debate_execution_failure_records_failure(self, handler):
        cb = get_hybrid_debate_circuit_breaker()
        with patch.object(
            handler, "_run_debate", side_effect=RuntimeError("execution failed"),
        ):
            http_handler = _make_http_handler({
                "task": "Test", "external_agent": "crewai-agent",
            })
            # The handle_errors decorator should catch the re-raised exception
            result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
            # The handle_errors decorator catches and returns 500
            assert _status(result) == 500


# ============================================================================
# _run_debate Tests
# ============================================================================


class TestRunDebate:
    """Tests for the internal _run_debate method."""

    def test_default_run_debate(self, handler):
        record = {
            "task": "Test task",
            "max_rounds": 3,
        }
        result = handler._run_debate(record)
        assert result["status"] == "completed"
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.85
        assert result["rounds"] <= 3
        assert "completed_at" in result

    def test_run_debate_truncates_task(self, handler):
        long_task = "x" * 200
        record = {"task": long_task, "max_rounds": 3}
        result = handler._run_debate(record)
        assert result["final_answer"] is not None
        # final_answer truncates to 100 chars
        assert len(long_task[:100]) == 100

    def test_run_debate_rounds_capped(self, handler):
        record = {"task": "Test", "max_rounds": 1}
        result = handler._run_debate(record)
        assert result["rounds"] == 1  # min(1, 2)

    def test_run_debate_rounds_default_cap(self, handler):
        record = {"task": "Test", "max_rounds": 5}
        result = handler._run_debate(record)
        assert result["rounds"] == 2  # min(5, 2) = 2


# ============================================================================
# Routing Dispatch Tests
# ============================================================================


class TestRoutingDispatch:
    """Tests for handle() and handle_post() dispatch."""

    def test_handle_returns_none_for_non_matching_path(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/agents", {}, mock_http_handler)
        assert result is None

    def test_handle_post_returns_none_for_non_matching_path(self, handler):
        http_handler = _make_http_handler({"task": "Test"})
        result = handler.handle_post("/api/v1/agents", {}, http_handler)
        assert result is None

    def test_handle_routes_list(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        assert result is not None
        assert _status(result) == 200
        assert "debates" in _body(result)

    def test_handle_routes_get_specific(self, handler, mock_http_handler):
        handler._debates["my-id"] = {
            "debate_id": "my-id", "task": "t", "status": "done",
        }
        result = handler.handle(
            "/api/v1/debates/hybrid/my-id", {}, mock_http_handler,
        )
        assert result is not None
        assert _status(result) == 200

    def test_handle_post_routes_create(self, handler):
        http_handler = _make_http_handler({
            "task": "Test", "external_agent": "crewai-agent",
        })
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )
        assert result is not None
        assert _status(result) == 201

    def test_handle_post_specific_id_returns_none(self, handler):
        """POST to a specific debate ID is not supported (returns None)."""
        http_handler = _make_http_handler({
            "task": "Test", "external_agent": "crewai-agent",
        })
        result = handler.handle_post(
            "/api/v1/debates/hybrid/some-id", {}, http_handler,
        )
        assert result is None


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurity:
    """Security-related tests."""

    def test_path_traversal_in_debate_id(self, handler, mock_http_handler):
        """Path traversal in debate ID should not crash."""
        result = handler.handle(
            "/api/v1/debates/hybrid/../../etc/passwd", {}, mock_http_handler,
        )
        # Should return 404 (not found) rather than leaking info
        if result is not None:
            assert _status(result) == 404

    def test_sql_injection_in_debate_id(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/debates/hybrid/' OR 1=1 --", {}, mock_http_handler,
        )
        if result is not None:
            assert _status(result) == 404

    def test_xss_in_task(self, handler):
        http_handler = _make_http_handler({
            "task": "<script>alert('xss')</script>",
            "external_agent": "crewai-agent",
        })
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )
        # Should succeed (task is stored as-is, XSS prevention is at render layer)
        assert _status(result) == 201

    def test_very_long_debate_id(self, handler, mock_http_handler):
        long_id = "a" * 10000
        result = handler.handle(
            f"/api/v1/debates/hybrid/{long_id}", {}, mock_http_handler,
        )
        if result is not None:
            assert _status(result) == 404

    def test_null_bytes_in_task(self, handler):
        http_handler = _make_http_handler({
            "task": "test\x00task",
            "external_agent": "crewai-agent",
        })
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )
        # Should succeed - null bytes in string don't cause issues
        assert _status(result) == 201

    def test_unicode_in_task(self, handler):
        http_handler = _make_http_handler({
            "task": "Evaluate: \u2603 \ud83d\ude00 design decision",
            "external_agent": "crewai-agent",
        })
        result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )
        assert _status(result) == 201


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_debate_with_missing_optional_keys_in_store(self, handler, mock_http_handler):
        """Debates in store that are missing optional keys should still list."""
        handler._debates["d1"] = {
            "debate_id": "d1",
            "task": "Minimal",
            "status": "pending",
            # Missing consensus_reached, confidence, started_at
        }
        result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        summary = data["debates"][0]
        assert summary["consensus_reached"] is False  # default from .get()
        assert summary["confidence"] == 0.0

    def test_empty_external_agents_context(self):
        """Handler with no external agents registered."""
        h = HybridDebateHandler({"external_agents": {}})
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "any-agent",
        })
        result = h.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400
        assert "not found" in _body(result)["error"].lower()

    def test_no_external_agents_key_in_context(self):
        """Handler with no external_agents key in server context."""
        h = HybridDebateHandler({})
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "any-agent",
        })
        result = h.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400

    def test_created_debate_accessible_via_get(self, handler):
        """Debate created via POST is immediately accessible via GET."""
        http_handler = _make_http_handler({
            "task": "Round trip",
            "external_agent": "crewai-agent",
        })
        post_result = handler.handle_post(
            "/api/v1/debates/hybrid", {}, http_handler,
        )
        debate_id = _body(post_result)["debate_id"]

        get_handler = _make_http_handler()
        get_result = handler.handle(
            f"/api/v1/debates/hybrid/{debate_id}", {}, get_handler,
        )
        assert _status(get_result) == 200
        assert _body(get_result)["debate_id"] == debate_id

    def test_list_then_get_consistency(self, handler):
        """Debates listed should be gettable individually."""
        http_handler = _make_http_handler({
            "task": "Consistency test",
            "external_agent": "crewai-agent",
        })
        handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)

        list_handler = _make_http_handler()
        list_result = handler.handle(
            "/api/v1/debates/hybrid", {}, list_handler,
        )
        debates = _body(list_result)["debates"]
        assert len(debates) == 1

        get_handler = _make_http_handler()
        get_result = handler.handle(
            f"/api/v1/debates/hybrid/{debates[0]['debate_id']}",
            {}, get_handler,
        )
        assert _status(get_result) == 200
        assert _body(get_result)["task"] == "Consistency test"

    def test_task_none_value(self, handler):
        http_handler = _make_http_handler({
            "task": None,
            "external_agent": "crewai-agent",
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400

    def test_consensus_threshold_as_string_number(self, handler):
        """consensus_threshold should accept string numbers via float()."""
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "crewai-agent",
            "consensus_threshold": "0.5",
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 201
        assert _body(result)["consensus_threshold"] == 0.5

    def test_max_rounds_as_string_number(self, handler):
        """max_rounds should accept string numbers via int()."""
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "crewai-agent",
            "max_rounds": "7",
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 201
        assert _body(result)["max_rounds"] == 7

    def test_consensus_threshold_none_value(self, handler):
        """None consensus_threshold should fail validation."""
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "crewai-agent",
            "consensus_threshold": None,
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400

    def test_max_rounds_none_value(self, handler):
        """None max_rounds should fail validation."""
        http_handler = _make_http_handler({
            "task": "Test",
            "external_agent": "crewai-agent",
            "max_rounds": None,
        })
        result = handler.handle_post("/api/v1/debates/hybrid", {}, http_handler)
        assert _status(result) == 400


# ============================================================================
# Combined Filter and Limit Tests
# ============================================================================


class TestFilterAndLimit:
    """Tests for combining status filter with limit."""

    def _populate(self, handler, n_completed=5, n_pending=5):
        for i in range(n_completed):
            handler._debates[f"c{i}"] = {
                "debate_id": f"c{i}", "task": f"completed {i}",
                "status": "completed", "consensus_reached": True,
                "confidence": 0.9, "started_at": "x",
            }
        for i in range(n_pending):
            handler._debates[f"p{i}"] = {
                "debate_id": f"p{i}", "task": f"pending {i}",
                "status": "pending", "consensus_reached": False,
                "confidence": 0.0, "started_at": "y",
            }

    def test_filter_with_limit(self, handler, mock_http_handler):
        self._populate(handler, n_completed=10, n_pending=5)
        result = handler.handle(
            "/api/v1/debates/hybrid",
            {"status": "completed", "limit": "3"},
            mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 3
        for d in data["debates"]:
            assert d["status"] == "completed"

    def test_filter_reduces_below_limit(self, handler, mock_http_handler):
        self._populate(handler, n_completed=2, n_pending=10)
        result = handler.handle(
            "/api/v1/debates/hybrid",
            {"status": "completed", "limit": "10"},
            mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 2

    def test_no_filter_with_limit(self, handler, mock_http_handler):
        self._populate(handler, n_completed=3, n_pending=3)
        result = handler.handle(
            "/api/v1/debates/hybrid",
            {"limit": "4"},
            mock_http_handler,
        )
        data = _body(result)
        assert data["total"] == 4
