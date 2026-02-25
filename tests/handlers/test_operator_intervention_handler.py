"""Tests for aragora.server.handlers.operator_intervention.

Covers the OperatorInterventionHandler HTTP endpoints:
- POST /api/v1/debates/{id}/pause
- POST /api/v1/debates/{id}/resume
- POST /api/v1/debates/{id}/restart
- POST /api/v1/debates/{id}/inject
- GET  /api/v1/debates/{id}/intervention-status
- GET  /api/v1/interventions/active
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.operator_intervention import (
    DebateInterventionManager,
    InterventionStatus,
    _reset_operator_manager,
)
from aragora.server.handlers.operator_intervention import OperatorInterventionHandler


# =========================================================================
# Helpers
# =========================================================================


def _parse_result(result) -> tuple[dict[str, Any], int]:
    """Parse a HandlerResult into (body_dict, status_code)."""
    if result is None:
        return {}, 0
    body = json.loads(result.body.decode("utf-8")) if result.body else {}
    return body, result.status_code


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(autouse=True)
def _reset_global():
    """Reset the global singleton between tests."""
    yield
    _reset_operator_manager()


@pytest.fixture
def manager():
    """Create and return a fresh DebateInterventionManager."""
    return DebateInterventionManager()


@pytest.fixture
def handler(manager):
    """Create an OperatorInterventionHandler with a patched manager."""
    h = OperatorInterventionHandler(ctx={})
    # Patch _get_manager to return our test manager
    h._get_manager = lambda: manager
    return h


def _make_http_handler(body: dict[str, Any] | None = None, command: str = "POST") -> MagicMock:
    """Create a mock HTTP request handler."""
    mock = MagicMock()
    mock.command = command
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = MagicMock()

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        mock.headers.get = lambda key, default=None: {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
            "X-Forwarded-For": None,
        }.get(key, default)
        mock.rfile = io.BytesIO(body_bytes)
    else:
        mock.headers.get = lambda key, default=None: {
            "Content-Length": "0",
            "X-Forwarded-For": None,
        }.get(key, default)
        mock.rfile = io.BytesIO(b"")

    return mock


# =========================================================================
# can_handle
# =========================================================================


class TestCanHandle:
    def test_can_handle_pause(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/pause") is True

    def test_can_handle_resume(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/resume") is True

    def test_can_handle_restart(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/restart") is True

    def test_can_handle_inject(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/inject") is True

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/intervention-status") is True

    def test_can_handle_active_list(self, handler):
        assert handler.can_handle("/api/v1/interventions/active") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/votes") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False


# =========================================================================
# GET /api/v1/debates/{id}/intervention-status
# =========================================================================


class TestGetStatus:
    def test_get_status_success(self, handler, manager):
        manager.register("d1", total_rounds=5)
        manager.update_round("d1", 2)

        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/debates/d1/intervention-status", {}, http)
        assert result is not None
        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["debate_id"] == "d1"
        assert body["data"]["state"] == "running"
        assert body["data"]["current_round"] == 2
        assert body["data"]["total_rounds"] == 5

    def test_get_status_not_found(self, handler, manager):
        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/debates/nonexistent/intervention-status", {}, http)
        assert result is not None
        body, status = _parse_result(result)
        assert "not found" in body.get("error", "").lower()
        assert status == 404


# =========================================================================
# GET /api/v1/interventions/active
# =========================================================================


class TestListActive:
    def test_list_active_empty(self, handler, manager):
        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/interventions/active", {}, http)
        assert result is not None
        body, status = _parse_result(result)
        assert status == 200
        assert body["count"] == 0
        assert body["data"] == []

    def test_list_active_with_debates(self, handler, manager):
        manager.register("d1", total_rounds=3)
        manager.register("d2", total_rounds=5)
        manager.pause("d2", reason="review")

        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/interventions/active", {}, http)
        assert result is not None
        body, status = _parse_result(result)
        assert status == 200
        assert body["count"] == 2
        ids = {d["debate_id"] for d in body["data"]}
        assert ids == {"d1", "d2"}

    def test_list_active_excludes_completed(self, handler, manager):
        manager.register("d1")
        manager.register("d2")
        manager.mark_completed("d1")

        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/interventions/active", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["count"] == 1
        assert body["data"][0]["debate_id"] == "d2"


# =========================================================================
# POST /api/v1/debates/{id}/pause
# =========================================================================


class TestPause:
    def test_pause_success(self, handler, manager):
        manager.register("d1", total_rounds=5)

        http = _make_http_handler(body={"reason": "Checking results"})
        result = handler.handle_post("/api/v1/debates/d1/pause", {}, http)
        assert result is not None
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["state"] == "paused"
        assert body["reason"] == "Checking results"

    def test_pause_without_reason(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/pause", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["reason"] is None

    def test_pause_not_found(self, handler, manager):
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/missing/pause", {}, http)
        body, status = _parse_result(result)
        assert "not found" in body.get("error", "").lower()
        assert status == 404

    def test_pause_already_paused(self, handler, manager):
        manager.register("d1")
        manager.pause("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/pause", {}, http)
        _, status = _parse_result(result)
        assert status == 409

    def test_pause_completed_debate(self, handler, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/pause", {}, http)
        _, status = _parse_result(result)
        assert status == 409


# =========================================================================
# POST /api/v1/debates/{id}/resume
# =========================================================================


class TestResume:
    def test_resume_success(self, handler, manager):
        manager.register("d1")
        manager.pause("d1")

        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/resume", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["state"] == "running"

    def test_resume_not_found(self, handler, manager):
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/missing/resume", {}, http)
        _, status = _parse_result(result)
        assert status == 404

    def test_resume_not_paused(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/resume", {}, http)
        _, status = _parse_result(result)
        assert status == 409


# =========================================================================
# POST /api/v1/debates/{id}/restart
# =========================================================================


class TestRestart:
    def test_restart_from_beginning(self, handler, manager):
        manager.register("d1", total_rounds=5)

        http = _make_http_handler(body={"from_round": 0})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["from_round"] == 0

    def test_restart_from_specific_round(self, handler, manager):
        manager.register("d1", total_rounds=5)

        http = _make_http_handler(body={"from_round": 3})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["from_round"] == 3

    def test_restart_default_round_zero(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["from_round"] == 0

    def test_restart_not_found(self, handler, manager):
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/missing/restart", {}, http)
        _, status = _parse_result(result)
        assert status == 404

    def test_restart_completed_debate(self, handler, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        _, status = _parse_result(result)
        assert status == 409

    def test_restart_negative_round_rejected(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={"from_round": -1})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        body, status = _parse_result(result)
        assert "non-negative" in body.get("error", "").lower()
        assert status == 400

    def test_restart_invalid_round_type_rejected(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={"from_round": "abc"})
        result = handler.handle_post("/api/v1/debates/d1/restart", {}, http)
        _, status = _parse_result(result)
        assert status == 400


# =========================================================================
# POST /api/v1/debates/{id}/inject
# =========================================================================


class TestInject:
    def test_inject_context_success(self, handler, manager):
        manager.register("d1")

        http = _make_http_handler(body={"context": "Additional background info"})
        result = handler.handle_post("/api/v1/debates/d1/inject", {}, http)
        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["context_length"] > 0

    def test_inject_missing_context(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/inject", {}, http)
        body, status = _parse_result(result)
        assert "context" in body.get("error", "").lower()
        assert status == 400

    def test_inject_empty_context(self, handler, manager):
        manager.register("d1")
        http = _make_http_handler(body={"context": "   "})
        result = handler.handle_post("/api/v1/debates/d1/inject", {}, http)
        _, status = _parse_result(result)
        assert status == 400

    def test_inject_not_found(self, handler, manager):
        http = _make_http_handler(body={"context": "hello"})
        result = handler.handle_post("/api/v1/debates/missing/inject", {}, http)
        _, status = _parse_result(result)
        assert status == 404

    def test_inject_completed_debate(self, handler, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        http = _make_http_handler(body={"context": "hello"})
        result = handler.handle_post("/api/v1/debates/d1/inject", {}, http)
        _, status = _parse_result(result)
        assert status == 409


# =========================================================================
# Module unavailable
# =========================================================================


class TestModuleUnavailable:
    def test_pause_when_module_unavailable(self):
        h = OperatorInterventionHandler(ctx={})
        h._get_manager = lambda: None

        http = _make_http_handler(body={})
        result = h.handle_post("/api/v1/debates/d1/pause", {}, http)
        _, status = _parse_result(result)
        assert status == 503

    def test_status_when_module_unavailable(self):
        h = OperatorInterventionHandler(ctx={})
        h._get_manager = lambda: None

        http = _make_http_handler(command="GET")
        result = h.handle("/api/v1/debates/d1/intervention-status", {}, http)
        _, status = _parse_result(result)
        assert status == 503

    def test_active_when_module_unavailable(self):
        h = OperatorInterventionHandler(ctx={})
        h._get_manager = lambda: None

        http = _make_http_handler(command="GET")
        result = h.handle("/api/v1/interventions/active", {}, http)
        _, status = _parse_result(result)
        assert status == 503


# =========================================================================
# Invalid debate ID
# =========================================================================


class TestInvalidDebateId:
    def test_get_status_invalid_id(self, handler):
        http = _make_http_handler(command="GET")
        result = handler.handle("/api/v1/debates/../etc/intervention-status", {}, http)
        # Should not match the pattern at all (returns None or 400)
        # The regex only allows [a-zA-Z0-9_-]+ so ../ won't match
        assert result is None

    def test_post_pause_invalid_id(self, handler):
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/../etc/pause", {}, http)
        assert result is None


# =========================================================================
# Unhandled routes
# =========================================================================


class TestUnhandledRoutes:
    def test_handle_returns_none_for_non_status_get(self, handler):
        http = _make_http_handler(command="GET")
        # pause is a POST endpoint; GET should return None
        result = handler.handle("/api/v1/debates/d1/pause", {}, http)
        assert result is None

    def test_handle_post_returns_none_for_status(self, handler):
        # intervention-status is a GET endpoint; POST should return None
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/debates/d1/intervention-status", {}, http)
        assert result is None
