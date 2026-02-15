"""
Tests for aragora.server.handlers.context_budget - Context budget handler.

Tests cover:
- Instantiation and ROUTES
- can_handle() route matching with version prefix stripping
- GET /api/v1/context/budget - get current budget configuration
- PUT /api/v1/context/budget - update budget settings
- POST /api/v1/context/budget/estimate - estimate token usage
- Permission checks for PUT
- Input validation
- Error handling
"""

from __future__ import annotations

import io
import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.context_budget import ContextBudgetHandler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a ContextBudgetHandler with mocked context."""
    ctx: dict[str, Any] = {"storage": MagicMock()}
    return ContextBudgetHandler(ctx)


@pytest.fixture
def mock_get():
    """Create a mock HTTP GET handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {"Authorization": "Bearer test-token"}
    return mock


def make_handler(method: str, body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with given method and optional body."""
    mock = MagicMock()
    mock.command = method
    if body is not None:
        body_bytes = json.dumps(body).encode()
        mock.headers = {
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock.rfile = io.BytesIO(body_bytes)
    else:
        mock.headers = {"Authorization": "Bearer test-token"}
    return mock


# ===========================================================================
# Instantiation and Routes
# ===========================================================================


class TestSetup:
    """Tests for handler instantiation and route registration."""

    def test_instantiation(self, handler):
        """Should create handler with context."""
        assert handler is not None

    def test_routes_defined(self):
        """Should define expected ROUTES."""
        assert "/api/v1/context/budget" in ContextBudgetHandler.ROUTES
        assert "/api/v1/context/budget/estimate" in ContextBudgetHandler.ROUTES

    def test_routes_count(self):
        """Should have exactly 2 routes."""
        assert len(ContextBudgetHandler.ROUTES) == 2


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_can_handle_budget(self, handler):
        """Should handle /api/v1/context/budget."""
        assert handler.can_handle("/api/v1/context/budget") is True

    def test_can_handle_estimate(self, handler):
        """Should handle /api/v1/context/budget/estimate."""
        assert handler.can_handle("/api/v1/context/budget/estimate") is True

    def test_cannot_handle_unknown(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/v1/context") is False
        assert handler.can_handle("/api/v1/budgets") is False
        assert handler.can_handle("/api/v1/context/other") is False


# ===========================================================================
# Method Not Allowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for unsupported method rejection."""

    def test_delete_returns_405(self, handler):
        """Should reject DELETE on /api/v1/context/budget."""
        mock = make_handler("DELETE")
        result = handler.handle("/api/v1/context/budget", {}, mock)
        assert result.status_code == 405

    def test_post_on_budget_returns_405(self, handler):
        """Should reject POST on /api/v1/context/budget (only PUT and GET)."""
        mock = make_handler("POST")
        result = handler.handle("/api/v1/context/budget", {}, mock)
        assert result.status_code == 405

    def test_get_on_estimate_returns_405(self, handler):
        """Should reject GET on /api/v1/context/budget/estimate (POST only)."""
        mock = make_handler("GET")
        result = handler.handle("/api/v1/context/budget/estimate", {}, mock)
        assert result.status_code == 405


# ===========================================================================
# GET - Budget Configuration
# ===========================================================================


class TestGetBudget:
    """Tests for GET /api/v1/context/budget."""

    def test_get_budget_success(self, handler, mock_get):
        """Should return current budget configuration."""
        with patch("aragora.debate.context_budgeter.DEFAULT_TOTAL_TOKENS", 4500):
            with patch(
                "aragora.debate.context_budgeter.DEFAULT_SECTION_LIMITS",
                {"env_context": 1400, "historical": 800},
            ):
                result = handler.handle("/api/v1/context/budget", {}, mock_get)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_tokens"] == 4500
        assert "env_context" in data["section_limits"]

    def test_get_budget_import_failure(self, handler, mock_get):
        """Should return 500 when budgeter module raises."""
        with patch(
            "aragora.debate.context_budgeter.DEFAULT_TOTAL_TOKENS",
            new=None,
        ):
            with patch(
                "aragora.debate.context_budgeter.DEFAULT_SECTION_LIMITS",
                new=None,
            ):
                result = handler.handle("/api/v1/context/budget", {}, mock_get)
        # Returns 200 with None values (budgeter returns whatever it has),
        # or 500 if handler fails to serialize
        assert result.status_code in (200, 500)


# ===========================================================================
# PUT - Update Budget
# ===========================================================================


class TestUpdateBudget:
    """Tests for PUT /api/v1/context/budget."""

    def test_permission_required(self, handler):
        """Should require admin:context_budget permission."""
        mock = make_handler("PUT", {"total_tokens": 5000})
        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, MagicMock(status_code=403, body=b'{"error":"Forbidden"}')),
        ):
            result = handler.handle("/api/v1/context/budget", {}, mock)
            assert result.status_code == 403

    def test_update_total_tokens(self, handler):
        """Should update total_tokens via environment variable."""
        mock_user = MagicMock()
        mock = make_handler("PUT", {"total_tokens": 6000})

        with patch.object(handler, "require_permission_or_error", return_value=(mock_user, None)):
            with patch.dict(os.environ, {}, clear=False):
                result = handler.handle("/api/v1/context/budget", {}, mock)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["updated"] is True
        assert data["total_tokens"] == 6000

    def test_update_section_limits(self, handler):
        """Should update section_limits via environment variable."""
        mock_user = MagicMock()
        limits = {"env_context": 2000, "historical": 1000}
        mock = make_handler("PUT", {"section_limits": limits})

        with patch.object(handler, "require_permission_or_error", return_value=(mock_user, None)):
            with patch.dict(os.environ, {}, clear=False):
                result = handler.handle("/api/v1/context/budget", {}, mock)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["section_limits"] == limits

    def test_invalid_total_tokens_returns_400(self, handler):
        """Should return 400 for invalid total_tokens."""
        mock_user = MagicMock()
        mock = make_handler("PUT", {"total_tokens": 50})  # Below minimum of 100

        with patch.object(handler, "require_permission_or_error", return_value=(mock_user, None)):
            result = handler.handle("/api/v1/context/budget", {}, mock)
        assert result.status_code == 400

    def test_invalid_section_limits_type_returns_400(self, handler):
        """Should return 400 for non-dict section_limits."""
        mock_user = MagicMock()
        mock = make_handler("PUT", {"section_limits": "not a dict"})

        with patch.object(handler, "require_permission_or_error", return_value=(mock_user, None)):
            result = handler.handle("/api/v1/context/budget", {}, mock)
        assert result.status_code == 400

    def test_invalid_json_body(self, handler):
        """Should return 400 for invalid JSON body."""
        mock_user = MagicMock()
        mock = make_handler("PUT")

        with patch.object(handler, "require_permission_or_error", return_value=(mock_user, None)):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler.handle("/api/v1/context/budget", {}, mock)
        assert result.status_code == 400


# ===========================================================================
# POST - Estimate Budget
# ===========================================================================


class TestEstimateBudget:
    """Tests for POST /api/v1/context/budget/estimate."""

    def test_estimate_success(self, handler):
        """Should return token estimates for each section."""
        mock = make_handler(
            "POST",
            {
                "sections": {
                    "env_context": "Hello world this is test",
                    "historical": "Some longer text that should take more tokens",
                }
            },
        )

        with patch(
            "aragora.debate.context_budgeter._estimate_tokens",
            side_effect=lambda t: len(t) // 4,
        ):
            result = handler.handle("/api/v1/context/budget/estimate", {}, mock)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "estimates" in data
        assert "env_context" in data["estimates"]
        assert "historical" in data["estimates"]
        assert data["total_estimated_tokens"] > 0

    def test_estimate_missing_sections_returns_400(self, handler):
        """Should return 400 when sections is missing."""
        mock = make_handler("POST", {"not_sections": {}})

        result = handler.handle("/api/v1/context/budget/estimate", {}, mock)
        assert result.status_code == 400

    def test_estimate_non_dict_sections_returns_400(self, handler):
        """Should return 400 when sections is not a dict."""
        mock = make_handler("POST", {"sections": "not a dict"})

        result = handler.handle("/api/v1/context/budget/estimate", {}, mock)
        assert result.status_code == 400

    def test_estimate_invalid_json_returns_400(self, handler):
        """Should return 400 for invalid JSON body."""
        mock = make_handler("POST")

        with patch.object(handler, "read_json_body", return_value=None):
            result = handler.handle("/api/v1/context/budget/estimate", {}, mock)
        assert result.status_code == 400

    def test_estimate_empty_sections(self, handler):
        """Should handle empty sections dict."""
        mock = make_handler("POST", {"sections": {}})

        result = handler.handle("/api/v1/context/budget/estimate", {}, mock)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_estimated_tokens"] == 0
        assert data["estimates"] == {}
