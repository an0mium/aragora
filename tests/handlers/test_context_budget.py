"""Tests for context budget handler.

Covers:
- Route matching (can_handle)
- GET /api/v1/context/budget - retrieve budget configuration
- PUT /api/v1/context/budget - update budget settings
- POST /api/v1/context/budget/estimate - estimate token usage
- RBAC permission checks (require_permission_or_error)
- Input validation (missing body, invalid JSON, malformed fields)
- Error handling (import failures, runtime errors)
- Method dispatch and 405 responses
- Edge cases for token estimation
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.context_budget import ContextBudgetHandler


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status_code from HandlerResult."""
    return result.status_code


def _make_handler(command: str = "GET", body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with command and optional JSON body."""
    handler = MagicMock()
    handler.command = command
    if body is not None:
        body_bytes = json.dumps(body).encode()
        handler.rfile.read.return_value = body_bytes
        handler.headers = {"Content-Length": str(len(body_bytes))}
    else:
        handler.rfile.read.return_value = b""
        handler.headers = {"Content-Length": "0"}
    return handler


# ============================================================================
# Initialization
# ============================================================================


class TestInit:
    """Test handler initialization."""

    def test_create_with_empty_context(self):
        h = ContextBudgetHandler({})
        assert h.ctx == {}

    def test_create_with_server_context(self):
        ctx = {"storage": MagicMock()}
        h = ContextBudgetHandler(ctx)
        assert h.ctx is ctx

    def test_routes_attribute(self):
        assert "/api/v1/context/budget" in ContextBudgetHandler.ROUTES
        assert "/api/v1/context/budget/estimate" in ContextBudgetHandler.ROUTES
        assert len(ContextBudgetHandler.ROUTES) == 2


# ============================================================================
# Route Matching (can_handle)
# ============================================================================


class TestCanHandle:
    """Test route matching via can_handle."""

    def test_budget_route(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/context/budget") is True

    def test_estimate_route(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/context/budget/estimate") is True

    def test_unknown_route(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/context/other") is False

    def test_root_api_route(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/") is False

    def test_no_version_prefix(self):
        h = ContextBudgetHandler({})
        # Without the version prefix, strip_version_prefix returns the path unchanged,
        # which is /api/context/budget -- this matches the expected stripped path
        assert h.can_handle("/api/context/budget") is True

    def test_different_version_prefix(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v2/context/budget") is True

    def test_partial_match_no_handle(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/context") is False

    def test_budget_with_trailing_slash(self):
        h = ContextBudgetHandler({})
        # trailing slash creates a different path
        assert h.can_handle("/api/v1/context/budget/") is False

    def test_estimate_with_extra_segment(self):
        h = ContextBudgetHandler({})
        assert h.can_handle("/api/v1/context/budget/estimate/extra") is False


# ============================================================================
# Method Dispatch
# ============================================================================


class TestMethodDispatch:
    """Test HTTP method routing in handle()."""

    def test_get_budget_dispatches(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200

    def test_put_budget_dispatches(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 5000})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200

    def test_post_estimate_dispatches(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"intro": "Hello world"}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200

    def test_delete_budget_returns_405(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("DELETE")
        mock_h.command = "DELETE"
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result)["error"]

    def test_post_budget_returns_405(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"total_tokens": 100})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 405

    def test_get_estimate_returns_405(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 405

    def test_patch_budget_returns_405(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PATCH")
        mock_h.command = "PATCH"
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 405

    def test_put_estimate_returns_405(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"sections": {"a": "b"}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 405


# ============================================================================
# GET /api/v1/context/budget
# ============================================================================


class TestGetBudget:
    """Test budget retrieval endpoint."""

    def test_returns_total_tokens(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        body = _body(result)
        assert "total_tokens" in body
        assert isinstance(body["total_tokens"], int)

    def test_returns_section_limits(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        body = _body(result)
        assert "section_limits" in body
        assert isinstance(body["section_limits"], dict)

    def test_returns_200(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200

    def test_default_total_tokens_value(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        body = _body(result)
        assert body["total_tokens"] >= 100  # sensible minimum

    def test_section_limits_has_known_keys(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        body = _body(result)
        # At least some known sections should exist
        limits = body["section_limits"]
        assert len(limits) > 0

    def test_import_error_returns_500(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        with patch.dict("sys.modules", {"aragora.debate.context_budgeter": None}):
            result = h.handle("/api/v1/context/budget", {}, mock_h)
            assert _status(result) == 500
            assert "Failed to get context budget" in _body(result)["error"]


# ============================================================================
# PUT /api/v1/context/budget
# ============================================================================


class TestUpdateBudget:
    """Test budget update endpoint."""

    def test_update_total_tokens(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 8000})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["updated"] is True
        assert body["total_tokens"] == 8000

    def test_update_section_limits(self):
        h = ContextBudgetHandler({})
        new_limits = {"env_context": 2000, "historical": 1000}
        mock_h = _make_handler("PUT", {"section_limits": new_limits})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["updated"] is True
        assert body["section_limits"] == new_limits

    def test_update_both_fields(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 6000, "section_limits": {"a": 100}})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_tokens"] == 6000
        assert body["section_limits"] == {"a": 100}

    def test_update_neither_field(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["updated"] is True
        assert body["total_tokens"] is None
        assert body["section_limits"] is None

    def test_sets_env_var_total_tokens(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 9999})
        with patch.dict(os.environ, {}, clear=False):
            result = h.handle("/api/v1/context/budget", {}, mock_h)
            assert _status(result) == 200
            assert os.environ.get("ARAGORA_CONTEXT_TOTAL_TOKENS") == "9999"

    def test_sets_env_var_section_limits(self):
        h = ContextBudgetHandler({})
        limits = {"env_context": 1500}
        mock_h = _make_handler("PUT", {"section_limits": limits})
        with patch.dict(os.environ, {}, clear=False):
            result = h.handle("/api/v1/context/budget", {}, mock_h)
            assert _status(result) == 200
            stored = json.loads(os.environ.get("ARAGORA_CONTEXT_SECTION_LIMITS", "{}"))
            assert stored == limits

    def test_invalid_json_body_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = MagicMock()
        mock_h.command = "PUT"
        mock_h.rfile.read.return_value = b"not json"
        mock_h.headers = {"Content-Length": "8"}
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "Invalid JSON body" in _body(result)["error"]

    def test_total_tokens_not_int_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": "abc"})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "total_tokens must be an integer >= 100" in _body(result)["error"]

    def test_total_tokens_float_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 100.5})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "total_tokens must be an integer >= 100" in _body(result)["error"]

    def test_total_tokens_too_small_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 50})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "total_tokens must be an integer >= 100" in _body(result)["error"]

    def test_total_tokens_negative_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": -1})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400

    def test_total_tokens_zero_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 0})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400

    def test_total_tokens_exactly_100_succeeds(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 100})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        assert _body(result)["total_tokens"] == 100

    def test_total_tokens_99_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 99})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400

    def test_section_limits_not_dict_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"section_limits": "not a dict"})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "section_limits must be a dict" in _body(result)["error"]

    def test_section_limits_list_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"section_limits": [1, 2, 3]})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400
        assert "section_limits must be a dict" in _body(result)["error"]

    def test_section_limits_int_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"section_limits": 42})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 400

    def test_total_tokens_bool_returns_400(self):
        """Booleans are ints in Python; True == 1 which is < 100."""
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": True})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        # True is isinstance(True, int) -> True, but True == 1 < 100
        assert _status(result) == 400

    def test_total_tokens_large_value_succeeds(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 1_000_000})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        assert _body(result)["total_tokens"] == 1_000_000

    def test_section_limits_empty_dict_succeeds(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"section_limits": {}})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        assert _body(result)["section_limits"] == {}


# ============================================================================
# POST /api/v1/context/budget/estimate
# ============================================================================


class TestEstimateBudget:
    """Test token estimation endpoint."""

    def test_estimate_single_section(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"intro": "Hello world test"}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "estimates" in body
        assert "total_estimated_tokens" in body
        assert "intro" in body["estimates"]
        assert body["estimates"]["intro"] > 0

    def test_estimate_multiple_sections(self):
        h = ContextBudgetHandler({})
        sections = {
            "context": "Some context text here.",
            "historical": "Historical data with more words.",
        }
        mock_h = _make_handler("POST", {"sections": sections})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["estimates"]) == 2
        assert body["total_estimated_tokens"] == sum(body["estimates"].values())

    def test_estimate_empty_text_section(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"empty": ""}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["estimates"]["empty"] == 0
        assert body["total_estimated_tokens"] == 0

    def test_estimate_with_non_string_value(self):
        """Non-string values get str()-converted by the handler."""
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"numbers": 12345678}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["estimates"]["numbers"] >= 1

    def test_estimate_empty_sections_dict(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["estimates"] == {}
        assert body["total_estimated_tokens"] == 0

    def test_estimate_missing_sections_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400
        assert "sections must be a dict" in _body(result)["error"]

    def test_estimate_sections_not_dict_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": "not a dict"})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400
        assert "sections must be a dict" in _body(result)["error"]

    def test_estimate_sections_list_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": ["a", "b"]})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400

    def test_estimate_sections_null_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": None})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400

    def test_estimate_invalid_json_body_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = MagicMock()
        mock_h.command = "POST"
        mock_h.rfile.read.return_value = b"not json"
        mock_h.headers = {"Content-Length": "8"}
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400
        assert "Invalid JSON body" in _body(result)["error"]

    def test_estimate_import_error_returns_500(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"intro": "text"}})
        with patch.dict("sys.modules", {"aragora.debate.context_budgeter": None}):
            result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
            assert _status(result) == 500
            assert "Failed to estimate context budget" in _body(result)["error"]

    def test_estimate_total_is_sum_of_estimates(self):
        h = ContextBudgetHandler({})
        # 16 chars -> 4 tokens, 32 chars -> 8 tokens (heuristic: len / 4)
        mock_h = _make_handler("POST", {"sections": {
            "a": "a" * 16,
            "b": "b" * 32,
        }})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_estimated_tokens"] == body["estimates"]["a"] + body["estimates"]["b"]

    def test_estimate_long_text(self):
        h = ContextBudgetHandler({})
        long_text = "x" * 10000
        mock_h = _make_handler("POST", {"sections": {"long": long_text}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["estimates"]["long"] == 2500  # 10000 / 4


# ============================================================================
# RBAC / Permission Checks
# ============================================================================


class TestPermissionChecks:
    """Test that permission checks are applied for each endpoint."""

    @pytest.mark.no_auto_auth
    def test_get_budget_requires_permission(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        # With no_auto_auth, require_permission_or_error should fail
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    def test_put_budget_requires_permission(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": 200})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    def test_estimate_budget_requires_permission(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"a": "text"}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) in (401, 403)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_body_on_put(self):
        """PUT with Content-Length 0 should return an empty-dict body, not None."""
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT")
        # Content-Length 0 -> read_json_body returns {}
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["updated"] is True

    def test_extra_fields_in_put_body_ignored(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {
            "total_tokens": 500,
            "extra_field": "should be ignored",
        })
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_tokens"] == 500
        assert "extra_field" not in body

    def test_estimate_sections_int_returns_400(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": 42})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 400

    def test_unicode_text_estimate(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"emoji": "Hello world! \U0001f600" * 100}})
        result = h.handle("/api/v1/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["estimates"]["emoji"] > 0

    def test_version_prefix_v2_get_budget(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("GET")
        result = h.handle("/api/v2/context/budget", {}, mock_h)
        assert _status(result) == 200

    def test_version_prefix_v2_estimate(self):
        h = ContextBudgetHandler({})
        mock_h = _make_handler("POST", {"sections": {"x": "y"}})
        result = h.handle("/api/v2/context/budget/estimate", {}, mock_h)
        assert _status(result) == 200

    def test_total_tokens_none_explicit(self):
        """Explicitly passing total_tokens: null in JSON should be treated as absent."""
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"total_tokens": None})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_tokens"] is None

    def test_section_limits_none_explicit(self):
        """Explicitly passing section_limits: null should be treated as absent."""
        h = ContextBudgetHandler({})
        mock_h = _make_handler("PUT", {"section_limits": None})
        result = h.handle("/api/v1/context/budget", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["section_limits"] is None
