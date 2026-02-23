"""Tests for audience suggestions handler (aragora/server/handlers/audience_suggestions.py).

Covers all routes and behavior of the AudienceSuggestionsHandler class:
- can_handle() routing
- GET /api/v1/audience/suggestions - list clustered suggestions
- POST /api/v1/audience/suggestions - submit a new suggestion
- Permission checks (audience:read, audience:write)
- Query parameter parsing (debate_id, max_clusters, threshold)
- Body validation (debate_id, suggestion text, length limits)
- Storage availability (503 on missing storage)
- Error handling (ImportError, ValueError, etc.)
- Sanitization of suggestion text
- Edge cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.audience_suggestions import AudienceSuggestionsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for AudienceSuggestionsHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


@dataclass
class MockCluster:
    """Mock cluster object returned by cluster_suggestions."""

    representative: str
    count: int
    user_ids: list[str]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AudienceSuggestionsHandler with a mock server context."""
    ctx: dict[str, Any] = {}
    return AudienceSuggestionsHandler(server_context=ctx)


@pytest.fixture
def handler_with_storage():
    """Create handler with a mock storage backend."""
    mock_storage = MagicMock()
    mock_storage.get_audience_suggestions.return_value = []
    mock_storage.save_audience_suggestion.return_value = None
    ctx: dict[str, Any] = {"storage": mock_storage}
    h = AudienceSuggestionsHandler(server_context=ctx)
    return h, mock_storage


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching."""

    def test_matches_audience_suggestions_route(self, handler):
        assert handler.can_handle("/api/v1/audience/suggestions") is True

    def test_rejects_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/audience/other") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/audience") is False

    def test_rejects_extra_segment(self, handler):
        assert handler.can_handle("/api/v1/audience/suggestions/123") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_root_path(self, handler):
        assert handler.can_handle("/") is False


# ---------------------------------------------------------------------------
# Method dispatch tests
# ---------------------------------------------------------------------------


class TestMethodDispatch:
    """Tests for HTTP method routing in handle()."""

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_get_dispatches_to_list(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle("/api/v1/audience/suggestions", {"debate_id": "d1"}, http)
        assert _status(result) == 200

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="clean")
    def test_post_dispatches_to_submit(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "good idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 201

    def test_put_returns_405(self, handler):
        http = MockHTTPHandler(method="PUT")
        result = handler.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result)["error"]

    def test_delete_returns_405(self, handler):
        http = MockHTTPHandler(method="DELETE")
        result = handler.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 405

    def test_patch_returns_405(self, handler):
        http = MockHTTPHandler(method="PATCH")
        result = handler.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 405


# ---------------------------------------------------------------------------
# GET /api/v1/audience/suggestions tests
# ---------------------------------------------------------------------------


class TestListSuggestions:
    """Tests for listing clustered suggestions."""

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_missing_debate_id_returns_400(self, mock_cluster, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "debate_id" in _body(result)["error"]

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_empty_debate_id_returns_400(self, mock_cluster, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle("/api/v1/audience/suggestions", {"debate_id": ""}, http)
        assert _status(result) == 400

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_successful_list_with_clusters(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = [
            {"text": "idea1", "user_id": "u1"},
            {"text": "idea2", "user_id": "u2"},
        ]
        mock_cluster.return_value = [
            MockCluster(representative="idea1", count=2, user_ids=["u1", "u2"]),
        ]
        http = MockHTTPHandler(method="GET")
        result = h.handle("/api/v1/audience/suggestions", {"debate_id": "d1"}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d1"
        assert body["total_clusters"] == 1
        assert len(body["clusters"]) == 1
        assert body["clusters"][0]["representative"] == "idea1"
        assert body["clusters"][0]["count"] == 2
        assert body["clusters"][0]["user_ids"] == ["u1", "u2"]

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_successful_list_empty_suggestions(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle("/api/v1/audience/suggestions", {"debate_id": "d1"}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["clusters"] == []
        assert body["total_clusters"] == 0

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_max_clusters_param_passed_through(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "max_clusters": "10"},
            http,
        )
        assert _status(result) == 200
        mock_cluster.assert_called_once()
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_clusters"] == 10

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_max_clusters_clamped_to_min(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "max_clusters": "0"},
            http,
        )
        assert _status(result) == 200
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_clusters"] == 1  # Clamped to min_val=1

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_max_clusters_clamped_to_max(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "max_clusters": "100"},
            http,
        )
        assert _status(result) == 200
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_clusters"] == 20  # Clamped to max_val=20

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_threshold_param_default(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 200
        _, kwargs = mock_cluster.call_args
        assert kwargs["similarity_threshold"] == 0.6

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_threshold_param_custom(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "0.8"},
            http,
        )
        assert _status(result) == 200
        _, kwargs = mock_cluster.call_args
        assert kwargs["similarity_threshold"] == 0.8

    def test_threshold_invalid_value_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "notanumber"},
            http,
        )
        assert _status(result) == 400
        assert "threshold" in _body(result)["error"].lower()

    def test_threshold_below_zero_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "-0.1"},
            http,
        )
        assert _status(result) == 400
        assert "threshold" in _body(result)["error"].lower()

    def test_threshold_above_one_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "1.5"},
            http,
        )
        assert _status(result) == 400
        assert "threshold" in _body(result)["error"].lower()

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_threshold_at_zero_boundary(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "0.0"},
            http,
        )
        assert _status(result) == 200

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_threshold_at_one_boundary(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": "1.0"},
            http,
        )
        assert _status(result) == 200

    def test_storage_not_available_returns_503(self, handler):
        http = MockHTTPHandler(method="GET")
        with patch(
            "aragora.audience.suggestions.cluster_suggestions",
        ):
            result = handler.handle(
                "/api/v1/audience/suggestions",
                {"debate_id": "d1"},
                http,
            )
        assert _status(result) == 503
        assert "Storage" in _body(result)["error"]

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=ValueError("bad data"),
    )
    def test_value_error_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = [{"text": "x"}]
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500
        assert "Failed to list" in _body(result)["error"]

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=RuntimeError("unexpected"),
    )
    def test_runtime_error_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=ImportError("no module"),
    )
    def test_import_error_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_multiple_clusters_returned(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = [
            {"text": "a"}, {"text": "b"}, {"text": "c"},
        ]
        mock_cluster.return_value = [
            MockCluster(representative="a", count=1, user_ids=["u1"]),
            MockCluster(representative="b", count=2, user_ids=["u2", "u3"]),
        ]
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total_clusters"] == 2
        assert len(body["clusters"]) == 2

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=TypeError("type error"),
    )
    def test_type_error_in_clustering_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=KeyError("missing key"),
    )
    def test_key_error_in_clustering_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=AttributeError("attr"),
    )
    def test_attribute_error_in_clustering_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch(
        "aragora.audience.suggestions.cluster_suggestions",
        side_effect=OSError("os error"),
    )
    def test_os_error_in_clustering_returns_500(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    def test_threshold_none_value_returns_400(self, handler_with_storage):
        """Threshold query param as None should trigger TypeError/ValueError."""
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "threshold": None},
            http,
        )
        # float(None) raises TypeError, caught by the except
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/audience/suggestions tests
# ---------------------------------------------------------------------------


class TestSubmitSuggestion:
    """Tests for submitting audience suggestions."""

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="clean text")
    def test_successful_submit(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "my idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "accepted"
        assert body["debate_id"] == "d1"
        assert body["sanitized_text"] == "clean text"

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="clean")
    def test_storage_called_with_correct_args(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "my idea"},
        )
        h.handle("/api/v1/audience/suggestions", {}, http)
        storage.save_audience_suggestion.assert_called_once_with(
            debate_id="d1",
            user_id="test-user-001",  # From conftest mock
            suggestion="clean",
        )

    def test_missing_body_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="POST")
        # Empty body {} -> debate_id and suggestion both missing
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400

    def test_missing_debate_id_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"suggestion": "my idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "debate_id" in _body(result)["error"]

    def test_missing_suggestion_text_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "suggestion" in _body(result)["error"]

    def test_empty_suggestion_text_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": ""},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "suggestion" in _body(result)["error"]

    def test_whitespace_only_suggestion_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "   "},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "suggestion" in _body(result)["error"]

    def test_suggestion_too_long_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        long_text = "x" * 501
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": long_text},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "500 characters" in _body(result)["error"]

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="ok")
    def test_suggestion_exactly_500_chars_accepted(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        text = "x" * 500
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": text},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 201

    def test_invalid_json_body_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(method="POST")
        # Simulate invalid JSON by making rfile return non-JSON bytes
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "JSON" in _body(result)["error"]

    def test_storage_not_available_returns_503(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        with patch(
            "aragora.audience.suggestions.sanitize_suggestion",
            return_value="idea",
        ):
            result = handler.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 503
        assert "Storage" in _body(result)["error"]

    @patch(
        "aragora.audience.suggestions.sanitize_suggestion",
        side_effect=ValueError("bad input"),
    )
    def test_sanitize_value_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500
        assert "Failed to submit" in _body(result)["error"]

    @patch(
        "aragora.audience.suggestions.sanitize_suggestion",
        side_effect=ImportError("no module"),
    )
    def test_sanitize_import_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500

    @patch(
        "aragora.audience.suggestions.sanitize_suggestion",
        side_effect=RuntimeError("runtime"),
    )
    def test_sanitize_runtime_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="ok")
    def test_save_raises_os_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        storage.save_audience_suggestion.side_effect = OSError("disk fail")
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="ok")
    def test_user_id_from_user_context(self, mock_sanitize, handler_with_storage):
        """Verify user_id is extracted from the authenticated user context."""
        h, storage = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        h.handle("/api/v1/audience/suggestions", {}, http)
        call_kwargs = storage.save_audience_suggestion.call_args[1]
        # The conftest mock returns a user with user_id="test-user-001"
        assert call_kwargs["user_id"] == "test-user-001"

    def test_empty_debate_id_in_body_returns_400(self, handler_with_storage):
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "debate_id" in _body(result)["error"]

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="trimmed")
    def test_suggestion_with_leading_trailing_spaces(self, mock_sanitize, handler_with_storage):
        """Suggestion text should be stripped before length check."""
        h, storage = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "  trimmed  "},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 201


# ---------------------------------------------------------------------------
# ROUTES constant tests
# ---------------------------------------------------------------------------


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_audience_suggestions(self):
        assert "/api/v1/audience/suggestions" in AudienceSuggestionsHandler.ROUTES

    def test_routes_is_list(self):
        assert isinstance(AudienceSuggestionsHandler.ROUTES, list)

    def test_routes_length(self):
        assert len(AudienceSuggestionsHandler.ROUTES) == 1


# ---------------------------------------------------------------------------
# Handler integration edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration behavior."""

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_max_clusters_default_is_5(self, mock_cluster, handler_with_storage):
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        h.handle("/api/v1/audience/suggestions", {"debate_id": "d1"}, http)
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_clusters"] == 5

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="ok")
    def test_suggestion_key_missing_from_body_uses_empty_default(self, mock_sanitize, handler_with_storage):
        """When 'suggestion' key is absent, body.get('suggestion', '') returns ''."""
        h, _ = handler_with_storage
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1"},  # no 'suggestion' key
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 400
        assert "suggestion" in _body(result)["error"]

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_max_clusters_non_numeric_uses_default(self, mock_cluster, handler_with_storage):
        """Non-numeric max_clusters falls back to default via safe_query_int."""
        h, storage = handler_with_storage
        storage.get_audience_suggestions.return_value = []
        mock_cluster.return_value = []
        http = MockHTTPHandler(method="GET")
        h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1", "max_clusters": "abc"},
            http,
        )
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_clusters"] == 5  # default

    @patch("aragora.audience.suggestions.cluster_suggestions")
    def test_storage_get_raises_attribute_error(self, mock_cluster, handler_with_storage):
        """AttributeError from storage is caught."""
        h, storage = handler_with_storage
        storage.get_audience_suggestions.side_effect = AttributeError("no attr")
        http = MockHTTPHandler(method="GET")
        result = h.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d1"},
            http,
        )
        assert _status(result) == 500

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="x")
    def test_save_raises_key_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        storage.save_audience_suggestion.side_effect = KeyError("k")
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500

    @patch("aragora.audience.suggestions.sanitize_suggestion", return_value="x")
    def test_save_raises_type_error_returns_500(self, mock_sanitize, handler_with_storage):
        h, storage = handler_with_storage
        storage.save_audience_suggestion.side_effect = TypeError("t")
        http = MockHTTPHandler(
            method="POST",
            body={"debate_id": "d1", "suggestion": "idea"},
        )
        result = h.handle("/api/v1/audience/suggestions", {}, http)
        assert _status(result) == 500
