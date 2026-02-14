"""
Tests for aragora.server.handlers.audience_suggestions - Audience suggestion handler.

Tests cover:
- Instantiation and ROUTES
- can_handle() route matching
- GET /api/v1/audience/suggestions - list clustered suggestions
- POST /api/v1/audience/suggestions - submit a suggestion
- Permission checks for submission
- Input validation (debate_id, suggestion text, threshold)
- Error handling when storage or clustering fails
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.audience_suggestions import AudienceSuggestionsHandler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create an AudienceSuggestionsHandler with mocked storage."""
    storage = MagicMock()
    storage.get_audience_suggestions = MagicMock(return_value=[])
    storage.save_audience_suggestion = MagicMock()
    ctx: dict[str, Any] = {"storage": storage}
    return AudienceSuggestionsHandler(ctx)


@pytest.fixture
def handler_no_storage():
    """Create handler without storage."""
    ctx: dict[str, Any] = {}
    return AudienceSuggestionsHandler(ctx)


@pytest.fixture
def mock_get():
    """Create a mock HTTP GET handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {"Authorization": "Bearer test-token"}
    return mock


def make_post(body: dict) -> MagicMock:
    """Create a mock HTTP POST handler with JSON body."""
    mock = MagicMock()
    mock.command = "POST"
    body_bytes = json.dumps(body).encode()
    mock.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
        "Content-Length": str(len(body_bytes)),
    }
    mock.rfile = io.BytesIO(body_bytes)
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
        assert "/api/v1/audience/suggestions" in AudienceSuggestionsHandler.ROUTES

    def test_routes_count(self):
        """Should have exactly 1 route."""
        assert len(AudienceSuggestionsHandler.ROUTES) == 1


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_can_handle_suggestions(self, handler):
        """Should handle /api/v1/audience/suggestions."""
        assert handler.can_handle("/api/v1/audience/suggestions") is True

    def test_cannot_handle_unknown_path(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/v1/audience") is False
        assert handler.can_handle("/api/v1/audience/other") is False
        assert handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Method Not Allowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for unsupported method rejection."""

    def test_delete_returns_405(self, handler):
        """Should reject DELETE requests with 405."""
        mock = MagicMock()
        mock.command = "DELETE"
        result = handler.handle("/api/v1/audience/suggestions", {}, mock)
        assert result.status_code == 405

    def test_put_returns_405(self, handler):
        """Should reject PUT requests with 405."""
        mock = MagicMock()
        mock.command = "PUT"
        result = handler.handle("/api/v1/audience/suggestions", {}, mock)
        assert result.status_code == 405


# ===========================================================================
# GET - List Suggestions
# ===========================================================================


@dataclass
class FakeCluster:
    representative: str
    count: int
    user_ids: list[str]


class TestListSuggestions:
    """Tests for GET /api/v1/audience/suggestions."""

    def test_missing_debate_id_returns_400(self, handler, mock_get):
        """Should return 400 when debate_id missing."""
        result = handler.handle("/api/v1/audience/suggestions", {}, mock_get)
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "debate_id" in data.get("error", "")

    def test_list_success(self, handler, mock_get):
        """Should return clustered suggestions."""
        clusters = [
            FakeCluster(representative="Improve performance", count=3, user_ids=["u1", "u2", "u3"]),
            FakeCluster(representative="Add dark mode", count=1, user_ids=["u4"]),
        ]

        with patch(
            "aragora.audience.suggestions.cluster_suggestions",
            return_value=clusters,
        ):
            result = handler.handle(
                "/api/v1/audience/suggestions",
                {"debate_id": "debate-123"},
                mock_get,
            )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert data["total_clusters"] == 2
        assert data["clusters"][0]["representative"] == "Improve performance"
        assert data["clusters"][0]["count"] == 3

    def test_list_with_max_clusters_param(self, handler, mock_get):
        """Should pass max_clusters to cluster_suggestions."""
        with patch(
            "aragora.audience.suggestions.cluster_suggestions",
            return_value=[],
        ) as mock_cluster:
            handler.handle(
                "/api/v1/audience/suggestions",
                {"debate_id": "d-1", "max_clusters": "10"},
                mock_get,
            )
            _, kwargs = mock_cluster.call_args
            assert kwargs["max_clusters"] == 10

    def test_list_with_threshold_param(self, handler, mock_get):
        """Should pass threshold to cluster_suggestions."""
        with patch(
            "aragora.audience.suggestions.cluster_suggestions",
            return_value=[],
        ) as mock_cluster:
            handler.handle(
                "/api/v1/audience/suggestions",
                {"debate_id": "d-1", "threshold": "0.8"},
                mock_get,
            )
            _, kwargs = mock_cluster.call_args
            assert kwargs["similarity_threshold"] == 0.8

    def test_invalid_threshold_returns_400(self, handler, mock_get):
        """Should return 400 for out-of-range threshold."""
        result = handler.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d-1", "threshold": "1.5"},
            mock_get,
        )
        assert result.status_code == 400

    def test_no_storage_returns_503(self, handler_no_storage, mock_get):
        """Should return 503 when storage not available."""
        result = handler_no_storage.handle(
            "/api/v1/audience/suggestions",
            {"debate_id": "d-1"},
            mock_get,
        )
        assert result.status_code == 503

    def test_cluster_failure_returns_500(self, handler, mock_get):
        """Should return 500 when clustering fails."""
        with patch(
            "aragora.audience.suggestions.cluster_suggestions",
            side_effect=RuntimeError("clustering error"),
        ):
            result = handler.handle(
                "/api/v1/audience/suggestions",
                {"debate_id": "d-1"},
                mock_get,
            )
        assert result.status_code == 500


# ===========================================================================
# POST - Submit Suggestion
# ===========================================================================


class TestSubmitSuggestion:
    """Tests for POST /api/v1/audience/suggestions."""

    def test_permission_required(self, handler):
        """Should require audience:write permission."""
        mock_post = make_post({"debate_id": "d-1", "suggestion": "test"})
        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, MagicMock(status_code=403, body=b'{"error":"Forbidden"}')),
        ):
            result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)
            assert result.status_code == 403

    def test_submit_success(self, handler):
        """Should accept and sanitize a valid suggestion."""
        mock_user = MagicMock()
        mock_user.user_id = "user-42"
        mock_post = make_post({"debate_id": "d-1", "suggestion": "Add caching layer"})

        with patch.object(
            handler, "require_permission_or_error", return_value=(mock_user, None)
        ):
            with patch(
                "aragora.audience.suggestions.sanitize_suggestion",
                return_value="Add caching layer",
            ):
                result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)

        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["status"] == "accepted"
        assert data["debate_id"] == "d-1"

    def test_missing_debate_id_returns_400(self, handler):
        """Should return 400 when debate_id is missing."""
        mock_user = MagicMock()
        mock_post = make_post({"suggestion": "test"})

        with patch.object(
            handler, "require_permission_or_error", return_value=(mock_user, None)
        ):
            result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)
        assert result.status_code == 400

    def test_missing_suggestion_returns_400(self, handler):
        """Should return 400 when suggestion is empty."""
        mock_user = MagicMock()
        mock_post = make_post({"debate_id": "d-1", "suggestion": ""})

        with patch.object(
            handler, "require_permission_or_error", return_value=(mock_user, None)
        ):
            result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)
        assert result.status_code == 400

    def test_too_long_suggestion_returns_400(self, handler):
        """Should return 400 when suggestion exceeds 500 chars."""
        mock_user = MagicMock()
        long_text = "x" * 501
        mock_post = make_post({"debate_id": "d-1", "suggestion": long_text})

        with patch.object(
            handler, "require_permission_or_error", return_value=(mock_user, None)
        ):
            result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)
        assert result.status_code == 400

    def test_invalid_json_body_returns_400(self, handler):
        """Should return 400 for invalid JSON body."""
        mock_user = MagicMock()
        mock_post = MagicMock()
        mock_post.command = "POST"
        mock_post.headers = {
            "Content-Type": "application/json",
            "Content-Length": "5",
        }
        mock_post.rfile = io.BytesIO(b"notjson")  # Will fail JSON parse

        with patch.object(
            handler, "require_permission_or_error", return_value=(mock_user, None)
        ):
            # read_json_body returns None on parse error
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler.handle("/api/v1/audience/suggestions", {}, mock_post)
        assert result.status_code == 400

    def test_no_storage_returns_503(self, handler_no_storage):
        """Should return 503 when storage not available."""
        mock_user = MagicMock()
        mock_user.user_id = "u-1"
        mock_post = make_post({"debate_id": "d-1", "suggestion": "test"})

        with patch.object(
            handler_no_storage, "require_permission_or_error", return_value=(mock_user, None)
        ):
            with patch(
                "aragora.audience.suggestions.sanitize_suggestion",
                return_value="test",
            ):
                result = handler_no_storage.handle(
                    "/api/v1/audience/suggestions", {}, mock_post
                )
        assert result.status_code == 503
