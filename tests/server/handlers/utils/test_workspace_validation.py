"""
Tests for X-Workspace-ID header validation.

Covers workspace ID extraction, membership validation,
anonymous user handling, and graceful fallback on errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.utils.auth import _extract_workspace_id


# ===========================================================================
# Helpers
# ===========================================================================


def _make_request(workspace_id: str | None = None, workspace_store=None):
    """Create a mock request with optional workspace header and app store."""
    request = MagicMock()
    headers = {}
    if workspace_id is not None:
        headers["X-Workspace-ID"] = workspace_id
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: headers.get(key, default)

    app_mock = MagicMock()
    if workspace_store is not None:
        app_mock.get = MagicMock(
            side_effect=lambda key, *a: workspace_store if key == "workspace_store" else None
        )
    else:
        app_mock.get = MagicMock(return_value=None)
    request.app = app_mock

    return request


# ===========================================================================
# Basic extraction
# ===========================================================================


class TestBasicExtraction:
    """Tests for basic workspace ID extraction from headers."""

    def test_returns_workspace_id_from_header(self):
        request = _make_request(workspace_id="ws-123")
        assert _extract_workspace_id(request) == "ws-123"

    def test_returns_none_when_no_header(self):
        request = _make_request(workspace_id=None)
        assert _extract_workspace_id(request) is None

    def test_returns_none_when_no_headers_attr(self):
        request = MagicMock(spec=[])  # No headers attribute
        assert _extract_workspace_id(request) is None


# ===========================================================================
# Anonymous user handling
# ===========================================================================


class TestAnonymousUser:
    """Tests for anonymous user workspace extraction."""

    def test_anonymous_gets_header_value(self):
        request = _make_request(workspace_id="ws-456")
        assert _extract_workspace_id(request, user_id=None) == "ws-456"

    def test_anonymous_string_gets_header_value(self):
        request = _make_request(workspace_id="ws-789")
        assert _extract_workspace_id(request, user_id="anonymous") == "ws-789"


# ===========================================================================
# Authenticated user — membership validation
# ===========================================================================


class TestMembershipValidation:
    """Tests for authenticated user workspace membership validation."""

    def test_valid_workspace_allowed(self):
        store = MagicMock()
        store.get_user_workspaces.return_value = [
            {"workspace_id": "ws-100"},
            {"workspace_id": "ws-200"},
        ]
        request = _make_request(workspace_id="ws-200", workspace_store=store)
        result = _extract_workspace_id(request, user_id="user-1")
        assert result == "ws-200"

    def test_invalid_workspace_rejected(self):
        store = MagicMock()
        store.get_user_workspaces.return_value = [
            {"workspace_id": "ws-100"},
        ]
        request = _make_request(workspace_id="ws-999", workspace_store=store)
        result = _extract_workspace_id(request, user_id="user-1")
        assert result is None

    def test_no_store_allows_header_through(self):
        """When workspace_store is not available, allow header through."""
        request = _make_request(workspace_id="ws-500")
        result = _extract_workspace_id(request, user_id="user-1")
        assert result == "ws-500"

    def test_store_error_allows_header_through(self):
        """On store errors, allow header through (graceful degradation)."""
        store = MagicMock()
        store.get_user_workspaces.side_effect = RuntimeError("DB unavailable")
        request = _make_request(workspace_id="ws-600", workspace_store=store)
        result = _extract_workspace_id(request, user_id="user-1")
        assert result == "ws-600"

    def test_empty_memberships_rejects(self):
        store = MagicMock()
        store.get_user_workspaces.return_value = []
        request = _make_request(workspace_id="ws-300", workspace_store=store)
        result = _extract_workspace_id(request, user_id="user-1")
        assert result is None

    def test_id_field_fallback(self):
        """Supports workspace objects with 'id' instead of 'workspace_id'."""
        store = MagicMock()
        store.get_user_workspaces.return_value = [
            {"id": "ws-400"},
        ]
        request = _make_request(workspace_id="ws-400", workspace_store=store)
        result = _extract_workspace_id(request, user_id="user-1")
        assert result == "ws-400"
