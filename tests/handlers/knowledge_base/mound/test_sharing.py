"""Tests for SharingOperationsMixin (aragora/server/handlers/knowledge_base/mound/sharing.py).

Covers all routes and behavior of the sharing mixin:
- POST   /api/v1/knowledge/mound/share          - Share item with workspace/user
- GET    /api/v1/knowledge/mound/shared-with-me  - List items shared with me
- DELETE /api/v1/knowledge/mound/share           - Revoke a share
- GET    /api/v1/knowledge/mound/my-shares       - List items I've shared
- PATCH  /api/v1/knowledge/mound/share           - Update share permissions
- Error cases: missing mound, invalid body, missing fields, server errors
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.sharing import (
    SharingOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock user context
# ---------------------------------------------------------------------------


@dataclass
class MockUser:
    """Mock user context returned by require_auth_or_error."""

    id: str = "test-user-001"
    user_id: str = "test-user-001"
    email: str = "test@example.com"
    roles: list[str] = field(default_factory=lambda: ["admin"])
    permissions: list[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for sharing tests."""

    command: str = "GET"
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Length": "0"}
    )
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def with_body(cls, body: dict, method: str = "POST") -> MockHTTPHandler:
        """Create a handler with a JSON body."""
        raw = json.dumps(body).encode("utf-8")
        return cls(
            command=method,
            headers={"Content-Length": str(len(raw))},
            rfile=io.BytesIO(raw),
        )

    @classmethod
    def empty(cls, method: str = "POST") -> MockHTTPHandler:
        """Create a handler with no body (Content-Length: 0)."""
        return cls(
            command=method,
            headers={"Content-Length": "0"},
            rfile=io.BytesIO(b""),
        )

    @classmethod
    def invalid_json(cls, method: str = "POST") -> MockHTTPHandler:
        """Create a handler with invalid JSON body."""
        raw = b"not valid json {"
        return cls(
            command=method,
            headers={"Content-Length": str(len(raw))},
            rfile=io.BytesIO(raw),
        )


# ---------------------------------------------------------------------------
# Mock shared item for get_shared_with_me
# ---------------------------------------------------------------------------


@dataclass
class MockSharedItem:
    """Mock shared item with to_dict support."""

    id: str = "item-001"
    content: str = "Shared knowledge item"
    shared_by: str = "user-001"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "shared_by": self.shared_by,
        }


@dataclass
class MockSharedItemNoDict:
    """Mock shared item without to_dict (fallback path)."""

    id: str = "item-002"
    content: str = "Item without to_dict"


@dataclass
class MockShareGrant:
    """Mock share grant with to_dict support."""

    item_id: str = "item-001"
    grantee_id: str = "ws-target"
    permissions: list[str] = field(default_factory=lambda: ["read"])
    shared_by: str = "user-001"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "shared_by": self.shared_by,
        }


@dataclass
class MockUpdatedGrant:
    """Mock updated grant with to_dict support."""

    item_id: str = "item-001"
    grantee_id: str = "ws-target"
    permissions: list[str] = field(default_factory=lambda: ["read", "write"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
        }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class SharingTestHandler(SharingOperationsMixin):
    """Concrete handler for testing the sharing mixin."""

    def __init__(self, mound=None, user=None):
        self._mound = mound
        self._user = user or MockUser()

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None


class SharingTestHandlerNoAuth(SharingOperationsMixin):
    """Handler that simulates auth failure."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Authentication required", 401)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with sharing methods."""
    mound = MagicMock()
    mound.share_with_workspace = AsyncMock(return_value=None)
    mound.share_with_user = AsyncMock(return_value=None)
    mound.get_shared_with_me = AsyncMock(
        return_value=[
            MockSharedItem(id="item-001", content="First item"),
            MockSharedItem(id="item-002", content="Second item"),
            MockSharedItem(id="item-003", content="Third item"),
        ]
    )
    mound.revoke_share = AsyncMock(return_value=True)
    mound.get_share_grants = AsyncMock(
        return_value=[
            MockShareGrant(item_id="item-001", grantee_id="ws-target"),
            MockShareGrant(item_id="item-002", grantee_id="user-target"),
        ]
    )
    mound.update_share_permissions = AsyncMock(
        return_value=MockUpdatedGrant()
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a SharingTestHandler with a mocked mound."""
    return SharingTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a SharingTestHandler with no mound (None)."""
    return SharingTestHandler(mound=None)


@pytest.fixture
def handler_no_auth(mock_mound):
    """Create a SharingTestHandler that fails auth."""
    return SharingTestHandlerNoAuth(mound=mock_mound)


# ============================================================================
# Tests: _handle_share_item (POST /api/knowledge/mound/share)
# ============================================================================


class TestShareItem:
    """Test _handle_share_item (POST /api/knowledge/mound/share)."""

    def test_share_with_workspace_success(self, handler, mock_mound):
        """Successfully sharing an item with a workspace returns 201."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
            "permissions": ["read", "write"],
            "from_workspace_id": "ws-source",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["share"]["item_id"] == "item-001"
        assert body["share"]["target_type"] == "workspace"
        assert body["share"]["target_id"] == "ws-target"
        assert body["share"]["permissions"] == ["read", "write"]
        assert body["share"]["shared_by"] == "test-user-001"
        mock_mound.share_with_workspace.assert_called_once()

    def test_share_with_user_success(self, handler, mock_mound):
        """Successfully sharing an item with a user returns 201."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["share"]["target_type"] == "user"
        assert body["share"]["target_id"] == "user-target"
        mock_mound.share_with_user.assert_called_once()

    def test_share_default_permissions(self, handler, mock_mound):
        """Default permissions are ['read'] when not specified."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["permissions"] == ["read"]

    def test_share_default_from_workspace_id(self, handler, mock_mound):
        """Default from_workspace_id is 'default' when not specified."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        handler._handle_share_item(http)
        call_kwargs = mock_mound.share_with_workspace.call_args
        assert call_kwargs.kwargs["from_workspace_id"] == "default"

    def test_share_with_expires_at(self, handler, mock_mound):
        """Sharing with expires_at parses ISO format correctly."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
            "expires_at": "2025-12-31T23:59:59Z",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["expires_at"] is not None
        assert "2025-12-31" in body["share"]["expires_at"]

    def test_share_with_message(self, handler, mock_mound):
        """Sharing with optional message includes it in response."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
            "message": "Check this out!",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["message"] == "Check this out!"

    def test_share_no_message_returns_null(self, handler, mock_mound):
        """When no message is provided, response message is None."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        body = _body(result)
        assert body["share"]["message"] is None

    def test_share_no_expires_returns_null(self, handler, mock_mound):
        """When no expires_at is provided, response expires_at is None."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        body = _body(result)
        assert body["share"]["expires_at"] is None

    def test_share_missing_item_id_returns_400(self, handler):
        """Missing item_id returns 400."""
        http = MockHTTPHandler.with_body({
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "item_id" in body["error"].lower()

    def test_share_empty_item_id_returns_400(self, handler):
        """Empty string item_id returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400

    def test_share_missing_target_type_returns_400(self, handler):
        """Missing target_type returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "target_type" in body["error"].lower()

    def test_share_invalid_target_type_returns_400(self, handler):
        """Invalid target_type (not 'workspace' or 'user') returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "group",
            "target_id": "grp-001",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "target_type" in body["error"].lower()

    def test_share_missing_target_id_returns_400(self, handler):
        """Missing target_id returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "target_id" in body["error"].lower()

    def test_share_empty_target_id_returns_400(self, handler):
        """Empty string target_id returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400

    def test_share_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower()

    def test_share_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower() or "body" in body["error"].lower()

    def test_share_invalid_expires_at_returns_400(self, handler):
        """Invalid expires_at format returns 400."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
            "expires_at": "not-a-date",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 400
        body = _body(result)
        assert "expires_at" in body["error"].lower() or "iso" in body["error"].lower()

    def test_share_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler_no_mound._handle_share_item(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_share_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler_no_auth._handle_share_item(http)
        assert _status(result) == 401

    def test_share_workspace_value_error_returns_404(self, handler, mock_mound):
        """ValueError from share_with_workspace returns 404."""
        mock_mound.share_with_workspace = AsyncMock(
            side_effect=ValueError("Item not found")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "missing-item",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_share_user_value_error_returns_404(self, handler, mock_mound):
        """ValueError from share_with_user returns 404."""
        mock_mound.share_with_user = AsyncMock(
            side_effect=ValueError("Item not found")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "missing-item",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 404

    def test_share_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.share_with_workspace = AsyncMock(
            side_effect=OSError("disk fail")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 500

    def test_share_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.share_with_workspace = AsyncMock(
            side_effect=KeyError("missing key")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 500

    def test_share_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.share_with_user = AsyncMock(
            side_effect=TypeError("wrong type")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 500

    def test_share_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.share_with_workspace = AsyncMock(
            side_effect=RuntimeError("runtime fail")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 500

    def test_share_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.share_with_user = AsyncMock(
            side_effect=AttributeError("no attr")
        )
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 500

    def test_share_workspace_forwards_correct_kwargs(self, handler, mock_mound):
        """share_with_workspace is called with correct keyword args."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-abc",
            "target_type": "workspace",
            "target_id": "ws-dest",
            "permissions": ["read", "write"],
            "from_workspace_id": "ws-src",
        })
        handler._handle_share_item(http)
        call_kwargs = mock_mound.share_with_workspace.call_args.kwargs
        assert call_kwargs["item_id"] == "item-abc"
        assert call_kwargs["from_workspace_id"] == "ws-src"
        assert call_kwargs["to_workspace_id"] == "ws-dest"
        assert call_kwargs["shared_by"] == "test-user-001"
        assert call_kwargs["permissions"] == ["read", "write"]

    def test_share_user_forwards_correct_kwargs(self, handler, mock_mound):
        """share_with_user is called with correct keyword args."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-xyz",
            "target_type": "user",
            "target_id": "user-dest",
            "permissions": ["read"],
            "from_workspace_id": "ws-src",
        })
        handler._handle_share_item(http)
        call_kwargs = mock_mound.share_with_user.call_args.kwargs
        assert call_kwargs["item_id"] == "item-xyz"
        assert call_kwargs["from_workspace_id"] == "ws-src"
        assert call_kwargs["user_id"] == "user-dest"
        assert call_kwargs["shared_by"] == "test-user-001"
        assert call_kwargs["permissions"] == ["read"]

    @patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share")
    def test_share_tracks_metrics_for_workspace(self, mock_track, handler, mock_mound):
        """track_share is called with correct action and target_type for workspace."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        handler._handle_share_item(http)
        mock_track.assert_called_once_with(action="share", target_type="workspace")

    @patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share")
    def test_share_tracks_metrics_for_user(self, mock_track, handler, mock_mound):
        """track_share is called with correct action and target_type for user."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        handler._handle_share_item(http)
        mock_track.assert_called_once_with(action="share", target_type="user")

    def test_share_user_id_from_user_id_attr(self, mock_mound):
        """Falls back to user_id attribute when id is not available."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-user"
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = h._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["shared_by"] == "fallback-user"

    def test_share_with_iso_expires_at_plus_offset(self, handler, mock_mound):
        """expires_at with timezone offset is parsed correctly."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
            "expires_at": "2025-06-15T12:00:00+05:00",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        call_kwargs = mock_mound.share_with_workspace.call_args.kwargs
        assert call_kwargs["expires_at"] is not None


# ============================================================================
# Tests: _handle_shared_with_me (GET /api/knowledge/mound/shared-with-me)
# ============================================================================


class TestSharedWithMe:
    """Test _handle_shared_with_me (GET /api/knowledge/mound/shared-with-me)."""

    def test_shared_with_me_success(self, handler, mock_mound):
        """Successfully listing shared items returns items with count."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3
        assert len(body["items"]) == 3
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_shared_with_me_items_have_to_dict(self, handler, mock_mound):
        """Items with to_dict are serialized using to_dict."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        body = _body(result)
        assert body["items"][0]["id"] == "item-001"
        assert body["items"][0]["content"] == "First item"

    def test_shared_with_me_items_without_to_dict(self, handler, mock_mound):
        """Items without to_dict use fallback serialization."""
        mock_mound.get_shared_with_me = AsyncMock(
            return_value=[MockSharedItemNoDict(id="item-fallback", content="Fallback")]
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        body = _body(result)
        assert body["items"][0]["id"] == "item-fallback"
        assert body["items"][0]["content"] == "Fallback"

    def test_shared_with_me_custom_workspace(self, handler, mock_mound):
        """Custom workspace_id is forwarded to mound."""
        http = MockHTTPHandler()
        handler._handle_shared_with_me({"workspace_id": ["ws-custom"]}, http)
        call_kwargs = mock_mound.get_shared_with_me.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-custom"

    def test_shared_with_me_default_workspace(self, handler, mock_mound):
        """Default workspace_id is 'default'."""
        http = MockHTTPHandler()
        handler._handle_shared_with_me({}, http)
        call_kwargs = mock_mound.get_shared_with_me.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"

    def test_shared_with_me_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to mound."""
        http = MockHTTPHandler()
        handler._handle_shared_with_me({"limit": ["10"]}, http)
        call_kwargs = mock_mound.get_shared_with_me.call_args.kwargs
        assert call_kwargs["limit"] == 10

    def test_shared_with_me_custom_offset(self, handler, mock_mound):
        """Custom offset is used for slicing results."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({"offset": ["1"]}, http)
        body = _body(result)
        assert body["offset"] == 1
        # With offset=1 and 3 items, should get 2 items
        assert len(body["items"]) == 2

    def test_shared_with_me_offset_beyond_count(self, handler, mock_mound):
        """Offset beyond count returns empty items list."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({"offset": ["100"]}, http)
        body = _body(result)
        assert body["count"] == 3
        assert len(body["items"]) == 0

    def test_shared_with_me_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler()
        result = handler_no_mound._handle_shared_with_me({}, http)
        assert _status(result) == 503

    def test_shared_with_me_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler()
        result = handler_no_auth._handle_shared_with_me({}, http)
        assert _status(result) == 401

    def test_shared_with_me_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=ValueError("bad data")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=OSError("db fail")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=KeyError("missing")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=TypeError("wrong type")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=RuntimeError("runtime")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_shared_with_me = AsyncMock(
            side_effect=AttributeError("attr")
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        assert _status(result) == 500

    def test_shared_with_me_empty_results(self, handler, mock_mound):
        """Empty results return count=0 and empty items list."""
        mock_mound.get_shared_with_me = AsyncMock(return_value=[])
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        body = _body(result)
        assert body["count"] == 0
        assert body["items"] == []

    def test_shared_with_me_user_id_forwarded(self, handler, mock_mound):
        """User ID from auth context is forwarded to mound."""
        http = MockHTTPHandler()
        handler._handle_shared_with_me({}, http)
        call_kwargs = mock_mound.get_shared_with_me.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user-001"

    def test_shared_with_me_include_expired_param(self, handler, mock_mound):
        """include_expired query param is accepted (though unused)."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me(
            {"include_expired": ["true"]}, http
        )
        assert _status(result) == 200


# ============================================================================
# Tests: _handle_revoke_share (DELETE /api/knowledge/mound/share)
# ============================================================================


class TestRevokeShare:
    """Test _handle_revoke_share (DELETE /api/knowledge/mound/share)."""

    def test_revoke_share_success(self, handler, mock_mound):
        """Successfully revoking a share returns success."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["item_id"] == "item-001"
        assert body["grantee_id"] == "ws-target"
        assert body["revoked_by"] == "test-user-001"

    def test_revoke_share_calls_mound_with_correct_args(self, handler, mock_mound):
        """revoke_share is called with correct keyword args."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-abc", "grantee_id": "user-xyz"},
            method="DELETE",
        )
        handler._handle_revoke_share(http)
        call_kwargs = mock_mound.revoke_share.call_args.kwargs
        assert call_kwargs["item_id"] == "item-abc"
        assert call_kwargs["grantee_id"] == "user-xyz"
        assert call_kwargs["revoked_by"] == "test-user-001"

    def test_revoke_share_missing_item_id_returns_400(self, handler):
        """Missing item_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"grantee_id": "ws-target"}, method="DELETE"
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 400
        body = _body(result)
        assert "item_id" in body["error"].lower() or "grantee_id" in body["error"].lower()

    def test_revoke_share_missing_grantee_id_returns_400(self, handler):
        """Missing grantee_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001"}, method="DELETE"
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 400

    def test_revoke_share_missing_both_returns_400(self, handler):
        """Missing both item_id and grantee_id returns 400."""
        http = MockHTTPHandler.with_body({}, method="DELETE")
        result = handler._handle_revoke_share(http)
        assert _status(result) == 400

    def test_revoke_share_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty(method="DELETE")
        result = handler._handle_revoke_share(http)
        assert _status(result) == 400

    def test_revoke_share_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json(method="DELETE")
        result = handler._handle_revoke_share(http)
        assert _status(result) == 400

    def test_revoke_share_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler_no_mound._handle_revoke_share(http)
        assert _status(result) == 503

    def test_revoke_share_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler_no_auth._handle_revoke_share(http)
        assert _status(result) == 401

    def test_revoke_share_value_error_returns_404(self, handler, mock_mound):
        """ValueError from revoke_share returns 404."""
        mock_mound.revoke_share = AsyncMock(
            side_effect=ValueError("Share not found")
        )
        http = MockHTTPHandler.with_body(
            {"item_id": "missing", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_revoke_share_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.revoke_share = AsyncMock(side_effect=OSError("db fail"))
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 500

    def test_revoke_share_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.revoke_share = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 500

    def test_revoke_share_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.revoke_share = AsyncMock(side_effect=TypeError("wrong"))
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 500

    def test_revoke_share_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.revoke_share = AsyncMock(
            side_effect=RuntimeError("runtime")
        )
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 500

    def test_revoke_share_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.revoke_share = AsyncMock(
            side_effect=AttributeError("attr")
        )
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = handler._handle_revoke_share(http)
        assert _status(result) == 500


# ============================================================================
# Tests: _handle_my_shares (GET /api/knowledge/mound/my-shares)
# ============================================================================


class TestMyShares:
    """Test _handle_my_shares (GET /api/knowledge/mound/my-shares)."""

    def test_my_shares_success(self, handler, mock_mound):
        """Successfully listing my shares returns grants with count."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["grants"]) == 2
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_my_shares_grants_have_to_dict(self, handler, mock_mound):
        """Grants with to_dict are serialized using to_dict."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        body = _body(result)
        assert body["grants"][0]["item_id"] == "item-001"
        assert body["grants"][0]["grantee_id"] == "ws-target"

    def test_my_shares_grants_without_to_dict(self, handler, mock_mound):
        """Grants without to_dict are passed through as-is."""
        mock_mound.get_share_grants = AsyncMock(
            return_value=[
                {"item_id": "raw-001", "grantee_id": "raw-grantee"},
            ]
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        body = _body(result)
        assert body["grants"][0]["item_id"] == "raw-001"

    def test_my_shares_custom_workspace(self, handler, mock_mound):
        """Custom workspace_id is forwarded to mound."""
        http = MockHTTPHandler()
        handler._handle_my_shares({"workspace_id": ["ws-custom"]}, http)
        call_kwargs = mock_mound.get_share_grants.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-custom"

    def test_my_shares_default_workspace(self, handler, mock_mound):
        """Default workspace_id is 'default'."""
        http = MockHTTPHandler()
        handler._handle_my_shares({}, http)
        call_kwargs = mock_mound.get_share_grants.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"

    def test_my_shares_user_id_forwarded(self, handler, mock_mound):
        """User ID from auth context is forwarded as shared_by."""
        http = MockHTTPHandler()
        handler._handle_my_shares({}, http)
        call_kwargs = mock_mound.get_share_grants.call_args.kwargs
        assert call_kwargs["shared_by"] == "test-user-001"

    def test_my_shares_custom_limit(self, handler, mock_mound):
        """Custom limit is reflected in response."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({"limit": ["5"]}, http)
        body = _body(result)
        assert body["limit"] == 5

    def test_my_shares_custom_offset(self, handler, mock_mound):
        """Custom offset is used for slicing results."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({"offset": ["1"]}, http)
        body = _body(result)
        assert body["offset"] == 1
        # With offset=1 and 2 grants, should get 1 grant
        assert len(body["grants"]) == 1

    def test_my_shares_offset_beyond_count(self, handler, mock_mound):
        """Offset beyond count returns empty grants list."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({"offset": ["100"]}, http)
        body = _body(result)
        assert body["count"] == 2
        assert len(body["grants"]) == 0

    def test_my_shares_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler()
        result = handler_no_mound._handle_my_shares({}, http)
        assert _status(result) == 503

    def test_my_shares_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler()
        result = handler_no_auth._handle_my_shares({}, http)
        assert _status(result) == 401

    def test_my_shares_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=ValueError("bad")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=OSError("db fail")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=KeyError("missing")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=TypeError("wrong")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=RuntimeError("runtime")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_share_grants = AsyncMock(
            side_effect=AttributeError("attr")
        )
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        assert _status(result) == 500

    def test_my_shares_empty_results(self, handler, mock_mound):
        """Empty results return count=0 and empty grants list."""
        mock_mound.get_share_grants = AsyncMock(return_value=[])
        http = MockHTTPHandler()
        result = handler._handle_my_shares({}, http)
        body = _body(result)
        assert body["count"] == 0
        assert body["grants"] == []


# ============================================================================
# Tests: _handle_update_share (PATCH /api/knowledge/mound/share)
# ============================================================================


class TestUpdateShare:
    """Test _handle_update_share (PATCH /api/knowledge/mound/share)."""

    def test_update_share_permissions_success(self, handler, mock_mound):
        """Successfully updating share permissions returns success."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read", "write"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "grant" in body

    def test_update_share_grant_has_to_dict(self, handler, mock_mound):
        """Updated grant with to_dict is serialized using to_dict."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read", "write"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        body = _body(result)
        assert body["grant"]["item_id"] == "item-001"
        assert body["grant"]["permissions"] == ["read", "write"]

    def test_update_share_grant_without_to_dict(self, handler, mock_mound):
        """Updated grant without to_dict uses fallback serialization."""
        mock_mound.update_share_permissions = AsyncMock(
            return_value="non-dict-result"
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        body = _body(result)
        # Fallback serialization
        assert body["grant"]["item_id"] == "item-001"
        assert body["grant"]["grantee_id"] == "ws-target"
        assert body["grant"]["permissions"] == ["read"]

    def test_update_share_with_expires_at(self, handler, mock_mound):
        """Updating share with expires_at parses ISO format."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "expires_at": "2025-12-31T23:59:59Z",
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200
        call_kwargs = mock_mound.update_share_permissions.call_args.kwargs
        assert call_kwargs["expires_at"] is not None

    def test_update_share_permissions_only(self, handler, mock_mound):
        """Updating only permissions (no expires_at) succeeds."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read", "write", "admin"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200
        call_kwargs = mock_mound.update_share_permissions.call_args.kwargs
        assert call_kwargs["permissions"] == ["read", "write", "admin"]
        assert call_kwargs["expires_at"] is None

    def test_update_share_expires_at_only(self, handler, mock_mound):
        """Updating only expires_at (no permissions) succeeds."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "expires_at": "2026-01-01T00:00:00Z",
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200

    def test_update_share_missing_item_id_returns_400(self, handler):
        """Missing item_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"grantee_id": "ws-target", "permissions": ["read"]},
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 400

    def test_update_share_missing_grantee_id_returns_400(self, handler):
        """Missing grantee_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "permissions": ["read"]},
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 400

    def test_update_share_missing_both_ids_returns_400(self, handler):
        """Missing both item_id and grantee_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"permissions": ["read"]}, method="PATCH"
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 400

    def test_update_share_no_permissions_or_expires_returns_400(self, handler):
        """Missing both permissions and expires_at returns 400."""
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 400
        body = _body(result)
        assert "permissions" in body["error"].lower() or "expires_at" in body["error"].lower()

    def test_update_share_invalid_expires_at_returns_400(self, handler):
        """Invalid expires_at format returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "expires_at": "not-a-date",
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 400
        body = _body(result)
        assert "expires_at" in body["error"].lower() or "iso" in body["error"].lower()

    def test_update_share_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty(method="PATCH")
        result = handler._handle_update_share(http)
        assert _status(result) == 400

    def test_update_share_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json(method="PATCH")
        result = handler._handle_update_share(http)
        assert _status(result) == 400

    def test_update_share_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler_no_mound._handle_update_share(http)
        assert _status(result) == 503

    def test_update_share_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler_no_auth._handle_update_share(http)
        assert _status(result) == 401

    def test_update_share_value_error_returns_404(self, handler, mock_mound):
        """ValueError from update_share_permissions returns 404."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=ValueError("Share not found")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "missing",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_update_share_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=OSError("db fail")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 500

    def test_update_share_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=KeyError("missing")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 500

    def test_update_share_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=TypeError("wrong")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 500

    def test_update_share_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=RuntimeError("runtime")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 500

    def test_update_share_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.update_share_permissions = AsyncMock(
            side_effect=AttributeError("attr")
        )
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 500

    def test_update_share_forwards_correct_kwargs(self, handler, mock_mound):
        """update_share_permissions is called with correct keyword args."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-abc",
                "grantee_id": "user-xyz",
                "permissions": ["read", "write"],
                "expires_at": "2025-12-31T23:59:59Z",
            },
            method="PATCH",
        )
        handler._handle_update_share(http)
        call_kwargs = mock_mound.update_share_permissions.call_args.kwargs
        assert call_kwargs["item_id"] == "item-abc"
        assert call_kwargs["grantee_id"] == "user-xyz"
        assert call_kwargs["permissions"] == ["read", "write"]
        assert call_kwargs["updated_by"] == "test-user-001"
        assert call_kwargs["expires_at"] is not None

    def test_update_share_with_iso_expires_at_plus_offset(self, handler, mock_mound):
        """expires_at with timezone offset is parsed correctly."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "expires_at": "2025-06-15T12:00:00+05:00",
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200


# ============================================================================
# Tests: routing integration
# ============================================================================


class TestSharingRouting:
    """Test routing dispatch for sharing endpoints via _route_share."""

    def test_route_share_post_dispatches_to_share_item(self, handler, mock_mound):
        """POST /share dispatches to _handle_share_item."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
        })
        http.command = "POST"
        result = RoutingMixin._route_share(handler, "/share", {}, http)
        assert result is not None
        assert _status(result) == 201

    def test_route_share_delete_dispatches_to_revoke(self, handler, mock_mound):
        """DELETE /share dispatches to _handle_revoke_share."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        http.command = "DELETE"
        result = RoutingMixin._route_share(handler, "/share", {}, http)
        assert result is not None
        assert _status(result) == 200

    def test_route_share_patch_dispatches_to_update(self, handler, mock_mound):
        """PATCH /share dispatches to _handle_update_share."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        http.command = "PATCH"
        result = RoutingMixin._route_share(handler, "/share", {}, http)
        assert result is not None
        assert _status(result) == 200

    def test_route_share_get_returns_none(self, handler):
        """GET /share returns None (unsupported method)."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler()
        http.command = "GET"
        result = RoutingMixin._route_share(handler, "/share", {}, http)
        assert result is None


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestSharingEdgeCases:
    """Test edge cases across sharing operations."""

    def test_share_item_user_id_fallback_to_unknown(self, mock_mound):
        """User without id or user_id falls back to 'unknown'."""
        user = MagicMock(spec=[])
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
        })
        result = h._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["shared_by"] == "unknown"

    def test_shared_with_me_user_id_fallback(self, mock_mound):
        """User without id attribute falls back to user_id."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-id"
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler()
        h._handle_shared_with_me({}, http)
        call_kwargs = mock_mound.get_shared_with_me.call_args.kwargs
        assert call_kwargs["user_id"] == "fallback-id"

    def test_my_shares_user_id_fallback(self, mock_mound):
        """User without id attribute falls back to user_id in my_shares."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-id"
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler()
        h._handle_my_shares({}, http)
        call_kwargs = mock_mound.get_share_grants.call_args.kwargs
        assert call_kwargs["shared_by"] == "fallback-id"

    def test_revoke_share_user_id_fallback(self, mock_mound):
        """User without id attribute falls back to user_id in revoke."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-id"
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {"item_id": "item-001", "grantee_id": "ws-target"},
            method="DELETE",
        )
        result = h._handle_revoke_share(http)
        body = _body(result)
        assert body["revoked_by"] == "fallback-id"

    def test_update_share_user_id_fallback(self, mock_mound):
        """User without id attribute falls back to user_id in update."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-id"
        h = SharingTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
            },
            method="PATCH",
        )
        h._handle_update_share(http)
        call_kwargs = mock_mound.update_share_permissions.call_args.kwargs
        assert call_kwargs["updated_by"] == "fallback-id"

    def test_shared_with_me_limit_max_clamp(self, handler, mock_mound):
        """Limit is clamped to max 200."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({"limit": ["999"]}, http)
        body = _body(result)
        assert body["limit"] == 200

    def test_shared_with_me_limit_min_clamp(self, handler, mock_mound):
        """Limit is clamped to min 1."""
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({"limit": ["0"]}, http)
        body = _body(result)
        assert body["limit"] == 1

    def test_my_shares_limit_max_clamp(self, handler, mock_mound):
        """Limit is clamped to max 200."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({"limit": ["999"]}, http)
        body = _body(result)
        assert body["limit"] == 200

    def test_my_shares_limit_min_clamp(self, handler, mock_mound):
        """Limit is clamped to min 1."""
        http = MockHTTPHandler()
        result = handler._handle_my_shares({"limit": ["0"]}, http)
        body = _body(result)
        assert body["limit"] == 1

    def test_share_item_both_permissions_and_expires(self, handler, mock_mound):
        """Share with both custom permissions and expires_at."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "workspace",
            "target_id": "ws-target",
            "permissions": ["read", "write", "admin"],
            "expires_at": "2025-12-31T23:59:59Z",
            "message": "Full access until year end",
            "from_workspace_id": "ws-source",
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["share"]["permissions"] == ["read", "write", "admin"]
        assert body["share"]["expires_at"] is not None
        assert body["share"]["message"] == "Full access until year end"

    def test_update_share_both_permissions_and_expires(self, handler, mock_mound):
        """Update with both permissions and expires_at."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "grantee_id": "ws-target",
                "permissions": ["read"],
                "expires_at": "2026-06-30T00:00:00Z",
            },
            method="PATCH",
        )
        result = handler._handle_update_share(http)
        assert _status(result) == 200
        call_kwargs = mock_mound.update_share_permissions.call_args.kwargs
        assert call_kwargs["permissions"] == ["read"]
        assert call_kwargs["expires_at"] is not None

    def test_shared_with_me_mixed_to_dict_items(self, handler, mock_mound):
        """Items mix of with and without to_dict are handled."""
        mock_mound.get_shared_with_me = AsyncMock(
            return_value=[
                MockSharedItem(id="with-dict", content="Has to_dict"),
                MockSharedItemNoDict(id="no-dict", content="No to_dict"),
            ]
        )
        http = MockHTTPHandler()
        result = handler._handle_shared_with_me({}, http)
        body = _body(result)
        assert body["count"] == 2
        assert body["items"][0]["id"] == "with-dict"
        assert "shared_by" in body["items"][0]  # from to_dict
        assert body["items"][1]["id"] == "no-dict"
        assert "shared_by" not in body["items"][1]  # fallback

    def test_share_item_empty_permissions_list(self, handler, mock_mound):
        """Empty permissions list is passed through (defaults to ['read'])."""
        http = MockHTTPHandler.with_body({
            "item_id": "item-001",
            "target_type": "user",
            "target_id": "user-target",
            "permissions": [],
        })
        result = handler._handle_share_item(http)
        assert _status(result) == 201
        body = _body(result)
        # Empty list is valid (no default override since key exists)
        assert body["share"]["permissions"] == []
