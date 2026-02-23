"""Tests for VisibilityOperationsMixin (aragora/server/handlers/knowledge_base/mound/visibility.py).

Covers all routes and behavior of the visibility mixin:
- PUT  /api/v1/knowledge/mound/nodes/:id/visibility - Set item visibility
- GET  /api/v1/knowledge/mound/nodes/:id/visibility - Get item visibility
- POST /api/v1/knowledge/mound/nodes/:id/access     - Grant access to item
- DELETE /api/v1/knowledge/mound/nodes/:id/access    - Revoke access
- GET  /api/v1/knowledge/mound/nodes/:id/access      - List access grants
- Error cases: missing mound, invalid body, missing fields, server errors
- Security: path traversal, injection
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.visibility import (
    VisibilityOperationsMixin,
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
    workspace_id: str = "ws-default"
    roles: list[str] = field(default_factory=lambda: ["admin"])
    permissions: list[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for visibility tests."""

    command: str = "GET"
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Length": "0"})
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def with_body(cls, body: dict, method: str = "PUT") -> MockHTTPHandler:
        """Create a handler with a JSON body."""
        raw = json.dumps(body).encode("utf-8")
        return cls(
            command=method,
            headers={"Content-Length": str(len(raw))},
            rfile=io.BytesIO(raw),
        )

    @classmethod
    def empty(cls, method: str = "PUT") -> MockHTTPHandler:
        """Create a handler with no body (Content-Length: 0)."""
        return cls(
            command=method,
            headers={"Content-Length": "0"},
            rfile=io.BytesIO(b""),
        )

    @classmethod
    def invalid_json(cls, method: str = "PUT") -> MockHTTPHandler:
        """Create a handler with invalid JSON body."""
        raw = b"not valid json {"
        return cls(
            command=method,
            headers={"Content-Length": str(len(raw))},
            rfile=io.BytesIO(raw),
        )


# ---------------------------------------------------------------------------
# Mock node for get_visibility
# ---------------------------------------------------------------------------


@dataclass
class MockNode:
    """Mock knowledge node with metadata."""

    id: str = "node-001"
    content: str = "Test node"
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "visibility": "workspace",
                "visibility_set_by": "admin-user",
                "is_discoverable": True,
            }


# ---------------------------------------------------------------------------
# Mock access grant
# ---------------------------------------------------------------------------


@dataclass
class MockAccessGrant:
    """Mock access grant with to_dict support."""

    item_id: str = "node-001"
    grantee_type: str = "user"
    grantee_id: str = "user-grantee"
    permissions: list[str] = field(default_factory=lambda: ["read"])
    granted_by: str = "admin-user"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "grantee_type": self.grantee_type,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "granted_by": self.granted_by,
        }


@dataclass
class MockAccessGrantNoDict:
    """Mock access grant without to_dict (fallback path)."""

    item_id: str = "node-002"
    grantee_id: str = "user-fallback"


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class VisibilityTestHandler(VisibilityOperationsMixin):
    """Concrete handler for testing the visibility mixin."""

    def __init__(self, mound=None, user=None):
        self._mound = mound
        self._user = user or MockUser()

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None


class VisibilityTestHandlerNoAuth(VisibilityOperationsMixin):
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
    """Create a mock KnowledgeMound with visibility methods."""
    mound = MagicMock()
    mound.set_visibility = AsyncMock(return_value=None)
    mound.get_node = AsyncMock(return_value=MockNode())
    mound.grant_access = AsyncMock(
        return_value=MockAccessGrant(
            item_id="node-001",
            grantee_type="user",
            grantee_id="user-grantee",
            permissions=["read"],
            granted_by="test-user-001",
        )
    )
    mound.revoke_access = AsyncMock(return_value=True)
    mound.get_access_grants = AsyncMock(
        return_value=[
            MockAccessGrant(item_id="node-001", grantee_id="user-a"),
            MockAccessGrant(item_id="node-001", grantee_id="user-b"),
        ]
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a VisibilityTestHandler with a mocked mound."""
    return VisibilityTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a VisibilityTestHandler with no mound (None)."""
    return VisibilityTestHandler(mound=None)


@pytest.fixture
def handler_no_auth(mock_mound):
    """Create a VisibilityTestHandler that fails auth."""
    return VisibilityTestHandlerNoAuth(mound=mock_mound)


# ============================================================================
# Tests: _handle_set_visibility (PUT /api/knowledge/mound/nodes/:id/visibility)
# ============================================================================


class TestSetVisibility:
    """Test _handle_set_visibility (PUT)."""

    def test_set_visibility_private_success(self, handler, mock_mound):
        """Successfully setting visibility to private returns 200."""
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["item_id"] == "node-001"
        assert body["visibility"] == "private"
        assert body["set_by"] == "test-user-001"

    def test_set_visibility_workspace_success(self, handler, mock_mound):
        """Setting visibility to workspace returns 200."""
        http = MockHTTPHandler.with_body({"visibility": "workspace"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["visibility"] == "workspace"

    def test_set_visibility_organization_success(self, handler, mock_mound):
        """Setting visibility to organization returns 200."""
        http = MockHTTPHandler.with_body({"visibility": "organization"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["visibility"] == "organization"

    def test_set_visibility_public_success(self, handler, mock_mound):
        """Setting visibility to public returns 200."""
        http = MockHTTPHandler.with_body({"visibility": "public"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["visibility"] == "public"

    def test_set_visibility_system_success(self, handler, mock_mound):
        """Setting visibility to system returns 200."""
        http = MockHTTPHandler.with_body({"visibility": "system"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["visibility"] == "system"

    def test_set_visibility_default_discoverable_true(self, handler, mock_mound):
        """Default is_discoverable is True when not specified."""
        http = MockHTTPHandler.with_body({"visibility": "workspace"})
        result = handler._handle_set_visibility("node-001", http)
        body = _body(result)
        assert body["is_discoverable"] is True

    def test_set_visibility_discoverable_false(self, handler, mock_mound):
        """is_discoverable=false is reflected in response."""
        http = MockHTTPHandler.with_body(
            {
                "visibility": "private",
                "is_discoverable": False,
            }
        )
        result = handler._handle_set_visibility("node-001", http)
        body = _body(result)
        assert body["is_discoverable"] is False

    def test_set_visibility_discoverable_true_explicit(self, handler, mock_mound):
        """Explicit is_discoverable=true is reflected in response."""
        http = MockHTTPHandler.with_body(
            {
                "visibility": "public",
                "is_discoverable": True,
            }
        )
        result = handler._handle_set_visibility("node-001", http)
        body = _body(result)
        assert body["is_discoverable"] is True

    def test_set_visibility_calls_mound_with_correct_args(self, handler, mock_mound):
        """set_visibility is called with correct keyword args."""
        http = MockHTTPHandler.with_body(
            {
                "visibility": "organization",
                "is_discoverable": False,
            }
        )
        handler._handle_set_visibility("node-xyz", http)
        call_args = mock_mound.set_visibility.call_args
        assert call_args.kwargs["item_id"] == "node-xyz"
        assert call_args.kwargs["visibility"] == "organization"
        assert call_args.kwargs["set_by"] == "test-user-001"
        assert call_args.kwargs["is_discoverable"] is False

    def test_set_visibility_missing_visibility_returns_400(self, handler):
        """Missing visibility field returns 400."""
        http = MockHTTPHandler.with_body({"is_discoverable": True})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "visibility" in body["error"].lower()

    def test_set_visibility_empty_visibility_returns_400(self, handler):
        """Empty string visibility returns 400."""
        http = MockHTTPHandler.with_body({"visibility": ""})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "visibility" in body["error"].lower()

    def test_set_visibility_invalid_level_returns_400(self, handler):
        """Invalid visibility level returns 400 with valid levels."""
        http = MockHTTPHandler.with_body({"visibility": "super_secret"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid visibility level" in body["error"].lower()
        assert "super_secret" in body["error"]
        # Valid levels should be listed
        assert "private" in body["error"]
        assert "workspace" in body["error"]

    def test_set_visibility_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower()

    def test_set_visibility_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower() or "body" in body["error"].lower()

    def test_set_visibility_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler_no_mound._handle_set_visibility("node-001", http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_set_visibility_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler_no_auth._handle_set_visibility("node-001", http)
        assert _status(result) == 401

    def test_set_visibility_value_error_returns_404(self, handler, mock_mound):
        """ValueError from set_visibility returns 404."""
        mock_mound.set_visibility = AsyncMock(side_effect=ValueError("Node not found"))
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("missing-node", http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_set_visibility_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.set_visibility = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 500

    def test_set_visibility_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.set_visibility = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 500

    def test_set_visibility_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.set_visibility = AsyncMock(side_effect=TypeError("wrong type"))
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 500

    def test_set_visibility_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.set_visibility = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 500

    def test_set_visibility_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.set_visibility = AsyncMock(side_effect=AttributeError("attr"))
        http = MockHTTPHandler.with_body({"visibility": "workspace"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_visibility_change")
    def test_set_visibility_tracks_metrics(self, mock_track, handler, mock_mound):
        """track_visibility_change is called with correct args."""
        http = MockHTTPHandler.with_body({"visibility": "public"})
        handler._handle_set_visibility("node-001", http)
        mock_track.assert_called_once_with(
            node_id="node-001",
            from_level="unknown",
            to_level="public",
            workspace_id="ws-default",
        )

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_visibility_change")
    def test_set_visibility_tracks_metrics_unknown_workspace(self, mock_track, mock_mound):
        """Workspace fallback to 'unknown' when user has no workspace_id."""
        user = MagicMock(spec=[])
        user.id = "user-no-ws"
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({"visibility": "private"})
        h._handle_set_visibility("node-001", http)
        mock_track.assert_called_once()
        assert mock_track.call_args.kwargs["workspace_id"] == "unknown"

    def test_set_visibility_user_id_from_user_id_attr(self, mock_mound):
        """Falls back to user_id attribute when id is not available."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-user"
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({"visibility": "workspace"})
        result = h._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["set_by"] == "fallback-user"

    def test_set_visibility_user_id_fallback_to_unknown(self, mock_mound):
        """User without id or user_id falls back to 'unknown'."""
        user = MagicMock(spec=[])
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({"visibility": "workspace"})
        result = h._handle_set_visibility("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["set_by"] == "unknown"

    def test_set_visibility_all_valid_levels(self, handler, mock_mound):
        """Every valid VisibilityLevel is accepted."""
        from aragora.knowledge.mound.types import VisibilityLevel

        for level in VisibilityLevel:
            mock_mound.set_visibility = AsyncMock(return_value=None)
            http = MockHTTPHandler.with_body({"visibility": level.value})
            result = handler._handle_set_visibility("node-001", http)
            assert _status(result) == 200, f"Level {level.value} should be valid"
            body = _body(result)
            assert body["visibility"] == level.value


# ============================================================================
# Tests: _handle_get_visibility (GET /api/knowledge/mound/nodes/:id/visibility)
# ============================================================================


class TestGetVisibility:
    """Test _handle_get_visibility (GET)."""

    def test_get_visibility_success(self, handler, mock_mound):
        """Successfully getting visibility returns node visibility data."""
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["item_id"] == "node-001"
        assert body["visibility"] == "workspace"
        assert body["visibility_set_by"] == "admin-user"
        assert body["is_discoverable"] is True

    def test_get_visibility_private_node(self, handler, mock_mound):
        """Getting visibility for a private node returns correct data."""
        mock_mound.get_node = AsyncMock(
            return_value=MockNode(
                id="node-priv",
                metadata={
                    "visibility": "private",
                    "visibility_set_by": "user-x",
                    "is_discoverable": False,
                },
            )
        )
        result = handler._handle_get_visibility("node-priv")
        body = _body(result)
        assert body["visibility"] == "private"
        assert body["visibility_set_by"] == "user-x"
        assert body["is_discoverable"] is False

    def test_get_visibility_default_values(self, handler, mock_mound):
        """Node with no visibility metadata returns defaults."""
        mock_mound.get_node = AsyncMock(return_value=MockNode(id="node-bare", metadata={}))
        result = handler._handle_get_visibility("node-bare")
        body = _body(result)
        assert body["visibility"] == "workspace"
        assert body["visibility_set_by"] is None
        assert body["is_discoverable"] is True

    def test_get_visibility_none_metadata(self, handler, mock_mound):
        """Node with None metadata returns defaults."""
        node = MockNode(id="node-none")
        node.metadata = None
        mock_mound.get_node = AsyncMock(return_value=node)
        result = handler._handle_get_visibility("node-none")
        body = _body(result)
        assert body["visibility"] == "workspace"
        assert body["visibility_set_by"] is None
        assert body["is_discoverable"] is True

    def test_get_visibility_node_not_found_returns_404(self, handler, mock_mound):
        """Node not found (None) returns 404."""
        mock_mound.get_node = AsyncMock(return_value=None)
        result = handler._handle_get_visibility("nonexistent")
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()
        assert "nonexistent" in body["error"]

    def test_get_visibility_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_visibility("node-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_visibility_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=RuntimeError("runtime"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_node = AsyncMock(side_effect=AttributeError("attr"))
        result = handler._handle_get_visibility("node-001")
        assert _status(result) == 500

    def test_get_visibility_item_id_matches_request(self, handler, mock_mound):
        """Response item_id matches the requested node_id."""
        result = handler._handle_get_visibility("my-special-node")
        body = _body(result)
        assert body["item_id"] == "my-special-node"

    def test_get_visibility_partial_metadata(self, handler, mock_mound):
        """Node with partial metadata uses defaults for missing fields."""
        mock_mound.get_node = AsyncMock(
            return_value=MockNode(
                id="node-partial",
                metadata={"visibility": "public"},
            )
        )
        result = handler._handle_get_visibility("node-partial")
        body = _body(result)
        assert body["visibility"] == "public"
        assert body["visibility_set_by"] is None
        assert body["is_discoverable"] is True


# ============================================================================
# Tests: _handle_grant_access (POST /api/knowledge/mound/nodes/:id/access)
# ============================================================================


class TestGrantAccess:
    """Test _handle_grant_access (POST)."""

    def test_grant_access_user_success(self, handler, mock_mound):
        """Successfully granting access to a user returns 201."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert "grant" in body

    def test_grant_access_workspace_success(self, handler, mock_mound):
        """Granting access to a workspace returns 201."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "workspace", "grantee_id": "ws-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201

    def test_grant_access_role_success(self, handler, mock_mound):
        """Granting access to a role returns 201."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "role", "grantee_id": "role-editor"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201

    def test_grant_access_organization_success(self, handler, mock_mound):
        """Granting access to an organization returns 201."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "organization", "grantee_id": "org-001"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201

    def test_grant_access_default_permissions(self, handler, mock_mound):
        """Default permissions are ['read'] when not specified."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        handler._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["permissions"] == ["read"]

    def test_grant_access_custom_permissions(self, handler, mock_mound):
        """Custom permissions are forwarded to mound."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "permissions": ["read", "write", "admin"],
            },
            method="POST",
        )
        handler._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["permissions"] == ["read", "write", "admin"]

    def test_grant_access_with_expires_at(self, handler, mock_mound):
        """expires_at is parsed and forwarded to mound."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "expires_at": "2025-12-31T23:59:59Z",
            },
            method="POST",
        )
        handler._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["expires_at"] is not None

    def test_grant_access_expires_at_with_offset(self, handler, mock_mound):
        """expires_at with timezone offset is parsed correctly."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "expires_at": "2025-06-15T12:00:00+05:00",
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["expires_at"] is not None

    def test_grant_access_no_expires_at(self, handler, mock_mound):
        """No expires_at means None is passed to mound."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        handler._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["expires_at"] is None

    def test_grant_access_invalid_expires_at_returns_400(self, handler):
        """Invalid expires_at format returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "expires_at": "not-a-date",
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "expires_at" in body["error"].lower() or "iso" in body["error"].lower()

    def test_grant_access_calls_mound_with_correct_args(self, handler, mock_mound):
        """grant_access is called with correct keyword args."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "workspace",
                "grantee_id": "ws-dest",
                "permissions": ["read", "write"],
            },
            method="POST",
        )
        handler._handle_grant_access("node-xyz", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["item_id"] == "node-xyz"
        assert call_kwargs["grantee_type"] == "workspace"
        assert call_kwargs["grantee_id"] == "ws-dest"
        assert call_kwargs["permissions"] == ["read", "write"]
        assert call_kwargs["granted_by"] == "test-user-001"

    def test_grant_access_missing_grantee_type_returns_400(self, handler):
        """Missing grantee_type returns 400."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "grantee_type" in body["error"].lower()

    def test_grant_access_missing_grantee_id_returns_400(self, handler):
        """Missing grantee_id returns 400."""
        http = MockHTTPHandler.with_body({"grantee_type": "user"}, method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "grantee_id" in body["error"].lower()

    def test_grant_access_missing_both_returns_400(self, handler):
        """Missing both grantee_type and grantee_id returns 400."""
        http = MockHTTPHandler.with_body({}, method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_empty_grantee_type_returns_400(self, handler):
        """Empty grantee_type returns 400."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "", "grantee_id": "user-target"}, method="POST"
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_empty_grantee_id_returns_400(self, handler):
        """Empty grantee_id returns 400."""
        http = MockHTTPHandler.with_body({"grantee_type": "user", "grantee_id": ""}, method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_invalid_grantee_type_returns_400(self, handler):
        """Invalid grantee_type returns 400 with valid types."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "invalid_type", "grantee_id": "target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid grantee_type" in body["error"].lower()
        assert "invalid_type" in body["error"]
        # Valid types should be listed
        assert "user" in body["error"]

    def test_grant_access_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty(method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower()

    def test_grant_access_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json(method="POST")
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler_no_mound._handle_grant_access("node-001", http)
        assert _status(result) == 503

    def test_grant_access_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler_no_auth._handle_grant_access("node-001", http)
        assert _status(result) == 401

    def test_grant_access_value_error_returns_404(self, handler, mock_mound):
        """ValueError from grant_access returns 404."""
        mock_mound.grant_access = AsyncMock(side_effect=ValueError("Node not found"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("missing-node", http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_grant_access_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.grant_access = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 500

    def test_grant_access_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.grant_access = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 500

    def test_grant_access_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.grant_access = AsyncMock(side_effect=TypeError("wrong"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 500

    def test_grant_access_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.grant_access = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 500

    def test_grant_access_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.grant_access = AsyncMock(side_effect=AttributeError("attr"))
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 500

    def test_grant_access_grant_with_to_dict(self, handler, mock_mound):
        """Grant with to_dict is serialized using to_dict."""
        mock_mound.grant_access = AsyncMock(
            return_value=MockAccessGrant(
                item_id="node-001",
                grantee_type="user",
                grantee_id="user-target",
                permissions=["read", "write"],
                granted_by="admin",
            )
        )
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        body = _body(result)
        assert body["grant"]["item_id"] == "node-001"
        assert body["grant"]["grantee_type"] == "user"
        assert body["grant"]["grantee_id"] == "user-target"

    def test_grant_access_grant_without_to_dict(self, handler, mock_mound):
        """Grant without to_dict uses fallback serialization."""
        mock_mound.grant_access = AsyncMock(return_value="non-dict-result")
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "permissions": ["read"],
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        body = _body(result)
        # Fallback serialization uses request data
        assert body["grant"]["item_id"] == "node-001"
        assert body["grant"]["grantee_type"] == "user"
        assert body["grant"]["grantee_id"] == "user-target"
        assert body["grant"]["permissions"] == ["read"]
        assert body["grant"]["granted_by"] == "test-user-001"

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_grant_access_tracks_metrics(self, mock_track, handler, mock_mound):
        """track_access_grant is called with correct args."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "workspace", "grantee_id": "ws-target"},
            method="POST",
        )
        handler._handle_grant_access("node-001", http)
        mock_track.assert_called_once_with(
            action="grant",
            grantee_type="workspace",
            workspace_id="ws-default",
        )

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_grant_access_metrics_unknown_workspace(self, mock_track, mock_mound):
        """Workspace fallback to 'unknown' when user has no workspace_id."""
        user = MagicMock(spec=[])
        user.id = "user-no-ws"
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        h._handle_grant_access("node-001", http)
        assert mock_track.call_args.kwargs["workspace_id"] == "unknown"

    def test_grant_access_user_id_fallback_to_user_id(self, mock_mound):
        """Falls back to user_id attribute when id is not available."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-user"
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        h._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["granted_by"] == "fallback-user"

    def test_grant_access_user_id_fallback_to_unknown(self, mock_mound):
        """User without id or user_id falls back to 'unknown'."""
        user = MagicMock(spec=[])
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        h._handle_grant_access("node-001", http)
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["granted_by"] == "unknown"

    def test_grant_access_all_valid_grantee_types(self, handler, mock_mound):
        """Every valid AccessGrantType is accepted."""
        from aragora.knowledge.mound.types import AccessGrantType

        for grant_type in AccessGrantType:
            mock_mound.grant_access = AsyncMock(
                return_value=MockAccessGrant(grantee_type=grant_type.value)
            )
            http = MockHTTPHandler.with_body(
                {"grantee_type": grant_type.value, "grantee_id": "target"},
                method="POST",
            )
            result = handler._handle_grant_access("node-001", http)
            assert _status(result) == 201, f"Type {grant_type.value} should be valid"


# ============================================================================
# Tests: _handle_revoke_access (DELETE /api/knowledge/mound/nodes/:id/access)
# ============================================================================


class TestRevokeAccess:
    """Test _handle_revoke_access (DELETE)."""

    def test_revoke_access_success(self, handler, mock_mound):
        """Successfully revoking access returns 200."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["item_id"] == "node-001"
        assert body["grantee_id"] == "user-target"
        assert body["revoked_by"] == "test-user-001"

    def test_revoke_access_calls_mound_with_correct_args(self, handler, mock_mound):
        """revoke_access is called with correct keyword args."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-xyz"}, method="DELETE")
        handler._handle_revoke_access("node-abc", http)
        call_kwargs = mock_mound.revoke_access.call_args.kwargs
        assert call_kwargs["item_id"] == "node-abc"
        assert call_kwargs["grantee_id"] == "user-xyz"
        assert call_kwargs["revoked_by"] == "test-user-001"

    def test_revoke_access_missing_grantee_id_returns_400(self, handler):
        """Missing grantee_id returns 400."""
        http = MockHTTPHandler.with_body({}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "grantee_id" in body["error"].lower()

    def test_revoke_access_empty_grantee_id_returns_400(self, handler):
        """Empty grantee_id returns 400."""
        http = MockHTTPHandler.with_body({"grantee_id": ""}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 400

    def test_revoke_access_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty(method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 400

    def test_revoke_access_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json(method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 400

    def test_revoke_access_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler_no_mound._handle_revoke_access("node-001", http)
        assert _status(result) == 503

    def test_revoke_access_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler_no_auth._handle_revoke_access("node-001", http)
        assert _status(result) == 401

    def test_revoke_access_value_error_returns_404(self, handler, mock_mound):
        """ValueError from revoke_access returns 404."""
        mock_mound.revoke_access = AsyncMock(side_effect=ValueError("Grant not found"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("missing", http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_revoke_access_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.revoke_access = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 500

    def test_revoke_access_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.revoke_access = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 500

    def test_revoke_access_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.revoke_access = AsyncMock(side_effect=TypeError("wrong"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 500

    def test_revoke_access_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.revoke_access = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 500

    def test_revoke_access_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.revoke_access = AsyncMock(side_effect=AttributeError("attr"))
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_revoke_access_tracks_metrics(self, mock_track, handler, mock_mound):
        """track_access_grant is called with action='revoke'."""
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        handler._handle_revoke_access("node-001", http)
        mock_track.assert_called_once_with(
            action="revoke",
            grantee_type="unknown",
            workspace_id="ws-default",
        )

    def test_revoke_access_user_id_fallback_to_user_id(self, mock_mound):
        """Falls back to user_id attribute when id is not available."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-user"
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = h._handle_revoke_access("node-001", http)
        body = _body(result)
        assert body["revoked_by"] == "fallback-user"

    def test_revoke_access_user_id_fallback_to_unknown(self, mock_mound):
        """User without id or user_id falls back to 'unknown'."""
        user = MagicMock(spec=[])
        h = VisibilityTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        result = h._handle_revoke_access("node-001", http)
        body = _body(result)
        assert body["revoked_by"] == "unknown"


# ============================================================================
# Tests: _handle_list_access_grants (GET /api/knowledge/mound/nodes/:id/access)
# ============================================================================


class TestListAccessGrants:
    """Test _handle_list_access_grants (GET)."""

    def test_list_grants_success(self, handler, mock_mound):
        """Successfully listing grants returns grants with count."""
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["item_id"] == "node-001"
        assert body["count"] == 2
        assert len(body["grants"]) == 2

    def test_list_grants_with_to_dict(self, handler, mock_mound):
        """Grants with to_dict are serialized using to_dict."""
        result = handler._handle_list_access_grants("node-001", {})
        body = _body(result)
        assert body["grants"][0]["grantee_id"] == "user-a"
        assert body["grants"][1]["grantee_id"] == "user-b"

    def test_list_grants_without_to_dict(self, handler, mock_mound):
        """Grants without to_dict are passed through as-is."""
        mock_mound.get_access_grants = AsyncMock(
            return_value=[
                {"item_id": "node-001", "grantee_id": "raw-grantee"},
            ]
        )
        result = handler._handle_list_access_grants("node-001", {})
        body = _body(result)
        assert body["grants"][0]["grantee_id"] == "raw-grantee"

    def test_list_grants_mixed_to_dict(self, handler, mock_mound):
        """Mixed grants (with and without to_dict) are handled correctly."""
        mock_mound.get_access_grants = AsyncMock(
            return_value=[
                MockAccessGrant(grantee_id="with-dict"),
                {"item_id": "node-001", "grantee_id": "without-dict"},
            ]
        )
        result = handler._handle_list_access_grants("node-001", {})
        body = _body(result)
        assert body["count"] == 2
        assert body["grants"][0]["grantee_id"] == "with-dict"
        assert body["grants"][1]["grantee_id"] == "without-dict"

    def test_list_grants_empty(self, handler, mock_mound):
        """Empty grants return count=0 and empty list."""
        mock_mound.get_access_grants = AsyncMock(return_value=[])
        result = handler._handle_list_access_grants("node-001", {})
        body = _body(result)
        assert body["count"] == 0
        assert body["grants"] == []

    def test_list_grants_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_list_access_grants("node-001", {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_list_grants_value_error_returns_404(self, handler, mock_mound):
        """ValueError from get_access_grants returns 404."""
        mock_mound.get_access_grants = AsyncMock(side_effect=ValueError("Node not found"))
        result = handler._handle_list_access_grants("missing", {})
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_list_grants_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_access_grants = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 500

    def test_list_grants_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_access_grants = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 500

    def test_list_grants_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_access_grants = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 500

    def test_list_grants_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_access_grants = AsyncMock(side_effect=RuntimeError("rt"))
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 500

    def test_list_grants_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_access_grants = AsyncMock(side_effect=AttributeError("attr"))
        result = handler._handle_list_access_grants("node-001", {})
        assert _status(result) == 500

    def test_list_grants_item_id_matches_request(self, handler, mock_mound):
        """Response item_id matches the requested node_id."""
        result = handler._handle_list_access_grants("my-node-id", {})
        body = _body(result)
        assert body["item_id"] == "my-node-id"

    def test_list_grants_count_matches_length(self, handler, mock_mound):
        """Count matches the number of grants returned."""
        mock_mound.get_access_grants = AsyncMock(
            return_value=[
                MockAccessGrant(grantee_id="a"),
                MockAccessGrant(grantee_id="b"),
                MockAccessGrant(grantee_id="c"),
            ]
        )
        result = handler._handle_list_access_grants("node-001", {})
        body = _body(result)
        assert body["count"] == len(body["grants"])
        assert body["count"] == 3


# ============================================================================
# Tests: routing integration
# ============================================================================


class TestVisibilityRouting:
    """Test routing dispatch for visibility/access endpoints."""

    def test_route_node_visibility_put_dispatches_to_set(self, handler, mock_mound):
        """PUT /nodes/:id/visibility dispatches to _handle_set_visibility."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body({"visibility": "private"})
        http.command = "PUT"
        result = RoutingMixin._route_node_visibility(handler, "node-001", {}, http)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["visibility"] == "private"

    def test_route_node_visibility_get_dispatches_to_get(self, handler, mock_mound):
        """GET /nodes/:id/visibility dispatches to _handle_get_visibility."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler()
        http.command = "GET"
        result = RoutingMixin._route_node_visibility(handler, "node-001", {}, http)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["item_id"] == "node-001"
        assert "visibility" in body

    def test_route_node_access_post_dispatches_to_grant(self, handler, mock_mound):
        """POST /nodes/:id/access dispatches to _handle_grant_access."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-target"},
            method="POST",
        )
        http.command = "POST"
        result = RoutingMixin._route_node_access(handler, "node-001", {}, http)
        assert result is not None
        assert _status(result) == 201

    def test_route_node_access_delete_dispatches_to_revoke(self, handler, mock_mound):
        """DELETE /nodes/:id/access dispatches to _handle_revoke_access."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler.with_body({"grantee_id": "user-target"}, method="DELETE")
        http.command = "DELETE"
        result = RoutingMixin._route_node_access(handler, "node-001", {}, http)
        assert result is not None
        assert _status(result) == 200

    def test_route_node_access_get_dispatches_to_list(self, handler, mock_mound):
        """GET /nodes/:id/access dispatches to _handle_list_access_grants."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        http = MockHTTPHandler()
        http.command = "GET"
        result = RoutingMixin._route_node_access(handler, "node-001", {}, http)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "grants" in body
        assert "count" in body


# ============================================================================
# Tests: security edge cases
# ============================================================================


class TestVisibilitySecurity:
    """Test security edge cases for visibility operations."""

    def test_set_visibility_path_traversal_node_id(self, handler, mock_mound):
        """Path traversal in node_id is passed through (handler trusts routing)."""
        http = MockHTTPHandler.with_body({"visibility": "private"})
        result = handler._handle_set_visibility("../../etc/passwd", http)
        # The handler doesn't validate node_id format; that's routing's job
        assert _status(result) == 200
        call_kwargs = mock_mound.set_visibility.call_args.kwargs
        assert call_kwargs["item_id"] == "../../etc/passwd"

    def test_grant_access_html_injection_grantee_id(self, handler, mock_mound):
        """HTML injection in grantee_id is passed through (no XSS in API)."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "<script>alert('xss')</script>",
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["grantee_id"] == "<script>alert('xss')</script>"

    def test_revoke_access_sql_injection_grantee_id(self, handler, mock_mound):
        """SQL injection in grantee_id is passed through (parameterized queries downstream)."""
        http = MockHTTPHandler.with_body(
            {"grantee_id": "'; DROP TABLE nodes; --"},
            method="DELETE",
        )
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 200
        call_kwargs = mock_mound.revoke_access.call_args.kwargs
        assert call_kwargs["grantee_id"] == "'; DROP TABLE nodes; --"

    def test_set_visibility_large_body_handled(self, handler, mock_mound):
        """Large body with extra fields is handled gracefully."""
        body_data = {"visibility": "public"}
        # Add extra fields
        for i in range(100):
            body_data[f"extra_field_{i}"] = f"value_{i}"
        http = MockHTTPHandler.with_body(body_data)
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200

    def test_grant_access_unicode_grantee_id(self, handler, mock_mound):
        """Unicode characters in grantee_id are handled."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-\u00e9\u00e8\u00ea"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201

    def test_get_visibility_special_chars_node_id(self, handler, mock_mound):
        """Special characters in node_id are handled."""
        result = handler._handle_get_visibility("node-with-special/chars?query=1")
        assert _status(result) == 200
        body = _body(result)
        assert body["item_id"] == "node-with-special/chars?query=1"

    def test_list_grants_very_long_node_id(self, handler, mock_mound):
        """Very long node_id is passed through."""
        long_id = "x" * 10000
        result = handler._handle_list_access_grants(long_id, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["item_id"] == long_id


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestVisibilityEdgeCases:
    """Test edge cases across visibility operations."""

    def test_set_visibility_null_visibility_returns_400(self, handler):
        """Null visibility value returns 400."""
        http = MockHTTPHandler.with_body({"visibility": None})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400

    def test_set_visibility_numeric_visibility_returns_400(self, handler):
        """Numeric visibility value returns 400."""
        http = MockHTTPHandler.with_body({"visibility": 42})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400

    def test_set_visibility_case_sensitive(self, handler):
        """Visibility level is case-sensitive ('PRIVATE' is invalid)."""
        http = MockHTTPHandler.with_body({"visibility": "PRIVATE"})
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid visibility level" in body["error"].lower()

    def test_grant_access_null_grantee_type_returns_400(self, handler):
        """Null grantee_type returns 400."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": None, "grantee_id": "user-target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_null_grantee_id_returns_400(self, handler):
        """Null grantee_id returns 400."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": None},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_grant_access_case_sensitive_grantee_type(self, handler):
        """grantee_type is case-sensitive ('USER' is invalid)."""
        http = MockHTTPHandler.with_body(
            {"grantee_type": "USER", "grantee_id": "target"},
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 400

    def test_revoke_access_null_grantee_id_returns_400(self, handler):
        """Null grantee_id returns 400."""
        http = MockHTTPHandler.with_body({"grantee_id": None}, method="DELETE")
        result = handler._handle_revoke_access("node-001", http)
        assert _status(result) == 400

    def test_get_visibility_mound_returns_node_with_empty_metadata(self, handler, mock_mound):
        """Node with empty dict metadata uses defaults."""
        mock_mound.get_node = AsyncMock(return_value=MockNode(id="node-empty-meta", metadata={}))
        result = handler._handle_get_visibility("node-empty-meta")
        body = _body(result)
        assert body["visibility"] == "workspace"

    def test_set_visibility_with_extra_fields_ignored(self, handler, mock_mound):
        """Extra fields in body are ignored."""
        http = MockHTTPHandler.with_body(
            {
                "visibility": "private",
                "extra_field": "ignored",
                "another": 123,
            }
        )
        result = handler._handle_set_visibility("node-001", http)
        assert _status(result) == 200

    def test_grant_access_empty_permissions_list(self, handler, mock_mound):
        """Empty permissions list is valid (uses default)."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "user",
                "grantee_id": "user-target",
                "permissions": [],
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        # Empty list is passed through to mound
        assert _status(result) == 201

    def test_multiple_operations_sequential(self, handler, mock_mound):
        """Multiple sequential operations work correctly."""
        # Set visibility
        http1 = MockHTTPHandler.with_body({"visibility": "public"})
        result1 = handler._handle_set_visibility("node-001", http1)
        assert _status(result1) == 200

        # Grant access
        http2 = MockHTTPHandler.with_body(
            {"grantee_type": "user", "grantee_id": "user-a"},
            method="POST",
        )
        result2 = handler._handle_grant_access("node-001", http2)
        assert _status(result2) == 201

        # List grants
        result3 = handler._handle_list_access_grants("node-001", {})
        assert _status(result3) == 200

        # Get visibility
        result4 = handler._handle_get_visibility("node-001")
        assert _status(result4) == 200

        # Revoke access
        http5 = MockHTTPHandler.with_body({"grantee_id": "user-a"}, method="DELETE")
        result5 = handler._handle_revoke_access("node-001", http5)
        assert _status(result5) == 200

    def test_set_visibility_content_length_mismatch(self, handler):
        """Content-Length header mismatch with actual body length is handled."""
        raw = json.dumps({"visibility": "private"}).encode("utf-8")
        http = MockHTTPHandler(
            command="PUT",
            headers={"Content-Length": str(len(raw) + 100)},
            rfile=io.BytesIO(raw),
        )
        # rfile.read() will return what's available regardless of Content-Length
        result = handler._handle_set_visibility("node-001", http)
        # Should still work since read returns available bytes
        assert _status(result) == 200

    def test_grant_access_with_all_fields(self, handler, mock_mound):
        """Grant access with all optional fields set."""
        http = MockHTTPHandler.with_body(
            {
                "grantee_type": "organization",
                "grantee_id": "org-001",
                "permissions": ["read", "write", "admin"],
                "expires_at": "2025-12-31T23:59:59Z",
            },
            method="POST",
        )
        result = handler._handle_grant_access("node-001", http)
        assert _status(result) == 201
        call_kwargs = mock_mound.grant_access.call_args.kwargs
        assert call_kwargs["grantee_type"] == "organization"
        assert call_kwargs["grantee_id"] == "org-001"
        assert call_kwargs["permissions"] == ["read", "write", "admin"]
        assert call_kwargs["expires_at"] is not None
        assert call_kwargs["granted_by"] == "test-user-001"
