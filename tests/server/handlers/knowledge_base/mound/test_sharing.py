"""Tests for SharingOperationsMixin."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.sharing import (
    SharingOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    user_id: str = "user-123"
    workspace_id: str = "ws-123"


@dataclass
class MockShareGrant:
    """Mock share grant."""

    item_id: str
    grantee_id: str
    permissions: List[str]
    shared_by: str
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "shared_by": self.shared_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class MockSharedItem:
    """Mock shared item."""

    id: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "content": self.content}


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    share_with_workspace: AsyncMock = field(default_factory=AsyncMock)
    share_with_user: AsyncMock = field(default_factory=AsyncMock)
    get_shared_with_me: AsyncMock = field(default_factory=AsyncMock)
    revoke_share: AsyncMock = field(default_factory=AsyncMock)
    get_share_grants: AsyncMock = field(default_factory=AsyncMock)
    update_share_permissions: AsyncMock = field(default_factory=AsyncMock)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: Optional[Dict[str, str]] = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class SharingHandler(SharingOperationsMixin):
    """Handler implementation for testing SharingOperationsMixin."""

    def __init__(self, mound: Optional[MockKnowledgeMound] = None, user: Optional[MockUser] = None):
        self._mound = mound
        self._user = user or MockUser()
        self.ctx = {}

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return MockUser()


@pytest.fixture
def handler(mock_mound, mock_user):
    """Create a test handler with mock mound."""
    return SharingHandler(mound=mock_mound, user=mock_user)


@pytest.fixture
def handler_no_mound(mock_user):
    """Create a test handler without mound."""
    return SharingHandler(mound=None, user=mock_user)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test share_item
# =============================================================================


class TestShareItem:
    """Tests for share item endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share")
    def test_share_with_workspace_success(self, mock_track, handler, mock_mound):
        """Test successful share with workspace."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
                "permissions": ["read", "write"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["success"] is True
        assert data["share"]["item_id"] == "item-123"
        assert data["share"]["target_type"] == "workspace"

    @patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share")
    def test_share_with_user_success(self, mock_track, handler, mock_mound):
        """Test successful share with user."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "user",
                "target_id": "user-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["share"]["target_type"] == "user"

    @patch("aragora.server.handlers.knowledge_base.mound.sharing.track_share")
    def test_share_with_expiration(self, mock_track, handler, mock_mound):
        """Test share with expiration date."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
                "expires_at": "2026-12-31T23:59:59Z",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["share"]["expires_at"] is not None

    def test_share_missing_item_id(self, handler):
        """Test share with missing item_id."""
        body = json.dumps(
            {
                "target_type": "workspace",
                "target_id": "ws-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_share_invalid_target_type(self, handler):
        """Test share with invalid target type."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "invalid",
                "target_id": "id-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 400
        assert "target_type" in parse_response(result)["error"]

    def test_share_invalid_expires_at(self, handler):
        """Test share with invalid expiration format."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
                "expires_at": "not-a-date",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_share_item(http_handler)

        assert result.status_code == 400
        assert "expires_at" in parse_response(result)["error"]

    def test_share_no_mound(self, handler_no_mound):
        """Test share when mound not available."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "target_type": "workspace",
                "target_id": "ws-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_share_item(http_handler)

        assert result.status_code == 503


# =============================================================================
# Test shared_with_me
# =============================================================================


class TestSharedWithMe:
    """Tests for shared with me endpoint."""

    def test_get_shared_items_success(self, handler, mock_mound):
        """Test successful retrieval of shared items."""
        items = [
            MockSharedItem(id="item-1", content="Content 1"),
            MockSharedItem(id="item-2", content="Content 2"),
        ]
        mock_mound.get_shared_with_me.return_value = items

        http_handler = MockHandler()
        result = handler._handle_shared_with_me({}, http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 2

    def test_get_shared_with_pagination(self, handler, mock_mound):
        """Test shared items with pagination."""
        items = [MockSharedItem(id=f"item-{i}", content=f"Content {i}") for i in range(10)]
        mock_mound.get_shared_with_me.return_value = items

        http_handler = MockHandler()
        result = handler._handle_shared_with_me({"limit": ["5"], "offset": ["2"]}, http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["limit"] == 5
        assert data["offset"] == 2

    def test_get_shared_no_mound(self, handler_no_mound):
        """Test shared items when mound not available."""
        http_handler = MockHandler()
        result = handler_no_mound._handle_shared_with_me({}, http_handler)

        assert result.status_code == 503


# =============================================================================
# Test revoke_share
# =============================================================================


class TestRevokeShare:
    """Tests for revoke share endpoint."""

    def test_revoke_success(self, handler, mock_mound):
        """Test successful share revocation."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "grantee-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_share(http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["success"] is True
        assert data["item_id"] == "item-123"

    def test_revoke_missing_params(self, handler):
        """Test revocation with missing parameters."""
        body = json.dumps({"item_id": "item-123"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_share(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_revoke_not_found(self, handler, mock_mound):
        """Test revocation when share not found."""
        mock_mound.revoke_share.side_effect = ValueError("Share not found")

        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "nonexistent",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_share(http_handler)

        assert result.status_code == 404

    def test_revoke_no_mound(self, handler_no_mound):
        """Test revocation when mound not available."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "grantee-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_revoke_share(http_handler)

        assert result.status_code == 503


# =============================================================================
# Test my_shares
# =============================================================================


class TestMyShares:
    """Tests for my shares endpoint."""

    def test_list_shares_success(self, handler, mock_mound):
        """Test successful listing of shares."""
        grants = [
            MockShareGrant(
                item_id="item-1",
                grantee_id="grantee-1",
                permissions=["read"],
                shared_by="user-123",
            ),
        ]
        mock_mound.get_share_grants.return_value = grants

        http_handler = MockHandler()
        result = handler._handle_my_shares({}, http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 1

    def test_list_shares_no_mound(self, handler_no_mound):
        """Test listing shares when mound not available."""
        http_handler = MockHandler()
        result = handler_no_mound._handle_my_shares({}, http_handler)

        assert result.status_code == 503


# =============================================================================
# Test update_share
# =============================================================================


class TestUpdateShare:
    """Tests for update share endpoint."""

    def test_update_permissions_success(self, handler, mock_mound):
        """Test successful permission update."""
        grant = MockShareGrant(
            item_id="item-123",
            grantee_id="grantee-456",
            permissions=["read", "write"],
            shared_by="user-123",
        )
        mock_mound.update_share_permissions.return_value = grant

        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "grantee-456",
                "permissions": ["read", "write"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_update_share(http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["success"] is True

    def test_update_missing_ids(self, handler):
        """Test update with missing identifiers."""
        body = json.dumps({"permissions": ["read"]}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_update_share(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_update_missing_changes(self, handler):
        """Test update with no changes specified."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "grantee-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_update_share(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_update_not_found(self, handler, mock_mound):
        """Test update when share not found."""
        mock_mound.update_share_permissions.side_effect = ValueError("Not found")

        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "nonexistent",
                "permissions": ["read"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_update_share(http_handler)

        assert result.status_code == 404

    def test_update_no_mound(self, handler_no_mound):
        """Test update when mound not available."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "grantee_id": "grantee-456",
                "permissions": ["read"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_update_share(http_handler)

        assert result.status_code == 503
