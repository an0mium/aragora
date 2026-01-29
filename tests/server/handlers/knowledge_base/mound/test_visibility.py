"""Tests for VisibilityOperationsMixin."""

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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.visibility import (
    VisibilityOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class VisibilityLevel(str, Enum):
    """Mock visibility level enum."""

    PRIVATE = "private"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class AccessGrantType(str, Enum):
    """Mock access grant type enum."""

    USER = "user"
    WORKSPACE = "workspace"
    ROLE = "role"


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    user_id: str = "user-123"
    workspace_id: str = "ws-123"


@dataclass
class MockNode:
    """Mock knowledge node."""

    id: str
    content: str
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "metadata": self.metadata}


@dataclass
class MockAccessGrant:
    """Mock access grant."""

    item_id: str
    grantee_type: str
    grantee_id: str
    permissions: list[str]
    granted_by: str
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "grantee_type": self.grantee_type,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "granted_by": self.granted_by,
        }


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    set_visibility: AsyncMock = field(default_factory=AsyncMock)
    get_node: AsyncMock = field(default_factory=AsyncMock)
    grant_access: AsyncMock = field(default_factory=AsyncMock)
    revoke_access: AsyncMock = field(default_factory=AsyncMock)
    get_access_grants: AsyncMock = field(default_factory=AsyncMock)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: Optional[dict[str, str]] = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class VisibilityHandler(VisibilityOperationsMixin):
    """Handler implementation for testing VisibilityOperationsMixin."""

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
    return VisibilityHandler(mound=mock_mound, user=mock_user)


@pytest.fixture
def handler_no_mound(mock_user):
    """Create a test handler without mound."""
    return VisibilityHandler(mound=None, user=mock_user)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test set_visibility
# =============================================================================


class TestSetVisibility:
    """Tests for set visibility endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_visibility_change")
    def test_set_visibility_success(self, mock_track, handler, mock_mound):
        """Test successful visibility change."""
        body = json.dumps(
            {
                "visibility": "workspace",
                "is_discoverable": True,
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_set_visibility("node-123", http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["success"] is True
        assert data["visibility"] == "workspace"
        assert data["is_discoverable"] is True

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_visibility_change")
    def test_set_visibility_to_private(self, mock_track, handler, mock_mound):
        """Test setting visibility to private."""
        body = json.dumps({"visibility": "private"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_set_visibility("node-123", http_handler)

        assert result.status_code == 200
        assert parse_response(result)["visibility"] == "private"

    def test_set_visibility_missing(self, handler):
        """Test set visibility with missing visibility field."""
        body = json.dumps({}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_set_visibility("node-123", http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_set_visibility_invalid(self, handler):
        """Test set visibility with invalid level."""
        body = json.dumps({"visibility": "invalid"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_set_visibility("node-123", http_handler)

        assert result.status_code == 400
        assert "Invalid visibility" in parse_response(result)["error"]

    def test_set_visibility_not_found(self, handler, mock_mound):
        """Test set visibility when node not found."""
        mock_mound.set_visibility.side_effect = ValueError("Node not found")

        body = json.dumps({"visibility": "workspace"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_set_visibility("nonexistent", http_handler)

        assert result.status_code == 404

    def test_set_visibility_no_mound(self, handler_no_mound):
        """Test set visibility when mound not available."""
        body = json.dumps({"visibility": "workspace"}).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_set_visibility("node-123", http_handler)

        assert result.status_code == 503


# =============================================================================
# Test get_visibility
# =============================================================================


class TestGetVisibility:
    """Tests for get visibility endpoint."""

    def test_get_visibility_success(self, handler, mock_mound):
        """Test successful visibility retrieval."""
        node = MockNode(
            id="node-123",
            content="Test content",
            metadata={
                "visibility": "workspace",
                "visibility_set_by": "user-123",
                "is_discoverable": True,
            },
        )
        mock_mound.get_node.return_value = node

        result = handler._handle_get_visibility("node-123")

        assert result.status_code == 200
        data = parse_response(result)
        assert data["item_id"] == "node-123"
        assert data["visibility"] == "workspace"
        assert data["is_discoverable"] is True

    def test_get_visibility_not_found(self, handler, mock_mound):
        """Test get visibility when node not found."""
        mock_mound.get_node.return_value = None

        result = handler._handle_get_visibility("nonexistent")

        assert result.status_code == 404

    def test_get_visibility_default_values(self, handler, mock_mound):
        """Test get visibility with default metadata values."""
        node = MockNode(id="node-123", content="Test", metadata={})
        mock_mound.get_node.return_value = node

        result = handler._handle_get_visibility("node-123")

        assert result.status_code == 200
        data = parse_response(result)
        assert data["visibility"] == "workspace"  # Default
        assert data["is_discoverable"] is True  # Default

    def test_get_visibility_no_mound(self, handler_no_mound):
        """Test get visibility when mound not available."""
        result = handler_no_mound._handle_get_visibility("node-123")

        assert result.status_code == 503


# =============================================================================
# Test grant_access
# =============================================================================


class TestGrantAccess:
    """Tests for grant access endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_grant_access_success(self, mock_track, handler, mock_mound):
        """Test successful access grant."""
        grant = MockAccessGrant(
            item_id="node-123",
            grantee_type="user",
            grantee_id="user-456",
            permissions=["read"],
            granted_by="user-123",
        )
        mock_mound.grant_access.return_value = grant

        body = json.dumps(
            {
                "grantee_type": "user",
                "grantee_id": "user-456",
                "permissions": ["read"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_grant_access("node-123", http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["success"] is True

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_grant_access_to_workspace(self, mock_track, handler, mock_mound):
        """Test granting access to workspace."""
        grant = MockAccessGrant(
            item_id="node-123",
            grantee_type="workspace",
            grantee_id="ws-456",
            permissions=["read", "write"],
            granted_by="user-123",
        )
        mock_mound.grant_access.return_value = grant

        body = json.dumps(
            {
                "grantee_type": "workspace",
                "grantee_id": "ws-456",
                "permissions": ["read", "write"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_grant_access("node-123", http_handler)

        assert result.status_code == 201

    def test_grant_access_missing_params(self, handler):
        """Test grant access with missing parameters."""
        body = json.dumps({"grantee_type": "user"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_grant_access("node-123", http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_grant_access_invalid_type(self, handler):
        """Test grant access with invalid grantee type."""
        body = json.dumps(
            {
                "grantee_type": "invalid",
                "grantee_id": "id-123",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_grant_access("node-123", http_handler)

        assert result.status_code == 400
        assert "Invalid grantee_type" in parse_response(result)["error"]

    def test_grant_access_not_found(self, handler, mock_mound):
        """Test grant access when node not found."""
        mock_mound.grant_access.side_effect = ValueError("Node not found")

        body = json.dumps(
            {
                "grantee_type": "user",
                "grantee_id": "user-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_grant_access("nonexistent", http_handler)

        assert result.status_code == 404

    def test_grant_access_no_mound(self, handler_no_mound):
        """Test grant access when mound not available."""
        body = json.dumps(
            {
                "grantee_type": "user",
                "grantee_id": "user-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_grant_access("node-123", http_handler)

        assert result.status_code == 503


# =============================================================================
# Test revoke_access
# =============================================================================


class TestRevokeAccess:
    """Tests for revoke access endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.visibility.track_access_grant")
    def test_revoke_access_success(self, mock_track, handler, mock_mound):
        """Test successful access revocation."""
        body = json.dumps({"grantee_id": "user-456"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_access("node-123", http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["success"] is True
        assert data["item_id"] == "node-123"
        assert data["grantee_id"] == "user-456"

    def test_revoke_access_missing_grantee(self, handler):
        """Test revoke access with missing grantee_id."""
        body = json.dumps({}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_access("node-123", http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_revoke_access_not_found(self, handler, mock_mound):
        """Test revoke access when grant not found."""
        mock_mound.revoke_access.side_effect = ValueError("Grant not found")

        body = json.dumps({"grantee_id": "nonexistent"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_revoke_access("node-123", http_handler)

        assert result.status_code == 404

    def test_revoke_access_no_mound(self, handler_no_mound):
        """Test revoke access when mound not available."""
        body = json.dumps({"grantee_id": "user-456"}).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_revoke_access("node-123", http_handler)

        assert result.status_code == 503


# =============================================================================
# Test list_access_grants
# =============================================================================


class TestListAccessGrants:
    """Tests for list access grants endpoint."""

    def test_list_grants_success(self, handler, mock_mound):
        """Test successful listing of access grants."""
        grants = [
            MockAccessGrant(
                item_id="node-123",
                grantee_type="user",
                grantee_id="user-456",
                permissions=["read"],
                granted_by="user-123",
            ),
        ]
        mock_mound.get_access_grants.return_value = grants

        result = handler._handle_list_access_grants("node-123", {})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["item_id"] == "node-123"
        assert data["count"] == 1

    def test_list_grants_empty(self, handler, mock_mound):
        """Test listing grants when none exist."""
        mock_mound.get_access_grants.return_value = []

        result = handler._handle_list_access_grants("node-123", {})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 0

    def test_list_grants_not_found(self, handler, mock_mound):
        """Test listing grants when node not found."""
        mock_mound.get_access_grants.side_effect = ValueError("Node not found")

        result = handler._handle_list_access_grants("nonexistent", {})

        assert result.status_code == 404

    def test_list_grants_no_mound(self, handler_no_mound):
        """Test listing grants when mound not available."""
        result = handler_no_mound._handle_list_access_grants("node-123", {})

        assert result.status_code == 503
