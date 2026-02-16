"""Tests for RelationshipOperationsMixin."""

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
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from aragora.server.handlers.knowledge_base.mound.relationships import (
    RelationshipOperationsMixin,
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


@dataclass
class MockNode:
    """Mock knowledge node."""

    id: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content}


@dataclass
class MockRelationship:
    """Mock relationship."""

    id: str
    from_node_id: str
    to_node_id: str
    relationship_type: str
    strength: float = 1.0
    created_at: datetime | None = None
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    get_node: AsyncMock = field(default_factory=AsyncMock)
    get_relationships: AsyncMock = field(default_factory=AsyncMock)
    add_relationship: AsyncMock = field(default_factory=AsyncMock)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class RelationshipHandler(RelationshipOperationsMixin):
    """Handler implementation for testing RelationshipOperationsMixin."""

    def __init__(self, mound: MockKnowledgeMound | None = None, user: MockUser | None = None):
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
    return RelationshipHandler(mound=mock_mound, user=mock_user)


@pytest.fixture
def handler_no_mound(mock_user):
    """Create a test handler without mound."""
    return RelationshipHandler(mound=None, user=mock_user)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test get_node_relationships
# =============================================================================


class TestGetNodeRelationships:
    """Tests for get node relationships endpoint."""

    def test_get_relationships_success(self, handler, mock_mound):
        """Test successful relationship retrieval."""
        node = MockNode(id="node-123", content="Test node")
        mock_mound.get_node.return_value = node

        relationships = [
            MockRelationship(
                id="rel-1",
                from_node_id="node-123",
                to_node_id="node-456",
                relationship_type="supports",
                created_at=datetime(2026, 1, 27, 12, 0, 0),
                created_by="user-123",
            ),
            MockRelationship(
                id="rel-2",
                from_node_id="node-789",
                to_node_id="node-123",
                relationship_type="contradicts",
            ),
        ]
        mock_mound.get_relationships.return_value = relationships

        result = handler._handle_get_node_relationships("node-123", {})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["node_id"] == "node-123"
        assert data["count"] == 2
        assert len(data["relationships"]) == 2

    def test_get_relationships_with_type_filter(self, handler, mock_mound):
        """Test relationship retrieval with type filter."""
        node = MockNode(id="node-123", content="Test")
        mock_mound.get_node.return_value = node
        mock_mound.get_relationships.return_value = []

        result = handler._handle_get_node_relationships(
            "node-123", {"relationship_type": ["supports"]}
        )

        assert result.status_code == 200

    def test_get_relationships_outgoing(self, handler, mock_mound):
        """Test outgoing relationships only."""
        node = MockNode(id="node-123", content="Test")
        mock_mound.get_node.return_value = node
        mock_mound.get_relationships.return_value = []

        result = handler._handle_get_node_relationships("node-123", {"direction": ["outgoing"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["direction"] == "outgoing"

    def test_get_relationships_incoming(self, handler, mock_mound):
        """Test incoming relationships only."""
        node = MockNode(id="node-123", content="Test")
        mock_mound.get_node.return_value = node
        mock_mound.get_relationships.return_value = []

        result = handler._handle_get_node_relationships("node-123", {"direction": ["incoming"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["direction"] == "incoming"

    def test_get_relationships_invalid_direction(self, handler, mock_mound):
        """Test relationships with invalid direction."""
        node = MockNode(id="node-123", content="Test")
        mock_mound.get_node.return_value = node

        result = handler._handle_get_node_relationships("node-123", {"direction": ["invalid"]})

        assert result.status_code == 400
        assert "direction" in parse_response(result)["error"]

    def test_get_relationships_node_not_found(self, handler, mock_mound):
        """Test relationships when node not found."""
        mock_mound.get_node.return_value = None

        result = handler._handle_get_node_relationships("nonexistent", {})

        assert result.status_code == 404

    def test_get_relationships_no_mound(self, handler_no_mound):
        """Test relationships when mound not available."""
        result = handler_no_mound._handle_get_node_relationships("node-123", {})

        assert result.status_code == 503


# =============================================================================
# Test create_relationship
# =============================================================================


class TestCreateRelationship:
    """Tests for create relationship endpoint."""

    def test_create_relationship_success(self, handler, mock_mound):
        """Test successful relationship creation."""
        mock_mound.add_relationship.return_value = "rel-123"

        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
                "relationship_type": "supports",
                "strength": 0.9,
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["id"] == "rel-123"
        assert data["relationship_type"] == "supports"

    def test_create_relationship_all_types(self, handler, mock_mound):
        """Test creating relationships of all valid types."""
        valid_types = ["supports", "contradicts", "derived_from", "related_to", "supersedes"]

        for rel_type in valid_types:
            mock_mound.add_relationship.return_value = f"rel-{rel_type}"

            body = json.dumps(
                {
                    "from_node_id": "node-1",
                    "to_node_id": "node-2",
                    "relationship_type": rel_type,
                }
            ).encode()
            http_handler = MockHandler(body=body)

            result = handler._handle_create_relationship(http_handler)

            assert result.status_code == 201
            assert parse_response(result)["relationship_type"] == rel_type

    def test_create_relationship_missing_from_node(self, handler):
        """Test relationship creation without from_node_id."""
        body = json.dumps(
            {
                "to_node_id": "node-2",
                "relationship_type": "supports",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_create_relationship_missing_to_node(self, handler):
        """Test relationship creation without to_node_id."""
        body = json.dumps(
            {
                "from_node_id": "node-1",
                "relationship_type": "supports",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_create_relationship_missing_type(self, handler):
        """Test relationship creation without relationship_type."""
        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_create_relationship_invalid_type(self, handler):
        """Test relationship creation with invalid type."""
        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
                "relationship_type": "invalid_type",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 400
        assert "Invalid relationship_type" in parse_response(result)["error"]

    def test_create_relationship_with_metadata(self, handler, mock_mound):
        """Test relationship creation with metadata."""
        mock_mound.add_relationship.return_value = "rel-456"

        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
                "relationship_type": "supports",
                "strength": 0.8,
                "created_by": "agent-1",
                "metadata": {"reason": "semantic similarity"},
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 201

    def test_create_relationship_no_mound(self, handler_no_mound):
        """Test relationship creation when mound not available."""
        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
                "relationship_type": "supports",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler_no_mound._handle_create_relationship(http_handler)

        assert result.status_code == 503

    def test_create_relationship_empty_body(self, handler):
        """Test relationship creation with empty body."""
        http_handler = MockHandler(body=b"")

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 400

    def test_create_relationship_error(self, handler, mock_mound):
        """Test relationship creation error handling."""
        mock_mound.add_relationship.side_effect = ValueError("Database error")

        body = json.dumps(
            {
                "from_node_id": "node-1",
                "to_node_id": "node-2",
                "relationship_type": "supports",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_create_relationship(http_handler)

        assert result.status_code == 500
