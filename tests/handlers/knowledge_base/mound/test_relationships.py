"""Tests for RelationshipOperationsMixin (aragora/server/handlers/knowledge_base/mound/relationships.py).

Covers all routes and behavior of the relationship operations mixin:
- GET  /api/v1/knowledge/mound/nodes/:id/relationships  - Get node relationships
- POST /api/v1/knowledge/mound/relationships             - Create relationship

Error cases: missing mound, invalid JSON body, missing required fields, server errors,
invalid relationship types, invalid direction, auth failures, and edge cases.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.relationships import (
    RelationshipOperationsMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-token-123"

_RUN_ASYNC_PATCH = "aragora.server.handlers.knowledge_base.mound.relationships._run_async"


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
# Autouse fixture: bypass @require_auth by making auth_config accept our token
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_require_auth(monkeypatch):
    """Patch auth_config so the @require_auth decorator accepts our test token."""
    from aragora.server import auth as auth_module

    monkeypatch.setattr(auth_module.auth_config, "api_token", _TEST_TOKEN)
    monkeypatch.setattr(
        auth_module.auth_config, "validate_token", lambda token: token == _TEST_TOKEN
    )


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for relationship tests."""

    command: str = "GET"
    path: str = ""
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": "test-agent",
            "Authorization": f"Bearer {_TEST_TOKEN}",
            "Content-Length": "0",
        }
    )
    client_address: tuple = ("127.0.0.1", 12345)
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def get(cls) -> MockHTTPHandler:
        return cls(command="GET")

    @classmethod
    def post(cls, body: dict | None = None) -> MockHTTPHandler:
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            return cls(
                command="POST",
                headers={
                    "User-Agent": "test-agent",
                    "Authorization": f"Bearer {_TEST_TOKEN}",
                    "Content-Length": str(len(raw)),
                },
                rfile=io.BytesIO(raw),
            )
        return cls(command="POST")


# ---------------------------------------------------------------------------
# Mock relationship / node objects
# ---------------------------------------------------------------------------


@dataclass
class MockKnowledgeNode:
    """Lightweight mock for KnowledgeNode."""

    node_id: str = "node-001"
    node_type: str = "fact"
    content: str = "Test content"


@dataclass
class MockRelationship:
    """Lightweight mock for a relationship between nodes."""

    id: str = "rel-001"
    from_node_id: str = "node-001"
    to_node_id: str = "node-002"
    relationship_type: str = "supports"
    strength: float = 0.9
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    )
    created_by: str = "test-user"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class RelTestHandler(RelationshipOperationsMixin):
    """Concrete handler for testing the relationship mixin."""

    def __init__(self, mound=None):
        self._mound_instance = mound

    def _get_mound(self):
        return self._mound_instance

    def require_auth_or_error(self, handler):
        """Mock auth that always succeeds."""
        user = MagicMock()
        user.authenticated = True
        user.user_id = "test-user-001"
        return user, None


class RelTestHandlerAuthFail(RelationshipOperationsMixin):
    """Concrete handler where auth always fails."""

    def __init__(self, mound=None):
        self._mound_instance = mound

    def _get_mound(self):
        return self._mound_instance

    def require_auth_or_error(self, handler):
        """Mock auth that always fails."""
        from aragora.server.handlers.utils.responses import error_response

        return None, error_response("Authentication required", 401)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with relationship-related methods."""
    mound = MagicMock()
    mound.get_node = MagicMock(return_value=MockKnowledgeNode(node_id="node-001"))
    mound.get_relationships = MagicMock(
        return_value=[
            MockRelationship(id="rel-001", from_node_id="node-001", to_node_id="node-002"),
            MockRelationship(
                id="rel-002",
                from_node_id="node-003",
                to_node_id="node-001",
                relationship_type="contradicts",
            ),
        ]
    )
    mound.add_relationship = MagicMock(return_value="rel-new-001")
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a RelTestHandler with a mocked mound."""
    return RelTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a RelTestHandler with no mound (None)."""
    return RelTestHandler(mound=None)


@pytest.fixture
def handler_auth_fail(mock_mound):
    """Create a RelTestHandlerAuthFail for auth failure testing."""
    return RelTestHandlerAuthFail(mound=mock_mound)


# ============================================================================
# Tests: _handle_get_node_relationships
# (GET /api/v1/knowledge/mound/nodes/:id/relationships)
# ============================================================================


class TestGetNodeRelationships:
    """Test _handle_get_node_relationships - get node relationships endpoint."""

    def test_success_default_direction(self, handler, mock_mound):
        """Successful query with default direction returns relationships."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-001"
        assert body["count"] == 2
        assert body["direction"] == "both"
        assert len(body["relationships"]) == 2

    def test_success_outgoing_direction(self, handler, mock_mound):
        """Direction 'outgoing' is accepted and forwarded."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {"direction": "outgoing"})
        assert _status(result) == 200
        body = _body(result)
        assert body["direction"] == "outgoing"

    def test_success_incoming_direction(self, handler, mock_mound):
        """Direction 'incoming' is accepted and forwarded."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {"direction": "incoming"})
        assert _status(result) == 200
        body = _body(result)
        assert body["direction"] == "incoming"

    def test_success_both_direction(self, handler, mock_mound):
        """Direction 'both' is accepted and forwarded."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {"direction": "both"})
        assert _status(result) == 200
        body = _body(result)
        assert body["direction"] == "both"

    def test_invalid_direction_returns_400(self, handler, mock_mound):
        """Invalid direction value returns 400."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {"direction": "sideways"})
        assert _status(result) == 400
        body = _body(result)
        assert "direction" in body["error"].lower()

    def test_with_relationship_type_filter(self, handler, mock_mound):
        """relationship_type query param is forwarded to mound."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships(
                "node-001", {"relationship_type": "supports"}
            )
        assert _status(result) == 200
        mock_mound.get_relationships.assert_called_once_with(
            node_id="node-001",
            relationship_type="supports",
            direction="both",
        )

    def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_node_relationships("node-001", {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_node_not_found_returns_404(self, handler, mock_mound):
        """Non-existent node returns 404."""
        mock_mound.get_node = MagicMock(return_value=None)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("nonexistent", {})
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()
        assert "nonexistent" in body["error"]

    def test_get_node_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound.get_node returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_node_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound.get_node returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_node_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound.get_node returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_node_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound.get_node returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_node_raises_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound.get_node returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_relationships_not_available_returns_503(self, handler, mock_mound):
        """Missing get_relationships method on mound returns 503."""
        del mock_mound.get_relationships
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_relationships_raises_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from get_relationships returns 500."""
        # First _run_async call succeeds (get_node), second fails (get_relationships)
        call_count = [0]

        def side_effect(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                return coro  # get_node succeeds
            raise RuntimeError("db fail")

        with patch(_RUN_ASYNC_PATCH, side_effect=side_effect):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_relationships_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from get_relationships returns 500."""
        call_count = [0]

        def side_effect(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                return coro
            raise KeyError("missing")

        with patch(_RUN_ASYNC_PATCH, side_effect=side_effect):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_get_relationships_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from get_relationships returns 500."""
        call_count = [0]

        def side_effect(coro):
            call_count[0] += 1
            if call_count[0] == 1:
                return coro
            raise OSError("disk")

        with patch(_RUN_ASYNC_PATCH, side_effect=side_effect):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 500

    def test_relationship_serialization_all_fields(self, handler, mock_mound):
        """Relationships are serialized with all expected fields."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        body = _body(result)
        rel = body["relationships"][0]
        assert rel["id"] == "rel-001"
        assert rel["from_node_id"] == "node-001"
        assert rel["to_node_id"] == "node-002"
        assert rel["relationship_type"] == "supports"
        assert rel["strength"] == 0.9
        assert rel["created_by"] == "test-user"
        assert rel["metadata"] == {}
        # created_at should be ISO format string
        assert "2026-01-15" in rel["created_at"]

    def test_relationship_created_at_none_serializes_as_none(self, handler, mock_mound):
        """Relationship with created_at=None serializes created_at as null."""
        mock_mound.get_relationships = MagicMock(return_value=[MockRelationship(created_at=None)])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        body = _body(result)
        assert body["relationships"][0]["created_at"] is None

    def test_empty_relationships_list(self, handler, mock_mound):
        """Empty relationships list returns count=0."""
        mock_mound.get_relationships = MagicMock(return_value=[])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        body = _body(result)
        assert body["count"] == 0
        assert body["relationships"] == []

    def test_relationship_type_param_truncated(self, handler, mock_mound):
        """Oversized relationship_type param is truncated to 50 chars."""
        long_type = "x" * 100
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_node_relationships("node-001", {"relationship_type": long_type})
        call_kwargs = mock_mound.get_relationships.call_args
        assert len(call_kwargs.kwargs.get("relationship_type", "")) <= 50

    def test_direction_param_truncated(self, handler, mock_mound):
        """Oversized direction param is truncated to 20 chars, then rejected as invalid."""
        long_dir = "x" * 100
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {"direction": long_dir})
        # Truncated to 20 chars, but still not a valid direction
        assert _status(result) == 400

    def test_mound_get_node_not_available_returns_node_none(self, handler, mock_mound):
        """When get_node attr does not exist, node is None and returns 404."""
        del mock_mound.get_node
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 404

    def test_response_structure_correct(self, handler, mock_mound):
        """Response contains node_id, relationships, count, direction."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        body = _body(result)
        assert "node_id" in body
        assert "relationships" in body
        assert "count" in body
        assert "direction" in body

    def test_count_matches_relationships_length(self, handler, mock_mound):
        """Count field matches the actual number of relationships."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        body = _body(result)
        assert body["count"] == len(body["relationships"])


# ============================================================================
# Tests: _handle_create_relationship
# (POST /api/v1/knowledge/mound/relationships)
# ============================================================================


class TestCreateRelationship:
    """Test _handle_create_relationship - create relationship endpoint."""

    def test_create_success(self, handler, mock_mound):
        """Successfully creating a relationship returns 201."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["id"] == "rel-new-001"
        assert body["from_node_id"] == "node-001"
        assert body["to_node_id"] == "node-002"
        assert body["relationship_type"] == "supports"

    def test_create_with_all_optional_fields(self, handler, mock_mound):
        """Creating a relationship with all optional fields works."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "contradicts",
                "strength": 0.75,
                "created_by": "admin",
                "metadata": {"reason": "logical inconsistency"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201
        mock_mound.add_relationship.assert_called_once_with(
            from_node_id="node-001",
            to_node_id="node-002",
            relationship_type="contradicts",
            strength=0.75,
            created_by="admin",
            metadata={"reason": "logical inconsistency"},
        )

    def test_create_default_strength_is_1(self, handler, mock_mound):
        """Default strength is 1.0 when not specified."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_relationship(http)
        call_kwargs = mock_mound.add_relationship.call_args
        assert call_kwargs.kwargs.get("strength") == 1.0

    def test_create_default_created_by_is_empty(self, handler, mock_mound):
        """Default created_by is empty string when not specified."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_relationship(http)
        call_kwargs = mock_mound.add_relationship.call_args
        assert call_kwargs.kwargs.get("created_by") == ""

    def test_create_default_metadata_is_none(self, handler, mock_mound):
        """Default metadata is None when not specified."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_relationship(http)
        call_kwargs = mock_mound.add_relationship.call_args
        assert call_kwargs.kwargs.get("metadata") is None

    def test_create_missing_from_node_id_returns_400(self, handler):
        """Missing from_node_id returns 400."""
        http = MockHTTPHandler.post(
            {
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "from_node_id" in body["error"]

    def test_create_missing_to_node_id_returns_400(self, handler):
        """Missing to_node_id returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "relationship_type": "supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "to_node_id" in body["error"]

    def test_create_missing_relationship_type_returns_400(self, handler):
        """Missing relationship_type returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "relationship_type" in body["error"]

    def test_create_empty_from_node_id_returns_400(self, handler):
        """Empty from_node_id returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "from_node_id" in body["error"]

    def test_create_empty_to_node_id_returns_400(self, handler):
        """Empty to_node_id returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "",
                "relationship_type": "supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "to_node_id" in body["error"]

    def test_create_empty_relationship_type_returns_400(self, handler):
        """Empty relationship_type returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "relationship_type" in body["error"]

    def test_create_valid_type_supports(self, handler, mock_mound):
        """relationship_type 'supports' is valid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_valid_type_contradicts(self, handler, mock_mound):
        """relationship_type 'contradicts' is valid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "contradicts",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_valid_type_derived_from(self, handler, mock_mound):
        """relationship_type 'derived_from' is valid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "derived_from",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_valid_type_related_to(self, handler, mock_mound):
        """relationship_type 'related_to' is valid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "related_to",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_valid_type_supersedes(self, handler, mock_mound):
        """relationship_type 'supersedes' is valid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supersedes",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_invalid_type_returns_400(self, handler):
        """Invalid relationship_type returns 400."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "unknown_type",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()
        assert "relationship_type" in body["error"].lower()

    def test_create_no_body_returns_400(self, handler):
        """Zero content-length body returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "0",
            },
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body required" in body["error"].lower()

    def test_create_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "11",
            },
            rfile=io.BytesIO(b"not valid j"),
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    def test_create_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        result = handler_no_mound._handle_create_relationship(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_create_add_relationship_not_available_returns_503(self, handler, mock_mound):
        """Missing add_relationship method on mound returns 503."""
        del mock_mound.add_relationship
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_create_mound_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound.add_relationship returns 500."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 500

    def test_create_mound_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound.add_relationship returns 500."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 500

    def test_create_mound_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound.add_relationship returns 500."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 500

    def test_create_mound_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound.add_relationship returns 500."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 500

    def test_create_mound_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound.add_relationship returns 500."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 500

    def test_create_auth_failure_returns_401(self, handler_auth_fail):
        """Authentication failure returns 401."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        result = handler_auth_fail._handle_create_relationship(http)
        assert _status(result) == 401

    def test_create_response_structure(self, handler, mock_mound):
        """Create response contains id, from_node_id, to_node_id, relationship_type."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "node-001",
                "to_node_id": "node-002",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        body = _body(result)
        assert "id" in body
        assert "from_node_id" in body
        assert "to_node_id" in body
        assert "relationship_type" in body

    def test_create_content_length_header_missing_defaults_zero(self, handler):
        """Content-Length header missing defaults to 0, returns 400 (body required)."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
            },
            rfile=io.BytesIO(b""),
        )
        # headers.get("Content-Length", 0) -> 0 -> body required
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400


# ============================================================================
# Tests: Security and Edge Cases
# ============================================================================


class TestSecurityAndEdgeCases:
    """Test security and edge case scenarios for relationship operations."""

    def test_create_sql_injection_in_from_node_id(self, handler, mock_mound):
        """SQL injection in from_node_id is passed as-is to mound."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "'; DROP TABLE nodes; --",
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_xss_in_metadata(self, handler, mock_mound):
        """XSS in metadata is stored as-is (rendering handles escaping)."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "supports",
                "metadata": {"note": "<script>alert('xss')</script>"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_get_relationships_path_traversal_in_node_id(self, handler, mock_mound):
        """Path traversal in node_id is treated as plain string."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("../../etc/passwd", {})
        # get_node returns the mock node regardless
        assert _status(result) == 200

    def test_create_unicode_node_ids(self, handler, mock_mound):
        """Unicode characters in node IDs are accepted."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "noeud-un",
                "to_node_id": "noeud-deux",
                "relationship_type": "related_to",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_create_very_long_node_ids(self, handler, mock_mound):
        """Very long node IDs are accepted (handler does not limit them)."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n" * 10000,
                "to_node_id": "m" * 10000,
                "relationship_type": "supports",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_relationship(http)
        assert _status(result) == 201

    def test_relationship_with_minimal_attributes(self, handler, mock_mound):
        """Relationship object with only id attribute still serializes."""
        minimal_rel = MagicMock(spec=[])  # No attributes at all
        mock_mound.get_relationships = MagicMock(return_value=[minimal_rel])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node_relationships("node-001", {})
        assert _status(result) == 200
        body = _body(result)
        rel = body["relationships"][0]
        # All fields should be None since the object has no attributes
        assert rel["id"] is None
        assert rel["from_node_id"] is None
        assert rel["to_node_id"] is None
        assert rel["relationship_type"] is None
        assert rel["strength"] is None
        assert rel["created_at"] is None
        assert rel["created_by"] is None
        assert rel["metadata"] is None

    def test_create_none_from_node_id_returns_400(self, handler):
        """None from_node_id returns 400 (falsy check)."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": None,
                "to_node_id": "n2",
                "relationship_type": "supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400

    def test_create_case_sensitive_relationship_type(self, handler):
        """Relationship types are case-sensitive; 'Supports' is invalid."""
        http = MockHTTPHandler.post(
            {
                "from_node_id": "n1",
                "to_node_id": "n2",
                "relationship_type": "Supports",
            }
        )
        result = handler._handle_create_relationship(http)
        assert _status(result) == 400
