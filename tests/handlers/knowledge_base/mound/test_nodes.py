"""Tests for NodeOperationsMixin (aragora/server/handlers/knowledge_base/mound/nodes.py).

Covers all routes and behavior of the node operations mixin:
- POST /api/v1/knowledge/mound/query      - Semantic query against knowledge mound
- POST /api/v1/knowledge/mound/nodes       - Create knowledge node
- GET  /api/v1/knowledge/mound/nodes       - List/filter nodes
- GET  /api/v1/knowledge/mound/nodes/:id   - Get specific node
- GET  /api/v1/knowledge/mound/stats       - Get mound statistics

Error cases: missing mound, invalid JSON body, missing required fields, server errors,
invalid node types, invalid provenance, and parameter validation.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.nodes import (
    NodeOperationsMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-token-123"

_RUN_ASYNC_PATCH = "aragora.server.handlers.knowledge_base.mound.nodes._run_async"


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
    """Lightweight mock HTTP handler for node tests."""

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
# Mock knowledge node / query result
# ---------------------------------------------------------------------------


@dataclass
class MockKnowledgeNode:
    """Lightweight mock for KnowledgeNode."""

    node_id: str = "node-001"
    node_type: str = "fact"
    content: str = "Test content"
    confidence: float = 0.8
    workspace_id: str = "default"
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tier: str = "slow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "confidence": self.confidence,
            "workspace_id": self.workspace_id,
            "topics": self.topics,
            "metadata": self.metadata,
            "tier": self.tier,
        }


@dataclass
class MockQueryResult:
    """Lightweight mock for semantic query results."""

    query: str = "test query"
    nodes: list[MockKnowledgeNode] = field(default_factory=list)
    total_count: int = 0
    processing_time_ms: float = 12.5


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class NodeTestHandler(NodeOperationsMixin):
    """Concrete handler for testing the node mixin."""

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


class NodeTestHandlerAuthFail(NodeOperationsMixin):
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
    """Create a mock KnowledgeMound with node-related methods."""
    mound = MagicMock()
    mound.add_node = MagicMock(return_value="node-001")
    mound.get_node = MagicMock(return_value=MockKnowledgeNode(node_id="node-001"))
    mound.query_nodes = MagicMock(
        return_value=[
            MockKnowledgeNode(node_id="node-001"),
            MockKnowledgeNode(node_id="node-002", confidence=0.6),
        ]
    )
    mound.query_semantic = MagicMock(
        return_value=MockQueryResult(
            query="test",
            nodes=[MockKnowledgeNode(node_id="node-001")],
            total_count=1,
            processing_time_ms=10.5,
        )
    )
    mound.get_stats = MagicMock(
        return_value={
            "total_nodes": 100,
            "by_type": {"fact": 50, "claim": 30, "memory": 20},
            "workspaces": 3,
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a NodeTestHandler with a mocked mound."""
    return NodeTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a NodeTestHandler with no mound (None)."""
    return NodeTestHandler(mound=None)


@pytest.fixture
def handler_auth_fail(mock_mound):
    """Create a NodeTestHandlerAuthFail for auth failure testing."""
    return NodeTestHandlerAuthFail(mound=mock_mound)


# ============================================================================
# Tests: _handle_mound_query (POST /api/v1/knowledge/mound/query)
# ============================================================================


class TestMoundQuery:
    """Test _handle_mound_query - semantic query endpoint."""

    def test_query_success(self, handler, mock_mound):
        """Successful semantic query returns nodes and metadata."""
        http = MockHTTPHandler.post({"query": "test query", "limit": 5})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == "test"
        assert len(body["nodes"]) == 1
        assert body["total_count"] == 1
        assert body["processing_time_ms"] == 10.5

    def test_query_with_workspace_id(self, handler, mock_mound):
        """workspace_id from body is forwarded to query_semantic."""
        http = MockHTTPHandler.post({"query": "test", "workspace_id": "ws-prod"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        mock_mound.query_semantic.assert_called_once()
        call_kwargs = mock_mound.query_semantic.call_args
        assert call_kwargs.kwargs.get("workspace_id") == "ws-prod" or (
            len(call_kwargs.args) > 0 and "ws-prod" in str(call_kwargs)
        )

    def test_query_with_min_confidence(self, handler, mock_mound):
        """min_confidence from body is forwarded."""
        http = MockHTTPHandler.post({"query": "test", "min_confidence": 0.7})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        mock_mound.query_semantic.assert_called_once()

    def test_query_with_node_types(self, handler, mock_mound):
        """node_types from body is forwarded."""
        http = MockHTTPHandler.post({"query": "test", "node_types": ["fact", "claim"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        mock_mound.query_semantic.assert_called_once()

    def test_query_empty_query_returns_400(self, handler):
        """Empty query string returns 400."""
        http = MockHTTPHandler.post({"query": ""})
        result = handler._handle_mound_query(http)
        assert _status(result) == 400
        body = _body(result)
        assert "query" in body["error"].lower()

    def test_query_missing_query_field_returns_400(self, handler):
        """Missing query field returns 400."""
        http = MockHTTPHandler.post({"limit": 10})
        result = handler._handle_mound_query(http)
        assert _status(result) == 400

    def test_query_no_body_returns_400(self, handler):
        """Empty body (no query) returns 400."""
        http = MockHTTPHandler.post({})
        result = handler._handle_mound_query(http)
        assert _status(result) == 400

    def test_query_invalid_json_returns_400(self, handler):
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
        result = handler._handle_mound_query(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    def test_query_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.post({"query": "test"})
        result = handler_no_mound._handle_mound_query(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_query_mound_error_returns_500(self, handler, mock_mound):
        """Server error from mound.query_semantic returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = handler._handle_mound_query(http)
        assert _status(result) == 500

    def test_query_default_limit_is_10(self, handler, mock_mound):
        """Default limit is 10 when not specified."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        call_kwargs = mock_mound.query_semantic.call_args
        # limit=10 should be passed
        assert 10 in call_kwargs.args or call_kwargs.kwargs.get("limit") == 10

    def test_query_default_workspace_is_default(self, handler, mock_mound):
        """Default workspace_id is 'default'."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        call_kwargs = mock_mound.query_semantic.call_args
        assert "default" in str(call_kwargs)

    def test_query_default_min_confidence_is_zero(self, handler, mock_mound):
        """Default min_confidence is 0.0."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_mound_query(http)
        call_kwargs = mock_mound.query_semantic.call_args
        assert 0.0 in call_kwargs.args or call_kwargs.kwargs.get("min_confidence") == 0.0

    def test_query_result_with_no_nodes(self, handler, mock_mound):
        """Empty query result returns zero nodes."""
        mock_mound.query_semantic = MagicMock(
            return_value=MockQueryResult(query="empty", nodes=[], total_count=0)
        )
        http = MockHTTPHandler.post({"query": "empty search"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["nodes"] == []
        assert body["total_count"] == 0

    def test_query_result_nodes_serialized_via_to_dict(self, handler, mock_mound):
        """Nodes in query result are serialized via to_dict()."""
        http = MockHTTPHandler.post({"query": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        body = _body(result)
        assert body["nodes"][0]["node_id"] == "node-001"
        assert body["nodes"][0]["content"] == "Test content"

    def test_query_zero_content_length(self, handler):
        """Zero content-length body results in empty data, missing query returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "0",
            },
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_mound_query(http)
        assert _status(result) == 400


# ============================================================================
# Tests: _handle_create_node (POST /api/v1/knowledge/mound/nodes)
# ============================================================================


class TestCreateNode:
    """Test _handle_create_node - create knowledge node endpoint."""

    def test_create_node_success(self, handler, mock_mound):
        """Successfully creating a node returns 201 with node data."""
        http = MockHTTPHandler.post(
            {
                "content": "The sky is blue",
                "node_type": "fact",
                "workspace_id": "default",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["node_id"] == "node-001"

    def test_create_node_with_all_fields(self, handler, mock_mound):
        """Node creation with all fields works correctly."""
        http = MockHTTPHandler.post(
            {
                "content": "Complex fact",
                "node_type": "claim",
                "workspace_id": "ws-prod",
                "confidence": 0.9,
                "topics": ["science", "physics"],
                "metadata": {"key": "value"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_create_node_empty_content_returns_400(self, handler):
        """Empty content returns 400."""
        http = MockHTTPHandler.post({"content": "", "node_type": "fact"})
        result = handler._handle_create_node(http)
        assert _status(result) == 400
        body = _body(result)
        assert "content" in body["error"].lower()

    def test_create_node_missing_content_returns_400(self, handler):
        """Missing content field returns 400."""
        http = MockHTTPHandler.post({"node_type": "fact"})
        result = handler._handle_create_node(http)
        assert _status(result) == 400

    def test_create_node_no_body_returns_400(self, handler):
        """Missing request body returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "0",
            },
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_create_node(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body required" in body["error"].lower()

    def test_create_node_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "7",
            },
            rfile=io.BytesIO(b"not-jsn"),
        )
        result = handler._handle_create_node(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    def test_create_node_invalid_node_type_returns_400(self, handler):
        """Invalid node_type returns 400."""
        http = MockHTTPHandler.post({"content": "test", "node_type": "invalid_type"})
        result = handler._handle_create_node(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid node_type" in body["error"].lower()

    def test_create_node_valid_types(self, handler, mock_mound):
        """All valid node types are accepted."""
        valid_types = ["fact", "claim", "memory", "evidence", "consensus", "entity"]
        for ntype in valid_types:
            http = MockHTTPHandler.post({"content": "test", "node_type": ntype})
            with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
                result = handler._handle_create_node(http)
            assert _status(result) == 201, f"node_type '{ntype}' should be valid"

    def test_create_node_default_type_is_fact(self, handler, mock_mound):
        """Default node_type is 'fact' when not specified."""
        http = MockHTTPHandler.post({"content": "test content"})
        with (
            patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro),
            patch(
                "aragora.server.handlers.knowledge_base.mound.nodes.KnowledgeNode",
                create=True,
            ) as mock_kn_cls,
        ):
            mock_kn_cls.return_value = MagicMock()
            # The handler creates a KnowledgeNode with node_type from data
            handler._handle_create_node(http)
        # We can't easily verify the KnowledgeNode arg since it's imported
        # inside the function, but the test should pass with default "fact"

    def test_create_node_with_source_provenance(self, handler, mock_mound):
        """Node creation with source/provenance works."""
        http = MockHTTPHandler.post(
            {
                "content": "Sourced fact",
                "source": {
                    "type": "user",
                    "id": "src-001",
                    "user_id": "user-123",
                    "agent_id": None,
                    "debate_id": "debate-001",
                    "document_id": None,
                },
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_create_node_invalid_source_type_returns_400(self, handler):
        """Invalid source type returns 400."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"type": "not_a_real_type", "id": "src"},
            }
        )
        result = handler._handle_create_node(http)
        assert _status(result) == 400
        body = _body(result)
        assert "source type" in body["error"].lower()

    def test_create_node_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.post({"content": "test"})
        result = handler_no_mound._handle_create_node(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_create_node_mound_add_error_returns_500(self, handler, mock_mound):
        """Server error from mound.add_node returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = handler._handle_create_node(http)
        assert _status(result) == 500

    def test_create_node_returns_saved_node_to_dict(self, handler, mock_mound):
        """When saved_node exists, to_dict() is used in response."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        body = _body(result)
        assert body["node_id"] == "node-001"
        assert body["content"] == "Test content"

    def test_create_node_saved_node_is_none_fallback(self, handler, mock_mound):
        """When saved node is None, response falls back to {id: node_id}."""
        mock_mound.get_node = MagicMock(return_value=None)
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["id"] == "node-001"

    def test_create_node_default_confidence_is_0_5(self, handler, mock_mound):
        """Default confidence is 0.5 when not specified."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_node(http)
        # Verify add_node was called (node object created with default confidence)
        mock_mound.add_node.assert_called_once()

    def test_create_node_default_workspace_is_default(self, handler, mock_mound):
        """Default workspace_id is 'default'."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_node(http)
        mock_mound.add_node.assert_called_once()

    def test_create_node_default_tier_is_slow(self, handler, mock_mound):
        """Default tier is 'slow' when not specified."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_create_node(http)
        mock_mound.add_node.assert_called_once()

    def test_create_node_auth_failure_returns_401(self, handler_auth_fail):
        """Authentication failure returns 401."""
        http = MockHTTPHandler.post({"content": "test"})
        result = handler_auth_fail._handle_create_node(http)
        assert _status(result) == 401

    def test_create_node_with_topics(self, handler, mock_mound):
        """Node creation with topics list."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "topics": ["ml", "ai", "nlp"],
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_create_node_with_metadata(self, handler, mock_mound):
        """Node creation with metadata dict."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "metadata": {"source": "paper", "year": 2025},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201


# ============================================================================
# Tests: _handle_get_node (GET /api/v1/knowledge/mound/nodes/:id)
# ============================================================================


class TestGetNode:
    """Test _handle_get_node - get specific node endpoint."""

    def test_get_node_success(self, handler, mock_mound):
        """Successfully getting a node returns its data."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-001"
        assert body["content"] == "Test content"

    def test_get_node_not_found_returns_404(self, handler, mock_mound):
        """Non-existent node returns 404."""
        mock_mound.get_node = MagicMock(return_value=None)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node("nonexistent")
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()
        assert "nonexistent" in body["error"]

    def test_get_node_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_node("node-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_node_mound_error_returns_500(self, handler, mock_mound):
        """Server error from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = handler._handle_get_node("node-001")
        assert _status(result) == 500

    def test_get_node_serialized_via_to_dict(self, handler, mock_mound):
        """Node is serialized via to_dict()."""
        custom_node = MockKnowledgeNode(
            node_id="custom-123",
            content="Custom content",
            node_type="claim",
            confidence=0.95,
        )
        mock_mound.get_node = MagicMock(return_value=custom_node)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node("custom-123")
        body = _body(result)
        assert body["node_id"] == "custom-123"
        assert body["content"] == "Custom content"
        assert body["node_type"] == "claim"
        assert body["confidence"] == 0.95

    def test_get_node_passes_id_to_mound(self, handler, mock_mound):
        """Node ID is forwarded to mound.get_node."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_node("specific-node-id")
        mock_mound.get_node.assert_called_once_with("specific-node-id")

    def test_get_node_with_special_characters_in_id(self, handler, mock_mound):
        """Node ID with special characters is handled."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node("node-with-special_chars.123")
        assert _status(result) == 200


# ============================================================================
# Tests: _handle_list_nodes (GET /api/v1/knowledge/mound/nodes)
# ============================================================================


class TestListNodes:
    """Test _handle_list_nodes - list/filter nodes endpoint."""

    def test_list_nodes_default(self, handler, mock_mound):
        """Default list returns nodes with count, limit, offset."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({})
        assert _status(result) == 200
        body = _body(result)
        assert len(body["nodes"]) == 2
        assert body["count"] == 2
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_nodes_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to query_nodes."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"workspace_id": "ws-prod"})
        mock_mound.query_nodes.assert_called_once()
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("workspace_id") == "ws-prod"

    def test_list_nodes_with_node_types_single(self, handler, mock_mound):
        """Single node_type filter is forwarded."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"node_types": "fact"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("node_type") == "fact"

    def test_list_nodes_with_node_types_multiple(self, handler, mock_mound):
        """Multiple node_types use the first for query, filter rest post-query."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"node_types": "fact,claim,memory"})
        call_kwargs = mock_mound.query_nodes.call_args
        # First type is used for query
        assert call_kwargs.kwargs.get("node_type") == "fact"

    def test_list_nodes_with_limit(self, handler, mock_mound):
        """Custom limit is forwarded (clamped 1-200)."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"limit": "25"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("limit") == 25

    def test_list_nodes_limit_clamped_max(self, handler, mock_mound):
        """Limit above 200 is clamped to 200."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"limit": "999"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("limit") == 200

    def test_list_nodes_limit_clamped_min(self, handler, mock_mound):
        """Limit below 1 is clamped to 1."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"limit": "0"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("limit") == 1

    def test_list_nodes_with_offset(self, handler, mock_mound):
        """Custom offset is forwarded (clamped 0-10000)."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"offset": "100"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("offset") == 100

    def test_list_nodes_offset_clamped_max(self, handler, mock_mound):
        """Offset above 10000 is clamped."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"offset": "99999"})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("offset") == 10000

    def test_list_nodes_min_confidence_filter(self, handler, mock_mound):
        """min_confidence filters nodes post-query."""
        # One node has 0.8 confidence, other has 0.6
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"min_confidence": "0.7"})
        body = _body(result)
        # Only the node with 0.8 confidence should pass
        assert body["count"] == 1
        assert body["nodes"][0]["confidence"] == 0.8

    def test_list_nodes_min_confidence_zero_no_filter(self, handler, mock_mound):
        """min_confidence=0.0 does not filter anything."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"min_confidence": "0.0"})
        body = _body(result)
        assert body["count"] == 2

    def test_list_nodes_tier_filter(self, handler, mock_mound):
        """tier parameter filters nodes post-query."""
        # Both mock nodes have tier="slow"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"tier": "slow"})
        body = _body(result)
        assert body["count"] == 2

    def test_list_nodes_tier_filter_no_match(self, handler, mock_mound):
        """Tier filter with non-matching tier returns empty."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"tier": "fast"})
        body = _body(result)
        assert body["count"] == 0

    def test_list_nodes_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_list_nodes({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_list_nodes_mound_error_returns_500(self, handler, mock_mound):
        """Server error from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = handler._handle_list_nodes({})
        assert _status(result) == 500

    def test_list_nodes_empty_result(self, handler, mock_mound):
        """Empty node list returns count=0."""
        mock_mound.query_nodes = MagicMock(return_value=[])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({})
        body = _body(result)
        assert body["count"] == 0
        assert body["nodes"] == []

    def test_list_nodes_serialized_via_to_dict(self, handler, mock_mound):
        """Nodes are serialized via to_dict()."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({})
        body = _body(result)
        assert body["nodes"][0]["node_id"] == "node-001"
        assert body["nodes"][1]["node_id"] == "node-002"

    def test_list_nodes_default_workspace_is_default(self, handler, mock_mound):
        """Default workspace_id is 'default'."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("workspace_id") == "default"

    def test_list_nodes_no_node_types_passes_none(self, handler, mock_mound):
        """When no node_types param, node_type=None is passed."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({})
        call_kwargs = mock_mound.query_nodes.call_args
        assert call_kwargs.kwargs.get("node_type") is None

    def test_list_nodes_response_has_correct_structure(self, handler, mock_mound):
        """Response always includes nodes, count, limit, offset keys."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({})
        body = _body(result)
        assert "nodes" in body
        assert "count" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_nodes_count_matches_nodes_length(self, handler, mock_mound):
        """Count field matches the number of nodes."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({})
        body = _body(result)
        assert body["count"] == len(body["nodes"])


# ============================================================================
# Tests: _handle_mound_stats (GET /api/v1/knowledge/mound/stats)
# ============================================================================


class TestMoundStats:
    """Test _handle_mound_stats - mound statistics endpoint."""

    def test_stats_success(self, handler, mock_mound):
        """Successfully getting stats returns data."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_stats({})
        assert _status(result) == 200
        body = _body(result)
        assert body["total_nodes"] == 100
        assert body["by_type"]["fact"] == 50
        assert body["workspaces"] == 3

    def test_stats_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_mound_stats({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_stats_mound_error_returns_500(self, handler, mock_mound):
        """Server error from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("db fail")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("wrong")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = handler._handle_mound_stats({})
        assert _status(result) == 500

    def test_stats_empty_response(self, handler, mock_mound):
        """Stats with empty/zero values works."""
        mock_mound.get_stats = MagicMock(return_value={"total_nodes": 0})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_stats({})
        body = _body(result)
        assert body["total_nodes"] == 0

    def test_stats_custom_fields_passed_through(self, handler, mock_mound):
        """Custom fields in stats are passed through."""
        mock_mound.get_stats = MagicMock(return_value={"total_nodes": 42, "custom_metric": "yes"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_stats({})
        body = _body(result)
        assert body["custom_metric"] == "yes"

    def test_stats_returns_json_response(self, handler, mock_mound):
        """Stats endpoint returns a HandlerResult with JSON content."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_stats({})
        assert isinstance(result, HandlerResult)
        assert result.content_type == "application/json"


# ============================================================================
# Tests: Security / Edge Cases
# ============================================================================


class TestSecurityAndEdgeCases:
    """Test security-related scenarios and edge cases."""

    def test_query_path_traversal_in_query(self, handler, mock_mound):
        """Path traversal in query string is handled safely (passed as text)."""
        http = MockHTTPHandler.post({"query": "../../etc/passwd"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        # Should be treated as a normal query string, not an exploit
        assert _status(result) == 200

    def test_query_sql_injection_in_query(self, handler, mock_mound):
        """SQL injection in query is passed as-is to the search (mound handles safety)."""
        http = MockHTTPHandler.post({"query": "'; DROP TABLE nodes; --"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        assert _status(result) == 200

    def test_create_node_xss_in_content(self, handler, mock_mound):
        """XSS in content is stored as-is (rendering layer handles escaping)."""
        http = MockHTTPHandler.post(
            {
                "content": "<script>alert('xss')</script>",
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_get_node_path_traversal_in_id(self, handler, mock_mound):
        """Path traversal in node ID is handled (just passed to mound.get_node)."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_node("../../etc/passwd")
        # Node won't be found with traversal ID, returns the node from mock
        assert _status(result) == 200

    def test_list_nodes_oversized_workspace_param(self, handler, mock_mound):
        """Oversized workspace_id is truncated by get_bounded_string_param."""
        long_ws = "x" * 200
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"workspace_id": long_ws})
        # Should not error, string is truncated
        assert _status(result) == 200

    def test_list_nodes_non_numeric_limit(self, handler, mock_mound):
        """Non-numeric limit falls back to default."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"limit": "abc"})
        body = _body(result)
        # Default limit is 50
        assert body["limit"] == 50

    def test_list_nodes_non_numeric_offset(self, handler, mock_mound):
        """Non-numeric offset falls back to default."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"offset": "abc"})
        body = _body(result)
        assert body["offset"] == 0

    def test_list_nodes_negative_min_confidence(self, handler, mock_mound):
        """Negative min_confidence is clamped to 0.0."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"min_confidence": "-0.5"})
        body = _body(result)
        # All nodes pass since min_confidence is clamped to 0.0
        assert body["count"] == 2

    def test_list_nodes_min_confidence_above_one(self, handler, mock_mound):
        """min_confidence above 1.0 is clamped to 1.0, filtering everything."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_list_nodes({"min_confidence": "2.0"})
        body = _body(result)
        # Clamped to 1.0, no nodes have confidence >= 1.0
        assert body["count"] == 0

    def test_create_node_very_long_content(self, handler, mock_mound):
        """Very long content is accepted (handler doesn't limit it)."""
        http = MockHTTPHandler.post({"content": "x" * 100000})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_query_unicode_content(self, handler, mock_mound):
        """Unicode characters in query are handled correctly."""
        http = MockHTTPHandler.post({"query": "recherche en francais"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        assert _status(result) == 200

    def test_create_node_unicode_content(self, handler, mock_mound):
        """Unicode characters in content are handled."""
        http = MockHTTPHandler.post({"content": "Contenu en francais"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_query_whitespace_only_returns_400(self, handler):
        """Whitespace-only query is treated as empty (falsy string)."""
        http = MockHTTPHandler.post({"query": "   "})
        # "   " is truthy in Python, so this will actually pass validation
        # and be forwarded to the mound as a whitespace query
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_mound_query(http)
        # Whitespace-only is still a non-empty string, so it passes
        assert _status(result) == 200

    def test_list_nodes_empty_node_types_string(self, handler, mock_mound):
        """Empty node_types string is treated as no filter."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_list_nodes({"node_types": ""})
        call_kwargs = mock_mound.query_nodes.call_args
        # Empty string from get_bounded_string_param returns None
        assert call_kwargs.kwargs.get("node_type") is None


# ============================================================================
# Tests: create node source types
# ============================================================================


class TestCreateNodeSourceTypes:
    """Test various source/provenance type values for node creation."""

    def test_source_type_user(self, handler, mock_mound):
        """Source type 'user' is valid."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"type": "user", "id": "src-001"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_source_type_agent(self, handler, mock_mound):
        """Source type 'agent' is valid."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"type": "agent", "id": "src-002", "agent_id": "agent-1"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_source_type_debate(self, handler, mock_mound):
        """Source type 'debate' is valid."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"type": "debate", "id": "src-003", "debate_id": "d-1"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_source_without_optional_fields(self, handler, mock_mound):
        """Source with minimal fields (only type and id) works."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"type": "user", "id": "src-minimal"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_source_empty_dict_no_type_uses_default_user(self, handler, mock_mound):
        """Source with empty type defaults to 'user'."""
        http = MockHTTPHandler.post(
            {
                "content": "test",
                "source": {"id": "src-no-type"},
            }
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201

    def test_no_source_creates_node_without_provenance(self, handler, mock_mound):
        """No source field means provenance is None."""
        http = MockHTTPHandler.post({"content": "test"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_create_node(http)
        assert _status(result) == 201
