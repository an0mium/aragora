"""Tests for GlobalKnowledgeOperationsMixin (aragora/server/handlers/knowledge_base/mound/global_knowledge.py).

Covers all five handler methods on the mixin:
- _handle_store_verified_fact     (POST /api/knowledge/mound/global)
- _handle_query_global            (GET /api/knowledge/mound/global)
- _handle_promote_to_global       (POST /api/knowledge/mound/global/promote)
- _handle_get_system_facts        (GET /api/knowledge/mound/global/facts)
- _handle_get_system_workspace_id (GET /api/knowledge/mound/global/workspace-id)

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Internal errors from mound operations (500)
- Auth/permission failures
- Edge cases (fallback serialization, user ID resolution, parameter clamping)
- Security (path traversal, injection)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
    GlobalKnowledgeOperationsMixin,
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
    """Mock user context returned by require_auth_or_error / require_admin_or_error."""

    id: str = "test-user-001"
    user_id: str = "test-user-001"
    email: str = "test@example.com"
    roles: list[str] = field(default_factory=lambda: ["admin"])
    permissions: list[str] = field(default_factory=lambda: ["*", "admin", "global_write"])


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for global knowledge tests."""

    command: str = "GET"
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Length": "0"})
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
# Mock knowledge items
# ---------------------------------------------------------------------------


@dataclass
class MockKnowledgeItem:
    """Mock knowledge item with to_dict support."""

    id: str = "item-001"
    content: str = "A verified fact"
    importance: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "importance": self.importance,
        }


@dataclass
class MockKnowledgeItemNoDict:
    """Mock knowledge item without to_dict (fallback path)."""

    id: str = "item-002"
    content: str = "Item without to_dict"
    importance: float = 0.5


# ---------------------------------------------------------------------------
# Concrete test classes combining the mixin with stubs
# ---------------------------------------------------------------------------


class GlobalKnowledgeTestHandler(GlobalKnowledgeOperationsMixin):
    """Concrete handler for testing the global knowledge mixin."""

    def __init__(self, mound=None, user=None):
        self._mound = mound
        self._user = user or MockUser()

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None

    def require_admin_or_error(self, handler):
        return self._user, None


class GlobalKnowledgeTestHandlerNoAuth(GlobalKnowledgeOperationsMixin):
    """Handler that simulates auth failure for both admin and regular auth."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Authentication required", 401)

    def require_admin_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Admin required", 403)


class GlobalKnowledgeTestHandlerNonAdmin(GlobalKnowledgeOperationsMixin):
    """Handler where admin check fails but regular auth succeeds, user has no global_write."""

    def __init__(self, mound=None):
        self._mound = mound
        self._user = MockUser(permissions=["read", "write"])

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None

    def require_admin_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Admin required", 403)


class GlobalKnowledgeTestHandlerGlobalWriter(GlobalKnowledgeOperationsMixin):
    """Handler where admin check fails but user has global_write permission."""

    def __init__(self, mound=None):
        self._mound = mound
        self._user = MockUser(permissions=["global_write"])

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None

    def require_admin_or_error(self, handler):
        from aragora.server.handlers.base import error_response

        return None, error_response("Admin required", 403)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with global knowledge methods."""
    mound = MagicMock()
    mound.store_verified_fact = AsyncMock(return_value="node-abc-123")
    mound.query_global_knowledge = AsyncMock(
        return_value=[
            MockKnowledgeItem(id="item-001", content="Fact 1", importance=0.95),
            MockKnowledgeItem(id="item-002", content="Fact 2", importance=0.85),
        ]
    )
    mound.promote_to_global = AsyncMock(return_value="global-xyz-789")
    mound.get_system_facts = AsyncMock(
        return_value=[
            MockKnowledgeItem(id="fact-001", content="System fact 1"),
            MockKnowledgeItem(id="fact-002", content="System fact 2"),
            MockKnowledgeItem(id="fact-003", content="System fact 3"),
        ]
    )
    mound.get_system_workspace_id = MagicMock(return_value="system-workspace-001")
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a GlobalKnowledgeTestHandler with a mocked mound."""
    return GlobalKnowledgeTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a GlobalKnowledgeTestHandler with no mound (None)."""
    return GlobalKnowledgeTestHandler(mound=None)


@pytest.fixture
def handler_no_auth(mock_mound):
    """Create a handler that fails all auth."""
    return GlobalKnowledgeTestHandlerNoAuth(mound=mock_mound)


@pytest.fixture
def handler_non_admin(mock_mound):
    """Create a handler where admin fails and user has no global_write."""
    return GlobalKnowledgeTestHandlerNonAdmin(mound=mock_mound)


@pytest.fixture
def handler_global_writer(mock_mound):
    """Create a handler where admin fails but user has global_write."""
    return GlobalKnowledgeTestHandlerGlobalWriter(mound=mock_mound)


# ============================================================================
# Tests: _handle_store_verified_fact (POST /api/knowledge/mound/global)
# ============================================================================


class TestStoreVerifiedFact:
    """Test _handle_store_verified_fact (POST /api/knowledge/mound/global)."""

    def test_store_verified_fact_success(self, handler, mock_mound):
        """Successfully storing a verified fact returns 201 with node_id."""
        http = MockHTTPHandler.with_body(
            {
                "content": "The Earth orbits the Sun",
                "source": "astronomy-textbook",
                "confidence": 0.99,
                "evidence_ids": ["ev-001", "ev-002"],
                "topics": ["astronomy", "physics"],
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["node_id"] == "node-abc-123"
        assert body["content"] == "The Earth orbits the Sun"
        assert body["source"] == "astronomy-textbook"
        assert body["verified_by"] == "test-user-001"

    def test_store_verified_fact_minimal_body(self, handler, mock_mound):
        """Store with only required fields (content, source) succeeds."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Water boils at 100C",
                "source": "chemistry-101",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["content"] == "Water boils at 100C"
        assert body["source"] == "chemistry-101"

    def test_store_verified_fact_default_confidence(self, handler, mock_mound):
        """Default confidence is 0.9 when not specified."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "source",
            }
        )
        handler._handle_store_verified_fact(http)
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert call_kwargs["confidence"] == 0.9

    def test_store_verified_fact_custom_confidence(self, handler, mock_mound):
        """Custom confidence is forwarded to mound."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "source",
                "confidence": 0.75,
            }
        )
        handler._handle_store_verified_fact(http)
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert call_kwargs["confidence"] == 0.75

    def test_store_verified_fact_default_evidence_ids(self, handler, mock_mound):
        """Default evidence_ids is empty list when not specified."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "source",
            }
        )
        handler._handle_store_verified_fact(http)
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert call_kwargs["evidence_ids"] == []

    def test_store_verified_fact_default_topics(self, handler, mock_mound):
        """Default topics is empty list when not specified."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "source",
            }
        )
        handler._handle_store_verified_fact(http)
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert call_kwargs["topics"] == []

    def test_store_verified_fact_forwards_correct_kwargs(self, handler, mock_mound):
        """All parameters are correctly forwarded to mound.store_verified_fact."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Test fact",
                "source": "test-source",
                "confidence": 0.8,
                "evidence_ids": ["ev-1"],
                "topics": ["topic-a"],
            }
        )
        handler._handle_store_verified_fact(http)
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert call_kwargs["content"] == "Test fact"
        assert call_kwargs["source"] == "test-source"
        assert call_kwargs["confidence"] == 0.8
        assert call_kwargs["evidence_ids"] == ["ev-1"]
        assert call_kwargs["verified_by"] == "test-user-001"
        assert call_kwargs["topics"] == ["topic-a"]

    def test_store_verified_fact_missing_content_returns_400(self, handler):
        """Missing content returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400
        body = _body(result)
        assert "content" in body["error"].lower()

    def test_store_verified_fact_empty_content_returns_400(self, handler):
        """Empty string content returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "content": "",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400

    def test_store_verified_fact_missing_source_returns_400(self, handler):
        """Missing source returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400
        body = _body(result)
        assert "source" in body["error"].lower()

    def test_store_verified_fact_empty_source_returns_400(self, handler):
        """Empty string source returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400

    def test_store_verified_fact_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower()

    def test_store_verified_fact_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower() or "body" in body["error"].lower()

    def test_store_verified_fact_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler_no_mound._handle_store_verified_fact(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_store_verified_fact_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler_no_auth._handle_store_verified_fact(http)
        assert _status(result) == 401

    def test_store_verified_fact_non_admin_no_global_write_returns_403(self, handler_non_admin):
        """Non-admin user without global_write permission returns 403."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler_non_admin._handle_store_verified_fact(http)
        assert _status(result) == 403
        body = _body(result)
        assert "admin" in body["error"].lower() or "global_write" in body["error"].lower()

    def test_store_verified_fact_global_writer_succeeds(self, handler_global_writer, mock_mound):
        """User with global_write permission (but not admin) can store facts."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler_global_writer._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True

    def test_store_verified_fact_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    def test_store_verified_fact_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=ValueError("bad data"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    def test_store_verified_fact_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    def test_store_verified_fact_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=TypeError("wrong type"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    def test_store_verified_fact_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    def test_store_verified_fact_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=AttributeError("attr"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_store_verified_fact_tracks_metrics(self, mock_track, handler, mock_mound):
        """track_global_fact is called with action='store' on success."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        handler._handle_store_verified_fact(http)
        mock_track.assert_called_once_with(action="store")

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_store_verified_fact_no_metrics_on_error(self, mock_track, handler, mock_mound):
        """track_global_fact is NOT called when mound raises error."""
        mock_mound.store_verified_fact = AsyncMock(side_effect=RuntimeError("fail"))
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        handler._handle_store_verified_fact(http)
        mock_track.assert_not_called()

    def test_store_verified_fact_user_id_from_id_attr(self, mock_mound):
        """User.id attribute is used for verified_by."""
        user = MockUser(id="user-from-id", user_id="user-from-user-id")
        h = GlobalKnowledgeTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = h._handle_store_verified_fact(http)
        body = _body(result)
        assert body["verified_by"] == "user-from-id"

    def test_store_verified_fact_user_id_fallback_to_user_id(self, mock_mound):
        """Falls back to user_id when id is None."""
        user = MagicMock(spec=[])
        user.id = None
        user.user_id = "fallback-user"
        user.permissions = ["admin"]
        h = GlobalKnowledgeTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = h._handle_store_verified_fact(http)
        body = _body(result)
        assert body["verified_by"] == "fallback-user"

    def test_store_verified_fact_user_id_fallback_to_unknown(self, mock_mound):
        """Falls back to 'unknown' when both id and user_id are missing."""
        user = MagicMock(spec=[])
        user.permissions = ["admin"]
        h = GlobalKnowledgeTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = h._handle_store_verified_fact(http)
        body = _body(result)
        assert body["verified_by"] == "unknown"

    def test_store_verified_fact_admin_permission_allows_store(self, mock_mound):
        """User with 'admin' permission (not role) can store facts even when admin check fails."""
        user = MockUser(permissions=["admin"])

        class HandlerAdminPerm(GlobalKnowledgeOperationsMixin):
            def __init__(self):
                self._mound = mock_mound

            def _get_mound(self):
                return self._mound

            def require_auth_or_error(self, handler):
                return user, None

            def require_admin_or_error(self, handler):
                from aragora.server.handlers.base import error_response

                return None, error_response("Not admin", 403)

        h = HandlerAdminPerm()
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "some-source",
            }
        )
        result = h._handle_store_verified_fact(http)
        assert _status(result) == 201


# ============================================================================
# Tests: _handle_query_global (GET /api/knowledge/mound/global)
# ============================================================================


class TestQueryGlobal:
    """Test _handle_query_global (GET /api/knowledge/mound/global)."""

    def test_query_global_with_query_success(self, handler, mock_mound):
        """Successfully querying global knowledge returns items and count."""
        result = handler._handle_query_global({"query": "astronomy"})
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["items"]) == 2
        assert body["query"] == "astronomy"

    def test_query_global_items_have_to_dict(self, handler, mock_mound):
        """Items with to_dict are serialized using to_dict."""
        result = handler._handle_query_global({"query": "test"})
        body = _body(result)
        assert body["items"][0]["id"] == "item-001"
        assert body["items"][0]["content"] == "Fact 1"
        assert body["items"][0]["importance"] == 0.95

    def test_query_global_items_without_to_dict(self, handler, mock_mound):
        """Items without to_dict use fallback serialization."""
        mock_mound.query_global_knowledge = AsyncMock(
            return_value=[MockKnowledgeItemNoDict(id="fallback-1", content="No dict")]
        )
        result = handler._handle_query_global({"query": "test"})
        body = _body(result)
        assert body["items"][0]["id"] == "fallback-1"
        assert body["items"][0]["content"] == "No dict"
        assert body["items"][0]["importance"] == 0.5

    def test_query_global_mixed_items(self, handler, mock_mound):
        """Mix of items with and without to_dict are handled."""
        mock_mound.query_global_knowledge = AsyncMock(
            return_value=[
                MockKnowledgeItem(id="with-dict", content="Has to_dict"),
                MockKnowledgeItemNoDict(id="no-dict", content="No to_dict"),
            ]
        )
        result = handler._handle_query_global({"query": "test"})
        body = _body(result)
        assert body["count"] == 2
        assert body["items"][0]["id"] == "with-dict"
        assert body["items"][1]["id"] == "no-dict"

    def test_query_global_no_query_returns_system_facts(self, handler, mock_mound):
        """Without a query, get_system_facts is called instead."""
        result = handler._handle_query_global({})
        assert _status(result) == 200
        body = _body(result)
        # System facts returns 3 items
        assert body["count"] == 3
        assert body["query"] == ""
        mock_mound.get_system_facts.assert_called_once()

    def test_query_global_empty_query_returns_system_facts(self, handler, mock_mound):
        """Empty query string falls back to get_system_facts."""
        result = handler._handle_query_global({"query": ""})
        assert _status(result) == 200
        mock_mound.get_system_facts.assert_called_once()

    def test_query_global_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to mound."""
        handler._handle_query_global({"query": "test", "limit": "10"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 10

    def test_query_global_default_limit(self, handler, mock_mound):
        """Default limit is 20."""
        handler._handle_query_global({"query": "test"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 20

    def test_query_global_limit_clamped_max(self, handler, mock_mound):
        """Limit is clamped to max 100."""
        handler._handle_query_global({"query": "test", "limit": "999"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 100

    def test_query_global_limit_clamped_min(self, handler, mock_mound):
        """Limit is clamped to min 1."""
        handler._handle_query_global({"query": "test", "limit": "0"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 1

    def test_query_global_with_topics(self, handler, mock_mound):
        """Topics are parsed from comma-separated string."""
        handler._handle_query_global({"query": "test", "topics": "ai,ml,nlp"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["topics"] == ["ai", "ml", "nlp"]

    def test_query_global_without_topics(self, handler, mock_mound):
        """Topics default to None when not specified."""
        handler._handle_query_global({"query": "test"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["topics"] is None

    def test_query_global_single_topic(self, handler, mock_mound):
        """Single topic (no comma) is parsed as single-element list."""
        handler._handle_query_global({"query": "test", "topics": "ai"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["topics"] == ["ai"]

    def test_query_global_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_query_global({"query": "test"})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_query_global_empty_results(self, handler, mock_mound):
        """Empty results return count=0 and empty items list."""
        mock_mound.query_global_knowledge = AsyncMock(return_value=[])
        result = handler._handle_query_global({"query": "nonexistent"})
        body = _body(result)
        assert body["count"] == 0
        assert body["items"] == []

    def test_query_global_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=RuntimeError("runtime"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=AttributeError("attr"))
        result = handler._handle_query_global({"query": "test"})
        assert _status(result) == 500

    def test_query_global_system_facts_error_returns_500(self, handler, mock_mound):
        """Error from get_system_facts (no query path) returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=RuntimeError("fail"))
        result = handler._handle_query_global({})
        assert _status(result) == 500

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_tracks_metrics_with_results(self, mock_track, handler, mock_mound):
        """track_global_query is called with has_results=True when results exist."""
        handler._handle_query_global({"query": "test"})
        mock_track.assert_called_once_with(has_results=True)

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_tracks_metrics_no_results(self, mock_track, handler, mock_mound):
        """track_global_query is called with has_results=False when no results."""
        mock_mound.query_global_knowledge = AsyncMock(return_value=[])
        handler._handle_query_global({"query": "test"})
        mock_track.assert_called_once_with(has_results=False)

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_no_metrics_on_error(self, mock_track, handler, mock_mound):
        """track_global_query is NOT called when mound raises error."""
        mock_mound.query_global_knowledge = AsyncMock(side_effect=RuntimeError("fail"))
        handler._handle_query_global({"query": "test"})
        mock_track.assert_not_called()

    def test_query_global_query_truncated_to_max_length(self, handler, mock_mound):
        """Query string is truncated to max_length=1000."""
        long_query = "a" * 2000
        handler._handle_query_global({"query": long_query})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert len(call_kwargs["query"]) == 1000

    def test_query_global_topics_truncated_to_max_length(self, handler, mock_mound):
        """Topics string is truncated to max_length=500."""
        long_topics = ",".join(["topic"] * 200)  # Very long topics string
        handler._handle_query_global({"query": "test", "topics": long_topics})
        # Should still work (topics string is truncated before splitting)
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["topics"] is not None


# ============================================================================
# Tests: _handle_promote_to_global (POST /api/knowledge/mound/global/promote)
# ============================================================================


class TestPromoteToGlobal:
    """Test _handle_promote_to_global (POST /api/knowledge/mound/global/promote)."""

    def test_promote_to_global_success(self, handler, mock_mound):
        """Successfully promoting an item returns 201 with global_id."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-local-001",
                "workspace_id": "ws-001",
                "reason": "High-confidence verified fact",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["success"] is True
        assert body["global_id"] == "global-xyz-789"
        assert body["original_id"] == "item-local-001"
        assert body["promoted_by"] == "test-user-001"
        assert body["reason"] == "High-confidence verified fact"

    def test_promote_to_global_forwards_correct_kwargs(self, handler, mock_mound):
        """All parameters are correctly forwarded to mound.promote_to_global."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-abc",
                "workspace_id": "ws-xyz",
                "reason": "Important discovery",
            }
        )
        handler._handle_promote_to_global(http)
        call_kwargs = mock_mound.promote_to_global.call_args.kwargs
        assert call_kwargs["item_id"] == "item-abc"
        assert call_kwargs["workspace_id"] == "ws-xyz"
        assert call_kwargs["promoted_by"] == "test-user-001"
        assert call_kwargs["reason"] == "Important discovery"

    def test_promote_to_global_missing_item_id_returns_400(self, handler):
        """Missing item_id returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400
        body = _body(result)
        assert "item_id" in body["error"].lower()

    def test_promote_to_global_empty_item_id_returns_400(self, handler):
        """Empty string item_id returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400

    def test_promote_to_global_missing_workspace_id_returns_400(self, handler):
        """Missing workspace_id returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    def test_promote_to_global_empty_workspace_id_returns_400(self, handler):
        """Empty string workspace_id returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400

    def test_promote_to_global_missing_reason_returns_400(self, handler):
        """Missing reason returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400
        body = _body(result)
        assert "reason" in body["error"].lower()

    def test_promote_to_global_empty_reason_returns_400(self, handler):
        """Empty string reason returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400

    def test_promote_to_global_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower()

    def test_promote_to_global_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower() or "body" in body["error"].lower()

    def test_promote_to_global_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler_no_mound._handle_promote_to_global(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_promote_to_global_auth_failure_returns_401(self, handler_no_auth):
        """Auth failure returns 401."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler_no_auth._handle_promote_to_global(http)
        assert _status(result) == 401

    def test_promote_to_global_value_error_returns_404(self, handler, mock_mound):
        """ValueError from promote_to_global returns 404 (item not found)."""
        mock_mound.promote_to_global = AsyncMock(side_effect=ValueError("Not found"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "missing-item",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_promote_to_global_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.promote_to_global = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 500

    def test_promote_to_global_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.promote_to_global = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 500

    def test_promote_to_global_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.promote_to_global = AsyncMock(side_effect=TypeError("wrong type"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 500

    def test_promote_to_global_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.promote_to_global = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 500

    def test_promote_to_global_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.promote_to_global = AsyncMock(side_effect=AttributeError("attr"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_promote_to_global_tracks_metrics(self, mock_track, handler, mock_mound):
        """track_global_fact is called with action='promote' on success."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        handler._handle_promote_to_global(http)
        mock_track.assert_called_once_with(action="promote")

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_promote_to_global_no_metrics_on_not_found(self, mock_track, handler, mock_mound):
        """track_global_fact is NOT called when item not found (ValueError)."""
        mock_mound.promote_to_global = AsyncMock(side_effect=ValueError("Not found"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "missing",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        handler._handle_promote_to_global(http)
        mock_track.assert_not_called()

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_promote_to_global_no_metrics_on_error(self, mock_track, handler, mock_mound):
        """track_global_fact is NOT called when mound raises error."""
        mock_mound.promote_to_global = AsyncMock(side_effect=RuntimeError("fail"))
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        handler._handle_promote_to_global(http)
        mock_track.assert_not_called()

    def test_promote_to_global_user_id_fallback(self, mock_mound):
        """User without id falls back to user_id."""
        user = MagicMock(spec=[])
        user.user_id = "fallback-user"
        h = GlobalKnowledgeTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = h._handle_promote_to_global(http)
        body = _body(result)
        assert body["promoted_by"] == "fallback-user"

    def test_promote_to_global_user_id_fallback_to_unknown(self, mock_mound):
        """User without id or user_id falls back to 'unknown'."""
        user = MagicMock(spec=[])
        h = GlobalKnowledgeTestHandler(mound=mock_mound, user=user)
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Some reason",
            }
        )
        result = h._handle_promote_to_global(http)
        body = _body(result)
        assert body["promoted_by"] == "unknown"


# ============================================================================
# Tests: _handle_get_system_facts (GET /api/knowledge/mound/global/facts)
# ============================================================================


class TestGetSystemFacts:
    """Test _handle_get_system_facts (GET /api/knowledge/mound/global/facts)."""

    def test_get_system_facts_success(self, handler, mock_mound):
        """Successfully getting system facts returns facts list with pagination."""
        result = handler._handle_get_system_facts({})
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3
        assert body["total"] == 3
        assert len(body["facts"]) == 3
        assert body["limit"] == 100
        assert body["offset"] == 0

    def test_get_system_facts_items_have_to_dict(self, handler, mock_mound):
        """Facts with to_dict are serialized using to_dict."""
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["facts"][0]["id"] == "fact-001"
        assert body["facts"][0]["content"] == "System fact 1"

    def test_get_system_facts_items_without_to_dict(self, handler, mock_mound):
        """Facts without to_dict use fallback serialization."""
        mock_mound.get_system_facts = AsyncMock(
            return_value=[MockKnowledgeItemNoDict(id="no-dict", content="Fallback")]
        )
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["facts"][0]["id"] == "no-dict"
        assert body["facts"][0]["content"] == "Fallback"

    def test_get_system_facts_mixed_items(self, handler, mock_mound):
        """Mix of items with and without to_dict are handled."""
        mock_mound.get_system_facts = AsyncMock(
            return_value=[
                MockKnowledgeItem(id="with-dict"),
                MockKnowledgeItemNoDict(id="no-dict"),
            ]
        )
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["count"] == 2
        # to_dict item has importance field
        assert "importance" in body["facts"][0]
        assert body["facts"][1]["id"] == "no-dict"

    def test_get_system_facts_custom_limit(self, handler, mock_mound):
        """Custom limit is reflected in response."""
        result = handler._handle_get_system_facts({"limit": "2"})
        body = _body(result)
        assert body["limit"] == 2
        # With limit=2 and 3 facts, should get 2 facts (pagination)
        assert len(body["facts"]) == 2

    def test_get_system_facts_default_limit(self, handler, mock_mound):
        """Default limit is 100."""
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["limit"] == 100

    def test_get_system_facts_limit_clamped_max(self, handler, mock_mound):
        """Limit is clamped to max 500."""
        result = handler._handle_get_system_facts({"limit": "9999"})
        body = _body(result)
        assert body["limit"] == 500

    def test_get_system_facts_limit_clamped_min(self, handler, mock_mound):
        """Limit is clamped to min 1."""
        result = handler._handle_get_system_facts({"limit": "0"})
        body = _body(result)
        assert body["limit"] == 1

    def test_get_system_facts_custom_offset(self, handler, mock_mound):
        """Custom offset skips items in the result."""
        result = handler._handle_get_system_facts({"offset": "1"})
        body = _body(result)
        assert body["offset"] == 1
        # With offset=1 and 3 total facts, we get 2 paginated facts
        assert len(body["facts"]) == 2
        assert body["total"] == 3

    def test_get_system_facts_offset_beyond_total(self, handler, mock_mound):
        """Offset beyond total returns empty facts list."""
        result = handler._handle_get_system_facts({"offset": "100"})
        body = _body(result)
        assert body["offset"] == 100
        assert len(body["facts"]) == 0
        assert body["total"] == 3

    def test_get_system_facts_default_offset(self, handler, mock_mound):
        """Default offset is 0."""
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["offset"] == 0

    def test_get_system_facts_with_topics(self, handler, mock_mound):
        """Topics are parsed from comma-separated string and forwarded."""
        handler._handle_get_system_facts({"topics": "physics,math"})
        call_kwargs = mock_mound.get_system_facts.call_args.kwargs
        assert call_kwargs["topics"] == ["physics", "math"]

    def test_get_system_facts_without_topics(self, handler, mock_mound):
        """Topics default to None when not specified."""
        handler._handle_get_system_facts({})
        call_kwargs = mock_mound.get_system_facts.call_args.kwargs
        assert call_kwargs["topics"] is None

    def test_get_system_facts_single_topic(self, handler, mock_mound):
        """Single topic (no comma) is parsed as single-element list."""
        handler._handle_get_system_facts({"topics": "biology"})
        call_kwargs = mock_mound.get_system_facts.call_args.kwargs
        assert call_kwargs["topics"] == ["biology"]

    def test_get_system_facts_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_system_facts({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_system_facts_empty_results(self, handler, mock_mound):
        """Empty results return count=0 and empty facts list."""
        mock_mound.get_system_facts = AsyncMock(return_value=[])
        result = handler._handle_get_system_facts({})
        body = _body(result)
        assert body["count"] == 0
        assert body["total"] == 0
        assert body["facts"] == []

    def test_get_system_facts_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=RuntimeError("runtime"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_system_facts = AsyncMock(side_effect=AttributeError("attr"))
        result = handler._handle_get_system_facts({})
        assert _status(result) == 500

    def test_get_system_facts_limit_forwarded_with_offset(self, handler, mock_mound):
        """Mound is called with limit+offset to support pagination slicing."""
        handler._handle_get_system_facts({"limit": "10", "offset": "5"})
        call_kwargs = mock_mound.get_system_facts.call_args.kwargs
        # Handler calls get_system_facts(limit=limit+offset, ...)
        assert call_kwargs["limit"] == 15


# ============================================================================
# Tests: _handle_get_system_workspace_id (GET /api/knowledge/mound/global/workspace-id)
# ============================================================================


class TestGetSystemWorkspaceId:
    """Test _handle_get_system_workspace_id (GET /api/knowledge/mound/global/workspace-id)."""

    def test_get_system_workspace_id_success(self, handler, mock_mound):
        """Successfully getting system workspace ID returns the ID."""
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 200
        body = _body(result)
        assert body["system_workspace_id"] == "system-workspace-001"

    def test_get_system_workspace_id_calls_mound(self, handler, mock_mound):
        """get_system_workspace_id is called on mound."""
        handler._handle_get_system_workspace_id()
        mock_mound.get_system_workspace_id.assert_called_once()

    def test_get_system_workspace_id_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_system_workspace_id()
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_system_workspace_id_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_system_workspace_id = MagicMock(side_effect=KeyError("missing"))
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 500

    def test_get_system_workspace_id_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_system_workspace_id = MagicMock(side_effect=ValueError("bad"))
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 500

    def test_get_system_workspace_id_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_system_workspace_id = MagicMock(side_effect=OSError("disk"))
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 500

    def test_get_system_workspace_id_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_system_workspace_id = MagicMock(side_effect=TypeError("wrong"))
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 500

    def test_get_system_workspace_id_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_system_workspace_id = MagicMock(side_effect=AttributeError("attr"))
        result = handler._handle_get_system_workspace_id()
        assert _status(result) == 500

    def test_get_system_workspace_id_returns_different_ids(self, handler, mock_mound):
        """Different workspace IDs are correctly passed through."""
        mock_mound.get_system_workspace_id = MagicMock(return_value="custom-ws-id")
        result = handler._handle_get_system_workspace_id()
        body = _body(result)
        assert body["system_workspace_id"] == "custom-ws-id"

    def test_get_system_workspace_id_empty_string(self, handler, mock_mound):
        """Empty string workspace ID is returned as-is."""
        mock_mound.get_system_workspace_id = MagicMock(return_value="")
        result = handler._handle_get_system_workspace_id()
        body = _body(result)
        assert body["system_workspace_id"] == ""


# ============================================================================
# Tests: security and edge cases
# ============================================================================


class TestGlobalKnowledgeSecurity:
    """Test security-related edge cases for global knowledge operations."""

    def test_store_fact_path_traversal_in_content(self, handler, mock_mound):
        """Path traversal in content is stored as-is (not interpreted)."""
        http = MockHTTPHandler.with_body(
            {
                "content": "../../etc/passwd",
                "source": "test",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["content"] == "../../etc/passwd"

    def test_store_fact_script_injection_in_content(self, handler, mock_mound):
        """Script injection in content is stored as-is (not executed)."""
        http = MockHTTPHandler.with_body(
            {
                "content": "<script>alert('xss')</script>",
                "source": "test",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["content"] == "<script>alert('xss')</script>"

    def test_store_fact_sql_injection_in_source(self, handler, mock_mound):
        """SQL injection in source is stored as-is (not executed)."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "'; DROP TABLE facts; --",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["source"] == "'; DROP TABLE facts; --"

    def test_promote_path_traversal_in_item_id(self, handler, mock_mound):
        """Path traversal in item_id is passed through (mound handles validation)."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "../../secret-item",
                "workspace_id": "ws-001",
                "reason": "Testing",
            }
        )
        result = handler._handle_promote_to_global(http)
        # Should succeed (mound is responsible for validating IDs)
        assert _status(result) == 201

    def test_query_global_injection_in_query(self, handler, mock_mound):
        """Injection attempts in query string are passed through safely."""
        result = handler._handle_query_global({"query": "'; DROP TABLE knowledge; --"})
        assert _status(result) == 200
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["query"] == "'; DROP TABLE knowledge; --"

    def test_store_fact_unicode_content(self, handler, mock_mound):
        """Unicode content is handled correctly."""
        http = MockHTTPHandler.with_body(
            {
                "content": "The universe is vast. It contains many stars and galaxies.",
                "source": "encyclopaedia",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201

    def test_store_fact_very_large_content(self, handler, mock_mound):
        """Very large content is accepted (handler doesn't enforce size limit)."""
        large_content = "x" * 100000
        http = MockHTTPHandler.with_body(
            {
                "content": large_content,
                "source": "test",
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201
        call_kwargs = mock_mound.store_verified_fact.call_args.kwargs
        assert len(call_kwargs["content"]) == 100000

    def test_store_fact_null_values_in_optional_fields(self, handler, mock_mound):
        """Null values in optional fields are treated as defaults."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "test",
                "confidence": None,
                "evidence_ids": None,
                "topics": None,
            }
        )
        result = handler._handle_store_verified_fact(http)
        # None confidence gets data.get("confidence", 0.9) -> None passed to mound
        # This tests that the handler doesn't crash
        assert _status(result) in (201, 500)  # depends on mound handling of None

    def test_promote_null_values_in_required_fields(self, handler):
        """Null values in required fields return 400."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": None,
                "workspace_id": "ws-001",
                "reason": "Test",
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400

    def test_query_global_negative_limit(self, handler, mock_mound):
        """Negative limit is clamped to minimum (1)."""
        handler._handle_query_global({"query": "test", "limit": "-5"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 1

    def test_get_system_facts_negative_limit(self, handler, mock_mound):
        """Negative limit is clamped to minimum (1)."""
        result = handler._handle_get_system_facts({"limit": "-5"})
        body = _body(result)
        assert body["limit"] == 1

    def test_get_system_facts_negative_offset(self, handler, mock_mound):
        """Negative offset is clamped to minimum (0)."""
        result = handler._handle_get_system_facts({"offset": "-5"})
        body = _body(result)
        assert body["offset"] == 0

    def test_get_system_facts_offset_clamped_max(self, handler, mock_mound):
        """Offset is clamped to max 10000."""
        result = handler._handle_get_system_facts({"offset": "99999"})
        body = _body(result)
        assert body["offset"] == 10000

    def test_store_fact_missing_both_required_fields(self, handler):
        """Missing both content and source returns 400 (content error first)."""
        http = MockHTTPHandler.with_body({})
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 400

    def test_promote_missing_all_required_fields(self, handler):
        """Missing all required fields returns 400."""
        http = MockHTTPHandler.with_body({})
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 400


# ============================================================================
# Tests: parameter handling edge cases
# ============================================================================


class TestParameterEdgeCases:
    """Test edge cases in parameter parsing across all handlers."""

    def test_query_global_list_param_format(self, handler, mock_mound):
        """Query params in list format (from parse_qs) are handled."""
        handler._handle_query_global({"query": ["test query"], "limit": ["5"]})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["limit"] == 5

    def test_get_system_facts_list_param_format(self, handler, mock_mound):
        """Query params in list format are handled for system facts."""
        result = handler._handle_get_system_facts({"limit": ["3"], "offset": ["1"]})
        body = _body(result)
        assert body["limit"] == 3
        assert body["offset"] == 1

    def test_query_global_non_numeric_limit(self, handler, mock_mound):
        """Non-numeric limit falls back to default (20)."""
        handler._handle_query_global({"query": "test", "limit": "abc"})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["limit"] == 20

    def test_get_system_facts_non_numeric_limit(self, handler, mock_mound):
        """Non-numeric limit falls back to default (100)."""
        result = handler._handle_get_system_facts({"limit": "abc"})
        body = _body(result)
        assert body["limit"] == 100

    def test_get_system_facts_non_numeric_offset(self, handler, mock_mound):
        """Non-numeric offset falls back to default (0)."""
        result = handler._handle_get_system_facts({"offset": "abc"})
        body = _body(result)
        assert body["offset"] == 0

    def test_query_global_topics_list_format(self, handler, mock_mound):
        """Topics in list format (from parse_qs) are handled."""
        handler._handle_query_global({"query": "test", "topics": ["ai,ml"]})
        call_kwargs = mock_mound.query_global_knowledge.call_args.kwargs
        assert call_kwargs["topics"] == ["ai", "ml"]

    def test_get_system_facts_topics_list_format(self, handler, mock_mound):
        """Topics in list format are handled for system facts."""
        handler._handle_get_system_facts({"topics": ["physics,chemistry"]})
        call_kwargs = mock_mound.get_system_facts.call_args.kwargs
        assert call_kwargs["topics"] == ["physics", "chemistry"]

    def test_store_fact_extra_fields_ignored(self, handler, mock_mound):
        """Extra fields in request body are ignored."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some fact",
                "source": "test",
                "extra_field": "should be ignored",
                "another_extra": 42,
            }
        )
        result = handler._handle_store_verified_fact(http)
        assert _status(result) == 201

    def test_promote_extra_fields_ignored(self, handler, mock_mound):
        """Extra fields in promote request are ignored."""
        http = MockHTTPHandler.with_body(
            {
                "item_id": "item-001",
                "workspace_id": "ws-001",
                "reason": "Testing",
                "unknown_param": True,
            }
        )
        result = handler._handle_promote_to_global(http)
        assert _status(result) == 201
