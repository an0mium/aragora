"""Tests for GlobalKnowledgeOperationsMixin."""

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
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.global_knowledge import (
    GlobalKnowledgeOperationsMixin,
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
    permissions: list[str] = field(default_factory=lambda: ["global_write"])


@dataclass
class MockFact:
    """Mock fact object."""

    id: str
    content: str
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "importance": self.importance}


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    store_verified_fact: AsyncMock = field(default_factory=AsyncMock)
    query_global_knowledge: AsyncMock = field(default_factory=AsyncMock)
    get_system_facts: AsyncMock = field(default_factory=AsyncMock)
    promote_to_global: AsyncMock = field(default_factory=AsyncMock)
    get_system_workspace_id: MagicMock = field(
        default_factory=lambda: MagicMock(return_value="system-ws-id")
    )


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class GlobalKnowledgeHandler(GlobalKnowledgeOperationsMixin):
    """Handler implementation for testing GlobalKnowledgeOperationsMixin."""

    def __init__(
        self,
        mound: MockKnowledgeMound | None = None,
        user: MockUser | None = None,
        is_admin: bool = False,
    ):
        self._mound = mound
        self._user = user or MockUser()
        self._is_admin = is_admin
        self.ctx = {}

    def _get_mound(self):
        return self._mound

    def require_auth_or_error(self, handler):
        return self._user, None

    def require_admin_or_error(self, handler):
        if self._is_admin:
            return self._user, None
        return None, ("error", 403)


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
    return GlobalKnowledgeHandler(mound=mock_mound, user=mock_user)


@pytest.fixture
def admin_handler(mock_mound, mock_user):
    """Create a test handler with admin privileges."""
    return GlobalKnowledgeHandler(mound=mock_mound, user=mock_user, is_admin=True)


@pytest.fixture
def handler_no_mound(mock_user):
    """Create a test handler without mound."""
    return GlobalKnowledgeHandler(mound=None, user=mock_user)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test store_verified_fact
# =============================================================================


class TestStoreVerifiedFact:
    """Tests for store verified fact endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_store_fact_success(self, mock_track, handler, mock_mound):
        """Test successful fact storage."""
        mock_mound.store_verified_fact.return_value = "node-123"

        body = json.dumps(
            {
                "content": "The sky is blue.",
                "source": "observation",
                "confidence": 0.95,
                "topics": ["nature", "science"],
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_store_verified_fact(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["success"] is True
        assert data["node_id"] == "node-123"

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_store_fact_as_admin(self, mock_track, admin_handler, mock_mound):
        """Test storing fact as admin."""
        mock_mound.store_verified_fact.return_value = "node-456"

        body = json.dumps(
            {
                "content": "Admin verified fact.",
                "source": "admin",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = admin_handler._handle_store_verified_fact(http_handler)

        assert result.status_code == 201

    def test_store_fact_missing_content(self, handler):
        """Test storing fact without content."""
        body = json.dumps({"source": "test"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_store_verified_fact(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_store_fact_missing_source(self, handler):
        """Test storing fact without source."""
        body = json.dumps({"content": "Test content"}).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_store_verified_fact(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_store_fact_no_mound(self, handler_no_mound):
        """Test storing fact when mound not available."""
        body = json.dumps(
            {
                "content": "Test fact",
                "source": "test",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler_no_mound._handle_store_verified_fact(http_handler)

        assert result.status_code == 503


# =============================================================================
# Test query_global
# =============================================================================


class TestQueryGlobal:
    """Tests for query global endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_success(self, mock_track, handler, mock_mound):
        """Test successful global query."""
        facts = [
            MockFact(id="fact-1", content="Fact one", importance=0.8),
            MockFact(id="fact-2", content="Fact two", importance=0.6),
        ]
        mock_mound.query_global_knowledge.return_value = facts

        result = handler._handle_query_global({"query": ["test query"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 2
        assert data["query"] == "test query"

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_empty_query(self, mock_track, handler, mock_mound):
        """Test global query without search term (returns system facts)."""
        facts = [MockFact(id="fact-1", content="System fact")]
        mock_mound.get_system_facts.return_value = facts

        result = handler._handle_query_global({})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 1

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_query")
    def test_query_global_with_topics(self, mock_track, handler, mock_mound):
        """Test global query with topic filter."""
        mock_mound.query_global_knowledge.return_value = []

        result = handler._handle_query_global(
            {"query": ["science"], "topics": ["physics,chemistry"]}
        )

        assert result.status_code == 200

    def test_query_global_no_mound(self, handler_no_mound):
        """Test global query when mound not available."""
        result = handler_no_mound._handle_query_global({"query": ["test"]})

        assert result.status_code == 503


# =============================================================================
# Test promote_to_global
# =============================================================================


class TestPromoteToGlobal:
    """Tests for promote to global endpoint."""

    @patch("aragora.server.handlers.knowledge_base.mound.global_knowledge.track_global_fact")
    def test_promote_success(self, mock_track, handler, mock_mound):
        """Test successful promotion to global."""
        mock_mound.promote_to_global.return_value = "global-123"

        body = json.dumps(
            {
                "item_id": "item-123",
                "workspace_id": "ws-456",
                "reason": "Universally applicable knowledge",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_promote_to_global(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["success"] is True
        assert data["global_id"] == "global-123"

    def test_promote_missing_item_id(self, handler):
        """Test promotion without item_id."""
        body = json.dumps(
            {
                "workspace_id": "ws-456",
                "reason": "Test reason",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_promote_to_global(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_promote_missing_workspace_id(self, handler):
        """Test promotion without workspace_id."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "reason": "Test reason",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_promote_to_global(http_handler)

        assert result.status_code == 400

    def test_promote_missing_reason(self, handler):
        """Test promotion without reason."""
        body = json.dumps(
            {
                "item_id": "item-123",
                "workspace_id": "ws-456",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_promote_to_global(http_handler)

        assert result.status_code == 400

    def test_promote_item_not_found(self, handler, mock_mound):
        """Test promotion when item not found."""
        mock_mound.promote_to_global.side_effect = ValueError("Item not found")

        body = json.dumps(
            {
                "item_id": "nonexistent",
                "workspace_id": "ws-456",
                "reason": "Test",
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_promote_to_global(http_handler)

        assert result.status_code == 404


# =============================================================================
# Test get_system_facts
# =============================================================================


class TestGetSystemFacts:
    """Tests for get system facts endpoint."""

    def test_get_facts_success(self, handler, mock_mound):
        """Test successful system facts retrieval."""
        facts = [
            MockFact(id="fact-1", content="System fact 1"),
            MockFact(id="fact-2", content="System fact 2"),
        ]
        mock_mound.get_system_facts.return_value = facts

        result = handler._handle_get_system_facts({})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 2
        assert len(data["facts"]) == 2

    def test_get_facts_with_pagination(self, handler, mock_mound):
        """Test system facts with pagination."""
        facts = [MockFact(id=f"fact-{i}", content=f"Fact {i}") for i in range(10)]
        mock_mound.get_system_facts.return_value = facts

        result = handler._handle_get_system_facts({"limit": ["5"], "offset": ["2"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["limit"] == 5
        assert data["offset"] == 2

    def test_get_facts_with_topics(self, handler, mock_mound):
        """Test system facts filtered by topics."""
        mock_mound.get_system_facts.return_value = []

        result = handler._handle_get_system_facts({"topics": ["science"]})

        assert result.status_code == 200

    def test_get_facts_no_mound(self, handler_no_mound):
        """Test system facts when mound not available."""
        result = handler_no_mound._handle_get_system_facts({})

        assert result.status_code == 503


# =============================================================================
# Test get_system_workspace_id
# =============================================================================


class TestGetSystemWorkspaceId:
    """Tests for get system workspace ID endpoint."""

    def test_get_workspace_id_success(self, handler, mock_mound):
        """Test successful system workspace ID retrieval."""
        result = handler._handle_get_system_workspace_id()

        assert result.status_code == 200
        data = parse_response(result)
        assert data["system_workspace_id"] == "system-ws-id"

    def test_get_workspace_id_no_mound(self, handler_no_mound):
        """Test system workspace ID when mound not available."""
        result = handler_no_mound._handle_get_system_workspace_id()

        assert result.status_code == 503
