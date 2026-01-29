"""Tests for CultureOperationsMixin."""

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
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.culture import (
    CultureOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockPattern:
    """Mock culture pattern."""

    name: str
    frequency: float = 0.5
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frequency": self.frequency,
            "description": self.description,
        }


@dataclass
class MockCultureProfile:
    """Mock culture profile."""

    workspace_id: str
    patterns: Dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[datetime] = None
    total_observations: int = 0


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    get_culture_profile: AsyncMock = field(default_factory=AsyncMock)
    add_node: AsyncMock = field(default_factory=AsyncMock)
    update: AsyncMock = field(default_factory=AsyncMock)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: Optional[Dict[str, str]] = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class CultureHandler(CultureOperationsMixin):
    """Handler implementation for testing CultureOperationsMixin."""

    def __init__(self, mound: Optional[MockKnowledgeMound] = None):
        self._mound = mound
        self.ctx = {}

    def _get_mound(self):
        return self._mound


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


@pytest.fixture
def handler(mock_mound):
    """Create a test handler with mock mound."""
    return CultureHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return CultureHandler(mound=None)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test get_culture
# =============================================================================


class TestGetCulture:
    """Tests for get culture endpoint."""

    def test_get_culture_success(self, handler, mock_mound):
        """Test successful culture retrieval."""
        profile = MockCultureProfile(
            workspace_id="ws-123",
            patterns={"communication": MockPattern(name="formal", frequency=0.8)},
            generated_at=datetime(2026, 1, 27, 12, 0, 0),
            total_observations=100,
        )
        mock_mound.get_culture_profile.return_value = profile

        result = handler._handle_get_culture({"workspace_id": ["ws-123"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["workspace_id"] == "ws-123"
        assert data["total_observations"] == 100

    def test_get_culture_default_workspace(self, handler, mock_mound):
        """Test culture retrieval with default workspace."""
        profile = MockCultureProfile(workspace_id="default", total_observations=50)
        mock_mound.get_culture_profile.return_value = profile

        result = handler._handle_get_culture({})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["workspace_id"] == "default"

    def test_get_culture_no_mound(self, handler_no_mound):
        """Test culture retrieval when mound not available."""
        result = handler_no_mound._handle_get_culture({})

        assert result.status_code == 503
        assert "not available" in parse_response(result)["error"]

    def test_get_culture_error(self, handler, mock_mound):
        """Test culture retrieval error handling."""
        mock_mound.get_culture_profile.side_effect = Exception("Database error")

        result = handler._handle_get_culture({})

        assert result.status_code == 500
        assert "Failed" in parse_response(result)["error"]


# =============================================================================
# Test add_culture_document
# =============================================================================


class TestAddCultureDocument:
    """Tests for add culture document endpoint."""

    def test_add_document_success(self, handler, mock_mound):
        """Test successful culture document addition."""
        mock_mound.add_node.return_value = "node-123"

        body = json.dumps(
            {
                "content": "Our company values include transparency and collaboration.",
                "workspace_id": "ws-123",
                "document_type": "values",
                "metadata": {"source": "handbook"},
            }
        ).encode()

        http_handler = MockHandler(body=body)
        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 201
        data = parse_response(result)
        assert data["node_id"] == "node-123"
        assert data["document_type"] == "values"
        assert data["workspace_id"] == "ws-123"

    def test_add_document_default_type(self, handler, mock_mound):
        """Test document addition with default type."""
        mock_mound.add_node.return_value = "node-456"

        body = json.dumps({"content": "Policy content"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 201
        assert parse_response(result)["document_type"] == "policy"

    def test_add_document_missing_content(self, handler):
        """Test document addition with missing content."""
        body = json.dumps({"workspace_id": "ws-123"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_add_document_empty_body(self, handler):
        """Test document addition with empty body."""
        http_handler = MockHandler(body=b"")

        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_add_document_invalid_json(self, handler):
        """Test document addition with invalid JSON."""
        http_handler = MockHandler(body=b"not json")

        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 400
        assert "Invalid JSON" in parse_response(result)["error"]

    def test_add_document_no_mound(self, handler_no_mound):
        """Test document addition when mound not available."""
        body = json.dumps({"content": "Test content"}).encode()
        http_handler = MockHandler(body=body)

        result = handler_no_mound._handle_add_culture_document(http_handler)

        assert result.status_code == 503

    def test_add_document_error(self, handler, mock_mound):
        """Test document addition error handling."""
        mock_mound.add_node.side_effect = Exception("Storage error")

        body = json.dumps({"content": "Test content"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_add_culture_document(http_handler)

        assert result.status_code == 500


# =============================================================================
# Test promote_to_culture
# =============================================================================


class TestPromoteToCulture:
    """Tests for promote to culture endpoint."""

    def test_promote_success(self, handler, mock_mound):
        """Test successful culture promotion."""
        mock_mound.update.return_value = True

        body = json.dumps({"node_id": "node-123"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_promote_to_culture(http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["node_id"] == "node-123"
        assert data["promoted"] is True

    def test_promote_missing_node_id(self, handler):
        """Test promotion with missing node_id."""
        body = json.dumps({}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_promote_to_culture(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_promote_node_not_found(self, handler, mock_mound):
        """Test promotion when node not found."""
        mock_mound.update.return_value = False

        body = json.dumps({"node_id": "nonexistent"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_promote_to_culture(http_handler)

        assert result.status_code == 404

    def test_promote_empty_body(self, handler):
        """Test promotion with empty body."""
        http_handler = MockHandler(body=b"")

        result = handler._handle_promote_to_culture(http_handler)

        assert result.status_code == 400

    def test_promote_no_mound(self, handler_no_mound):
        """Test promotion when mound not available."""
        body = json.dumps({"node_id": "node-123"}).encode()
        http_handler = MockHandler(body=body)

        result = handler_no_mound._handle_promote_to_culture(http_handler)

        assert result.status_code == 503

    def test_promote_error(self, handler, mock_mound):
        """Test promotion error handling."""
        mock_mound.update.side_effect = Exception("Update failed")

        body = json.dumps({"node_id": "node-123"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_promote_to_culture(http_handler)

        assert result.status_code == 500
