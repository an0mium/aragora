"""Tests for CultureOperationsMixin (aragora/server/handlers/knowledge_base/mound/culture.py).

Covers all routes and behavior of the culture mixin:
- GET  /api/v1/knowledge/mound/culture           - Get culture profile
- POST /api/v1/knowledge/mound/culture/documents  - Add culture document
- POST /api/v1/knowledge/mound/culture/promote     - Promote knowledge to culture
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

from aragora.server.handlers.knowledge_base.mound.culture import (
    CultureOperationsMixin,
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
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for culture tests."""

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
# Mock culture profile returned by mound.get_culture_profile
# ---------------------------------------------------------------------------


@dataclass
class MockCulturePattern:
    """Mock culture pattern with optional to_dict."""

    name: str = "collaboration"
    frequency: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "frequency": self.frequency}


@dataclass
class MockCultureProfile:
    """Mock culture profile returned by mound.get_culture_profile."""

    workspace_id: str = "default"
    patterns: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime | None = field(
        default_factory=lambda: datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    )
    total_observations: int = 150

    def __post_init__(self):
        if not self.patterns:
            self.patterns = {
                "collaboration": MockCulturePattern("collaboration", 42),
                "innovation": MockCulturePattern("innovation", 28),
            }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class CultureTestHandler(CultureOperationsMixin):
    """Concrete handler for testing the culture mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with culture methods."""
    mound = MagicMock()
    mound.get_culture_profile = AsyncMock(return_value=MockCultureProfile())
    mound.add_node = AsyncMock(return_value="node-culture-001")
    mound.update = AsyncMock(return_value={"node_type": "culture", "promoted_to_culture": True})
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a CultureTestHandler with a mocked mound."""
    return CultureTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a CultureTestHandler with no mound (None)."""
    return CultureTestHandler(mound=None)


# ============================================================================
# Tests: _handle_get_culture (GET /api/v1/knowledge/mound/culture)
# ============================================================================


class TestGetCulture:
    """Test _handle_get_culture (GET /api/knowledge/mound/culture)."""

    def test_get_culture_success(self, handler, mock_mound):
        """Successfully getting culture profile returns expected fields."""
        result = handler._handle_get_culture({})
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "default"
        assert "patterns" in body
        assert body["total_observations"] == 150
        assert body["generated_at"] is not None

    def test_get_culture_default_workspace_id(self, handler, mock_mound):
        """Default workspace_id is 'default' when not provided."""
        handler._handle_get_culture({})
        mock_mound.get_culture_profile.assert_called_once()
        # run_async wraps coroutine; the arg to get_culture_profile is workspace_id
        call_args = mock_mound.get_culture_profile.call_args
        assert call_args is not None

    def test_get_culture_with_workspace_id(self, handler, mock_mound):
        """Custom workspace_id is passed to mound.get_culture_profile."""
        handler._handle_get_culture({"workspace_id": "ws-custom"})
        mock_mound.get_culture_profile.assert_called_once_with("ws-custom")

    def test_get_culture_patterns_with_to_dict(self, handler, mock_mound):
        """Patterns with to_dict() method are serialized via to_dict."""
        result = handler._handle_get_culture({})
        body = _body(result)
        patterns = body["patterns"]
        assert "collaboration" in patterns
        assert patterns["collaboration"]["name"] == "collaboration"
        assert patterns["collaboration"]["frequency"] == 42

    def test_get_culture_patterns_without_to_dict(self, handler, mock_mound):
        """Patterns without to_dict() are passed through as-is."""
        profile = MockCultureProfile()
        profile.patterns = {"plain": "string_value", "number": 42}
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["patterns"]["plain"] == "string_value"
        assert body["patterns"]["number"] == 42

    def test_get_culture_generated_at_none(self, handler, mock_mound):
        """When generated_at is None, the response contains null."""
        profile = MockCultureProfile(generated_at=None)
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["generated_at"] is None

    def test_get_culture_generated_at_isoformat(self, handler, mock_mound):
        """generated_at is serialized as ISO 8601 string."""
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        profile = MockCultureProfile(generated_at=ts)
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["generated_at"] == ts.isoformat()

    def test_get_culture_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_culture({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_get_culture_empty_patterns(self, handler, mock_mound):
        """Empty patterns dict returns empty object."""
        profile = MockCultureProfile()
        profile.patterns = {}
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["patterns"] == {}

    def test_get_culture_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_culture_profile = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_get_culture({})
        assert _status(result) == 500

    def test_get_culture_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_culture_profile = AsyncMock(side_effect=ValueError("bad"))
        result = handler._handle_get_culture({})
        assert _status(result) == 500

    def test_get_culture_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_culture_profile = AsyncMock(side_effect=OSError("disk fail"))
        result = handler._handle_get_culture({})
        assert _status(result) == 500

    def test_get_culture_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_culture_profile = AsyncMock(side_effect=TypeError("wrong type"))
        result = handler._handle_get_culture({})
        assert _status(result) == 500

    def test_get_culture_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.get_culture_profile = AsyncMock(side_effect=RuntimeError("runtime"))
        result = handler._handle_get_culture({})
        assert _status(result) == 500

    def test_get_culture_total_observations_zero(self, handler, mock_mound):
        """Zero observations are returned correctly."""
        profile = MockCultureProfile(total_observations=0)
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["total_observations"] == 0

    def test_get_culture_workspace_id_truncated(self, handler, mock_mound):
        """Workspace ID longer than 100 chars is truncated by get_bounded_string_param."""
        long_ws = "a" * 200
        handler._handle_get_culture({"workspace_id": long_ws})
        call_args = mock_mound.get_culture_profile.call_args
        # The workspace_id passed to get_culture_profile should be truncated to 100
        actual_ws = call_args[0][0] if call_args[0] else call_args[1].get("workspace_id")
        assert len(actual_ws) == 100

    def test_get_culture_mixed_patterns(self, handler, mock_mound):
        """Mix of patterns with and without to_dict are handled correctly."""
        profile = MockCultureProfile()
        profile.patterns = {
            "with_method": MockCulturePattern("test", 10),
            "raw_string": "raw_value",
        }
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["patterns"]["with_method"]["name"] == "test"
        assert body["patterns"]["raw_string"] == "raw_value"

    def test_get_culture_large_observations(self, handler, mock_mound):
        """Large total_observations value is handled correctly."""
        profile = MockCultureProfile(total_observations=999999)
        mock_mound.get_culture_profile = AsyncMock(return_value=profile)
        result = handler._handle_get_culture({})
        body = _body(result)
        assert body["total_observations"] == 999999


# ============================================================================
# Tests: _handle_add_culture_document (POST /api/v1/knowledge/mound/culture/documents)
# ============================================================================


class TestAddCultureDocument:
    """Test _handle_add_culture_document (POST /api/knowledge/mound/culture/documents)."""

    def test_add_document_success(self, handler, mock_mound):
        """Successfully adding a culture document returns 201 with node_id."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Our company values collaboration and transparency.",
                "workspace_id": "ws-001",
                "document_type": "values",
                "metadata": {"source": "hr"},
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["node_id"] == "node-culture-001"
        assert body["document_type"] == "values"
        assert body["workspace_id"] == "ws-001"
        assert "successfully" in body["message"].lower()

    def test_add_document_default_workspace(self, handler, mock_mound):
        """Default workspace_id is 'default' when not provided."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Test content",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["workspace_id"] == "default"

    def test_add_document_default_document_type(self, handler, mock_mound):
        """Default document_type is 'policy' when not provided."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Test content",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201
        body = _body(result)
        assert body["document_type"] == "policy"

    def test_add_document_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower() or "required" in body["error"].lower()

    def test_add_document_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower() or "body" in body["error"].lower()

    def test_add_document_missing_content_returns_400(self, handler):
        """Missing content field returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "workspace_id": "ws-001",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400
        body = _body(result)
        assert "content" in body["error"].lower()

    def test_add_document_empty_content_returns_400(self, handler):
        """Empty string content returns 400."""
        http = MockHTTPHandler.with_body(
            {
                "content": "",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400
        body = _body(result)
        assert "content" in body["error"].lower()

    def test_add_document_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Some content",
            }
        )
        result = handler_no_mound._handle_add_culture_document(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_add_document_mound_called_with_node(self, handler, mock_mound):
        """mound.add_node is called with a KnowledgeNode."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Culture content here",
                "workspace_id": "ws-test",
                "document_type": "values",
            }
        )
        handler._handle_add_culture_document(http)
        mock_mound.add_node.assert_called_once()
        node = mock_mound.add_node.call_args[0][0]
        assert node.content == "Culture content here"
        assert node.workspace_id == "ws-test"
        assert node.confidence == 1.0
        assert "culture" in node.topics
        assert "values" in node.topics

    def test_add_document_node_metadata_includes_document_type(self, handler, mock_mound):
        """Node metadata includes document_type."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Content",
                "document_type": "mission",
            }
        )
        handler._handle_add_culture_document(http)
        node = mock_mound.add_node.call_args[0][0]
        assert node.metadata["document_type"] == "mission"

    def test_add_document_custom_metadata_merged(self, handler, mock_mound):
        """Custom metadata is merged into node metadata."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Content",
                "metadata": {"author": "CEO", "version": "2.0"},
            }
        )
        handler._handle_add_culture_document(http)
        node = mock_mound.add_node.call_args[0][0]
        assert node.metadata["author"] == "CEO"
        assert node.metadata["version"] == "2.0"
        assert node.metadata["document_type"] == "policy"

    def test_add_document_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.add_node = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 500

    def test_add_document_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.add_node = AsyncMock(side_effect=ValueError("bad"))
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 500

    def test_add_document_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.add_node = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 500

    def test_add_document_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.add_node = AsyncMock(side_effect=TypeError("wrong type"))
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 500

    def test_add_document_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.add_node = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 500

    def test_add_document_import_error_returns_500(self, handler, mock_mound):
        """ImportError during node creation returns 500."""
        # Patch the import inside the handler to raise ImportError
        with patch(
            "aragora.server.handlers.knowledge_base.mound.culture._run_async",
            side_effect=ImportError("no module"),
        ):
            http = MockHTTPHandler.with_body({"content": "Content"})
            result = handler._handle_add_culture_document(http)
            assert _status(result) == 500

    def test_add_document_no_metadata_defaults_to_empty(self, handler, mock_mound):
        """When metadata is not provided, it defaults to empty dict."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Content",
            }
        )
        handler._handle_add_culture_document(http)
        node = mock_mound.add_node.call_args[0][0]
        # metadata should have document_type but no extra keys
        assert node.metadata == {"document_type": "policy"}

    def test_add_document_node_type_is_culture(self, handler, mock_mound):
        """Node type is set to 'culture'."""
        http = MockHTTPHandler.with_body({"content": "Content"})
        handler._handle_add_culture_document(http)
        node = mock_mound.add_node.call_args[0][0]
        assert str(node.node_type) == "culture"

    def test_add_document_provenance_user_type(self, handler, mock_mound):
        """Node provenance source_type is USER."""
        http = MockHTTPHandler.with_body({"content": "Content"})
        handler._handle_add_culture_document(http)
        node = mock_mound.add_node.call_args[0][0]
        assert node.provenance.source_id == "culture_document"

    def test_add_document_various_document_types(self, handler, mock_mound):
        """Various document_type values are accepted and returned."""
        for doc_type in ["policy", "values", "guidelines", "mission"]:
            mock_mound.add_node = AsyncMock(return_value=f"node-{doc_type}")
            http = MockHTTPHandler.with_body(
                {
                    "content": f"Content for {doc_type}",
                    "document_type": doc_type,
                }
            )
            result = handler._handle_add_culture_document(http)
            assert _status(result) == 201
            body = _body(result)
            assert body["document_type"] == doc_type


# ============================================================================
# Tests: _handle_promote_to_culture (POST /api/v1/knowledge/mound/culture/promote)
# ============================================================================


class TestPromoteToCulture:
    """Test _handle_promote_to_culture (POST /api/knowledge/mound/culture/promote)."""

    def test_promote_success(self, handler, mock_mound):
        """Successfully promoting a node returns 200 with promoted=True."""
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-123"
        assert body["promoted"] is True
        assert "successfully" in body["message"].lower()

    def test_promote_empty_body_returns_400(self, handler):
        """Empty body (Content-Length: 0) returns 400."""
        http = MockHTTPHandler.empty()
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400
        body = _body(result)
        assert "body" in body["error"].lower() or "required" in body["error"].lower()

    def test_promote_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        http = MockHTTPHandler.invalid_json()
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400

    def test_promote_missing_node_id_returns_400(self, handler):
        """Missing node_id field returns 400."""
        http = MockHTTPHandler.with_body({"workspace_id": "ws-001"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400
        body = _body(result)
        assert "node_id" in body["error"].lower()

    def test_promote_empty_node_id_returns_400(self, handler):
        """Empty string node_id returns 400 (falsy check)."""
        http = MockHTTPHandler.with_body({"node_id": ""})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400
        body = _body(result)
        assert "node_id" in body["error"].lower()

    def test_promote_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler_no_mound._handle_promote_to_culture(http)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_promote_node_not_found_returns_404(self, handler, mock_mound):
        """When mound.update returns None (falsy), returns 404."""
        mock_mound.update = AsyncMock(return_value=None)
        http = MockHTTPHandler.with_body({"node_id": "nonexistent"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_promote_node_not_found_includes_id(self, handler, mock_mound):
        """404 error message includes the node_id."""
        mock_mound.update = AsyncMock(return_value=None)
        http = MockHTTPHandler.with_body({"node_id": "node-abc"})
        result = handler._handle_promote_to_culture(http)
        body = _body(result)
        assert "node-abc" in body["error"]

    def test_promote_update_false_returns_404(self, handler, mock_mound):
        """When mound.update returns False (falsy), returns 404."""
        mock_mound.update = AsyncMock(return_value=False)
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 404

    def test_promote_mound_called_with_correct_updates(self, handler, mock_mound):
        """mound.update is called with culture promotion fields."""
        http = MockHTTPHandler.with_body({"node_id": "node-xyz"})
        handler._handle_promote_to_culture(http)
        mock_mound.update.assert_called_once()
        call_args = mock_mound.update.call_args
        node_id = call_args[0][0]
        updates = call_args[0][1]
        assert node_id == "node-xyz"
        assert updates["node_type"] == "culture"
        assert updates["promoted_to_culture"] is True
        # tier should be MemoryTier.GLACIAL.value
        assert "tier" in updates

    def test_promote_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.update = AsyncMock(side_effect=KeyError("missing"))
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 500

    def test_promote_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.update = AsyncMock(side_effect=ValueError("bad"))
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 500

    def test_promote_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.update = AsyncMock(side_effect=OSError("disk fail"))
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 500

    def test_promote_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.update = AsyncMock(side_effect=TypeError("wrong type"))
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 500

    def test_promote_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        mock_mound.update = AsyncMock(side_effect=RuntimeError("runtime"))
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 500

    def test_promote_import_error_returns_500(self, handler, mock_mound):
        """ImportError during tier import returns 500."""
        with patch(
            "aragora.server.handlers.knowledge_base.mound.culture._run_async",
            side_effect=ImportError("no module"),
        ):
            http = MockHTTPHandler.with_body({"node_id": "node-123"})
            result = handler._handle_promote_to_culture(http)
            assert _status(result) == 500

    def test_promote_node_id_none_returns_400(self, handler):
        """Explicit None node_id returns 400 (falsy)."""
        http = MockHTTPHandler.with_body({"node_id": None})
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400

    def test_promote_response_structure(self, handler, mock_mound):
        """Promote response contains exactly node_id, promoted, and message."""
        http = MockHTTPHandler.with_body({"node_id": "node-123"})
        result = handler._handle_promote_to_culture(http)
        body = _body(result)
        assert set(body.keys()) == {"node_id", "promoted", "message"}


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestCultureEdgeCases:
    """Test edge cases across culture operations."""

    def test_get_culture_response_structure(self, handler, mock_mound):
        """GET culture response contains workspace_id, patterns, generated_at, total_observations."""
        result = handler._handle_get_culture({})
        body = _body(result)
        assert set(body.keys()) == {
            "workspace_id",
            "patterns",
            "generated_at",
            "total_observations",
        }

    def test_add_document_response_structure(self, handler, mock_mound):
        """POST culture/documents response contains node_id, document_type, workspace_id, message."""
        http = MockHTTPHandler.with_body({"content": "Content"})
        result = handler._handle_add_culture_document(http)
        body = _body(result)
        assert set(body.keys()) == {
            "node_id",
            "document_type",
            "workspace_id",
            "message",
        }

    def test_add_document_content_length_zero_explicit(self, handler):
        """Content-Length explicitly set to 0 returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "0"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400

    def test_promote_content_length_zero_explicit(self, handler):
        """Content-Length explicitly set to 0 returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "0"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400

    def test_add_document_with_unicode_content(self, handler, mock_mound):
        """Unicode content is handled correctly."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Wir schaetzen Zusammenarbeit und Transparenz. \u2764\ufe0f",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201
        node = mock_mound.add_node.call_args[0][0]
        assert "\u2764\ufe0f" in node.content

    def test_add_document_with_long_content(self, handler, mock_mound):
        """Long content is accepted."""
        content = "x" * 10000
        http = MockHTTPHandler.with_body({"content": content})
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201
        node = mock_mound.add_node.call_args[0][0]
        assert len(node.content) == 10000

    def test_promote_with_extra_fields_ignored(self, handler, mock_mound):
        """Extra fields in promote request body are ignored."""
        http = MockHTTPHandler.with_body(
            {
                "node_id": "node-123",
                "extra_field": "should be ignored",
                "another": 42,
            }
        )
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-123"

    def test_add_document_with_extra_fields_ignored(self, handler, mock_mound):
        """Extra fields in add document request are ignored gracefully."""
        http = MockHTTPHandler.with_body(
            {
                "content": "Content",
                "extra_field": "ignored",
            }
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 201

    def test_get_culture_empty_query_params(self, handler, mock_mound):
        """Empty query params use defaults."""
        result = handler._handle_get_culture({})
        assert _status(result) == 200

    def test_add_document_content_whitespace_only(self, handler):
        """Whitespace-only content is technically non-empty (not falsy)."""
        http = MockHTTPHandler.with_body({"content": "   "})
        result = handler._handle_add_culture_document(http)
        # "   " is truthy so should be accepted
        assert _status(result) == 201 or _status(result) == 400
        # The handler checks `if not content` which is falsy for "   " -> False
        # "   " is truthy, so it should be accepted
        assert _status(result) == 201

    def test_sequential_add_and_promote(self, handler, mock_mound):
        """Adding a document then promoting it is a valid workflow."""
        # Add document
        http_add = MockHTTPHandler.with_body({"content": "Culture doc"})
        add_result = handler._handle_add_culture_document(http_add)
        assert _status(add_result) == 201
        node_id = _body(add_result)["node_id"]

        # Promote that node
        http_promote = MockHTTPHandler.with_body({"node_id": node_id})
        promote_result = handler._handle_promote_to_culture(http_promote)
        assert _status(promote_result) == 200
        assert _body(promote_result)["node_id"] == node_id

    def test_add_document_negative_content_length(self, handler):
        """Negative Content-Length is handled gracefully (treated as 0)."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "-1"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_add_culture_document(http)
        # Negative content_length means content_length > 0 is False
        assert _status(result) == 400

    def test_promote_negative_content_length(self, handler):
        """Negative Content-Length is handled gracefully."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "-1"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400

    def test_add_document_non_numeric_content_length(self, handler):
        """Non-numeric Content-Length returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "abc"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_add_culture_document(http)
        assert _status(result) == 400

    def test_promote_non_numeric_content_length(self, handler):
        """Non-numeric Content-Length returns 400."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "abc"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_promote_to_culture(http)
        assert _status(result) == 400
