"""
Tests for KnowledgeHandler.

Tests cover:
- Route registration and can_handle
- Facts CRUD (list, get, create, update, delete)
- Query endpoint
- Search and stats endpoints
- Path normalization (/api/v1/facts/* alias)
- Error handling
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from tests.server.handlers.conftest import parse_handler_response


# ===========================================================================
# Route Registration Tests
# ===========================================================================


class TestKnowledgeHandlerRoutes:
    """Verify that KnowledgeHandler declares the expected routes."""

    def test_has_routes(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert hasattr(KnowledgeHandler, "ROUTES")
        assert len(KnowledgeHandler.ROUTES) >= 4

    def test_knowledge_query_route(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert "/api/v1/knowledge/query" in KnowledgeHandler.ROUTES

    def test_knowledge_facts_route(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert "/api/v1/knowledge/facts" in KnowledgeHandler.ROUTES

    def test_knowledge_search_route(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert "/api/v1/knowledge/search" in KnowledgeHandler.ROUTES

    def test_knowledge_stats_route(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert "/api/v1/knowledge/stats" in KnowledgeHandler.ROUTES

    def test_facts_alias_route(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert "/api/v1/facts" in KnowledgeHandler.ROUTES


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestKnowledgeHandlerCanHandle:
    """Test the can_handle routing method."""

    @pytest.fixture
    def handler(self, mock_server_context):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        return KnowledgeHandler(server_context=mock_server_context)

    def test_can_handle_facts_route(self, handler):
        assert handler.can_handle("/api/v1/knowledge/facts") is True

    def test_can_handle_facts_with_id(self, handler):
        assert handler.can_handle("/api/v1/knowledge/facts/fact-123") is True

    def test_can_handle_query(self, handler):
        assert handler.can_handle("/api/v1/knowledge/query") is True

    def test_can_handle_search(self, handler):
        assert handler.can_handle("/api/v1/knowledge/search") is True

    def test_can_handle_stats(self, handler):
        assert handler.can_handle("/api/v1/knowledge/stats") is True

    def test_can_handle_sdk_alias(self, handler):
        assert handler.can_handle("/api/v1/facts") is True

    def test_can_handle_sdk_alias_with_id(self, handler):
        assert handler.can_handle("/api/v1/facts/fact-456") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/v1/knowledge") is False


# ===========================================================================
# Path Normalization Tests
# ===========================================================================


class TestPathNormalization:
    """Test _normalize_facts_path for SDK alias support."""

    def test_normalizes_facts_root(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert KnowledgeHandler._normalize_facts_path("/api/v1/facts") == "/api/v1/knowledge/facts"

    def test_normalizes_facts_with_id(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert (
            KnowledgeHandler._normalize_facts_path("/api/v1/facts/fact-123")
            == "/api/v1/knowledge/facts/fact-123"
        )

    def test_normalizes_facts_with_subpath(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert (
            KnowledgeHandler._normalize_facts_path("/api/v1/facts/fact-123/verify")
            == "/api/v1/knowledge/facts/fact-123/verify"
        )

    def test_does_not_normalize_knowledge_path(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        path = "/api/v1/knowledge/facts/fact-123"
        assert KnowledgeHandler._normalize_facts_path(path) == path

    def test_does_not_normalize_unrelated_path(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        path = "/api/v1/debates/debate-123"
        assert KnowledgeHandler._normalize_facts_path(path) == path


# ===========================================================================
# Import and Class Hierarchy Tests
# ===========================================================================


class TestKnowledgeHandlerImport:
    """Verify the handler can be imported and has the correct class hierarchy."""

    def test_importable(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert KnowledgeHandler is not None

    def test_has_handle_method(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert hasattr(KnowledgeHandler, "handle")

    def test_has_can_handle_method(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert hasattr(KnowledgeHandler, "can_handle")

    def test_has_permission_constants(self):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        assert KnowledgeHandler.KNOWLEDGE_READ_PERMISSION == "knowledge.read"
        assert KnowledgeHandler.KNOWLEDGE_WRITE_PERMISSION == "knowledge.write"
        assert KnowledgeHandler.KNOWLEDGE_DELETE_PERMISSION == "knowledge.delete"


# ===========================================================================
# Facts CRUD Tests
# ===========================================================================


class TestKnowledgeHandlerFacts:
    """Test facts CRUD operations via the handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler

        h = KnowledgeHandler(server_context=mock_server_context)
        # Inject a mock fact store with objects that have to_dict()
        mock_store = MagicMock()

        mock_fact = MagicMock()
        mock_fact.to_dict.return_value = {
            "id": "fact-001",
            "content": "Test fact",
            "confidence": 0.9,
        }
        mock_fact.id = "fact-001"

        mock_store.list_facts.return_value = [mock_fact]
        mock_store.get_fact.return_value = mock_fact
        mock_new_fact = MagicMock()
        mock_new_fact.to_dict.return_value = {
            "id": "fact-new",
            "content": "New fact",
            "confidence": 0.8,
        }
        mock_new_fact.id = "fact-new"
        mock_store.add_fact.return_value = mock_new_fact
        mock_store.update_fact.return_value = True
        mock_store.delete_fact.return_value = True
        mock_store.count.return_value = 1
        h._fact_store = mock_store
        return h

    @pytest.fixture
    def mock_get_handler(self):
        mock = MagicMock()
        mock.command = "GET"
        mock.client_address = ("127.0.0.1", 12345)
        mock.headers = {"Content-Length": "0"}
        return mock

    @pytest.fixture
    def mock_post_handler(self):
        mock = MagicMock()
        mock.command = "POST"
        mock.client_address = ("127.0.0.1", 12345)
        body = json.dumps({"statement": "New fact", "source": "test"}).encode()
        mock.headers = {"Content-Length": str(len(body))}
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = body
        return mock

    def test_list_facts(self, handler, mock_get_handler):
        result = handler.handle("/api/v1/knowledge/facts", {}, mock_get_handler)
        assert result is not None
        assert result.status_code == 200

    def test_list_facts_via_alias(self, handler, mock_get_handler):
        result = handler.handle("/api/v1/facts", {}, mock_get_handler)
        assert result is not None
        assert result.status_code == 200

    def test_get_fact_by_id(self, handler, mock_get_handler):
        result = handler.handle("/api/v1/knowledge/facts/fact-001", {}, mock_get_handler)
        assert result is not None
        assert result.status_code == 200

    def test_get_stats(self, handler, mock_get_handler):
        result = handler.handle("/api/v1/knowledge/stats", {}, mock_get_handler)
        assert result is not None
        assert result.status_code == 200

    def test_search(self, handler, mock_get_handler):
        # Search uses async internally - mock at handler level
        async def mock_search(*args, **kwargs):
            return []

        mock_engine = MagicMock()
        mock_engine.search = mock_search
        handler._query_engine = mock_engine
        result = handler.handle("/api/v1/knowledge/search", {"q": "test"}, mock_get_handler)
        assert result is not None
        assert result.status_code == 200

    def test_create_fact(self, handler, mock_post_handler):
        result = handler.handle("/api/v1/knowledge/facts", {}, mock_post_handler)
        assert result is not None
        assert result.status_code in (200, 201)

    def test_create_fact_missing_statement(self, handler):
        mock = MagicMock()
        mock.command = "POST"
        mock.client_address = ("127.0.0.1", 12345)
        body = json.dumps({"source": "test"}).encode()
        mock.headers = {"Content-Length": str(len(body))}
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = body

        result = handler.handle("/api/v1/knowledge/facts", {}, mock)
        assert result is not None
        assert result.status_code == 400

    def test_query_endpoint(self, handler, mock_post_handler):
        body = json.dumps({"question": "What is the meaning of life?"}).encode()
        mock_post_handler.rfile.read.return_value = body
        mock_post_handler.headers = {"Content-Length": str(len(body))}

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"answer": "42", "sources": []}

        async def mock_query(*args, **kwargs):
            return mock_result

        mock_engine = MagicMock()
        mock_engine.query = mock_query
        handler._query_engine = mock_engine

        result = handler.handle("/api/v1/knowledge/query", {}, mock_post_handler)
        assert result is not None
        assert result.status_code == 200
