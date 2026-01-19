"""Tests for Knowledge handler endpoints.

Validates the REST API endpoints for the enterprise knowledge base:
- Facts API (CRUD operations)
- Query/search functionality
- Fact relationships and contradictions
- Statistics
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge import KnowledgeHandler


@pytest.fixture
def knowledge_handler():
    """Create a knowledge handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = KnowledgeHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address and headers."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_http_handler_post():
    """Create a mock HTTP handler for POST requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    return handler


def create_request_body(data: dict) -> MagicMock:
    """Create a mock handler with request body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


class TestKnowledgeHandlerCanHandle:
    """Test KnowledgeHandler.can_handle method."""

    def test_can_handle_query(self, knowledge_handler):
        """Test can_handle returns True for query endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/query")

    def test_can_handle_facts(self, knowledge_handler):
        """Test can_handle returns True for facts endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/facts")

    def test_can_handle_search(self, knowledge_handler):
        """Test can_handle returns True for search endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/search")

    def test_can_handle_stats(self, knowledge_handler):
        """Test can_handle returns True for stats endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/stats")

    def test_can_handle_fact_by_id(self, knowledge_handler):
        """Test can_handle returns True for fact by ID endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/facts/fact-123")

    def test_can_handle_fact_verify(self, knowledge_handler):
        """Test can_handle returns True for fact verify endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/facts/fact-123/verify")

    def test_can_handle_fact_contradictions(self, knowledge_handler):
        """Test can_handle returns True for fact contradictions endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/facts/fact-123/contradictions")

    def test_can_handle_fact_relations(self, knowledge_handler):
        """Test can_handle returns True for fact relations endpoint."""
        assert knowledge_handler.can_handle("/api/knowledge/facts/fact-123/relations")

    def test_cannot_handle_unknown(self, knowledge_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not knowledge_handler.can_handle("/api/knowledge/unknown")
        assert not knowledge_handler.can_handle("/api/debates")


class TestKnowledgeHandlerListFacts:
    """Test GET /api/knowledge/facts endpoint."""

    def test_list_facts_default(self, knowledge_handler, mock_http_handler):
        """Test listing facts with default parameters."""
        result = knowledge_handler.handle("/api/knowledge/facts", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "facts" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_facts_with_filters(self, knowledge_handler, mock_http_handler):
        """Test listing facts with filter parameters."""
        query_params = {
            "workspace_id": "test-workspace",
            "topic": "security",
            "min_confidence": "0.7",
            "limit": "10",
            "offset": "0",
        }
        result = knowledge_handler.handle("/api/knowledge/facts", query_params, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "facts" in body
        assert body["limit"] == 10
        assert body["offset"] == 0

    def test_list_facts_with_status_filter(self, knowledge_handler, mock_http_handler):
        """Test listing facts filtered by validation status."""
        query_params = {
            "status": "majority_agreed",
            "include_superseded": "false",
        }
        result = knowledge_handler.handle("/api/knowledge/facts", query_params, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "facts" in body


class TestKnowledgeHandlerGetFact:
    """Test GET /api/knowledge/facts/:id endpoint."""

    def test_get_fact_not_found(self, knowledge_handler, mock_http_handler):
        """Test getting non-existent fact returns 404."""
        result = knowledge_handler.handle(
            "/api/knowledge/facts/nonexistent-id", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body

    def test_get_fact_success(self, knowledge_handler, mock_http_handler):
        """Test getting existing fact."""
        # First create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Test fact statement",
            workspace_id="default",
            confidence=0.8,
        )

        # Then retrieve it
        result = knowledge_handler.handle(
            f"/api/knowledge/facts/{fact.id}", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["statement"] == "Test fact statement"
        assert body["confidence"] == 0.8


class TestKnowledgeHandlerCreateFact:
    """Test POST /api/knowledge/facts endpoint."""

    def test_create_fact_requires_auth(self, knowledge_handler):
        """Test creating fact requires authentication."""
        handler = create_request_body({
            "statement": "New test fact",
            "workspace_id": "default",
        })

        # Patch require_auth_or_error to return auth error
        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            from aragora.server.handlers.base import error_response
            mock_auth.return_value = (None, error_response("Unauthorized", 401))

            result = knowledge_handler.handle("/api/knowledge/facts", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_create_fact_success(self, knowledge_handler):
        """Test creating fact with valid data."""
        handler = create_request_body({
            "statement": "New test fact",
            "workspace_id": "default",
            "confidence": 0.9,
            "topics": ["testing"],
        })

        # Patch require_auth_or_error to return success
        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle("/api/knowledge/facts", {}, handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["statement"] == "New test fact"
        assert body["confidence"] == 0.9

    def test_create_fact_missing_statement(self, knowledge_handler):
        """Test creating fact without statement returns error."""
        handler = create_request_body({
            "workspace_id": "default",
        })

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle("/api/knowledge/facts", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestKnowledgeHandlerUpdateFact:
    """Test PUT /api/knowledge/facts/:id endpoint."""

    def test_update_fact_not_found(self, knowledge_handler):
        """Test updating non-existent fact returns 404."""
        handler = create_request_body({
            "confidence": 0.95,
        })
        handler.command = "PUT"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle(
                "/api/knowledge/facts/nonexistent-id", {}, handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_update_fact_success(self, knowledge_handler):
        """Test updating existing fact."""
        # First create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Original statement",
            workspace_id="default",
            confidence=0.5,
        )

        # Then update it
        handler = create_request_body({
            "confidence": 0.95,
            "topics": ["updated"],
        })
        handler.command = "PUT"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle(
                f"/api/knowledge/facts/{fact.id}", {}, handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["confidence"] == 0.95


class TestKnowledgeHandlerDeleteFact:
    """Test DELETE /api/knowledge/facts/:id endpoint."""

    def test_delete_fact_not_found(self, knowledge_handler, mock_http_handler):
        """Test deleting non-existent fact returns 404."""
        mock_http_handler.command = "DELETE"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle(
                "/api/knowledge/facts/nonexistent-id", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_delete_fact_success(self, knowledge_handler, mock_http_handler):
        """Test deleting existing fact."""
        # First create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Fact to delete",
            workspace_id="default",
        )

        mock_http_handler.command = "DELETE"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)

            result = knowledge_handler.handle(
                f"/api/knowledge/facts/{fact.id}", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deleted"] is True


class TestKnowledgeHandlerVerifyFact:
    """Test POST /api/knowledge/facts/:id/verify endpoint."""

    def test_verify_fact_not_found(self, knowledge_handler, mock_http_handler_post):
        """Test verifying non-existent fact returns 404."""
        result = knowledge_handler.handle(
            "/api/knowledge/facts/nonexistent-id/verify", {}, mock_http_handler_post
        )

        assert result is not None
        assert result.status_code == 404

    def test_verify_fact_queued(self, knowledge_handler, mock_http_handler_post):
        """Test verifying fact queues when agents not available."""
        # Create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Fact to verify",
            workspace_id="default",
        )

        result = knowledge_handler.handle(
            f"/api/knowledge/facts/{fact.id}/verify", {}, mock_http_handler_post
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "queued"


class TestKnowledgeHandlerContradictions:
    """Test GET /api/knowledge/facts/:id/contradictions endpoint."""

    def test_get_contradictions_not_found(self, knowledge_handler, mock_http_handler):
        """Test getting contradictions for non-existent fact returns 404."""
        result = knowledge_handler.handle(
            "/api/knowledge/facts/nonexistent-id/contradictions", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404

    def test_get_contradictions_success(self, knowledge_handler, mock_http_handler):
        """Test getting contradictions for existing fact."""
        # Create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Original fact",
            workspace_id="default",
        )

        result = knowledge_handler.handle(
            f"/api/knowledge/facts/{fact.id}/contradictions", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "contradictions" in body
        assert "count" in body


class TestKnowledgeHandlerRelations:
    """Test /api/knowledge/facts/:id/relations endpoints."""

    def test_get_relations_not_found(self, knowledge_handler, mock_http_handler):
        """Test getting relations for non-existent fact returns 404."""
        result = knowledge_handler.handle(
            "/api/knowledge/facts/nonexistent-id/relations", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404

    def test_get_relations_success(self, knowledge_handler, mock_http_handler):
        """Test getting relations for existing fact."""
        # Create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Fact with relations",
            workspace_id="default",
        )

        result = knowledge_handler.handle(
            f"/api/knowledge/facts/{fact.id}/relations", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "relations" in body
        assert "count" in body

    def test_get_relations_with_type_filter(self, knowledge_handler, mock_http_handler):
        """Test getting relations filtered by type."""
        # Create a fact
        store = knowledge_handler._get_fact_store()
        fact = store.add_fact(
            statement="Fact for filtering",
            workspace_id="default",
        )

        query_params = {"type": "supports"}
        result = knowledge_handler.handle(
            f"/api/knowledge/facts/{fact.id}/relations", query_params, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "relations" in body


class TestKnowledgeHandlerStats:
    """Test GET /api/knowledge/stats endpoint."""

    def test_get_stats(self, knowledge_handler, mock_http_handler):
        """Test getting knowledge base statistics."""
        result = knowledge_handler.handle("/api/knowledge/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert isinstance(body, dict)

    def test_get_stats_with_workspace(self, knowledge_handler, mock_http_handler):
        """Test getting stats for specific workspace."""
        query_params = {"workspace_id": "test-workspace"}
        result = knowledge_handler.handle("/api/knowledge/stats", query_params, mock_http_handler)

        assert result is not None
        assert result.status_code == 200


class TestKnowledgeHandlerSearch:
    """Test GET /api/knowledge/search endpoint."""

    def test_search_basic(self, knowledge_handler, mock_http_handler):
        """Test basic search functionality."""
        query_params = {"q": "security"}
        result = knowledge_handler.handle("/api/knowledge/search", query_params, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert isinstance(body, dict)


class TestKnowledgeHandlerQuery:
    """Test POST /api/knowledge/query endpoint."""

    def test_query_missing_question(self, knowledge_handler):
        """Test query without question returns error."""
        handler = create_request_body({})

        result = knowledge_handler.handle("/api/knowledge/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_query_with_question(self, knowledge_handler):
        """Test query with valid question."""
        handler = create_request_body({
            "question": "What are the security requirements?",
            "workspace_id": "default",
        })

        result = knowledge_handler.handle("/api/knowledge/query", {}, handler)

        assert result is not None
        body = json.loads(result.body)
        assert isinstance(body, dict)

    def test_query_with_options(self, knowledge_handler):
        """Test query with custom options."""
        handler = create_request_body({
            "question": "What are the compliance rules?",
            "workspace_id": "compliance",
            "options": {
                "max_chunks": 5,
                "search_alpha": 0.7,
                "use_agents": False,
            },
        })

        result = knowledge_handler.handle("/api/knowledge/query", {}, handler)

        assert result is not None
        body = json.loads(result.body)
        assert isinstance(body, dict)


class TestKnowledgeHandlerIntegration:
    """Integration tests for Knowledge handler."""

    def test_full_fact_lifecycle(self, knowledge_handler):
        """Test full fact lifecycle: create -> read -> update -> delete."""
        # 1. Create a fact
        create_handler = create_request_body({
            "statement": "Integration test fact",
            "workspace_id": "integration",
            "confidence": 0.7,
            "topics": ["testing", "integration"],
        })

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)
            create_result = knowledge_handler.handle("/api/knowledge/facts", {}, create_handler)

        assert create_result is not None
        assert create_result.status_code == 201
        create_body = json.loads(create_result.body)
        fact_id = create_body["id"]

        # 2. Read the fact
        read_handler = MagicMock()
        read_handler.client_address = ("127.0.0.1", 12345)
        read_handler.command = "GET"

        read_result = knowledge_handler.handle(f"/api/knowledge/facts/{fact_id}", {}, read_handler)
        assert read_result is not None
        assert read_result.status_code == 200

        # 3. Update the fact
        update_handler = create_request_body({
            "confidence": 0.95,
        })
        update_handler.command = "PUT"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)
            update_result = knowledge_handler.handle(f"/api/knowledge/facts/{fact_id}", {}, update_handler)

        assert update_result is not None
        assert update_result.status_code == 200
        update_body = json.loads(update_result.body)
        assert update_body["confidence"] == 0.95

        # 4. Delete the fact
        delete_handler = MagicMock()
        delete_handler.client_address = ("127.0.0.1", 12345)
        delete_handler.command = "DELETE"

        with patch.object(knowledge_handler, 'require_auth_or_error') as mock_auth:
            mock_auth.return_value = ({"user_id": "test-user"}, None)
            delete_result = knowledge_handler.handle(f"/api/knowledge/facts/{fact_id}", {}, delete_handler)

        assert delete_result is not None
        assert delete_result.status_code == 200

        # 5. Verify fact is gone
        verify_result = knowledge_handler.handle(f"/api/knowledge/facts/{fact_id}", {}, read_handler)
        assert verify_result.status_code == 404
