"""
Tests for knowledge handler.

Tests:
- KnowledgeHandler initialization
- Route matching (can_handle)
- Authentication requirements for write endpoints
- Fact CRUD operations
- Verification queueing when agents unavailable
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from aragora.server.handlers.knowledge import KnowledgeHandler
from aragora.server.handlers.base import HandlerResult


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a KnowledgeHandler instance."""
    return KnowledgeHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.headers = {"Content-Type": "application/json", "Content-Length": "100"}
    mock.command = "GET"
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b'{}'
    return mock


@pytest.fixture
def mock_fact_store():
    """Create a mock fact store."""
    from aragora.knowledge.fact_store import Fact

    store = MagicMock()

    # Create a proper Fact mock
    mock_fact = MagicMock(spec=Fact)
    mock_fact.id = "fact-1"
    mock_fact.statement = "Test fact"
    mock_fact.confidence = 0.8
    mock_fact.workspace_id = "default"
    mock_fact.topics = []
    mock_fact.metadata = {}
    mock_fact.to_dict.return_value = {
        "id": "fact-1",
        "statement": "Test fact",
        "confidence": 0.8,
        "workspace_id": "default",
        "topics": [],
        "metadata": {}
    }

    store.add_fact.return_value = mock_fact
    store.get_fact.return_value = mock_fact
    store.update_fact.return_value = mock_fact
    store.delete_fact.return_value = True
    store.list_facts.return_value = [mock_fact]
    store.query_facts.return_value = [mock_fact]
    store.get_contradictions.return_value = []
    store.get_relations.return_value = []

    return store


def get_status(result: HandlerResult) -> int:
    """Extract status code from handler result."""
    return result.status_code


def get_body(result: HandlerResult) -> dict:
    """Extract JSON body from handler result."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# Test KnowledgeHandler Initialization
# ===========================================================================


class TestKnowledgeHandlerInit:
    """Tests for KnowledgeHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = KnowledgeHandler({})
        assert handler is not None


# ===========================================================================
# Test Route Matching (can_handle)
# ===========================================================================


class TestKnowledgeHandlerCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_facts_endpoint(self, handler):
        """Should handle /api/knowledge/facts."""
        assert handler.can_handle("/api/knowledge/facts") is True

    def test_can_handle_facts_by_id(self, handler):
        """Should handle /api/knowledge/facts/:id."""
        assert handler.can_handle("/api/knowledge/facts/fact-1") is True

    def test_can_handle_query_endpoint(self, handler):
        """Should handle /api/knowledge/query."""
        assert handler.can_handle("/api/knowledge/query") is True

    def test_can_handle_stats_endpoint(self, handler):
        """Should handle /api/knowledge/stats."""
        assert handler.can_handle("/api/knowledge/stats") is True

    def test_cannot_handle_unknown_path(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/knowledge") is False
        assert handler.can_handle("/api/unknown") is False
        assert handler.can_handle("/api/other/path") is False


# ===========================================================================
# Test List Facts (GET - no auth required)
# ===========================================================================


class TestListFacts:
    """Tests for GET /api/knowledge/facts endpoint."""

    def test_list_facts_success(self, handler, mock_http_handler, mock_fact_store):
        """Should list facts without auth."""
        mock_http_handler.command = "GET"

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            result = handler.handle("/api/knowledge/facts", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "facts" in body

    def test_list_facts_with_filters(self, handler, mock_http_handler, mock_fact_store):
        """Should filter facts by workspace."""
        mock_http_handler.command = "GET"

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            result = handler.handle("/api/knowledge/facts", {"workspace_id": ["ws-1"]}, mock_http_handler)

        assert get_status(result) == 200


# ===========================================================================
# Test Get Fact (GET - no auth required)
# ===========================================================================


class TestGetFact:
    """Tests for GET /api/knowledge/facts/:id endpoint."""

    def test_get_fact_success(self, handler, mock_http_handler, mock_fact_store):
        """Should get fact by id."""
        mock_http_handler.command = "GET"

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            result = handler.handle("/api/knowledge/facts/fact-1", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["id"] == "fact-1"

    def test_get_fact_not_found(self, handler, mock_http_handler, mock_fact_store):
        """Should return 404 for nonexistent fact."""
        mock_http_handler.command = "GET"
        mock_fact_store.get_fact.return_value = None

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            result = handler.handle("/api/knowledge/facts/nonexistent", {}, mock_http_handler)

        assert get_status(result) == 404


# ===========================================================================
# Test Create Fact (POST - auth required)
# ===========================================================================


class TestCreateFact:
    """Tests for POST /api/knowledge/facts endpoint."""

    def test_create_requires_auth(self, handler, mock_http_handler, mock_fact_store):
        """Should require authentication for fact creation."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "statement": "Test fact"
        }).encode()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(None, HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}))):
                result = handler.handle("/api/knowledge/facts", {}, mock_http_handler)

        assert get_status(result) == 401

    def test_create_fact_success(self, handler, mock_http_handler, mock_fact_store):
        """Should create fact successfully with auth."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "statement": "Test fact",
            "confidence": 0.8
        }).encode()
        mock_http_handler.headers = {"Content-Type": "application/json", "Content-Length": "50"}

        mock_user = MagicMock()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                result = handler.handle("/api/knowledge/facts", {}, mock_http_handler)

        assert get_status(result) == 201
        body = get_body(result)
        assert body["id"] == "fact-1"

    def test_create_fact_missing_statement(self, handler, mock_http_handler, mock_fact_store):
        """Should reject fact without statement."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "confidence": 0.8
        }).encode()
        mock_http_handler.headers = {"Content-Type": "application/json", "Content-Length": "20"}

        mock_user = MagicMock()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                result = handler.handle("/api/knowledge/facts", {}, mock_http_handler)

        assert get_status(result) == 400
        assert "Statement" in get_body(result)["error"]


# ===========================================================================
# Test Update Fact (PUT - auth required)
# ===========================================================================


class TestUpdateFact:
    """Tests for PUT /api/knowledge/facts/:id endpoint."""

    def test_update_requires_auth(self, handler, mock_http_handler, mock_fact_store):
        """Should require authentication for fact update."""
        mock_http_handler.command = "PUT"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "confidence": 0.9
        }).encode()
        mock_http_handler.headers = {"Content-Type": "application/json", "Content-Length": "20"}

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(None, HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}))):
                result = handler.handle("/api/knowledge/facts/fact-1", {}, mock_http_handler)

        assert get_status(result) == 401

    def test_update_fact_success(self, handler, mock_http_handler, mock_fact_store):
        """Should update fact successfully with auth."""
        mock_http_handler.command = "PUT"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "confidence": 0.9
        }).encode()
        mock_http_handler.headers = {"Content-Type": "application/json", "Content-Length": "20"}

        mock_user = MagicMock()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                result = handler.handle("/api/knowledge/facts/fact-1", {}, mock_http_handler)

        assert get_status(result) == 200


# ===========================================================================
# Test Delete Fact (DELETE - auth required)
# ===========================================================================


class TestDeleteFact:
    """Tests for DELETE /api/knowledge/facts/:id endpoint."""

    def test_delete_requires_auth(self, handler, mock_http_handler, mock_fact_store):
        """Should require authentication for fact deletion."""
        mock_http_handler.command = "DELETE"

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(None, HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}))):
                result = handler.handle("/api/knowledge/facts/fact-1", {}, mock_http_handler)

        assert get_status(result) == 401

    def test_delete_fact_success(self, handler, mock_http_handler, mock_fact_store):
        """Should delete fact successfully with auth."""
        mock_http_handler.command = "DELETE"

        mock_user = MagicMock()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                result = handler.handle("/api/knowledge/facts/fact-1", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["deleted"] is True

    def test_delete_fact_not_found(self, handler, mock_http_handler, mock_fact_store):
        """Should return 404 for nonexistent fact."""
        mock_http_handler.command = "DELETE"
        mock_fact_store.delete_fact.return_value = False

        mock_user = MagicMock()

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                result = handler.handle("/api/knowledge/facts/nonexistent", {}, mock_http_handler)

        assert get_status(result) == 404


# ===========================================================================
# Test Verification Queueing
# ===========================================================================


class TestVerificationQueueing:
    """Tests for POST /api/knowledge/facts/:id/verify endpoint."""

    def test_verify_queues_when_agents_unavailable(self, handler, mock_http_handler, mock_fact_store):
        """Should queue verification when agent capability unavailable."""
        mock_http_handler.command = "POST"

        from aragora.knowledge.query_engine import SimpleQueryEngine
        mock_engine = MagicMock(spec=SimpleQueryEngine)

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "_get_query_engine", return_value=mock_engine):
                result = handler.handle("/api/knowledge/facts/fact-1/verify", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["status"] == "queued"
        assert body["verified"] is None
        assert "queued for verification" in body["message"]

    def test_verify_updates_fact_metadata(self, handler, mock_http_handler, mock_fact_store):
        """Should update fact metadata with pending verification."""
        mock_http_handler.command = "POST"

        from aragora.knowledge.query_engine import SimpleQueryEngine
        mock_engine = MagicMock(spec=SimpleQueryEngine)

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            with patch.object(handler, "_get_query_engine", return_value=mock_engine):
                handler.handle("/api/knowledge/facts/fact-1/verify", {}, mock_http_handler)

        # Verify update_fact was called with pending verification metadata
        mock_fact_store.update_fact.assert_called()
        call_args = mock_fact_store.update_fact.call_args
        metadata = call_args[1].get("metadata", {})
        assert metadata.get("_pending_verification") is True


# ===========================================================================
# Test Query Endpoint
# ===========================================================================


class TestQueryFacts:
    """Tests for POST /api/knowledge/query endpoint."""

    def test_query_success(self, handler, mock_http_handler, mock_fact_store):
        """Should query facts successfully."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps({
            "question": "test query"  # API expects "question", not "query"
        }).encode()
        mock_http_handler.headers = {"Content-Type": "application/json", "Content-Length": "30"}

        with patch.object(handler, "_get_fact_store", return_value=mock_fact_store):
            result = handler.handle("/api/knowledge/query", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "answer" in body  # Query returns an answer, not results list
