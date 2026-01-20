"""
Tests for the Knowledge Base endpoint handlers.

Tests KnowledgeHandler and KnowledgeMoundHandler classes.
"""

import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class MockFact:
    """Mock Fact for testing."""

    id: str = "fact-123"
    statement: str = "Test fact statement"
    workspace_id: str = "default"
    confidence: float = 0.8
    topics: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    evidence_ids: list = field(default_factory=list)
    source_documents: list = field(default_factory=list)
    validation_status: str = "unverified"
    superseded_by: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "workspace_id": self.workspace_id,
            "confidence": self.confidence,
            "topics": self.topics,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evidence_ids": self.evidence_ids,
            "source_documents": self.source_documents,
            "validation_status": self.validation_status,
            "superseded_by": self.superseded_by,
        }


@dataclass
class MockRelation:
    """Mock Fact Relation for testing."""

    id: str = "rel-123"
    source_fact_id: str = "fact-1"
    target_fact_id: str = "fact-2"
    relation_type: str = "supports"
    confidence: float = 0.7
    created_by: str = "test-user"
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_fact_id": self.source_fact_id,
            "target_fact_id": self.target_fact_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class MockQueryResult:
    """Mock Query Result for testing."""

    answer: str = "Test answer"
    citations: list = field(default_factory=list)
    facts_extracted: list = field(default_factory=list)
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "facts_extracted": self.facts_extracted,
            "confidence": self.confidence,
        }


@dataclass
class MockSearchResult:
    """Mock Search Result for testing."""

    chunk_id: str = "chunk-1"
    content: str = "Test content"
    score: float = 0.9

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
        }


@dataclass
class MockHandler:
    """Mock HTTP Handler for testing."""

    command: str = "GET"
    headers: dict = field(default_factory=lambda: {"Content-Length": "0"})
    rfile: Any = None
    client_address: tuple = ("127.0.0.1", 12345)

    def __post_init__(self):
        if self.rfile is None:
            self.rfile = io.BytesIO(b"")


def make_handler_with_body(body: dict, method: str = "POST") -> MockHandler:
    """Create a mock handler with JSON body."""
    body_bytes = json.dumps(body).encode("utf-8")
    return MockHandler(
        command=method,
        headers={"Content-Length": str(len(body_bytes))},
        rfile=io.BytesIO(body_bytes),
    )


class TestKnowledgeHandlerInit:
    """Tests for KnowledgeHandler initialization."""

    def test_init_with_server_context(self):
        """Should initialize with server context."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        context = {"some": "context"}
        handler = KnowledgeHandler(context)

        assert handler.ctx == context
        assert handler._fact_store is None
        assert handler._query_engine is None

    def test_routes_defined(self):
        """Should have expected routes defined."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        assert "/api/knowledge/query" in KnowledgeHandler.ROUTES
        assert "/api/knowledge/facts" in KnowledgeHandler.ROUTES
        assert "/api/knowledge/search" in KnowledgeHandler.ROUTES
        assert "/api/knowledge/stats" in KnowledgeHandler.ROUTES


class TestKnowledgeHandlerCanHandle:
    """Tests for can_handle path matching."""

    def test_matches_static_routes(self):
        """Should match static routes."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        assert handler.can_handle("/api/knowledge/query")
        assert handler.can_handle("/api/knowledge/facts")
        assert handler.can_handle("/api/knowledge/search")
        assert handler.can_handle("/api/knowledge/stats")

    def test_matches_dynamic_fact_routes(self):
        """Should match dynamic fact routes."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        assert handler.can_handle("/api/knowledge/facts/fact-123")
        assert handler.can_handle("/api/knowledge/facts/fact-123/verify")
        assert handler.can_handle("/api/knowledge/facts/fact-123/contradictions")
        assert handler.can_handle("/api/knowledge/facts/fact-123/relations")

    def test_rejects_non_matching_paths(self):
        """Should reject non-matching paths."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other/endpoint")
        assert not handler.can_handle("/knowledge/facts")


class TestKnowledgeHandlerGetStores:
    """Tests for lazy store initialization."""

    def test_creates_fact_store_on_demand(self):
        """Should create fact store on first access."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        # Mock FactStore to avoid actual initialization
        with patch("aragora.server.handlers.knowledge.FactStore") as mock_fs:
            mock_fs.return_value = MagicMock()
            store = handler._get_fact_store()

            assert store is not None
            mock_fs.assert_called_once()

    def test_falls_back_to_in_memory_store(self):
        """Should fall back to in-memory store on error."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        # Mock FactStore to raise error
        with patch("aragora.server.handlers.knowledge.FactStore") as mock_fs:
            mock_fs.side_effect = Exception("Database error")
            store = handler._get_fact_store()

            # Should be InMemoryFactStore
            assert store is not None

    def test_creates_query_engine_on_demand(self):
        """Should create query engine on first access."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch.object(handler, "_get_fact_store") as mock_store:
            mock_store.return_value = MagicMock()
            engine = handler._get_query_engine()

            assert engine is not None


class TestKnowledgeHandlerListFacts:
    """Tests for GET /api/knowledge/facts endpoint."""

    def test_list_facts_returns_facts(self):
        """Should return list of facts."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.list_facts.return_value = [MockFact(), MockFact(id="fact-456")]
        handler._fact_store = mock_store

        # Clear rate limiter
        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert len(body["facts"]) == 2

    def test_list_facts_with_filters(self):
        """Should pass filters to fact store."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.list_facts.return_value = []
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            handler.handle(
                "/api/knowledge/facts",
                {"topic": "security", "min_confidence": "0.7", "limit": "10"},
                MockHandler(),
            )

        # Verify filters were passed
        call_args = mock_store.list_facts.call_args
        filters = call_args[0][0]
        assert filters.topics == ["security"]
        assert filters.min_confidence == 0.7
        assert filters.limit == 10


class TestKnowledgeHandlerGetFact:
    """Tests for GET /api/knowledge/facts/:id endpoint."""

    def test_get_fact_returns_fact(self):
        """Should return specific fact."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_fact.return_value = MockFact(id="fact-123", statement="Test")
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/fact-123",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["id"] == "fact-123"
        assert body["statement"] == "Test"

    def test_get_fact_not_found(self):
        """Should return 404 for non-existent fact."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_fact.return_value = None
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/nonexistent",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 404


class TestKnowledgeHandlerCreateFact:
    """Tests for POST /api/knowledge/facts endpoint."""

    def test_create_fact_requires_auth(self):
        """Should require authentication for fact creation."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, MagicMock(status_code=401))

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/facts",
                    {},
                    make_handler_with_body({"statement": "Test"}, "POST"),
                )

        assert result is not None
        assert result.status_code == 401

    def test_create_fact_requires_statement(self):
        """Should require statement in request body."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/facts",
                    {},
                    make_handler_with_body({}, "POST"),
                )

        assert result is not None
        assert result.status_code == 400
        assert b"Statement is required" in result.body

    def test_create_fact_success(self):
        """Should create fact and return 201."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.add_fact.return_value = MockFact(id="new-fact", statement="New statement")
        handler._fact_store = mock_store

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/facts",
                    {},
                    make_handler_with_body(
                        {"statement": "New statement", "confidence": 0.9}, "POST"
                    ),
                )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["id"] == "new-fact"


class TestKnowledgeHandlerUpdateFact:
    """Tests for PUT /api/knowledge/facts/:id endpoint."""

    def test_update_fact_requires_auth(self):
        """Should require authentication for fact update."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, MagicMock(status_code=401))

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/facts/fact-123",
                    {},
                    make_handler_with_body({"confidence": 0.9}, "PUT"),
                )

        assert result is not None
        assert result.status_code == 401

    def test_update_fact_not_found(self):
        """Should return 404 for non-existent fact."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.update_fact.return_value = None
        handler._fact_store = mock_store

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/facts/nonexistent",
                    {},
                    make_handler_with_body({"confidence": 0.9}, "PUT"),
                )

        assert result is not None
        assert result.status_code == 404


class TestKnowledgeHandlerDeleteFact:
    """Tests for DELETE /api/knowledge/facts/:id endpoint."""

    def test_delete_fact_requires_auth(self):
        """Should require authentication for fact deletion."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, MagicMock(status_code=401))

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                mock_handler = MockHandler(command="DELETE")
                result = handler.handle(
                    "/api/knowledge/facts/fact-123",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 401

    def test_delete_fact_success(self):
        """Should delete fact and return success."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.delete_fact.return_value = True
        handler._fact_store = mock_store

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                mock_handler = MockHandler(command="DELETE")
                result = handler.handle(
                    "/api/knowledge/facts/fact-123",
                    {},
                    mock_handler,
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deleted"] is True


class TestKnowledgeHandlerQuery:
    """Tests for POST /api/knowledge/query endpoint."""

    def test_query_requires_question(self):
        """Should require question in request body."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/query",
                {},
                make_handler_with_body({}, "POST"),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"Question is required" in result.body

    def test_query_success(self):
        """Should execute query and return results."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=MockQueryResult())
        handler._query_engine = mock_engine

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/query",
                {},
                make_handler_with_body({"question": "What is the answer?"}, "POST"),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["answer"] == "Test answer"


class TestKnowledgeHandlerSearch:
    """Tests for GET /api/knowledge/search endpoint."""

    def test_search_requires_query(self):
        """Should require 'q' parameter."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/search",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"Query parameter 'q' is required" in result.body

    def test_search_success(self):
        """Should search and return results."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(
            return_value=[MockSearchResult(), MockSearchResult(chunk_id="chunk-2")]
        )
        handler._query_engine = mock_engine

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/search",
                {"q": "test query"},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 2
        assert len(body["results"]) == 2


class TestKnowledgeHandlerRelations:
    """Tests for fact relations endpoints."""

    def test_get_relations(self):
        """Should return fact relations."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_fact.return_value = MockFact()
        mock_store.get_relations.return_value = [MockRelation()]
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/fact-123/relations",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_add_relation(self):
        """Should add relation between facts."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_fact.return_value = MockFact()
        mock_store.add_relation.return_value = MockRelation()
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/fact-1/relations",
                {},
                make_handler_with_body(
                    {
                        "target_fact_id": "fact-2",
                        "relation_type": "supports",
                    },
                    "POST",
                ),
            )

        assert result is not None
        assert result.status_code == 201

    def test_add_relation_missing_target(self):
        """Should require target_fact_id."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/fact-1/relations",
                {},
                make_handler_with_body({"relation_type": "supports"}, "POST"),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"target_fact_id is required" in result.body


class TestKnowledgeHandlerContradictions:
    """Tests for contradictions endpoint."""

    def test_get_contradictions(self):
        """Should return contradicting facts."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_fact.return_value = MockFact()
        mock_store.get_contradictions.return_value = [
            MockFact(id="contra-1", statement="Contradicting fact")
        ]
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/facts/fact-123/contradictions",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1


class TestKnowledgeHandlerStats:
    """Tests for stats endpoint."""

    def test_get_stats(self):
        """Should return knowledge base statistics."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})
        mock_store = MagicMock()
        mock_store.get_statistics.return_value = {
            "total_facts": 100,
            "verified_facts": 50,
        }
        handler._fact_store = mock_store

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/stats",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_facts"] == 100


class TestKnowledgeHandlerRateLimit:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self):
        """Should return 429 when rate limit exceeded."""
        from aragora.server.handlers.knowledge import KnowledgeHandler

        handler = KnowledgeHandler({})

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = handler.handle(
                "/api/knowledge/facts",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 429
        assert b"Rate limit exceeded" in result.body


# =============================================================================
# KnowledgeMoundHandler Tests
# =============================================================================


@dataclass
class MockKnowledgeNode:
    """Mock Knowledge Node for testing."""

    id: str = "node-123"
    node_type: str = "fact"
    content: str = "Test content"
    confidence: float = 0.8
    workspace_id: str = "default"
    topics: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "content": self.content,
            "confidence": self.confidence,
            "workspace_id": self.workspace_id,
            "topics": self.topics,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class MockRelationship:
    """Mock Relationship for testing."""

    id: str = "rel-123"
    from_node_id: str = "node-1"
    to_node_id: str = "node-2"
    relationship_type: str = "supports"
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    metadata: Optional[dict] = None


@dataclass
class MockQuerySemanticResult:
    """Mock semantic query result."""

    query: str = "test query"
    nodes: list = field(default_factory=list)
    total_count: int = 0
    processing_time_ms: float = 10.5


@dataclass
class MockGraphResult:
    """Mock graph query result."""

    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0


@dataclass
class MockCultureProfile:
    """Mock culture profile."""

    workspace_id: str = "default"
    patterns: dict = field(default_factory=dict)
    generated_at: Optional[datetime] = None
    total_observations: int = 0


@dataclass
class MockStaleItem:
    """Mock stale knowledge item."""

    node_id: str = "node-123"
    staleness_score: float = 0.7
    reasons: list = field(default_factory=list)
    last_validated_at: Optional[datetime] = None
    recommended_action: str = "revalidate"


class TestKnowledgeMoundHandlerInit:
    """Tests for KnowledgeMoundHandler initialization."""

    def test_init_with_server_context(self):
        """Should initialize with server context."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        context = {"some": "context"}
        handler = KnowledgeMoundHandler(context)

        assert handler.ctx == context
        assert handler._mound is None
        assert handler._mound_initialized is False


class TestKnowledgeMoundHandlerCanHandle:
    """Tests for can_handle path matching."""

    def test_matches_mound_paths(self):
        """Should match mound API paths."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})

        assert handler.can_handle("/api/knowledge/mound/query")
        assert handler.can_handle("/api/knowledge/mound/nodes")
        assert handler.can_handle("/api/knowledge/mound/nodes/node-123")
        assert handler.can_handle("/api/knowledge/mound/relationships")
        assert handler.can_handle("/api/knowledge/mound/stats")
        assert handler.can_handle("/api/knowledge/mound/graph/node-123")

    def test_rejects_non_mound_paths(self):
        """Should reject non-mound paths."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})

        assert not handler.can_handle("/api/knowledge/facts")
        assert not handler.can_handle("/api/debates")


class TestKnowledgeMoundHandlerQuery:
    """Tests for mound query endpoint."""

    def test_query_requires_query_field(self):
        """Should require query in request body."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/query",
                {},
                make_handler_with_body({}, "POST"),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"Query is required" in result.body

    def test_query_success(self):
        """Should execute semantic query."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(
            return_value=MockQuerySemanticResult(
                query="test",
                nodes=[MockKnowledgeNode()],
                total_count=1,
            )
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/query",
                {},
                make_handler_with_body({"query": "test query"}, "POST"),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_count"] == 1


class TestKnowledgeMoundHandlerNodes:
    """Tests for mound nodes endpoints."""

    def test_list_nodes(self):
        """Should list knowledge nodes."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.query_nodes = AsyncMock(
            return_value=[MockKnowledgeNode(), MockKnowledgeNode(id="node-456")]
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/nodes",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 2

    def test_get_node(self):
        """Should get specific node."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_node = AsyncMock(
            return_value=MockKnowledgeNode(id="node-123", content="Test")
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/nodes/node-123",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["id"] == "node-123"

    def test_get_node_not_found(self):
        """Should return 404 for non-existent node."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_node = AsyncMock(return_value=None)
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/nodes/nonexistent",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 404

    def test_create_node_requires_auth(self):
        """Should require authentication for node creation."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, MagicMock(status_code=401))

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/mound/nodes",
                    {},
                    make_handler_with_body({"content": "Test"}, "POST"),
                )

        assert result is not None
        assert result.status_code == 401

    def test_create_node_requires_content(self):
        """Should require content in request body."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        handler._mound = mock_mound

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/mound/nodes",
                    {},
                    make_handler_with_body({}, "POST"),
                )

        assert result is not None
        assert result.status_code == 400
        assert b"Content is required" in result.body


class TestKnowledgeMoundHandlerRelationships:
    """Tests for mound relationships endpoints."""

    def test_get_node_relationships(self):
        """Should return node relationships."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_node = AsyncMock(return_value=MockKnowledgeNode())
        mock_mound.get_relationships = AsyncMock(return_value=[MockRelationship()])
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/nodes/node-123/relationships",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_create_relationship_requires_auth(self):
        """Should require authentication for relationship creation."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, MagicMock(status_code=401))

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/mound/relationships",
                    {},
                    make_handler_with_body(
                        {
                            "from_node_id": "node-1",
                            "to_node_id": "node-2",
                            "relationship_type": "supports",
                        },
                        "POST",
                    ),
                )

        assert result is not None
        assert result.status_code == 401

    def test_create_relationship_validates_type(self):
        """Should validate relationship type."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        handler._mound = mock_mound

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(), None)

            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/mound/relationships",
                    {},
                    make_handler_with_body(
                        {
                            "from_node_id": "node-1",
                            "to_node_id": "node-2",
                            "relationship_type": "invalid-type",
                        },
                        "POST",
                    ),
                )

        assert result is not None
        assert result.status_code == 400
        assert b"Invalid relationship_type" in result.body


class TestKnowledgeMoundHandlerGraph:
    """Tests for graph traversal endpoints."""

    def test_graph_traversal(self):
        """Should traverse graph from node."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.query_graph = AsyncMock(
            return_value=[MockKnowledgeNode(), MockKnowledgeNode(id="node-456")]
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/graph/node-123",
                {"depth": "2"},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 2

    def test_graph_traversal_validates_direction(self):
        """Should validate direction parameter."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/graph/node-123",
                {"direction": "invalid"},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"direction must be" in result.body


class TestKnowledgeMoundHandlerStats:
    """Tests for mound stats endpoint."""

    def test_get_mound_stats(self):
        """Should return mound statistics."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_stats = AsyncMock(
            return_value={
                "total_nodes": 100,
                "total_relationships": 50,
            }
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/stats",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_nodes"] == 100


class TestKnowledgeMoundHandlerCulture:
    """Tests for culture management endpoints."""

    def test_get_culture_profile(self):
        """Should return culture profile."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_culture_profile = AsyncMock(
            return_value=MockCultureProfile(
                workspace_id="default",
                total_observations=50,
            )
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/culture",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["workspace_id"] == "default"

    def test_add_culture_document_requires_content(self):
        """Should require content for culture document."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/culture/documents",
                {},
                make_handler_with_body({}, "POST"),
            )

        assert result is not None
        assert result.status_code == 400
        assert b"Content is required" in result.body


class TestKnowledgeMoundHandlerStaleness:
    """Tests for staleness detection endpoints."""

    def test_get_stale_knowledge(self):
        """Should return stale knowledge items."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.get_stale_knowledge = AsyncMock(
            return_value=[MockStaleItem(node_id="node-1", staleness_score=0.8)]
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/stale",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    def test_revalidate_node(self):
        """Should revalidate a node."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.mark_validated = AsyncMock()
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/revalidate/node-123",
                {},
                make_handler_with_body({"validator": "api"}, "POST"),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["validated"] is True


class TestKnowledgeMoundHandlerSync:
    """Tests for sync endpoints."""

    def test_sync_continuum_not_implemented(self):
        """Should handle unimplemented sync gracefully."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.sync_from_continuum = AsyncMock(side_effect=AttributeError())
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/sync/continuum",
                {},
                make_handler_with_body({}, "POST"),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "not yet implemented" in body["message"]


class TestKnowledgeMoundHandlerExport:
    """Tests for graph export endpoints."""

    def test_export_d3(self):
        """Should export graph as D3 JSON."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.export_graph_d3 = AsyncMock(
            return_value={
                "nodes": [{"id": "node-1"}],
                "links": [{"source": "node-1", "target": "node-2"}],
            }
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/export/d3",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["format"] == "d3"
        assert len(body["nodes"]) == 1

    def test_export_graphml(self):
        """Should export graph as GraphML."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        mock_mound = MagicMock()
        mock_mound.export_graph_graphml = AsyncMock(
            return_value='<?xml version="1.0"?><graphml></graphml>'
        )
        handler._mound = mock_mound

        with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/knowledge/mound/export/graphml",
                {},
                MockHandler(),
            )

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/xml"
        assert "graphml" in result.body


class TestKnowledgeMoundHandlerMoundUnavailable:
    """Tests for handling when Knowledge Mound is unavailable."""

    def test_query_mound_unavailable(self):
        """Should return 503 when mound is unavailable."""
        from aragora.server.handlers.knowledge import KnowledgeMoundHandler

        handler = KnowledgeMoundHandler({})
        handler._mound = None

        # Mock _get_mound to return None
        with patch.object(handler, "_get_mound", return_value=None):
            with patch("aragora.server.handlers.knowledge._knowledge_limiter") as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                result = handler.handle(
                    "/api/knowledge/mound/query",
                    {},
                    make_handler_with_body({"query": "test"}, "POST"),
                )

        assert result is not None
        assert result.status_code == 503
        assert b"Knowledge Mound not available" in result.body
