"""
Knowledge Base endpoint handlers.

Provides API endpoints for the enterprise document auditing knowledge base:

- POST /api/knowledge/query - Natural language query against dataset
- GET /api/knowledge/facts - List facts with filtering
- GET /api/knowledge/facts/:id - Get specific fact
- POST /api/knowledge/facts - Add a new fact
- PUT /api/knowledge/facts/:id - Update a fact
- DELETE /api/knowledge/facts/:id - Delete a fact
- POST /api/knowledge/facts/:id/verify - Verify fact with agents
- GET /api/knowledge/facts/:id/contradictions - Get contradicting facts
- GET /api/knowledge/facts/:id/relations - Get fact relations
- POST /api/knowledge/facts/relations - Add relation between facts
- GET /api/knowledge/search - Search chunks via embeddings
- GET /api/knowledge/stats - Get knowledge base statistics
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.knowledge import (
    DatasetQueryEngine,
    FactFilters,
    FactRelationType,
    FactStore,
    InMemoryEmbeddingService,
    InMemoryFactStore,
    QueryOptions,
    SimpleQueryEngine,
    ValidationStatus,
)

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bool_param,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    ttl_cache,
)
from .utils.rate_limit import RateLimiter, get_client_ip

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Rate limiter for knowledge endpoints (60 requests per minute)
_knowledge_limiter = RateLimiter(requests_per_minute=60)

# Cache TTLs
CACHE_TTL_FACTS = 60  # 1 minute for fact listings
CACHE_TTL_STATS = 300  # 5 minutes for statistics


class KnowledgeHandler(BaseHandler):
    """Handler for knowledge base API endpoints."""

    ROUTES = [
        "/api/knowledge/query",
        "/api/knowledge/facts",
        "/api/knowledge/search",
        "/api/knowledge/stats",
    ]

    def __init__(self, server_context: dict):
        """Initialize knowledge handler.

        Args:
            server_context: Server context with shared resources
        """
        super().__init__(server_context)

        # Initialize stores - use in-memory for now, can be configured later
        self._fact_store: Optional[FactStore | InMemoryFactStore] = None
        self._query_engine: Optional[DatasetQueryEngine | SimpleQueryEngine] = None

    def _get_fact_store(self) -> FactStore | InMemoryFactStore:
        """Get or create fact store."""
        if self._fact_store is None:
            try:
                self._fact_store = FactStore()
            except Exception as e:
                logger.warning(f"Failed to create FactStore, using in-memory: {e}")
                self._fact_store = InMemoryFactStore()
        return self._fact_store

    def _get_query_engine(self) -> DatasetQueryEngine | SimpleQueryEngine:
        """Get or create query engine."""
        if self._query_engine is None:
            fact_store = self._get_fact_store()
            embedding_service = InMemoryEmbeddingService()
            self._query_engine = SimpleQueryEngine(
                fact_store=fact_store,
                embedding_service=embedding_service,
            )
        return self._query_engine

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle dynamic routes
        if path.startswith("/api/knowledge/facts/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route knowledge requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _knowledge_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for knowledge endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Query endpoint (POST)
        if path == "/api/knowledge/query":
            return self._handle_query(query_params, handler)

        # Facts listing (GET) or creation (POST)
        if path == "/api/knowledge/facts":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_fact(handler)
            return self._handle_list_facts(query_params)

        # Search chunks
        if path == "/api/knowledge/search":
            return self._handle_search(query_params)

        # Statistics
        if path == "/api/knowledge/stats":
            workspace_id = get_bounded_string_param(
                query_params, "workspace_id", None, max_length=100
            )
            return self._handle_stats(workspace_id)

        # Dynamic fact routes
        if path.startswith("/api/knowledge/facts/"):
            return self._handle_fact_routes(path, query_params, handler)

        return None

    def _handle_fact_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/knowledge/facts/:id/* routes."""
        parts = path.strip("/").split("/")

        # /api/knowledge/facts/:id
        if len(parts) == 4:
            fact_id = parts[3]
            method = getattr(handler, "command", "GET")
            if method == "GET":
                return self._handle_get_fact(fact_id)
            elif method == "PUT":
                return self._handle_update_fact(fact_id, handler)
            elif method == "DELETE":
                return self._handle_delete_fact(fact_id)

        # /api/knowledge/facts/:id/verify
        if len(parts) == 5 and parts[4] == "verify":
            fact_id = parts[3]
            return self._handle_verify_fact(fact_id, handler)

        # /api/knowledge/facts/:id/contradictions
        if len(parts) == 5 and parts[4] == "contradictions":
            fact_id = parts[3]
            return self._handle_get_contradictions(fact_id)

        # /api/knowledge/facts/:id/relations
        if len(parts) == 5 and parts[4] == "relations":
            fact_id = parts[3]
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_add_relation(fact_id, handler)
            return self._handle_get_relations(fact_id, query_params)

        # /api/knowledge/facts/relations (POST - add relation)
        if len(parts) == 4 and parts[3] == "relations":
            return self._handle_add_relation_bulk(handler)

        return error_response("Unknown endpoint", 404)

    @handle_errors("knowledge query")
    def _handle_query(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/query - Natural language query."""
        import asyncio

        # Read request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        question = data.get("question", "")
        if not question:
            return error_response("Question is required", 400)

        workspace_id = data.get("workspace_id", "default")
        options_data = data.get("options", {})

        options = QueryOptions(
            max_chunks=options_data.get("max_chunks", 10),
            search_alpha=options_data.get("search_alpha", 0.5),
            use_agents=options_data.get("use_agents", False),
            extract_facts=options_data.get("extract_facts", True),
            include_citations=options_data.get("include_citations", True),
        )

        engine = self._get_query_engine()

        # Run async query
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(engine.query(question, workspace_id, options))
            loop.close()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return error_response(f"Query failed: {e}", 500)

        return json_response(result.to_dict())

    @ttl_cache(ttl_seconds=CACHE_TTL_FACTS, key_prefix="knowledge_facts", skip_first=True)
    @handle_errors("list facts")
    def _handle_list_facts(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/facts - List facts."""
        workspace_id = get_bounded_string_param(query_params, "workspace_id", None, max_length=100)
        topic = get_bounded_string_param(query_params, "topic", None, max_length=200)
        min_confidence = get_bounded_float_param(
            query_params, "min_confidence", 0.0, min_val=0.0, max_val=1.0
        )
        status = get_bounded_string_param(query_params, "status", None, max_length=50)
        include_superseded = get_bool_param(query_params, "include_superseded", False)
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)

        filters = FactFilters(
            workspace_id=workspace_id,
            topics=[topic] if topic else None,
            min_confidence=min_confidence,
            validation_status=ValidationStatus(status) if status else None,
            include_superseded=include_superseded,
            limit=limit,
            offset=offset,
        )

        store = self._get_fact_store()
        facts = store.list_facts(filters)

        return json_response(
            {
                "facts": [f.to_dict() for f in facts],
                "total": len(facts),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get fact")
    def _handle_get_fact(self, fact_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id - Get specific fact."""
        store = self._get_fact_store()
        fact = store.get_fact(fact_id)

        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response(fact.to_dict())

    @handle_errors("create fact")
    def _handle_create_fact(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts - Create new fact."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        statement = data.get("statement", "")
        if not statement:
            return error_response("Statement is required", 400)

        workspace_id = data.get("workspace_id", "default")

        store = self._get_fact_store()
        fact = store.add_fact(
            statement=statement,
            workspace_id=workspace_id,
            evidence_ids=data.get("evidence_ids", []),
            source_documents=data.get("source_documents", []),
            confidence=data.get("confidence", 0.5),
            topics=data.get("topics", []),
            metadata=data.get("metadata", {}),
        )

        return json_response(fact.to_dict(), status=201)

    @handle_errors("update fact")
    def _handle_update_fact(self, fact_id: str, handler: Any) -> HandlerResult:
        """Handle PUT /api/knowledge/facts/:id - Update fact."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        store = self._get_fact_store()

        # Build update kwargs
        kwargs = {}
        if "confidence" in data:
            kwargs["confidence"] = data["confidence"]
        if "validation_status" in data:
            kwargs["validation_status"] = ValidationStatus(data["validation_status"])
        if "evidence_ids" in data:
            kwargs["evidence_ids"] = data["evidence_ids"]
        if "topics" in data:
            kwargs["topics"] = data["topics"]
        if "metadata" in data:
            kwargs["metadata"] = data["metadata"]
        if "superseded_by" in data:
            kwargs["superseded_by"] = data["superseded_by"]

        updated = store.update_fact(fact_id, **kwargs)

        if not updated:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response(updated.to_dict())

    @handle_errors("delete fact")
    def _handle_delete_fact(self, fact_id: str) -> HandlerResult:
        """Handle DELETE /api/knowledge/facts/:id - Delete fact."""
        store = self._get_fact_store()
        deleted = store.delete_fact(fact_id)

        if not deleted:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response({"deleted": True, "fact_id": fact_id})

    @handle_errors("verify fact")
    def _handle_verify_fact(self, fact_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/:id/verify - Verify fact."""
        import asyncio

        store = self._get_fact_store()
        fact = store.get_fact(fact_id)

        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        engine = self._get_query_engine()

        # Verify requires DatasetQueryEngine with agents
        if not isinstance(engine, DatasetQueryEngine):
            return error_response("Agent verification not available", 503)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            verified = loop.run_until_complete(engine.verify_fact(fact_id))
            loop.close()
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return error_response(f"Verification failed: {e}", 500)

        return json_response(verified.to_dict())

    @handle_errors("get contradictions")
    def _handle_get_contradictions(self, fact_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id/contradictions."""
        store = self._get_fact_store()

        fact = store.get_fact(fact_id)
        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        contradictions = store.get_contradictions(fact_id)

        return json_response(
            {
                "fact_id": fact_id,
                "contradictions": [c.to_dict() for c in contradictions],
                "count": len(contradictions),
            }
        )

    @handle_errors("get relations")
    def _handle_get_relations(self, fact_id: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id/relations."""
        store = self._get_fact_store()

        fact = store.get_fact(fact_id)
        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        relation_type_str = get_bounded_string_param(query_params, "type", None, max_length=50)
        relation_type = FactRelationType(relation_type_str) if relation_type_str else None

        as_source = get_bool_param(query_params, "as_source", True)
        as_target = get_bool_param(query_params, "as_target", True)

        relations = store.get_relations(
            fact_id,
            relation_type=relation_type,
            as_source=as_source,
            as_target=as_target,
        )

        return json_response(
            {
                "fact_id": fact_id,
                "relations": [r.to_dict() for r in relations],
                "count": len(relations),
            }
        )

    @handle_errors("add relation")
    def _handle_add_relation(self, fact_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/:id/relations - Add relation from fact."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        target_fact_id = data.get("target_fact_id")
        if not target_fact_id:
            return error_response("target_fact_id is required", 400)

        relation_type_str = data.get("relation_type")
        if not relation_type_str:
            return error_response("relation_type is required", 400)

        try:
            relation_type = FactRelationType(relation_type_str)
        except ValueError:
            return error_response(f"Invalid relation_type: {relation_type_str}", 400)

        store = self._get_fact_store()

        # Verify both facts exist
        if not store.get_fact(fact_id):
            return error_response(f"Source fact not found: {fact_id}", 404)
        if not store.get_fact(target_fact_id):
            return error_response(f"Target fact not found: {target_fact_id}", 404)

        relation = store.add_relation(
            source_fact_id=fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=data.get("confidence", 0.5),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata"),
        )

        return json_response(relation.to_dict(), status=201)

    @handle_errors("add relation bulk")
    def _handle_add_relation_bulk(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/relations - Add relation between facts."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        source_fact_id = data.get("source_fact_id")
        target_fact_id = data.get("target_fact_id")
        relation_type_str = data.get("relation_type")

        if not source_fact_id:
            return error_response("source_fact_id is required", 400)
        if not target_fact_id:
            return error_response("target_fact_id is required", 400)
        if not relation_type_str:
            return error_response("relation_type is required", 400)

        try:
            relation_type = FactRelationType(relation_type_str)
        except ValueError:
            return error_response(f"Invalid relation_type: {relation_type_str}", 400)

        store = self._get_fact_store()

        relation = store.add_relation(
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=data.get("confidence", 0.5),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata"),
        )

        return json_response(relation.to_dict(), status=201)

    @handle_errors("search chunks")
    def _handle_search(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/search - Search chunks."""
        import asyncio

        query = get_bounded_string_param(query_params, "q", "", max_length=500)
        if not query:
            return error_response("Query parameter 'q' is required", 400)

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )
        limit = get_clamped_int_param(query_params, "limit", 10, min_val=1, max_val=50)

        engine = self._get_query_engine()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(engine.search(query, workspace_id, limit))
            loop.close()
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return error_response(f"Search failed: {e}", 500)

        return json_response(
            {
                "query": query,
                "workspace_id": workspace_id,
                "results": [r.to_dict() for r in results],
                "count": len(results),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_STATS, key_prefix="knowledge_stats", skip_first=True)
    @handle_errors("get stats")
    def _handle_stats(self, workspace_id: Optional[str]) -> HandlerResult:
        """Handle GET /api/knowledge/stats - Get statistics."""
        store = self._get_fact_store()
        stats = store.get_statistics(workspace_id)

        return json_response(
            {
                "workspace_id": workspace_id,
                **stats,
            }
        )
