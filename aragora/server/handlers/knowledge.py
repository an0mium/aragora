"""
Knowledge Base endpoint handlers.

Provides API endpoints for the enterprise document auditing knowledge base:

Facts API (FactStore):
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

Knowledge Mound API (unified knowledge storage):
- POST /api/knowledge/mound/query - Semantic query against knowledge mound
- POST /api/knowledge/mound/nodes - Add a knowledge node
- GET /api/knowledge/mound/nodes/:id - Get specific node
- GET /api/knowledge/mound/nodes - List/filter nodes
- POST /api/knowledge/mound/relationships - Add relationship between nodes
- GET /api/knowledge/mound/graph/:id - Get graph traversal from node
- GET /api/knowledge/mound/stats - Get mound statistics
- POST /api/knowledge/mound/index/repository - Index a repository
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

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


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from sync context safely.

    Uses asyncio.run() which creates a new event loop, runs the coroutine,
    and closes the loop. This is the recommended pattern for calling async
    code from sync handlers.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    return asyncio.run(coro)


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
                return self._handle_delete_fact(fact_id, handler)

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
            result = _run_async(engine.query(question, workspace_id, options))
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
        # Require authentication for fact creation
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

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
        # Require authentication for fact updates
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

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
    def _handle_delete_fact(self, fact_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/knowledge/facts/:id - Delete fact."""
        # Require authentication for fact deletion
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

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
            # Queue for later verification when capability becomes available
            # Update fact metadata to track pending verification
            store.update_fact(
                fact_id,
                metadata={
                    **fact.metadata,
                    "_pending_verification": True,
                    "_verification_queued_at": __import__("time").time(),
                },
            )
            return json_response({
                "fact_id": fact_id,
                "verified": None,
                "status": "queued",
                "message": "Agent verification not currently available. Fact queued for verification when capability becomes available.",
            })

        try:
            verified = _run_async(engine.verify_fact(fact_id))
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
            results = _run_async(engine.search(query, workspace_id, limit))
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


class KnowledgeMoundHandler(BaseHandler):
    """Handler for Knowledge Mound API endpoints (unified knowledge storage).

    Extended endpoints for Phase A1:
    - Culture management (get profile, add documents, promote knowledge)
    - Staleness detection and revalidation
    - Sync with legacy memory systems (ContinuumMemory, ConsensusMemory, FactStore)
    - Enhanced graph traversal (lineage, related nodes)
    """

    ROUTES = [
        "/api/knowledge/mound/query",
        "/api/knowledge/mound/nodes",
        "/api/knowledge/mound/relationships",
        "/api/knowledge/mound/stats",
        "/api/knowledge/mound/culture",
        "/api/knowledge/mound/culture/*",
        "/api/knowledge/mound/stale",
        "/api/knowledge/mound/revalidate/*",
        "/api/knowledge/mound/schedule-revalidation",
        "/api/knowledge/mound/sync/*",
        "/api/knowledge/mound/graph/*/lineage",
        "/api/knowledge/mound/graph/*/related",
    ]

    def __init__(self, server_context: dict):
        """Initialize knowledge mound handler."""
        super().__init__(server_context)
        self._mound = None
        self._mound_initialized = False

    def _get_mound(self):
        """Get or create Knowledge Mound instance."""
        if self._mound is None:
            import asyncio
            from aragora.knowledge.mound import KnowledgeMound

            self._mound = KnowledgeMound(workspace_id="default")
            # Initialize synchronously for handler use
            try:
                _run_async(self._mound.initialize())
                self._mound_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Knowledge Mound: {e}")
                self._mound = None
        return self._mound

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/knowledge/mound/")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route knowledge mound requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _knowledge_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for mound endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Semantic query
        if path == "/api/knowledge/mound/query":
            return self._handle_mound_query(handler)

        # Nodes listing/creation
        if path == "/api/knowledge/mound/nodes":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_node(handler)
            return self._handle_list_nodes(query_params)

        # Relationships
        if path == "/api/knowledge/mound/relationships":
            return self._handle_create_relationship(handler)

        # Statistics
        if path == "/api/knowledge/mound/stats":
            return self._handle_mound_stats(query_params)

        # Dynamic routes
        if path.startswith("/api/knowledge/mound/nodes/"):
            return self._handle_node_routes(path, query_params, handler)

        if path.startswith("/api/knowledge/mound/graph/"):
            # Check for lineage or related sub-routes
            if "/lineage" in path:
                return self._handle_graph_lineage(path, query_params)
            if "/related" in path:
                return self._handle_graph_related(path, query_params)
            return self._handle_graph_traversal(path, query_params)

        if path == "/api/knowledge/mound/index/repository":
            return self._handle_index_repository(handler)

        # Culture endpoints
        if path == "/api/knowledge/mound/culture":
            return self._handle_get_culture(query_params)

        if path == "/api/knowledge/mound/culture/documents":
            return self._handle_add_culture_document(handler)

        if path == "/api/knowledge/mound/culture/promote":
            return self._handle_promote_to_culture(handler)

        # Staleness endpoints
        if path == "/api/knowledge/mound/stale":
            return self._handle_get_stale(query_params)

        if path.startswith("/api/knowledge/mound/revalidate/"):
            node_id = path.split("/")[-1]
            return self._handle_revalidate_node(node_id, handler)

        if path == "/api/knowledge/mound/schedule-revalidation":
            return self._handle_schedule_revalidation(handler)

        # Sync endpoints
        if path == "/api/knowledge/mound/sync/continuum":
            return self._handle_sync_continuum(handler)

        if path == "/api/knowledge/mound/sync/consensus":
            return self._handle_sync_consensus(handler)

        if path == "/api/knowledge/mound/sync/facts":
            return self._handle_sync_facts(handler)

        return None

    def _handle_node_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/knowledge/mound/nodes/:id routes."""
        parts = path.strip("/").split("/")
        if len(parts) >= 5:
            node_id = parts[4]
            return self._handle_get_node(node_id)
        return error_response("Invalid node path", 400)

    @handle_errors("mound query")
    def _handle_mound_query(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/query - Semantic query."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        query = data.get("query", "")
        if not query:
            return error_response("Query is required", 400)

        workspace_id = data.get("workspace_id", "default")
        limit = data.get("limit", 10)
        node_types = data.get("node_types")
        min_confidence = data.get("min_confidence", 0.0)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.query_semantic(
                    query=query,
                    limit=limit,
                    node_types=node_types,
                    min_confidence=min_confidence,
                    workspace_id=workspace_id,
                )
            )
        except Exception as e:
            logger.error(f"Mound query failed: {e}")
            return error_response(f"Query failed: {e}", 500)

        return json_response({
            "query": result.query,
            "nodes": [n.to_dict() for n in result.nodes],
            "total_count": result.total_count,
            "processing_time_ms": result.processing_time_ms,
        })

    @handle_errors("create node")
    def _handle_create_node(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/nodes - Create knowledge node."""
        # Require authentication for node creation
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        import asyncio
        from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
        from aragora.memory.tier_manager import MemoryTier

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        content = data.get("content", "")
        if not content:
            return error_response("Content is required", 400)

        node_type = data.get("node_type", "fact")
        if node_type not in ("fact", "claim", "memory", "evidence", "consensus", "entity"):
            return error_response(f"Invalid node_type: {node_type}", 400)

        workspace_id = data.get("workspace_id", "default")

        # Build provenance if provided
        provenance = None
        if data.get("source"):
            source = data["source"]
            try:
                provenance = ProvenanceChain(
                    source_type=ProvenanceType(source.get("type", "user")),
                    source_id=source.get("id", ""),
                    user_id=source.get("user_id"),
                    agent_id=source.get("agent_id"),
                    debate_id=source.get("debate_id"),
                    document_id=source.get("document_id"),
                )
            except ValueError as e:
                return error_response(f"Invalid source type: {e}", 400)

        node = KnowledgeNode(
            node_type=node_type,
            content=content,
            confidence=data.get("confidence", 0.5),
            provenance=provenance,
            tier=MemoryTier(data.get("tier", "slow")),
            workspace_id=workspace_id,
            topics=data.get("topics", []),
            metadata=data.get("metadata", {}),
        )

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            node_id = _run_async(mound.add_node(node))
            # Fetch the saved node
            saved_node = _run_async(mound.get_node(node_id))
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return error_response(f"Failed to create node: {e}", 500)

        return json_response(saved_node.to_dict() if saved_node else {"id": node_id}, status=201)

    @handle_errors("get node")
    def _handle_get_node(self, node_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/mound/nodes/:id - Get specific node."""
        import asyncio

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            node = _run_async(mound.get_node(node_id))
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return error_response(f"Failed to get node: {e}", 500)

        if not node:
            return error_response(f"Node not found: {node_id}", 404)

        return json_response(node.to_dict())

    @handle_errors("list nodes")
    def _handle_list_nodes(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/nodes - List/filter nodes."""
        import asyncio
        from aragora.memory.tier_manager import MemoryTier

        workspace_id = get_bounded_string_param(query_params, "workspace_id", "default", max_length=100)
        node_types_str = get_bounded_string_param(query_params, "node_types", None, max_length=200)
        node_types = node_types_str.split(",") if node_types_str else None
        min_confidence = get_bounded_float_param(query_params, "min_confidence", 0.0, min_val=0.0, max_val=1.0)
        tier_str = get_bounded_string_param(query_params, "tier", None, max_length=20)
        tier = MemoryTier(tier_str) if tier_str else None
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            nodes = _run_async(
                mound.query_nodes(
                    workspace_id=workspace_id,
                    node_types=node_types,
                    min_confidence=min_confidence,
                    tier=tier,
                    limit=limit,
                    offset=offset,
                )
            )
        except Exception as e:
            logger.error(f"Failed to list nodes: {e}")
            return error_response(f"Failed to list nodes: {e}", 500)

        return json_response({
            "nodes": [n.to_dict() for n in nodes],
            "count": len(nodes),
            "limit": limit,
            "offset": offset,
        })

    @handle_errors("create relationship")
    def _handle_create_relationship(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/relationships - Add relationship."""
        # Require authentication for relationship creation
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        from_node_id = data.get("from_node_id")
        to_node_id = data.get("to_node_id")
        relationship_type = data.get("relationship_type")

        if not from_node_id:
            return error_response("from_node_id is required", 400)
        if not to_node_id:
            return error_response("to_node_id is required", 400)
        if not relationship_type:
            return error_response("relationship_type is required", 400)

        valid_types = ("supports", "contradicts", "derived_from", "related_to", "supersedes")
        if relationship_type not in valid_types:
            return error_response(f"Invalid relationship_type. Must be one of: {valid_types}", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            rel_id = _run_async(
                mound.add_relationship(
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    relationship_type=relationship_type,
                    strength=data.get("strength", 1.0),
                    created_by=data.get("created_by", ""),
                    metadata=data.get("metadata"),
                )
            )
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return error_response(f"Failed to create relationship: {e}", 500)

        return json_response({
            "id": rel_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relationship_type": relationship_type,
        }, status=201)

    @handle_errors("graph traversal")
    def _handle_graph_traversal(self, path: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id - Graph traversal."""
        import asyncio

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type = get_bounded_string_param(query_params, "relationship_type", None, max_length=50)
        depth = get_clamped_int_param(query_params, "depth", 2, min_val=1, max_val=5)
        direction = get_bounded_string_param(query_params, "direction", "outgoing", max_length=20)

        if direction not in ("outgoing", "incoming", "both"):
            return error_response("direction must be 'outgoing', 'incoming', or 'both'", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            nodes = _run_async(
                mound.query_graph(
                    start_node_id=node_id,
                    relationship_type=relationship_type,
                    depth=depth,
                    direction=direction,
                )
            )
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return error_response(f"Graph traversal failed: {e}", 500)

        return json_response({
            "start_node_id": node_id,
            "depth": depth,
            "direction": direction,
            "relationship_type": relationship_type,
            "nodes": [n.to_dict() for n in nodes],
            "count": len(nodes),
        })

    @handle_errors("mound stats")
    def _handle_mound_stats(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/stats - Get mound statistics."""
        import asyncio

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            stats = _run_async(mound.get_stats())
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return error_response(f"Failed to get stats: {e}", 500)

        return json_response(stats)

    @handle_errors("index repository")
    def _handle_index_repository(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/index/repository - Index a repository."""
        # Require authentication for repository indexing
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        repo_path = data.get("repo_path")
        if not repo_path:
            return error_response("repo_path is required", 400)

        workspace_id = data.get("workspace_id", "default")

        # Get Knowledge Mound
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.connectors.repository_crawler import (
                CrawlConfig,
                RepositoryCrawler,
            )

            # Build crawl config from request
            config = CrawlConfig(
                include_patterns=data.get("include_patterns", ["*", "**/*"]),
                exclude_patterns=data.get("exclude_patterns", [
                    "**/node_modules/**",
                    "**/.git/**",
                    "**/venv/**",
                    "**/__pycache__/**",
                    "**/.venv/**",
                    "**/dist/**",
                    "**/build/**",
                ]),
                max_file_size_bytes=data.get("max_file_size_bytes", 1_000_000),
                max_files=data.get("max_files", 10_000),
                extract_symbols=data.get("extract_symbols", True),
                extract_dependencies=data.get("extract_dependencies", True),
                extract_docstrings=data.get("extract_docstrings", True),
            )

            # Create crawler and run
            crawler = RepositoryCrawler(config=config, workspace_id=workspace_id)
            crawl_result = _run_async(crawler.crawl(repo_path, incremental=data.get("incremental", True)))

            # Index to Knowledge Mound
            nodes_created = _run_async(crawler.index_to_mound(crawl_result, mound))

            return json_response({
                "status": "completed",
                "repository": crawl_result.repository_name,
                "repository_path": crawl_result.repository_path,
                "workspace_id": workspace_id,
                "total_files": crawl_result.total_files,
                "total_lines": crawl_result.total_lines,
                "total_bytes": crawl_result.total_bytes,
                "nodes_created": nodes_created,
                "file_type_counts": crawl_result.file_type_counts,
                "symbol_counts": crawl_result.symbol_counts,
                "crawl_duration_ms": crawl_result.crawl_duration_ms,
                "errors": crawl_result.errors[:10] if crawl_result.errors else [],
                "warnings": crawl_result.warnings[:10] if crawl_result.warnings else [],
                "git_info": crawl_result.git_info,
            })

        except FileNotFoundError as e:
            return error_response(f"Repository not found: {e}", 404)
        except Exception as e:
            logger.error(f"Failed to index repository: {e}")
            return error_response(f"Failed to index repository: {e}", 500)

    # =========================================================================
    # Culture Management Endpoints (Phase A1)
    # =========================================================================

    @handle_errors("get culture")
    def _handle_get_culture(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/culture - Get organization culture profile."""
        import asyncio

        workspace_id = get_bounded_string_param(query_params, "workspace_id", "default", max_length=100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            culture = _run_async(mound.get_culture_profile(workspace_id))
        except Exception as e:
            logger.error(f"Failed to get culture profile: {e}")
            return error_response(f"Failed to get culture profile: {e}", 500)

        return json_response({
            "workspace_id": culture.workspace_id,
            "patterns": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in culture.patterns.items()},
            "generated_at": culture.generated_at.isoformat() if culture.generated_at else None,
            "total_observations": culture.total_observations,
        })

    @handle_errors("add culture document")
    def _handle_add_culture_document(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/culture/documents - Add culture document."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        content = data.get("content", "")
        if not content:
            return error_response("Content is required", 400)

        workspace_id = data.get("workspace_id", "default")
        document_type = data.get("document_type", "policy")
        metadata = data.get("metadata", {})

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        # Store as a culture-related knowledge node
        try:
            from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
            from aragora.memory.tier_manager import MemoryTier

            provenance = ProvenanceChain(
                source_type=ProvenanceType.USER,
                source_id="culture_document",
            )

            node = KnowledgeNode(
                node_type="culture",
                content=content,
                confidence=1.0,  # Culture documents are authoritative
                provenance=provenance,
                tier=MemoryTier.GLACIAL,  # Long-term storage
                workspace_id=workspace_id,
                topics=["culture", document_type],
                metadata={"document_type": document_type, **metadata},
            )

            node_id = _run_async(mound.add_node(node))
        except Exception as e:
            logger.error(f"Failed to add culture document: {e}")
            return error_response(f"Failed to add culture document: {e}", 500)

        return json_response({
            "node_id": node_id,
            "document_type": document_type,
            "workspace_id": workspace_id,
            "message": "Culture document added successfully",
        }, status=201)

    @handle_errors("promote to culture")
    def _handle_promote_to_culture(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/culture/promote - Promote knowledge to culture."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        node_id = data.get("node_id")
        if not node_id:
            return error_response("node_id is required", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.memory.tier_manager import MemoryTier

            # Update node to culture tier and type
            updated = _run_async(mound.update(node_id, {
                "node_type": "culture",
                "tier": MemoryTier.GLACIAL.value,
                "promoted_to_culture": True,
            }))

            if not updated:
                return error_response(f"Node not found: {node_id}", 404)

        except Exception as e:
            logger.error(f"Failed to promote to culture: {e}")
            return error_response(f"Failed to promote to culture: {e}", 500)

        return json_response({
            "node_id": node_id,
            "promoted": True,
            "message": "Knowledge promoted to culture successfully",
        })

    # =========================================================================
    # Staleness and Revalidation Endpoints (Phase A1)
    # =========================================================================

    @handle_errors("get stale knowledge")
    def _handle_get_stale(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/stale - Get stale knowledge items."""
        import asyncio

        workspace_id = get_bounded_string_param(query_params, "workspace_id", "default", max_length=100)
        threshold = get_bounded_float_param(query_params, "threshold", 0.5, min_val=0.0, max_val=1.0)
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            stale_items = _run_async(
                mound.get_stale_knowledge(
                    threshold=threshold,
                    limit=limit,
                    workspace_id=workspace_id,
                )
            )
        except Exception as e:
            logger.error(f"Failed to get stale knowledge: {e}")
            return error_response(f"Failed to get stale knowledge: {e}", 500)

        return json_response({
            "stale_items": [
                {
                    "node_id": item.node_id,
                    "staleness_score": item.staleness_score,
                    "reasons": [r.value if hasattr(r, 'value') else r for r in item.reasons],
                    "last_validated_at": item.last_validated_at.isoformat() if item.last_validated_at else None,
                    "recommended_action": item.recommended_action,
                }
                for item in stale_items
            ],
            "total": len(stale_items),
            "threshold": threshold,
            "workspace_id": workspace_id,
        })

    @handle_errors("revalidate node")
    def _handle_revalidate_node(self, node_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/revalidate/:id - Trigger revalidation."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        validator = data.get("validator", "api")
        new_confidence = data.get("confidence")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            _run_async(mound.mark_validated(node_id, validator, new_confidence))
        except Exception as e:
            logger.error(f"Failed to revalidate node: {e}")
            return error_response(f"Failed to revalidate node: {e}", 500)

        return json_response({
            "node_id": node_id,
            "validated": True,
            "validator": validator,
            "new_confidence": new_confidence,
            "message": "Node revalidated successfully",
        })

    @handle_errors("schedule revalidation")
    def _handle_schedule_revalidation(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/schedule-revalidation - Schedule batch."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        node_ids = data.get("node_ids", [])
        if not node_ids:
            return error_response("node_ids is required", 400)

        priority = data.get("priority", "low")
        if priority not in ("low", "medium", "high"):
            return error_response("priority must be 'low', 'medium', or 'high'", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            scheduled = _run_async(mound.schedule_revalidation(node_ids, priority))
        except Exception as e:
            logger.error(f"Failed to schedule revalidation: {e}")
            return error_response(f"Failed to schedule revalidation: {e}", 500)

        return json_response({
            "scheduled": scheduled,
            "priority": priority,
            "count": len(scheduled),
            "message": f"Scheduled {len(scheduled)} nodes for revalidation",
        }, status=202)

    # =========================================================================
    # Sync Endpoints (Phase A1)
    # =========================================================================

    @handle_errors("sync continuum")
    def _handle_sync_continuum(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/continuum - Sync from ContinuumMemory."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")  # ISO timestamp
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_continuum(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from ContinuumMemory not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from continuum: {e}")
            return error_response(f"Failed to sync from continuum: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from ContinuumMemory completed",
        })

    @handle_errors("sync consensus")
    def _handle_sync_consensus(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/consensus - Sync from ConsensusMemory."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_consensus(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from ConsensusMemory not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from consensus: {e}")
            return error_response(f"Failed to sync from consensus: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from ConsensusMemory completed",
        })

    @handle_errors("sync facts")
    def _handle_sync_facts(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/sync/facts - Sync from FactStore."""
        import asyncio

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        workspace_id = data.get("workspace_id", "default")
        since = data.get("since")
        limit = data.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.sync_from_facts(workspace_id=workspace_id, since=since, limit=limit)
            )
        except AttributeError:
            return json_response({
                "synced": 0,
                "message": "Sync from FactStore not yet implemented",
                "workspace_id": workspace_id,
            })
        except Exception as e:
            logger.error(f"Failed to sync from facts: {e}")
            return error_response(f"Failed to sync from facts: {e}", 500)

        return json_response({
            "synced": result.nodes_synced if hasattr(result, 'nodes_synced') else 0,
            "workspace_id": workspace_id,
            "message": "Sync from FactStore completed",
        })

    # =========================================================================
    # Extended Graph Endpoints (Phase A1)
    # =========================================================================

    @handle_errors("graph lineage")
    def _handle_graph_lineage(self, path: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id/lineage - Get node lineage."""
        import asyncio

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        depth = get_clamped_int_param(query_params, "depth", 5, min_val=1, max_val=10)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            # Query graph with derived_from relationships only
            result = _run_async(
                mound.query_graph(
                    start_id=node_id,
                    relationship_types=["derived_from"],
                    depth=depth,
                )
            )
        except Exception as e:
            logger.error(f"Graph lineage failed: {e}")
            return error_response(f"Graph lineage failed: {e}", 500)

        return json_response({
            "node_id": node_id,
            "lineage": {
                "nodes": [n.to_dict() for n in result.nodes],
                "edges": [e.to_dict() if hasattr(e, 'to_dict') else e for e in result.edges],
                "total_nodes": result.total_nodes,
                "total_edges": result.total_edges,
            },
            "depth": depth,
        })

    @handle_errors("graph related")
    def _handle_graph_related(self, path: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id/related - Get related nodes."""
        import asyncio

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type = get_bounded_string_param(query_params, "relationship_type", None, max_length=50)
        limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            # Query graph with depth 1 for immediate relations
            rel_types = [relationship_type] if relationship_type else None
            result = _run_async(
                mound.query_graph(
                    start_id=node_id,
                    relationship_types=rel_types,
                    depth=1,
                    max_nodes=limit,
                )
            )
        except Exception as e:
            logger.error(f"Get related nodes failed: {e}")
            return error_response(f"Get related nodes failed: {e}", 500)

        return json_response({
            "node_id": node_id,
            "related": [n.to_dict() for n in result.nodes if n.id != node_id],
            "relationship_type": relationship_type,
            "total": len(result.nodes) - 1,  # Exclude start node
        })
