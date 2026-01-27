"""
Main Knowledge Handler.

Combines all mixins to provide the complete Knowledge Base API:

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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.knowledge import (
    DatasetQueryEngine,
    FactStore,
    InMemoryEmbeddingService,
    InMemoryFactStore,
    SimpleQueryEngine,
)
from aragora.rbac.decorators import require_permission

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_string_param,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

from .facts import FactsOperationsMixin
from .query import QueryOperationsMixin
from .search import SearchOperationsMixin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Rate limiter for knowledge endpoints (60 requests per minute)
_knowledge_limiter = RateLimiter(requests_per_minute=60)


class KnowledgeHandler(
    FactsOperationsMixin,
    QueryOperationsMixin,
    SearchOperationsMixin,
    BaseHandler,
):
    """Handler for knowledge base API endpoints.

    Combines mixins for:
    - Fact CRUD operations (FactsOperationsMixin)
    - Natural language queries (QueryOperationsMixin)
    - Search and statistics (SearchOperationsMixin)
    """

    # RBAC permission keys
    KNOWLEDGE_READ_PERMISSION = "knowledge.read"
    KNOWLEDGE_WRITE_PERMISSION = "knowledge.write"
    KNOWLEDGE_DELETE_PERMISSION = "knowledge.delete"

    ROUTES = [
        "/api/v1/knowledge/query",
        "/api/v1/knowledge/facts",
        "/api/v1/knowledge/search",
        "/api/v1/knowledge/stats",
    ]

    def __init__(self, server_context: dict):
        """Initialize knowledge handler.

        Args:
            server_context: Server context with shared resources
        """
        super().__init__(server_context)  # type: ignore[arg-type]
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
        if path.startswith("/api/v1/knowledge/facts/"):
            return True
        return False

    def _check_permission(self, handler: Any, permission: str) -> Optional[HandlerResult]:
        """Check RBAC permission and return error response if denied."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission
        permissions = getattr(user, "permissions", []) or []
        roles = getattr(user, "roles", []) or []
        if permission not in permissions and "admin" not in roles and "admin" not in permissions:
            return error_response(f"Permission denied: requires {permission}", 403)
        return None

    @require_permission("knowledge:read")
    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route knowledge requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _knowledge_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for knowledge endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Check read permission for GET requests
        method = getattr(handler, "command", "GET")
        if method == "GET":
            perm_error = self._check_permission(handler, self.KNOWLEDGE_READ_PERMISSION)
            if perm_error:
                return perm_error
        elif method == "POST":
            # Query is read, create fact is write
            if path == "/api/v1/knowledge/query":
                perm_error = self._check_permission(handler, self.KNOWLEDGE_READ_PERMISSION)
            else:
                perm_error = self._check_permission(handler, self.KNOWLEDGE_WRITE_PERMISSION)
            if perm_error:
                return perm_error
        elif method == "PUT":
            perm_error = self._check_permission(handler, self.KNOWLEDGE_WRITE_PERMISSION)
            if perm_error:
                return perm_error
        elif method == "DELETE":
            perm_error = self._check_permission(handler, self.KNOWLEDGE_DELETE_PERMISSION)
            if perm_error:
                return perm_error

        # Query endpoint (POST)
        if path == "/api/v1/knowledge/query":
            return self._handle_query(query_params, handler)

        # Facts listing (GET) or creation (POST)
        if path == "/api/v1/knowledge/facts":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_fact(handler)
            return self._handle_list_facts(query_params)

        # Search chunks
        if path == "/api/v1/knowledge/search":
            return self._handle_search(query_params)

        # Statistics
        if path == "/api/v1/knowledge/stats":
            workspace_id = get_bounded_string_param(
                query_params, "workspace_id", None, max_length=100
            )
            return self._handle_stats(workspace_id)

        # Dynamic fact routes
        if path.startswith("/api/v1/knowledge/facts/"):
            return self._handle_fact_routes(path, query_params, handler)

        return None

    def _handle_fact_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/v1/knowledge/facts/:id/* routes."""
        parts = path.strip("/").split("/")

        # /api/v1/knowledge/facts/:id (5 parts: api, v1, knowledge, facts, id)
        if len(parts) == 5:
            fact_id = parts[4]
            method = getattr(handler, "command", "GET")
            if method == "GET":
                return self._handle_get_fact(fact_id)
            elif method == "PUT":
                return self._handle_update_fact(fact_id, handler)
            elif method == "DELETE":
                return self._handle_delete_fact(fact_id, handler)

        # /api/v1/knowledge/facts/:id/verify (6 parts)
        if len(parts) == 6 and parts[5] == "verify":
            fact_id = parts[4]
            return self._handle_verify_fact(fact_id, handler)

        # /api/v1/knowledge/facts/:id/contradictions (6 parts)
        if len(parts) == 6 and parts[5] == "contradictions":
            fact_id = parts[4]
            return self._handle_get_contradictions(fact_id)

        # /api/v1/knowledge/facts/:id/relations (6 parts)
        if len(parts) == 6 and parts[5] == "relations":
            fact_id = parts[4]
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_add_relation(fact_id, handler)
            return self._handle_get_relations(fact_id, query_params)

        # /api/v1/knowledge/facts/relations (POST - add relation) (5 parts)
        if len(parts) == 5 and parts[4] == "relations":
            return self._handle_add_relation_bulk(handler)

        return error_response("Unknown endpoint", 404)
