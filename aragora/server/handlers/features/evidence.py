"""
Evidence API Handler.

Provides REST API endpoints for the evidence collection, storage, and retrieval system.

Endpoints:
- GET  /api/evidence                    - List all evidence with filtering/pagination
- GET  /api/evidence/:id                - Get specific evidence by ID
- POST /api/evidence/search             - Search evidence with full-text query
- POST /api/evidence/collect            - Collect evidence for a topic/task
- GET  /api/evidence/debate/:debate_id  - Get evidence for a specific debate
- POST /api/evidence/debate/:debate_id  - Associate evidence with a debate
- GET  /api/evidence/statistics         - Get evidence store statistics
- DELETE /api/evidence/:id              - Delete evidence by ID
"""

import logging
from typing import Any, Optional

from aragora.evidence import (
    EvidenceCollector,
    EvidenceStore,
    QualityContext,
)
from aragora.server.handlers.base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    PaginatedHandlerMixin,
    error_response,
    get_float_param,
    get_int_param,
    get_string_param,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip
from aragora.server.validation.security import (
    validate_search_query_redos_safe,
    MAX_SEARCH_QUERY_LENGTH,
)
from aragora.server.handlers.utils.responses import HandlerResult

logger = logging.getLogger(__name__)

# Rate limiters for evidence endpoints
# Read operations are more permissive
_evidence_read_limiter = RateLimiter(requests_per_minute=60)
# Write/collect operations are more restrictive (expensive external API calls)
_evidence_write_limiter = RateLimiter(requests_per_minute=10)


class EvidenceHandler(BaseHandler, PaginatedHandlerMixin):
    """Handler for evidence-related API endpoints."""

    # Routes this handler responds to
    routes = [
        "GET /api/evidence",
        "GET /api/evidence/statistics",
        "GET /api/evidence/:id",
        "GET /api/evidence/debate/:debate_id",
        "POST /api/evidence/search",
        "POST /api/evidence/collect",
        "POST /api/evidence/debate/:debate_id",
        "DELETE /api/evidence/:id",
    ]

    # Static routes for exact matching
    ROUTES = [
        "/api/evidence",
        "/api/evidence/statistics",
        "/api/evidence/search",
        "/api/evidence/collect",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/evidence")

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)
        self._evidence_store: Optional[EvidenceStore] = None
        self._evidence_collector: Optional[EvidenceCollector] = None

    def _get_evidence_store(self) -> EvidenceStore:
        """Get or create evidence store instance."""
        if self._evidence_store is None:
            # Check if store exists in context
            if "evidence_store" in self.ctx:
                self._evidence_store = self.ctx["evidence_store"]
            else:
                # Create new store
                self._evidence_store = EvidenceStore()
                self.ctx["evidence_store"] = self._evidence_store
        return self._evidence_store

    def _get_evidence_collector(self) -> EvidenceCollector:
        """Get or create evidence collector instance."""
        if self._evidence_collector is None:
            # Check if collector exists in context
            if "evidence_collector" in self.ctx:
                self._evidence_collector = self.ctx["evidence_collector"]
            else:
                # Get connectors from context if available
                connectors = self.ctx.get("connectors", {})
                event_emitter = self.ctx.get("event_emitter")
                self._evidence_collector = EvidenceCollector(
                    connectors=connectors,
                    event_emitter=event_emitter,
                )
                self.ctx["evidence_collector"] = self._evidence_collector
        return self._evidence_collector

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests for evidence endpoints."""
        # Rate limit check for read operations
        client_ip = get_client_ip(handler)
        if not _evidence_read_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for evidence GET: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # GET /api/evidence/statistics
        if path == "/api/evidence/statistics":
            return self._handle_statistics()

        # GET /api/evidence/debate/:debate_id
        if path.startswith("/api/evidence/debate/"):
            debate_id, err = self.extract_path_param(path, 3, "debate_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_get_debate_evidence(debate_id, query_params)

        # GET /api/evidence/:id
        if path.startswith("/api/evidence/") and path.count("/") == 3:
            evidence_id, err = self.extract_path_param(path, 2, "evidence_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_get_evidence(evidence_id)

        # GET /api/evidence - list all
        if path == "/api/evidence":
            return self._handle_list_evidence(query_params)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for evidence endpoints."""
        # Rate limit check for write operations
        client_ip = get_client_ip(handler)
        if not _evidence_write_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for evidence POST: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # POST /api/evidence/search
        if path == "/api/evidence/search":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_search(body)

        # POST /api/evidence/collect
        if path == "/api/evidence/collect":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_collect(body)

        # POST /api/evidence/debate/:debate_id
        if path.startswith("/api/evidence/debate/"):
            debate_id, err = self.extract_path_param(path, 3, "debate_id", SAFE_ID_PATTERN)
            if err:
                return err
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_associate_evidence(debate_id, body)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests for evidence endpoints."""
        # Rate limit check for delete operations (uses write limiter)
        client_ip = get_client_ip(handler)
        if not _evidence_write_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for evidence DELETE: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # DELETE /api/evidence/:id
        if path.startswith("/api/evidence/") and path.count("/") == 3:
            evidence_id, err = self.extract_path_param(path, 2, "evidence_id", SAFE_ID_PATTERN)
            if err:
                return err
            return self._handle_delete_evidence(evidence_id)

        return None

    def _handle_list_evidence(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/evidence - list all evidence with pagination."""
        limit, offset = self.get_pagination(query_params)
        source_filter = get_string_param(query_params, "source", None)
        min_reliability = get_float_param(query_params, "min_reliability", 0.0)

        store = self._get_evidence_store()

        # Get total count
        stats = store.get_statistics()
        total = stats.get("total_evidence", 0)

        # For listing, use a broad search or direct query
        # Since EvidenceStore doesn't have a list_all, we'll use search with empty query
        # or implement pagination differently
        try:
            # Use search with wildcard-like behavior
            results = store.search_evidence(
                query="*",
                limit=limit,
                source_filter=source_filter,
                min_reliability=min_reliability,
            )
        except Exception:  # noqa: BLE001 - FTS might not support * wildcard
            # Fallback to empty results when search fails
            results = []

        return self.paginated_response(
            items=results,
            total=total,
            limit=limit,
            offset=offset,
            items_key="evidence",
        )

    def _handle_get_evidence(self, evidence_id: str) -> HandlerResult:
        """Handle GET /api/evidence/:id - get specific evidence."""
        store = self._get_evidence_store()
        evidence = store.get_evidence(evidence_id)

        if evidence is None:
            return error_response(f"Evidence not found: {evidence_id}", 404)

        return json_response({"evidence": evidence})

    def _handle_get_debate_evidence(self, debate_id: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/evidence/debate/:debate_id - get evidence for debate."""
        round_number = get_int_param(query_params, "round", None)

        store = self._get_evidence_store()
        evidence_list = store.get_debate_evidence(debate_id, round_number)

        return json_response(
            {
                "debate_id": debate_id,
                "round": round_number,
                "evidence": evidence_list,
                "count": len(evidence_list),
            }
        )

    def _handle_search(self, body: dict) -> HandlerResult:
        """Handle POST /api/evidence/search - full-text search."""
        query = body.get("query", "").strip()
        if not query:
            return error_response("Query is required", 400)

        # Validate search query for ReDoS safety
        validation_result = validate_search_query_redos_safe(
            query, max_length=MAX_SEARCH_QUERY_LENGTH
        )
        if not validation_result.is_valid:
            logger.warning("Evidence search query validation failed: %s", validation_result.error)
            return error_response(validation_result.error or "Invalid search query", 400)

        limit = body.get("limit", 20)
        source_filter = body.get("source")
        min_reliability = body.get("min_reliability", 0.0)

        # Optional quality context for scoring
        context = None
        if "context" in body:
            ctx_data = body["context"]
            context = QualityContext(
                query=ctx_data.get("topic", ctx_data.get("query", "")),
                keywords=ctx_data.get("keywords", []),
                required_topics=set(ctx_data.get("required_topics", [])),
                preferred_sources=set(
                    ctx_data.get("preferred_sources", ctx_data.get("required_sources", []))
                ),
                blocked_sources=set(ctx_data.get("blocked_sources", [])),
                max_age_days=ctx_data.get("max_age_days", 365),
                min_word_count=ctx_data.get("min_word_count", 50),
                require_citations=ctx_data.get("require_citations", False),
            )

        store = self._get_evidence_store()
        results = store.search_evidence(
            query=query,
            limit=limit,
            source_filter=source_filter,
            min_reliability=min_reliability,
            context=context,
        )

        return json_response(
            {
                "query": query,
                "results": results,
                "count": len(results),
            }
        )

    def _handle_collect(self, body: dict) -> HandlerResult:
        """Handle POST /api/evidence/collect - collect evidence for topic."""
        task = body.get("task", "").strip()
        if not task:
            return error_response("Task/topic is required", 400)

        enabled_connectors = body.get("connectors")  # Optional list
        debate_id = body.get("debate_id")  # Optional association
        round_number = body.get("round")

        collector = self._get_evidence_collector()

        # Run collection asynchronously
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            evidence_pack = loop.run_until_complete(
                collector.collect_evidence(task, enabled_connectors)
            )
        except Exception as e:
            logger.exception(f"Evidence collection failed: {e}")
            return error_response(f"Evidence collection failed: {str(e)}", 500)

        # Save to store if debate_id provided
        saved_ids = []
        if debate_id:
            store = self._get_evidence_store()
            saved_ids = store.save_evidence_pack(evidence_pack, debate_id, round_number)

        return json_response(
            {
                "task": task,
                "keywords": evidence_pack.topic_keywords,
                "snippets": [s.to_dict() for s in evidence_pack.snippets],
                "count": len(evidence_pack.snippets),
                "total_searched": evidence_pack.total_searched,
                "average_reliability": evidence_pack.average_reliability,
                "average_freshness": evidence_pack.average_freshness,
                "saved_ids": saved_ids,
                "debate_id": debate_id,
            }
        )

    def _handle_associate_evidence(self, debate_id: str, body: dict) -> HandlerResult:
        """Handle POST /api/evidence/debate/:debate_id - associate evidence."""
        evidence_ids = body.get("evidence_ids", [])
        if not evidence_ids:
            return error_response("evidence_ids is required", 400)

        round_number = body.get("round")
        store = self._get_evidence_store()

        associated = []
        for evidence_id in evidence_ids:
            # Check if evidence exists
            evidence = store.get_evidence(evidence_id)
            if evidence:
                # Re-save to create association (deduplication handles existing)
                store.save_evidence(
                    evidence_id=evidence_id,
                    source=evidence["source"],
                    title=evidence["title"],
                    snippet=evidence["snippet"],
                    url=evidence.get("url", ""),
                    reliability_score=evidence.get("reliability_score", 0.5),
                    metadata=evidence.get("metadata"),
                    debate_id=debate_id,
                    round_number=round_number,
                    enrich=False,  # Already enriched
                    score_quality=False,  # Already scored
                )
                associated.append(evidence_id)

        return json_response(
            {
                "debate_id": debate_id,
                "associated": associated,
                "count": len(associated),
            }
        )

    def _handle_delete_evidence(self, evidence_id: str) -> HandlerResult:
        """Handle DELETE /api/evidence/:id - delete evidence."""
        store = self._get_evidence_store()
        deleted = store.delete_evidence(evidence_id)

        if not deleted:
            return error_response(f"Evidence not found: {evidence_id}", 404)

        return json_response(
            {
                "deleted": True,
                "evidence_id": evidence_id,
            }
        )

    def _handle_statistics(self) -> HandlerResult:
        """Handle GET /api/evidence/statistics - get store stats."""
        store = self._get_evidence_store()
        stats = store.get_statistics()

        return json_response(
            {
                "statistics": stats,
            }
        )
