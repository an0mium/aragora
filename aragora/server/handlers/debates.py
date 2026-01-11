"""
Debate-related endpoint handlers.

Endpoints:
- GET /api/debates - List all debates
- GET /api/debates/{slug} - Get debate by slug
- GET /api/debates/slug/{slug} - Get debate by slug (alternative)
- GET /api/debates/{id}/export/{format} - Export debate
- GET /api/debates/{id}/impasse - Detect debate impasse
- GET /api/debates/{id}/convergence - Get convergence status
- GET /api/debates/{id}/citations - Get evidence citations for debate
- GET /api/debates/{id}/evidence - Get comprehensive evidence trail
- GET /api/debate/{id}/meta-critique - Get meta-level debate analysis
- GET /api/debate/{id}/graph/stats - Get argument graph statistics
- POST /api/debates/{id}/fork - Fork debate at a branch point
- PATCH /api/debates/{id} - Update debate metadata (title, tags, status)
- GET /api/search - Cross-debate search by query
"""

from __future__ import annotations

import logging
from typing import Optional
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    require_storage,
    get_int_param,
    ttl_cache,
    safe_json_parse,
    safe_error_message,
)
from .utils.rate_limit import rate_limit
from aragora.server.validation import validate_debate_id

logger = logging.getLogger(__name__)

# Cache TTLs for debates endpoints (in seconds)
CACHE_TTL_DEBATES_LIST = 30  # Short TTL for list (may change frequently)
CACHE_TTL_SEARCH = 60  # Search results cache
CACHE_TTL_CONVERGENCE = 120  # Convergence status (changes less often)
CACHE_TTL_IMPASSE = 120  # Impasse detection


class DebatesHandler(BaseHandler):
    """Handler for debate-related endpoints."""

    # Route patterns this handler manages
    ROUTES = [
        "/api/debates",
        "/api/debates/",  # With trailing slash
        "/api/debates/slug/",
        "/api/debates/*/export/",
        "/api/debates/*/impasse",
        "/api/debates/*/convergence",
        "/api/debates/*/citations",
        "/api/debates/*/messages",  # Paginated message history
        "/api/debates/*/fork",  # POST - counterfactual fork
        "/api/debates/*/followup",  # POST - crux-driven follow-up debate
        "/api/debates/*/followups",  # GET - list follow-up suggestions
        "/api/debates/*/verification-report",  # Verification feedback
        "/api/search",  # Cross-debate search
    ]

    # Endpoints that require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/debates",  # List all debates - prevents enumeration
        "/export/",  # Export debate data
        "/citations",  # Evidence citations
        "/fork",  # Fork debate
        "/followup",  # Create follow-up debate
    ]

    # Allowed export formats and tables for input validation
    ALLOWED_EXPORT_FORMATS = {"json", "csv", "html"}
    ALLOWED_EXPORT_TABLES = {"summary", "messages", "critiques", "votes"}

    # Endpoints that expose debate artifacts - require auth unless debate is_public
    ARTIFACT_ENDPOINTS = {"/messages", "/evidence", "/verification-report"}

    # Route dispatch table: (suffix, handler_method_name, needs_debate_id, extra_params)
    # extra_params is a callable that extracts additional params from (path, query_params)
    SUFFIX_ROUTES = [
        ("/impasse", "_get_impasse", True, None),
        ("/convergence", "_get_convergence", True, None),
        ("/citations", "_get_citations", True, None),
        ("/evidence", "_get_evidence", True, None),
        ("/messages", "_get_debate_messages", True, lambda p, q: {
            "limit": get_int_param(q, 'limit', 50),
            "offset": get_int_param(q, 'offset', 0),
        }),
        ("/meta-critique", "_get_meta_critique", True, None),
        ("/graph/stats", "_get_graph_stats", True, None),
        ("/verification-report", "_get_verification_report", True, None),
        ("/followups", "_get_followup_suggestions", True, None),
    ]

    def _check_auth(self, handler) -> Optional[HandlerResult]:
        """Check authentication for sensitive endpoints.

        Returns:
            None if auth passes, HandlerResult with 401 if auth fails.
        """
        from aragora.server.auth import auth_config

        if handler is None:
            logger.debug("No handler provided for auth check")
            return None  # Can't check auth without handler

        # If auth is disabled globally, allow access
        if not auth_config.enabled:
            return None

        # Extract auth token from Authorization header
        auth_header = None
        if hasattr(handler, 'headers'):
            auth_header = handler.headers.get('Authorization', '')

        token = None
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]

        # Check if API token is configured
        if not auth_config.api_token:
            logger.debug("No API token configured, skipping auth")
            return None

        # Validate the provided token
        if not token or not auth_config.validate_token(token):
            return error_response("Invalid or missing authentication token", 401)

        return None

    def _requires_auth(self, path: str) -> bool:
        """Check if the given path requires authentication."""
        for pattern in self.AUTH_REQUIRED_ENDPOINTS:
            if pattern in path:
                return True
        return False

    def _check_artifact_access(
        self, debate_id: str, suffix: str, handler
    ) -> Optional[HandlerResult]:
        """Check access to artifact endpoints.

        Returns None if access allowed, 401 error if auth required but missing.
        Artifacts are accessible if:
        - Debate is marked as is_public=True, OR
        - Valid auth token is provided
        """
        if suffix not in self.ARTIFACT_ENDPOINTS:
            return None  # Not an artifact endpoint

        # Check if debate is public
        storage = self.get_storage()
        if storage and storage.is_public(debate_id):
            return None  # Public debate, no auth needed

        # Private debate - require authentication
        auth_result = self._check_auth(handler)
        if auth_result:
            return auth_result  # Auth failed

        return None  # Auth passed

    def _dispatch_suffix_route(
        self, path: str, query_params: dict, handler
    ) -> Optional[HandlerResult]:
        """Dispatch routes based on path suffix using SUFFIX_ROUTES table.

        Returns:
            HandlerResult if a route matched, None otherwise.
        """
        for suffix, method_name, needs_id, extra_params_fn in self.SUFFIX_ROUTES:
            if not path.endswith(suffix):
                continue

            # Extract debate_id if needed
            if needs_id:
                debate_id, err = self._extract_debate_id(path)
                if err:
                    return error_response(err, 400)
                if not debate_id:
                    continue

                # Check artifact access (auth required for private debates)
                access_error = self._check_artifact_access(debate_id, suffix, handler)
                if access_error:
                    return access_error

            # Get handler method
            method = getattr(self, method_name, None)
            if not method:
                continue

            # Build arguments
            if needs_id:
                if extra_params_fn:
                    extra = extra_params_fn(path, query_params)
                    # Methods like _get_debate_messages don't take handler
                    if method_name == "_get_debate_messages":
                        return method(debate_id, **extra)
                    return method(handler, debate_id, **extra)
                else:
                    # Methods like _get_meta_critique only take debate_id
                    if method_name in ("_get_meta_critique", "_get_graph_stats"):
                        return method(debate_id)
                    return method(handler, debate_id)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/debates":
            return True
        if path == "/api/search":
            return True
        if path.startswith("/api/debates/"):
            return True
        # Also handle /api/debate/{id}/meta-critique and /api/debate/{id}/graph/stats
        if path.startswith("/api/debate/") and (
            path.endswith("/meta-critique") or path.endswith("/graph/stats")
        ):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route debate requests to appropriate handler methods."""
        # Check authentication for protected endpoints
        if self._requires_auth(path):
            auth_error = self._check_auth(handler)
            if auth_error:
                return auth_error

        # Search endpoint
        if path == "/api/search":
            query = query_params.get('q', query_params.get('query', ''))
            if isinstance(query, list):
                query = query[0] if query else ''
            limit = min(get_int_param(query_params, 'limit', 20), 100)
            offset = get_int_param(query_params, 'offset', 0)
            # Get authenticated user for org-scoped search
            user = self.get_current_user(handler)
            org_id = user.org_id if user else None
            return self._search_debates(query, limit, offset, org_id)

        # Exact path matches
        if path == "/api/debates":
            limit = min(get_int_param(query_params, 'limit', 20), 100)
            # Get authenticated user for org-scoped results
            user = self.get_current_user(handler)
            org_id = user.org_id if user else None
            return self._list_debates(limit, org_id)

        if path.startswith("/api/debates/slug/"):
            slug = path.split("/")[-1]
            return self._get_debate_by_slug(handler, slug)

        # Dispatch suffix-based routes (impasse, convergence, citations, messages, etc.)
        result = self._dispatch_suffix_route(path, query_params, handler)
        if result:
            return result

        # Export route (special handling for format/table validation)
        if "/export/" in path:
            parts = path.split("/")
            if len(parts) >= 6:
                debate_id = parts[3]
                # Validate debate ID for export
                is_valid, err = validate_debate_id(debate_id)
                if not is_valid:
                    return error_response(err, 400)
                export_format = parts[5]
                # Validate export format
                if export_format not in self.ALLOWED_EXPORT_FORMATS:
                    return error_response(
                        f"Invalid format '{export_format}'. Allowed: {sorted(self.ALLOWED_EXPORT_FORMATS)}", 400
                    )
                table = query_params.get('table', 'summary')
                # Validate table parameter
                if table not in self.ALLOWED_EXPORT_TABLES:
                    return error_response(
                        f"Invalid table '{table}'. Allowed: {sorted(self.ALLOWED_EXPORT_TABLES)}", 400
                    )
                return self._export_debate(handler, debate_id, export_format, table)

        # Default: treat as slug lookup
        if path.startswith("/api/debates/"):
            slug = path.split("/")[-1]
            if slug and slug not in ("impasse", "convergence"):
                return self._get_debate_by_slug(handler, slug)

        return None

    def _extract_debate_id(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """Extract and validate debate ID from path like /api/debates/{id}/impasse.

        Returns:
            Tuple of (debate_id, error_message). If error_message is set, debate_id is None.
        """
        parts = path.split("/")
        if len(parts) < 4:
            return None, "Invalid path"

        debate_id = parts[3]
        is_valid, err = validate_debate_id(debate_id)
        if not is_valid:
            return None, err

        return debate_id, None

    @rate_limit(rpm=30, limiter_name="debates_list")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_DEBATES_LIST, key_prefix="debates_list", skip_first=True)
    @handle_errors("list debates")
    def _list_debates(self, limit: int, org_id: Optional[str] = None) -> HandlerResult:
        """List recent debates, optionally filtered by organization.

        Args:
            limit: Maximum number of debates to return
            org_id: If provided, only return debates for this organization.
                    If None, returns all debates (backwards compatible).

        Cached for 30 seconds. Cache key includes org_id for per-org isolation.
        """
        storage = self.get_storage()
        debates = storage.list_recent(limit=limit, org_id=org_id)
        # Convert DebateMetadata objects to dicts
        debates_list = [d.__dict__ if hasattr(d, '__dict__') else d for d in debates]
        return json_response({"debates": debates_list, "count": len(debates_list)})

    @rate_limit(rpm=30, limiter_name="debates_search")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_SEARCH, key_prefix="debates_search", skip_first=True)
    def _search_debates(
        self, query: str, limit: int, offset: int, org_id: Optional[str] = None
    ) -> HandlerResult:
        """Search debates by query string, optionally filtered by organization.

        Searches across debate tasks/topics using SQL LIKE pattern matching.

        Args:
            query: Search query string
            limit: Maximum results to return
            offset: Offset for pagination
            org_id: If provided, only search within this organization's debates

        Returns:
            HandlerResult with matching debates and pagination metadata
        """
        storage = self.get_storage()
        try:
            import sqlite3
            from aragora.config import DB_TIMEOUT_SECONDS

            # Get all recent debates and filter in Python
            # (More robust than raw SQL for now)
            all_debates = storage.list_recent(limit=500, org_id=org_id)

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                matching = []
                for d in all_debates:
                    task = getattr(d, 'task', '') or ''
                    topic = getattr(d, 'topic', '') or task
                    slug = getattr(d, 'slug', '') or ''

                    if (query_lower in task.lower() or
                        query_lower in topic.lower() or
                        query_lower in slug.lower()):
                        matching.append(d)
            else:
                matching = list(all_debates)

            # Apply pagination
            total = len(matching)
            paginated = matching[offset:offset + limit]

            # Convert to dicts
            results = []
            for d in paginated:
                if hasattr(d, '__dict__'):
                    results.append(d.__dict__)
                elif isinstance(d, dict):
                    results.append(d)
                else:
                    results.append({"data": str(d)})

            return json_response({
                "results": results,
                "query": query,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(results) < total,
            })
        except Exception as e:
            logger.error("Search failed for query '%s': %s: %s", query, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "search debates"), 500)

    @require_storage
    @handle_errors("get debate by slug")
    def _get_debate_by_slug(self, handler, slug: str) -> HandlerResult:
        """Get a debate by slug."""
        storage = self.get_storage()
        debate = storage.get_debate(slug)
        if debate:
            return json_response(debate)
        return error_response(f"Debate not found: {slug}", 404)

    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_IMPASSE, key_prefix="debates_impasse", skip_first=True)
    @handle_errors("impasse detection")
    def _get_impasse(self, handler, debate_id: str) -> HandlerResult:
        """Detect impasse in a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        # Analyze for impasse indicators
        critiques = debate.get("critiques", [])

        # Simple impasse detection: repetitive critiques without progress
        impasse_indicators = {
            "repeated_critiques": False,
            "no_convergence": not debate.get("consensus_reached", False),
            "high_severity_critiques": any(c.get("severity", 0) > 0.7 for c in critiques),
        }

        is_impasse = sum(impasse_indicators.values()) >= 2

        return json_response({
            "debate_id": debate_id,
            "is_impasse": is_impasse,
            "indicators": impasse_indicators,
        })

    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_convergence", skip_first=True)
    @handle_errors("convergence check")
    def _get_convergence(self, handler, debate_id: str) -> HandlerResult:
        """Get convergence status for a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        return json_response({
            "debate_id": debate_id,
            "convergence_status": debate.get("convergence_status", "unknown"),
            "convergence_similarity": debate.get("convergence_similarity", 0.0),
            "consensus_reached": debate.get("consensus_reached", False),
            "rounds_used": debate.get("rounds_used", 0),
        })

    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_verification", skip_first=True)
    @handle_errors("verification report")
    def _get_verification_report(self, handler, debate_id: str) -> HandlerResult:
        """Get verification report for a debate.

        Returns verification results and bonuses applied during consensus,
        useful for analyzing claim quality and feedback loop effectiveness.
        """
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        verification_results = debate.get("verification_results", {})
        verification_bonuses = debate.get("verification_bonuses", {})

        # Calculate summary stats
        total_verified = sum(v for v in verification_results.values() if v > 0)
        agents_verified = sum(1 for v in verification_results.values() if v > 0)
        total_bonus = sum(verification_bonuses.values())

        return json_response({
            "debate_id": debate_id,
            "verification_enabled": bool(verification_results),
            "verification_results": verification_results,
            "verification_bonuses": verification_bonuses,
            "summary": {
                "total_verified_claims": total_verified,
                "agents_with_verified_claims": agents_verified,
                "total_bonus_applied": round(total_bonus, 3),
            },
            "winner": debate.get("winner"),
            "consensus_reached": debate.get("consensus_reached", False),
        })

    @require_storage
    def _export_debate(self, handler, debate_id: str, format: str, table: str) -> HandlerResult:
        """Export debate in specified format."""
        valid_formats = {"json", "csv", "html"}
        if format not in valid_formats:
            return error_response(f"Invalid format: {format}. Valid: {valid_formats}", 400)

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            if format == "json":
                return json_response(debate)
            elif format == "csv":
                return self._format_csv(debate, table)
            else:  # format == "html"
                return self._format_html(debate)

        except Exception as e:
            logger.error("Export failed for %s (format=%s): %s: %s", debate_id, format, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "export debate"), 500)

    def _format_csv(self, debate: dict, table: str) -> HandlerResult:
        """Format debate as CSV for the specified table type."""
        from aragora.server.debate_export import format_debate_csv

        result = format_debate_csv(debate, table)
        return HandlerResult(
            status_code=200,
            content_type=result.content_type,
            body=result.content,
            headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
        )

    def _format_html(self, debate: dict) -> HandlerResult:
        """Format debate as standalone HTML page."""
        from aragora.server.debate_export import format_debate_html

        result = format_debate_html(debate)
        return HandlerResult(
            status_code=200,
            content_type=result.content_type,
            body=result.content,
            headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
        )

    @require_storage
    def _get_citations(self, handler, debate_id: str) -> HandlerResult:
        """Get evidence citations for a debate.

        Returns the grounded verdict including:
        - Claims extracted from final answer
        - Evidence snippets linked to each claim
        - Overall grounding score
        - Full citation list with sources
        """
        import json as json_module

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Check if grounded_verdict is stored
            grounded_verdict_raw = debate.get("grounded_verdict")

            if not grounded_verdict_raw:
                return json_response({
                    "debate_id": debate_id,
                    "has_citations": False,
                    "message": "No evidence citations available for this debate",
                    "grounded_verdict": None,
                })

            # Parse grounded_verdict JSON if it's a string
            grounded_verdict = safe_json_parse(grounded_verdict_raw)

            if not grounded_verdict:
                return json_response({
                    "debate_id": debate_id,
                    "has_citations": False,
                    "message": "Evidence citations could not be parsed",
                    "grounded_verdict": None,
                })

            return json_response({
                "debate_id": debate_id,
                "has_citations": True,
                "grounding_score": grounded_verdict.get("grounding_score", 0),
                "confidence": grounded_verdict.get("confidence", 0),
                "claims": grounded_verdict.get("claims", []),
                "all_citations": grounded_verdict.get("all_citations", []),
                "verdict": grounded_verdict.get("verdict", ""),
            })

        except Exception as e:
            logger.error("Failed to get citations for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get citations"), 500)

    @require_storage
    def _get_evidence(self, handler, debate_id: str) -> HandlerResult:
        """Get comprehensive evidence trail for a debate.

        Combines grounded verdict with related evidence from ContinuumMemory.

        Returns:
            - grounded_verdict: Claim analysis with citations
            - related_evidence: Evidence snippets from memory
            - metadata: Search context and quality metrics
        """
        import json as json_module

        storage = self.get_storage()

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get grounded verdict from debate
            grounded_verdict = safe_json_parse(debate.get("grounded_verdict"))

            # Try to get related evidence from ContinuumMemory
            related_evidence = []
            task = debate.get("task", "")

            try:
                from aragora.memory.continuum import ContinuumMemory

                continuum = self.ctx.get("continuum_memory")
                if continuum and task:
                    # Query for evidence-type memories related to this task
                    memories = continuum.search(
                        query=task[:200],
                        limit=10,
                        min_importance=0.3,
                    )

                    # Filter to evidence type
                    for memory in memories:
                        metadata = getattr(memory, "metadata", {}) or {}
                        if metadata.get("type") == "evidence":
                            related_evidence.append({
                                "id": getattr(memory, "id", ""),
                                "content": getattr(memory, "content", ""),
                                "source": metadata.get("source", "unknown"),
                                "importance": getattr(memory, "importance", 0.5),
                                "tier": str(getattr(memory, "tier", "medium")),
                            })
            except Exception as e:
                logger.debug(f"Could not fetch ContinuumMemory evidence: {e}")

            # Build response
            response = {
                "debate_id": debate_id,
                "task": task,
                "has_evidence": bool(grounded_verdict or related_evidence),
            }

            if grounded_verdict:
                response["grounded_verdict"] = {
                    "grounding_score": grounded_verdict.get("grounding_score", 0),
                    "confidence": grounded_verdict.get("confidence", 0),
                    "claims_count": len(grounded_verdict.get("claims", [])),
                    "citations_count": len(grounded_verdict.get("all_citations", [])),
                    "verdict": grounded_verdict.get("verdict", ""),
                }
                response["claims"] = grounded_verdict.get("claims", [])
                response["citations"] = grounded_verdict.get("all_citations", [])
            else:
                response["grounded_verdict"] = None
                response["claims"] = []
                response["citations"] = []

            response["related_evidence"] = related_evidence
            response["evidence_count"] = len(related_evidence)

            return json_response(response)

        except Exception as e:
            logger.exception(f"Failed to get evidence for {debate_id}")
            return error_response(safe_error_message(e, "get evidence"), 500)

    @require_storage
    def _get_debate_messages(self, debate_id: str, limit: int = 50, offset: int = 0) -> HandlerResult:
        """Get paginated message history for a debate.

        Args:
            debate_id: The debate ID
            limit: Maximum messages to return (default 50, max 200)
            offset: Starting offset for pagination

        Returns:
            Paginated list of messages with metadata
        """
        storage = self.get_storage()
        # Clamp limit
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            messages = debate.get("messages", [])
            total = len(messages)

            # Apply pagination
            paginated_messages = messages[offset:offset + limit]

            # Format messages for API response
            formatted_messages = []
            for i, msg in enumerate(paginated_messages):
                formatted_msg = {
                    "index": offset + i,
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", ""),
                    "agent": msg.get("agent") or msg.get("name"),
                    "round": msg.get("round", 0),
                }
                # Include optional fields if present
                if "timestamp" in msg:
                    formatted_msg["timestamp"] = msg["timestamp"]
                if "metadata" in msg:
                    formatted_msg["metadata"] = msg["metadata"]
                formatted_messages.append(formatted_msg)

            return json_response({
                "debate_id": debate_id,
                "messages": formatted_messages,
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(paginated_messages) < total,
            })

        except Exception as e:
            logger.error("Failed to get messages for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get messages"), 500)

    def _get_meta_critique(self, debate_id: str) -> HandlerResult:
        """Get meta-level analysis of a debate (repetition, circular arguments, etc)."""
        try:
            from aragora.debate.meta import MetaCritiqueAnalyzer
            from aragora.debate.traces import DebateTrace
        except ImportError:
            return error_response("Meta critique module not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            trace_path = nomic_dir / "traces" / f"{debate_id}.json"
            if not trace_path.exists():
                return error_response("Debate trace not found", 404)

            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()  # type: ignore[attr-defined]

            analyzer = MetaCritiqueAnalyzer()
            critique = analyzer.analyze(result)

            return json_response({
                "debate_id": debate_id,
                "overall_quality": critique.overall_quality,
                "productive_rounds": critique.productive_rounds,
                "unproductive_rounds": critique.unproductive_rounds,
                "observations": [
                    {
                        "type": o.observation_type,
                        "severity": o.severity,
                        "agent": getattr(o, 'agent', None),
                        "round": getattr(o, 'round_num', None),
                        "description": o.description,
                    }
                    for o in critique.observations
                ],
                "recommendations": critique.recommendations,
            })
        except Exception as e:
            logger.error("Failed to get meta critique for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get meta critique"), 500)

    def _get_graph_stats(self, debate_id: str) -> HandlerResult:
        """Get argument graph statistics for a debate.

        Returns node counts, edge counts, depth, branching factor, and complexity.
        """
        try:
            from aragora.visualization.mapper import ArgumentCartographer
            from aragora.debate.traces import DebateTrace
        except ImportError:
            return error_response("Graph analysis module not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            trace_path = nomic_dir / "traces" / f"{debate_id}.json"

            if not trace_path.exists():
                # Try replays directory as fallback
                replay_path = nomic_dir / "replays" / debate_id / "events.jsonl"
                if replay_path.exists():
                    return self._build_graph_from_replay(debate_id, replay_path)
                return error_response("Debate not found", 404)

            # Load from trace file
            trace = DebateTrace.load(trace_path)
            result = trace.to_debate_result()  # type: ignore[attr-defined]

            # Build cartographer from debate result
            cartographer = ArgumentCartographer()
            cartographer.set_debate_context(debate_id, result.task or "")

            # Process messages from the debate
            for msg in result.messages:
                cartographer.update_from_message(
                    agent=msg.agent,
                    content=msg.content,
                    role=msg.role,
                    round_num=msg.round,
                )

            # Process critiques
            for critique in result.critiques:
                cartographer.update_from_critique(
                    critic_agent=critique.agent,
                    target_agent=critique.target or "",
                    severity=critique.severity,
                    round_num=getattr(critique, 'round', 1),
                    critique_text=critique.reasoning,
                )

            stats = cartographer.get_statistics()
            return json_response(stats)

        except Exception as e:
            logger.error("Failed to get graph stats for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get graph stats"), 500)

    def _build_graph_from_replay(self, debate_id: str, replay_path) -> HandlerResult:
        """Build graph stats from replay events file."""
        import json as json_mod

        try:
            from aragora.visualization.mapper import ArgumentCartographer
        except ImportError:
            return error_response("Graph analysis module not available", 503)

        try:
            cartographer = ArgumentCartographer()
            cartographer.set_debate_context(debate_id, "")

            with replay_path.open() as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            event = json_mod.loads(line)
                        except json_mod.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSONL line {line_num}")
                            continue

                        if event.get("type") == "agent_message":
                            cartographer.update_from_message(
                                agent=event.get("agent", "unknown"),
                                content=event.get("data", {}).get("content", ""),
                                role=event.get("data", {}).get("role", "proposer"),
                                round_num=event.get("round", 1),
                            )
                        elif event.get("type") == "critique":
                            cartographer.update_from_critique(
                                critic_agent=event.get("agent", "unknown"),
                                target_agent=event.get("data", {}).get("target", "unknown"),
                                severity=event.get("data", {}).get("severity", 0.5),
                                round_num=event.get("round", 1),
                                critique_text=event.get("data", {}).get("content", ""),
                            )

            stats = cartographer.get_statistics()
            return json_response(stats)
        except Exception as e:
            logger.error("Failed to build graph from replay %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "build graph from replay"), 500)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path.endswith("/fork"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._fork_debate(handler, debate_id)

        if path.endswith("/verify"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._verify_outcome(handler, debate_id)

        if path.endswith("/followup"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._create_followup_debate(handler, debate_id)

        return None

    def handle_patch(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route PATCH requests to appropriate methods."""
        # Handle /api/debates/{id} pattern for updates
        if path.startswith("/api/debates/") and path.count("/") == 3:
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._patch_debate(handler, debate_id)
        return None

    @require_storage
    def _patch_debate(self, handler, debate_id: str) -> HandlerResult:
        """Update debate metadata.

        Request body can include:
            {
                "title": str,  # Optional: update debate title
                "tags": list[str],  # Optional: update tags
                "status": str,  # Optional: "active", "paused", "concluded"
                "metadata": dict  # Optional: custom metadata
            }

        Returns:
            Updated debate summary
        """
        # Read and validate request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        if not body:
            return error_response("Empty update body", 400)

        # Get storage and find debate
        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Apply updates (only allowed fields)
            allowed_fields = {"title", "tags", "status", "metadata"}
            updates = {k: v for k, v in body.items() if k in allowed_fields}

            if not updates:
                return error_response(
                    f"No valid fields to update. Allowed: {', '.join(allowed_fields)}",
                    400
                )

            # Validate status if provided
            if "status" in updates:
                valid_statuses = {"active", "paused", "concluded", "archived"}
                if updates["status"] not in valid_statuses:
                    return error_response(
                        f"Invalid status. Must be one of: {', '.join(valid_statuses)}",
                        400
                    )

            # Apply updates to debate
            for key, value in updates.items():
                debate[key] = value

            # Save updated debate
            storage.save_debate(debate_id, debate)

            logger.info(f"Debate {debate_id} updated: {list(updates.keys())}")

            return json_response({
                "success": True,
                "debate_id": debate_id,
                "updated_fields": list(updates.keys()),
                "debate": {
                    "id": debate_id,
                    "title": debate.get("title", debate.get("task", "")),
                    "status": debate.get("status", "active"),
                    "tags": debate.get("tags", []),
                }
            })
        except Exception as e:
            logger.error(f"Failed to update debate {debate_id}: {e}")
            return error_response(safe_error_message(e, "update debate"), 500)

    @require_storage
    def _fork_debate(self, handler, debate_id: str) -> HandlerResult:
        """Create a counterfactual fork of a debate at a specific branch point.

        Request body:
            {
                "branch_point": int,  # Round number to branch from
                "modified_context": str  # Optional: context for the counterfactual
            }

        Returns:
            Information about the created branch
        """
        from aragora.server.validation import FORK_REQUEST_SCHEMA, validate_against_schema

        # Read and validate request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        validation = validate_against_schema(body, FORK_REQUEST_SCHEMA)
        if not validation.is_valid:
            return error_response(validation.error, 400)

        branch_point = body.get("branch_point", 0)
        modified_context = body.get("modified_context")

        # Get the original debate
        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            messages = debate.get("messages", [])
            if branch_point > len(messages):
                return error_response(
                    f"Branch point {branch_point} exceeds message count {len(messages)}",
                    400
                )

            # Import counterfactual module
            try:
                from aragora.debate.counterfactual import (
                    CounterfactualOrchestrator,
                    PivotClaim,
                    CounterfactualBranch,
                )
            except ImportError:
                return error_response("Counterfactual module not available", 503)

            # Create a pivot claim from the context
            import uuid as uuid_mod
            pivot = PivotClaim(
                claim_id=f"pivot-{uuid_mod.uuid4().hex[:8]}",
                statement=modified_context or f"Branch at round {branch_point}",
                author="user",
                disagreement_score=1.0,
                importance_score=1.0,
                blocking_agents=[],
                branch_reason=f"User-initiated fork at round {branch_point}",
            )

            # Create the branch record
            branch_id = f"fork-{debate_id}-r{branch_point}-{uuid_mod.uuid4().hex[:8]}"

            branch = CounterfactualBranch(
                branch_id=branch_id,
                parent_debate_id=debate_id,
                pivot_claim=pivot,
                assumption=True,  # Default to exploring the "true" branch
                messages=messages[:branch_point] if branch_point > 0 else [],
            )

            # Store the branch info
            branch_data = {
                "branch_id": branch_id,
                "parent_debate_id": debate_id,
                "branch_point": branch_point,
                "modified_context": modified_context,
                "pivot_claim": pivot.statement,
                "status": "created",
                "messages_inherited": branch_point,
            }

            # Try to store in nomic dir
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                import json as json_mod
                branches_dir = nomic_dir / "branches"
                branches_dir.mkdir(exist_ok=True)
                branch_file = branches_dir / f"{branch_id}.json"
                with open(branch_file, "w") as f:
                    json_mod.dump(branch_data, f, indent=2)

            return json_response({
                "success": True,
                "branch_id": branch_id,
                "parent_debate_id": debate_id,
                "branch_point": branch_point,
                "messages_inherited": branch_point,
                "modified_context": modified_context,
                "status": "created",
                "message": f"Created fork '{branch_id}' from debate '{debate_id}' at round {branch_point}",
            })

        except Exception as e:
            logger.error("Failed to create fork for %s at round %s: %s: %s", debate_id, branch_point, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "create fork"), 500)

    @handle_errors("verify debate outcome")
    def _verify_outcome(self, handler, debate_id: str) -> HandlerResult:
        """Record verification of whether a debate's winning position was correct.

        POST body:
            correct: bool - whether the winning position was actually correct
            source: str - verification source (default: "manual")

        Completes the truth-grounding feedback loop by linking positions to outcomes.
        """
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        correct = body.get('correct', False)
        source = body.get('source', 'manual')

        # Get position tracker from context
        position_tracker = self.ctx.get("position_tracker")

        if position_tracker:
            position_tracker.record_verification(debate_id, correct, source)
            return json_response({
                "status": "verified",
                "debate_id": debate_id,
                "correct": correct,
                "source": source,
            })

        # Try to create a temporary tracker
        try:
            from aragora.agents.truth_grounding import PositionTracker
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                db_path = nomic_dir / "aragora_positions.db"
                if db_path.exists():
                    tracker = PositionTracker(db_path=str(db_path))
                    tracker.record_verification(debate_id, correct, source)
                    return json_response({
                        "status": "verified",
                        "debate_id": debate_id,
                        "correct": correct,
                        "source": source,
                    })
            return error_response("Position tracking not configured", 503)
        except ImportError:
            return error_response("PositionTracker module not available", 503)

    @require_storage
    def _get_followup_suggestions(self, debate_id: str) -> HandlerResult:
        """Get follow-up debate suggestions based on identified cruxes.

        Analyzes the debate's uncertainty metrics and generates suggestions
        for follow-up debates to resolve key disagreement points.

        Returns:
            List of follow-up suggestions with priority and suggested task
        """
        storage = self.get_storage()

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Import uncertainty analysis components
            from aragora.uncertainty.estimator import DisagreementAnalyzer, DisagreementCrux

            # Extract cruxes from debate data
            cruxes = []

            # Check for stored uncertainty metrics
            uncertainty = debate.get("uncertainty_metrics", {})
            if uncertainty and "cruxes" in uncertainty:
                for crux_data in uncertainty["cruxes"]:
                    crux = DisagreementCrux(
                        description=crux_data.get("description", ""),
                        divergent_agents=crux_data.get("agents", []),
                        evidence_needed=crux_data.get("evidence_needed", ""),
                        severity=crux_data.get("severity", 0.5),
                    )
                    cruxes.append(crux)

            # If no stored cruxes, analyze the debate
            if not cruxes:
                from aragora.core import Vote, Message
                votes_data = debate.get("votes", [])
                messages_data = debate.get("messages", [])
                proposals = debate.get("proposals", {})

                # Convert to Vote/Message objects if needed
                votes = []
                for v in votes_data:
                    if hasattr(v, 'choice'):
                        votes.append(v)
                    else:
                        votes.append(Vote(
                            agent=v.get("agent", "unknown"),
                            choice=v.get("choice", ""),
                            confidence=v.get("confidence", 0.5),
                            reasoning=v.get("reasoning", ""),
                        ))

                messages = []
                for m in messages_data:
                    if hasattr(m, 'content'):
                        messages.append(m)
                    else:
                        messages.append(Message(
                            agent=m.get("agent", "unknown"),
                            content=m.get("content", ""),
                            role=m.get("role", "proposer"),
                            round=m.get("round", 1),
                        ))

                # Run disagreement analysis
                analyzer = DisagreementAnalyzer()
                metrics = analyzer.analyze_disagreement(messages, votes, proposals)
                cruxes = metrics.cruxes

            if not cruxes:
                return json_response({
                    "debate_id": debate_id,
                    "suggestions": [],
                    "message": "No significant disagreement cruxes identified",
                })

            # Generate follow-up suggestions
            analyzer = DisagreementAnalyzer()
            available_agents = debate.get("agents", [])
            suggestions = analyzer.suggest_followups(
                cruxes=cruxes,
                parent_debate_id=debate_id,
                available_agents=available_agents,
            )

            return json_response({
                "debate_id": debate_id,
                "suggestions": [s.to_dict() for s in suggestions],
                "count": len(suggestions),
            })

        except Exception as e:
            logger.error("Failed to get followup suggestions for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get followup suggestions"), 500)

    @require_storage
    def _create_followup_debate(self, handler, debate_id: str) -> HandlerResult:
        """Create a follow-up debate to resolve a specific crux.

        POST body:
            crux_id: str - ID of the crux to explore
            task: str (optional) - Custom task, otherwise generated from crux
            agents: list[str] (optional) - Specific agents to include

        Returns:
            Created debate metadata with parent lineage
        """
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        crux_id = body.get("crux_id")
        custom_task = body.get("task")
        requested_agents = body.get("agents", [])

        if not crux_id and not custom_task:
            return error_response("Either crux_id or task is required", 400)

        storage = self.get_storage()

        try:
            # Get parent debate
            parent_debate = storage.get_debate(debate_id)
            if not parent_debate:
                return error_response(f"Parent debate not found: {debate_id}", 404)

            # Find the crux if crux_id provided
            crux_data = None
            if crux_id:
                uncertainty = parent_debate.get("uncertainty_metrics", {})
                for c in uncertainty.get("cruxes", []):
                    if c.get("id") == crux_id:
                        crux_data = c
                        break

                if not crux_data:
                    return error_response(f"Crux not found: {crux_id}", 404)

            # Generate task
            if custom_task:
                task = custom_task
            elif crux_data:
                # Generate task from crux description
                from aragora.uncertainty.estimator import DisagreementAnalyzer, DisagreementCrux
                analyzer = DisagreementAnalyzer()
                crux = DisagreementCrux(
                    description=crux_data.get("description", ""),
                    divergent_agents=crux_data.get("agents", []),
                    severity=crux_data.get("severity", 0.5),
                )
                task = analyzer._generate_followup_task(crux)
            else:
                return error_response("Could not generate task", 400)

            # Determine agents
            if requested_agents:
                agents = requested_agents
            elif crux_data:
                agents = crux_data.get("agents", [])
                # Add parent debate agents if not enough
                if len(agents) < 2:
                    for agent in parent_debate.get("agents", []):
                        if agent not in agents:
                            agents.append(agent)
                        if len(agents) >= 3:
                            break
            else:
                agents = parent_debate.get("agents", [])[:3]

            # Create unique ID for follow-up debate
            import time
            import hashlib
            followup_id = f"followup-{debate_id[:8]}-{int(time.time()) % 100000}"

            # Store follow-up debate metadata
            followup_data = {
                "id": followup_id,
                "task": task,
                "agents": agents,
                "parent_debate_id": debate_id,
                "crux_id": crux_id,
                "crux_description": crux_data.get("description") if crux_data else None,
                "status": "pending",
                "created_at": time.time(),
            }

            # Store in nomic dir
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                import json as json_mod
                followups_dir = nomic_dir / "followups"
                followups_dir.mkdir(exist_ok=True)
                followup_file = followups_dir / f"{followup_id}.json"
                with open(followup_file, "w") as f:
                    json_mod.dump(followup_data, f, indent=2)

            logger.info(f"Created follow-up debate {followup_id} from parent {debate_id}")

            return json_response({
                "success": True,
                "followup_id": followup_id,
                "parent_debate_id": debate_id,
                "task": task,
                "agents": agents,
                "crux_id": crux_id,
                "status": "pending",
                "message": f"Created follow-up debate to explore: {task[:100]}",
            })

        except Exception as e:
            logger.error("Failed to create followup debate for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "create followup debate"), 500)
