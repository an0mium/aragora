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

import importlib
import logging
from typing import Optional

from aragora.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    StorageError,
)
from aragora.server.middleware.tier_enforcement import require_quota
from aragora.server.validation import validate_debate_id
from aragora.server.validation.schema import (
    DEBATE_START_SCHEMA,
    DEBATE_UPDATE_SCHEMA,
    validate_against_schema,
)

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    handle_errors,
    json_response,
    require_storage,
    safe_json_parse,
    ttl_cache,
)
from .debates_batch import BatchOperationsMixin
from .debates_fork import ForkOperationsMixin
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Status normalization map: converts internal status values to canonical SDK-compatible values
# Server uses: active, paused, concluded, archived
# SDKs expect: pending, running, completed, failed, cancelled, paused
STATUS_MAP = {
    "active": "running",
    "concluded": "completed",
    "archived": "completed",
    "starting": "created",
    "in_progress": "running",
}

# Reverse map for accepting SDK status values in updates
STATUS_REVERSE_MAP = {
    "running": "active",
    "completed": "concluded",
    "pending": "active",
    "created": "active",
    "in_progress": "active",
}


def normalize_status(status: str) -> str:
    """Normalize internal status to SDK-compatible canonical status.

    Args:
        status: Internal status value (active, paused, concluded, archived)

    Returns:
        Canonical status (pending, running, completed, failed, cancelled, paused)
    """
    return STATUS_MAP.get(status, status)


def denormalize_status(status: str) -> str:
    """Convert SDK status to internal status for storage.

    Args:
        status: SDK canonical status (running, completed, etc.)

    Returns:
        Internal status (active, concluded, etc.)
    """
    return STATUS_REVERSE_MAP.get(status, status)


def normalize_debate_response(debate: dict) -> dict:
    """Normalize debate dict for API response, ensuring SDK compatibility.

    Normalizes status values and ensures both field name variants are present
    for consensus fields (agreement/confidence, conclusion/final_answer).

    Args:
        debate: Raw debate dict from storage

    Returns:
        Normalized debate dict with SDK-compatible fields
    """
    if not debate:
        return debate

    # Normalize status
    if "status" in debate:
        debate["status"] = normalize_status(debate["status"])
    else:
        debate["status"] = "completed"

    # Ensure debate_id/id aliases exist for SDK/front-end parity
    if "debate_id" in debate and "id" not in debate:
        debate["id"] = debate["debate_id"]
    if "id" in debate and "debate_id" not in debate:
        debate["debate_id"] = debate["id"]

    # Promote consensus_proof into consensus if needed
    if "consensus" not in debate and "consensus_proof" in debate:
        consensus_proof = debate.get("consensus_proof") or {}
        vote_breakdown = consensus_proof.get("vote_breakdown") or {}
        supporting_agents = [agent for agent, agreed in vote_breakdown.items() if agreed]
        dissenting_agents = [agent for agent, agreed in vote_breakdown.items() if not agreed]
        debate["consensus"] = {
            "reached": consensus_proof.get("reached", False),
            "agreement": consensus_proof.get("confidence"),
            "confidence": consensus_proof.get("confidence"),
            "final_answer": consensus_proof.get("final_answer"),
            "conclusion": consensus_proof.get("final_answer"),
            "supporting_agents": supporting_agents,
            "dissenting_agents": dissenting_agents,
        }

    # consensus_reached/concordance helpers for UI
    if "consensus_reached" not in debate:
        consensus = debate.get("consensus") or {}
        debate["consensus_reached"] = bool(consensus.get("reached", False))
    if "confidence" not in debate:
        consensus = debate.get("consensus") or {}
        confidence = consensus.get("confidence", consensus.get("agreement"))
        if confidence is not None:
            debate["confidence"] = confidence

    # rounds_used defaults for list views
    if "rounds_used" not in debate:
        rounds_value = debate.get("rounds")
        if isinstance(rounds_value, int):
            debate["rounds_used"] = rounds_value
        elif isinstance(rounds_value, list):
            debate["rounds_used"] = len(rounds_value)
        else:
            debate["rounds_used"] = 0
    debate.setdefault("duration_seconds", 0)

    # Ensure consensus field aliases (for SDK compatibility)
    # confidence <-> agreement
    if "confidence" in debate and "agreement" not in debate:
        debate["agreement"] = debate["confidence"]
    elif "agreement" in debate and "confidence" not in debate:
        debate["confidence"] = debate["agreement"]

    # conclusion <-> final_answer
    if "conclusion" in debate and "final_answer" not in debate:
        debate["final_answer"] = debate["conclusion"]
    elif "final_answer" in debate and "conclusion" not in debate:
        debate["conclusion"] = debate["final_answer"]

    return debate


# Cache TTLs for debates endpoints (in seconds)
CACHE_TTL_DEBATES_LIST = 30  # Short TTL for list (may change frequently)
CACHE_TTL_SEARCH = 60  # Search results cache
CACHE_TTL_CONVERGENCE = 120  # Convergence status (changes less often)
CACHE_TTL_IMPASSE = 120  # Impasse detection


class DebatesHandler(ForkOperationsMixin, BatchOperationsMixin, BaseHandler):
    """Handler for debate-related endpoints."""

    # Route patterns this handler manages
    ROUTES = [
        "/api/debate",  # POST - create new debate (legacy endpoint)
        "/api/debates",
        "/api/debates/",  # With trailing slash
        "/api/debates/batch",  # POST - batch debate submission
        "/api/debates/batch/",
        "/api/debates/batch/*/status",  # GET - batch status
        "/api/debates/queue/status",  # GET - queue status
        "/api/debates/slug/",
        "/api/debates/*/export/",
        "/api/debates/*/impasse",
        "/api/debates/*/convergence",
        "/api/debates/*/citations",
        "/api/debates/*/messages",  # Paginated message history
        "/api/debates/*/fork",  # POST - counterfactual fork
        "/api/debates/*/followup",  # POST - crux-driven follow-up debate
        "/api/debates/*/followups",  # GET - list follow-up suggestions
        "/api/debates/*/forks",  # GET - list all forks for a debate
        "/api/debates/*/verification-report",  # Verification feedback
        "/api/debates/*/summary",  # GET - human-readable summary
        "/api/search",  # Cross-debate search
    ]

    # Endpoints that require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/debates",  # List all debates - prevents enumeration
        "/api/debates/batch",  # Batch submission requires auth
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
        (
            "/messages",
            "_get_debate_messages",
            True,
            lambda p, q: {
                "limit": get_int_param(q, "limit", 50),
                "offset": get_int_param(q, "offset", 0),
            },
        ),
        ("/meta-critique", "_get_meta_critique", True, None),
        ("/graph/stats", "_get_graph_stats", True, None),
        ("/verification-report", "_get_verification_report", True, None),
        ("/followups", "_get_followup_suggestions", True, None),
        ("/forks", "_list_debate_forks", True, None),
        ("/summary", "_get_summary", True, None),
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
        if hasattr(handler, "headers"):
            auth_header = handler.headers.get("Authorization", "")

        token = None
        if auth_header and auth_header.startswith("Bearer "):
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
                    if method_name in (
                        "_get_meta_critique",
                        "_get_graph_stats",
                        "_get_followup_suggestions",
                    ):
                        return method(debate_id)
                    return method(handler, debate_id)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/debate":
            return True  # POST - create debate
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
            query = query_params.get("q", query_params.get("query", ""))
            if isinstance(query, list):
                query = query[0] if query else ""
            limit = min(get_int_param(query_params, "limit", 20), 100)
            offset = get_int_param(query_params, "offset", 0)
            # Get authenticated user for org-scoped search
            user = self.get_current_user(handler)
            org_id = user.org_id if user else None
            return self._search_debates(query, limit, offset, org_id)

        # Queue status endpoint
        if path == "/api/debates/queue/status":
            return self._get_queue_status()

        # Batch status endpoint (GET /api/debates/batch/{id}/status)
        if path.startswith("/api/debates/batch/") and path.endswith("/status"):
            parts = path.split("/")
            if len(parts) >= 5:
                batch_id = parts[4]
                return self._get_batch_status(batch_id)

        # List batches (GET /api/debates/batch)
        if path in ("/api/debates/batch", "/api/debates/batch/"):
            limit = min(get_int_param(query_params, "limit", 50), 100)
            status_filter = query_params.get("status")
            return self._list_batches(limit, status_filter)

        # Exact path matches
        if path == "/api/debates":
            limit = min(get_int_param(query_params, "limit", 20), 100)
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
                        f"Invalid format '{export_format}'. Allowed: {sorted(self.ALLOWED_EXPORT_FORMATS)}",
                        400,
                    )
                table = query_params.get("table", "summary")
                # Validate table parameter
                if table not in self.ALLOWED_EXPORT_TABLES:
                    return error_response(
                        f"Invalid table '{table}'. Allowed: {sorted(self.ALLOWED_EXPORT_TABLES)}",
                        400,
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
        # Convert DebateMetadata objects to dicts and normalize for SDK compatibility
        debates_list = [
            normalize_debate_response(d.__dict__ if hasattr(d, "__dict__") else d) for d in debates
        ]
        return json_response({"debates": debates_list, "count": len(debates_list)})

    @rate_limit(rpm=30, limiter_name="debates_search")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_SEARCH, key_prefix="debates_search", skip_first=True)
    def _search_debates(
        self, query: str, limit: int, offset: int, org_id: Optional[str] = None
    ) -> HandlerResult:
        """Search debates by query string, optionally filtered by organization.

        Uses efficient SQL LIKE queries instead of loading all debates into memory.
        This is optimized for large debate databases.

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
            # Use efficient SQL search if query provided
            if query:
                matching, total = storage.search(
                    query=query,
                    limit=limit,
                    offset=offset,
                    org_id=org_id,
                )
            else:
                # No query - just list recent debates
                matching = storage.list_recent(limit=limit, org_id=org_id)
                total = len(matching)  # Approximate for no-query case

            # Convert to dicts and normalize for SDK compatibility
            results = []
            for d in matching:
                if hasattr(d, "__dict__"):
                    results.append(normalize_debate_response(d.__dict__))
                elif isinstance(d, dict):
                    results.append(normalize_debate_response(d))
                else:
                    results.append({"data": str(d)})

            return json_response(
                {
                    "results": results,
                    "query": query,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + len(results) < total,
                }
            )
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Search failed for query '%s': %s: %s", query, type(e).__name__, e, exc_info=True
            )
            return error_response("Database error during search", 500)
        except ValueError as e:
            logger.warning("Invalid search query '%s': %s", query, e)
            return error_response(f"Invalid search query: {e}", 400)

    @require_storage
    @handle_errors("get debate by slug")
    def _get_debate_by_slug(self, handler, slug: str) -> HandlerResult:
        """Get a debate by slug."""
        storage = self.get_storage()
        debate = storage.get_debate(slug)
        if debate:
            return json_response(normalize_debate_response(debate))
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

        return json_response(
            {
                "debate_id": debate_id,
                "is_impasse": is_impasse,
                "indicators": impasse_indicators,
            }
        )

    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_convergence", skip_first=True)
    @handle_errors("convergence check")
    def _get_convergence(self, handler, debate_id: str) -> HandlerResult:
        """Get convergence status for a debate."""
        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        return json_response(
            {
                "debate_id": debate_id,
                "convergence_status": debate.get("convergence_status", "unknown"),
                "convergence_similarity": debate.get("convergence_similarity", 0.0),
                "consensus_reached": debate.get("consensus_reached", False),
                "rounds_used": debate.get("rounds_used", 0),
            }
        )

    @require_storage
    @ttl_cache(
        ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_verification", skip_first=True
    )
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

        return json_response(
            {
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
            }
        )

    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_CONVERGENCE, key_prefix="debates_summary", skip_first=True)
    @handle_errors("get summary")
    def _get_summary(self, handler, debate_id: str) -> HandlerResult:
        """Get human-readable summary for a debate.

        Returns a structured summary with:
        - One-liner verdict
        - Key points and conclusions
        - Agreement and disagreement areas
        - Confidence assessment
        - Actionable next steps
        """
        from aragora.debate.summarizer import summarize_debate

        storage = self.get_storage()
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        # Generate summary
        summary = summarize_debate(debate)

        return json_response(
            {
                "debate_id": debate_id,
                "summary": summary.to_dict(),
                "task": debate.get("task", ""),
                "consensus_reached": debate.get("consensus_reached", False),
                "confidence": debate.get("confidence", 0.0),
            }
        )

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

        except RecordNotFoundError:
            logger.info("Export failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Export failed for %s (format=%s): %s: %s",
                debate_id,
                format,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error during export", 500)
        except ValueError as e:
            logger.warning("Export failed for %s - invalid format: %s", debate_id, e)
            return error_response(f"Invalid export format: {e}", 400)

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

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Check if grounded_verdict is stored
            grounded_verdict_raw = debate.get("grounded_verdict")

            if not grounded_verdict_raw:
                return json_response(
                    {
                        "debate_id": debate_id,
                        "has_citations": False,
                        "message": "No evidence citations available for this debate",
                        "grounded_verdict": None,
                    }
                )

            # Parse grounded_verdict JSON if it's a string
            grounded_verdict = safe_json_parse(grounded_verdict_raw)

            if not grounded_verdict:
                return json_response(
                    {
                        "debate_id": debate_id,
                        "has_citations": False,
                        "message": "Evidence citations could not be parsed",
                        "grounded_verdict": None,
                    }
                )

            return json_response(
                {
                    "debate_id": debate_id,
                    "has_citations": True,
                    "grounding_score": grounded_verdict.get("grounding_score", 0),
                    "confidence": grounded_verdict.get("confidence", 0),
                    "claims": grounded_verdict.get("claims", []),
                    "all_citations": grounded_verdict.get("all_citations", []),
                    "verdict": grounded_verdict.get("verdict", ""),
                }
            )

        except RecordNotFoundError:
            logger.info("Citations failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get citations for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving citations", 500)

    @require_storage
    def _get_evidence(self, handler, debate_id: str) -> HandlerResult:
        """Get comprehensive evidence trail for a debate.

        Combines grounded verdict with related evidence from ContinuumMemory.

        Returns:
            - grounded_verdict: Claim analysis with citations
            - related_evidence: Evidence snippets from memory
            - metadata: Search context and quality metrics
        """

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
                            related_evidence.append(
                                {
                                    "id": getattr(memory, "id", ""),
                                    "content": getattr(memory, "content", ""),
                                    "source": metadata.get("source", "unknown"),
                                    "importance": getattr(memory, "importance", 0.5),
                                    "tier": str(getattr(memory, "tier", "medium")),
                                }
                            )
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

        except RecordNotFoundError:
            logger.info("Evidence failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get evidence for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving evidence", 500)

    @require_storage
    def _get_debate_messages(
        self, debate_id: str, limit: int = 50, offset: int = 0
    ) -> HandlerResult:
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
            paginated_messages = messages[offset : offset + limit]

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

            return json_response(
                {
                    "debate_id": debate_id,
                    "messages": formatted_messages,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + len(paginated_messages) < total,
                }
            )

        except RecordNotFoundError:
            logger.info("Messages failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get messages for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving messages", 500)

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

            return json_response(
                {
                    "debate_id": debate_id,
                    "overall_quality": critique.overall_quality,
                    "productive_rounds": critique.productive_rounds,
                    "unproductive_rounds": critique.unproductive_rounds,
                    "observations": [
                        {
                            "type": o.observation_type,
                            "severity": o.severity,
                            "agent": getattr(o, "agent", None),
                            "round": getattr(o, "round_num", None),
                            "description": o.description,
                        }
                        for o in critique.observations
                    ],
                    "recommendations": critique.recommendations,
                }
            )
        except RecordNotFoundError:
            logger.info("Meta critique failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get meta critique for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving meta critique", 500)
        except ValueError as e:
            logger.warning("Invalid meta critique request for %s: %s", debate_id, e)
            return error_response(f"Invalid request: {e}", 400)

    def _get_graph_stats(self, debate_id: str) -> HandlerResult:
        """Get argument graph statistics for a debate.

        Returns node counts, edge counts, depth, branching factor, and complexity.
        """
        try:
            from aragora.debate.traces import DebateTrace
            from aragora.visualization.mapper import ArgumentCartographer
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
                    round_num=getattr(critique, "round", 1),
                    critique_text=critique.reasoning,
                )

            stats = cartographer.get_statistics()
            return json_response(stats)

        except RecordNotFoundError:
            logger.info("Graph stats failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to get graph stats for %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error retrieving graph stats", 500)
        except ValueError as e:
            logger.warning("Invalid graph stats request for %s: %s", debate_id, e)
            return error_response(f"Invalid request: {e}", 400)

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
        except FileNotFoundError:
            logger.info("Build graph failed - replay file not found: %s", replay_path)
            return error_response(f"Replay file not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to build graph from replay %s: %s: %s",
                debate_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error building graph", 500)
        except ValueError as e:
            logger.warning("Invalid replay data for %s: %s", debate_id, e)
            return error_response(f"Invalid replay data: {e}", 400)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        # Create debate endpoint - both legacy and RESTful
        # POST /api/debates (canonical) or POST /api/debate (legacy, deprecated)
        if path in ("/api/debate", "/api/debates"):
            result = self._create_debate(handler)

            # Add deprecation headers for legacy endpoint
            if path == "/api/debate" and result:
                # RFC 8594 Sunset header - 6 months from now
                result.headers = result.headers or {}
                result.headers["Deprecation"] = "true"
                result.headers["Sunset"] = "Sat, 01 Aug 2026 00:00:00 GMT"
                result.headers["Link"] = '</api/debates>; rel="successor-version"'
                logger.warning("Legacy endpoint /api/debate used. Use /api/debates instead.")
            return result

        # Batch submission endpoint
        if path in ("/api/debates/batch", "/api/debates/batch/"):
            return self._submit_batch(handler)

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

    @rate_limit(rpm=5, limiter_name="debates_create")
    @require_quota("debate")
    def _create_debate(self, handler) -> HandlerResult:
        """Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default varies)
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")
            auto_select: Whether to auto-select agents (optional, default: False)
            use_trending: Whether to use trending topic (optional, default: False)

        Rate limited and quota enforced via decorators.
        Returns 402 Payment Required if monthly debate quota exceeded.
        """
        logger.info("[_create_debate] Called via DebatesHandler")

        # Rate limit expensive debate creation
        try:
            if hasattr(handler, "_check_rate_limit") and not handler._check_rate_limit():
                logger.info("[_create_debate] Rate limit check failed")
                return error_response("Rate limit exceeded", 429)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.exception(f"[_create_debate] Rate limit check error: {e}")
            return error_response(f"Rate limit check failed: {e}", 500)

        logger.info("[_create_debate] Rate limit passed")

        # Tier-aware rate limiting based on subscription
        try:
            if hasattr(handler, "_check_tier_rate_limit") and not handler._check_tier_rate_limit():
                return error_response("Tier rate limit exceeded", 429)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning(f"Tier rate limit check failed, proceeding: {e}")

        # Check if debate orchestrator is available
        debate_available = False
        try:
            importlib.import_module("aragora.debate.orchestrator")
            debate_available = True
        except ImportError:
            pass

        if not debate_available:
            return error_response("Debate orchestrator not available", 500)

        stream_emitter = getattr(handler, "stream_emitter", None)
        if not stream_emitter:
            return error_response("Event streaming not configured", 500)

        # Read and validate request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        if not body:
            return error_response("No content provided", 400)

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, DEBATE_START_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        # Parse and validate request using DebateRequest
        try:
            from aragora.server.debate_controller import DebateRequest

            request = DebateRequest.from_dict(body)
        except ValueError as e:
            return error_response(str(e), 400)

        # Get debate controller and start debate
        try:
            controller = handler._get_debate_controller()
            response = controller.start_debate(request)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
            logger.exception(f"Failed to start debate: {e}")
            return error_response(f"Failed to start debate: {str(e)}", 500)

        # Note: Usage increment is handled by @require_quota decorator on success
        return json_response(response.to_dict(), status=response.status_code)

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

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, DEBATE_UPDATE_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

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
                    f"No valid fields to update. Allowed: {', '.join(allowed_fields)}", 400
                )

            # Validate and normalize status if provided
            if "status" in updates:
                # Accept both internal and SDK status values
                valid_internal = {"active", "paused", "concluded", "archived"}
                valid_sdk = {"pending", "running", "completed", "failed", "cancelled"}
                input_status = updates["status"]

                if input_status in valid_sdk:
                    # Convert SDK status to internal status for storage
                    updates["status"] = denormalize_status(input_status)
                elif input_status not in valid_internal:
                    all_valid = valid_internal | valid_sdk
                    return error_response(
                        f"Invalid status. Must be one of: {', '.join(sorted(all_valid))}", 400
                    )

            # Apply updates to debate
            for key, value in updates.items():
                debate[key] = value

            # Save updated debate
            storage.save_debate(debate_id, debate)

            logger.info(f"Debate {debate_id} updated: {list(updates.keys())}")

            # Return normalized status for SDK compatibility
            return json_response(
                {
                    "success": True,
                    "debate_id": debate_id,
                    "updated_fields": list(updates.keys()),
                    "debate": {
                        "id": debate_id,
                        "title": debate.get("title", debate.get("task", "")),
                        "status": normalize_status(debate.get("status", "active")),
                        "tags": debate.get("tags", []),
                    },
                }
            )
        except RecordNotFoundError:
            logger.info("Update failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Failed to update debate %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True
            )
            return error_response("Database error updating debate", 500)
        except ValueError as e:
            logger.warning("Invalid update request for %s: %s", debate_id, e)
            return error_response(f"Invalid update data: {e}", 400)
