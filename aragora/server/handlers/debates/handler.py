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
- GET /api/debates/{id}/rhetorical - Get rhetorical pattern observations
- GET /api/debates/{id}/trickster - Get trickster hollow consensus status
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
from aragora.rbac.decorators import require_permission
from aragora.server.debate_utils import _active_debates
from aragora.server.middleware.abac import Action, ResourceType, check_resource_access
from aragora.server.middleware.tier_enforcement import require_quota
from aragora.server.validation import validate_debate_id
from aragora.server.validation.schema import (
    DEBATE_START_SCHEMA,
    DEBATE_UPDATE_SCHEMA,
    validate_against_schema,
)
from aragora.server.http_utils import run_async
from aragora.resilience_patterns import with_timeout_sync

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    handle_errors,
    json_response,
    require_storage,
    safe_error_message,
    safe_json_parse,
    ttl_cache,
)
from .analysis import AnalysisOperationsMixin
from .batch import BatchOperationsMixin
from .export import ExportOperationsMixin
from .fork import ForkOperationsMixin
from .response_formatting import (
    CACHE_TTL_CONVERGENCE,
    CACHE_TTL_DEBATES_LIST,
    CACHE_TTL_IMPASSE,
    denormalize_status,
    normalize_debate_response,
    normalize_status,
)
from .search import SearchOperationsMixin
from ..utils.rate_limit import rate_limit, user_rate_limit

logger = logging.getLogger(__name__)


class DebatesHandler(
    AnalysisOperationsMixin,
    ExportOperationsMixin,
    ForkOperationsMixin,
    SearchOperationsMixin,
    BatchOperationsMixin,
    BaseHandler,
):
    """Handler for debate-related endpoints."""

    # Route patterns this handler manages
    ROUTES = [
        "/api/v1/debate",  # POST - create new debate (legacy endpoint)
        "/api/v1/debates",
        "/api/v1/debates/",  # With trailing slash
        "/api/v1/debates/batch",  # POST - batch debate submission
        "/api/v1/debates/batch/",
        "/api/v1/debates/batch/*/status",  # GET - batch status
        "/api/v1/debates/queue/status",  # GET - queue status
        "/api/v1/debates/export/batch",  # POST - start batch export
        "/api/v1/debates/export/batch/",
        "/api/v1/debates/export/batch/*/status",  # GET - export job status
        "/api/v1/debates/export/batch/*/results",  # GET - export job results
        "/api/v1/debates/export/batch/*/stream",  # GET - SSE progress stream
        "/api/v1/debates/slug/",
        "/api/v1/debates/*/export/",
        "/api/v1/debates/*/impasse",
        "/api/v1/debates/*/convergence",
        "/api/v1/debates/*/citations",
        "/api/v1/debates/*/messages",  # Paginated message history
        "/api/v1/debates/*/fork",  # POST - counterfactual fork
        "/api/v1/debates/*/followup",  # POST - crux-driven follow-up debate
        "/api/v1/debates/*/followups",  # GET - list follow-up suggestions
        "/api/v1/debates/*/forks",  # GET - list all forks for a debate
        "/api/v1/debates/*/verification-report",  # Verification feedback
        "/api/v1/debates/*/summary",  # GET - human-readable summary
        "/api/v1/debates/*/cancel",  # POST - cancel running debate
        "/api/v1/search",  # Cross-debate search
    ]

    # Endpoints that require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/v1/debates",  # List all debates - prevents enumeration
        "/api/v1/debates/batch",  # Batch submission requires auth
        "/export/",  # Export debate data
        "/citations",  # Evidence citations
        "/fork",  # Fork debate
        "/followup",  # Create follow-up debate
    ]

    # Allowed export formats and tables for input validation
    ALLOWED_EXPORT_FORMATS = {"json", "csv", "html", "txt", "md"}
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
        ("/rhetorical", "_get_rhetorical_observations", True, None),
        ("/trickster", "_get_trickster_status", True, None),
    ]

    def _check_auth(self, handler) -> Optional[HandlerResult]:
        """Check authentication for sensitive endpoints.

        Supports both:
        - JWT tokens (from Google OAuth, etc.)
        - API tokens (ara_* prefix)
        - Legacy HMAC tokens (for backwards compatibility)

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

        # If no API token is configured on the server, skip token authentication
        if not auth_config.api_token:
            return None

        # Extract auth token from Authorization header
        auth_header = None
        if hasattr(handler, "headers"):
            auth_header = handler.headers.get("Authorization", "")

        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not token:
            return error_response("Missing authentication token", 401)

        # Check for JWT tokens (3 base64url parts separated by dots)
        if token.count(".") == 2:
            try:
                from aragora.billing.auth import validate_access_token

                jwt_result = validate_access_token(token)
                if jwt_result:
                    return None  # JWT valid
                logger.debug("JWT token validation failed")
            except Exception as e:
                logger.debug(f"JWT validation error: {e}")

        # Check for API tokens (ara_* prefix)
        if token.startswith("ara_"):
            # API tokens are validated by rate limiter, just check format
            return None

        # Check if API token is configured for legacy HMAC tokens
        if not auth_config.api_token:
            logger.debug("No API token configured, skipping legacy auth check")
            return None

        # Validate legacy HMAC token
        if auth_config.validate_token(token):
            return None

        return error_response("Invalid or expired authentication token", 401)

    def _requires_auth(self, path: str) -> bool:
        """Check if the given path requires authentication."""
        # Normalize path for consistent checking
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")
        for pattern in self.AUTH_REQUIRED_ENDPOINTS:
            # Also normalize the pattern for comparison
            norm_pattern = pattern.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")
            if norm_pattern in normalized:
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
                        "_get_rhetorical_observations",
                        "_get_trickster_status",
                    ):
                        return method(debate_id)
                    return method(handler, debate_id)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path.

        Note: Paths may be normalized (version stripped) by handler_registry,
        so we check both versioned and unversioned variants.
        """
        # Normalize to unversioned for consistent checking
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")

        if normalized in ("/api/debate", "/api/debates"):
            return True  # POST - create debate, GET - list debates
        if normalized == "/api/search":
            return True
        if normalized.startswith("/api/debates/"):
            return True
        # Also handle /api/debate/{id}/meta-critique and /api/debate/{id}/graph/stats
        if normalized.startswith("/api/debate/") and (
            normalized.endswith("/meta-critique") or normalized.endswith("/graph/stats")
        ):
            return True
        return False

    @require_permission("debates:read")
    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route debate requests to appropriate handler methods.

        Note: Paths may be normalized (version stripped) by handler_registry,
        so we normalize to unversioned for consistent route matching.
        """
        # Normalize to unversioned for consistent checking
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")

        # Check authentication for protected endpoints
        if self._requires_auth(path):
            auth_error = self._check_auth(handler)
            if auth_error:
                return auth_error

        # Search endpoint
        if normalized == "/api/search":
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
        if normalized == "/api/debates/queue/status":
            return self._get_queue_status()

        # Batch status endpoint (GET /api/debates/batch/{id}/status)
        if normalized.startswith("/api/debates/batch/") and normalized.endswith("/status"):
            parts = normalized.split("/")
            if len(parts) >= 5:
                batch_id = parts[3]  # Index 3 for unversioned paths
                return self._get_batch_status(batch_id)

        # List batches (GET /api/debates/batch)
        if normalized in ("/api/debates/batch", "/api/debates/batch/"):
            limit = min(get_int_param(query_params, "limit", 50), 100)
            status_filter = query_params.get("status")
            return self._list_batches(limit, status_filter)

        # Batch export endpoints
        if normalized.startswith("/api/debates/export/batch"):
            return self._handle_batch_export(normalized, query_params, handler)

        # Exact path matches - list debates
        if normalized == "/api/debates":
            limit = min(get_int_param(query_params, "limit", 20), 100)
            # Get authenticated user for org-scoped results
            user = self.get_current_user(handler)
            org_id = user.org_id if user else None
            return self._list_debates(limit, org_id)

        if normalized.startswith("/api/debates/slug/"):
            slug = normalized.split("/")[-1]
            return self._get_debate_by_slug(handler, slug)

        # Dispatch suffix-based routes (impasse, convergence, citations, messages, etc.)
        result = self._dispatch_suffix_route(normalized, query_params, handler)
        if result:
            return result

        # Export route (special handling for format/table validation)
        # URL: /api/debates/{id}/export/{format}
        # Parts: ['', 'api', 'debates', '{id}', 'export', '{format}']
        if "/export/" in normalized:
            parts = normalized.split("/")
            if len(parts) >= 6:
                debate_id = parts[3]  # Index 3 for unversioned paths
                # Validate debate ID for export
                is_valid, err = validate_debate_id(debate_id)
                if not is_valid:
                    return error_response(err, 400)
                export_format = parts[5]  # Index 5 for unversioned paths
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
        if normalized.startswith("/api/debates/"):
            slug = normalized.split("/")[-1]
            if slug and slug not in ("impasse", "convergence"):
                return self._get_debate_by_slug(handler, slug)

        return None

    def _extract_debate_id(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """Extract and validate debate ID from path like /api/debates/{id}/impasse.

        Handles both versioned (/api/v1/debates/{id}) and unversioned (/api/debates/{id}) paths.

        Returns:
            Tuple of (debate_id, error_message). If error_message is set, debate_id is None.
        """
        # Normalize to unversioned path
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")
        parts = normalized.split("/")
        if len(parts) < 4:
            return None, "Invalid path"

        # For unversioned routes: ['', 'api', 'debates', '{id}', ...]
        debate_id = parts[3]
        is_valid, err = validate_debate_id(debate_id)
        if not is_valid:
            return None, err

        return debate_id, None

    @handle_errors("batch export")
    def _handle_batch_export(
        self, path: str, query_params: dict, handler
    ) -> Optional[HandlerResult]:
        """Route batch export requests to appropriate methods."""
        from aragora.server.http_utils import run_async

        # Normalize to unversioned for consistent checking
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")

        # POST /api/debates/export/batch - start batch export
        if normalized in ("/api/debates/export/batch", "/api/debates/export/batch/"):
            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid or missing JSON body", 400)
            debate_ids = body.get("debate_ids", [])
            format = body.get("format", "json")
            return self._start_batch_export(handler, debate_ids, format)  # type: ignore[misc]

        # GET /api/debates/export/batch - list export jobs
        if normalized == "/api/debates/export/batch":
            limit = min(get_int_param(query_params, "limit", 50), 100)
            return self._list_batch_exports(limit)  # type: ignore[misc]

        # Extract job ID from normalized path
        parts = normalized.split("/")
        if len(parts) < 5:
            return error_response("Invalid batch export path", 400)

        job_id = parts[4]

        # GET /api/debates/export/batch/{job_id}/status
        if path.endswith("/status"):
            return self._get_batch_export_status(job_id)  # type: ignore[misc]

        # GET /api/debates/export/batch/{job_id}/results
        if path.endswith("/results"):
            return self._get_batch_export_results(job_id)  # type: ignore[misc]

        # GET /api/debates/export/batch/{job_id}/stream - SSE stream
        if path.endswith("/stream"):

            async def stream():
                async for chunk in self._stream_batch_export_progress(job_id):
                    yield chunk

            return HandlerResult(
                status_code=200,
                content_type="text/event-stream",
                body=run_async(stream()),
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return error_response(f"Unknown batch export endpoint: {path}", 404)

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
            normalize_debate_response(d.__dict__ if hasattr(d, "__dict__") else d)  # type: ignore[arg-type]
            for d in debates
        ]
        return json_response({"debates": debates_list, "count": len(debates_list)})

    @require_storage
    @handle_errors("get debate by slug")
    def _get_debate_by_slug(self, handler, slug: str) -> HandlerResult:
        """Get a debate by slug.

        Checks both persistent storage and in-progress debates (_active_debates).
        In-progress debates haven't been persisted yet but should still be queryable.
        """
        # First check persistent storage
        storage = self.get_storage()
        debate = storage.get_debate(slug)
        if debate:
            return json_response(normalize_debate_response(debate))

        # Fallback: check in-progress debates that haven't been persisted yet
        if slug in _active_debates:
            active = _active_debates[slug]
            # Return minimal info for in-progress debate
            # Support both "task" (new) and "question" (legacy) field names
            return json_response(
                {
                    "id": slug,
                    "debate_id": slug,
                    "task": active.get("task") or active.get("question", ""),
                    "status": normalize_status(active.get("status", "starting")),
                    "agents": (
                        active.get("agents", "").split(",")
                        if isinstance(active.get("agents"), str)
                        else active.get("agents", [])
                    ),
                    "rounds": active.get("rounds", 3),
                    "in_progress": True,
                }
            )

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
                    memories = continuum.search(  # type: ignore[attr-defined]
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

    @require_permission("debates:create")
    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        # Create debate endpoint - both legacy and RESTful
        # POST /api/debates (canonical) or POST /api/debate (legacy, deprecated)
        # Note: path is normalized (version stripped), so check both versioned and unversioned
        if path in ("/api/v1/debate", "/api/v1/debates", "/api/debate", "/api/debates"):
            result = self._create_debate(handler)

            # Add deprecation headers for legacy endpoint
            if path in ("/api/v1/debate", "/api/debate") and result:
                # RFC 8594 Sunset header - 6 months from now
                result.headers = result.headers or {}
                result.headers["Deprecation"] = "true"
                result.headers["Sunset"] = "Sat, 01 Aug 2026 00:00:00 GMT"
                result.headers["Link"] = '</api/debates>; rel="successor-version"'
                logger.warning("Legacy endpoint /api/debate used. Use /api/debates instead.")
            return result

        # Batch submission endpoint
        if path in (
            "/api/v1/debates/batch",
            "/api/v1/debates/batch/",
            "/api/debates/batch",
            "/api/debates/batch/",
        ):
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

        if path.endswith("/cancel"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._cancel_debate(handler, debate_id)

        return None

    @with_timeout_sync(120.0)
    @user_rate_limit(action="debate_create")
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

        Routes through DecisionRouter for unified routing, deduplication,
        and caching. Falls back to direct controller if router unavailable.

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

        # Spam check for debate content
        spam_result = self._check_spam_content(body)
        if spam_result:
            return spam_result

        # Use direct controller approach for ad-hoc debates.
        # The DecisionRouter is designed for synchronous decisions and blocks
        # until the debate completes (minutes). For HTTP requests, we need to
        # return immediately with the debate_id and let the client poll/stream.
        #
        # DecisionRouter can still be used for:
        # - Chat connector decisions (Slack/Telegram) that need sync responses
        # - Internal orchestration where blocking is acceptable
        return self._create_debate_direct(handler, body)

    async def _route_through_decision_router(
        self, handler, body: dict, headers: dict
    ) -> HandlerResult:
        """Route debate creation through DecisionRouter.

        This provides unified handling including:
        - Deduplication of concurrent identical requests
        - Result caching
        - Origin registration for bidirectional routing
        - Unified metrics and tracing
        """
        from aragora.core.decision import (
            DecisionRequest,
            DecisionType,
            InputSource,
            get_decision_router,
        )

        # Create unified decision request from HTTP body
        request = DecisionRequest.from_http(body, headers)

        # Force debate type for this endpoint
        request.decision_type = DecisionType.DEBATE
        request.source = InputSource.HTTP_API

        # Get authenticated user context
        user = self.get_current_user(handler)
        if user:
            if not request.context.user_id:
                request.context.user_id = user.user_id
            if not request.context.workspace_id:
                request.context.workspace_id = getattr(user, "org_id", None)

        # Route through DecisionRouter
        router = get_decision_router()
        result = await router.route(request)

        logger.info(
            f"DecisionRouter completed debate {request.request_id} "
            f"(success={result.success}, debate_id={getattr(result, 'debate_id', 'N/A')})"
        )

        # Build response
        status_code = 200 if result.success else 500
        response_data = {
            "request_id": request.request_id,
            "debate_id": getattr(result, "debate_id", request.request_id),
            "status": "completed" if result.success else "failed",
            "decision_type": result.decision_type.value,
            "answer": result.answer,
            "confidence": result.confidence,
            "consensus_reached": result.consensus_reached,
            "reasoning": result.reasoning,
            "evidence_used": result.evidence_used,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

        return json_response(response_data, status=status_code)

    def _create_debate_direct(self, handler, body: dict) -> HandlerResult:
        """Direct debate creation via controller (fallback path)."""
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
            return error_response(safe_error_message(e, "start debate"), 500)

        # Note: Usage increment is handled by @require_quota decorator on success
        return json_response(response.to_dict(), status=response.status_code)

    def _cancel_debate(self, handler, debate_id: str) -> HandlerResult:
        """Cancel a running debate.

        Marks the debate as cancelled and attempts to cancel any running tasks.

        Args:
            handler: The HTTP handler
            debate_id: ID of the debate to cancel

        Returns:
            HandlerResult with cancellation status
        """
        from aragora.server.debate_utils import update_debate_status
        from aragora.server.state import get_state_manager
        from aragora.server.stream import StreamEvent, StreamEventType

        manager = get_state_manager()
        state = manager.get_debate(debate_id)

        if not state:
            # Check if debate exists in storage but already completed
            storage = self.get_storage()
            if storage:
                debate = storage.get_debate(debate_id)
                if debate:
                    return error_response(
                        f"Debate {debate_id} already completed (status: {debate.get('status', 'unknown')})",
                        400,
                    )
            return error_response(f"Debate not found: {debate_id}", 404)

        # Check if debate is in a cancellable state
        if state.status not in ("running", "starting"):
            return error_response(
                f"Debate {debate_id} cannot be cancelled (status: {state.status})",
                400,
            )

        # Mark as cancelled
        update_debate_status(debate_id, "cancelled", error="Cancelled by user")
        manager.update_debate_status(debate_id, status="cancelled")

        # Try to cancel the asyncio task if tracked
        task = state.metadata.get("_task")
        if task and hasattr(task, "cancel") and not getattr(task, "done", lambda: True)():
            try:
                task.cancel()
                logger.info(f"Cancelled running task for debate {debate_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel task for {debate_id}: {e}")

        # Emit cancellation event to all subscribers
        stream_emitter = getattr(handler, "stream_emitter", None)
        if stream_emitter:
            stream_emitter.emit(
                StreamEvent(
                    type=StreamEventType.DEBATE_END,
                    data={
                        "debate_id": debate_id,
                        "status": "cancelled",
                        "reason": "Cancelled by user",
                    },
                    loop_id=debate_id,
                )
            )

        logger.info(f"Debate {debate_id} cancelled by user")

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "status": "cancelled",
                "message": "Debate cancelled successfully",
            }
        )

    @require_permission("debates:update")
    def handle_patch(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route PATCH requests to appropriate methods."""
        # Handle /api/debates/{id} pattern for updates
        if path.startswith("/api/v1/debates/") and path.count("/") == 4:
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

            # ABAC: Check if user has write access to this debate
            user = self.get_current_user(handler)
            if user:
                debate_owner_id = debate.get("user_id") or debate.get("owner_id")
                debate_workspace_id = debate.get("workspace_id") or debate.get("org_id")

                access_decision = check_resource_access(
                    user_id=user.user_id,
                    user_role=getattr(user, "role", "user"),
                    user_plan=getattr(user, "plan", "free"),
                    resource_type=ResourceType.DEBATE,
                    resource_id=debate_id,
                    action=Action.WRITE,
                    resource_owner_id=debate_owner_id,
                    resource_workspace_id=debate_workspace_id,
                    user_workspace_id=getattr(user, "org_id", None),
                    user_workspace_role=getattr(user, "org_role", None),
                )

                if not access_decision.allowed:
                    logger.warning(
                        f"ABAC denied WRITE access to debate {debate_id} for user {user.user_id}: "
                        f"{access_decision.reason}"
                    )
                    return error_response(
                        "You do not have permission to update this debate",
                        403,
                    )

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
            storage.save_debate(debate_id, debate)  # type: ignore[attr-defined]

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

    def _check_spam_content(self, body: dict) -> Optional[HandlerResult]:
        """
        Check debate input content for spam.

        Runs the spam moderation integration against the debate task/question
        and context. Returns an error response if content should be blocked,
        or None if content passes.

        Args:
            body: The request body containing task/question and context

        Returns:
            HandlerResult with 400 error if spam detected, None otherwise
        """
        try:
            from aragora.moderation import check_debate_content, ContentModerationError

            # Extract content to check
            proposal = body.get("task") or body.get("question", "")
            context = body.get("context", "")

            if not proposal:
                return None  # Let schema validation handle this

            # Run async spam check
            result = run_async(check_debate_content(proposal, context))

            if result.should_block:
                logger.warning(
                    f"Spam content blocked: verdict={result.verdict.value}, "
                    f"confidence={result.confidence:.2f}, "
                    f"reasons={result.reasons[:3]}"
                )
                return error_response(  # type: ignore[call-arg]
                    "Content blocked by spam filter. Please revise your input.",
                    400,
                    extra={
                        "verdict": result.verdict.value,
                        "reasons": result.reasons[:3],
                    },
                )

            if result.should_flag_for_review:
                # Log suspicious content but allow it through
                logger.info(
                    f"Suspicious content flagged: verdict={result.verdict.value}, "
                    f"confidence={result.confidence:.2f}"
                )

            return None  # Content passed

        except ImportError:
            # Moderation module not available - allow content through
            logger.debug("Spam moderation not available, skipping check")
            return None
        except ContentModerationError as e:
            # Moderation explicitly rejected content
            logger.warning(f"Content moderation error: {e}")
            return error_response(str(e), 400)
        except Exception as e:
            # Unexpected error - log but allow content through (fail-open)
            logger.error(f"Spam check failed unexpectedly: {e}", exc_info=True)
            return None
