"""
Debate-related endpoint handlers.

This is the main DebatesHandler class that composes functionality from specialized mixins:
- AnalysisOperationsMixin: Meta-critique and argument graph analysis
- BatchOperationsMixin: Batch debate submission and processing
- CreateOperationsMixin: Debate creation and cancellation
- CrudOperationsMixin: List, get, update, delete debates
- EvidenceOperationsMixin: Citations, evidence, verification reports
- ExportOperationsMixin: Export in various formats
- ForkOperationsMixin: Counterfactual forks and follow-ups
- RoutingMixin: Route dispatch and authentication
- SearchOperationsMixin: Cross-debate search

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
- DELETE /api/debates/{id} - Permanently delete a debate (cascades to critiques)
- GET /api/search - Cross-debate search by query
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.http_utils import run_async
from aragora.server.validation import validate_debate_id
from aragora.server.debate_utils import _active_debates

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    handle_errors,
    json_response,
)
from ..openapi_decorator import api_endpoint

# Import all mixins
from .analysis import AnalysisOperationsMixin
from .batch import BatchOperationsMixin
from .create import CreateOperationsMixin
from .crud import CrudOperationsMixin
from .evidence import EvidenceOperationsMixin
from .export import ExportOperationsMixin
from .fork import ForkOperationsMixin
from .routing import (
    ALLOWED_EXPORT_FORMATS,
    ALLOWED_EXPORT_TABLES,
    ARTIFACT_ENDPOINTS,
    AUTH_REQUIRED_ENDPOINTS,
    ROUTES,
    SUFFIX_ROUTES,
    RoutingMixin,
)
from .search import SearchOperationsMixin


logger = logging.getLogger(__name__)


class DebatesHandler(
    AnalysisOperationsMixin,
    BatchOperationsMixin,
    CreateOperationsMixin,
    CrudOperationsMixin,
    EvidenceOperationsMixin,
    ExportOperationsMixin,
    ForkOperationsMixin,
    RoutingMixin,
    SearchOperationsMixin,
    BaseHandler,
):
    """Handler for debate-related endpoints.

    Composes functionality from specialized mixins for better modularity.
    Each mixin provides a specific category of operations:
    - Analysis: Meta-critique and graph analysis
    - Batch: Batch submission and processing
    - Create: Debate creation and cancellation
    - CRUD: List, get, update, delete
    - Evidence: Citations and verification reports
    - Export: Export in various formats
    - Fork: Counterfactual forks and follow-ups
    - Routing: Route dispatch and authentication
    - Search: Cross-debate search
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        if server_context is not None:
            self.ctx = server_context
        else:
            self.ctx = ctx or {}  # type: ignore[assignment]  # dict is runtime-compatible with ServerContext

    # Route patterns this handler manages (from routing module)
    ROUTES = ROUTES
    AUTH_REQUIRED_ENDPOINTS = AUTH_REQUIRED_ENDPOINTS
    ALLOWED_EXPORT_FORMATS = ALLOWED_EXPORT_FORMATS
    ALLOWED_EXPORT_TABLES = ALLOWED_EXPORT_TABLES
    ARTIFACT_ENDPOINTS = ARTIFACT_ENDPOINTS
    SUFFIX_ROUTES = SUFFIX_ROUTES

    @require_permission("debates:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
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
        if normalized in ("/api/search", "/api/debates/search"):
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

    @handle_errors("batch export")
    def _handle_batch_export(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route batch export requests to appropriate methods."""
        # Normalize to unversioned for consistent checking
        normalized = path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")

        # POST /api/debates/export/batch - start batch export
        if normalized in ("/api/debates/export/batch", "/api/debates/export/batch/"):
            body = self.read_json_body(handler)
            if not body:
                return error_response("Invalid or missing JSON body", 400)
            debate_ids = body.get("debate_ids", [])
            format = body.get("format", "json")
            return self._start_batch_export(handler, debate_ids, format)  # type: ignore[misc]  # mixin protocol self

        # GET /api/debates/export/batch - list export jobs
        if normalized == "/api/debates/export/batch":
            limit = min(get_int_param(query_params, "limit", 50), 100)
            return self._list_batch_exports(limit)  # type: ignore[misc]  # mixin protocol self

        # Extract job ID from normalized path
        parts = normalized.split("/")
        if len(parts) < 5:
            return error_response("Invalid batch export path", 400)

        job_id = parts[4]

        # GET /api/debates/export/batch/{job_id}/status
        if path.endswith("/status"):
            return self._get_batch_export_status(job_id)  # type: ignore[misc]  # mixin protocol self

        # GET /api/debates/export/batch/{job_id}/results
        if path.endswith("/results"):
            return self._get_batch_export_results(job_id)  # type: ignore[misc]  # mixin protocol self

        # GET /api/debates/export/batch/{job_id}/stream - SSE stream
        if path.endswith("/stream"):

            async def stream() -> AsyncIterator[Any]:
                async for chunk in self._stream_batch_export_progress(job_id):  # type: ignore[misc]  # mixin protocol self
                    yield chunk

            return HandlerResult(
                status_code=200,
                content_type="text/event-stream",
                body=run_async(stream()),  # type: ignore[arg-type]  # async generator used as SSE stream body
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return error_response(f"Unknown batch export endpoint: {path}", 404)

    @require_permission("debates:create")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
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

    @require_permission("debates:update")
    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route PATCH requests to appropriate methods."""
        # Handle /api/debates/{id} pattern for updates
        if path.startswith("/api/v1/debates/") and path.count("/") == 4:
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._patch_debate(handler, debate_id)
        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route DELETE requests to appropriate methods."""
        # Handle DELETE /api/debates/{id}
        if path.startswith("/api/v1/debates/") and path.count("/") == 4:
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            if debate_id:
                return self._delete_debate(handler, debate_id)
        return None


# Backward compatibility alias
DebateHandler = DebatesHandler

__all__ = ["DebatesHandler", "DebateHandler"]
