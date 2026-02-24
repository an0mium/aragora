"""
HTTP Handler for Knowledge Gap Detection.

Provides endpoints for knowledge gap analysis:
- GET /api/v1/knowledge/gaps - Detect coverage gaps, staleness, contradictions
- GET /api/v1/knowledge/gaps/recommendations - Get prioritized recommendations
- GET /api/v1/knowledge/gaps/score - Get coverage score for a domain
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.decorators import handle_errors
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for gap detection endpoints (expensive queries)
_gaps_limiter = RateLimiter(requests_per_minute=20)


class KnowledgeGapHandler(BaseHandler):
    """Handler for knowledge gap detection endpoints.

    Endpoints:
        GET /api/v1/knowledge/gaps - Detect gaps (coverage, staleness, contradictions)
        GET /api/v1/knowledge/gaps/recommendations - Get improvement recommendations
        GET /api/v1/knowledge/gaps/score - Get domain coverage score
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = server_context or ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/knowledge/gaps")

    @handle_errors
    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Handle GET requests for knowledge gap detection."""
        # Rate limit check
        if hasattr(handler, "get_client_ip"):
            client_ip = handler.get_client_ip()
        else:
            client_ip = get_client_ip(handler)
        if not _gaps_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        workspace_id = query_params.get("workspace_id", "default")

        if path == "/api/v1/knowledge/gaps":
            return await self._get_gaps(workspace_id, query_params)

        if path == "/api/v1/knowledge/gaps/recommendations":
            return await self._get_recommendations(workspace_id, query_params)

        if path == "/api/v1/knowledge/gaps/score":
            return await self._get_score(workspace_id, query_params)

        return None

    async def _get_gaps(
        self,
        workspace_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get knowledge gaps across all dimensions."""
        try:
            detector = self._create_detector(workspace_id)
            if detector is None:
                return self._unavailable_response(workspace_id)

            domain = query_params.get("domain")
            max_age_days = int(query_params.get("max_age_days", "90"))

            response: dict[str, Any] = {
                "workspace_id": workspace_id,
            }

            # Coverage gaps
            if domain:
                coverage_gaps = await detector.detect_coverage_gaps(domain)
                response["coverage_gaps"] = [g.to_dict() for g in coverage_gaps]
            else:
                response["coverage_gaps"] = []

            # Staleness
            stale = await detector.detect_staleness(max_age_days=max_age_days)
            response["stale_entries"] = [s.to_dict() for s in stale[:50]]
            response["stale_count"] = len(stale)

            # Contradictions
            contradictions = await detector.detect_contradictions()
            response["contradictions"] = [c.to_dict() for c in contradictions[:50]]
            response["contradiction_count"] = len(contradictions)

            return json_response({"data": response})

        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Failed to detect knowledge gaps: %s", e)
            return error_response("Failed to detect knowledge gaps", 500)

    async def _get_recommendations(
        self,
        workspace_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get prioritized recommendations for knowledge improvement."""
        try:
            detector = self._create_detector(workspace_id)
            if detector is None:
                return self._unavailable_response(workspace_id)

            domain = query_params.get("domain")
            limit = int(query_params.get("limit", "20"))

            recommendations = await detector.get_recommendations(
                domain=domain,
                limit=limit,
            )

            return json_response({
                "data": {
                    "recommendations": [r.to_dict() for r in recommendations],
                    "count": len(recommendations),
                    "workspace_id": workspace_id,
                }
            })

        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Failed to get recommendations: %s", e)
            return error_response("Failed to get recommendations", 500)

    async def _get_score(
        self,
        workspace_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get coverage score for a specific domain."""
        domain = query_params.get("domain")
        if not domain:
            return error_response("Missing required parameter: domain", 400)

        try:
            detector = self._create_detector(workspace_id)
            if detector is None:
                return self._unavailable_response(workspace_id)

            score = await detector.get_coverage_score(domain)

            return json_response({
                "data": {
                    "domain": domain,
                    "coverage_score": score,
                    "workspace_id": workspace_id,
                }
            })

        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Failed to get coverage score: %s", e)
            return error_response("Failed to get coverage score", 500)

    def _create_detector(self, workspace_id: str) -> Any:
        """Create a KnowledgeGapDetector instance with lazy imports.

        Returns:
            KnowledgeGapDetector instance, or None if KM is unavailable
        """
        try:
            from aragora.knowledge.gap_detector import KnowledgeGapDetector
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound(workspace_id)
            return KnowledgeGapDetector(mound=mound, workspace_id=workspace_id)
        except (ImportError, RuntimeError) as e:
            logger.warning("Knowledge Mound unavailable for gap detection: %s", e)
            return None

    def _unavailable_response(self, workspace_id: str) -> HandlerResult:
        """Return a response indicating KM is unavailable."""
        return json_response({
            "data": {
                "coverage_gaps": [],
                "stale_entries": [],
                "stale_count": 0,
                "contradictions": [],
                "contradiction_count": 0,
                "workspace_id": workspace_id,
                "status": "knowledge_mound_unavailable",
            }
        })


__all__ = ["KnowledgeGapHandler"]
