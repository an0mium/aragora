"""
Explainability API handler.

Provides endpoints for understanding debate decisions:
- GET /api/v1/debates/{id}/explanation - Full decision explanation
- GET /api/v1/debates/{id}/evidence - Evidence chain
- GET /api/v1/debates/{id}/votes/pivots - Vote influence analysis
- GET /api/v1/debates/{id}/counterfactuals - Counterfactual analysis
- GET /api/v1/debates/{id}/summary - Human-readable summary
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Cache for built Decision objects (simple TTL cache)
_decision_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_decision(debate_id: str) -> Optional[Any]:
    """Get cached decision if not expired."""
    import time

    if debate_id not in _decision_cache:
        return None

    timestamp = _cache_timestamps.get(debate_id, 0)
    if time.time() - timestamp > CACHE_TTL_SECONDS:
        del _decision_cache[debate_id]
        del _cache_timestamps[debate_id]
        return None

    return _decision_cache[debate_id]


def _cache_decision(debate_id: str, decision: Any) -> None:
    """Cache a decision."""
    import time

    _decision_cache[debate_id] = decision
    _cache_timestamps[debate_id] = time.time()

    # Prune old entries
    if len(_decision_cache) > 100:
        oldest = min(_cache_timestamps, key=_cache_timestamps.get)  # type: ignore
        del _decision_cache[oldest]
        del _cache_timestamps[oldest]


class ExplainabilityHandler(BaseHandler):
    """Handler for debate explainability endpoints."""

    # API v1 routes
    ROUTES = [
        "/api/v1/debates/*/explanation",
        "/api/v1/debates/*/evidence",
        "/api/v1/debates/*/votes/pivots",
        "/api/v1/debates/*/counterfactuals",
        "/api/v1/debates/*/summary",
        "/api/v1/explain/*",
        # Legacy routes (deprecated)
        "/api/debates/*/explanation",
        "/api/explain/*",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        if method != "GET":
            return False

        # Check versioned routes
        if path.startswith("/api/v1/debates/") and any(
            path.endswith(suffix)
            for suffix in ["/explanation", "/evidence", "/votes/pivots", "/counterfactuals", "/summary"]
        ):
            return True

        # Check explain shortcut
        if path.startswith("/api/v1/explain/"):
            return True

        # Legacy routes
        if path.startswith("/api/debates/") and path.endswith("/explanation"):
            return True
        if path.startswith("/api/explain/"):
            return True

        return False

    def _is_legacy_route(self, path: str) -> bool:
        """Check if this is a legacy (non-versioned) route."""
        return not path.startswith("/api/v1/")

    @rate_limit(rpm=60)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route explainability requests."""
        # Add deprecation headers for legacy routes
        is_legacy = self._is_legacy_route(path)

        # Normalize path
        if path.startswith("/api/v1/"):
            normalized = path[8:]  # Remove /api/v1/
        else:
            normalized = path[5:]  # Remove /api/

        # Extract debate_id
        parts = normalized.split("/")

        # Handle /explain/{id} shortcut
        if parts[0] == "explain" and len(parts) >= 2:
            debate_id = parts[1]
            return self._handle_full_explanation(debate_id, query_params, is_legacy)

        # Handle /debates/{id}/...
        if parts[0] == "debates" and len(parts) >= 3:
            debate_id = parts[1]
            endpoint = "/".join(parts[2:])

            if endpoint == "explanation":
                return self._handle_full_explanation(debate_id, query_params, is_legacy)
            elif endpoint == "evidence":
                return self._handle_evidence(debate_id, query_params, is_legacy)
            elif endpoint == "votes/pivots":
                return self._handle_vote_pivots(debate_id, query_params, is_legacy)
            elif endpoint == "counterfactuals":
                return self._handle_counterfactuals(debate_id, query_params, is_legacy)
            elif endpoint == "summary":
                return self._handle_summary(debate_id, query_params, is_legacy)

        return error_response("Invalid explainability endpoint", 400)

    def _add_headers(self, result: HandlerResult, is_legacy: bool) -> HandlerResult:
        """Add version and deprecation headers."""
        if result.headers is None:
            result.headers = {}

        result.headers["X-API-Version"] = "v1"

        if is_legacy:
            result.headers["Deprecation"] = "true"
            result.headers["Sunset"] = "2026-06-01"

        return result

    async def _get_or_build_decision(self, debate_id: str) -> Optional[Any]:
        """Get decision from cache or build it."""
        # Check cache
        decision = _get_cached_decision(debate_id)
        if decision:
            return decision

        # Get debate result from storage
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if not db:
                return None

            debate_data = db.get(debate_id)
            if not debate_data:
                return None

            # Build decision
            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()

            # Convert dict to simple object for builder
            class ResultProxy:
                def __init__(self, data: Dict[str, Any]):
                    for k, v in data.items():
                        setattr(self, k, v)

            result = ResultProxy(debate_data)
            result.id = debate_id  # Ensure id is set

            decision = await builder.build(result)
            _cache_decision(debate_id, decision)

            return decision

        except Exception as e:
            logger.error(f"Failed to build decision for {debate_id}: {e}")
            return None

    def _handle_full_explanation(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle full explanation request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get format preference
            format_type = get_string_param(query_params, "format", "json")

            if format_type == "summary":
                from aragora.explainability import ExplanationBuilder

                builder = ExplanationBuilder()
                summary = builder.generate_summary(decision)

                result = HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=summary.encode("utf-8"),
                )
            else:
                result = json_response(decision.to_dict())

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Explanation error for {debate_id}: {e}")
            return error_response(f"Failed to generate explanation: {str(e)[:100]}", 500)

    def _handle_evidence(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle evidence chain request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            limit = int(get_string_param(query_params, "limit", "20"))
            min_relevance = float(get_string_param(query_params, "min_relevance", "0.0"))

            evidence = decision.evidence_chain
            if min_relevance > 0:
                evidence = [e for e in evidence if e.relevance_score >= min_relevance]

            evidence = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)[:limit]

            result = json_response({
                "debate_id": debate_id,
                "evidence_count": len(evidence),
                "evidence_quality_score": decision.evidence_quality_score,
                "evidence": [e.to_dict() for e in evidence],
            })

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Evidence error for {debate_id}: {e}")
            return error_response(f"Failed to get evidence: {str(e)[:100]}", 500)

    def _handle_vote_pivots(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle vote pivot analysis request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            min_influence = float(get_string_param(query_params, "min_influence", "0.0"))

            pivots = decision.vote_pivots
            if min_influence > 0:
                pivots = [p for p in pivots if p.influence_score >= min_influence]

            result = json_response({
                "debate_id": debate_id,
                "total_votes": len(decision.vote_pivots),
                "pivotal_votes": len(pivots),
                "agent_agreement_score": decision.agent_agreement_score,
                "votes": [p.to_dict() for p in pivots],
            })

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Vote pivot error for {debate_id}: {e}")
            return error_response(f"Failed to get vote pivots: {str(e)[:100]}", 500)

    def _handle_counterfactuals(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle counterfactual analysis request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Get filter params
            min_sensitivity = float(get_string_param(query_params, "min_sensitivity", "0.0"))

            counterfactuals = decision.counterfactuals
            if min_sensitivity > 0:
                counterfactuals = [
                    c for c in counterfactuals if c.sensitivity >= min_sensitivity
                ]

            result = json_response({
                "debate_id": debate_id,
                "counterfactual_count": len(counterfactuals),
                "counterfactuals": [c.to_dict() for c in counterfactuals],
            })

            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Counterfactual error for {debate_id}: {e}")
            return error_response(f"Failed to get counterfactuals: {str(e)[:100]}", 500)

    def _handle_summary(
        self, debate_id: str, query_params: Dict[str, Any], is_legacy: bool
    ) -> HandlerResult:
        """Handle human-readable summary request."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            decision = loop.run_until_complete(self._get_or_build_decision(debate_id))

            if not decision:
                return error_response(f"Debate not found: {debate_id}", 404)

            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()
            summary = builder.generate_summary(decision)

            # Get format preference
            format_type = get_string_param(query_params, "format", "markdown")

            if format_type == "json":
                result = json_response({
                    "debate_id": debate_id,
                    "summary": summary,
                    "confidence": decision.confidence,
                    "consensus_reached": decision.consensus_reached,
                })
            elif format_type == "html":
                import markdown
                html_content = f"""
<!DOCTYPE html>
<html>
<head><title>Decision Summary - {debate_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 1rem; }}
h2 {{ color: #333; }}
h3 {{ color: #666; }}
</style>
</head>
<body>
{markdown.markdown(summary)}
</body>
</html>
"""
                result = HandlerResult(
                    status_code=200,
                    content_type="text/html",
                    body=html_content.encode("utf-8"),
                )
            else:
                result = HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=summary.encode("utf-8"),
                )

            return self._add_headers(result, is_legacy)

        except ImportError:
            # markdown not available, return plain text
            from aragora.explainability import ExplanationBuilder

            builder = ExplanationBuilder()
            summary = builder.generate_summary(decision)

            result = HandlerResult(
                status_code=200,
                content_type="text/plain",
                body=summary.encode("utf-8"),
            )
            return self._add_headers(result, is_legacy)

        except Exception as e:
            logger.error(f"Summary error for {debate_id}: {e}")
            return error_response(f"Failed to generate summary: {str(e)[:100]}", 500)


# Handler factory
_explainability_handler: Optional["ExplainabilityHandler"] = None


def get_explainability_handler(server_context: Optional[Dict] = None) -> "ExplainabilityHandler":
    """Get or create the explainability handler instance."""
    global _explainability_handler
    if _explainability_handler is None:
        if server_context is None:
            server_context = {}
        _explainability_handler = ExplainabilityHandler(server_context)
    return _explainability_handler


__all__ = ["ExplainabilityHandler", "get_explainability_handler"]
