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
"""

from typing import Optional
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param


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
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/debates":
            return True
        if path.startswith("/api/debates/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """
        Route debate requests to appropriate handler methods.

        Note: This delegates to the unified server's existing methods
        to maintain backward compatibility.
        """
        if path == "/api/debates":
            limit = get_int_param(query_params, 'limit', 20)
            limit = min(limit, 100)  # Cap at 100
            return self._list_debates(handler, limit)

        if path.startswith("/api/debates/slug/"):
            slug = path.split("/")[-1]
            return self._get_debate_by_slug(handler, slug)

        if path.endswith("/impasse"):
            debate_id = self._extract_debate_id(path)
            if debate_id:
                return self._get_impasse(handler, debate_id)

        if path.endswith("/convergence"):
            debate_id = self._extract_debate_id(path)
            if debate_id:
                return self._get_convergence(handler, debate_id)

        if path.endswith("/citations"):
            debate_id = self._extract_debate_id(path)
            if debate_id:
                return self._get_citations(handler, debate_id)

        if "/export/" in path:
            parts = path.split("/")
            if len(parts) >= 6:
                debate_id = parts[3]
                export_format = parts[5]
                table = query_params.get('table', 'summary')
                return self._export_debate(handler, debate_id, export_format, table)

        # Default: treat as slug lookup
        if path.startswith("/api/debates/"):
            slug = path.split("/")[-1]
            if slug and slug not in ("impasse", "convergence"):
                return self._get_debate_by_slug(handler, slug)

        return None

    def _extract_debate_id(self, path: str) -> Optional[str]:
        """Extract debate ID from path like /api/debates/{id}/impasse."""
        parts = path.split("/")
        if len(parts) >= 4:
            return parts[3]
        return None

    def _list_debates(self, handler, limit: int) -> HandlerResult:
        """List recent debates."""
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        try:
            debates = storage.list_debates(limit=limit)
            return json_response({"debates": debates, "count": len(debates)})
        except Exception as e:
            return error_response(f"Failed to list debates: {e}", 500)

    def _get_debate_by_slug(self, handler, slug: str) -> HandlerResult:
        """Get a debate by slug."""
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        try:
            debate = storage.get_debate(slug)
            if debate:
                return json_response(debate)
            return error_response(f"Debate not found: {slug}", 404)
        except Exception as e:
            return error_response(f"Failed to get debate: {e}", 500)

    def _get_impasse(self, handler, debate_id: str) -> HandlerResult:
        """Detect impasse in a debate."""
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Analyze for impasse indicators
            messages = debate.get("messages", [])
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
        except Exception as e:
            return error_response(f"Impasse detection failed: {e}", 500)

    def _get_convergence(self, handler, debate_id: str) -> HandlerResult:
        """Get convergence status for a debate."""
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        try:
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
        except Exception as e:
            return error_response(f"Convergence check failed: {e}", 500)

    def _export_debate(self, handler, debate_id: str, format: str, table: str) -> HandlerResult:
        """Export debate in specified format."""
        valid_formats = {"json", "csv", "html"}
        if format not in valid_formats:
            return error_response(f"Invalid format: {format}. Valid: {valid_formats}", 400)

        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            if format == "json":
                return json_response(debate)

            # CSV/HTML would need additional formatting
            return error_response(f"Format {format} not yet implemented", 501)

        except Exception as e:
            return error_response(f"Export failed: {e}", 500)

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
        if not storage:
            return error_response("Storage not available", 503)

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
            if isinstance(grounded_verdict_raw, str):
                try:
                    grounded_verdict = json_module.loads(grounded_verdict_raw)
                except json_module.JSONDecodeError:
                    grounded_verdict = None
            else:
                grounded_verdict = grounded_verdict_raw

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
            return error_response(f"Failed to get citations: {e}", 500)
