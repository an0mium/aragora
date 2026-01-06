"""
Persona laboratory endpoint handlers.

Endpoints:
- GET /api/laboratory/emergent-traits - Get emergent traits from agent performance
- POST /api/laboratory/cross-pollinations/suggest - Suggest beneficial trait transfers
"""

import logging
from typing import Optional

from aragora.utils.optional_imports import try_import
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
    get_bounded_float_param,
    get_clamped_int_param,
)

logger = logging.getLogger(__name__)

# Optional PersonaLaboratory import
_lab_imports, LABORATORY_AVAILABLE = try_import(
    "aragora.agents.laboratory",
    "PersonaLaboratory"
)
PersonaLaboratory = _lab_imports.get("PersonaLaboratory")


class LaboratoryHandler(BaseHandler):
    """Handler for persona laboratory endpoints."""

    ROUTES = [
        "/api/laboratory/emergent-traits",
        "/api/laboratory/cross-pollinations/suggest",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/laboratory/emergent-traits":
            min_confidence = get_bounded_float_param(query_params, 'min_confidence', 0.5, min_val=0.0, max_val=1.0)
            limit = get_clamped_int_param(query_params, 'limit', 20, min_val=1, max_val=100)
            return self._get_emergent_traits(min_confidence, limit)
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/laboratory/cross-pollinations/suggest":
            return self._suggest_cross_pollinations(handler)
        return None

    @handle_errors("emergent traits")
    def _get_emergent_traits(self, min_confidence: float, limit: int) -> HandlerResult:
        """Get emergent traits detected from agent performance patterns.

        Query params:
            min_confidence: Minimum confidence threshold (0.0-1.0, default: 0.5)
            limit: Maximum traits to return (default: 20, max: 100)

        Returns:
            List of emergent traits with agent, domain, confidence, and evidence.
        """
        if not LABORATORY_AVAILABLE or not PersonaLaboratory:
            return error_response("Persona laboratory not available", 503)

        nomic_dir = self.get_nomic_dir()
        persona_manager = self.ctx.get("persona_manager")

        lab = PersonaLaboratory(
            db_path=str(nomic_dir / "laboratory.db") if nomic_dir else None,
            persona_manager=persona_manager,
        )
        traits = lab.detect_emergent_traits()

        # Filter by confidence and limit
        filtered = [t for t in traits if t.confidence >= min_confidence][:limit]

        return json_response({
            "emergent_traits": [
                {
                    "agent": t.agent_name,
                    "trait": t.trait_name,
                    "domain": t.domain,
                    "confidence": t.confidence,
                    "evidence": t.evidence,
                    "detected_at": t.detected_at,
                }
                for t in filtered
            ],
            "count": len(filtered),
            "min_confidence": min_confidence,
        })

    @handle_errors("cross pollinations")
    def _suggest_cross_pollinations(self, handler) -> HandlerResult:
        """Suggest beneficial trait transfers for a target agent.

        POST body:
            target_agent: The agent to suggest traits for (required)

        Returns:
            List of suggested trait transfers with source, trait, and reason.
        """
        if not LABORATORY_AVAILABLE or not PersonaLaboratory:
            return error_response("Persona laboratory not available", 503)

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body or body too large", 400)

        target_agent = body.get('target_agent')
        if not target_agent:
            return error_response("target_agent required", 400)

        nomic_dir = self.get_nomic_dir()
        persona_manager = self.ctx.get("persona_manager")

        lab = PersonaLaboratory(
            db_path=str(nomic_dir / "laboratory.db") if nomic_dir else None,
            persona_manager=persona_manager,
        )
        suggestions = lab.suggest_cross_pollinations(target_agent)

        return json_response({
            "target_agent": target_agent,
            "suggestions": [
                {
                    "source_agent": s[0],
                    "trait_or_domain": s[1],
                    "reason": s[2],
                }
                for s in suggestions
            ],
            "count": len(suggestions),
        })
