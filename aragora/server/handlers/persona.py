"""
Persona-related endpoint handlers.

Endpoints:
- GET /api/personas - Get all agent personas
- GET /api/agent/{name}/persona - Get agent persona
- GET /api/agent/{name}/grounded-persona - Get truth-grounded persona
- GET /api/agent/{name}/identity-prompt - Get identity prompt
- GET /api/agent/{name}/performance - Get agent performance summary
- GET /api/agent/{name}/domains - Get agent expertise domains
- GET /api/agent/{name}/accuracy - Get position accuracy stats
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_string_param,
    SAFE_AGENT_PATTERN,
)
from aragora.utils.optional_imports import try_import_class

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies using centralized utility
PersonaSynthesizer, GROUNDED_AVAILABLE = try_import_class(
    "aragora.agents.grounded", "PersonaSynthesizer"
)
PositionTracker, POSITION_TRACKER_AVAILABLE = try_import_class(
    "aragora.agents.truth_grounding", "PositionTracker"
)


class PersonaHandler(BaseHandler):
    """Handler for persona-related endpoints."""

    ROUTES = [
        "/api/personas",
        "/api/agent/*/persona",
        "/api/agent/*/grounded-persona",
        "/api/agent/*/identity-prompt",
        "/api/agent/*/performance",
        "/api/agent/*/domains",
        "/api/agent/*/accuracy",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/personas":
            return True
        if path.startswith("/api/agent/") and any(
            path.endswith(suffix) for suffix in (
                "/persona",
                "/grounded-persona",
                "/identity-prompt",
                "/performance",
                "/domains",
                "/accuracy",
            )
        ):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route persona requests to appropriate methods."""
        # List all personas
        if path == "/api/personas":
            return self._get_all_personas()

        # Agent-specific endpoints
        if path.startswith("/api/agent/"):
            # Extract agent name from path: /api/agent/{name}/endpoint
            agent, err = self.extract_path_param(path, 2, "agent", SAFE_AGENT_PATTERN)
            if err:
                return err

            if path.endswith("/persona"):
                return self._get_agent_persona(agent)
            elif path.endswith("/grounded-persona"):
                return self._get_grounded_persona(agent)
            elif path.endswith("/identity-prompt"):
                sections = get_string_param(query_params, 'sections')
                return self._get_identity_prompt(agent, sections)
            elif path.endswith("/performance"):
                return self._get_agent_performance(agent)
            elif path.endswith("/domains"):
                limit = get_int_param(query_params, 'limit', 10)
                return self._get_agent_domains(agent, limit)
            elif path.endswith("/accuracy"):
                return self._get_agent_accuracy(agent)

        return None

    def get_persona_manager(self):
        """Get persona manager instance."""
        return self.ctx.get("persona_manager")

    def get_position_ledger(self):
        """Get position ledger instance."""
        return self.ctx.get("position_ledger")

    def _get_all_personas(self) -> HandlerResult:
        """Get all agent personas."""
        persona_manager = self.get_persona_manager()
        if not persona_manager:
            return json_response({
                "error": "Persona management not configured",
                "personas": []
            })

        try:
            personas = persona_manager.get_all_personas()
            return json_response({
                "personas": [
                    {
                        "agent_name": p.agent_name,
                        "description": p.description,
                        "traits": p.traits,
                        "expertise": p.expertise,
                        "created_at": p.created_at,
                        "updated_at": p.updated_at,
                    }
                    for p in personas
                ],
                "count": len(personas),
            })
        except Exception as e:
            logger.error(f"Error getting personas: {e}", exc_info=True)
            return json_response({"error": "Failed to get personas", "personas": []})

    def _get_agent_persona(self, agent: str) -> HandlerResult:
        """Get persona for a specific agent."""
        persona_manager = self.get_persona_manager()
        if not persona_manager:
            return error_response("Persona management not configured", 503)

        try:
            persona = persona_manager.get_persona(agent)
            if persona:
                return json_response({
                    "persona": {
                        "agent_name": persona.agent_name,
                        "description": persona.description,
                        "traits": persona.traits,
                        "expertise": persona.expertise,
                        "created_at": persona.created_at,
                        "updated_at": persona.updated_at,
                    }
                })
            else:
                return json_response({
                    "error": f"No persona found for agent '{agent}'",
                    "persona": None
                })
        except Exception as e:
            logger.error(f"Error getting persona for {agent}: {e}", exc_info=True)
            return error_response("Failed to get persona", 500)

    def _get_agent_performance(self, agent: str) -> HandlerResult:
        """Get performance summary for an agent."""
        persona_manager = self.get_persona_manager()
        if not persona_manager:
            return error_response("Persona management not configured", 503)

        try:
            summary = persona_manager.get_performance_summary(agent)
            return json_response({
                "agent": agent,
                "performance": summary,
            })
        except Exception as e:
            logger.error(f"Error getting performance for {agent}: {e}", exc_info=True)
            return error_response("Failed to get performance", 500)

    def _get_agent_domains(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's best expertise domains by calibration."""
        elo_system = self.get_elo_system()
        if not elo_system:
            return error_response("Ranking system not available", 503)

        try:
            domains = elo_system.get_best_domains(agent, limit=limit)
            return json_response({
                "agent": agent,
                "domains": [
                    {"domain": d[0], "calibration_score": d[1]}
                    for d in domains
                ],
                "count": len(domains),
            })
        except Exception as e:
            logger.error(f"Error getting domains for {agent}: {e}", exc_info=True)
            return error_response("Failed to get domains", 500)

    def _get_grounded_persona(self, agent: str) -> HandlerResult:
        """Get truth-grounded persona synthesized from performance data."""
        if not GROUNDED_AVAILABLE or not PersonaSynthesizer:
            return error_response("Grounded personas module not available", 503)

        try:
            synthesizer = PersonaSynthesizer(
                persona_manager=self.get_persona_manager(),
                elo_system=self.get_elo_system(),
                position_ledger=self.get_position_ledger(),
                relationship_tracker=None,
            )
            persona = synthesizer.get_grounded_persona(agent)
            if persona:
                return json_response({
                    "agent": agent,
                    "elo": persona.elo,
                    "domain_elos": persona.domain_elos,
                    "games_played": persona.games_played,
                    "win_rate": persona.win_rate,
                    "calibration_score": persona.calibration_score,
                    "position_accuracy": persona.position_accuracy,
                    "positions_taken": persona.positions_taken,
                    "reversals": persona.reversals,
                })
            else:
                return json_response({
                    "agent": agent,
                    "message": "No grounded persona data"
                })
        except Exception as e:
            logger.error(f"Error getting grounded persona for {agent}: {e}", exc_info=True)
            return error_response("Failed to get grounded persona", 500)

    def _get_identity_prompt(self, agent: str, sections: str | None = None) -> HandlerResult:
        """Get evidence-grounded identity prompt for agent initialization."""
        if not GROUNDED_AVAILABLE or not PersonaSynthesizer:
            return error_response("Grounded personas module not available", 503)

        try:
            synthesizer = PersonaSynthesizer(
                persona_manager=self.get_persona_manager(),
                elo_system=self.get_elo_system(),
                position_ledger=self.get_position_ledger(),
                relationship_tracker=None,
            )
            include_sections = sections.split(',') if sections else None
            prompt = synthesizer.synthesize_identity_prompt(agent, include_sections=include_sections)
            return json_response({
                "agent": agent,
                "identity_prompt": prompt,
                "sections": include_sections,
            })
        except Exception as e:
            logger.error(f"Error getting identity prompt for {agent}: {e}", exc_info=True)
            return error_response("Failed to get identity prompt", 500)

    def _get_agent_accuracy(self, agent: str) -> HandlerResult:
        """Get position accuracy stats for an agent from PositionTracker."""
        if not POSITION_TRACKER_AVAILABLE or not PositionTracker:
            return error_response("PositionTracker module not available", 503)

        try:
            # Try to get position tracker from context or create one
            nomic_dir = self.get_nomic_dir()
            if not nomic_dir:
                return error_response("Position tracking not configured", 503)

            db_path = nomic_dir / "aragora_positions.db"
            if not db_path.exists():
                return json_response({
                    "agent": agent,
                    "total_positions": 0,
                    "verified_positions": 0,
                    "accuracy_rate": 0.0,
                    "message": "No position accuracy data available",
                })

            tracker = PositionTracker(db_path=str(db_path))
            accuracy = tracker.get_agent_position_accuracy(agent)

            if accuracy:
                return json_response({
                    "agent": agent,
                    "total_positions": accuracy.get("total_positions", 0),
                    "verified_positions": accuracy.get("verified_positions", 0),
                    "correct_positions": accuracy.get("correct_positions", 0),
                    "accuracy_rate": accuracy.get("accuracy_rate", 0.0),
                    "by_type": accuracy.get("by_type", {}),
                })
            else:
                return json_response({
                    "agent": agent,
                    "total_positions": 0,
                    "verified_positions": 0,
                    "accuracy_rate": 0.0,
                    "message": "No position accuracy data available",
                })
        except Exception as e:
            logger.error(f"Error getting accuracy for {agent}: {e}", exc_info=True)
            return error_response("Failed to get accuracy", 500)
