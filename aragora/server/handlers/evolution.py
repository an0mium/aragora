"""
Prompt evolution endpoint handlers.

Endpoints:
- GET /api/evolution/{agent}/history - Get prompt evolution history for an agent
"""

import logging
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    SAFE_AGENT_PATTERN,
)
from aragora.persistence.db_config import DatabaseType, get_db_path

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
EVOLUTION_AVAILABLE = False
PromptEvolver = None

try:
    from aragora.evolution.evolver import PromptEvolver as _PE
    PromptEvolver = _PE
    EVOLUTION_AVAILABLE = True
except ImportError:
    pass


class EvolutionHandler(BaseHandler):
    """Handler for prompt evolution endpoints."""

    ROUTES = [
        "/api/evolution/*/history",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/evolution/") and path.endswith("/history")

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route evolution requests to appropriate methods."""
        if not path.startswith("/api/evolution/") or not path.endswith("/history"):
            return None

        # Extract agent name: /api/evolution/{agent}/history
        agent, err = self.extract_path_param(path, 2, "agent", SAFE_AGENT_PATTERN)
        if err:
            return err

        limit = get_int_param(query_params, 'limit', 10)
        limit = min(max(limit, 1), 50)
        return self._get_evolution_history(agent, limit)

    def _get_evolution_history(self, agent: str, limit: int) -> HandlerResult:
        """Get prompt evolution history for an agent."""
        if not EVOLUTION_AVAILABLE or not PromptEvolver:
            return error_response("Prompt evolution not available", 503)

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            evolver = PromptEvolver(db_path=str(get_db_path(DatabaseType.PROMPT_EVOLUTION, nomic_dir)))
            history = evolver.get_evolution_history(agent, limit=limit)

            return json_response({
                "agent": agent,
                "history": history,
                "count": len(history),
            })
        except Exception as e:
            logger.error(f"Error getting evolution history for {agent}: {e}", exc_info=True)
            return error_response("Failed to get evolution history", 500)
