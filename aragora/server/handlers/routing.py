"""
Agent routing and team selection endpoint handlers.

Endpoints:
- GET /api/routing/best-teams - Get best-performing team combinations
- POST /api/routing/recommendations - Get agent recommendations for a task
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
    get_int_param,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_routing_imports, ROUTING_AVAILABLE = try_import(
    "aragora.routing.selection",
    "AgentSelector", "TaskRequirements"
)
AgentSelector = _routing_imports.get("AgentSelector")
TaskRequirements = _routing_imports.get("TaskRequirements")


class RoutingHandler(BaseHandler):
    """Handler for agent routing endpoints."""

    ROUTES = [
        "/api/routing/best-teams",
        "/api/routing/recommendations",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/routing/best-teams":
            min_debates = get_int_param(query_params, 'min_debates', 3)
            min_debates = min(max(min_debates, 1), 20)
            limit = get_int_param(query_params, 'limit', 10)
            limit = min(max(limit, 1), 50)
            return self._get_best_team_combinations(min_debates, limit)
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/routing/recommendations":
            return self._get_recommendations(handler)
        return None

    @handle_errors("best team combinations")
    def _get_best_team_combinations(self, min_debates: int, limit: int) -> HandlerResult:
        """Get best-performing team combinations from history."""
        if not ROUTING_AVAILABLE or not AgentSelector:
            return error_response("Agent selector not available", 503)

        elo_system = self.get_elo_system()
        persona_manager = self.ctx.get("persona_manager")

        selector = AgentSelector(
            elo_system=elo_system,
            persona_manager=persona_manager,
        )
        combinations = selector.get_best_team_combinations(min_debates=min_debates)[:limit]

        return json_response({
            "min_debates": min_debates,
            "combinations": combinations,
            "count": len(combinations),
        })

    @handle_errors("routing recommendations")
    def _get_recommendations(self, handler) -> HandlerResult:
        """Get agent recommendations for a task.

        POST body:
            primary_domain: Primary domain for the task (default: 'general')
            secondary_domains: List of secondary domains
            required_traits: List of required agent traits
            task_id: Optional task identifier
            limit: Maximum recommendations to return (default: 5, max: 20)

        Returns:
            Ranked list of agent recommendations with scores.
        """
        if not ROUTING_AVAILABLE or not AgentSelector or not TaskRequirements:
            return error_response("Agent selector not available", 503)

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body or body too large", 400)

        primary_domain = body.get('primary_domain', 'general')
        secondary_domains = body.get('secondary_domains', [])
        required_traits = body.get('required_traits', [])
        limit = min(body.get('limit', 5), 20)
        task_id = body.get('task_id', 'ad-hoc')

        requirements = TaskRequirements(
            task_id=task_id,
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            required_traits=required_traits,
        )

        elo_system = self.get_elo_system()
        persona_manager = self.ctx.get("persona_manager")

        selector = AgentSelector(
            elo_system=elo_system,
            persona_manager=persona_manager,
        )
        recommendations = selector.get_recommendations(requirements, limit=limit)

        return json_response({
            "task_id": task_id,
            "primary_domain": primary_domain,
            "recommendations": recommendations,
            "count": len(recommendations),
        })
