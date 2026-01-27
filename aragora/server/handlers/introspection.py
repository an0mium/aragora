"""
Introspection endpoint handlers.

Provides agent self-awareness and introspection capabilities.

Endpoints:
- GET /api/introspection/all - Get introspection for all agents
- GET /api/introspection/leaderboard - Get agents ranked by reputation
- GET /api/introspection/agents - List available agents
- GET /api/introspection/agents/{name} - Get introspection for specific agent
"""

from __future__ import annotations

__all__ = [
    "IntrospectionHandler",
]

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.server.versioning.compat import strip_version_prefix
from aragora.utils.optional_imports import try_import_class

from .base import (
    SAFE_AGENT_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    ttl_cache,
)
from aragora.rbac.decorators import require_permission
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for introspection endpoints (20 requests per minute)
_introspection_limiter = RateLimiter(requests_per_minute=20)

# Lazy imports for optional dependencies using centralized utility
get_agent_introspection, INTROSPECTION_AVAILABLE = try_import_class(
    "aragora.introspection", "get_agent_introspection"
)
CritiqueStore, CRITIQUE_STORE_AVAILABLE = try_import_class("aragora.memory.store", "CritiqueStore")
PersonaManager, PERSONA_MANAGER_AVAILABLE = try_import_class(
    "aragora.agents.personas", "PersonaManager"
)


class IntrospectionHandler(BaseHandler):
    """Handler for introspection-related endpoints."""

    ROUTES = [
        "/api/introspection/all",
        "/api/introspection/leaderboard",
        "/api/introspection/agents",
        "/api/introspection/agents/availability",
        "/api/introspection/agents/*",
    ]

    DEFAULT_AGENTS = ["gemini", "claude", "codex", "grok", "deepseek"]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)
        if path in (
            "/api/introspection/all",
            "/api/introspection/leaderboard",
            "/api/introspection/agents",
            "/api/introspection/agents/availability",
        ):
            return True
        if path.startswith("/api/introspection/agents/"):
            return True
        return False

    @require_permission("introspection:read")
    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route introspection requests to appropriate methods."""
        path = strip_version_prefix(path)
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _introspection_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for introspection endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/introspection/all":
            return self._get_all_introspection()
        elif path == "/api/introspection/leaderboard":
            limit = get_int_param(query_params, "limit", 10)
            return self._get_introspection_leaderboard(min(limit, 50))
        elif path == "/api/introspection/agents":
            return self._list_agents()
        elif path == "/api/introspection/agents/availability":
            return self._get_agent_availability()
        elif path.startswith("/api/introspection/agents/"):
            # Path: /api/introspection/agents/{name}
            # After strip().split("/") = ["api", "introspection", "agents", "{name}"]
            # Agent name is at index 4
            agent, err = self.extract_path_param(path, 4, "agent", SAFE_AGENT_PATTERN)
            if err:
                return err
            return self._get_agent_introspection(agent)
        return None

    def _get_critique_store(self) -> Optional[object]:
        """Get or create a CritiqueStore instance."""
        if not CRITIQUE_STORE_AVAILABLE or not CritiqueStore:
            return None
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return None
        db_path = nomic_dir / "debates.db"
        if not db_path.exists():
            return None
        return CritiqueStore(str(db_path))

    def _get_persona_manager(self) -> Optional[object]:
        """Get or create a PersonaManager instance."""
        if not PERSONA_MANAGER_AVAILABLE or not PersonaManager:
            return None
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return None
        persona_db = get_db_path(DatabaseType.PERSONAS, nomic_dir=nomic_dir)
        if not persona_db.exists():
            return None
        return PersonaManager(persona_db)

    def _get_known_agents(self, store: Any) -> list[str]:
        """Get list of known agents from critique store or defaults."""
        if store:
            try:
                reputations = store.get_all_reputations()
                agents = [r.agent_name for r in reputations]
                if agents:
                    return agents
            except Exception as e:
                logger.debug(f"Could not fetch agent reputations: {e}")
        return self.DEFAULT_AGENTS

    @ttl_cache(ttl_seconds=60, key_prefix="lb_introspection_agents")
    def _list_agents(self) -> HandlerResult:
        """List available agents for introspection.

        Returns a lightweight list of agent names with basic metadata.
        """
        try:
            memory = self._get_critique_store()
            agents = self._get_known_agents(memory)

            agent_list = []
            for agent in agents:
                agent_info: dict[str, Any] = {"name": agent}

                # Add reputation if available
                if memory and hasattr(memory, "get_agent_reputation"):
                    try:
                        reputation = memory.get_agent_reputation(agent)  # type: ignore[union-attr]
                        if reputation:
                            agent_info["reputation_score"] = getattr(reputation, "score", 0.5)
                            agent_info["total_critiques"] = getattr(
                                reputation, "total_critiques", 0
                            )
                    except Exception as e:
                        logger.debug(f"Failed to get reputation for {agent}: {e}")

                agent_list.append(agent_info)

            # Sort by reputation score descending
            agent_list.sort(key=lambda x: x.get("reputation_score", 0), reverse=True)

            return json_response(
                {
                    "agents": agent_list,
                    "count": len(agent_list),
                }
            )
        except Exception as e:
            logger.error(f"Error listing agents: {e}", exc_info=True)
            return error_response("Failed to list agents", 500)

    def _get_agent_introspection(self, agent: str) -> HandlerResult:
        """Get introspection data for a specific agent."""
        if not INTROSPECTION_AVAILABLE or not get_agent_introspection:
            return error_response("Introspection module not available", 503)

        try:
            memory = self._get_critique_store()
            persona_manager = self._get_persona_manager()
            snapshot = get_agent_introspection(
                agent, memory=memory, persona_manager=persona_manager
            )
            if snapshot is None:
                return error_response(f"Agent '{agent}' not found", 404)
            return json_response(snapshot.to_dict())
        except Exception as e:
            logger.error(f"Error getting introspection for {agent}: {e}", exc_info=True)
            return error_response("Failed to get introspection", 500)

    def _get_agent_availability(self) -> HandlerResult:
        """Return credential availability for known agent types."""
        try:
            from aragora.agents.credential_validator import get_agent_credential_status
        except ImportError:
            return error_response("Credential validator not available", 503)

        try:
            statuses = get_agent_credential_status()
            available = []
            missing = []
            details: dict[str, Any] = {}

            for agent_type, status in statuses.items():
                details[agent_type] = {
                    "available": status.is_available,
                    "required_vars": status.required_vars,
                    "missing_vars": status.missing_vars,
                    "available_via": status.available_via,
                }
                if status.is_available:
                    available.append(agent_type)
                else:
                    missing.append(agent_type)

            return json_response(
                {
                    "available": sorted(available),
                    "missing": sorted(missing),
                    "details": details,
                }
            )
        except Exception as e:
            logger.error(f"Error getting agent availability: {e}", exc_info=True)
            return error_response("Failed to determine agent availability", 500)

    @ttl_cache(ttl_seconds=120, key_prefix="lb_introspection_all")
    def _get_all_introspection(self) -> HandlerResult:
        """Get introspection data for all known agents.

        Cached for 2 minutes since this is an expensive operation.
        """
        if not INTROSPECTION_AVAILABLE or not get_agent_introspection:
            return error_response("Introspection module not available", 503)

        try:
            memory = self._get_critique_store()
            persona_manager = self._get_persona_manager()
            agents = self._get_known_agents(memory)

            snapshots = {}
            for agent in agents:
                try:
                    snapshot = get_agent_introspection(
                        agent, memory=memory, persona_manager=persona_manager
                    )
                    snapshots[agent] = snapshot.to_dict()
                except Exception as e:
                    logger.debug(f"Error getting introspection for {agent}: {e}")
                    continue

            return json_response(
                {
                    "agents": snapshots,
                    "count": len(snapshots),
                }
            )
        except Exception as e:
            logger.error(f"Error getting all introspection: {e}", exc_info=True)
            return error_response("Failed to get introspection data", 500)

    @ttl_cache(ttl_seconds=120, key_prefix="lb_introspection_lb")
    def _get_introspection_leaderboard(self, limit: int) -> HandlerResult:
        """Get agents ranked by reputation score.

        Cached for 2 minutes since this is an expensive operation.
        """
        if not INTROSPECTION_AVAILABLE or not get_agent_introspection:
            return error_response("Introspection module not available", 503)

        try:
            memory = self._get_critique_store()
            persona_manager = self._get_persona_manager()
            agents = self._get_known_agents(memory)

            snapshots = []
            for agent in agents:
                try:
                    snapshot = get_agent_introspection(
                        agent, memory=memory, persona_manager=persona_manager
                    )
                    snapshots.append(snapshot.to_dict())
                except Exception as e:
                    logger.debug(f"Error getting introspection for {agent}: {e}")
                    continue

            # Sort by reputation score descending
            snapshots.sort(key=lambda x: x.get("reputation_score", 0), reverse=True)

            return json_response(
                {
                    "leaderboard": snapshots[:limit],
                    "total_agents": len(snapshots),
                }
            )
        except Exception as e:
            logger.error(f"Error getting introspection leaderboard: {e}", exc_info=True)
            return error_response("Failed to get leaderboard", 500)
