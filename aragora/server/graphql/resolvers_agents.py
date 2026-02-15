"""
GraphQL Resolvers for Agent operations.

Contains query and mutation resolvers for agents,
plus transform functions for agent data.

Separated from resolvers.py for maintainability.
"""

from __future__ import annotations

import logging
from typing import Any

from .resolvers import (
    ResolverContext,
    ResolverResult,
    _normalize_agent_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Transform Functions
# =============================================================================


def _transform_agent(agent: Any, agent_id: str | None = None) -> dict[str, Any]:
    """Transform internal agent format to GraphQL format."""
    if isinstance(agent, dict):
        name = agent.get("name") or agent.get("agent_name") or agent_id or "unknown"
        return {
            "id": name,
            "name": name,
            "status": _normalize_agent_status(agent.get("status")),
            "capabilities": agent.get("capabilities", []),
            "region": agent.get("region"),
            "currentTask": None,
            "stats": {
                "totalGames": agent.get("games", 0) + agent.get("matches", 0),
                "wins": agent.get("wins", 0),
                "losses": agent.get("losses", 0),
                "draws": agent.get("draws", 0),
                "winRate": agent.get("win_rate", 0.0),
                "elo": agent.get("elo", 1500),
                "calibrationAccuracy": agent.get("calibration_accuracy"),
                "consistencyScore": agent.get("consistency"),
            },
            "elo": agent.get("elo", 1500),
            "model": agent.get("model"),
            "provider": agent.get("provider"),
        }

    # Handle object-based agent (e.g., AgentRating)
    name = (
        getattr(agent, "name", None) or getattr(agent, "agent_name", None) or agent_id or "unknown"
    )
    wins = getattr(agent, "wins", 0)
    losses = getattr(agent, "losses", 0)
    draws = getattr(agent, "draws", 0)
    total_games = wins + losses + draws

    return {
        "id": name,
        "name": name,
        "status": "AVAILABLE",
        "capabilities": getattr(agent, "capabilities", []),
        "region": getattr(agent, "region", None),
        "currentTask": None,
        "stats": {
            "totalGames": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winRate": getattr(agent, "win_rate", wins / total_games if total_games > 0 else 0.0),
            "elo": getattr(agent, "elo", 1500),
            "calibrationAccuracy": getattr(agent, "calibration_accuracy", None),
            "consistencyScore": getattr(agent, "consistency", None),
        },
        "elo": getattr(agent, "elo", 1500),
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
    }


# =============================================================================
# Agent Query Resolvers
# =============================================================================


class AgentQueryResolvers:
    """Query resolvers for agent operations."""

    @staticmethod
    async def resolve_agent(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Resolve a single agent by ID.

        Args:
            ctx: Resolver context
            id: Agent ID/name

        Returns:
            ResolverResult with agent data
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            rating = elo_system.get_rating(id)
            if not rating:
                return ResolverResult(errors=[f"Agent not found: {id}"])

            data = _transform_agent(rating, id)
            return ResolverResult(data=data)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.warning("GraphQL resolver error in resolve_agent: %s", e)
            return ResolverResult(errors=["Failed to resolve agent: internal error"])

    @staticmethod
    async def resolve_agents(
        ctx: ResolverContext,
        status: str | None = None,
        capability: str | None = None,
        region: str | None = None,
    ) -> ResolverResult:
        """Resolve a list of agents with optional filtering.

        Args:
            ctx: Resolver context
            status: Optional status filter
            capability: Optional capability filter
            region: Optional region filter

        Returns:
            ResolverResult with list of agents
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            # Get all agents from leaderboard
            rankings = elo_system.get_leaderboard(limit=500)

            agents = []
            for agent in rankings:
                agent_data = _transform_agent(agent)

                # Apply filters
                if status and agent_data.get("status") != status:
                    continue
                if capability:
                    caps = agent_data.get("capabilities", [])
                    if capability not in caps:
                        continue
                if region and agent_data.get("region") != region:
                    continue

                agents.append(agent_data)

            return ResolverResult(data=agents)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.warning("GraphQL resolver error in resolve_agents: %s", e)
            return ResolverResult(errors=["Failed to resolve agents: internal error"])

    @staticmethod
    async def resolve_leaderboard(
        ctx: ResolverContext,
        limit: int = 20,
        domain: str | None = None,
    ) -> ResolverResult:
        """Get agent leaderboard.

        Args:
            ctx: Resolver context
            limit: Maximum results
            domain: Optional domain filter

        Returns:
            ResolverResult with list of agents sorted by ELO
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            # Get leaderboard
            if hasattr(elo_system, "get_cached_leaderboard") and domain is None:
                rankings = elo_system.get_cached_leaderboard(limit=min(limit, 50))
            else:
                rankings = elo_system.get_leaderboard(limit=min(limit, 50), domain=domain)

            agents = [_transform_agent(agent) for agent in rankings]
            return ResolverResult(data=agents)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.warning("GraphQL resolver error in resolve_leaderboard: %s", e)
            return ResolverResult(errors=["Failed to resolve leaderboard: internal error"])


# =============================================================================
# Agent Mutation Resolvers
# =============================================================================


class AgentMutationResolvers:
    """Mutation resolvers for agent operations."""

    @staticmethod
    async def resolve_register_agent(
        ctx: ResolverContext,
        input: dict[str, Any],
    ) -> ResolverResult:
        """Register a new agent with the control plane.

        Args:
            ctx: Resolver context
            input: RegisterAgentInput fields

        Returns:
            ResolverResult with registered agent data
        """
        try:
            agent_id = input.get("agentId")
            if not agent_id:
                return ResolverResult(errors=["Agent ID is required"])

            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            _agent = await coordinator.register_agent(
                agent_id=agent_id,
                capabilities=input.get("capabilities", []),
                model=input.get("model", "unknown"),
                provider=input.get("provider", "unknown"),
                metadata=input.get("metadata", {}),
            )

            return ResolverResult(
                data={
                    "id": agent_id,
                    "name": agent_id,
                    "status": "AVAILABLE",
                    "capabilities": input.get("capabilities", []),
                    "region": input.get("metadata", {}).get("region"),
                    "currentTask": None,
                    "stats": {
                        "totalGames": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "winRate": 0.0,
                        "elo": 1500,
                        "calibrationAccuracy": None,
                        "consistencyScore": None,
                    },
                    "elo": 1500,
                    "model": input.get("model"),
                    "provider": input.get("provider"),
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # Agent registration or data transformation errors
            logger.warning("GraphQL resolver error in resolve_register_agent: %s", e)
            return ResolverResult(errors=["Failed to register agent: internal error"])

    @staticmethod
    async def resolve_unregister_agent(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Unregister an agent from the control plane.

        Args:
            ctx: Resolver context
            id: Agent ID

        Returns:
            ResolverResult with boolean success
        """
        try:
            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            success = await coordinator.unregister_agent(id)
            if not success:
                return ResolverResult(errors=[f"Agent not found: {id}"])

            return ResolverResult(data=True)

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # Agent lookup or unregistration errors
            logger.warning("GraphQL resolver error in resolve_unregister_agent: %s", e)
            return ResolverResult(errors=["Failed to unregister agent: internal error"])
