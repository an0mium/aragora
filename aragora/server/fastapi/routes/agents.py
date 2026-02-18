"""
Agent Endpoints (FastAPI v2).

Provides async agent management endpoints:
- List available agents
- Get agent details by ID
- Get agent leaderboard
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Agents"])


# =============================================================================
# Pydantic Models
# =============================================================================


class AgentSummary(BaseModel):
    """Summary of an agent for list views."""

    name: str
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    available: bool = True

    model_config = {"extra": "allow"}


class AgentListResponse(BaseModel):
    """Response for agent listing."""

    agents: list[AgentSummary]
    total: int


class AgentDetail(BaseModel):
    """Full agent details."""

    name: str
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    available: bool = True
    type: str = ""
    requires_api_key: bool = False
    api_key_configured: bool = False
    uses_openrouter_fallback: bool = False
    fallback_model: str | None = None

    model_config = {"extra": "allow"}


class LeaderboardEntry(BaseModel):
    """Entry in the agent leaderboard."""

    rank: int
    name: str
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    model_config = {"extra": "allow"}


class LeaderboardResponse(BaseModel):
    """Response for agent leaderboard."""

    leaderboard: list[LeaderboardEntry]
    total: int
    domain: str | None = None


# =============================================================================
# Dependencies
# =============================================================================


async def get_elo_system(request: Request):
    """Dependency to get the ELO system from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx and "elo_system" in ctx:
        return ctx.get("elo_system")  # Return None if explicitly set to None

    # Fall back to global ELO system
    try:
        from aragora.ranking.elo import EloSystem

        return EloSystem()
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"ELO system not available: {e}")
        return None


# =============================================================================
# Helpers
# =============================================================================


def _agent_to_dict(agent: Any) -> dict[str, Any]:
    """Convert an agent object to a dictionary."""
    if isinstance(agent, dict):
        return agent
    if hasattr(agent, "__dict__"):
        return {
            "name": getattr(agent, "name", getattr(agent, "agent_name", "")),
            "elo": getattr(agent, "elo", getattr(agent, "rating", 1500.0)),
            "matches": getattr(agent, "matches", getattr(agent, "total_matches", 0)),
            "wins": getattr(agent, "wins", 0),
            "losses": getattr(agent, "losses", 0),
        }
    return {"name": str(agent)}


def _get_known_agents() -> list[str]:
    """Get list of known agent types from config."""
    try:
        from aragora.config import ALLOWED_AGENT_TYPES

        return list(ALLOWED_AGENT_TYPES)
    except (ImportError, AttributeError):
        return ["claude", "codex", "gemini", "grok", "deepseek"]


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    request: Request,
    include_stats: bool = Query(False, description="Include ELO stats"),
    elo=Depends(get_elo_system),
) -> AgentListResponse:
    """
    List all available agents.

    Returns a list of all known agents with optional statistics.
    """
    try:
        agents: list[AgentSummary] = []

        # Get agents from ELO system if available
        if elo:
            try:
                rankings = elo.get_leaderboard(limit=500)
                for agent in rankings:
                    agent_dict = _agent_to_dict(agent)
                    name = agent_dict.get("name", "")
                    if include_stats:
                        agents.append(
                            AgentSummary(
                                name=name,
                                elo=agent_dict.get("elo", 1500.0),
                                matches=agent_dict.get("matches", 0),
                                wins=agent_dict.get("wins", 0),
                                losses=agent_dict.get("losses", 0),
                            )
                        )
                    else:
                        agents.append(AgentSummary(name=name))
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Could not get agents from ELO: {e}")

        # Fallback to known agent types if no ELO data
        if not agents:
            agent_names = _get_known_agents()
            agents = [AgentSummary(name=name) for name in agent_names]

        return AgentListResponse(
            agents=agents,
            total=len(agents),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.get("/agents/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
    domain: str | None = Query(None, description="Filter by domain"),
    elo=Depends(get_elo_system),
) -> LeaderboardResponse:
    """
    Get agent leaderboard.

    Returns agents ranked by ELO rating with win/loss statistics.
    """
    try:
        leaderboard: list[LeaderboardEntry] = []

        if elo:
            try:
                if domain:
                    rankings = elo.get_leaderboard(limit=limit, domain=domain)
                else:
                    rankings = elo.get_leaderboard(limit=limit)

                for rank, agent in enumerate(rankings[:limit], 1):
                    agent_dict = _agent_to_dict(agent)
                    matches = agent_dict.get("matches", 0)
                    wins = agent_dict.get("wins", 0)
                    win_rate = wins / matches if matches > 0 else 0.0

                    leaderboard.append(
                        LeaderboardEntry(
                            rank=rank,
                            name=agent_dict.get("name", ""),
                            elo=agent_dict.get("elo", 1500.0),
                            matches=matches,
                            wins=wins,
                            losses=agent_dict.get("losses", 0),
                            win_rate=round(win_rate, 3),
                        )
                    )
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Could not get leaderboard from ELO: {e}")

        return LeaderboardResponse(
            leaderboard=leaderboard,
            total=len(leaderboard),
            domain=domain,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get leaderboard")


@router.get("/agents/{agent_id}", response_model=AgentDetail)
async def get_agent(
    agent_id: str,
    elo=Depends(get_elo_system),
) -> AgentDetail:
    """
    Get agent details by ID.

    Returns detailed information about a specific agent including ELO rating and availability.
    """
    try:
        agent_data: dict[str, Any] | None = None

        # Try to get agent from ELO system
        if elo:
            try:
                # Try direct lookup first
                if hasattr(elo, "get_agent"):
                    agent_obj = elo.get_agent(agent_id)
                    if agent_obj:
                        agent_data = _agent_to_dict(agent_obj)

                # Fall back to searching the leaderboard
                if agent_data is None and hasattr(elo, "get_leaderboard"):
                    rankings = elo.get_leaderboard(limit=500)
                    for agent in rankings:
                        d = _agent_to_dict(agent)
                        if d.get("name") == agent_id:
                            agent_data = d
                            break
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Could not get agent {agent_id} from ELO: {e}")

        # Check if the agent ID is in known agent types
        known_agents = _get_known_agents()
        if agent_data is None and agent_id in known_agents:
            agent_data = {"name": agent_id}

        if agent_data is None:
            raise NotFoundError(f"Agent {agent_id} not found")

        # Get agent registry info if available
        agent_type_info: dict[str, Any] = {}
        try:
            from aragora.agents.registry import AgentRegistry, register_all_agents

            register_all_agents()
            all_agents = AgentRegistry.list_all()
            if agent_id in all_agents:
                agent_type_info = all_agents[agent_id]
        except (ImportError, RuntimeError, AttributeError):
            pass

        return AgentDetail(
            name=agent_data.get("name", agent_id),
            elo=agent_data.get("elo", 1500.0),
            matches=agent_data.get("matches", 0),
            wins=agent_data.get("wins", 0),
            losses=agent_data.get("losses", 0),
            available=True,
            type=agent_type_info.get("type", ""),
            requires_api_key=bool(agent_type_info.get("env_vars")),
            api_key_configured=False,  # Don't expose key status via API
            uses_openrouter_fallback=False,
            fallback_model=None,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent")
