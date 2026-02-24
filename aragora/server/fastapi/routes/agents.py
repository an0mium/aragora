"""
Agent Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/ (aiohttp handler)

Provides async agent management endpoints:
- GET  /api/v2/agents                - List available agents with optional filters
- GET  /api/v2/agents/rankings       - Get ELO rankings
- GET  /api/v2/agents/leaderboard    - Get agent leaderboard
- GET  /api/v2/agents/{agent_id}     - Get agent details by ID
- POST /api/v2/agents                - Register a new agent

Migration Notes:
    This module replaces legacy agent handler endpoints with native FastAPI
    routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
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


class RegisterAgentRequest(BaseModel):
    """Request body for POST /agents."""

    name: str = Field(..., min_length=1, max_length=100, description="Agent name/identifier")
    type: str = Field("custom", description="Agent type (e.g. api, cli, custom)")
    config: dict[str, Any] = Field(default_factory=dict, description="Agent configuration")


class RegisterAgentResponse(BaseModel):
    """Response for POST /agents."""

    success: bool
    agent: AgentDetail


class AgentCapabilitiesResponse(BaseModel):
    """Response for agent capabilities/metadata."""

    name: str
    provider: str = ""
    model_id: str = ""
    context_window: int = 0
    specialties: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    release_date: str | None = None

    model_config = {"extra": "allow"}


class AgentStatsResponse(BaseModel):
    """Response for agent performance stats."""

    name: str
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    domains: list[str] = Field(default_factory=list)
    recent_performance: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class CalibrationBucket(BaseModel):
    """A single calibration bucket."""

    bucket: str = ""
    predicted: float = 0.0
    actual: float = 0.0
    count: int = 0

    model_config = {"extra": "allow"}


class AgentCalibrationResponse(BaseModel):
    """Response for agent calibration scores."""

    name: str
    calibration_score: float = 0.0
    buckets: list[CalibrationBucket] = Field(default_factory=list)
    total_predictions: int = 0

    model_config = {"extra": "allow"}


class DomainInfo(BaseModel):
    """Information about an agent domain/specialization."""

    name: str
    description: str = ""
    agent_count: int = 0

    model_config = {"extra": "allow"}


class DomainsResponse(BaseModel):
    """Response for agent domains listing."""

    domains: list[DomainInfo]
    total: int


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
        logger.warning("ELO system not available: %s", e)
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
                logger.warning("Could not get agents from ELO: %s", e)

        # Fallback to known agent types if no ELO data
        if not agents:
            agent_names = _get_known_agents()
            agents = [AgentSummary(name=name) for name in agent_names]

        return AgentListResponse(
            agents=agents,
            total=len(agents),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing agents: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.get("/agents/rankings", response_model=LeaderboardResponse)
async def get_rankings(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
    domain: str | None = Query(None, description="Filter by domain"),
    elo=Depends(get_elo_system),
) -> LeaderboardResponse:
    """
    Get ELO rankings.

    Alias for /agents/leaderboard. Returns agents ranked by ELO rating.
    """
    return await get_leaderboard(request=request, limit=limit, domain=domain, elo=elo)


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
                logger.warning("Could not get leaderboard from ELO: %s", e)

        return LeaderboardResponse(
            leaderboard=leaderboard,
            total=len(leaderboard),
            domain=domain,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting leaderboard: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get leaderboard")


@router.get("/agents/domains", response_model=DomainsResponse)
async def get_agent_domains(
    request: Request,
    elo=Depends(get_elo_system),
) -> DomainsResponse:
    """List available agent domains and specializations."""
    try:
        domains: list[DomainInfo] = []

        # Try to get domains from ELO system
        if elo and hasattr(elo, "get_domains"):
            try:
                raw_domains = elo.get_domains()
                for d in raw_domains:
                    if isinstance(d, str):
                        domains.append(DomainInfo(name=d))
                    elif isinstance(d, dict):
                        domains.append(DomainInfo(
                            name=d.get("name", d.get("id", "")),
                            description=d.get("description", ""),
                            agent_count=d.get("agent_count", 0),
                        ))
                    else:
                        domains.append(DomainInfo(
                            name=getattr(d, "name", str(d)),
                            description=getattr(d, "description", ""),
                            agent_count=getattr(d, "agent_count", 0),
                        ))
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug("Could not get domains from ELO: %s", e)

        # Fall back to known domain categories
        if not domains:
            default_domains = [
                "security", "architecture", "testing", "performance",
                "compliance", "general",
            ]
            domains = [DomainInfo(name=d) for d in default_domains]

        return DomainsResponse(domains=domains, total=len(domains))

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting agent domains: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get agent domains")


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
                logger.debug("Could not get agent %s from ELO: %s", agent_id, e)

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
        logger.exception("Error getting agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail="Failed to get agent")


@router.post("/agents", response_model=RegisterAgentResponse, status_code=201)
async def register_agent(
    body: RegisterAgentRequest,
    auth: AuthorizationContext = Depends(require_permission("agents:write")),
    elo=Depends(get_elo_system),
) -> RegisterAgentResponse:
    """
    Register a new agent.

    Registers a custom agent in the system with an initial ELO rating.
    Requires `agents:write` permission.
    """
    try:
        # Check if agent already exists
        known_agents = _get_known_agents()
        if body.name in known_agents:
            raise HTTPException(
                status_code=409,
                detail=f"Agent '{body.name}' already exists",
            )

        # Register in ELO system if available
        if elo and hasattr(elo, "register_agent"):
            try:
                elo.register_agent(body.name, config=body.config)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Could not register agent %s in ELO: %s", body.name, e)

        # Register in agent registry if available
        try:
            from aragora.agents.registry import AgentRegistry

            AgentRegistry.register(body.name, agent_type=body.type)
        except (ImportError, RuntimeError, AttributeError, TypeError) as e:
            logger.debug("Could not register agent %s in registry: %s", body.name, e)

        logger.info("Registered agent: %s (type=%s)", body.name, body.type)

        return RegisterAgentResponse(
            success=True,
            agent=AgentDetail(
                name=body.name,
                type=body.type,
                elo=1500.0,
                available=True,
            ),
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error registering agent: %s", e)
        raise HTTPException(status_code=500, detail="Failed to register agent")


# =============================================================================
# New Endpoints (Stats, Calibration)
# =============================================================================


@router.get("/agents/{agent_id}/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    agent_id: str,
    elo=Depends(get_elo_system),
) -> AgentStatsResponse:
    """Get per-agent performance statistics."""
    try:
        agent_data: dict[str, Any] | None = None

        if elo:
            try:
                if hasattr(elo, "get_agent"):
                    agent_obj = elo.get_agent(agent_id)
                    if agent_obj:
                        agent_data = _agent_to_dict(agent_obj)

                if agent_data is None and hasattr(elo, "get_leaderboard"):
                    rankings = elo.get_leaderboard(limit=500)
                    for agent in rankings:
                        d = _agent_to_dict(agent)
                        if d.get("name") == agent_id:
                            agent_data = d
                            break
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug("Could not get agent %s from ELO: %s", agent_id, e)

        # Check known agents as fallback
        if agent_data is None:
            known_agents = _get_known_agents()
            if agent_id in known_agents:
                agent_data = {"name": agent_id}
            else:
                raise NotFoundError(f"Agent {agent_id} not found")

        matches = agent_data.get("matches", 0)
        wins = agent_data.get("wins", 0)
        win_rate = wins / matches if matches > 0 else 0.0

        # Try to get extended stats
        recent_performance: list[dict[str, Any]] = []
        agent_domains: list[str] = []
        avg_confidence = 0.0

        if elo:
            try:
                if hasattr(elo, "get_agent_stats"):
                    ext_stats = elo.get_agent_stats(agent_id)
                    if isinstance(ext_stats, dict):
                        recent_performance = ext_stats.get("recent_performance", [])
                        agent_domains = ext_stats.get("domains", [])
                        avg_confidence = ext_stats.get("avg_confidence", 0.0)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

        return AgentStatsResponse(
            name=agent_data.get("name", agent_id),
            elo=agent_data.get("elo", 1500.0),
            matches=matches,
            wins=wins,
            losses=agent_data.get("losses", 0),
            win_rate=round(win_rate, 3),
            avg_confidence=avg_confidence,
            domains=agent_domains,
            recent_performance=recent_performance,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting stats for agent %s: %s", agent_id, e)
        raise HTTPException(status_code=500, detail="Failed to get agent stats")


@router.get(
    "/agents/{agent_id}/calibration", response_model=AgentCalibrationResponse
)
async def get_agent_calibration(
    agent_id: str,
    domain: str | None = Query(None, description="Filter by domain"),
    elo=Depends(get_elo_system),
) -> AgentCalibrationResponse:
    """Get calibration scores for a specific agent."""
    try:
        calibration_score = 0.0
        buckets: list[CalibrationBucket] = []
        total_predictions = 0

        if elo:
            # Try dedicated calibration methods
            try:
                if hasattr(elo, "get_calibration_by_bucket"):
                    raw_buckets = elo.get_calibration_by_bucket(
                        agent_id, domain=domain
                    )
                    for b in raw_buckets:
                        if isinstance(b, dict):
                            buckets.append(CalibrationBucket(
                                bucket=b.get("bucket", ""),
                                predicted=b.get("predicted", 0.0),
                                actual=b.get("actual", 0.0),
                                count=b.get("count", 0),
                            ))
                            total_predictions += b.get("count", 0)

                if hasattr(elo, "get_calibration_leaderboard"):
                    cal_lb = elo.get_calibration_leaderboard(limit=500)
                    for entry in cal_lb:
                        d = (
                            entry
                            if isinstance(entry, dict)
                            else _agent_to_dict(entry)
                        )
                        if d.get("name") == agent_id:
                            calibration_score = d.get(
                                "calibration_score", d.get("score", 0.0)
                            )
                            break
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(
                    "Could not get calibration for %s: %s", agent_id, e
                )

        # Verify agent exists
        if not buckets and calibration_score == 0.0:
            known_agents = _get_known_agents()
            found = False
            if elo and hasattr(elo, "get_leaderboard"):
                try:
                    for a in elo.get_leaderboard(limit=500):
                        if _agent_to_dict(a).get("name") == agent_id:
                            found = True
                            break
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass
            if not found and agent_id not in known_agents:
                raise NotFoundError(f"Agent {agent_id} not found")

        return AgentCalibrationResponse(
            name=agent_id,
            calibration_score=calibration_score,
            buckets=buckets,
            total_predictions=total_predictions,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(
            "Error getting calibration for agent %s: %s", agent_id, e
        )
        raise HTTPException(
            status_code=500, detail="Failed to get agent calibration"
        )
