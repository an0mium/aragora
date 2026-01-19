"""
Agent-related endpoint handlers.

Endpoints:
- GET /api/leaderboard - Get agent rankings
- GET /api/rankings - Get agent rankings (alias)
- GET /api/agents/local - List detected local LLM servers
- GET /api/agents/local/status - Get local LLM availability status
- GET /api/agent/{name}/profile - Get agent profile
- GET /api/agent/{name}/history - Get agent match history
- GET /api/agent/{name}/calibration - Get calibration scores
- GET /api/agent/{name}/consistency - Get consistency score
- GET /api/agent/{name}/flips - Get agent flip history
- GET /api/agent/{name}/network - Get relationship network
- GET /api/agent/{name}/rivals - Get top rivals
- GET /api/agent/{name}/allies - Get top allies
- GET /api/agent/{name}/moments - Get significant moments
- GET /api/agent/{name}/positions - Get position history
- GET /api/agent/{name}/metadata - Get agent metadata (model, capabilities)
- GET /api/agent/compare - Compare multiple agents
- GET /api/agent/{name}/head-to-head/{opponent} - Get head-to-head stats
- GET /api/flips/recent - Get recent flips across all agents
- GET /api/flips/summary - Get flip summary for dashboard
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from aragora.config import (
    CACHE_TTL_AGENT_FLIPS,
    CACHE_TTL_AGENT_H2H,
    CACHE_TTL_AGENT_PROFILE,
    CACHE_TTL_CALIBRATION_LB,
    CACHE_TTL_FLIPS_RECENT,
    CACHE_TTL_FLIPS_SUMMARY,
    CACHE_TTL_LEADERBOARD,
    CACHE_TTL_RECENT_MATCHES,
)

logger = logging.getLogger(__name__)
from aragora.persistence.db_config import DatabaseType, get_db_path

from ..base import (
    SAFE_AGENT_PATTERN,
    SAFE_ID_PATTERN,
    BaseHandler,
    HandlerResult,
    agent_to_dict,
    error_response,
    get_agent_name,
    get_int_param,
    get_string_param,
    handle_errors,
    json_response,
    safe_error_message,
    ttl_cache,
    validate_path_segment,
)
from ..utils.rate_limit import rate_limit


class AgentsHandler(BaseHandler):
    """Handler for agent-related endpoints."""

    ROUTES = [
        "/api/agents",
        "/api/agents/health",
        "/api/agents/local",
        "/api/agents/local/status",
        "/api/leaderboard",
        "/api/rankings",
        # Note: /api/calibration/leaderboard handled by CalibrationHandler
        "/api/matches/recent",
        "/api/agent/compare",
        "/api/agent/*/profile",
        "/api/agent/*/history",
        "/api/agent/*/calibration",
        "/api/agent/*/consistency",
        "/api/agent/*/flips",
        "/api/agent/*/network",
        "/api/agent/*/rivals",
        "/api/agent/*/allies",
        "/api/agent/*/moments",
        "/api/agent/*/positions",
        "/api/agent/*/domains",
        "/api/agent/*/performance",
        "/api/agent/*/metadata",
        "/api/agent/*/head-to-head/*",
        "/api/agent/*/opponent-briefing/*",
        "/api/agent/*/introspect",
        "/api/flips/recent",
        "/api/flips/summary",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/agents":
            return True
        if path == "/api/agents/health":
            return True
        if path in ("/api/agents/local", "/api/agents/local/status"):
            return True
        if path in ("/api/leaderboard", "/api/rankings"):
            return True
        if path == "/api/matches/recent":
            return True
        if path == "/api/agent/compare":
            return True
        if path.startswith("/api/agent/"):
            return True
        if path.startswith("/api/flips/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route agent requests to appropriate methods."""
        # Agent health endpoint (must come before /api/agents check)
        if path == "/api/agents/health":
            return self._get_agent_health()

        # Local LLM endpoints (must come before /api/agents check)
        if path == "/api/agents/local":
            return self._list_local_agents()

        if path == "/api/agents/local/status":
            return self._get_local_status()

        # List all agents
        if path == "/api/agents":
            include_stats = (
                get_string_param(query_params, "include_stats", "false").lower() == "true"
            )
            return self._list_agents(include_stats)

        # Leaderboard endpoints
        if path in ("/api/leaderboard", "/api/rankings"):
            limit = get_int_param(query_params, "limit", 20)
            domain = get_string_param(query_params, "domain")
            if domain:
                is_valid, err = validate_path_segment(domain, "domain", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._get_leaderboard(limit, domain)

        # Note: /api/calibration/leaderboard now handled by CalibrationHandler

        if path == "/api/matches/recent":
            limit = get_int_param(query_params, "limit", 10)
            loop_id = get_string_param(query_params, "loop_id")
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._get_recent_matches(limit, loop_id)

        # Agent comparison
        if path == "/api/agent/compare":
            agents = query_params.get("agents", [])
            if isinstance(agents, str):
                agents = [agents]
            return self._compare_agents(agents)

        # Per-agent endpoints
        if path.startswith("/api/agent/"):
            return self._handle_agent_endpoint(path, query_params)

        # Flip endpoints (not per-agent)
        if path == "/api/flips/recent":
            limit = get_int_param(query_params, "limit", 20)
            return self._get_recent_flips(limit)

        if path == "/api/flips/summary":
            return self._get_flip_summary()

        return None

    def _handle_agent_endpoint(self, path: str, query_params: dict) -> Optional[HandlerResult]:
        """Handle /api/agent/{name}/* endpoints."""
        parts = path.split("/")
        if len(parts) < 4:
            return error_response("Invalid agent path", 400)

        # Extract and validate agent name
        agent_name, err = self.extract_path_param(path, 2, "agent", SAFE_AGENT_PATTERN)
        if err:
            return err

        # Head-to-head: /api/agent/{name}/head-to-head/{opponent}
        if len(parts) >= 6 and parts[4] == "head-to-head":
            opponent, err = self.extract_path_param(path, 4, "opponent", SAFE_AGENT_PATTERN)
            if err:
                return err
            return self._get_head_to_head(agent_name, opponent)

        # Opponent briefing: /api/agent/{name}/opponent-briefing/{opponent}
        if len(parts) >= 6 and parts[4] == "opponent-briefing":
            opponent, err = self.extract_path_param(path, 4, "opponent", SAFE_AGENT_PATTERN)
            if err:
                return err
            return self._get_opponent_briefing(agent_name, opponent)

        # Other endpoints: /api/agent/{name}/{endpoint}
        if len(parts) >= 5:
            endpoint = parts[4]
            return self._dispatch_agent_endpoint(agent_name, endpoint, query_params)

        return None

    def _dispatch_agent_endpoint(
        self, agent: str, endpoint: str, params: dict
    ) -> Optional[HandlerResult]:
        """Dispatch to specific agent endpoint handler."""
        handlers = {
            "profile": lambda: self._get_profile(agent),
            "history": lambda: self._get_history(agent, get_int_param(params, "limit", 30)),
            "calibration": lambda: self._get_calibration(agent, params.get("domain")),
            "consistency": lambda: self._get_consistency(agent),
            "flips": lambda: self._get_agent_flips(agent, get_int_param(params, "limit", 20)),
            "network": lambda: self._get_network(agent),
            "rivals": lambda: self._get_rivals(agent, get_int_param(params, "limit", 5)),
            "allies": lambda: self._get_allies(agent, get_int_param(params, "limit", 5)),
            "moments": lambda: self._get_moments(agent, get_int_param(params, "limit", 10)),
            "positions": lambda: self._get_positions(agent, get_int_param(params, "limit", 20)),
            "domains": lambda: self._get_domains(agent),
            "performance": lambda: self._get_performance(agent),
            "metadata": lambda: self._get_metadata(agent),
            "introspect": lambda: self._get_agent_introspect(agent, params.get("debate_id")),
        }

        if endpoint in handlers:
            return handlers[endpoint]()

        return None

    @rate_limit(rpm=30, limiter_name="agents_list")
    @handle_errors("list agents")
    @ttl_cache(ttl_seconds=CACHE_TTL_LEADERBOARD, key_prefix="agents_list")
    def _list_agents(self, include_stats: bool = False) -> HandlerResult:
        """List all known agents.

        Args:
            include_stats: If True, include basic stats (ELO, match count)

        Returns:
            List of agent names or agent objects with stats
        """
        elo = self.get_elo_system()
        agents = []

        # Get agents from ELO system if available
        if elo:
            try:
                # Get all agents from leaderboard (large limit to get all)
                rankings = elo.get_leaderboard(limit=500)
                for agent in rankings:
                    agent_dict = agent_to_dict(agent)
                    name = agent_dict.get("name", "")
                    if include_stats:
                        agents.append(
                            {
                                "name": name,
                                "elo": agent_dict.get("elo", 1500),
                                "matches": agent_dict.get("matches", 0),
                                "wins": agent_dict.get("wins", 0),
                                "losses": agent_dict.get("losses", 0),
                            }
                        )
                    else:
                        agents.append({"name": name})
            except Exception as e:
                logger.warning(f"Could not get agents from ELO: {e}")

        # Fallback to known agent types if no ELO data
        if not agents:
            from aragora.agents.cli_agents import AGENT_TYPES  # type: ignore[attr-defined]

            agents = [{"name": name} for name in AGENT_TYPES.keys()]

        return json_response(
            {
                "agents": agents,
                "total": len(agents),
            }
        )

    @rate_limit(rpm=10, limiter_name="local_agents")
    @handle_errors("list local agents")
    def _list_local_agents(self) -> HandlerResult:
        """List detected local LLM servers (Ollama, LM Studio, etc.).

        Returns:
            List of detected local LLM servers with their available models
        """
        try:
            from aragora.agents.registry import AgentRegistry

            local_agents = AgentRegistry.detect_local_agents()

            return json_response(
                {
                    "servers": local_agents,
                    "total": len(local_agents),
                    "available_count": sum(1 for a in local_agents if a.get("available", False)),
                }
            )
        except Exception as e:
            logger.warning(f"Could not detect local LLMs: {e}")
            return json_response(
                {
                    "servers": [],
                    "total": 0,
                    "available_count": 0,
                    "error": str(e),
                }
            )

    @rate_limit(rpm=10, limiter_name="local_status")
    @handle_errors("get local status")
    def _get_local_status(self) -> HandlerResult:
        """Get overall local LLM availability status with recommendations.

        Returns:
            Status including availability, recommended server/model
        """
        try:
            from aragora.agents.registry import AgentRegistry

            status = AgentRegistry.get_local_status()

            return json_response(
                {
                    "available": status.get("any_available", False),
                    "total_models": status.get("total_models", 0),
                    "recommended": {
                        "server": status.get("recommended_server"),
                        "model": status.get("recommended_model"),
                    },
                    "available_agents": status.get("available_agents", []),
                    "servers": status.get("servers", []),
                }
            )
        except Exception as e:
            logger.warning(f"Could not get local LLM status: {e}")
            return json_response(
                {
                    "available": False,
                    "total_models": 0,
                    "recommended": {"server": None, "model": None},
                    "available_agents": [],
                    "servers": [],
                    "error": str(e),
                }
            )

    @rate_limit(rpm=30, limiter_name="agent_health")
    @handle_errors("get agent health")
    def _get_agent_health(self) -> HandlerResult:
        """Get runtime health status for all agents.

        Returns agent availability based on circuit breaker states,
        fallback chain status, and recent error metrics.

        Returns:
            Dict with health status for each agent type and overall system health
        """
        import time

        health: dict[str, Any] = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "agents": {},
            "circuit_breakers": {},
            "fallback": {},
        }

        # Get circuit breaker status
        try:
            from aragora.resilience import get_circuit_breaker

            cb = get_circuit_breaker()
            if cb:
                # Get all tracked agents from circuit breaker
                states = cb.get_all_states() if hasattr(cb, "get_all_states") else {}
                for agent_name, state in states.items():
                    health["circuit_breakers"][agent_name] = {
                        "state": state.get("state", "unknown"),
                        "failure_count": state.get("failure_count", 0),
                        "last_failure": state.get("last_failure_time"),
                        "available": state.get("state") != "open",
                    }

                    # Mark overall as degraded if any circuit is open
                    if state.get("state") == "open":
                        health["overall_status"] = "degraded"
        except ImportError:
            health["circuit_breakers"]["_note"] = "CircuitBreaker module not available"
        except Exception as e:
            logger.debug(f"Could not get circuit breaker status: {e}")
            health["circuit_breakers"]["_error"] = str(e)

        # Get fallback chain status
        try:
            from aragora.agents.fallback import get_local_fallback_providers, is_local_llm_available

            health["fallback"] = {
                "openrouter_available": bool(__import__("os").environ.get("OPENROUTER_API_KEY")),
                "local_llm_available": is_local_llm_available(),
                "local_providers": get_local_fallback_providers(),
            }
        except ImportError:
            health["fallback"]["_note"] = "Fallback module not available"
        except Exception as e:
            logger.debug(f"Could not get fallback status: {e}")
            health["fallback"]["_error"] = str(e)

        # Get registered agent types and their availability
        try:
            from aragora.agents.registry import AgentRegistry, register_all_agents

            register_all_agents()
            all_agents = AgentRegistry.list_all()

            for agent_type, spec in all_agents.items():
                agent_health = {
                    "type": spec.get("category", "unknown"),
                    "requires_api_key": spec.get("requires_api_key", False),
                    "api_key_configured": False,
                    "available": False,
                }

                # Check if required API key is configured
                env_var = spec.get("env_var")
                if env_var:
                    agent_health["api_key_configured"] = bool(__import__("os").environ.get(env_var))
                    agent_health["available"] = agent_health["api_key_configured"]
                else:
                    # CLI agents or agents without API keys
                    agent_health["available"] = True

                # Check circuit breaker state
                cb_state = health["circuit_breakers"].get(agent_type, {})
                if cb_state.get("state") == "open":
                    agent_health["available"] = False
                    agent_health["circuit_breaker_open"] = True

                health["agents"][agent_type] = agent_health
        except ImportError:
            health["agents"]["_note"] = "AgentRegistry not available"
        except Exception as e:
            logger.debug(f"Could not get agent registry: {e}")
            health["agents"]["_error"] = str(e)

        # Calculate summary
        available_count = sum(
            1
            for a in health["agents"].values()
            if isinstance(a, dict) and a.get("available", False)
        )
        total_count = sum(1 for a in health["agents"].values() if isinstance(a, dict))

        health["summary"] = {
            "available_agents": available_count,
            "total_agents": total_count,
            "availability_rate": (
                round(available_count / total_count, 2) if total_count > 0 else 0.0
            ),
        }

        # Downgrade status if too few agents available
        if total_count > 0 and available_count / total_count < 0.5:
            health["overall_status"] = "degraded"
        if available_count == 0 and total_count > 0:
            health["overall_status"] = "unhealthy"

        return json_response(health)

    @rate_limit(rpm=30, limiter_name="leaderboard")
    def _get_leaderboard(self, limit: int, domain: Optional[str]) -> HandlerResult:
        """Get agent leaderboard with consistency scores (batched to avoid N+1)."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            # Use cached leaderboard when available (no domain filter needed for cache)
            if domain is None:
                rankings = elo.get_cached_leaderboard(limit=min(limit, 50))
            else:
                rankings = elo.get_leaderboard(limit=min(limit, 50), domain=domain)

            # Batch fetch consistency scores (uses 3 queries instead of 3*N)
            consistency_map = {}
            try:
                from aragora.insights.flip_detector import FlipDetector

                nomic_dir = self.get_nomic_dir()
                if nomic_dir:
                    detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
                    # Extract all agent names first
                    agent_names = []
                    for agent in rankings:
                        name = get_agent_name(agent)
                        if name:
                            agent_names.append(name)

                    # Batch fetch all consistency scores at once
                    if agent_names:
                        try:
                            scores = detector.get_agents_consistency_batch(agent_names)
                            for agent_name, score in scores.items():
                                total_positions = max(score.total_positions, 1)
                                consistency = 1.0 - (score.total_flips / total_positions)
                                consistency_map[agent_name] = {
                                    "consistency": round(consistency, 3),
                                    "consistency_class": (
                                        "high"
                                        if consistency >= 0.8
                                        else "medium" if consistency >= 0.6 else "low"
                                    ),
                                }
                        except Exception as e:
                            # Fallback: set default consistency for all agents
                            logger.warning(
                                "Consistency detection failed, using fallback: %s: %s",
                                type(e).__name__,
                                e,
                            )
                            for name in agent_names:
                                consistency_map[name] = {
                                    "consistency": 1.0,
                                    "consistency_class": "high",
                                    "degraded": True,
                                    "degraded_reason": "Consistency detection unavailable",
                                }
            except ImportError:
                # FlipDetector not available - set degraded flag so clients know
                logger.debug("FlipDetector import failed - consistency scores unavailable")
                for agent in rankings:
                    name = get_agent_name(agent)
                    if name:
                        consistency_map[name] = {
                            "consistency": None,
                            "consistency_class": "unknown",
                            "degraded": True,
                            "degraded_reason": "FlipDetector module not available",
                        }

            # Enhance rankings with consistency data
            enhanced_rankings = []
            for agent in rankings:
                agent_dict = agent_to_dict(agent)
                agent_name = agent_dict.get("name", "unknown")

                if agent_name in consistency_map:
                    agent_dict.update(consistency_map[agent_name])

                enhanced_rankings.append(agent_dict)

            return json_response({"rankings": enhanced_rankings, "agents": enhanced_rankings})
        except Exception as e:
            return error_response(safe_error_message(e, "get leaderboard"), 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_CALIBRATION_LB, key_prefix="calibration_lb", skip_first=True)
    def _get_calibration_leaderboard(self, limit: int) -> HandlerResult:
        """Get calibration leaderboard."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            # Get calibration scores from ELO system
            rankings = elo.get_leaderboard(limit=min(limit, 50))
            # Add calibration data if available
            return json_response({"rankings": rankings})
        except Exception as e:
            return error_response(safe_error_message(e, "get calibration leaderboard"), 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_RECENT_MATCHES, key_prefix="recent_matches", skip_first=True)
    def _get_recent_matches(self, limit: int, loop_id: Optional[str]) -> HandlerResult:
        """Get recent matches (uses JSON snapshot cache for fast reads)."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            # Use cached matches from JSON snapshot (avoids SQLite locking)
            if hasattr(elo, "get_cached_recent_matches"):
                matches = elo.get_cached_recent_matches(limit=min(limit, 50))
            else:
                matches = elo.get_recent_matches(limit=min(limit, 50))
            return json_response({"matches": matches})
        except Exception as e:
            return error_response(safe_error_message(e, "get recent matches"), 500)

    def _compare_agents(self, agents: List[str]) -> HandlerResult:
        """Compare multiple agents using batch rating lookups."""
        if len(agents) < 2:
            return error_response("Need at least 2 agents to compare", 400)

        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            # Limit to 5 agents and batch fetch ratings in single query
            limited_agents = agents[:5]
            ratings_map = elo.get_ratings_batch(limited_agents)

            profiles = []
            for agent in limited_agents:
                rating = ratings_map.get(agent)
                stats = elo.get_agent_stats(agent) if hasattr(elo, "get_agent_stats") else {}
                profiles.append(
                    {
                        "name": agent,
                        "rating": rating,
                        **stats,
                    }
                )

            # Get head-to-head if exactly 2 agents
            head_to_head = None
            if len(agents) == 2:
                try:
                    h2h = elo.get_head_to_head(agents[0], agents[1])
                    head_to_head = h2h
                except Exception as e:
                    logger.warning(
                        "Head-to-head lookup failed for %s vs %s: %s: %s",
                        agents[0],
                        agents[1],
                        type(e).__name__,
                        e,
                    )

            return json_response(
                {
                    "agents": profiles,
                    "head_to_head": head_to_head,
                }
            )
        except Exception as e:
            return error_response(safe_error_message(e, "comparison"), 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_AGENT_PROFILE, key_prefix="agent_profile", skip_first=True)
    @handle_errors("agent profile")
    def _get_profile(self, agent: str) -> HandlerResult:
        """Get complete agent profile."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rating = elo.get_rating(agent)
        stats: dict[str, Any] = {}
        if hasattr(elo, "get_agent_stats"):
            stats = elo.get_agent_stats(agent) or {}

        return json_response(
            {
                "name": agent,
                "rating": rating,
                "rank": stats.get("rank"),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "win_rate": stats.get("win_rate", 0.0),
            }
        )

    @handle_errors("agent history")
    def _get_history(self, agent: str, limit: int) -> HandlerResult:
        """Get agent match history."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        history = elo.get_elo_history(agent, limit=min(limit, 100))
        # Convert list of (timestamp, elo) tuples to list of dicts for JSON
        history_list = [{"timestamp": ts, "elo": rating} for ts, rating in history]
        return json_response({"agent": agent, "history": history_list})

    @handle_errors("agent calibration")
    def _get_calibration(self, agent: str, domain: Optional[str]) -> HandlerResult:
        """Get agent calibration scores."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, "get_calibration"):
            calibration = elo.get_calibration(agent, domain=domain)
        else:
            calibration = {"agent": agent, "score": 0.5}
        return json_response(calibration)

    @handle_errors("agent consistency")
    def _get_consistency(self, agent: str) -> HandlerResult:
        """Get agent consistency score."""
        from aragora.insights.flip_detector import FlipDetector

        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
            score = detector.get_agent_consistency(agent)
            return json_response({"agent": agent, "consistency_score": score})
        return json_response({"agent": agent, "consistency_score": 1.0})

    @handle_errors("agent network")
    def _get_network(self, agent: str) -> HandlerResult:
        """Get agent relationship network."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rivals = elo.get_rivals(agent, limit=5) if hasattr(elo, "get_rivals") else []
        allies = elo.get_allies(agent, limit=5) if hasattr(elo, "get_allies") else []
        return json_response(
            {
                "agent": agent,
                "rivals": rivals,
                "allies": allies,
            }
        )

    @handle_errors("agent rivals")
    def _get_rivals(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top rivals."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rivals = elo.get_rivals(agent, limit=limit) if hasattr(elo, "get_rivals") else []
        return json_response({"agent": agent, "rivals": rivals})

    @handle_errors("agent allies")
    def _get_allies(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top allies."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        allies = elo.get_allies(agent, limit=limit) if hasattr(elo, "get_allies") else []
        return json_response({"agent": agent, "allies": allies})

    @handle_errors("agent moments")
    def _get_moments(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's significant moments."""
        from aragora.agents.grounded import MomentDetector

        elo = self.get_elo_system()
        if elo:
            detector = MomentDetector(elo_system=elo)
            moments = detector.get_agent_moments(agent, limit=limit)
            # Convert moments to dicts for JSON serialization
            moments_data = [
                {
                    "id": m.id,
                    "moment_type": m.moment_type,
                    "agent_name": m.agent_name,
                    "description": m.description,
                    "significance_score": m.significance_score,
                    "timestamp": (
                        getattr(m, "timestamp", None).isoformat()
                        if getattr(m, "timestamp", None)
                        else None
                    ),
                    "debate_id": m.debate_id,
                }
                for m in moments
            ]
            return json_response({"agent": agent, "moments": moments_data})
        return json_response({"agent": agent, "moments": []})

    @handle_errors("agent positions")
    def _get_positions(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's position history."""
        from aragora.agents.grounded import PositionLedger

        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            ledger = PositionLedger(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
            positions = ledger.get_agent_positions(agent, limit=limit)
            return json_response({"agent": agent, "positions": positions})
        return json_response({"agent": agent, "positions": []})

    @handle_errors("agent domains")
    def _get_domains(self, agent: str) -> HandlerResult:
        """Get agent's domain-specific ELO ratings.

        Returns domain expertise breakdown showing how the agent
        performs across different topic areas.
        """
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rating = elo.get_rating(agent)

        # Extract domain ELOs from rating
        domain_elos = rating.domain_elos if hasattr(rating, "domain_elos") else {}

        # Sort domains by ELO descending
        sorted_domains = sorted(domain_elos.items(), key=lambda x: x[1], reverse=True)

        domains = [
            {
                "domain": domain,
                "elo": elo_score,
                "relative": round(elo_score - rating.elo, 1),  # Relative to overall
            }
            for domain, elo_score in sorted_domains
        ]

        return json_response(
            {
                "agent": agent,
                "overall_elo": rating.elo,
                "domains": domains,
                "domain_count": len(domains),
            }
        )

    @handle_errors("agent performance")
    def _get_performance(self, agent: str) -> HandlerResult:
        """Get detailed agent performance statistics.

        Returns win rates, average scores, and performance trends.
        """
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rating = elo.get_rating(agent)

        # Calculate derived metrics
        total_games = rating.wins + rating.losses + rating.draws
        win_rate = rating.wins / total_games if total_games > 0 else 0.0

        # Get recent match history for trend analysis
        recent_matches = (
            elo.get_agent_history(agent, limit=20) if hasattr(elo, "get_agent_history") else []
        )

        # Calculate recent win rate (last 10 matches)
        recent_wins = sum(1 for m in recent_matches[:10] if m.get("result") == "win")
        recent_total = min(10, len(recent_matches))
        recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0.0

        # Calculate ELO trend from history
        elo_history = (
            elo.get_elo_history(agent, limit=20) if hasattr(elo, "get_elo_history") else []
        )
        elo_trend = 0.0
        if len(elo_history) >= 2:
            elo_trend = elo_history[0][1] - elo_history[-1][1]  # Most recent minus oldest

        return json_response(
            {
                "agent": agent,
                "elo": rating.elo,
                "total_games": total_games,
                "wins": rating.wins,
                "losses": rating.losses,
                "draws": rating.draws,
                "win_rate": round(win_rate, 3),
                "recent_win_rate": round(recent_win_rate, 3),
                "elo_trend": round(elo_trend, 1),
                "critiques_accepted": rating.critiques_accepted,
                "critiques_total": rating.critiques_total,
                "critique_acceptance_rate": round(rating.critique_acceptance_rate, 3),
                "calibration": {
                    "accuracy": round(rating.calibration_accuracy, 3),
                    "brier_score": round(rating.calibration_brier_score, 3),
                    "prediction_count": rating.calibration_total,
                },
            }
        )

    @handle_errors("agent metadata")
    def _get_metadata(self, agent: str) -> HandlerResult:
        """Get rich metadata about an agent.

        Returns model information, capabilities, and provider details
        from the agent_metadata table (populated by seed script).

        Args:
            agent: Agent name to look up

        Returns:
            JSON with agent metadata including:
            - provider: LLM provider (anthropic, openai, google, etc.)
            - model_id: Full model identifier
            - context_window: Maximum context window size
            - specialties: Areas of expertise
            - strengths: Key capabilities
            - release_date: When the model was released
        """
        import json
        import sqlite3

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return json_response({
                "agent": agent,
                "metadata": None,
                "message": "Database not available",
            })

        elo_path = get_db_path(DatabaseType.ELO, nomic_dir)
        if not elo_path.exists():
            return json_response({
                "agent": agent,
                "metadata": None,
                "message": "ELO database not found",
            })

        try:
            conn = sqlite3.connect(elo_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT agent_name, provider, model_id, context_window,
                       specialties, strengths, release_date, updated_at
                FROM agent_metadata
                WHERE agent_name = ?
                """,
                (agent,),
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return json_response({
                    "agent": agent,
                    "metadata": None,
                    "message": "Agent metadata not found",
                })

            # Parse JSON fields
            specialties = []
            strengths = []
            try:
                if row["specialties"]:
                    specialties = json.loads(row["specialties"])
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                if row["strengths"]:
                    strengths = json.loads(row["strengths"])
            except (json.JSONDecodeError, TypeError):
                pass

            return json_response({
                "agent": agent,
                "metadata": {
                    "provider": row["provider"],
                    "model_id": row["model_id"],
                    "context_window": row["context_window"],
                    "specialties": specialties,
                    "strengths": strengths,
                    "release_date": row["release_date"],
                    "updated_at": row["updated_at"],
                },
            })
        except sqlite3.OperationalError as e:
            # Table may not exist yet
            if "no such table" in str(e):
                return json_response({
                    "agent": agent,
                    "metadata": None,
                    "message": "Agent metadata table not initialized. Run seed_agents.py to populate.",
                })
            raise

    @handle_errors("agent introspect")
    def _get_agent_introspect(self, agent: str, debate_id: Optional[str] = None) -> HandlerResult:
        """Get agent introspection data for self-awareness and debugging.

        This endpoint provides comprehensive internal state information that
        agents can query to understand their own cognitive state, useful for
        debugging, self-improvement, and mid-debate introspection.

        Args:
            agent: Agent name to introspect
            debate_id: Optional debate ID for debate-specific state

        Returns:
            JSON with agent's internal state including:
            - identity: Basic agent info and persona
            - calibration: Prediction accuracy metrics
            - positions: Recent stance history
            - performance: Win/loss and rating data
            - memory_summary: Memory tier statistics
            - fatigue_indicators: Signs of cognitive fatigue (if available)
        """
        elo = self.get_elo_system()

        introspection: dict[str, Any] = {
            "agent_id": agent,
            "timestamp": self._get_timestamp(),
            "identity": {"name": agent},
            "calibration": {},
            "positions": [],
            "performance": {},
            "memory_summary": {},
            "fatigue_indicators": None,  # Placeholder for fatigue detection
            "debate_context": None,
        }

        # Get basic rating/performance data
        if elo:
            try:
                rating = elo.get_rating(agent)
                total_games = rating.wins + rating.losses + rating.draws
                introspection["performance"] = {
                    "elo": rating.elo,
                    "total_games": total_games,
                    "wins": rating.wins,
                    "losses": rating.losses,
                    "win_rate": rating.wins / total_games if total_games > 0 else 0.0,
                }
                introspection["calibration"] = {
                    "accuracy": round(rating.calibration_accuracy, 3),
                    "brier_score": round(rating.calibration_brier_score, 3),
                    "prediction_count": rating.calibration_total,
                    "confidence_level": self._compute_confidence(rating),
                }
            except Exception as e:
                logger.debug(f"Could not get ELO data for {agent}: {e}")

        # Get position history
        try:
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                from aragora.ranking.position_tracker import PositionTracker

                tracker_path = nomic_dir / "position_tracker.json"
                if tracker_path.exists():
                    tracker = PositionTracker(str(tracker_path))
                    positions = tracker.get_agent_positions(agent, limit=5)
                    introspection["positions"] = [
                        {
                            "topic": p.get("topic", ""),
                            "stance": p.get("stance", ""),
                            "confidence": p.get("confidence", 0.5),
                            "timestamp": p.get("timestamp", ""),
                        }
                        for p in positions
                    ]
        except Exception as e:
            logger.debug(f"Could not get position data for {agent}: {e}")

        # Get memory tier summary
        try:
            from aragora.memory.continuum import ContinuumMemory

            memory = ContinuumMemory()
            tier_stats = memory.get_tier_counts()
            introspection["memory_summary"] = {
                "tier_counts": tier_stats,
                "total_memories": sum(tier_stats.values()),
                "red_line_count": len(memory.get_red_line_memories()),
            }
        except Exception as e:
            logger.debug(f"Could not get memory data: {e}")

        # Get persona info if available
        try:
            from aragora.agents.personas import PersonaManager

            persona_mgr = PersonaManager()
            persona = persona_mgr.get_persona(agent)
            if persona:
                introspection["identity"]["persona"] = {
                    "style": getattr(persona, "style", None),
                    "temperature": getattr(persona, "temperature", None),
                    "system_prompt_preview": (
                        getattr(persona, "system_prompt", "")[:200]
                        if getattr(persona, "system_prompt", None)
                        else None
                    ),
                }
        except Exception as e:
            logger.debug(f"Could not get persona data for {agent}: {e}")

        # Add debate-specific context if debate_id provided
        if debate_id:
            try:
                storage = self.get_storage()
                if storage:
                    debate = storage.get_debate(debate_id)
                    if debate:
                        # Find agent's messages in this debate
                        agent_msgs = [
                            m for m in debate.get("messages", []) if m.get("agent") == agent
                        ]
                        introspection["debate_context"] = {
                            "debate_id": debate_id,
                            "messages_sent": len(agent_msgs),
                            "current_round": debate.get("current_round", 0),
                            "debate_status": debate.get("status", "unknown"),
                        }
            except Exception as e:
                logger.debug(f"Could not get debate context: {e}")

        return json_response(introspection)

    def _compute_confidence(self, rating) -> str:
        """Compute confidence level from calibration data."""
        accuracy = rating.calibration_accuracy
        count = rating.calibration_total
        if count < 5:
            return "insufficient_data"
        if accuracy >= 0.8:
            return "high"
        if accuracy >= 0.6:
            return "medium"
        return "low"

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    @ttl_cache(ttl_seconds=CACHE_TTL_AGENT_H2H, key_prefix="agent_h2h", skip_first=True)
    @handle_errors("head-to-head stats")
    def _get_head_to_head(self, agent: str, opponent: str) -> HandlerResult:
        """Get head-to-head stats between two agents."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, "get_head_to_head"):
            stats = elo.get_head_to_head(agent, opponent)
        else:
            stats = {"matches": 0, "agent1_wins": 0, "agent2_wins": 0}
        return json_response(
            {
                "agent1": agent,
                "agent2": opponent,
                **stats,
            }
        )

    @handle_errors("opponent briefing")
    def _get_opponent_briefing(self, agent: str, opponent: str) -> HandlerResult:
        """Get strategic briefing about an opponent for an agent."""
        elo = self.get_elo_system()
        nomic_dir = self.get_nomic_dir()

        from aragora.agents.grounded import PersonaSynthesizer

        # Get position ledger if available
        position_ledger = None
        if nomic_dir:
            try:
                from aragora.agents.grounded import PositionLedger

                db_path = get_db_path(DatabaseType.POSITIONS, nomic_dir)
                if db_path.exists():
                    position_ledger = PositionLedger(str(db_path))
            except ImportError:
                pass

        # Get calibration tracker if available
        calibration_tracker = None
        try:
            from aragora.ranking.calibration import CalibrationTracker

            calibration_tracker = CalibrationTracker()
        except ImportError:
            pass

        synthesizer = PersonaSynthesizer(  # type: ignore[call-arg]
            elo_system=elo,
            calibration_tracker=calibration_tracker,
            position_ledger=position_ledger,
        )
        briefing = synthesizer.get_opponent_briefing(agent, opponent)

        if briefing:
            return json_response(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": briefing,
                }
            )
        else:
            return json_response(
                {
                    "agent": agent,
                    "opponent": opponent,
                    "briefing": None,
                    "message": "No opponent data available",
                }
            )

    # ==================== Flip Detector Endpoints ====================

    @ttl_cache(ttl_seconds=CACHE_TTL_AGENT_FLIPS, key_prefix="agent_flips", skip_first=True)
    @handle_errors("agent flips")
    def _get_agent_flips(self, agent: str, limit: int) -> HandlerResult:
        """Get recent position flips for an agent."""
        from aragora.insights.flip_detector import FlipDetector

        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
            flips = detector.detect_flips_for_agent(agent, lookback_positions=min(limit, 100))
            consistency = detector.get_agent_consistency(agent)
            return json_response(
                {
                    "agent": agent,
                    "flips": [f.to_dict() for f in flips],
                    "consistency": consistency.to_dict(),
                    "count": len(flips),
                }
            )
        return json_response(
            {
                "agent": agent,
                "flips": [],
                "consistency": {
                    "agent_name": agent,
                    "total_positions": 0,
                    "total_flips": 0,
                    "consistency_score": 1.0,
                },
                "count": 0,
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_FLIPS_RECENT, key_prefix="flips_recent", skip_first=True)
    @handle_errors("recent flips")
    def _get_recent_flips(self, limit: int) -> HandlerResult:
        """Get recent flips across all agents."""
        from aragora.insights.flip_detector import FlipDetector

        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
            flips = detector.get_recent_flips(limit=min(limit, 100))
            summary = detector.get_flip_summary()
            return json_response(
                {
                    "flips": [f.to_dict() for f in flips],
                    "summary": summary,
                    "count": len(flips),
                }
            )
        return json_response(
            {
                "flips": [],
                "summary": {"total_flips": 0, "by_type": {}, "by_agent": {}, "recent_24h": 0},
                "count": 0,
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_FLIPS_SUMMARY, key_prefix="flips_summary", skip_first=True)
    @handle_errors("flip summary")
    def _get_flip_summary(self) -> HandlerResult:
        """Get flip summary for dashboard."""
        from aragora.insights.flip_detector import FlipDetector

        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
            summary = detector.get_flip_summary()
            return json_response(summary)
        return json_response(
            {
                "total_flips": 0,
                "by_type": {},
                "by_agent": {},
                "recent_24h": 0,
            }
        )
