"""
Agent-related endpoint handlers.

Endpoints:
- GET /api/leaderboard - Get agent rankings
- GET /api/rankings - Get agent rankings (alias)
- GET /api/agent/{name}/profile - Get agent profile
- GET /api/agent/{name}/history - Get agent match history
- GET /api/agent/{name}/calibration - Get calibration scores
- GET /api/agent/{name}/consistency - Get consistency score
- GET /api/agent/{name}/network - Get relationship network
- GET /api/agent/{name}/rivals - Get top rivals
- GET /api/agent/{name}/allies - Get top allies
- GET /api/agent/{name}/moments - Get significant moments
- GET /api/agent/{name}/positions - Get position history
- GET /api/agent/compare - Compare multiple agents
- GET /api/agent/{name}/head-to-head/{opponent} - Get head-to-head stats
"""

from typing import Optional, List
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param, ttl_cache


class AgentsHandler(BaseHandler):
    """Handler for agent-related endpoints."""

    ROUTES = [
        "/api/leaderboard",
        "/api/rankings",
        "/api/calibration/leaderboard",
        "/api/matches/recent",
        "/api/agent/compare",
        "/api/agent/*/profile",
        "/api/agent/*/history",
        "/api/agent/*/calibration",
        "/api/agent/*/consistency",
        "/api/agent/*/network",
        "/api/agent/*/rivals",
        "/api/agent/*/allies",
        "/api/agent/*/moments",
        "/api/agent/*/positions",
        "/api/agent/*/head-to-head/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in ("/api/leaderboard", "/api/rankings", "/api/calibration/leaderboard"):
            return True
        if path == "/api/matches/recent":
            return True
        if path == "/api/agent/compare":
            return True
        if path.startswith("/api/agent/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route agent requests to appropriate methods."""
        # Leaderboard endpoints
        if path in ("/api/leaderboard", "/api/rankings"):
            limit = get_int_param(query_params, 'limit', 20)
            domain = query_params.get('domain')
            return self._get_leaderboard(limit, domain)

        if path == "/api/calibration/leaderboard":
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_calibration_leaderboard(limit)

        if path == "/api/matches/recent":
            limit = get_int_param(query_params, 'limit', 10)
            loop_id = query_params.get('loop_id')
            return self._get_recent_matches(limit, loop_id)

        # Agent comparison
        if path == "/api/agent/compare":
            agents = query_params.get('agents', [])
            if isinstance(agents, str):
                agents = [agents]
            return self._compare_agents(agents)

        # Per-agent endpoints
        if path.startswith("/api/agent/"):
            return self._handle_agent_endpoint(path, query_params)

        return None

    def _handle_agent_endpoint(self, path: str, query_params: dict) -> Optional[HandlerResult]:
        """Handle /api/agent/{name}/* endpoints."""
        parts = path.split("/")
        if len(parts) < 4:
            return error_response("Invalid agent path", 400)

        agent_name = parts[3]

        # Head-to-head: /api/agent/{name}/head-to-head/{opponent}
        if len(parts) >= 6 and parts[4] == "head-to-head":
            opponent = parts[5]
            return self._get_head_to_head(agent_name, opponent)

        # Other endpoints: /api/agent/{name}/{endpoint}
        if len(parts) >= 5:
            endpoint = parts[4]
            return self._dispatch_agent_endpoint(agent_name, endpoint, query_params)

        return None

    def _dispatch_agent_endpoint(self, agent: str, endpoint: str, params: dict) -> Optional[HandlerResult]:
        """Dispatch to specific agent endpoint handler."""
        handlers = {
            "profile": lambda: self._get_profile(agent),
            "history": lambda: self._get_history(agent, get_int_param(params, 'limit', 30)),
            "calibration": lambda: self._get_calibration(agent, params.get('domain')),
            "consistency": lambda: self._get_consistency(agent),
            "network": lambda: self._get_network(agent),
            "rivals": lambda: self._get_rivals(agent, get_int_param(params, 'limit', 5)),
            "allies": lambda: self._get_allies(agent, get_int_param(params, 'limit', 5)),
            "moments": lambda: self._get_moments(agent, get_int_param(params, 'limit', 10)),
            "positions": lambda: self._get_positions(agent, get_int_param(params, 'limit', 20)),
        }

        if endpoint in handlers:
            return handlers[endpoint]()

        return None

    @ttl_cache(ttl_seconds=300, key_prefix="leaderboard")
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

            # Batch fetch consistency scores to avoid N+1 queries
            consistency_map = {}
            try:
                from aragora.insights.flip_detector import FlipDetector
                nomic_dir = self.get_nomic_dir()
                if nomic_dir:
                    detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
                    for agent in rankings:
                        agent_name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)
                        if agent_name:
                            try:
                                score = detector.get_agent_consistency(agent_name)
                                # Handle both dict and object returns
                                if hasattr(score, 'total_flips'):
                                    total_positions = max(score.total_positions, 1)
                                    consistency = 1.0 - (score.total_flips / total_positions)
                                elif isinstance(score, dict):
                                    total_positions = max(score.get('total_positions', 1), 1)
                                    consistency = 1.0 - (score.get('total_flips', 0) / total_positions)
                                else:
                                    consistency = 1.0
                                consistency_map[agent_name] = {
                                    "consistency": round(consistency, 3),
                                    "consistency_class": "high" if consistency >= 0.8 else "medium" if consistency >= 0.6 else "low"
                                }
                            except Exception:
                                consistency_map[agent_name] = {"consistency": 1.0, "consistency_class": "high"}
            except ImportError:
                pass  # FlipDetector not available, continue without consistency

            # Enhance rankings with consistency data
            enhanced_rankings = []
            for agent in rankings:
                if isinstance(agent, dict):
                    agent_dict = agent.copy()
                    agent_name = agent.get("name")
                else:
                    agent_dict = {
                        "name": getattr(agent, "name", "unknown"),
                        "elo": getattr(agent, "elo", 1500),
                        "wins": getattr(agent, "wins", 0),
                        "losses": getattr(agent, "losses", 0),
                        "draws": getattr(agent, "draws", 0),
                        "win_rate": getattr(agent, "win_rate", 0),
                        "games": getattr(agent, "games", 0),
                    }
                    agent_name = agent_dict["name"]

                if agent_name in consistency_map:
                    agent_dict.update(consistency_map[agent_name])

                enhanced_rankings.append(agent_dict)

            return json_response({"rankings": enhanced_rankings, "agents": enhanced_rankings})
        except Exception as e:
            return error_response(f"Failed to get leaderboard: {e}", 500)

    @ttl_cache(ttl_seconds=300, key_prefix="calibration_lb")
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
            return error_response(f"Failed to get calibration leaderboard: {e}", 500)

    @ttl_cache(ttl_seconds=120, key_prefix="recent_matches")
    def _get_recent_matches(self, limit: int, loop_id: Optional[str]) -> HandlerResult:
        """Get recent matches (uses JSON snapshot cache for fast reads)."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            # Use cached matches from JSON snapshot (avoids SQLite locking)
            if hasattr(elo, 'get_cached_recent_matches'):
                matches = elo.get_cached_recent_matches(limit=min(limit, 50))
            else:
                matches = elo.get_recent_matches(limit=min(limit, 50))
            return json_response({"matches": matches})
        except Exception as e:
            return error_response(f"Failed to get recent matches: {e}", 500)

    def _compare_agents(self, agents: List[str]) -> HandlerResult:
        """Compare multiple agents."""
        if len(agents) < 2:
            return error_response("Need at least 2 agents to compare", 400)

        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            profiles = []
            for agent in agents[:5]:  # Limit to 5 agents
                rating = elo.get_rating(agent)
                stats = elo.get_agent_stats(agent) if hasattr(elo, 'get_agent_stats') else {}
                profiles.append({
                    "name": agent,
                    "rating": rating,
                    **stats,
                })

            # Get head-to-head if exactly 2 agents
            head_to_head = None
            if len(agents) == 2:
                try:
                    h2h = elo.get_head_to_head(agents[0], agents[1])
                    head_to_head = h2h
                except Exception:
                    pass

            return json_response({
                "agents": profiles,
                "head_to_head": head_to_head,
            })
        except Exception as e:
            return error_response(f"Comparison failed: {e}", 500)

    @ttl_cache(ttl_seconds=600, key_prefix="agent_profile")
    def _get_profile(self, agent: str) -> HandlerResult:
        """Get complete agent profile."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            rating = elo.get_rating(agent)
            stats = {}
            if hasattr(elo, 'get_agent_stats'):
                stats = elo.get_agent_stats(agent) or {}

            return json_response({
                "name": agent,
                "rating": rating,
                "rank": stats.get("rank"),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "win_rate": stats.get("win_rate", 0.0),
            })
        except Exception as e:
            return error_response(f"Failed to get profile: {e}", 500)

    def _get_history(self, agent: str, limit: int) -> HandlerResult:
        """Get agent match history."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            history = elo.get_agent_history(agent, limit=min(limit, 100))
            return json_response({"agent": agent, "history": history})
        except Exception as e:
            return error_response(f"Failed to get history: {e}", 500)

    def _get_calibration(self, agent: str, domain: Optional[str]) -> HandlerResult:
        """Get agent calibration scores."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            if hasattr(elo, 'get_calibration'):
                calibration = elo.get_calibration(agent, domain=domain)
            else:
                calibration = {"agent": agent, "score": 0.5}
            return json_response(calibration)
        except Exception as e:
            return error_response(f"Failed to get calibration: {e}", 500)

    def _get_consistency(self, agent: str) -> HandlerResult:
        """Get agent consistency score."""
        try:
            from aragora.insights.flip_detector import FlipDetector
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
                score = detector.get_agent_consistency(agent)
                return json_response({"agent": agent, "consistency_score": score})
            return json_response({"agent": agent, "consistency_score": 1.0})
        except Exception as e:
            return error_response(f"Failed to get consistency: {e}", 500)

    def _get_network(self, agent: str) -> HandlerResult:
        """Get agent relationship network."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            rivals = elo.get_rivals(agent, limit=5) if hasattr(elo, 'get_rivals') else []
            allies = elo.get_allies(agent, limit=5) if hasattr(elo, 'get_allies') else []
            return json_response({
                "agent": agent,
                "rivals": rivals,
                "allies": allies,
            })
        except Exception as e:
            return error_response(f"Failed to get network: {e}", 500)

    def _get_rivals(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top rivals."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            rivals = elo.get_rivals(agent, limit=limit) if hasattr(elo, 'get_rivals') else []
            return json_response({"agent": agent, "rivals": rivals})
        except Exception as e:
            return error_response(f"Failed to get rivals: {e}", 500)

    def _get_allies(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top allies."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            allies = elo.get_allies(agent, limit=limit) if hasattr(elo, 'get_allies') else []
            return json_response({"agent": agent, "allies": allies})
        except Exception as e:
            return error_response(f"Failed to get allies: {e}", 500)

    def _get_moments(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's significant moments."""
        try:
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
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                        "debate_id": m.debate_id,
                    }
                    for m in moments
                ]
                return json_response({"agent": agent, "moments": moments_data})
            return json_response({"agent": agent, "moments": []})
        except Exception as e:
            return error_response(f"Failed to get moments: {e}", 500)

    def _get_positions(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's position history."""
        try:
            from aragora.agents.grounded import PositionLedger
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                ledger = PositionLedger(str(nomic_dir / "grounded_positions.db"))
                positions = ledger.get_agent_positions(agent, limit=limit)
                return json_response({"agent": agent, "positions": positions})
            return json_response({"agent": agent, "positions": []})
        except Exception as e:
            return error_response(f"Failed to get positions: {e}", 500)

    @ttl_cache(ttl_seconds=600, key_prefix="agent_h2h")
    def _get_head_to_head(self, agent: str, opponent: str) -> HandlerResult:
        """Get head-to-head stats between two agents."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        try:
            if hasattr(elo, 'get_head_to_head'):
                stats = elo.get_head_to_head(agent, opponent)
            else:
                stats = {"matches": 0, "agent1_wins": 0, "agent2_wins": 0}
            return json_response({
                "agent1": agent,
                "agent2": opponent,
                **stats,
            })
        except Exception as e:
            return error_response(f"Failed to get head-to-head: {e}", 500)
