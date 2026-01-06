"""
Agent-related endpoint handlers.

Endpoints:
- GET /api/leaderboard - Get agent rankings
- GET /api/rankings - Get agent rankings (alias)
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
- GET /api/agent/compare - Compare multiple agents
- GET /api/agent/{name}/head-to-head/{opponent} - Get head-to-head stats
- GET /api/flips/recent - Get recent flips across all agents
- GET /api/flips/summary - Get flip summary for dashboard
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_string_param,
    ttl_cache,
    handle_errors,
    validate_agent_name,
    validate_path_segment,
    SAFE_ID_PATTERN,
)


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
        "/api/agent/*/flips",
        "/api/agent/*/network",
        "/api/agent/*/rivals",
        "/api/agent/*/allies",
        "/api/agent/*/moments",
        "/api/agent/*/positions",
        "/api/agent/*/head-to-head/*",
        "/api/agent/*/opponent-briefing/*",
        "/api/flips/recent",
        "/api/flips/summary",
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
        if path.startswith("/api/flips/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route agent requests to appropriate methods."""
        # Leaderboard endpoints
        if path in ("/api/leaderboard", "/api/rankings"):
            limit = get_int_param(query_params, 'limit', 20)
            domain = get_string_param(query_params, 'domain')
            if domain:
                is_valid, err = validate_path_segment(domain, "domain", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._get_leaderboard(limit, domain)

        if path == "/api/calibration/leaderboard":
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_calibration_leaderboard(limit)

        if path == "/api/matches/recent":
            limit = get_int_param(query_params, 'limit', 10)
            loop_id = get_string_param(query_params, 'loop_id')
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
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

        # Flip endpoints (not per-agent)
        if path == "/api/flips/recent":
            limit = get_int_param(query_params, 'limit', 20)
            return self._get_recent_flips(limit)

        if path == "/api/flips/summary":
            return self._get_flip_summary()

        return None

    def _handle_agent_endpoint(self, path: str, query_params: dict) -> Optional[HandlerResult]:
        """Handle /api/agent/{name}/* endpoints."""
        parts = path.split("/")
        if len(parts) < 4:
            return error_response("Invalid agent path", 400)

        agent_name = parts[3]

        # Validate agent name
        is_valid, err = validate_agent_name(agent_name)
        if not is_valid:
            return error_response(err, 400)

        # Head-to-head: /api/agent/{name}/head-to-head/{opponent}
        if len(parts) >= 6 and parts[4] == "head-to-head":
            opponent = parts[5]
            # Validate opponent name
            is_valid, err = validate_agent_name(opponent)
            if not is_valid:
                return error_response(err, 400)
            return self._get_head_to_head(agent_name, opponent)

        # Opponent briefing: /api/agent/{name}/opponent-briefing/{opponent}
        if len(parts) >= 6 and parts[4] == "opponent-briefing":
            opponent = parts[5]
            # Validate opponent name
            is_valid, err = validate_agent_name(opponent)
            if not is_valid:
                return error_response(err, 400)
            return self._get_opponent_briefing(agent_name, opponent)

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
            "flips": lambda: self._get_agent_flips(agent, get_int_param(params, 'limit', 20)),
            "network": lambda: self._get_network(agent),
            "rivals": lambda: self._get_rivals(agent, get_int_param(params, 'limit', 5)),
            "allies": lambda: self._get_allies(agent, get_int_param(params, 'limit', 5)),
            "moments": lambda: self._get_moments(agent, get_int_param(params, 'limit', 10)),
            "positions": lambda: self._get_positions(agent, get_int_param(params, 'limit', 20)),
        }

        if endpoint in handlers:
            return handlers[endpoint]()

        return None

    @ttl_cache(ttl_seconds=300, key_prefix="leaderboard", skip_first=True)
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
                    detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
                    # Extract all agent names first
                    agent_names = []
                    for agent in rankings:
                        name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)
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
                                    "consistency_class": "high" if consistency >= 0.8 else "medium" if consistency >= 0.6 else "low"
                                }
                        except Exception as e:
                            # Fallback: set default consistency for all agents
                            logger.warning("Consistency detection failed, using fallback: %s: %s", type(e).__name__, e)
                            for name in agent_names:
                                consistency_map[name] = {
                                    "consistency": 1.0,
                                    "consistency_class": "high",
                                    "degraded": True,
                                    "degraded_reason": "Consistency detection unavailable"
                                }
            except ImportError:
                # FlipDetector not available - set degraded flag so clients know
                logger.debug("FlipDetector import failed - consistency scores unavailable")
                for agent in rankings:
                    name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)
                    if name:
                        consistency_map[name] = {
                            "consistency": None,
                            "consistency_class": "unknown",
                            "degraded": True,
                            "degraded_reason": "FlipDetector module not available"
                        }

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

    @ttl_cache(ttl_seconds=300, key_prefix="calibration_lb", skip_first=True)
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

    @ttl_cache(ttl_seconds=120, key_prefix="recent_matches", skip_first=True)
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
                except Exception as e:
                    logger.warning("Head-to-head lookup failed for %s vs %s: %s: %s", agents[0], agents[1], type(e).__name__, e)

            return json_response({
                "agents": profiles,
                "head_to_head": head_to_head,
            })
        except Exception as e:
            return error_response(f"Comparison failed: {e}", 500)

    @ttl_cache(ttl_seconds=600, key_prefix="agent_profile", skip_first=True)
    @handle_errors("agent profile")
    def _get_profile(self, agent: str) -> HandlerResult:
        """Get complete agent profile."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

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

    @handle_errors("agent history")
    def _get_history(self, agent: str, limit: int) -> HandlerResult:
        """Get agent match history."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        history = elo.get_agent_history(agent, limit=min(limit, 100))
        return json_response({"agent": agent, "history": history})

    @handle_errors("agent calibration")
    def _get_calibration(self, agent: str, domain: Optional[str]) -> HandlerResult:
        """Get agent calibration scores."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, 'get_calibration'):
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
            detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
            score = detector.get_agent_consistency(agent)
            return json_response({"agent": agent, "consistency_score": score})
        return json_response({"agent": agent, "consistency_score": 1.0})

    @handle_errors("agent network")
    def _get_network(self, agent: str) -> HandlerResult:
        """Get agent relationship network."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rivals = elo.get_rivals(agent, limit=5) if hasattr(elo, 'get_rivals') else []
        allies = elo.get_allies(agent, limit=5) if hasattr(elo, 'get_allies') else []
        return json_response({
            "agent": agent,
            "rivals": rivals,
            "allies": allies,
        })

    @handle_errors("agent rivals")
    def _get_rivals(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top rivals."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rivals = elo.get_rivals(agent, limit=limit) if hasattr(elo, 'get_rivals') else []
        return json_response({"agent": agent, "rivals": rivals})

    @handle_errors("agent allies")
    def _get_allies(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top allies."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        allies = elo.get_allies(agent, limit=limit) if hasattr(elo, 'get_allies') else []
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
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
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
            ledger = PositionLedger(str(nomic_dir / "grounded_positions.db"))
            positions = ledger.get_agent_positions(agent, limit=limit)
            return json_response({"agent": agent, "positions": positions})
        return json_response({"agent": agent, "positions": []})

    @ttl_cache(ttl_seconds=600, key_prefix="agent_h2h", skip_first=True)
    @handle_errors("head-to-head stats")
    def _get_head_to_head(self, agent: str, opponent: str) -> HandlerResult:
        """Get head-to-head stats between two agents."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, 'get_head_to_head'):
            stats = elo.get_head_to_head(agent, opponent)
        else:
            stats = {"matches": 0, "agent1_wins": 0, "agent2_wins": 0}
        return json_response({
            "agent1": agent,
            "agent2": opponent,
            **stats,
        })

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
                db_path = nomic_dir / "grounded_positions.db"
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

        synthesizer = PersonaSynthesizer(
            elo_system=elo,
            calibration_tracker=calibration_tracker,
            position_ledger=position_ledger,
        )
        briefing = synthesizer.get_opponent_briefing(agent, opponent)

        if briefing:
            return json_response({
                "agent": agent,
                "opponent": opponent,
                "briefing": briefing,
            })
        else:
            return json_response({
                "agent": agent,
                "opponent": opponent,
                "briefing": None,
                "message": "No opponent data available"
            })

    # ==================== Flip Detector Endpoints ====================

    @ttl_cache(ttl_seconds=300, key_prefix="agent_flips", skip_first=True)
    @handle_errors("agent flips")
    def _get_agent_flips(self, agent: str, limit: int) -> HandlerResult:
        """Get recent position flips for an agent."""
        from aragora.insights.flip_detector import FlipDetector
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
            flips = detector.detect_flips_for_agent(agent, lookback_positions=min(limit, 100))
            consistency = detector.get_agent_consistency(agent)
            return json_response({
                "agent": agent,
                "flips": [f.to_dict() for f in flips],
                "consistency": consistency.to_dict(),
                "count": len(flips),
            })
        return json_response({
            "agent": agent,
            "flips": [],
            "consistency": {"agent_name": agent, "total_positions": 0, "total_flips": 0, "consistency_score": 1.0},
            "count": 0,
        })

    @ttl_cache(ttl_seconds=300, key_prefix="flips_recent", skip_first=True)
    @handle_errors("recent flips")
    def _get_recent_flips(self, limit: int) -> HandlerResult:
        """Get recent flips across all agents."""
        from aragora.insights.flip_detector import FlipDetector
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
            flips = detector.get_recent_flips(limit=min(limit, 100))
            summary = detector.get_flip_summary()
            return json_response({
                "flips": [f.to_dict() for f in flips],
                "summary": summary,
                "count": len(flips),
            })
        return json_response({
            "flips": [],
            "summary": {"total_flips": 0, "by_type": {}, "by_agent": {}, "recent_24h": 0},
            "count": 0,
        })

    @ttl_cache(ttl_seconds=600, key_prefix="flips_summary", skip_first=True)
    @handle_errors("flip summary")
    def _get_flip_summary(self) -> HandlerResult:
        """Get flip summary for dashboard."""
        from aragora.insights.flip_detector import FlipDetector
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
            summary = detector.get_flip_summary()
            return json_response(summary)
        return json_response({
            "total_flips": 0,
            "by_type": {},
            "by_agent": {},
            "recent_24h": 0,
        })
