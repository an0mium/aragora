"""
Leaderboard view endpoint handler.

Consolidates 6 separate leaderboard-related endpoints into a single request:
- GET /api/leaderboard-view - Returns all leaderboard data in one response

This reduces frontend latency by 80% (1 request instead of 6).
"""

from typing import Optional
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
)


class LeaderboardViewHandler(BaseHandler):
    """Handler for consolidated leaderboard view endpoint."""

    ROUTES = ["/api/leaderboard-view"]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path == "/api/leaderboard-view"

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route leaderboard view requests."""
        if path == "/api/leaderboard-view":
            limit = get_int_param(query_params, 'limit', 10)
            domain = query_params.get('domain')
            loop_id = query_params.get('loop_id')
            return self._get_leaderboard_view(limit, domain, loop_id)
        return None

    def _get_leaderboard_view(self, limit: int, domain: Optional[str], loop_id: Optional[str]) -> HandlerResult:
        """
        Get consolidated leaderboard view with all tab data.

        Returns all 6 data sources in a single response with per-section
        error handling for graceful degradation.
        """
        data = {}
        errors = {}

        # 1. Rankings (with consistency from FlipDetector)
        try:
            data["rankings"] = self._fetch_rankings(limit, domain)
        except Exception as e:
            errors["rankings"] = str(e)
            data["rankings"] = {"agents": [], "count": 0}

        # 2. Recent matches
        try:
            data["matches"] = self._fetch_matches(limit, loop_id)
        except Exception as e:
            errors["matches"] = str(e)
            data["matches"] = {"matches": [], "count": 0}

        # 3. Reputation
        try:
            data["reputation"] = self._fetch_reputations()
        except Exception as e:
            errors["reputation"] = str(e)
            data["reputation"] = {"reputations": [], "count": 0}

        # 4. Best teams
        try:
            data["teams"] = self._fetch_teams(min_debates=3, limit=10)
        except Exception as e:
            errors["teams"] = str(e)
            data["teams"] = {"combinations": [], "count": 0}

        # 5. Ranking stats
        try:
            data["stats"] = self._fetch_stats()
        except Exception as e:
            errors["stats"] = str(e)
            data["stats"] = {
                "mean_elo": 1500, "median_elo": 1500, "total_agents": 0,
                "total_matches": 0, "rating_distribution": {},
                "trending_up": [], "trending_down": []
            }

        # 6. Introspection (agent self-models)
        try:
            data["introspection"] = self._fetch_introspection()
        except Exception as e:
            errors["introspection"] = str(e)
            data["introspection"] = {"agents": {}, "count": 0}

        return json_response({
            "data": data,
            "errors": {
                "partial_failure": len(errors) > 0,
                "failed_sections": list(errors.keys()),
                "messages": errors,
            }
        })

    @ttl_cache(ttl_seconds=300, key_prefix="lb_rankings", skip_first=True)
    def _fetch_rankings(self, limit: int, domain: Optional[str]) -> dict:
        """Fetch agent rankings with consistency scores."""
        elo = self.get_elo_system()
        if not elo:
            return {"agents": [], "count": 0}

        # Get base rankings
        if domain:
            rankings = elo.get_leaderboard(limit=min(limit, 50), domain=domain)
        else:
            rankings = elo.get_cached_leaderboard(limit=min(limit, 50)) if hasattr(elo, 'get_cached_leaderboard') else elo.get_leaderboard(limit=min(limit, 50))

        # Enhance with consistency data (batch fetch)
        consistency_map = {}
        try:
            from aragora.insights.flip_detector import FlipDetector
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                detector = FlipDetector(str(nomic_dir / "grounded_positions.db"))
                agent_names = []
                for agent in rankings:
                    name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)
                    if name:
                        agent_names.append(name)

                if agent_names:
                    scores = detector.get_agents_consistency_batch(agent_names)
                    for agent_name, score in scores.items():
                        total_positions = max(score.total_positions, 1)
                        consistency = 1.0 - (score.total_flips / total_positions)
                        consistency_map[agent_name] = {
                            "consistency": round(consistency, 3),
                            "consistency_class": "high" if consistency >= 0.8 else "medium" if consistency >= 0.6 else "low"
                        }
        except ImportError:
            pass

        # Build response
        enhanced = []
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
            enhanced.append(agent_dict)

        return {"agents": enhanced, "count": len(enhanced)}

    @ttl_cache(ttl_seconds=120, key_prefix="lb_matches", skip_first=True)
    def _fetch_matches(self, limit: int, loop_id: Optional[str]) -> dict:
        """Fetch recent matches."""
        elo = self.get_elo_system()
        if not elo:
            return {"matches": [], "count": 0}

        if hasattr(elo, 'get_cached_recent_matches'):
            matches = elo.get_cached_recent_matches(limit=min(limit, 50))
        else:
            matches = elo.get_recent_matches(limit=min(limit, 50))

        return {"matches": matches, "count": len(matches)}

    @ttl_cache(ttl_seconds=300, key_prefix="lb_reputation", skip_first=True)
    def _fetch_reputations(self) -> dict:
        """Fetch all agent reputations."""
        try:
            from aragora.memory.critique_store import CritiqueStore
        except ImportError:
            return {"reputations": [], "count": 0}

        nomic_dir = self.get_nomic_dir()
        db_path = nomic_dir / "debates.db" if nomic_dir else None
        if not db_path or not db_path.exists():
            return {"reputations": [], "count": 0}

        store = CritiqueStore(str(db_path))
        reputations = store.get_all_reputations()

        return {
            "reputations": [
                {
                    "agent": r.agent_name,
                    "score": r.reputation_score,
                    "vote_weight": r.vote_weight,
                    "proposal_acceptance_rate": r.proposal_acceptance_rate,
                    "critique_value": r.critique_value,
                    "debates_participated": r.debates_participated,
                }
                for r in reputations
            ],
            "count": len(reputations),
        }

    @ttl_cache(ttl_seconds=600, key_prefix="lb_teams", skip_first=True)
    def _fetch_teams(self, min_debates: int, limit: int) -> dict:
        """Fetch best team combinations."""
        try:
            from aragora.routing.selection import AgentSelector
        except ImportError:
            return {"combinations": [], "count": 0}

        elo = self.get_elo_system()
        selector = AgentSelector(elo_system=elo, persona_manager=None)
        combinations = selector.get_best_team_combinations(min_debates=min_debates)[:limit]

        return {"combinations": combinations, "count": len(combinations)}

    @ttl_cache(ttl_seconds=900, key_prefix="lb_stats", skip_first=True)
    def _fetch_stats(self) -> dict:
        """Fetch ranking statistics."""
        elo = self.get_elo_system()
        if not elo:
            return {
                "mean_elo": 1500, "median_elo": 1500, "total_agents": 0,
                "total_matches": 0, "rating_distribution": {},
                "trending_up": [], "trending_down": []
            }

        stats = elo.get_stats()
        # Ensure all expected fields are present
        return {
            "mean_elo": stats.get("avg_elo", stats.get("mean_elo", 1500)),
            "median_elo": stats.get("median_elo", 1500),
            "total_agents": stats.get("total_agents", 0),
            "total_matches": stats.get("total_matches", 0),
            "rating_distribution": stats.get("rating_distribution", {}),
            "trending_up": stats.get("trending_up", []),
            "trending_down": stats.get("trending_down", []),
        }

    @ttl_cache(ttl_seconds=600, key_prefix="lb_introspection", skip_first=True)
    def _fetch_introspection(self) -> dict:
        """Fetch agent introspection data."""
        try:
            from aragora.introspection import get_agent_introspection
            from aragora.memory.critique_store import CritiqueStore
        except ImportError:
            return {"agents": {}, "count": 0}

        nomic_dir = self.get_nomic_dir()

        # Get known agents from reputation store
        agents = []
        memory = None
        db_path = nomic_dir / "debates.db" if nomic_dir else None
        if db_path and db_path.exists():
            memory = CritiqueStore(str(db_path))
            reputations = memory.get_all_reputations()
            agents = [r.agent_name for r in reputations]

        if not agents:
            agents = ["gemini", "claude", "codex", "grok", "deepseek"]

        # Get persona manager if available
        persona_manager = None
        try:
            from aragora.agents.personas import PersonaManager
            persona_db = nomic_dir / "personas.db" if nomic_dir else None
            if persona_db and persona_db.exists():
                persona_manager = PersonaManager(str(persona_db))
        except ImportError:
            pass

        snapshots = {}
        for agent in agents:
            try:
                snapshot = get_agent_introspection(agent, memory=memory, persona_manager=persona_manager)
                snapshots[agent] = snapshot.to_dict()
            except Exception:
                # Skip agents that fail introspection
                continue

        return {"agents": snapshots, "count": len(snapshots)}
