"""
Leaderboard view endpoint handler.

Consolidates 6 separate leaderboard-related endpoints into a single request:
- GET /api/leaderboard-view - Returns all leaderboard data in one response

This reduces frontend latency by 80% (1 request instead of 6).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.config import (
    CACHE_TTL_LB_INTROSPECTION,
    CACHE_TTL_LB_MATCHES,
    CACHE_TTL_LB_RANKINGS,
    CACHE_TTL_LB_REPUTATION,
    CACHE_TTL_LB_STATS,
    CACHE_TTL_LB_TEAMS,
)
from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)

# Introspection limits
MAX_INTROSPECTION_AGENTS = 50  # Prevent unbounded introspection calls

from ..base import (
    SAFE_ID_PATTERN,
    HandlerResult,
    agent_to_dict,
    error_response,
    get_agent_name,
    get_int_param,
    get_string_param,
    json_response,
    ttl_cache,
    validate_path_segment,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip

# RBAC permission for leaderboard endpoints
LEADERBOARD_PERMISSION = "agents:read"

# Rate limiter for leaderboard endpoints (60 requests per minute - cached data)
_leaderboard_limiter = RateLimiter(requests_per_minute=60)


class LeaderboardViewHandler(SecureHandler):
    """Handler for consolidated leaderboard view endpoint.

    Requires authentication and agent:read permission (RBAC).
    """

    ROUTES = ["/api/leaderboard-view"]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return strip_version_prefix(path) == "/api/leaderboard-view"

    def _safe_fetch_section(
        self, data: dict, errors: dict, key: str, fetch_fn, fallback: dict
    ) -> None:
        """Safely fetch a leaderboard section with error handling and logging.

        Args:
            data: Dictionary to store successful results
            errors: Dictionary to store error messages
            key: Section name (e.g., "rankings", "matches")
            fetch_fn: Callable that fetches the section data
            fallback: Default value to use on error
        """
        try:
            data[key] = fetch_fn()
        except Exception as e:
            logger.error(
                "Leaderboard section '%s' failed: %s: %s", key, type(e).__name__, e, exc_info=True
            )
            errors[key] = str(e)
            data[key] = fallback

    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler
    ) -> Optional[HandlerResult]:
        """Route leaderboard view requests with RBAC."""
        path = strip_version_prefix(path)
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _leaderboard_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for leaderboard endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and agent:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, LEADERBOARD_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required to access leaderboard data", 401)
        except ForbiddenError as e:
            logger.warning(f"Leaderboard access denied: {e}")
            return error_response(str(e), 403)

        logger.debug(f"Leaderboard request: {path} params={query_params}")
        if path == "/api/leaderboard-view":
            limit = get_int_param(query_params, "limit", 10)
            domain = get_string_param(query_params, "domain")
            if domain:
                is_valid, err = validate_path_segment(domain, "domain", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            loop_id = get_string_param(query_params, "loop_id")
            if loop_id:
                is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
            return self._get_leaderboard_view(limit, domain, loop_id)
        return None

    def _get_leaderboard_view(
        self, limit: int, domain: Optional[str], loop_id: Optional[str]
    ) -> HandlerResult:
        """
        Get consolidated leaderboard view with all tab data.

        Returns all 6 data sources in a single response with per-section
        error handling for graceful degradation.
        """
        data: dict[str, Any] = {}
        errors: dict[str, str] = {}

        # Fetch all sections with graceful error handling
        self._safe_fetch_section(
            data,
            errors,
            "rankings",
            lambda: self._fetch_rankings(limit, domain),
            {"agents": [], "count": 0},
        )
        self._safe_fetch_section(
            data,
            errors,
            "matches",
            lambda: self._fetch_matches(limit, loop_id),
            {"matches": [], "count": 0},
        )
        self._safe_fetch_section(
            data, errors, "reputation", self._fetch_reputations, {"reputations": [], "count": 0}
        )
        self._safe_fetch_section(
            data,
            errors,
            "teams",
            lambda: self._fetch_teams(min_debates=3, limit=10),
            {"combinations": [], "count": 0},
        )
        self._safe_fetch_section(
            data,
            errors,
            "stats",
            self._fetch_stats,
            {
                "mean_elo": 1500,
                "median_elo": 1500,
                "total_agents": 0,
                "total_matches": 0,
                "rating_distribution": {},
                "trending_up": [],
                "trending_down": [],
            },
        )
        self._safe_fetch_section(
            data, errors, "introspection", self._fetch_introspection, {"agents": {}, "count": 0}
        )

        logger.info(
            f"Leaderboard view: {len(errors)} failed sections, {len(data.get('rankings', {}).get('agents', []))} agents"
        )
        return json_response(
            {
                "data": data,
                "errors": {
                    "partial_failure": len(errors) > 0,
                    "failed_sections": list(errors.keys()),
                    "messages": errors,
                },
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_LB_RANKINGS, key_prefix="lb_rankings", skip_first=True)
    def _fetch_rankings(self, limit: int, domain: Optional[str]) -> dict:
        """Fetch agent rankings with consistency scores."""
        elo = self.get_elo_system()
        if not elo:
            return {"agents": [], "count": 0}

        # Get base rankings
        if domain:
            rankings = elo.get_leaderboard(limit=min(limit, 50), domain=domain)
        else:
            rankings = (
                elo.get_cached_leaderboard(limit=min(limit, 50))
                if hasattr(elo, "get_cached_leaderboard")
                else elo.get_leaderboard(limit=min(limit, 50))
            )

        # Enhance with consistency data (batch fetch)
        consistency_map = {}
        try:
            from aragora.insights.flip_detector import FlipDetector

            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                detector = FlipDetector(str(get_db_path(DatabaseType.POSITIONS, nomic_dir)))
                agent_names = []
                for agent in rankings:
                    name = get_agent_name(agent)
                    if name:
                        agent_names.append(name)

                if agent_names:
                    scores = detector.get_agents_consistency_batch(agent_names)
                    for agent_name, score in scores.items():
                        total_positions = max(score.total_positions, 1)
                        consistency = 1.0 - (score.total_flips / total_positions)
                        consistency_map[agent_name] = {
                            "consistency": round(consistency, 3),
                            "consistency_class": (
                                "high"
                                if consistency >= 0.8
                                else "medium"
                                if consistency >= 0.6
                                else "low"
                            ),
                        }
        except ImportError:
            pass

        # Build response
        enhanced = []
        for agent in rankings:
            agent_dict = agent_to_dict(agent)
            agent_name = agent_dict.get("name")

            if agent_name in consistency_map:
                agent_dict.update(consistency_map[agent_name])
            enhanced.append(agent_dict)

        return {"agents": enhanced, "count": len(enhanced)}

    @ttl_cache(ttl_seconds=CACHE_TTL_LB_MATCHES, key_prefix="lb_matches", skip_first=True)
    def _fetch_matches(self, limit: int, loop_id: Optional[str]) -> dict:
        """Fetch recent matches."""
        elo = self.get_elo_system()
        if not elo:
            return {"matches": [], "count": 0}

        if hasattr(elo, "get_cached_recent_matches"):
            matches = elo.get_cached_recent_matches(limit=min(limit, 50))
        else:
            matches = elo.get_recent_matches(limit=min(limit, 50))

        return {"matches": matches, "count": len(matches)}

    @ttl_cache(ttl_seconds=CACHE_TTL_LB_REPUTATION, key_prefix="lb_reputation", skip_first=True)
    def _fetch_reputations(self) -> dict:
        """Fetch all agent reputations."""
        try:
            from aragora.memory.store import CritiqueStore
        except ImportError:
            return {"reputations": [], "count": 0}

        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return {"reputations": [], "count": 0}
        db_path = get_db_path(DatabaseType.AGORA_MEMORY, nomic_dir)
        if not db_path.exists():
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

    @ttl_cache(ttl_seconds=CACHE_TTL_LB_TEAMS, key_prefix="lb_teams", skip_first=True)
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

    @ttl_cache(ttl_seconds=CACHE_TTL_LB_STATS, key_prefix="lb_stats", skip_first=True)
    def _fetch_stats(self) -> dict:
        """Fetch ranking statistics."""
        elo = self.get_elo_system()
        if not elo:
            return {
                "mean_elo": 1500,
                "median_elo": 1500,
                "total_agents": 0,
                "total_matches": 0,
                "rating_distribution": {},
                "trending_up": [],
                "trending_down": [],
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

    @ttl_cache(
        ttl_seconds=CACHE_TTL_LB_INTROSPECTION, key_prefix="lb_introspection", skip_first=True
    )
    def _fetch_introspection(self) -> dict:
        """Fetch agent introspection data."""
        try:
            from aragora.introspection import get_agent_introspection
            from aragora.memory.store import CritiqueStore
        except ImportError:
            return {"agents": {}, "count": 0}

        nomic_dir = self.get_nomic_dir()

        # Get known agents from reputation store
        agents = []
        memory = None
        if nomic_dir:
            db_path = get_db_path(DatabaseType.AGORA_MEMORY, nomic_dir)
            if db_path.exists():
                memory = CritiqueStore(str(db_path))
                reputations = memory.get_all_reputations()
                agents = [r.agent_name for r in reputations]

        if not agents:
            agents = ["gemini", "claude", "codex", "grok", "deepseek"]

        # Limit agents for introspection to prevent unbounded calls
        agents = agents[:MAX_INTROSPECTION_AGENTS]

        # Get persona manager if available
        persona_manager = None
        try:
            from aragora.agents.personas import PersonaManager

            if nomic_dir:
                persona_db = get_db_path(DatabaseType.PERSONAS, nomic_dir)
                if persona_db.exists():
                    persona_manager = PersonaManager(str(persona_db))
        except ImportError:
            pass

        snapshots = {}
        for agent in agents:
            try:
                snapshot = get_agent_introspection(
                    agent, memory=memory, persona_manager=persona_manager
                )
                snapshots[agent] = snapshot.to_dict()
            except Exception as e:
                # Skip agents that fail introspection
                logger.warning(
                    "Agent introspection failed for %s: %s: %s", agent, type(e).__name__, e
                )
                continue

        return {"agents": snapshots, "count": len(snapshots)}
