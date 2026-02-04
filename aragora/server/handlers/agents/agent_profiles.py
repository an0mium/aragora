"""Agent profile and per-agent stat endpoint methods (AgentProfilesMixin).

Extracted from agents.py to reduce file size.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.config import (
    CACHE_TTL_AGENT_PROFILE,
    ELO_INITIAL_RATING,
)
from aragora.persistence.db_config import DatabaseType, get_db_path

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint

logger = logging.getLogger(__name__)


class AgentProfilesMixin:
    """Mixin providing agent profile and per-agent stat endpoints."""

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/profile",
        summary="Get complete agent profile",
        tags=["Agents"],
    )
    @ttl_cache(ttl_seconds=CACHE_TTL_AGENT_PROFILE, key_prefix="agent_profile", skip_first=True)
    @handle_errors("agent profile")
    def _get_profile(self, agent: str) -> HandlerResult:
        """Get complete agent profile."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rating = elo.get_rating(agent) or ELO_INITIAL_RATING
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/history",
        summary="Get agent match history",
        tags=["Agents"],
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/calibration",
        summary="Get agent calibration scores",
        tags=["Agents"],
    )
    @handle_errors("agent calibration")
    def _get_calibration(self, agent: str, domain: str | None) -> HandlerResult:
        """Get agent calibration scores."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        if hasattr(elo, "get_calibration"):
            calibration = elo.get_calibration(agent, domain=domain)
        else:
            calibration = {"agent": agent, "score": 0.5}
        return json_response(calibration)

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/consistency",
        summary="Get agent consistency score",
        tags=["Agents"],
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/network",
        summary="Get agent relationship network",
        tags=["Agents"],
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/rivals",
        summary="Get agent top rivals",
        tags=["Agents"],
    )
    @handle_errors("agent rivals")
    def _get_rivals(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top rivals."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        rivals = elo.get_rivals(agent, limit=limit) if hasattr(elo, "get_rivals") else []
        return json_response({"agent": agent, "rivals": rivals})

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/allies",
        summary="Get agent top allies",
        tags=["Agents"],
    )
    @handle_errors("agent allies")
    def _get_allies(self, agent: str, limit: int) -> HandlerResult:
        """Get agent's top allies."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("ELO system not available", 503)

        allies = elo.get_allies(agent, limit=limit) if hasattr(elo, "get_allies") else []
        return json_response({"agent": agent, "allies": allies})

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/moments",
        summary="Get agent significant moments",
        tags=["Agents"],
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/positions",
        summary="Get agent position history",
        tags=["Agents"],
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/domains",
        summary="Get agent domain-specific ELO ratings",
        tags=["Agents"],
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/agent/{name}/performance",
        summary="Get detailed agent performance statistics",
        tags=["Agents"],
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
