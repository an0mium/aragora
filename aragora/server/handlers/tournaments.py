"""
Tournament-related endpoint handlers.

Endpoints:
- GET /api/tournaments - List all tournaments
- GET /api/tournaments/{id}/standings - Get tournament standings
"""

import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
)

# Optional import for tournament functionality
try:
    from aragora.ranking.tournaments import TournamentManager
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False
    TournamentManager = None


class TournamentHandler(BaseHandler):
    """Handler for tournament-related endpoints."""

    ROUTES = [
        "/api/tournaments",
        "/api/tournaments/*/standings",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path == "/api/tournaments":
            return True
        if path.startswith("/api/tournaments/") and path.endswith("/standings"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route tournament requests to appropriate handler methods."""
        if path == "/api/tournaments":
            return self._list_tournaments()

        # Match /api/tournaments/{id}/standings
        if path.startswith("/api/tournaments/") and path.endswith("/standings"):
            tournament_id, err = self.extract_path_param(path, 2, "tournament_id")
            if err:
                return err
            return self._get_tournament_standings(tournament_id)

        return None

    def _list_tournaments(self) -> HandlerResult:
        """List all available tournaments."""
        if not TOURNAMENT_AVAILABLE:
            return error_response("Tournament system not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            tournaments_dir = Path(nomic_dir) / "tournaments"
            tournaments = []

            if tournaments_dir.exists():
                for db_file in tournaments_dir.glob("*.db"):
                    tournament_id = db_file.stem
                    try:
                        manager = TournamentManager(db_path=str(db_file))
                        standings = manager.get_current_standings()
                        tournaments.append({
                            "tournament_id": tournament_id,
                            "participants": len(standings),
                            "total_matches": sum(s.wins + s.losses + s.draws for s in standings) // 2,
                            "top_agent": standings[0].agent if standings else None,
                        })
                    except Exception as e:
                        # Skip corrupted or invalid tournament files
                        logger.warning("Failed to load tournament %s: %s: %s", tournament_id, type(e).__name__, e)
                        continue

            return json_response({
                "tournaments": tournaments,
                "count": len(tournaments),
            })
        except Exception as e:
            return error_response(f"Failed to list tournaments: {e}", 500)

    def _get_tournament_standings(self, tournament_id: str) -> HandlerResult:
        """Get current tournament standings."""
        if not TOURNAMENT_AVAILABLE:
            return error_response("Tournament system not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            tournament_path = Path(nomic_dir) / "tournaments" / f"{tournament_id}.db"
            if not tournament_path.exists():
                return error_response("Tournament not found", 404)

            manager = TournamentManager(db_path=str(tournament_path))
            standings = manager.get_current_standings()

            return json_response({
                "tournament_id": tournament_id,
                "standings": [
                    {
                        "agent": s.agent,
                        "wins": s.wins,
                        "losses": s.losses,
                        "draws": s.draws,
                        "points": s.points,
                        "total_score": s.total_score,
                        "win_rate": s.win_rate,
                    }
                    for s in standings
                ],
                "count": len(standings),
            })
        except Exception as e:
            return error_response(f"Failed to get tournament standings: {e}", 500)
