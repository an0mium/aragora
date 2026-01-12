"""
Tournament-related endpoint handlers.

Endpoints:
- GET /api/tournaments - List all tournaments
- GET /api/tournaments/{id}/standings - Get tournament standings
"""

from __future__ import annotations

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
    handle_errors,
)
from .utils.rate_limit import RateLimiter, get_client_ip

# Rate limiter for tournament endpoints (30 requests per minute)
_tournament_limiter = RateLimiter(requests_per_minute=30)

# Tournament listing limits
MAX_TOURNAMENTS_TO_LIST = 100  # Prevent unbounded directory iteration

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
        logger.debug(f"Tournament request: {path}")

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _tournament_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for tournament endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/tournaments":
            return self._list_tournaments()

        # Match /api/tournaments/{id}/standings
        if path.startswith("/api/tournaments/") and path.endswith("/standings"):
            tournament_id, err = self.extract_path_param(path, 2, "tournament_id")
            if err:
                return err
            return self._get_tournament_standings(tournament_id)

        return None

    @handle_errors("tournament listing")
    def _list_tournaments(self) -> HandlerResult:
        """List all available tournaments."""
        if not TOURNAMENT_AVAILABLE:
            return error_response("Tournament system not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        tournaments_dir = Path(nomic_dir) / "tournaments"
        tournaments = []

        if tournaments_dir.exists():
            scanned = 0
            for db_file in tournaments_dir.glob("*.db"):
                # Limit directory iteration to prevent unbounded growth
                if scanned >= MAX_TOURNAMENTS_TO_LIST:
                    break
                scanned += 1

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

        logger.info(f"Listed {len(tournaments)} tournaments")
        return json_response({
            "tournaments": tournaments,
            "count": len(tournaments),
        })

    @handle_errors("tournament standings retrieval")
    def _get_tournament_standings(self, tournament_id: str) -> HandlerResult:
        """Get current tournament standings."""
        if not TOURNAMENT_AVAILABLE:
            return error_response("Tournament system not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        tournament_path = Path(nomic_dir) / "tournaments" / f"{tournament_id}.db"
        if not tournament_path.exists():
            return error_response("Tournament not found", 404)

        manager = TournamentManager(db_path=str(tournament_path))
        standings = manager.get_current_standings()

        logger.info(f"Retrieved standings for tournament {tournament_id}: {len(standings)} participants")
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
