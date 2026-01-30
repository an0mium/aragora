"""
Leaderboard and Snapshot Access.

Extends LeaderboardEngine with snapshot-based fast access methods.
These methods provide cached reads from JSON snapshot files to avoid
SQLite locking in high-read scenarios.

For the core leaderboard engine, see leaderboard_engine.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from aragora.ranking.snapshot import (
    read_snapshot_leaderboard,
    read_snapshot_matches,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def get_snapshot_leaderboard(
    db_path: Path,
    get_leaderboard: Callable,
    limit: int = 20,
) -> list[dict]:
    """Get leaderboard from JSON snapshot file.

    Falls back to database query if snapshot is unavailable.

    Args:
        db_path: Path to the database (used to locate snapshot file)
        get_leaderboard: Callback to get leaderboard from database
        limit: Maximum entries to return

    Returns:
        List of leaderboard entry dicts
    """
    snapshot_path = db_path.parent / "elo_snapshot.json"
    result = read_snapshot_leaderboard(snapshot_path, limit)
    if result is not None:
        return result
    # Fall back to database
    leaderboard = get_leaderboard(limit)
    return [
        {
            "agent_name": r.agent_name,
            "elo": r.elo,
            "wins": r.wins,
            "losses": r.losses,
            "draws": r.draws,
            "games_played": r.games_played,
            "win_rate": r.win_rate,
        }
        for r in leaderboard
    ]


def get_cached_recent_matches(
    db_path: Path,
    get_recent_matches: Callable,
    limit: int = 10,
) -> list[dict]:
    """Get recent matches from cache if available.

    Falls back to database query if snapshot is unavailable.

    Args:
        db_path: Path to the database (used to locate snapshot file)
        get_recent_matches: Callback to get recent matches from database
        limit: Maximum matches to return

    Returns:
        List of match dicts
    """
    snapshot_path = db_path.parent / "elo_snapshot.json"
    result = read_snapshot_matches(snapshot_path, limit)
    if result is not None:
        return result
    return get_recent_matches(limit)


__all__ = [
    "get_snapshot_leaderboard",
    "get_cached_recent_matches",
]
