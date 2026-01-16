"""
ELO Snapshot Engine for fast reads.

Extracted from EloSystem to separate snapshot/persistence concerns.
Provides JSON snapshot files for fast reads that avoid SQLite locking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aragora.ranking.elo import AgentRating

logger = logging.getLogger(__name__)


def write_snapshot(
    snapshot_path: Path,
    leaderboard_getter: Callable[[int], list["AgentRating"]],
    matches_getter: Callable[[int], list[dict]],
    leaderboard_limit: int = 100,
    matches_limit: int = 50,
) -> None:
    """Write JSON snapshot for fast reads.

    Creates an atomic JSON file with current leaderboard and recent matches.
    This avoids SQLite locking issues when multiple readers access data.

    Args:
        snapshot_path: Path to write the snapshot file
        leaderboard_getter: Function to get leaderboard data
        matches_getter: Function to get recent matches
        leaderboard_limit: Max leaderboard entries
        matches_limit: Max match entries
    """
    # Gather current state
    leaderboard = leaderboard_getter(leaderboard_limit)
    data = {
        "leaderboard": [
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
        ],
        "recent_matches": matches_getter(matches_limit),
        "updated_at": datetime.now().isoformat(),
    }

    # Atomic write: write to temp file then rename
    temp_path = snapshot_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f)
        temp_path.rename(snapshot_path)
    except Exception as e:
        # Snapshot is optional, don't fail on write errors
        logger.debug(f"Failed to write ELO snapshot: {e}")
        if temp_path.exists():
            temp_path.unlink()


def read_snapshot_leaderboard(
    snapshot_path: Path,
    limit: int = 20,
) -> list[dict[str, Any]] | None:
    """Read leaderboard from JSON snapshot file.

    Args:
        snapshot_path: Path to the snapshot file
        limit: Maximum number of entries to return

    Returns:
        List of leaderboard dicts or None if snapshot unavailable
    """
    if not snapshot_path.exists():
        return None

    try:
        with open(snapshot_path) as f:
            data = json.load(f)
        return data.get("leaderboard", [])[:limit]
    except json.JSONDecodeError as e:
        logger.debug(f"ELO snapshot corrupted: {e}")
        return None
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot read ELO snapshot (I/O error): {e}")
        return None


def read_snapshot_matches(
    snapshot_path: Path,
    limit: int = 10,
) -> list[dict[str, Any]] | None:
    """Read recent matches from JSON snapshot file.

    Args:
        snapshot_path: Path to the snapshot file
        limit: Maximum number of matches to return

    Returns:
        List of match dicts or None if snapshot unavailable
    """
    if not snapshot_path.exists():
        return None

    try:
        with open(snapshot_path) as f:
            data = json.load(f)
        return data.get("recent_matches", [])[:limit]
    except json.JSONDecodeError as e:
        logger.debug(f"Recent matches snapshot corrupted: {e}")
        return None
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot read recent matches snapshot (I/O error): {e}")
        return None


__all__ = [
    "write_snapshot",
    "read_snapshot_leaderboard",
    "read_snapshot_matches",
]
