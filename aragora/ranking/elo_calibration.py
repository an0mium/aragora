"""
Calibration Leaderboard and History Queries.

Extracted from EloSystem to separate calibration-specific query operations.
These are database queries for calibration leaderboards and prediction history
that were inline in EloSystem.

For calibration recording and scoring logic, see calibration_engine.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.config import ELO_CALIBRATION_MIN_COUNT
from aragora.utils.json_helpers import safe_json_loads

if TYPE_CHECKING:
    from aragora.ranking.database import EloDatabase
    from aragora.ranking.elo import AgentRating
    from aragora.utils.cache import TTLCache

logger = logging.getLogger(__name__)

CALIBRATION_MIN_COUNT = ELO_CALIBRATION_MIN_COUNT


def get_calibration_leaderboard(
    db: EloDatabase,
    calibration_cache: TTLCache,
    limit: int = 20,
    use_cache: bool = True,
) -> list[AgentRating]:
    """
    Get agents ranked by calibration score.

    Only includes agents with minimum predictions.

    Args:
        db: EloDatabase instance for queries
        calibration_cache: TTLCache for caching results
        limit: Maximum number of agents to return
        use_cache: Whether to use cached value (default True)

    Returns:
        List of AgentRating sorted by calibration score
    """
    from aragora.ranking.elo import AgentRating

    cache_key = f"calibration_lb:{limit}"

    if use_cache:
        cached = calibration_cache.get(cache_key)
        if cached is not None:
            return cached

    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT agent_name, elo, domain_elos, wins, losses, draws,
                   debates_count, critiques_accepted, critiques_total,
                   calibration_correct, calibration_total, calibration_brier_sum,
                   updated_at
            FROM ratings
            WHERE calibration_total >= ?
            ORDER BY (1.0 - calibration_brier_sum / calibration_total) DESC
            LIMIT ?
            """,
            (CALIBRATION_MIN_COUNT, limit),
        )
        rows = cursor.fetchall()

    result = [
        AgentRating(
            agent_name=row[0],
            elo=row[1],
            domain_elos=safe_json_loads(row[2], {}),
            wins=row[3],
            losses=row[4],
            draws=row[5],
            debates_count=row[6],
            critiques_accepted=row[7],
            critiques_total=row[8],
            calibration_correct=row[9] or 0,
            calibration_total=row[10] or 0,
            calibration_brier_sum=row[11] or 0.0,
            updated_at=row[12],
        )
        for row in rows
    ]

    calibration_cache.set(cache_key, result)
    return result


def get_agent_calibration_history(
    db: EloDatabase,
    agent_name: str,
    limit: int = 50,
) -> list[dict]:
    """Get recent predictions made by an agent.

    Args:
        db: EloDatabase instance for queries
        agent_name: Agent to query
        limit: Maximum predictions to return

    Returns:
        List of prediction dicts with tournament_id, predicted_winner,
        confidence, and created_at
    """
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT tournament_id, predicted_winner, confidence, created_at
            FROM calibration_predictions
            WHERE predictor_agent = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )
        rows = cursor.fetchall()

    return [
        {
            "tournament_id": row[0],
            "predicted_winner": row[1],
            "confidence": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]


__all__ = [
    "get_calibration_leaderboard",
    "get_agent_calibration_history",
]
