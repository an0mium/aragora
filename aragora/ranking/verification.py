"""
Formal Verification Integration for ELO system.

Extracted from EloSystem to separate verification-related ELO adjustments.
Adjusts ELO based on formal proof verification results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aragora.config import ELO_INITIAL_RATING

if TYPE_CHECKING:
    from aragora.ranking.database import EloDatabase
    from aragora.ranking.elo import AgentRating

logger = logging.getLogger(__name__)

DEFAULT_ELO = ELO_INITIAL_RATING

# Maximum agent name length for validation
MAX_AGENT_NAME_LENGTH = 32


def _validate_agent_name(agent_name: str) -> None:
    """Validate agent name length."""
    if len(agent_name) > MAX_AGENT_NAME_LENGTH:
        raise ValueError(
            f"Agent name exceeds {MAX_AGENT_NAME_LENGTH} characters: {len(agent_name)}"
        )


def calculate_verification_elo_change(
    verified_count: int,
    disproven_count: int,
    k_factor: float = 16.0,
) -> float:
    """
    Calculate ELO change from verification results.

    Args:
        verified_count: Number of claims that were formally proven
        disproven_count: Number of claims that were formally disproven
        k_factor: ELO adjustment factor (default 16, half of standard)

    Returns:
        Net ELO change to apply
    """
    if verified_count == 0 and disproven_count == 0:
        return 0.0

    # Calculate ELO adjustment:
    # - Each verified claim: +k_factor * 0.5 (half a "win" vs verification)
    # - Each disproven claim: -k_factor * 0.5 (half a "loss" vs verification)
    # Using half because verification is against an objective standard,
    # not another agent.
    verification_bonus = verified_count * k_factor * 0.5
    disproven_penalty = disproven_count * k_factor * 0.5
    return verification_bonus - disproven_penalty


def update_rating_from_verification(
    rating: "AgentRating",
    domain: str,
    net_change: float,
    default_elo: float = DEFAULT_ELO,
) -> None:
    """
    Apply verification ELO change to a rating object.

    Modifies the rating in place.

    Args:
        rating: AgentRating to update
        domain: Domain of the verification
        net_change: Net ELO change to apply
        default_elo: Default ELO for new domain ratings
    """
    # Apply to overall ELO
    rating.elo = max(100, rating.elo + net_change)  # Floor at 100

    # Apply to domain-specific ELO
    if domain:
        old_domain_elo = rating.domain_elos.get(domain, default_elo)
        rating.domain_elos[domain] = max(100, old_domain_elo + net_change)

    rating.updated_at = datetime.now().isoformat()


def get_verification_history(
    db: "EloDatabase",
    agent_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Get verification-related ELO history entries.

    Args:
        db: EloDatabase instance
        agent_name: Agent to query
        limit: Maximum entries to return

    Returns:
        List of history entry dicts
    """
    _validate_agent_name(agent_name)

    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT debate_id, elo, created_at
            FROM elo_history
            WHERE agent_name = ? AND debate_id LIKE 'verification:%'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )
        rows = cursor.fetchall()

    history: list[dict[str, Any]] = []
    prev_elo: float | None = None

    for row in reversed(rows):
        debate_id, elo, created_at = row
        if prev_elo is not None:
            change = elo - prev_elo
            history.append(
                {
                    "debate_id": debate_id,
                    "elo": elo,
                    "change": change,
                    "created_at": created_at,
                }
            )
        prev_elo = elo

    return list(reversed(history))


def calculate_verification_impact(
    db: "EloDatabase",
    agent_name: str,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get summary of verification impact on an agent's ELO.

    Args:
        db: EloDatabase instance
        agent_name: Agent to query
        limit: Maximum history entries to consider

    Returns:
        Dict with agent_name, verification_events, total_impact, and history
    """
    history = get_verification_history(db, agent_name, limit)
    total_impact = sum(entry.get("change", 0) for entry in history)

    return {
        "agent_name": agent_name,
        "verification_events": len(history),
        "total_impact": total_impact,
        "history": history,
    }


__all__ = [
    "calculate_verification_elo_change",
    "update_rating_from_verification",
    "get_verification_history",
    "calculate_verification_impact",
]
