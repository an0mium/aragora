"""
Consolidation and batch tier operations for Continuum Memory System.

Extracted from continuum.py to reduce module size while maintaining functionality.
All functions operate on ContinuumMemory instances passed as the first parameter.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aragora.memory.tier_manager import DEFAULT_TIER_CONFIGS, MemoryTier

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory

logger = logging.getLogger(__name__)


def emit_tier_event(
    cms: "ContinuumMemory",
    event_type: str,
    memory_id: str,
    from_tier: MemoryTier,
    to_tier: MemoryTier,
    surprise_score: float,
) -> None:
    """Emit MEMORY_TIER_PROMOTION or MEMORY_TIER_DEMOTION event."""
    if not cms.event_emitter:
        return

    try:
        from aragora.server.stream import StreamEvent, StreamEventType

        stream_type = (
            StreamEventType.MEMORY_TIER_PROMOTION
            if event_type == "promotion"
            else StreamEventType.MEMORY_TIER_DEMOTION
        )

        cms.event_emitter.emit(
            StreamEvent(
                type=stream_type,
                data={
                    "memory_id": memory_id,
                    "from_tier": from_tier.value,
                    "to_tier": to_tier.value,
                    "surprise_score": surprise_score,
                },
            )
        )
    except ImportError:
        # Stream module not available - expected in minimal installations
        logger.debug("[memory] Stream module not available for tier event emission")
    except (AttributeError, TypeError) as e:
        # event_emitter not properly configured or emit() signature mismatch
        logger.debug(f"[memory] Event emitter configuration error: {e}")
    except (ValueError, KeyError) as e:
        # Invalid event data or missing StreamEventType
        logger.warning(f"[memory] Invalid tier event data: {e}")
    except (ConnectionError, OSError) as e:
        # Network/IO errors during event emission - non-critical
        logger.debug(f"[memory] Event emission network error: {e}")


def promote_batch(
    cms: "ContinuumMemory",
    from_tier: MemoryTier,
    to_tier: MemoryTier,
    ids: List[str],
) -> int:
    """
    Batch promote memories from one tier to another.

    Uses executemany for efficient batch updates instead of N+1 queries.
    Thread-safe: uses lock to prevent race conditions with single-item operations.

    Args:
        cms: ContinuumMemory instance
        from_tier: Source tier
        to_tier: Target tier (must be one level faster)
        ids: List of memory IDs to promote

    Returns:
        Number of successfully promoted entries
    """
    if not ids:
        return 0

    now = datetime.now().isoformat()
    cooldown_hours = cms.hyperparams["promotion_cooldown_hours"]
    cutoff_time = (datetime.now() - timedelta(hours=cooldown_hours)).isoformat()

    with cms._tier_lock, cms.connection() as conn:
        cursor = conn.cursor()

        # Batch UPDATE with cooldown check
        # Only promote entries where last_promotion_at is NULL or older than cooldown
        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"""
            UPDATE continuum_memory
            SET tier = ?, last_promotion_at = ?, updated_at = ?
            WHERE id IN ({placeholders})
              AND tier = ?
              AND (last_promotion_at IS NULL OR last_promotion_at < ?)
            """,
            (to_tier.value, now, now, *ids, from_tier.value, cutoff_time),
        )
        promoted_count = cursor.rowcount

        # Batch INSERT tier transitions for promoted entries
        # Only insert for entries that were actually updated
        if promoted_count > 0:
            cursor.execute(
                f"""
                SELECT id, surprise_score FROM continuum_memory
                WHERE id IN ({placeholders}) AND tier = ?
                """,
                (*ids, to_tier.value),
            )
            promoted_entries = cursor.fetchall()

            if promoted_entries:
                cursor.executemany(
                    """
                    INSERT INTO tier_transitions
                    (memory_id, from_tier, to_tier, reason, surprise_score)
                    VALUES (?, ?, ?, 'high_surprise', ?)
                    """,
                    [
                        (entry[0], from_tier.value, to_tier.value, entry[1])
                        for entry in promoted_entries
                    ],
                )

        conn.commit()

    if promoted_count > 0:
        logger.info(
            f"[memory] Batch promoted {promoted_count}/{len(ids)} entries: "
            f"{from_tier.value} -> {to_tier.value}"
        )

    return promoted_count


def demote_batch(
    cms: "ContinuumMemory",
    from_tier: MemoryTier,
    to_tier: MemoryTier,
    ids: List[str],
) -> int:
    """
    Batch demote memories from one tier to another.

    Uses executemany for efficient batch updates instead of N+1 queries.
    Thread-safe: uses lock to prevent race conditions with single-item operations.

    Args:
        cms: ContinuumMemory instance
        from_tier: Source tier
        to_tier: Target tier (must be one level slower)
        ids: List of memory IDs to demote

    Returns:
        Number of successfully demoted entries
    """
    if not ids:
        return 0

    now = datetime.now().isoformat()

    with cms._tier_lock, cms.connection() as conn:
        cursor = conn.cursor()

        # Batch UPDATE - update_count check is already done in candidate selection
        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"""
            UPDATE continuum_memory
            SET tier = ?, updated_at = ?
            WHERE id IN ({placeholders}) AND tier = ?
            """,
            (to_tier.value, now, *ids, from_tier.value),
        )
        demoted_count = cursor.rowcount

        # Batch INSERT tier transitions for demoted entries
        if demoted_count > 0:
            cursor.execute(
                f"""
                SELECT id, surprise_score FROM continuum_memory
                WHERE id IN ({placeholders}) AND tier = ?
                """,
                (*ids, to_tier.value),
            )
            demoted_entries = cursor.fetchall()

            if demoted_entries:
                cursor.executemany(
                    """
                    INSERT INTO tier_transitions
                    (memory_id, from_tier, to_tier, reason, surprise_score)
                    VALUES (?, ?, ?, 'high_stability', ?)
                    """,
                    [
                        (entry[0], from_tier.value, to_tier.value, entry[1])
                        for entry in demoted_entries
                    ],
                )

        conn.commit()

    if demoted_count > 0:
        logger.info(
            f"[memory] Batch demoted {demoted_count}/{len(ids)} entries: "
            f"{from_tier.value} -> {to_tier.value}"
        )

    return demoted_count


def consolidate(cms: "ContinuumMemory") -> Dict[str, int]:
    """
    Run tier consolidation: promote/demote memories based on surprise.

    This should be called periodically (e.g., after each nomic cycle).

    Uses batch operations to avoid N+1 query patterns for better performance
    with large memory stores.

    Each entry is only promoted/demoted once per consolidate call (one level
    at a time), matching the behavior of the individual promote/demote methods.

    Args:
        cms: ContinuumMemory instance

    Returns:
        Dict with counts of promotions and demotions
    """
    logger.debug("[memory] Starting tier consolidation")
    promotions = 0
    demotions = 0

    # Tier order for promotions: glacial -> slow -> medium -> fast
    promotion_pairs = [
        (MemoryTier.GLACIAL, MemoryTier.SLOW),
        (MemoryTier.SLOW, MemoryTier.MEDIUM),
        (MemoryTier.MEDIUM, MemoryTier.FAST),
    ]

    # Tier order for demotions: fast -> medium -> slow -> glacial
    demotion_pairs = [
        (MemoryTier.FAST, MemoryTier.MEDIUM),
        (MemoryTier.MEDIUM, MemoryTier.SLOW),
        (MemoryTier.SLOW, MemoryTier.GLACIAL),
    ]

    # Collect ALL candidates upfront before any processing
    # This ensures each entry only moves one level per consolidate call
    promotion_candidates: Dict[tuple, List[str]] = {}
    demotion_candidates: Dict[tuple, List[str]] = {}

    with cms.connection() as conn:
        cursor = conn.cursor()

        # Collect promotion candidates for all tier pairs
        # Limit to 1000 per tier to prevent memory issues with large databases
        batch_limit = 1000
        for from_tier, to_tier in promotion_pairs:
            config = DEFAULT_TIER_CONFIGS[from_tier]
            cursor.execute(
                """
                SELECT id FROM continuum_memory
                WHERE tier = ? AND surprise_score > ?
                ORDER BY surprise_score DESC
                LIMIT ?
                """,
                (from_tier.value, config.promotion_threshold, batch_limit),
            )
            ids = [row[0] for row in cursor.fetchall()]
            if ids:
                promotion_candidates[(from_tier, to_tier)] = ids

        # Collect demotion candidates for all tier pairs
        for from_tier, to_tier in demotion_pairs:
            config = DEFAULT_TIER_CONFIGS[from_tier]
            cursor.execute(
                """
                SELECT id FROM continuum_memory
                WHERE tier = ?
                  AND (1.0 - surprise_score) > ?
                  AND update_count > 10
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (from_tier.value, config.demotion_threshold, batch_limit),
            )
            ids = [row[0] for row in cursor.fetchall()]
            if ids:
                demotion_candidates[(from_tier, to_tier)] = ids

    # Process all promotions (outside the collection connection)
    for (from_tier, to_tier), ids in promotion_candidates.items():
        count = promote_batch(cms, from_tier, to_tier, ids)
        promotions += count
        logger.debug(
            f"Promoted {count}/{len(ids)} entries from {from_tier.value} to {to_tier.value}"
        )

    # Process all demotions (outside the collection connection)
    for (from_tier, to_tier), ids in demotion_candidates.items():
        count = demote_batch(cms, from_tier, to_tier, ids)
        demotions += count
        logger.debug(
            f"Demoted {count}/{len(ids)} entries from {from_tier.value} to {to_tier.value}"
        )

    if promotions > 0 or demotions > 0:
        logger.info(
            f"[memory] Consolidation complete: {promotions} promotions, {demotions} demotions"
        )
    else:
        logger.debug("[memory] Consolidation complete: no tier changes")

    return {"promotions": promotions, "demotions": demotions}
