"""Convergence detection lifecycle helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
convergence detector initialization, reinitialization per debate, and
embedding cache cleanup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.debate.convergence import (
    ConvergenceDetector,
    cleanup_embedding_cache,
)

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena

logger = logging.getLogger(__name__)


def init_convergence(arena: Arena, debate_id: str | None = None) -> None:
    """Initialize convergence detection if enabled.

    Sets up the convergence detector with protocol thresholds and
    creates the previous round responses tracker.

    Args:
        arena: Arena instance to initialize.
        debate_id: Optional debate ID for cache scoping.
    """
    arena.convergence_detector = None
    arena._convergence_debate_id = debate_id
    if arena.protocol.convergence_detection:
        arena.convergence_detector = ConvergenceDetector(
            convergence_threshold=arena.protocol.convergence_threshold,
            divergence_threshold=arena.protocol.divergence_threshold,
            min_rounds_before_check=1,
            debate_id=debate_id,
        )
    arena._previous_round_responses = {}


def reinit_convergence_for_debate(arena: Arena, debate_id: str) -> None:
    """Reinitialize convergence detector with debate-specific cache.

    Avoids redundant reinitialization if the debate ID matches.

    Args:
        arena: Arena instance.
        debate_id: New debate ID for cache scoping.
    """
    if arena._convergence_debate_id == debate_id:
        return
    arena._convergence_debate_id = debate_id
    if arena.protocol.convergence_detection:
        arena.convergence_detector = ConvergenceDetector(
            convergence_threshold=arena.protocol.convergence_threshold,
            divergence_threshold=arena.protocol.divergence_threshold,
            min_rounds_before_check=1,
            debate_id=debate_id,
        )
        logger.debug(f"Reinitialized convergence detector for debate {debate_id}")


def cleanup_convergence(arena: Arena) -> None:
    """Cleanup embedding cache for the current debate.

    Args:
        arena: Arena instance.
    """
    if arena._convergence_debate_id:
        cleanup_embedding_cache(arena._convergence_debate_id)
        logger.debug(f"Cleaned up embedding cache for debate {arena._convergence_debate_id}")


__all__ = [
    "init_convergence",
    "reinit_convergence_for_debate",
    "cleanup_convergence",
]
