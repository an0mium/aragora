"""
Safety gates for self-modifying debate features.
"""

from __future__ import annotations

__all__ = [
    "resolve_auto_evolve",
    "resolve_prompt_evolution",
]

import logging
import os

logger = logging.getLogger(__name__)


def resolve_auto_evolve(requested: bool) -> bool:
    """Allow population evolution only when explicitly enabled."""
    if not requested:
        return False
    if os.environ.get("ARAGORA_ALLOW_AUTO_EVOLVE", "0") == "1":
        return True
    logger.warning("[safety] auto_evolve disabled; set ARAGORA_ALLOW_AUTO_EVOLVE=1 to enable")
    return False


def resolve_prompt_evolution(requested: bool) -> bool:
    """Allow prompt evolution only when explicitly enabled."""
    if not requested:
        return False
    if os.environ.get("ARAGORA_ALLOW_PROMPT_EVOLVE", "0") == "1":
        return True
    logger.warning(
        "[safety] prompt evolution disabled; set ARAGORA_ALLOW_PROMPT_EVOLVE=1 to enable"
    )
    return False
