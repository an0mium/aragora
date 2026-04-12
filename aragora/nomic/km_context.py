"""
Knowledge Mound context provider for Nomic loop.

Initializes and provides a Knowledge Mound instance for use during
Nomic context gathering and debate phases. This bridges the canonical
KM singleton to the NomicContextBuilder.

Usage:
    from aragora.nomic.km_context import get_nomic_knowledge_mound

    km = get_nomic_knowledge_mound()
    builder = NomicContextBuilder(aragora_path=Path("."), knowledge_mound=km)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_nomic_km_instance: Any | None = None


def _load_knowledge_mound_factory() -> Any:
    """Load the canonical Knowledge Mound factory."""
    from aragora.knowledge.mound import get_knowledge_mound

    return get_knowledge_mound


def _create_nomic_knowledge_mound() -> Any | None:
    """Create a Nomic-scoped Knowledge Mound instance."""
    get_knowledge_mound = _load_knowledge_mound_factory()
    return get_knowledge_mound(workspace_id="nomic")


def get_nomic_knowledge_mound() -> Any | None:
    """Get a Knowledge Mound instance for Nomic context.

    Returns the canonical KM singleton if available, or None if KM
    is not configured. The result is cached for the process lifetime.

    Environment Variables:
        NOMIC_KM_ENABLED: Set to "0" to disable KM integration (default: "1")

    Returns:
        KnowledgeMound instance or None
    """
    global _nomic_km_instance

    if _nomic_km_instance is not None:
        return _nomic_km_instance

    if os.environ.get("NOMIC_KM_ENABLED", "1") == "0":
        logger.debug("[nomic-km] KM disabled via NOMIC_KM_ENABLED=0")
        return None

    try:
        _nomic_km_instance = _create_nomic_knowledge_mound()
        if _nomic_km_instance is None:
            logger.debug("[nomic-km] KM returned no instance for Nomic context")
            return None

        logger.info("[nomic-km] Knowledge Mound initialized for Nomic context")
        return _nomic_km_instance
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.debug("[nomic-km] KM not available: %s", e)
        return None


def reset_nomic_knowledge_mound() -> None:
    """Reset the cached KM instance (for testing)."""
    global _nomic_km_instance
    _nomic_km_instance = None
