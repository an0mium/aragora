"""
Domain-specific Ratings and Knowledge Mound Integration.

Extracted from EloSystem to separate domain expertise tracking and
Knowledge Mound bidirectional sync operations. Provides:
- KM adapter management
- Agent skill history queries via KM
- Domain expertise discovery via KM
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter

logger = logging.getLogger(__name__)


class KMAdapterMixin:
    """Mixin providing Knowledge Mound adapter integration.

    Manages the KM adapter lifecycle and provides methods for
    bidirectional sync between ELO ratings and Knowledge Mound.

    Attributes:
        _km_adapter: Optional EloAdapter instance for KM integration
    """

    _km_adapter: Optional["EloAdapter"]

    def set_km_adapter(self, adapter: "EloAdapter") -> None:
        """Set the Knowledge Mound adapter for bidirectional sync.

        Args:
            adapter: EloAdapter instance for KM integration
        """
        self._km_adapter = adapter

    def query_km_agent_skill_history(
        self,
        agent_name: str,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query Knowledge Mound for agent skill history (reverse flow).

        This enables cross-session skill tracking and team selection.

        Args:
            agent_name: Agent to get history for
            domain: Optional domain filter
            limit: Maximum results

        Returns:
            List of skill history entries from Knowledge Mound
        """
        if not self._km_adapter:
            return []

        try:
            return self._km_adapter.get_agent_skill_history(  # type: ignore[call-arg]
                agent_name=agent_name,
                domain=domain,
                limit=limit,
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"Failed to query KM for agent skill history: {e}")
            return []

    def query_km_domain_expertise(
        self,
        domain: str,
        limit: int = 10,
    ) -> list[dict]:
        """Query Knowledge Mound for agents with domain expertise (reverse flow).

        This enables smart team selection for domain-specific debates.

        Args:
            domain: Domain to find experts for
            limit: Maximum results

        Returns:
            List of agents with domain expertise from Knowledge Mound
        """
        if not self._km_adapter:
            return []

        try:
            return self._km_adapter.get_domain_expertise(
                domain=domain,
                limit=limit,
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"Failed to query KM for domain expertise: {e}")
            return []


__all__ = [
    "KMAdapterMixin",
]
