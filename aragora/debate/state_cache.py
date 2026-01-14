"""DebateStateCache - Centralized cache management for debate state.

Extracted from Arena orchestrator to reduce complexity and provide
clear ownership of cached debate state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DebateStateCache:
    """Manages cached state during debate execution.

    This class centralizes all per-debate cached values that were previously
    scattered across Arena instance variables. Benefits:
    - Single point of ownership for cache lifecycle
    - Easy to clear/reset between debates
    - Clear distinction between ephemeral cache and persistent state
    """

    # Historical context cache (computed once per debate)
    historical_context: str = ""

    # Research context cache (computed once per debate)
    research_context: Optional[str] = None

    # Evidence pack cache (for grounding verdict with citations)
    evidence_pack: Any = None

    # Continuum memory context (retrieved once per debate)
    continuum_context: str = ""
    continuum_retrieved_ids: list[str] = field(default_factory=list)
    continuum_retrieved_tiers: dict[str, Any] = field(default_factory=dict)

    # Similarity backend cache (avoids recreating per vote grouping call)
    similarity_backend: Any = None

    # Debate domain cache (computed once per debate)
    debate_domain: Optional[str] = None

    def clear(self) -> None:
        """Reset all cached values to initial state.

        Called during cleanup to ensure fresh state for next debate.
        """
        self.historical_context = ""
        self.research_context = None
        self.evidence_pack = None
        self.continuum_context = ""
        self.continuum_retrieved_ids = []
        self.continuum_retrieved_tiers = {}
        self.similarity_backend = None
        self.debate_domain = None

    def clear_continuum_tracking(self) -> None:
        """Clear continuum memory tracking after outcome update.

        Called after memory outcomes have been recorded.
        """
        self.continuum_retrieved_ids = []
        self.continuum_retrieved_tiers = {}

    def has_continuum_context(self) -> bool:
        """Check if continuum context has been retrieved."""
        return bool(self.continuum_context)

    def has_debate_domain(self) -> bool:
        """Check if debate domain has been computed."""
        return self.debate_domain is not None

    def track_continuum_retrieval(
        self,
        context: str,
        retrieved_ids: list[str],
        retrieved_tiers: dict[str, Any],
    ) -> None:
        """Record continuum memory retrieval results.

        Args:
            context: Retrieved context string
            retrieved_ids: List of memory IDs that were retrieved
            retrieved_tiers: Mapping of ID to MemoryTier for analytics
        """
        self.continuum_context = context
        self.continuum_retrieved_ids = retrieved_ids
        self.continuum_retrieved_tiers = retrieved_tiers
