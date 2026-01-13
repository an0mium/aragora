"""
Optional imports for the debate orchestrator.

This module provides lazy loading for optional components to avoid
circular imports. Components are loaded on first access and cached.

Usage:
    from aragora.debate.optional_imports import OptionalImports

    # Get component (returns None if not available)
    tracker_cls = OptionalImports.get_position_tracker()
    if tracker_cls:
        tracker = tracker_cls(...)
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Type

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.truth_grounding import PositionTracker
    from aragora.insights import InsightExtractor, InsightStore
    from aragora.memory.store import CritiqueStore
    from aragora.reasoning.belief import BeliefNetwork, BeliefPropagationAnalyzer
    from aragora.reasoning.citations import CitationExtractor
    from aragora.visualization.mapper import ArgumentCartographer


class OptionalImports:
    """
    Centralized lazy loading for optional debate components.

    All imports are cached after first access. Failed imports return None
    and are logged at debug level (not errors, since they're optional).
    """

    # Cache for loaded modules
    _cache: dict[str, Any] = {}

    @classmethod
    def _get_cached(cls, key: str, module: str, class_name: str) -> Optional[Type]:
        """Get a class from cache, loading if necessary."""
        if key in cls._cache:
            return cls._cache[key]

        try:
            mod = __import__(module, fromlist=[class_name])
            cls._cache[key] = getattr(mod, class_name)
            return cls._cache[key]
        except (ImportError, AttributeError) as e:
            logger.debug(f"Optional import {module}.{class_name} not available: {e}")
            cls._cache[key] = None
            return None

    @classmethod
    def get_position_tracker(cls) -> Optional[Type["PositionTracker"]]:
        """Get PositionTracker class for truth-grounded personas."""
        return cls._get_cached(
            "position_tracker", "aragora.agents.truth_grounding", "PositionTracker"
        )

    @classmethod
    def get_calibration_tracker(cls) -> Optional[Type["CalibrationTracker"]]:
        """Get CalibrationTracker class for prediction accuracy."""
        return cls._get_cached(
            "calibration_tracker", "aragora.agents.calibration", "CalibrationTracker"
        )

    @classmethod
    def get_belief_network(cls) -> Optional[Type["BeliefNetwork"]]:
        """Get BeliefNetwork class for belief propagation."""
        return cls._get_cached("belief_network", "aragora.reasoning.belief", "BeliefNetwork")

    @classmethod
    def get_belief_propagation_analyzer(cls) -> Optional[Type["BeliefPropagationAnalyzer"]]:
        """Get BeliefPropagationAnalyzer class."""
        return cls._get_cached(
            "belief_propagation_analyzer", "aragora.reasoning.belief", "BeliefPropagationAnalyzer"
        )

    @classmethod
    def get_belief_analyzer(
        cls,
    ) -> tuple[Optional[Type["BeliefNetwork"]], Optional[Type["BeliefPropagationAnalyzer"]]]:
        """Get both belief analysis classes as a tuple.

        Returns:
            Tuple of (BeliefNetwork, BeliefPropagationAnalyzer), either may be None.
        """
        return cls.get_belief_network(), cls.get_belief_propagation_analyzer()

    @classmethod
    def get_citation_extractor(cls) -> Optional[Type["CitationExtractor"]]:
        """Get CitationExtractor class for evidence citations."""
        return cls._get_cached(
            "citation_extractor", "aragora.reasoning.citations", "CitationExtractor"
        )

    @classmethod
    def get_insight_extractor(cls) -> Optional[Type["InsightExtractor"]]:
        """Get InsightExtractor class for debate learnings."""
        return cls._get_cached("insight_extractor", "aragora.insights", "InsightExtractor")

    @classmethod
    def get_insight_store(cls) -> Optional[Type["InsightStore"]]:
        """Get InsightStore class for storing insights."""
        return cls._get_cached("insight_store", "aragora.insights", "InsightStore")

    @classmethod
    def get_critique_store(cls) -> Optional[Type["CritiqueStore"]]:
        """Get CritiqueStore class for critique patterns."""
        return cls._get_cached("critique_store", "aragora.memory.store", "CritiqueStore")

    @classmethod
    def get_argument_cartographer(cls) -> Optional[Type["ArgumentCartographer"]]:
        """Get ArgumentCartographer class for graph visualization."""
        return cls._get_cached(
            "argument_cartographer", "aragora.visualization.mapper", "ArgumentCartographer"
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the import cache (mainly for testing)."""
        cls._cache.clear()
