"""
Arena factory for dependency injection.

Centralizes the creation of Arena instances with properly injected
dependencies, eliminating lazy imports in orchestrator.py.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.core import Agent, Environment

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.spectate.stream import SpectatorStream

logger = logging.getLogger(__name__)


class ArenaFactory:
    """
    Factory for creating Arena instances with injected dependencies.

    This factory centralizes dependency resolution, allowing Arena to
    accept Protocol-typed dependencies rather than importing concrete
    implementations directly. This breaks circular import chains.

    Usage:
        factory = ArenaFactory()
        arena = factory.create(
            environment=env,
            agents=agents,
            protocol=protocol,
            enable_insights=True,
            enable_belief_analysis=True,
        )
    """

    def __init__(self):
        """Initialize the factory with cached class references."""
        self._position_tracker_cls = None
        self._calibration_tracker_cls = None
        self._belief_network_cls = None
        self._belief_analyzer_cls = None
        self._citation_extractor_cls = None
        self._insight_extractor_cls = None
        self._insight_store_cls = None
        self._critique_store_cls = None
        self._argument_cartographer_cls = None

    def _get_position_tracker_cls(self):
        """Lazy load PositionTracker class."""
        if self._position_tracker_cls is None:
            try:
                from aragora.agents.truth_grounding import PositionTracker

                self._position_tracker_cls = PositionTracker
            except ImportError:
                logger.debug("PositionTracker not available")
        return self._position_tracker_cls

    def _get_calibration_tracker_cls(self):
        """Lazy load CalibrationTracker class."""
        if self._calibration_tracker_cls is None:
            try:
                from aragora.agents.calibration import CalibrationTracker

                self._calibration_tracker_cls = CalibrationTracker
            except ImportError:
                logger.debug("CalibrationTracker not available")
        return self._calibration_tracker_cls

    def _get_belief_classes(self):
        """Lazy load BeliefNetwork and BeliefPropagationAnalyzer."""
        if self._belief_network_cls is None:
            try:
                from aragora.reasoning.belief import (
                    BeliefNetwork,
                    BeliefPropagationAnalyzer,
                )

                self._belief_network_cls = BeliefNetwork
                self._belief_analyzer_cls = BeliefPropagationAnalyzer
            except ImportError:
                logger.debug("Belief classes not available")
        return self._belief_network_cls, self._belief_analyzer_cls

    def _get_citation_extractor_cls(self):
        """Lazy load CitationExtractor class."""
        if self._citation_extractor_cls is None:
            try:
                from aragora.reasoning.citations import CitationExtractor

                self._citation_extractor_cls = CitationExtractor
            except ImportError:
                logger.debug("CitationExtractor not available")
        return self._citation_extractor_cls

    def _get_insight_classes(self):
        """Lazy load InsightExtractor and InsightStore."""
        if self._insight_extractor_cls is None:
            try:
                from aragora.insights import InsightExtractor, InsightStore

                self._insight_extractor_cls = InsightExtractor
                self._insight_store_cls = InsightStore
            except ImportError:
                logger.debug("Insight classes not available")
        return self._insight_extractor_cls, self._insight_store_cls

    def _get_critique_store_cls(self):
        """Lazy load CritiqueStore class."""
        if self._critique_store_cls is None:
            try:
                from aragora.memory.store import CritiqueStore

                self._critique_store_cls = CritiqueStore
            except ImportError:
                logger.debug("CritiqueStore not available")
        return self._critique_store_cls

    def _get_argument_cartographer_cls(self):
        """Lazy load ArgumentCartographer class."""
        if self._argument_cartographer_cls is None:
            try:
                from aragora.visualization.mapper import ArgumentCartographer

                self._argument_cartographer_cls = ArgumentCartographer
            except ImportError:
                logger.debug("ArgumentCartographer not available")
        return self._argument_cartographer_cls

    def create_position_tracker(self, **kwargs: Any) -> Optional[Any]:
        """Create a PositionTracker instance."""
        cls = self._get_position_tracker_cls()
        if cls:
            return cls(**kwargs)
        return None

    def create_calibration_tracker(self, **kwargs: Any) -> Optional[Any]:
        """Create a CalibrationTracker instance."""
        cls = self._get_calibration_tracker_cls()
        if cls:
            return cls(**kwargs)
        return None

    def create_belief_network(self, **kwargs: Any) -> Optional[Any]:
        """Create a BeliefNetwork instance."""
        cls, _ = self._get_belief_classes()
        if cls:
            return cls(**kwargs)
        return None

    def create_belief_analyzer(self, **kwargs: Any) -> Optional[Any]:
        """Create a BeliefPropagationAnalyzer instance."""
        _, cls = self._get_belief_classes()
        if cls:
            return cls(**kwargs)
        return None

    def create_citation_extractor(self, **kwargs: Any) -> Optional[Any]:
        """Create a CitationExtractor instance."""
        cls = self._get_citation_extractor_cls()
        if cls:
            return cls(**kwargs)
        return None

    def create_insight_extractor(self, **kwargs: Any) -> Optional[Any]:
        """Create an InsightExtractor instance."""
        cls, _ = self._get_insight_classes()
        if cls:
            return cls(**kwargs)
        return None

    def create_insight_store(self, **kwargs: Any) -> Optional[Any]:
        """Create an InsightStore instance."""
        _, cls = self._get_insight_classes()
        if cls:
            return cls(**kwargs)
        return None

    def create_critique_store(self, **kwargs: Any) -> Optional[Any]:
        """Create a CritiqueStore instance."""
        cls = self._get_critique_store_cls()
        if cls:
            return cls(**kwargs)
        return None

    def create_argument_cartographer(self, **kwargs: Any) -> Optional[Any]:
        """Create an ArgumentCartographer instance."""
        cls = self._get_argument_cartographer_cls()
        if cls:
            return cls(**kwargs)
        return None

    def create(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: Optional[DebateProtocol] = None,
        # Optional dependencies - set to True to auto-create
        enable_position_tracking: bool = False,
        enable_calibration: bool = False,
        enable_insights: bool = False,
        enable_belief_analysis: bool = False,
        enable_critique_patterns: bool = False,
        enable_argument_mapping: bool = False,
        # Or pass explicit instances
        memory: Optional[Any] = None,
        event_hooks: Optional[dict] = None,
        event_emitter: Optional[Any] = None,
        spectator: Optional[SpectatorStream] = None,
        debate_embeddings: Optional[Any] = None,
        insight_store: Optional[Any] = None,
        recorder: Optional[Any] = None,
        agent_weights: Optional[dict[str, float]] = None,
        position_tracker: Optional[Any] = None,
        position_ledger: Optional[Any] = None,
        elo_system: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
        dissent_retriever: Optional[Any] = None,
        flip_detector: Optional[Any] = None,
        calibration_tracker: Optional[Any] = None,
        continuum_memory: Optional[Any] = None,
        relationship_tracker: Optional[Any] = None,
        moment_detector: Optional[Any] = None,
        loop_id: str = "",
        strict_loop_scoping: bool = False,
        circuit_breaker: Optional[CircuitBreaker] = None,
        initial_messages: Optional[list] = None,
        trending_topic: Optional[Any] = None,
    ) -> "Arena":
        """
        Create an Arena instance with injected dependencies.

        Args:
            environment: The debate environment
            agents: List of participating agents
            protocol: Debate protocol (defaults to DebateProtocol())
            enable_*: Flags to auto-create optional components
            *: Explicit component instances (override enable_* flags)

        Returns:
            Configured Arena instance
        """
        # Import Arena here to avoid circular import
        from aragora.debate.orchestrator import Arena

        # Auto-create components if requested and not provided
        if enable_position_tracking and position_tracker is None:
            position_tracker = self.create_position_tracker()

        if enable_calibration and calibration_tracker is None:
            calibration_tracker = self.create_calibration_tracker()

        if enable_insights and insight_store is None:
            insight_store = self.create_insight_store()

        if enable_critique_patterns and memory is None:
            memory = self.create_critique_store()

        return Arena(
            environment=environment,
            agents=agents,
            protocol=protocol,
            memory=memory,
            event_hooks=event_hooks,
            event_emitter=event_emitter,
            spectator=spectator,
            debate_embeddings=debate_embeddings,
            insight_store=insight_store,
            recorder=recorder,
            agent_weights=agent_weights,
            position_tracker=position_tracker,
            position_ledger=position_ledger,
            elo_system=elo_system,
            persona_manager=persona_manager,
            dissent_retriever=dissent_retriever,
            flip_detector=flip_detector,
            calibration_tracker=calibration_tracker,
            continuum_memory=continuum_memory,
            relationship_tracker=relationship_tracker,
            moment_detector=moment_detector,
            loop_id=loop_id,
            strict_loop_scoping=strict_loop_scoping,
            circuit_breaker=circuit_breaker,
            initial_messages=initial_messages,
            trending_topic=trending_topic,
        )


# Singleton factory instance
_factory: Optional[ArenaFactory] = None


def get_arena_factory() -> ArenaFactory:
    """Get the singleton ArenaFactory instance."""
    global _factory
    if _factory is None:
        _factory = ArenaFactory()
    return _factory


def create_arena(
    environment: Environment,
    agents: list[Agent],
    **kwargs: Any,
) -> "Arena":
    """
    Convenience function to create an Arena with the default factory.

    This is the recommended way to create Arena instances, as it
    centralizes dependency injection.

    Example:
        arena = create_arena(
            environment=env,
            agents=agents,
            enable_insights=True,
        )
    """
    return get_arena_factory().create(environment, agents, **kwargs)
