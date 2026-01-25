"""
Knowledge management for debates.

Extracted from Arena to improve code organization and testability.
Handles Knowledge Mound operations including:
- KnowledgeMoundOperations initialization
- KnowledgeBridgeHub access
- RevalidationScheduler management
- Bidirectional coordinator and adapter factory
- Culture hints retrieval and application
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult, Environment
    from aragora.knowledge.mound import KnowledgeMound
    from aragora.knowledge.mound.operations import KnowledgeMoundOperations

logger = logging.getLogger(__name__)


class ArenaKnowledgeManager:
    """Manages Knowledge Mound operations for Arena debates.

    Consolidates KM initialization, context retrieval, outcome ingestion,
    and culture-based protocol hints into a single manager class.

    Example:
        km_manager = ArenaKnowledgeManager(
            knowledge_mound=mound,
            enable_retrieval=True,
            enable_ingestion=True,
            notify_callback=arena._notify_spectator,
        )
        km_manager.initialize(arena)

        # During debate
        await km_manager.init_context(debate_id, domain, env, agents, protocol)
        context = await km_manager.fetch_context(task)

        # After debate
        await km_manager.ingest_outcome(result, env)
    """

    def __init__(
        self,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        enable_retrieval: bool = False,
        enable_ingestion: bool = False,
        enable_auto_revalidation: bool = False,
        revalidation_staleness_threshold: float = 0.7,
        revalidation_check_interval_seconds: int = 3600,
        notify_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Initialize the knowledge manager.

        Args:
            knowledge_mound: KnowledgeMound instance for storage/retrieval
            enable_retrieval: Whether to fetch knowledge context for debates
            enable_ingestion: Whether to store debate outcomes in KM
            enable_auto_revalidation: Whether to auto-revalidate stale knowledge
            revalidation_staleness_threshold: Staleness threshold for revalidation
            revalidation_check_interval_seconds: Interval between revalidation checks
            notify_callback: Callback for KM event notifications
        """
        self.knowledge_mound = knowledge_mound
        self.enable_retrieval = enable_retrieval
        self.enable_ingestion = enable_ingestion
        self.enable_auto_revalidation = enable_auto_revalidation
        self.revalidation_staleness_threshold = revalidation_staleness_threshold
        self.revalidation_check_interval_seconds = revalidation_check_interval_seconds
        self._notify_callback = notify_callback

        # Components initialized during initialize()
        self._knowledge_ops: Optional["KnowledgeMoundOperations"] = None
        self._km_metrics: Optional[Any] = None
        self.knowledge_bridge_hub: Optional[Any] = None
        self.revalidation_scheduler: Optional[Any] = None
        self._km_coordinator: Optional[Any] = None
        self._km_adapters: Dict[str, Any] = {}

        # Culture hint storage
        self._culture_consensus_hint: Optional[str] = None
        self._culture_extra_critiques: int = 0
        self._culture_early_consensus: Optional[float] = None
        self._culture_domain_patterns: Dict[str, Any] = {}

    def initialize(
        self,
        continuum_memory: Optional[Any] = None,
        consensus_memory: Optional[Any] = None,
        elo_system: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        insight_store: Optional[Any] = None,
        flip_detector: Optional[Any] = None,
        evidence_store: Optional[Any] = None,
        pulse_manager: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        """Initialize KM infrastructure components.

        Creates:
        - KnowledgeMoundOperations for query/store
        - KMMetrics for observability
        - KnowledgeBridgeHub for unified bridge access
        - RevalidationScheduler if auto-revalidation enabled
        - BidirectionalCoordinator and adapters for subsystem sync

        Args:
            continuum_memory: ContinuumMemory instance
            consensus_memory: ConsensusMemory instance
            elo_system: EloSystem for agent rankings
            cost_tracker: CostTracker for budget tracking
            insight_store: InsightStore for pattern insights
            flip_detector: FlipDetector for position tracking
            evidence_store: EvidenceStore for evidence storage
            pulse_manager: PulseManager for trending topics
            memory: Legacy memory instance
        """
        from aragora.knowledge.mound.operations import KnowledgeMoundOperations

        # Initialize KM metrics for observability
        self._km_metrics = None
        if self.knowledge_mound:
            try:
                from aragora.knowledge.mound.metrics import KMMetrics

                self._km_metrics = KMMetrics()
                logger.debug("[knowledge_mound] KMMetrics initialized for observability")
            except ImportError:
                pass

        # Create KnowledgeMoundOperations
        self._knowledge_ops = KnowledgeMoundOperations(
            knowledge_mound=self.knowledge_mound,
            enable_retrieval=self.enable_retrieval,
            enable_ingestion=self.enable_ingestion,
            notify_callback=self._notify_callback,
            metrics=self._km_metrics,
        )

        # Initialize KnowledgeBridgeHub for unified bridge access
        self.knowledge_bridge_hub = None
        if self.knowledge_mound:
            from aragora.knowledge.bridges import KnowledgeBridgeHub

            self.knowledge_bridge_hub = KnowledgeBridgeHub(self.knowledge_mound)

        # Initialize RevalidationScheduler for automatic knowledge revalidation
        self.revalidation_scheduler = None
        if self.enable_auto_revalidation and self.knowledge_mound:
            try:
                from aragora.knowledge.mound.revalidation_scheduler import RevalidationScheduler

                self.revalidation_scheduler = RevalidationScheduler(
                    knowledge_mound=self.knowledge_mound,
                    staleness_threshold=self.revalidation_staleness_threshold,
                    check_interval_seconds=self.revalidation_check_interval_seconds,
                )
                logger.info(
                    "[knowledge_mound] RevalidationScheduler initialized "
                    "(staleness_threshold=%.2f)",
                    self.revalidation_staleness_threshold,
                )
            except ImportError as e:
                logger.debug(f"[knowledge_mound] RevalidationScheduler unavailable: {e}")

        # Initialize KM adapter factory and coordinator for bidirectional sync
        self._km_coordinator = None
        self._km_adapters = {}
        if self.knowledge_mound:
            try:
                from aragora.knowledge.mound.adapters import AdapterFactory
                from aragora.knowledge.mound.bidirectional_coordinator import (
                    BidirectionalCoordinator,
                )

                # Create coordinator
                self._km_coordinator = BidirectionalCoordinator()

                # Create adapters from available subsystems
                factory = AdapterFactory(
                    event_callback=self._notify_callback,
                )

                self._km_adapters = factory.create_from_subsystems(
                    continuum_memory=continuum_memory,
                    consensus_memory=consensus_memory,
                    elo_system=elo_system,
                    cost_tracker=cost_tracker,
                    insight_store=insight_store,
                    flip_detector=flip_detector,
                    evidence_store=evidence_store,
                    pulse_manager=pulse_manager,
                    memory=memory,
                )

                # Register adapters with coordinator
                if self._km_adapters:
                    registered = factory.register_with_coordinator(
                        self._km_coordinator, self._km_adapters
                    )
                    logger.info(
                        "[knowledge_mound] AdapterFactory created %d adapters, "
                        "registered %d with coordinator",
                        len(self._km_adapters),
                        registered,
                    )
            except ImportError as e:
                logger.debug(f"[knowledge_mound] AdapterFactory unavailable: {e}")
            except Exception as e:
                logger.warning(f"[knowledge_mound] Failed to initialize adapters: {e}")

    async def init_context(
        self,
        debate_id: str,
        domain: str,
        env: "Environment",
        agents: list,
        protocol: Any,
    ) -> None:
        """Initialize Knowledge Mound context for the debate.

        Emits DEBATE_START event to trigger cross-subsystem handlers:
        - mound_to_belief: Initialize belief priors from historical cruxes
        - mound_to_team_selection: Query domain experts for team assembly
        - mound_to_trickster: Load flip history for consistency checking
        - culture_to_debate: Load learned culture patterns

        Args:
            debate_id: Unique debate identifier
            domain: Detected debate domain for targeted retrieval
            env: Environment with task info
            agents: List of agents participating
            protocol: Debate protocol settings
        """
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager
            from aragora.events.types import StreamEvent, StreamEventType

            manager = get_cross_subscriber_manager()

            # Emit DEBATE_START to trigger KM→subsystem flows
            event = StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={
                    "debate_id": debate_id,
                    "domain": domain,
                    "question": env.task,
                    "agent_count": len(agents),
                    "protocol": {
                        "rounds": protocol.rounds,
                        "consensus": protocol.consensus,
                    },
                },
            )
            manager.dispatch(event)
            logger.debug(f"[arena] KM context initialized for debate {debate_id}")

        except ImportError:
            logger.debug("[arena] KM context initialization skipped (events not available)")
        except Exception as e:
            logger.warning(f"[arena] Failed to initialize KM context: {e}")

    def get_culture_hints(self, debate_id: str) -> Dict[str, Any]:
        """Retrieve culture hints from cross-subscriber manager.

        Args:
            debate_id: Debate identifier

        Returns:
            Dict of protocol hints derived from organizational culture
        """
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()
            hints = manager.get_debate_culture_hints(debate_id)
            if hints:
                logger.debug(f"[arena] Retrieved {len(hints)} culture hints for debate {debate_id}")
            return hints

        except ImportError:
            return {}
        except Exception as e:
            logger.debug(f"[arena] Failed to get culture hints: {e}")
            return {}

    def apply_culture_hints(self, hints: Dict[str, Any]) -> None:
        """Apply culture-derived hints to protocol and debate configuration.

        Args:
            hints: Protocol hints from organizational culture patterns
        """
        if not hints:
            return

        try:
            # Apply recommended consensus method if available
            if "recommended_consensus" in hints:
                recommended = hints["recommended_consensus"]
                if recommended in ("unanimous", "majority", "consensus"):
                    logger.info(f"[arena] Culture recommends {recommended} consensus")
                    self._culture_consensus_hint = recommended

            # Apply extra critique rounds for conservative cultures
            if hints.get("extra_critique_rounds", 0) > 0:
                extra = hints["extra_critique_rounds"]
                logger.info(f"[arena] Culture suggests {extra} extra critique rounds")
                self._culture_extra_critiques = extra

            # Apply early consensus threshold for aggressive cultures
            if "early_consensus_threshold" in hints:
                threshold = hints["early_consensus_threshold"]
                logger.info(f"[arena] Culture suggests early consensus at {threshold:.0%}")
                self._culture_early_consensus = threshold

            # Store domain-specific patterns
            if "domain_patterns" in hints:
                patterns = hints["domain_patterns"]
                logger.debug(f"[arena] Loaded {len(patterns)} domain-specific culture patterns")
                self._culture_domain_patterns = patterns

        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"[arena] Failed to apply culture hints (data error): {e}")
        except Exception as e:
            logger.warning(f"[arena] Unexpected error applying culture hints: {e}")

    async def fetch_context(self, task: str, limit: int = 10) -> Optional[str]:
        """Fetch relevant knowledge from Knowledge Mound for debate context.

        Args:
            task: The debate task/question
            limit: Maximum number of knowledge items to retrieve

        Returns:
            Formatted knowledge context string or None
        """
        if self._knowledge_ops is None:
            return None
        return await self._knowledge_ops.fetch_knowledge_context(task, limit)

    async def ingest_outcome(self, result: "DebateResult", env: "Environment") -> None:
        """Store debate outcome in Knowledge Mound for future retrieval.

        Args:
            result: The debate result to store
            env: Environment with task context
        """
        if self._knowledge_ops is None:
            return
        await self._knowledge_ops.ingest_debate_outcome(result, env=env)

    @property
    def culture_consensus_hint(self) -> Optional[str]:
        """Get culture-recommended consensus method."""
        return self._culture_consensus_hint

    @property
    def culture_extra_critiques(self) -> int:
        """Get culture-recommended extra critique rounds."""
        return self._culture_extra_critiques

    @property
    def culture_early_consensus(self) -> Optional[float]:
        """Get culture-recommended early consensus threshold."""
        return self._culture_early_consensus

    @property
    def culture_domain_patterns(self) -> Dict[str, Any]:
        """Get culture domain patterns."""
        return self._culture_domain_patterns
