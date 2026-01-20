"""
Subsystem coordinator for Arena tracking and detection systems.

This module extracts subsystem management from the Arena god object,
following the Single Responsibility Principle. The coordinator handles:

1. **Position Tracking**: PositionTracker, PositionLedger
2. **Agent Ranking**: ELO system, calibration tracking
3. **Memory Systems**: ConsensusMemory, DissentRetriever, ContinuumMemory
4. **Detection Systems**: FlipDetector, MomentDetector
5. **Relationship Tracking**: RelationshipTracker, TierAnalyticsTracker

Usage:
    # Create coordinator with optional pre-configured subsystems
    coordinator = SubsystemCoordinator(
        protocol=protocol,
        loop_id="debate-123",
        elo_system=elo,  # Pre-configured
        enable_position_ledger=True,  # Auto-create
    )

    # Access subsystems (lazy initialization)
    ledger = coordinator.position_ledger
    if coordinator.has_calibration:
        tracker = coordinator.calibration_tracker

    # After debate, update tracking
    coordinator.on_debate_complete(ctx, result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.grounded import MomentDetector
    from aragora.agents.positions import PositionLedger
    from aragora.agents.truth_grounding import PositionTracker
    from aragora.core import DebateResult
    from aragora.insights.flip_detector import FlipDetector
    from aragora.debate.context import DebateContext
    from aragora.debate.protocol import DebateProtocol
    from aragora.memory.consensus import ConsensusMemory, DissentRetriever
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.tier_analytics import TierAnalyticsTracker
    from aragora.ranking.elo import EloSystem
    from aragora.relationships.tracker import RelationshipTracker


@runtime_checkable
class Resettable(Protocol):
    """Protocol for objects that can be reset."""

    def reset(self) -> None:
        """Reset internal state."""
        ...


logger = logging.getLogger(__name__)


@dataclass
class SubsystemCoordinator:
    """Coordinates tracking and detection subsystems for Arena.

    Provides a centralized place to manage optional subsystems that enhance
    debate capabilities. Handles lazy initialization and graceful fallbacks.

    Subsystems are grouped by function:

    **Position Systems** (track agent stances):
    - position_tracker: Real-time position tracking during debate
    - position_ledger: Persistent record of all positions across debates

    **Agent Ranking** (track agent skill):
    - elo_system: ELO ratings for agent skill ranking
    - calibration_tracker: Prediction accuracy tracking

    **Memory Systems** (cross-debate learning):
    - consensus_memory: Historical debate outcomes
    - dissent_retriever: Historical minority viewpoints
    - continuum_memory: Cross-debate learning memory

    **Detection Systems** (identify patterns):
    - flip_detector: Position reversal detection
    - moment_detector: Significant moment identification

    **Relationship Systems** (agent interactions):
    - relationship_tracker: Inter-agent relationship tracking
    - tier_analytics_tracker: Memory tier ROI analysis
    """

    # Protocol reference for breakpoint configuration
    protocol: Optional["DebateProtocol"] = None
    loop_id: str = ""

    # Position tracking subsystems
    position_tracker: Optional["PositionTracker"] = None
    position_ledger: Optional["PositionLedger"] = None
    enable_position_ledger: bool = False

    # Agent ranking subsystems
    elo_system: Optional["EloSystem"] = None
    calibration_tracker: Optional["CalibrationTracker"] = None
    enable_calibration: bool = False

    # Persona management
    persona_manager: Optional[Any] = None

    # Memory subsystems
    consensus_memory: Optional["ConsensusMemory"] = None
    dissent_retriever: Optional["DissentRetriever"] = None
    continuum_memory: Optional["ContinuumMemory"] = None

    # Detection subsystems
    flip_detector: Optional["FlipDetector"] = None
    moment_detector: Optional["MomentDetector"] = None
    enable_moment_detection: bool = False

    # Relationship subsystems
    relationship_tracker: Optional["RelationshipTracker"] = None
    tier_analytics_tracker: Optional["TierAnalyticsTracker"] = None

    # Hook system
    hook_manager: Optional[Any] = None  # HookManager for lifecycle hooks
    hook_handler_registry: Optional[Any] = None  # HookHandlerRegistry for auto-wiring
    enable_hook_handlers: bool = True  # Auto-register default handlers if hook_manager provided

    # ==========================================================================
    # Phase 9: Cross-Pollination Bridges
    # These bridges connect subsystems for self-improving feedback loops
    # ==========================================================================

    # Performance → Agent Router Bridge
    performance_router_bridge: Optional[Any] = None  # PerformanceRouterBridge
    enable_performance_router: bool = True  # Auto-create if performance_monitor available
    performance_monitor: Optional[Any] = None  # AgentPerformanceMonitor (source)
    agent_router: Optional[Any] = None  # AgentRouter (target)

    # Outcome → Complexity Governor Bridge
    outcome_complexity_bridge: Optional[Any] = None  # OutcomeComplexityBridge
    enable_outcome_complexity: bool = True  # Auto-create if outcome_tracker available
    outcome_tracker: Optional[Any] = None  # OutcomeTracker (source)
    complexity_governor: Optional[Any] = None  # ComplexityGovernor (target)

    # Analytics → Team Selection Bridge
    analytics_selection_bridge: Optional[Any] = None  # AnalyticsSelectionBridge
    enable_analytics_selection: bool = True  # Auto-create if analytics available
    analytics_coordinator: Optional[Any] = None  # AnalyticsCoordinator (source)
    team_selector: Optional[Any] = None  # TeamSelector (target)

    # Novelty → Selection Feedback Bridge
    novelty_selection_bridge: Optional[Any] = None  # NoveltySelectionBridge
    enable_novelty_selection: bool = True  # Auto-create if novelty_tracker available
    novelty_tracker: Optional[Any] = None  # NoveltyTracker (source)
    selection_feedback_loop: Optional[Any] = None  # SelectionFeedbackLoop (target)

    # Relationship → Bias Mitigation Bridge
    relationship_bias_bridge: Optional[Any] = None  # RelationshipBiasBridge
    enable_relationship_bias: bool = True  # Auto-create if relationship_tracker available
    # relationship_tracker already defined above (source)
    # bias_mitigation target is implicit in vote processing

    # RLM → Selection Feedback Bridge
    rlm_selection_bridge: Optional[Any] = None  # RLMSelectionBridge
    enable_rlm_selection: bool = True  # Auto-create if rlm_bridge available
    rlm_bridge: Optional[Any] = None  # RLMBridge (source)
    # selection_feedback_loop already defined above (target)

    # Calibration → Cost Optimizer Bridge
    calibration_cost_bridge: Optional[Any] = None  # CalibrationCostBridge
    enable_calibration_cost: bool = True  # Auto-create if calibration_tracker available
    # calibration_tracker already defined above (source)
    cost_tracker: Optional[Any] = None  # CostTracker (target)

    # ==========================================================================
    # Phase 10: Bidirectional Knowledge Mound Integration
    # ==========================================================================

    # Knowledge Mound core
    knowledge_mound: Optional[Any] = None  # KnowledgeMound instance
    enable_km_bidirectional: bool = True  # Master switch for bidirectional sync

    # Bidirectional Coordinator
    km_coordinator: Optional[Any] = None  # BidirectionalCoordinator
    enable_km_coordinator: bool = True  # Auto-create if KM available

    # KM Adapters (for manual configuration)
    km_continuum_adapter: Optional[Any] = None
    km_elo_adapter: Optional[Any] = None
    km_belief_adapter: Optional[Any] = None
    km_insights_adapter: Optional[Any] = None
    km_critique_adapter: Optional[Any] = None
    km_pulse_adapter: Optional[Any] = None

    # KM Configuration
    km_sync_interval_seconds: int = 300  # 5 minutes
    km_min_confidence_for_reverse: float = 0.7
    km_parallel_sync: bool = True

    # Internal state
    _initialized: bool = field(default=False, repr=False)
    _init_errors: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize subsystems after dataclass fields are set."""
        self._auto_init_subsystems()
        self._initialized = True

    # =========================================================================
    # Property accessors with capability checks
    # =========================================================================

    @property
    def has_position_tracking(self) -> bool:
        """Check if position tracking is available."""
        return self.position_tracker is not None or self.position_ledger is not None

    @property
    def has_elo(self) -> bool:
        """Check if ELO ranking is available."""
        return self.elo_system is not None

    @property
    def has_calibration(self) -> bool:
        """Check if calibration tracking is available."""
        return self.calibration_tracker is not None

    @property
    def has_consensus_memory(self) -> bool:
        """Check if consensus memory is available."""
        return self.consensus_memory is not None

    @property
    def has_dissent_retrieval(self) -> bool:
        """Check if dissent retrieval is available."""
        return self.dissent_retriever is not None

    @property
    def has_moment_detection(self) -> bool:
        """Check if moment detection is available."""
        return self.moment_detector is not None

    @property
    def has_relationship_tracking(self) -> bool:
        """Check if relationship tracking is available."""
        return self.relationship_tracker is not None

    @property
    def has_continuum_memory(self) -> bool:
        """Check if cross-debate memory is available."""
        return self.continuum_memory is not None

    # =========================================================================
    # Phase 9: Cross-Pollination Bridge Capability Checks
    # =========================================================================

    @property
    def has_performance_router_bridge(self) -> bool:
        """Check if performance-based routing bridge is available."""
        return self.performance_router_bridge is not None

    @property
    def has_outcome_complexity_bridge(self) -> bool:
        """Check if outcome-complexity bridge is available."""
        return self.outcome_complexity_bridge is not None

    @property
    def has_analytics_selection_bridge(self) -> bool:
        """Check if analytics-selection bridge is available."""
        return self.analytics_selection_bridge is not None

    @property
    def has_novelty_selection_bridge(self) -> bool:
        """Check if novelty-selection bridge is available."""
        return self.novelty_selection_bridge is not None

    @property
    def has_relationship_bias_bridge(self) -> bool:
        """Check if relationship-bias bridge is available."""
        return self.relationship_bias_bridge is not None

    @property
    def has_rlm_selection_bridge(self) -> bool:
        """Check if RLM-selection bridge is available."""
        return self.rlm_selection_bridge is not None

    @property
    def has_calibration_cost_bridge(self) -> bool:
        """Check if calibration-cost bridge is available."""
        return self.calibration_cost_bridge is not None

    @property
    def active_bridges_count(self) -> int:
        """Count of active cross-pollination bridges."""
        return sum(
            [
                self.has_performance_router_bridge,
                self.has_outcome_complexity_bridge,
                self.has_analytics_selection_bridge,
                self.has_novelty_selection_bridge,
                self.has_relationship_bias_bridge,
                self.has_rlm_selection_bridge,
                self.has_calibration_cost_bridge,
            ]
        )

    # =========================================================================
    # Phase 10: Knowledge Mound Capability Checks
    # =========================================================================

    @property
    def has_knowledge_mound(self) -> bool:
        """Check if Knowledge Mound is available."""
        return self.knowledge_mound is not None

    @property
    def has_km_coordinator(self) -> bool:
        """Check if KM bidirectional coordinator is available."""
        return self.km_coordinator is not None

    @property
    def has_km_bidirectional(self) -> bool:
        """Check if full KM bidirectional sync is available."""
        return self.has_knowledge_mound and self.has_km_coordinator

    @property
    def active_km_adapters_count(self) -> int:
        """Count of active KM adapters registered with coordinator."""
        adapters = [
            self.km_continuum_adapter,
            self.km_elo_adapter,
            self.km_belief_adapter,
            self.km_insights_adapter,
            self.km_critique_adapter,
            self.km_pulse_adapter,
        ]
        return sum(1 for a in adapters if a is not None)

    # =========================================================================
    # Auto-initialization methods
    # =========================================================================

    def _auto_init_subsystems(self) -> None:
        """Auto-initialize subsystems based on flags and dependencies."""
        # Position ledger
        if self.enable_position_ledger and self.position_ledger is None:
            self._auto_init_position_ledger()

        # Calibration tracker
        if self.enable_calibration and self.calibration_tracker is None:
            self._auto_init_calibration_tracker()

        # Dissent retriever (requires consensus_memory)
        if self.consensus_memory is not None and self.dissent_retriever is None:
            self._auto_init_dissent_retriever()

        # Moment detector (benefits from elo_system)
        if self.enable_moment_detection and self.moment_detector is None:
            self._auto_init_moment_detector()

        # Hook handler registry (requires hook_manager)
        if self.hook_manager is not None and self.enable_hook_handlers:
            self._auto_init_hook_handlers()

        # =======================================================================
        # Phase 9: Cross-Pollination Bridges
        # =======================================================================

        # Performance → Router bridge
        if self.enable_performance_router and self.performance_router_bridge is None:
            self._auto_init_performance_router_bridge()

        # Outcome → Complexity bridge
        if self.enable_outcome_complexity and self.outcome_complexity_bridge is None:
            self._auto_init_outcome_complexity_bridge()

        # Analytics → Selection bridge
        if self.enable_analytics_selection and self.analytics_selection_bridge is None:
            self._auto_init_analytics_selection_bridge()

        # Novelty → Selection Feedback bridge
        if self.enable_novelty_selection and self.novelty_selection_bridge is None:
            self._auto_init_novelty_selection_bridge()

        # Relationship → Bias Mitigation bridge
        if self.enable_relationship_bias and self.relationship_bias_bridge is None:
            self._auto_init_relationship_bias_bridge()

        # RLM → Selection Feedback bridge
        if self.enable_rlm_selection and self.rlm_selection_bridge is None:
            self._auto_init_rlm_selection_bridge()

        # Calibration → Cost bridge
        if self.enable_calibration_cost and self.calibration_cost_bridge is None:
            self._auto_init_calibration_cost_bridge()

        # =======================================================================
        # Phase 10: Bidirectional Knowledge Mound
        # =======================================================================

        # KM Bidirectional Coordinator
        if self.enable_km_coordinator and self.enable_km_bidirectional:
            if self.km_coordinator is None:
                self._auto_init_km_coordinator()

    def _auto_init_position_ledger(self) -> None:
        """Auto-initialize PositionLedger for tracking agent positions.

        PositionLedger tracks every position agents take across debates,
        including outcomes and reversals.
        """
        try:
            from aragora.agents.positions import PositionLedger

            self.position_ledger = PositionLedger()
            logger.debug("Auto-initialized PositionLedger for position tracking")
        except ImportError:
            logger.warning("PositionLedger not available - position tracking disabled")
            self._init_errors.append("PositionLedger import failed")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("PositionLedger auto-init failed: %s", e)
            self._init_errors.append(f"PositionLedger init failed: {e}")

    def _auto_init_calibration_tracker(self) -> None:
        """Auto-initialize CalibrationTracker for prediction accuracy."""
        try:
            from aragora.agents.calibration import CalibrationTracker

            self.calibration_tracker = CalibrationTracker()
            logger.debug("Auto-initialized CalibrationTracker for prediction calibration")
        except ImportError:
            logger.warning("CalibrationTracker not available - calibration disabled")
            self._init_errors.append("CalibrationTracker import failed")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("CalibrationTracker auto-init failed: %s", e)
            self._init_errors.append(f"CalibrationTracker init failed: {e}")

    def _auto_init_dissent_retriever(self) -> None:
        """Auto-initialize DissentRetriever for historical minority views.

        The DissentRetriever enables seeding new debates with historical minority
        views, helping agents avoid past groupthink.
        """
        try:
            from aragora.memory.consensus import DissentRetriever

            self.dissent_retriever = DissentRetriever(self.consensus_memory)
            logger.debug("Auto-initialized DissentRetriever for historical minority views")
        except ImportError:
            logger.debug("DissentRetriever not available - historical dissent disabled")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.warning("DissentRetriever auto-init failed: %s", e)
            self._init_errors.append(f"DissentRetriever init failed: {e}")

    def _auto_init_moment_detector(self) -> None:
        """Auto-initialize MomentDetector for significant moment detection."""
        try:
            from aragora.agents.grounded import MomentDetector

            self.moment_detector = MomentDetector(
                elo_system=self.elo_system,
                position_ledger=self.position_ledger,
                relationship_tracker=self.relationship_tracker,
            )
            logger.debug("Auto-initialized MomentDetector for significant moment detection")
        except ImportError:
            logger.debug("MomentDetector not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("MomentDetector auto-init failed: %s", e)
            self._init_errors.append(f"MomentDetector init failed: {e}")

    def _auto_init_hook_handlers(self) -> None:
        """Auto-initialize HookHandlerRegistry to wire subsystems to HookManager.

        Creates a registry that connects available subsystems to the hook lifecycle,
        enabling automatic event propagation across components.
        """
        if self.hook_handler_registry is not None:
            # Already have a registry
            return

        try:
            from aragora.debate.hook_handlers import HookHandlerRegistry

            # Collect available subsystems for the registry
            subsystems = {}
            if self.continuum_memory:
                subsystems["continuum_memory"] = self.continuum_memory
            if self.consensus_memory:
                subsystems["consensus_memory"] = self.consensus_memory
            if self.calibration_tracker:
                subsystems["calibration_tracker"] = self.calibration_tracker
            if self.flip_detector:
                subsystems["flip_detector"] = self.flip_detector
            if self.elo_system:
                subsystems["elo_system"] = self.elo_system
            if self.relationship_tracker:
                subsystems["relationship_tracker"] = self.relationship_tracker
            if self.tier_analytics_tracker:
                subsystems["tier_analytics_tracker"] = self.tier_analytics_tracker

            self.hook_handler_registry = HookHandlerRegistry(
                hook_manager=self.hook_manager,
                subsystems=subsystems,
            )
            count = self.hook_handler_registry.register_all()
            logger.debug(f"Auto-initialized HookHandlerRegistry with {count} handlers")
        except ImportError:
            logger.debug("HookHandlerRegistry not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("HookHandlerRegistry auto-init failed: %s", e)
            self._init_errors.append(f"HookHandlerRegistry init failed: {e}")

    # =========================================================================
    # Phase 9: Cross-Pollination Bridge Auto-Initialization
    # =========================================================================

    def _auto_init_performance_router_bridge(self) -> None:
        """Auto-initialize PerformanceRouterBridge for performance-based routing."""
        if self.performance_monitor is None:
            # No source data available
            return

        try:
            from aragora.debate.performance_router_bridge import (
                create_performance_router_bridge,
            )

            self.performance_router_bridge = create_performance_router_bridge(
                performance_monitor=self.performance_monitor,
                agent_router=self.agent_router,
            )
            logger.debug("Auto-initialized PerformanceRouterBridge")
        except ImportError:
            logger.debug("PerformanceRouterBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("PerformanceRouterBridge auto-init failed: %s", e)
            self._init_errors.append(f"PerformanceRouterBridge init failed: {e}")

    def _auto_init_outcome_complexity_bridge(self) -> None:
        """Auto-initialize OutcomeComplexityBridge for outcome-based complexity governance."""
        if self.outcome_tracker is None:
            # No source data available
            return

        try:
            from aragora.debate.outcome_complexity_bridge import (
                create_outcome_complexity_bridge,
            )

            self.outcome_complexity_bridge = create_outcome_complexity_bridge(
                outcome_tracker=self.outcome_tracker,
                complexity_governor=self.complexity_governor,
            )
            logger.debug("Auto-initialized OutcomeComplexityBridge")
        except ImportError:
            logger.debug("OutcomeComplexityBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("OutcomeComplexityBridge auto-init failed: %s", e)
            self._init_errors.append(f"OutcomeComplexityBridge init failed: {e}")

    def _auto_init_analytics_selection_bridge(self) -> None:
        """Auto-initialize AnalyticsSelectionBridge for analytics-driven team selection."""
        if self.analytics_coordinator is None:
            # No source data available
            return

        try:
            from aragora.debate.analytics_selection_bridge import (
                create_analytics_selection_bridge,
            )

            self.analytics_selection_bridge = create_analytics_selection_bridge(
                analytics_coordinator=self.analytics_coordinator,
                team_selector=self.team_selector,
            )
            logger.debug("Auto-initialized AnalyticsSelectionBridge")
        except ImportError:
            logger.debug("AnalyticsSelectionBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("AnalyticsSelectionBridge auto-init failed: %s", e)
            self._init_errors.append(f"AnalyticsSelectionBridge init failed: {e}")

    def _auto_init_novelty_selection_bridge(self) -> None:
        """Auto-initialize NoveltySelectionBridge for novelty-based selection feedback."""
        if self.novelty_tracker is None:
            # No source data available
            return

        try:
            from aragora.debate.novelty_selection_bridge import (
                create_novelty_selection_bridge,
            )

            self.novelty_selection_bridge = create_novelty_selection_bridge(
                novelty_tracker=self.novelty_tracker,
                selection_feedback=self.selection_feedback_loop,
            )
            logger.debug("Auto-initialized NoveltySelectionBridge")
        except ImportError:
            logger.debug("NoveltySelectionBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("NoveltySelectionBridge auto-init failed: %s", e)
            self._init_errors.append(f"NoveltySelectionBridge init failed: {e}")

    def _auto_init_relationship_bias_bridge(self) -> None:
        """Auto-initialize RelationshipBiasBridge for echo chamber detection and bias mitigation."""
        if self.relationship_tracker is None:
            # No source data available
            return

        try:
            from aragora.debate.relationship_bias_bridge import (
                create_relationship_bias_bridge,
            )

            self.relationship_bias_bridge = create_relationship_bias_bridge(
                relationship_tracker=self.relationship_tracker,
            )
            logger.debug("Auto-initialized RelationshipBiasBridge")
        except ImportError:
            logger.debug("RelationshipBiasBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("RelationshipBiasBridge auto-init failed: %s", e)
            self._init_errors.append(f"RelationshipBiasBridge init failed: {e}")

    def _auto_init_rlm_selection_bridge(self) -> None:
        """Auto-initialize RLMSelectionBridge for RLM-efficient agent selection."""
        if self.rlm_bridge is None:
            # No source data available
            return

        try:
            from aragora.rlm.rlm_selection_bridge import create_rlm_selection_bridge

            self.rlm_selection_bridge = create_rlm_selection_bridge(
                rlm_bridge=self.rlm_bridge,
                selection_feedback=self.selection_feedback_loop,
            )
            logger.debug("Auto-initialized RLMSelectionBridge")
        except ImportError:
            logger.debug("RLMSelectionBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("RLMSelectionBridge auto-init failed: %s", e)
            self._init_errors.append(f"RLMSelectionBridge init failed: {e}")

    def _auto_init_calibration_cost_bridge(self) -> None:
        """Auto-initialize CalibrationCostBridge for calibration-based cost optimization."""
        if self.calibration_tracker is None:
            # No source data available
            return

        try:
            from aragora.billing.calibration_cost_bridge import (
                create_calibration_cost_bridge,
            )

            self.calibration_cost_bridge = create_calibration_cost_bridge(
                calibration_tracker=self.calibration_tracker,
                cost_tracker=self.cost_tracker,
            )
            logger.debug("Auto-initialized CalibrationCostBridge")
        except ImportError:
            logger.debug("CalibrationCostBridge not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("CalibrationCostBridge auto-init failed: %s", e)
            self._init_errors.append(f"CalibrationCostBridge init failed: {e}")

    def _auto_init_km_coordinator(self) -> None:
        """Auto-initialize BidirectionalCoordinator for KM sync.

        BidirectionalCoordinator manages bidirectional data flow between
        the Knowledge Mound and connected subsystems (adapters).

        The coordinator is configured with:
        - sync_interval_seconds: How often to run bidirectional sync
        - min_confidence_for_reverse: Minimum confidence for reverse flow
        - parallel_sync: Whether to run adapter syncs in parallel

        After initialization, adapters are registered if available.
        """
        try:
            from aragora.knowledge.mound.bidirectional_coordinator import (
                BidirectionalCoordinator,
                CoordinatorConfig,
            )

            # Create configuration from SubsystemCoordinator fields
            config = CoordinatorConfig(
                sync_interval_seconds=self.km_sync_interval_seconds,
                min_confidence_for_reverse=self.km_min_confidence_for_reverse,
                parallel_sync=self.km_parallel_sync,
            )

            # Initialize coordinator with config and optional KM reference
            self.km_coordinator = BidirectionalCoordinator(
                config=config,
                knowledge_mound=self.knowledge_mound,
            )

            # Register available adapters
            self._register_km_adapters()

            logger.debug("Auto-initialized BidirectionalCoordinator for KM sync")
        except ImportError:
            logger.debug("BidirectionalCoordinator not available")
        except (TypeError, ValueError, RuntimeError) as e:
            logger.debug("BidirectionalCoordinator auto-init failed: %s", e)
            self._init_errors.append(f"BidirectionalCoordinator init failed: {e}")

    def _register_km_adapters(self) -> None:
        """Register available KM adapters with the BidirectionalCoordinator.

        Adapters are registered in priority order:
        1. ContinuumAdapter (highest impact on memory quality)
        2. ELOAdapter (critical for agent selection)
        3. BeliefAdapter (improves crux detection)
        4. InsightsAdapter (improves consistency analysis)
        5. CritiqueAdapter (boosts successful patterns)
        6. PulseAdapter (improves topic scheduling)

        Each adapter provides:
        - forward_method: Source → KM sync
        - reverse_method: KM → Source sync (optional)
        """
        if self.km_coordinator is None:
            return

        # Register pre-configured adapters or create them dynamically

        # 1. Continuum adapter (memory tier management)
        if self.km_continuum_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="continuum",
                    adapter=self.km_continuum_adapter,
                    forward_method="sync_to_km",
                    reverse_method="update_continuum_from_km",
                    priority=1,
                )
                logger.debug("Registered ContinuumAdapter with KM coordinator")
            except Exception as e:
                logger.debug("ContinuumAdapter registration failed: %s", e)

        # 2. ELO adapter (ranking adjustments)
        if self.km_elo_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="elo",
                    adapter=self.km_elo_adapter,
                    forward_method="sync_to_km",
                    reverse_method="update_elo_from_km_patterns",
                    priority=2,
                )
                logger.debug("Registered ELOAdapter with KM coordinator")
            except Exception as e:
                logger.debug("ELOAdapter registration failed: %s", e)

        # 3. Belief adapter (belief network calibration)
        if self.km_belief_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="belief",
                    adapter=self.km_belief_adapter,
                    forward_method="sync_to_km",
                    reverse_method="update_belief_thresholds_from_km",
                    priority=3,
                )
                logger.debug("Registered BeliefAdapter with KM coordinator")
            except Exception as e:
                logger.debug("BeliefAdapter registration failed: %s", e)

        # 4. Insights adapter (flip detection thresholds)
        if self.km_insights_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="insights",
                    adapter=self.km_insights_adapter,
                    forward_method="sync_to_km",
                    reverse_method="update_flip_thresholds_from_km",
                    priority=4,
                )
                logger.debug("Registered InsightsAdapter with KM coordinator")
            except Exception as e:
                logger.debug("InsightsAdapter registration failed: %s", e)

        # 5. Critique adapter (pattern boosting)
        if self.km_critique_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="critique",
                    adapter=self.km_critique_adapter,
                    forward_method="sync_to_km",
                    reverse_method="boost_pattern_from_km",
                    priority=5,
                )
                logger.debug("Registered CritiqueAdapter with KM coordinator")
            except Exception as e:
                logger.debug("CritiqueAdapter registration failed: %s", e)

        # 6. Pulse adapter (topic scheduling feedback)
        if self.km_pulse_adapter is not None:
            try:
                self.km_coordinator.register_adapter(
                    name="pulse",
                    adapter=self.km_pulse_adapter,
                    forward_method="sync_to_km",
                    reverse_method="sync_validations_from_km",
                    priority=6,
                )
                logger.debug("Registered PulseAdapter with KM coordinator")
            except Exception as e:
                logger.debug("PulseAdapter registration failed: %s", e)

        registered = (
            self.km_coordinator.adapter_count
            if hasattr(self.km_coordinator, "adapter_count")
            else 0
        )
        logger.debug("Registered %d KM adapters with coordinator", registered)

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================

    def on_debate_start(self, ctx: "DebateContext") -> None:
        """Called when a debate starts.

        Args:
            ctx: The debate context being initialized
        """
        # Reset moment detector for new debate if it supports reset
        if self.moment_detector and isinstance(self.moment_detector, Resettable):
            try:
                self.moment_detector.reset()
            except Exception as e:
                logger.debug("MomentDetector reset failed: %s", e)

    def on_round_complete(
        self,
        ctx: "DebateContext",
        round_num: int,
        positions: dict[str, str],
    ) -> None:
        """Called when a debate round completes.

        Args:
            ctx: The debate context
            round_num: The round number that completed
            positions: Agent name -> position mapping
        """
        # Record positions in ledger
        if self.position_ledger:
            for agent_name, position in positions.items():
                try:
                    self.position_ledger.record_position(
                        agent_name=agent_name,
                        claim=position,
                        confidence=0.5,  # Default confidence when not specified
                        debate_id=ctx.debate_id,
                        round_num=round_num,
                    )
                except Exception as e:
                    logger.debug("Position recording failed: %s", e)

    def on_debate_complete(
        self,
        ctx: "DebateContext",
        result: "DebateResult",
    ) -> None:
        """Called when a debate completes.

        Updates all tracking subsystems with debate outcome.

        Args:
            ctx: The debate context
            result: The final debate result
        """
        # Update consensus memory
        if self.consensus_memory and result:
            try:
                # Get task from environment
                task = ctx.env.task if ctx.env else ""
                consensus_text = getattr(result, "consensus", "") or ""
                confidence = getattr(result, "consensus_confidence", 0.0)
                participants = [a.name for a in ctx.agents] if ctx.agents else []

                # Import ConsensusStrength for the call
                from aragora.memory.consensus import ConsensusStrength

                # Determine strength based on confidence
                if confidence >= 0.9:
                    strength = ConsensusStrength.UNANIMOUS
                elif confidence >= 0.8:
                    strength = ConsensusStrength.STRONG
                elif confidence >= 0.6:
                    strength = ConsensusStrength.MODERATE
                elif confidence >= 0.5:
                    strength = ConsensusStrength.WEAK
                else:
                    strength = ConsensusStrength.SPLIT

                self.consensus_memory.store_consensus(
                    topic=task,
                    conclusion=consensus_text,
                    strength=strength,
                    confidence=confidence,
                    participating_agents=participants,
                    agreeing_agents=participants,  # Simplified: assume all agree at consensus
                    metadata={"debate_id": ctx.debate_id},
                )
            except Exception as e:
                logger.warning("Consensus memory update failed: %s", e)

        # Update calibration if agents made predictions
        if self.calibration_tracker and result:
            try:
                # Record prediction outcomes for calibration
                predictions: dict[str, Any] = getattr(result, "predictions", {})
                actual_outcome = getattr(result, "consensus", "")
                for agent_name, prediction in predictions.items():
                    # CalibrationTracker.record_prediction expects:
                    # (agent, confidence, correct, domain, debate_id, position_id)
                    predicted_value = (
                        prediction.get("prediction", "")
                        if isinstance(prediction, dict)
                        else str(prediction)
                    )
                    pred_confidence = (
                        prediction.get("confidence", 0.5) if isinstance(prediction, dict) else 0.5
                    )
                    is_correct = predicted_value == actual_outcome
                    self.calibration_tracker.record_prediction(
                        agent=agent_name,
                        confidence=pred_confidence,
                        correct=is_correct,
                        domain=ctx.domain,
                        debate_id=ctx.debate_id,
                    )
            except Exception as e:
                logger.debug("Calibration update failed: %s", e)

        # Update continuum memory with debate outcome
        if self.continuum_memory and result:
            try:
                # ContinuumMemory uses add() method
                # Store the debate outcome as a memory entry
                task = ctx.env.task if ctx.env else ""
                consensus_text = getattr(result, "consensus", "") or ""
                confidence = getattr(result, "consensus_confidence", 0.0)

                from aragora.memory.continuum import MemoryTier

                self.continuum_memory.add(
                    id=f"debate:{ctx.debate_id}",
                    content=f"Debate outcome: {consensus_text[:200]}",
                    tier=MemoryTier.MEDIUM,
                    importance=confidence,
                    metadata={
                        "debate_id": ctx.debate_id,
                        "task": task,
                        "consensus": consensus_text,
                        "confidence": confidence,
                    },
                )
            except Exception as e:
                logger.debug("Continuum memory update failed: %s", e)

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_historical_dissent(
        self,
        task: str,
        limit: int = 3,
    ) -> list[dict]:
        """Get historical minority viewpoints related to a task.

        Args:
            task: The debate task/question
            limit: Maximum number of dissenting views to return

        Returns:
            List of dissenting view records with position, agent, outcome
        """
        if not self.dissent_retriever:
            return []

        try:
            # DissentRetriever uses retrieve_for_new_debate() method
            result = self.dissent_retriever.retrieve_for_new_debate(task)
            # Extract relevant dissents from the result dict
            dissents = result.get("relevant_dissents", [])
            return dissents[:limit]
        except Exception as e:
            logger.debug("Dissent retrieval failed: %s", e)
            return []

    def get_agent_calibration_weight(self, agent_name: str) -> float:
        """Get calibration weight for an agent.

        Higher weights indicate better prediction accuracy.

        Args:
            agent_name: Name of the agent

        Returns:
            Weight between 0.5 and 2.0, default 1.0
        """
        if not self.calibration_tracker:
            return 1.0

        try:
            # CalibrationTracker uses get_calibration_summary() method
            summary = self.calibration_tracker.get_calibration_summary(agent_name)
            if summary and summary.total_predictions > 0:
                # Convert calibration score to weight
                # CalibrationSummary has brier_score (lower is better)
                # Convert: perfect (0.0) -> weight 1.5, poor (0.25) -> weight 0.8
                # Using 1 - brier_score as calibration quality
                calibration_quality = 1.0 - min(summary.brier_score, 0.5)
                return 0.5 + calibration_quality  # Range: 0.5 to 1.5
            return 1.0
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Could not get calibration weight for {agent_name}: {e}")
            return 1.0

    def get_continuum_context(self, task: str, limit: int = 5) -> str:
        """Get cross-debate context from continuum memory.

        Args:
            task: The debate task for context retrieval
            limit: Maximum number of relevant memories

        Returns:
            Formatted context string or empty string
        """
        if not self.continuum_memory:
            return ""

        try:
            memories = self.continuum_memory.retrieve(query=task, limit=limit)
            if not memories:
                return ""

            # Format memories for prompt injection
            # ContinuumMemory.retrieve() returns List[ContinuumMemoryEntry]
            lines = ["Relevant learnings from past debates:"]
            for mem in memories:
                # ContinuumMemoryEntry has content attribute and metadata dict
                summary = mem.metadata.get("summary", "") if mem.metadata else ""
                content = summary or mem.content
                lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Continuum context retrieval failed: %s", e)
            return ""

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @property
    def has_hook_handlers(self) -> bool:
        """Check if hook handlers are registered."""
        return self.hook_handler_registry is not None and getattr(
            self.hook_handler_registry, "is_registered", False
        )

    def get_status(self) -> dict:
        """Get status of all subsystems.

        Returns:
            Dictionary with subsystem availability and any init errors
        """
        hook_count = 0
        if self.hook_handler_registry:
            hook_count = getattr(self.hook_handler_registry, "registered_count", 0)

        return {
            "subsystems": {
                "position_tracker": self.position_tracker is not None,
                "position_ledger": self.position_ledger is not None,
                "elo_system": self.elo_system is not None,
                "calibration_tracker": self.calibration_tracker is not None,
                "consensus_memory": self.consensus_memory is not None,
                "dissent_retriever": self.dissent_retriever is not None,
                "continuum_memory": self.continuum_memory is not None,
                "flip_detector": self.flip_detector is not None,
                "moment_detector": self.moment_detector is not None,
                "relationship_tracker": self.relationship_tracker is not None,
                "tier_analytics_tracker": self.tier_analytics_tracker is not None,
                "persona_manager": self.persona_manager is not None,
                "hook_manager": self.hook_manager is not None,
                "hook_handler_registry": self.hook_handler_registry is not None,
            },
            "capabilities": {
                "position_tracking": self.has_position_tracking,
                "elo_ranking": self.has_elo,
                "calibration": self.has_calibration,
                "consensus_memory": self.has_consensus_memory,
                "dissent_retrieval": self.has_dissent_retrieval,
                "moment_detection": self.has_moment_detection,
                "relationship_tracking": self.has_relationship_tracking,
                "continuum_memory": self.has_continuum_memory,
                "hook_handlers": self.has_hook_handlers,
            },
            "cross_pollination_bridges": {
                "performance_router": self.has_performance_router_bridge,
                "outcome_complexity": self.has_outcome_complexity_bridge,
                "analytics_selection": self.has_analytics_selection_bridge,
                "novelty_selection": self.has_novelty_selection_bridge,
                "relationship_bias": self.has_relationship_bias_bridge,
                "rlm_selection": self.has_rlm_selection_bridge,
                "calibration_cost": self.has_calibration_cost_bridge,
            },
            "knowledge_mound": {
                "available": self.has_knowledge_mound,
                "coordinator_active": self.has_km_coordinator,
                "bidirectional_enabled": self.has_km_bidirectional,
                "adapters": {
                    "continuum": self.km_continuum_adapter is not None,
                    "elo": self.km_elo_adapter is not None,
                    "belief": self.km_belief_adapter is not None,
                    "insights": self.km_insights_adapter is not None,
                    "critique": self.km_critique_adapter is not None,
                    "pulse": self.km_pulse_adapter is not None,
                },
                "active_adapters_count": self.active_km_adapters_count,
                "config": {
                    "sync_interval_seconds": self.km_sync_interval_seconds,
                    "min_confidence_for_reverse": self.km_min_confidence_for_reverse,
                    "parallel_sync": self.km_parallel_sync,
                },
            },
            "active_bridges_count": self.active_bridges_count,
            "hook_handlers_registered": hook_count,
            "init_errors": self._init_errors,
            "initialized": self._initialized,
        }


@dataclass
class SubsystemConfig:
    """Configuration for creating SubsystemCoordinator.

    This provides a clean way to configure subsystems before
    creating the coordinator.
    """

    # Enable flags
    enable_position_ledger: bool = False
    enable_calibration: bool = False
    enable_moment_detection: bool = False
    enable_hook_handlers: bool = True

    # Phase 9: Cross-Pollination Bridge enable flags
    enable_performance_router: bool = True
    enable_outcome_complexity: bool = True
    enable_analytics_selection: bool = True
    enable_novelty_selection: bool = True
    enable_relationship_bias: bool = True
    enable_rlm_selection: bool = True
    enable_calibration_cost: bool = True

    # Pre-configured subsystems (optional)
    position_tracker: Optional[Any] = None
    position_ledger: Optional[Any] = None
    elo_system: Optional[Any] = None
    calibration_tracker: Optional[Any] = None
    persona_manager: Optional[Any] = None
    consensus_memory: Optional[Any] = None
    dissent_retriever: Optional[Any] = None
    continuum_memory: Optional[Any] = None
    flip_detector: Optional[Any] = None
    moment_detector: Optional[Any] = None
    relationship_tracker: Optional[Any] = None
    tier_analytics_tracker: Optional[Any] = None
    hook_manager: Optional[Any] = None
    hook_handler_registry: Optional[Any] = None

    # Phase 9: Cross-Pollination Bridge sources and pre-configured bridges
    performance_monitor: Optional[Any] = None
    agent_router: Optional[Any] = None
    performance_router_bridge: Optional[Any] = None
    outcome_tracker: Optional[Any] = None
    complexity_governor: Optional[Any] = None
    outcome_complexity_bridge: Optional[Any] = None
    analytics_coordinator: Optional[Any] = None
    team_selector: Optional[Any] = None
    analytics_selection_bridge: Optional[Any] = None
    novelty_tracker: Optional[Any] = None
    selection_feedback_loop: Optional[Any] = None
    novelty_selection_bridge: Optional[Any] = None
    relationship_bias_bridge: Optional[Any] = None
    rlm_bridge: Optional[Any] = None
    rlm_selection_bridge: Optional[Any] = None
    cost_tracker: Optional[Any] = None
    calibration_cost_bridge: Optional[Any] = None

    # Phase 10: Bidirectional Knowledge Mound Integration
    enable_km_bidirectional: bool = True  # Master switch for bidirectional sync
    enable_km_coordinator: bool = True  # Auto-create coordinator if KM available
    knowledge_mound: Optional[Any] = None  # KnowledgeMound instance
    km_coordinator: Optional[Any] = None  # BidirectionalCoordinator
    km_continuum_adapter: Optional[Any] = None
    km_elo_adapter: Optional[Any] = None
    km_belief_adapter: Optional[Any] = None
    km_insights_adapter: Optional[Any] = None
    km_critique_adapter: Optional[Any] = None
    km_pulse_adapter: Optional[Any] = None
    km_sync_interval_seconds: int = 300  # 5 minutes
    km_min_confidence_for_reverse: float = 0.7
    km_parallel_sync: bool = True

    def create_coordinator(
        self,
        protocol: Optional["DebateProtocol"] = None,
        loop_id: str = "",
    ) -> SubsystemCoordinator:
        """Create SubsystemCoordinator from this configuration.

        Args:
            protocol: The debate protocol (for breakpoint config)
            loop_id: Loop ID for multi-loop scoping

        Returns:
            Configured SubsystemCoordinator instance
        """
        return SubsystemCoordinator(
            protocol=protocol,
            loop_id=loop_id,
            position_tracker=self.position_tracker,
            position_ledger=self.position_ledger,
            enable_position_ledger=self.enable_position_ledger,
            elo_system=self.elo_system,
            calibration_tracker=self.calibration_tracker,
            enable_calibration=self.enable_calibration,
            persona_manager=self.persona_manager,
            consensus_memory=self.consensus_memory,
            dissent_retriever=self.dissent_retriever,
            continuum_memory=self.continuum_memory,
            flip_detector=self.flip_detector,
            moment_detector=self.moment_detector,
            enable_moment_detection=self.enable_moment_detection,
            relationship_tracker=self.relationship_tracker,
            tier_analytics_tracker=self.tier_analytics_tracker,
            hook_manager=self.hook_manager,
            hook_handler_registry=self.hook_handler_registry,
            enable_hook_handlers=self.enable_hook_handlers,
            # Phase 9: Cross-Pollination Bridges
            performance_monitor=self.performance_monitor,
            agent_router=self.agent_router,
            performance_router_bridge=self.performance_router_bridge,
            enable_performance_router=self.enable_performance_router,
            outcome_tracker=self.outcome_tracker,
            complexity_governor=self.complexity_governor,
            outcome_complexity_bridge=self.outcome_complexity_bridge,
            enable_outcome_complexity=self.enable_outcome_complexity,
            analytics_coordinator=self.analytics_coordinator,
            team_selector=self.team_selector,
            analytics_selection_bridge=self.analytics_selection_bridge,
            enable_analytics_selection=self.enable_analytics_selection,
            novelty_tracker=self.novelty_tracker,
            selection_feedback_loop=self.selection_feedback_loop,
            novelty_selection_bridge=self.novelty_selection_bridge,
            enable_novelty_selection=self.enable_novelty_selection,
            relationship_bias_bridge=self.relationship_bias_bridge,
            enable_relationship_bias=self.enable_relationship_bias,
            rlm_bridge=self.rlm_bridge,
            rlm_selection_bridge=self.rlm_selection_bridge,
            enable_rlm_selection=self.enable_rlm_selection,
            cost_tracker=self.cost_tracker,
            calibration_cost_bridge=self.calibration_cost_bridge,
            enable_calibration_cost=self.enable_calibration_cost,
            # Phase 10: Bidirectional Knowledge Mound
            enable_km_bidirectional=self.enable_km_bidirectional,
            enable_km_coordinator=self.enable_km_coordinator,
            knowledge_mound=self.knowledge_mound,
            km_coordinator=self.km_coordinator,
            km_continuum_adapter=self.km_continuum_adapter,
            km_elo_adapter=self.km_elo_adapter,
            km_belief_adapter=self.km_belief_adapter,
            km_insights_adapter=self.km_insights_adapter,
            km_critique_adapter=self.km_critique_adapter,
            km_pulse_adapter=self.km_pulse_adapter,
            km_sync_interval_seconds=self.km_sync_interval_seconds,
            km_min_confidence_for_reverse=self.km_min_confidence_for_reverse,
            km_parallel_sync=self.km_parallel_sync,
        )


__all__ = ["SubsystemCoordinator", "SubsystemConfig"]
