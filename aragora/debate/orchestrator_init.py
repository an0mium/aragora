"""Constructor delegation helpers for Arena.__init__.

Extracted from orchestrator.py to reduce its size. Contains the post-config-merge
initialization logic: unpacking CoreComponents/TrackerComponents, storing config
flags, and running subsystem initialization sequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aragora.container import try_resolve, BudgetCoordinatorProtocol
from aragora.debate.budget_coordinator import BudgetCoordinator
from aragora.debate.event_bus import EventBus
from aragora.logging_config import get_logger as get_structured_logger

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena

logger = get_structured_logger(__name__)

# Sentinel for distinguishing "not provided" from explicit None
_KNOWLEDGE_MOUND_UNSET = object()


def apply_core_components(arena: Arena, core: Any) -> None:
    """Unpack CoreComponents dataclass to Arena instance attributes.

    Args:
        arena: Arena instance to populate.
        core: CoreComponents dataclass from ArenaInitializer.init_core().
    """
    arena.env = core.env
    arena.agents = core.agents
    arena.protocol = core.protocol
    arena.memory = core.memory
    arena.hooks = core.hooks
    arena.hook_manager = core.hook_manager
    arena.event_emitter = core.event_emitter
    arena.spectator = core.spectator
    arena.debate_embeddings = core.debate_embeddings
    arena.insight_store = core.insight_store
    arena.recorder = core.recorder
    arena.agent_weights = core.agent_weights
    arena.loop_id = core.loop_id
    arena.strict_loop_scoping = core.strict_loop_scoping
    arena.circuit_breaker = core.circuit_breaker
    arena.agent_pool = core.agent_pool
    arena.immune_system = core.immune_system
    arena.chaos_director = core.chaos_director
    arena.performance_monitor = core.performance_monitor
    arena.prompt_evolver = core.prompt_evolver
    arena.autonomic = core.autonomic
    arena.initial_messages = core.initial_messages
    arena.trending_topic = core.trending_topic
    arena.pulse_manager = core.pulse_manager
    arena.auto_fetch_trending = core.auto_fetch_trending
    arena.population_manager = core.population_manager
    arena.auto_evolve = core.auto_evolve
    arena.breeding_threshold = core.breeding_threshold
    arena.evidence_collector = core.evidence_collector
    arena.breakpoint_manager = core.breakpoint_manager
    arena.agent_selector = core.agent_selector
    arena.use_performance_selection = core.use_performance_selection
    arena.checkpoint_manager = core.checkpoint_manager
    arena.org_id = core.org_id
    arena.user_id = core.user_id
    # Try DI container first, fall back to direct instantiation
    arena._budget_coordinator = try_resolve(BudgetCoordinatorProtocol)
    if arena._budget_coordinator is None:
        arena._budget_coordinator = BudgetCoordinator(
            org_id=arena.org_id,
            user_id=arena.user_id,
        )
    else:
        # Configure resolved coordinator with org/user context
        arena._budget_coordinator.org_id = arena.org_id
        arena._budget_coordinator.user_id = arena.user_id
    arena.extensions = core.extensions
    arena.cartographer = core.cartographer
    arena.event_bridge = core.event_bridge
    # ML Integration
    arena.enable_ml_delegation = core.enable_ml_delegation
    arena.ml_delegation_weight = core.ml_delegation_weight
    arena.enable_quality_gates = core.enable_quality_gates
    arena.quality_gate_threshold = core.quality_gate_threshold
    arena.enable_consensus_estimation = core.enable_consensus_estimation
    arena.consensus_early_termination_threshold = core.consensus_early_termination_threshold
    arena._ml_delegation_strategy = core.ml_delegation_strategy
    arena._ml_quality_gate = core.ml_quality_gate
    arena._ml_consensus_estimator = core.ml_consensus_estimator
    # Event bus initialized later in _init_event_bus() after audience_manager exists
    arena.event_bus: EventBus | None = None


def apply_tracker_components(arena: Arena, trackers: Any) -> None:
    """Unpack TrackerComponents dataclass to Arena instance attributes.

    Args:
        arena: Arena instance to populate.
        trackers: TrackerComponents dataclass from ArenaInitializer.init_trackers().
    """
    arena.position_tracker = trackers.position_tracker
    arena.position_ledger = trackers.position_ledger
    arena.elo_system = trackers.elo_system
    arena.persona_manager = trackers.persona_manager
    arena.dissent_retriever = trackers.dissent_retriever
    arena.consensus_memory = trackers.consensus_memory
    arena.flip_detector = trackers.flip_detector
    arena.calibration_tracker = trackers.calibration_tracker
    arena.continuum_memory = trackers.continuum_memory
    arena.relationship_tracker = trackers.relationship_tracker
    arena.moment_detector = trackers.moment_detector
    arena.tier_analytics_tracker = trackers.tier_analytics_tracker
    arena.knowledge_mound = trackers.knowledge_mound
    arena.enable_knowledge_retrieval = trackers.enable_knowledge_retrieval
    arena.enable_knowledge_ingestion = trackers.enable_knowledge_ingestion
    arena.enable_knowledge_extraction = trackers.enable_knowledge_extraction
    arena.extraction_min_confidence = trackers.extraction_min_confidence
    arena.enable_belief_guidance = trackers.enable_belief_guidance
    arena._trackers = trackers.coordinator
    arena.vertical = trackers.vertical
    arena.vertical_persona_manager = trackers.vertical_persona_manager


def store_post_tracker_config(
    arena: Arena,
    cfg: Any,
    *,
    document_store: Any = None,
    evidence_store: Any = None,
) -> None:
    """Store additional config flags not tracked via CoreComponents or TrackerComponents.

    Args:
        arena: Arena instance to populate.
        cfg: MergedConfig from merge_config_objects.
        document_store: Optional document store for context injection.
        evidence_store: Optional evidence store for context injection.
    """
    arena.enable_auto_revalidation = cfg.enable_auto_revalidation
    arena.revalidation_staleness_threshold = cfg.revalidation_staleness_threshold
    arena.revalidation_check_interval_seconds = cfg.revalidation_check_interval_seconds
    arena.revalidation_scheduler = cfg.revalidation_scheduler
    # Document/evidence stores for context injection
    arena.document_store = document_store
    arena.evidence_store = evidence_store
    # Supermemory integration (external memory persistence)
    arena.enable_supermemory = cfg.enable_supermemory
    arena.supermemory_adapter = cfg.supermemory_adapter
    arena.supermemory_inject_on_start = cfg.supermemory_inject_on_start
    arena.supermemory_max_context_items = cfg.supermemory_max_context_items
    arena.supermemory_context_container_tag = cfg.supermemory_context_container_tag
    arena.supermemory_sync_on_conclusion = cfg.supermemory_sync_on_conclusion
    arena.supermemory_min_confidence_for_sync = cfg.supermemory_min_confidence_for_sync
    arena.supermemory_outcome_container_tag = cfg.supermemory_outcome_container_tag
    arena.supermemory_enable_privacy_filter = cfg.supermemory_enable_privacy_filter
    arena.supermemory_enable_resilience = cfg.supermemory_enable_resilience
    arena.supermemory_enable_km_adapter = cfg.supermemory_enable_km_adapter
    # Cross-debate institutional memory
    arena.cross_debate_memory = cfg.cross_debate_memory
    arena.enable_cross_debate_memory = cfg.enable_cross_debate_memory


def run_init_subsystems(arena: Arena) -> None:
    """Run the sequence of subsystem initialization calls on Arena.

    This handles the tail of __init__ after core/tracker setup:
    user participation, event bus, roles, convergence, caches,
    lifecycle, events, checkpoints, grounded ops, hierarchy,
    knowledge ops, RLM limiter, phases, context, and termination.

    Args:
        arena: Arena instance to initialize.
    """
    # Initialize user participation and roles
    arena._init_user_participation()
    arena._init_event_bus()
    arena._init_roles_and_stances()

    # Initialize convergence detection and caches
    arena._init_convergence()
    arena._init_caches()

    # Initialize extracted helper classes for lifecycle, events, and checkpoints
    arena._init_lifecycle_manager()
    arena._init_event_emitter()
    arena._init_checkpoint_ops()
    arena._init_checkpoint_bridge()

    # Initialize grounded operations helper (uses position_ledger, elo_system)
    arena._init_grounded_operations()

    # Initialize knowledge mound operations
    arena._init_knowledge_ops()

    # Initialize phase classes for orchestrator decomposition
    arena._init_phases()

    # Initialize prompt context builder (uses persona_manager, flip_detector, etc.)
    arena._init_prompt_context_builder()

    # Initialize context delegator (after phases since it needs evidence_grounder)
    arena._init_context_delegator()

    # Initialize termination checker
    arena._init_termination_checker()

    # Initialize cross-subscriber bridge for event cross-pollination
    arena._init_cross_subscriber_bridge()
