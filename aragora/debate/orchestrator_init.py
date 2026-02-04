"""Constructor delegation helpers for Arena.__init__.

Extracted from orchestrator.py to reduce its size. Contains the post-config-merge
initialization logic: unpacking CoreComponents/TrackerComponents, storing config
flags, and running subsystem initialization sequences.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.container import try_resolve, BudgetCoordinatorProtocol
from aragora.debate.budget_coordinator import BudgetCoordinator
from aragora.debate.convergence import ConvergenceDetector, cleanup_embedding_cache
from aragora.debate.event_bus import EventBus
from aragora.debate.prompt_context import PromptContextBuilder
from aragora.debate.context_delegation import ContextDelegator
from aragora.debate.roles_manager import RolesManager
from aragora.debate.termination_checker import TerminationChecker
from aragora.debate.audience_manager import AudienceManager
from aragora.logging_config import get_logger as get_structured_logger

if TYPE_CHECKING:
    from aragora.core_types import Agent, Message
    from aragora.debate.orchestrator import Arena

_conv_logger = logging.getLogger("aragora.debate.convergence")

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
    arena._budget_coordinator = try_resolve(BudgetCoordinatorProtocol)  # type: ignore[type-abstract]
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
    arena.event_bus = None


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


# =============================================================================
# Role rotation initialization (from orchestrator_roles.py)
# =============================================================================


def init_roles_and_stances(arena: Arena) -> None:
    """Initialize cognitive role rotation and agent stances."""
    arena.roles_manager = RolesManager(
        agents=arena.agents,
        protocol=arena.protocol,
        prompt_builder=arena.prompt_builder if hasattr(arena, "prompt_builder") else None,
        calibration_tracker=(
            arena.calibration_tracker if hasattr(arena, "calibration_tracker") else None
        ),
        persona_manager=arena.persona_manager if hasattr(arena, "persona_manager") else None,
    )
    arena.role_rotator = arena.roles_manager.role_rotator
    arena.role_matcher = arena.roles_manager.role_matcher
    arena.current_role_assignments = arena.roles_manager.current_role_assignments
    arena.roles_manager.assign_initial_roles()
    arena.roles_manager.assign_stances(round_num=0)
    arena.roles_manager.apply_agreement_intensity()


# =============================================================================
# User participation initialization (from orchestrator_participation.py)
# =============================================================================


def init_user_participation(arena: Arena) -> None:
    """Initialize user participation tracking and event subscription."""
    arena.audience_manager = AudienceManager(
        loop_id=arena.loop_id,
        strict_loop_scoping=arena.strict_loop_scoping,
    )
    arena.audience_manager.set_notify_callback(arena._notify_spectator)
    if arena.event_emitter:
        arena.audience_manager.subscribe_to_emitter(arena.event_emitter)


def init_event_bus(arena: Arena) -> None:
    """Initialize EventBus for pub/sub event handling."""
    arena.event_bus = EventBus(
        event_bridge=arena.event_bridge,
        audience_manager=arena.audience_manager,
        immune_system=arena.immune_system,
        spectator=arena.spectator,
    )


# =============================================================================
# Context building initialization (from orchestrator_context.py)
# =============================================================================


def init_prompt_context_builder(arena: Arena) -> None:
    """Initialize PromptContextBuilder for agent prompt context."""
    arena._prompt_context = PromptContextBuilder(
        persona_manager=arena.persona_manager,
        flip_detector=arena.flip_detector,
        protocol=arena.protocol,
        prompt_builder=arena.prompt_builder,
        audience_manager=arena.audience_manager,
        spectator=arena.spectator,
        notify_callback=arena._notify_spectator,
        vertical=getattr(arena, "vertical", None),
        vertical_persona_manager=getattr(arena, "vertical_persona_manager", None),
    )


def init_context_delegator(arena: Arena) -> None:
    """Initialize ContextDelegator for context gathering operations."""
    arena._context_delegator = ContextDelegator(
        context_gatherer=arena.context_gatherer,
        memory_manager=arena.memory_manager,
        cache=arena._cache,
        evidence_grounder=getattr(arena, "evidence_grounder", None),
        continuum_memory=arena.continuum_memory,
        env=arena.env,
        auth_context=getattr(arena, "auth_context", None),
        extract_domain_fn=arena._extract_debate_domain,
    )


# =============================================================================
# Termination checking initialization (from orchestrator_termination.py)
# =============================================================================


def init_termination_checker(arena: Arena) -> None:
    """Initialize the termination checker for early debate termination."""

    async def generate_fn(agent: Agent, prompt: str, ctx: list[Message]) -> str:
        return await arena.autonomic.generate(agent, prompt, ctx)

    async def select_judge_fn(proposals: dict[str, str], context: list[Message]) -> Agent:
        return await arena._select_judge(proposals, context)

    arena.termination_checker = TerminationChecker(
        protocol=arena.protocol,
        agents=arena._require_agents() if arena.agents else [],
        generate_fn=generate_fn,
        task=arena.env.task if arena.env else "",
        select_judge_fn=select_judge_fn,
        hooks=arena.hooks,
    )


# =============================================================================
# Convergence detection initialization (from orchestrator_convergence.py)
# =============================================================================


def init_convergence(arena: Arena, debate_id: str | None = None) -> None:
    """Initialize convergence detection if enabled."""
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
    """Reinitialize convergence detector with debate-specific cache."""
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
        _conv_logger.debug(f"Reinitialized convergence detector for debate {debate_id}")


def cleanup_convergence(arena: Arena) -> None:
    """Cleanup embedding cache for the current debate."""
    if arena._convergence_debate_id:
        cleanup_embedding_cache(arena._convergence_debate_id)
        _conv_logger.debug(f"Cleaned up embedding cache for debate {arena._convergence_debate_id}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constructor delegation
    "apply_core_components",
    "apply_tracker_components",
    "store_post_tracker_config",
    "run_init_subsystems",
    "_KNOWLEDGE_MOUND_UNSET",
    # Roles
    "init_roles_and_stances",
    # Participation
    "init_user_participation",
    "init_event_bus",
    # Context
    "init_prompt_context_builder",
    "init_context_delegator",
    # Termination
    "init_termination_checker",
    # Convergence
    "init_convergence",
    "reinit_convergence_for_debate",
    "cleanup_convergence",
]
