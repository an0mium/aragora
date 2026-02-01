"""
Multi-agent debate orchestrator.

Implements the propose -> critique -> revise loop with configurable
debate protocols and consensus mechanisms.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.arena_config import (
    ArenaConfig,
    AgentConfig,
    DebateConfig,
    MemoryConfig,
    ObservabilityConfig,
    StreamingConfig,
)
from aragora.debate.arena_initializer import ArenaInitializer
from aragora.debate.arena_phases import create_phase_executor, init_phases
from aragora.debate.batch_loaders import debate_loader_context
from aragora.debate.budget_coordinator import BudgetCoordinator
from aragora.debate.knowledge_manager import ArenaKnowledgeManager
from aragora.debate.context import DebateContext
from aragora.debate.context_delegation import ContextDelegator
from aragora.debate.grounded_operations import GroundedOperations
from aragora.debate.hierarchy import AgentHierarchy, HierarchyConfig
from aragora.debate.prompt_context import PromptContextBuilder
from aragora.debate.event_bus import EventBus
from aragora.debate.judge_selector import JudgeSelector
from aragora.debate.protocol import CircuitBreaker, DebateProtocol
from aragora.debate.sanitization import OutputSanitizer
from aragora.debate.termination_checker import TerminationChecker
from aragora.logging_config import get_logger as get_structured_logger
from aragora.observability.n1_detector import n1_detection_scope
from aragora.observability.tracing import add_span_attributes, get_tracer
from aragora.debate.performance_monitor import get_debate_monitor
from aragora.server.metrics import ACTIVE_DEBATES
from aragora.spectate.stream import SpectatorStream

# Extracted sibling modules
from aragora.debate.orchestrator_agents import (
    assign_hierarchy_roles as _agents_assign_hierarchy_roles,
    filter_responses_by_quality as _agents_filter_responses_by_quality,
    get_fabric_agents_sync as _agents_get_fabric_agents_sync,
    init_agent_hierarchy as _agents_init_agent_hierarchy,
    select_debate_team as _agents_select_debate_team,
    should_terminate_early as _agents_should_terminate_early,
)
from aragora.debate.orchestrator_checkpoints import (
    cleanup_checkpoints as _cp_cleanup_checkpoints,
    list_checkpoints as _cp_list_checkpoints,
    restore_from_checkpoint as _cp_restore_from_checkpoint,
    save_checkpoint as _cp_save_checkpoint,
)
from aragora.debate.orchestrator_config import merge_config_objects
from aragora.debate.orchestrator_convergence import (
    cleanup_convergence as _conv_cleanup_convergence,
    init_convergence as _conv_init_convergence,
    reinit_convergence_for_debate as _conv_reinit_convergence_for_debate,
)
from aragora.debate.orchestrator_lifecycle import (
    init_caches as _lifecycle_init_caches,
    init_checkpoint_ops as _lifecycle_init_checkpoint_ops,
    init_event_emitter as _lifecycle_init_event_emitter,
    init_lifecycle_manager as _lifecycle_init_lifecycle_manager,
)
from aragora.debate.orchestrator_participation import (
    init_event_bus as _participation_init_event_bus,
    init_user_participation as _participation_init_user_participation,
)
from aragora.debate.orchestrator_roles import (
    init_roles_and_stances as _roles_init_roles_and_stances,
)
from aragora.debate.orchestrator_delegates import ArenaDelegatesMixin
from aragora.debate.orchestrator_domains import (
    compute_domain_from_task as _compute_domain_from_task,
)
from aragora.debate.orchestrator_memory import (
    auto_create_knowledge_mound as _mem_auto_create_knowledge_mound,
    init_checkpoint_bridge as _mem_init_checkpoint_bridge,
    init_cross_subscriber_bridge as _mem_init_cross_subscriber_bridge,
    init_rlm_limiter_state as _mem_init_rlm_limiter_state,
)
from aragora.debate.orchestrator_runner import (
    cleanup_debate_resources as _runner_cleanup_debate_resources,
    execute_debate_phases as _runner_execute_debate_phases,
    handle_debate_completion as _runner_handle_debate_completion,
    initialize_debate_context as _runner_initialize_debate_context,
    record_debate_metrics as _runner_record_debate_metrics,
    setup_debate_infrastructure as _runner_setup_debate_infrastructure,
)

# Structured logger for all debate events (JSON-formatted in production)
logger = get_structured_logger(__name__)

# TYPE_CHECKING imports for type hints without runtime import overhead
if TYPE_CHECKING:
    from aragora.debate.checkpoint_manager import CheckpointManager
    from aragora.debate.context_gatherer import ContextGatherer
    from aragora.debate.memory_manager import MemoryManager
    from aragora.debate.phases import (
        AnalyticsPhase,
        ConsensusPhase,
        ContextInitializer,
        DebateRoundsPhase,
        FeedbackPhase,
        ProposalPhase,
        VotingPhase,
    )
    from aragora.debate.prompt_builder import PromptBuilder
    from aragora.debate.revalidation_scheduler import RevalidationScheduler
    from aragora.debate.strategy import DebateStrategy
    from aragora.knowledge.mound.core import KnowledgeMound
    from aragora.memory.consensus import ConsensusMemory
    from aragora.memory.continuum import ContinuumMemory
    from aragora.ml.delegation import MLDelegationStrategy
    from aragora.ranking.elo import EloSystem
    from aragora.reasoning.citations import CitationExtractor
    from aragora.reasoning.evidence_grounding import EvidenceGrounder
    from aragora.rlm.cognitive_limiter import RLMCognitiveLoadLimiter
    from aragora.types.protocols import EventEmitterProtocol
    from aragora.workflow.engine import Workflow


class Arena(ArenaDelegatesMixin):
    """
    Orchestrates multi-agent debates.

    The Arena manages the flow of a debate:
    1. Proposers generate initial proposals
    2. Critics critique each proposal
    3. Proposers revise based on critique
    4. Repeat for configured rounds
    5. Consensus mechanism selects final answer

    Configuration Patterns
    ----------------------
    Arena supports two configuration patterns:

    1. **Config Objects (Recommended)** - Group related parameters for cleaner code::

        from aragora.debate.arena_config import (
            DebateConfig, AgentConfig, MemoryConfig,
            StreamingConfig, ObservabilityConfig
        )

        arena = Arena(
            environment=env,
            agents=agents,
            debate_config=DebateConfig(rounds=5, consensus_threshold=0.8),
            agent_config=AgentConfig(use_airlock=True),
            memory_config=MemoryConfig(enable_knowledge_retrieval=True),
            streaming_config=StreamingConfig(loop_id="debate-123"),
            observability_config=ObservabilityConfig(enable_telemetry=True),
        )
        result = await arena.run()

    2. **Individual Parameters (Legacy)** - Still supported for backward compatibility::

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            use_airlock=True,
            enable_knowledge_retrieval=True,
            loop_id="debate-123",
        )

    Factory Methods
    ---------------
    - ``Arena.from_configs()`` - Create from config objects (preferred)
    - ``Arena.from_config()`` - Create from ArenaConfig (legacy)
    """

    # Phase class attributes (initialized by init_phases)
    voting_phase: "VotingPhase"
    citation_extractor: Optional["CitationExtractor"]
    evidence_grounder: "EvidenceGrounder"
    prompt_builder: "PromptBuilder"
    memory_manager: "MemoryManager"
    context_gatherer: "ContextGatherer"
    context_initializer: "ContextInitializer"
    proposal_phase: "ProposalPhase"
    debate_rounds_phase: "DebateRoundsPhase"
    consensus_phase: "ConsensusPhase"
    analytics_phase: "AnalyticsPhase"
    feedback_phase: "FeedbackPhase"

    # Convergence attributes (initialized by orchestrator_convergence.init_convergence)
    convergence_detector: Optional[Any]
    _convergence_debate_id: Optional[str]
    _previous_round_responses: dict[str, str]

    # Role attributes (initialized by orchestrator_roles.init_roles_and_stances)
    roles_manager: Any
    role_rotator: Any
    role_matcher: Any
    current_role_assignments: Any

    def __init__(
        self,
        environment: Environment,
        agents: list[Agent],
        protocol: Optional[DebateProtocol] = None,
        # =====================================================================
        # Config Objects (Preferred - cleaner interface)
        # =====================================================================
        debate_config: Optional["DebateConfig"] = None,
        agent_config: Optional["AgentConfig"] = None,
        memory_config: Optional["MemoryConfig"] = None,
        streaming_config: Optional["StreamingConfig"] = None,
        observability_config: Optional["ObservabilityConfig"] = None,
        # =====================================================================
        # Individual Parameters (Deprecated - use config objects instead)
        # =====================================================================
        memory: Any = None,
        event_hooks: Optional[dict[str, Any]] = None,
        hook_manager: Any = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
        spectator: Optional[SpectatorStream] = None,
        debate_embeddings: Any = None,
        insight_store: Any = None,
        recorder: Any = None,
        agent_weights: Optional[dict[str, float]] = None,
        position_tracker: Any = None,
        position_ledger: Any = None,
        enable_position_ledger: bool = False,
        elo_system: Optional["EloSystem"] = None,
        persona_manager: Any = None,
        vertical: Optional[str] = None,
        vertical_persona_manager: Any = None,
        auto_detect_vertical: bool = True,
        dissent_retriever: Any = None,
        consensus_memory: Optional["ConsensusMemory"] = None,
        flip_detector: Any = None,
        calibration_tracker: Any = None,
        continuum_memory: Optional["ContinuumMemory"] = None,
        relationship_tracker: Any = None,
        moment_detector: Any = None,
        tier_analytics_tracker: Any = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        auto_create_knowledge_mound: bool = True,
        enable_knowledge_retrieval: bool = True,
        enable_knowledge_ingestion: bool = True,
        enable_knowledge_extraction: bool = False,
        extraction_min_confidence: float = 0.3,
        enable_belief_guidance: bool = False,
        enable_auto_revalidation: bool = False,
        revalidation_staleness_threshold: float = 0.8,
        revalidation_check_interval_seconds: int = 3600,
        revalidation_scheduler: Optional["RevalidationScheduler"] = None,
        loop_id: str = "",
        strict_loop_scoping: bool = False,
        circuit_breaker: Optional[CircuitBreaker] = None,
        initial_messages: Optional[list[Message]] = None,
        trending_topic: Any = None,
        pulse_manager: Any = None,
        auto_fetch_trending: bool = False,
        population_manager: Any = None,
        auto_evolve: bool = False,
        breeding_threshold: float = 0.8,
        evidence_collector: Any = None,
        skill_registry: Any = None,
        enable_skills: bool = False,
        propulsion_engine: Any = None,
        enable_propulsion: bool = False,
        breakpoint_manager: Any = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        enable_checkpointing: bool = True,
        performance_monitor: Any = None,
        enable_performance_monitor: bool = True,
        enable_telemetry: bool = False,
        use_airlock: bool = False,
        airlock_config: Any = None,
        agent_selector: Any = None,
        use_performance_selection: bool = False,
        enable_agent_hierarchy: bool = True,
        hierarchy_config: Optional[HierarchyConfig] = None,
        prompt_evolver: Any = None,
        enable_prompt_evolution: bool = False,
        org_id: str = "",
        user_id: str = "",
        usage_tracker: Any = None,
        broadcast_pipeline: Any = None,
        auto_broadcast: bool = False,
        broadcast_min_confidence: float = 0.8,
        training_exporter: Any = None,
        auto_export_training: bool = False,
        training_export_min_confidence: float = 0.75,
        enable_ml_delegation: bool = True,
        ml_delegation_strategy: Optional["MLDelegationStrategy"] = None,
        ml_delegation_weight: float = 0.3,
        enable_quality_gates: bool = True,
        quality_gate_threshold: float = 0.6,
        enable_consensus_estimation: bool = True,
        consensus_early_termination_threshold: float = 0.85,
        use_rlm_limiter: bool = True,
        rlm_limiter: Optional["RLMCognitiveLoadLimiter"] = None,
        rlm_compression_threshold: int = 3000,
        rlm_max_recent_messages: int = 5,
        rlm_summary_level: str = "SUMMARY",
        rlm_compression_round_threshold: int = 3,
        enable_adaptive_rounds: bool = False,
        debate_strategy: Optional["DebateStrategy"] = None,
        cross_debate_memory: Any = None,
        enable_cross_debate_memory: bool = True,
        post_debate_workflow: Optional["Workflow"] = None,
        enable_post_debate_workflow: bool = False,
        post_debate_workflow_threshold: float = 0.7,
        fabric: Any = None,
        fabric_config: Any = None,
    ) -> None:
        """Initialize the Arena with environment, agents, and optional subsystems.

        See inline parameter comments for subsystem descriptions.
        Initialization delegates to ArenaInitializer for core/tracker setup.
        """
        # =====================================================================
        # Config Object Merging (config objects take precedence over individual params)
        # =====================================================================
        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=agent_config,
            memory_config=memory_config,
            streaming_config=streaming_config,
            observability_config=observability_config,
            protocol=protocol,
            enable_adaptive_rounds=enable_adaptive_rounds,
            debate_strategy=debate_strategy,
            enable_agent_hierarchy=enable_agent_hierarchy,
            hierarchy_config=hierarchy_config,
            agent_weights=agent_weights,
            agent_selector=agent_selector,
            use_performance_selection=use_performance_selection,
            circuit_breaker=circuit_breaker,
            use_airlock=use_airlock,
            airlock_config=airlock_config,
            position_tracker=position_tracker,
            position_ledger=position_ledger,
            enable_position_ledger=enable_position_ledger,
            elo_system=elo_system,
            calibration_tracker=calibration_tracker,
            relationship_tracker=relationship_tracker,
            persona_manager=persona_manager,
            vertical=vertical,
            vertical_persona_manager=vertical_persona_manager,
            auto_detect_vertical=auto_detect_vertical,
            fabric=fabric,
            fabric_config=fabric_config,
            memory=memory,
            continuum_memory=continuum_memory,
            consensus_memory=consensus_memory,
            debate_embeddings=debate_embeddings,
            insight_store=insight_store,
            dissent_retriever=dissent_retriever,
            flip_detector=flip_detector,
            moment_detector=moment_detector,
            tier_analytics_tracker=tier_analytics_tracker,
            cross_debate_memory=cross_debate_memory,
            enable_cross_debate_memory=enable_cross_debate_memory,
            knowledge_mound=knowledge_mound,
            auto_create_knowledge_mound=auto_create_knowledge_mound,
            enable_knowledge_retrieval=enable_knowledge_retrieval,
            enable_knowledge_ingestion=enable_knowledge_ingestion,
            enable_knowledge_extraction=enable_knowledge_extraction,
            extraction_min_confidence=extraction_min_confidence,
            enable_belief_guidance=enable_belief_guidance,
            enable_auto_revalidation=enable_auto_revalidation,
            revalidation_staleness_threshold=revalidation_staleness_threshold,
            revalidation_check_interval_seconds=revalidation_check_interval_seconds,
            revalidation_scheduler=revalidation_scheduler,
            use_rlm_limiter=use_rlm_limiter,
            rlm_limiter=rlm_limiter,
            rlm_compression_threshold=rlm_compression_threshold,
            rlm_max_recent_messages=rlm_max_recent_messages,
            rlm_summary_level=rlm_summary_level,
            rlm_compression_round_threshold=rlm_compression_round_threshold,
            checkpoint_manager=checkpoint_manager,
            enable_checkpointing=enable_checkpointing,
            event_hooks=event_hooks,
            hook_manager=hook_manager,
            event_emitter=event_emitter,
            spectator=spectator,
            recorder=recorder,
            loop_id=loop_id,
            strict_loop_scoping=strict_loop_scoping,
            skill_registry=skill_registry,
            enable_skills=enable_skills,
            propulsion_engine=propulsion_engine,
            enable_propulsion=enable_propulsion,
            performance_monitor=performance_monitor,
            enable_performance_monitor=enable_performance_monitor,
            enable_telemetry=enable_telemetry,
            prompt_evolver=prompt_evolver,
            enable_prompt_evolution=enable_prompt_evolution,
            breakpoint_manager=breakpoint_manager,
            trending_topic=trending_topic,
            pulse_manager=pulse_manager,
            auto_fetch_trending=auto_fetch_trending,
            population_manager=population_manager,
            auto_evolve=auto_evolve,
            breeding_threshold=breeding_threshold,
            evidence_collector=evidence_collector,
            org_id=org_id,
            user_id=user_id,
            usage_tracker=usage_tracker,
            broadcast_pipeline=broadcast_pipeline,
            auto_broadcast=auto_broadcast,
            broadcast_min_confidence=broadcast_min_confidence,
            training_exporter=training_exporter,
            auto_export_training=auto_export_training,
            training_export_min_confidence=training_export_min_confidence,
            enable_ml_delegation=enable_ml_delegation,
            ml_delegation_strategy=ml_delegation_strategy,
            ml_delegation_weight=ml_delegation_weight,
            enable_quality_gates=enable_quality_gates,
            quality_gate_threshold=quality_gate_threshold,
            enable_consensus_estimation=enable_consensus_estimation,
            consensus_early_termination_threshold=consensus_early_termination_threshold,
            post_debate_workflow=post_debate_workflow,
            enable_post_debate_workflow=enable_post_debate_workflow,
            post_debate_workflow_threshold=post_debate_workflow_threshold,
            initial_messages=initial_messages,
        )

        # Unpack merged config back to local variables for use below
        # (config objects may have overridden individual params)
        fabric = cfg.fabric
        fabric_config = cfg.fabric_config
        agent_weights = cfg.agent_weights
        use_airlock = cfg.use_airlock
        airlock_config = cfg.airlock_config
        use_performance_selection = cfg.use_performance_selection
        agent_selector = cfg.agent_selector
        circuit_breaker = cfg.circuit_breaker
        memory = cfg.memory
        event_hooks = cfg.event_hooks
        hook_manager = cfg.hook_manager
        event_emitter = cfg.event_emitter
        spectator = cfg.spectator
        recorder = cfg.recorder
        loop_id = cfg.loop_id
        strict_loop_scoping = cfg.strict_loop_scoping
        initial_messages = cfg.initial_messages
        debate_embeddings = cfg.debate_embeddings
        insight_store = cfg.insight_store
        trending_topic = cfg.trending_topic
        pulse_manager = cfg.pulse_manager
        auto_fetch_trending = cfg.auto_fetch_trending
        population_manager = cfg.population_manager
        auto_evolve = cfg.auto_evolve
        breeding_threshold = cfg.breeding_threshold
        evidence_collector = cfg.evidence_collector
        breakpoint_manager = cfg.breakpoint_manager
        checkpoint_manager = cfg.checkpoint_manager
        enable_checkpointing = cfg.enable_checkpointing
        performance_monitor = cfg.performance_monitor
        enable_performance_monitor = cfg.enable_performance_monitor
        enable_telemetry = cfg.enable_telemetry
        prompt_evolver = cfg.prompt_evolver
        enable_prompt_evolution = cfg.enable_prompt_evolution
        org_id = cfg.org_id
        user_id = cfg.user_id
        usage_tracker = cfg.usage_tracker
        broadcast_pipeline = cfg.broadcast_pipeline
        auto_broadcast = cfg.auto_broadcast
        broadcast_min_confidence = cfg.broadcast_min_confidence
        training_exporter = cfg.training_exporter
        auto_export_training = cfg.auto_export_training
        training_export_min_confidence = cfg.training_export_min_confidence
        enable_ml_delegation = cfg.enable_ml_delegation
        ml_delegation_strategy = cfg.ml_delegation_strategy
        ml_delegation_weight = cfg.ml_delegation_weight
        enable_quality_gates = cfg.enable_quality_gates
        quality_gate_threshold = cfg.quality_gate_threshold
        enable_consensus_estimation = cfg.enable_consensus_estimation
        consensus_early_termination_threshold = cfg.consensus_early_termination_threshold
        position_tracker = cfg.position_tracker
        position_ledger = cfg.position_ledger
        enable_position_ledger = cfg.enable_position_ledger
        elo_system = cfg.elo_system
        persona_manager = cfg.persona_manager
        dissent_retriever = cfg.dissent_retriever
        consensus_memory = cfg.consensus_memory
        flip_detector = cfg.flip_detector
        calibration_tracker = cfg.calibration_tracker
        continuum_memory = cfg.continuum_memory
        relationship_tracker = cfg.relationship_tracker
        moment_detector = cfg.moment_detector
        tier_analytics_tracker = cfg.tier_analytics_tracker
        knowledge_mound = cfg.knowledge_mound
        auto_create_knowledge_mound = cfg.auto_create_knowledge_mound
        enable_knowledge_retrieval = cfg.enable_knowledge_retrieval
        enable_knowledge_ingestion = cfg.enable_knowledge_ingestion
        enable_knowledge_extraction = cfg.enable_knowledge_extraction
        extraction_min_confidence = cfg.extraction_min_confidence
        enable_belief_guidance = cfg.enable_belief_guidance
        vertical = cfg.vertical
        vertical_persona_manager = cfg.vertical_persona_manager
        auto_detect_vertical = cfg.auto_detect_vertical
        enable_auto_revalidation = cfg.enable_auto_revalidation
        revalidation_staleness_threshold = cfg.revalidation_staleness_threshold
        revalidation_check_interval_seconds = cfg.revalidation_check_interval_seconds
        revalidation_scheduler = cfg.revalidation_scheduler
        enable_adaptive_rounds = cfg.enable_adaptive_rounds
        debate_strategy = cfg.debate_strategy
        cross_debate_memory = cfg.cross_debate_memory
        enable_cross_debate_memory = cfg.enable_cross_debate_memory
        enable_agent_hierarchy = cfg.enable_agent_hierarchy
        hierarchy_config = cfg.hierarchy_config
        use_rlm_limiter = cfg.use_rlm_limiter
        rlm_limiter = cfg.rlm_limiter
        rlm_compression_threshold = cfg.rlm_compression_threshold
        rlm_max_recent_messages = cfg.rlm_max_recent_messages
        rlm_summary_level = cfg.rlm_summary_level
        rlm_compression_round_threshold = cfg.rlm_compression_round_threshold
        skill_registry = cfg.skill_registry
        enable_skills = cfg.enable_skills
        propulsion_engine = cfg.propulsion_engine
        enable_propulsion = cfg.enable_propulsion
        post_debate_workflow = cfg.post_debate_workflow
        enable_post_debate_workflow = cfg.enable_post_debate_workflow
        post_debate_workflow_threshold = cfg.post_debate_workflow_threshold

        # Handle fabric integration - get agents from fabric pool if configured
        if fabric is not None and fabric_config is not None:
            if agents:
                raise ValueError(
                    "Cannot specify both 'agents' and 'fabric'/'fabric_config'. "
                    "Use either direct agents or fabric-managed agents."
                )
            agents = self._get_fabric_agents_sync(fabric, fabric_config)
            self._fabric = fabric
            self._fabric_config = fabric_config
            logger.info(
                f"[fabric] Arena using fabric pool {fabric_config.pool_id} "
                f"with {len(agents)} agents"
            )
        else:
            self._fabric = None
            self._fabric_config = None

        if not agents:
            raise ValueError("Must specify either 'agents' or both 'fabric' and 'fabric_config'")

        # Create initializer with broadcast callback
        initializer = ArenaInitializer(broadcast_callback=self._broadcast_health_event)

        # Initialize core configuration via ArenaInitializer
        core = initializer.init_core(
            environment=environment,
            agents=agents,
            protocol=protocol,
            memory=memory,
            event_hooks=event_hooks,
            hook_manager=hook_manager,
            event_emitter=event_emitter,
            spectator=spectator,
            debate_embeddings=debate_embeddings,
            insight_store=insight_store,
            recorder=recorder,
            agent_weights=agent_weights,
            loop_id=loop_id,
            strict_loop_scoping=strict_loop_scoping,
            circuit_breaker=circuit_breaker,
            initial_messages=initial_messages,
            trending_topic=trending_topic,
            pulse_manager=pulse_manager,
            auto_fetch_trending=auto_fetch_trending,
            population_manager=population_manager,
            auto_evolve=auto_evolve,
            breeding_threshold=breeding_threshold,
            evidence_collector=evidence_collector,
            breakpoint_manager=breakpoint_manager,
            checkpoint_manager=checkpoint_manager,
            enable_checkpointing=enable_checkpointing,
            performance_monitor=performance_monitor,
            enable_performance_monitor=enable_performance_monitor,
            enable_telemetry=enable_telemetry,
            use_airlock=use_airlock,
            airlock_config=airlock_config,
            agent_selector=agent_selector,
            use_performance_selection=use_performance_selection,
            prompt_evolver=prompt_evolver,
            enable_prompt_evolution=enable_prompt_evolution,
            org_id=org_id,
            user_id=user_id,
            usage_tracker=usage_tracker,
            broadcast_pipeline=broadcast_pipeline,
            auto_broadcast=auto_broadcast,
            broadcast_min_confidence=broadcast_min_confidence,
            training_exporter=training_exporter,
            auto_export_training=auto_export_training,
            training_export_min_confidence=training_export_min_confidence,
            enable_ml_delegation=enable_ml_delegation,
            ml_delegation_strategy=ml_delegation_strategy,
            ml_delegation_weight=ml_delegation_weight,
            enable_quality_gates=enable_quality_gates,
            quality_gate_threshold=quality_gate_threshold,
            enable_consensus_estimation=enable_consensus_estimation,
            consensus_early_termination_threshold=consensus_early_termination_threshold,
        )

        # Unpack core components to instance attributes
        self._apply_core_components(core)

        # Channel integration (initialized per debate run)
        self._channel_integration = None

        # Skills system for extensible capabilities (evidence collection, etc.)
        self.skill_registry = skill_registry
        self.enable_skills = enable_skills
        if skill_registry and enable_skills:
            logger.info(
                f"[skills] Skill registry attached with {skill_registry.count()} skills "
                f"(debate evidence collection enabled)"
            )

        # Propulsion engine for push-based work assignment (Gastown pattern)
        self.propulsion_engine = propulsion_engine
        self.enable_propulsion = enable_propulsion
        if propulsion_engine and enable_propulsion:
            logger.info("[propulsion] PropulsionEngine attached (reactive debate flow enabled)")

        # Auto-create Knowledge Mound if not provided (recommended for decision engine)
        knowledge_mound = _mem_auto_create_knowledge_mound(
            knowledge_mound=knowledge_mound,
            auto_create=auto_create_knowledge_mound,
            enable_retrieval=enable_knowledge_retrieval,
            enable_ingestion=enable_knowledge_ingestion,
            org_id=org_id,
        )

        # Initialize tracking subsystems via ArenaInitializer
        trackers = initializer.init_trackers(
            protocol=self.protocol,
            loop_id=self.loop_id,
            agent_pool=self.agent_pool,
            position_tracker=position_tracker,
            position_ledger=position_ledger,
            enable_position_ledger=enable_position_ledger,
            elo_system=elo_system,
            persona_manager=persona_manager,
            dissent_retriever=dissent_retriever,
            consensus_memory=consensus_memory,
            flip_detector=flip_detector,
            calibration_tracker=calibration_tracker,
            continuum_memory=continuum_memory,
            relationship_tracker=relationship_tracker,
            moment_detector=moment_detector,
            tier_analytics_tracker=tier_analytics_tracker,
            knowledge_mound=knowledge_mound,
            enable_knowledge_retrieval=enable_knowledge_retrieval,
            enable_knowledge_ingestion=enable_knowledge_ingestion,
            enable_knowledge_extraction=enable_knowledge_extraction,
            extraction_min_confidence=extraction_min_confidence,
            enable_belief_guidance=enable_belief_guidance,
            vertical=vertical,
            vertical_persona_manager=vertical_persona_manager,
            auto_detect_vertical=auto_detect_vertical,
            task=environment.task,
        )

        # Unpack tracker components to instance attributes
        self._apply_tracker_components(trackers)

        # Store additional config flags not tracked via TrackerComponents
        self.enable_auto_revalidation = enable_auto_revalidation
        self.revalidation_staleness_threshold = revalidation_staleness_threshold
        self.revalidation_check_interval_seconds = revalidation_check_interval_seconds
        self.revalidation_scheduler = revalidation_scheduler

        # Adaptive rounds (memory-based debate strategy)
        self.enable_adaptive_rounds = enable_adaptive_rounds
        self.debate_strategy = debate_strategy
        if self.enable_adaptive_rounds and self.debate_strategy is None:
            try:
                from aragora.debate.strategy import DebateStrategy

                self.debate_strategy = DebateStrategy(
                    continuum_memory=self.continuum_memory,
                )
                logger.info("debate_strategy auto-initialized for adaptive rounds")
            except ImportError:
                logger.debug("DebateStrategy not available")
                self.debate_strategy = None
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to initialize DebateStrategy: {e}")
                self.debate_strategy = None
            except Exception as e:
                logger.exception(f"Unexpected error initializing DebateStrategy: {e}")
                self.debate_strategy = None

        # Cross-debate institutional memory
        self.cross_debate_memory = cross_debate_memory
        self.enable_cross_debate_memory = enable_cross_debate_memory

        # Post-debate workflow automation
        if enable_post_debate_workflow and post_debate_workflow is None:
            try:
                from aragora.workflow.patterns.post_debate import get_default_post_debate_workflow

                post_debate_workflow = get_default_post_debate_workflow()
                logger.debug("[arena] Auto-created default post-debate workflow")
            except ImportError:
                logger.warning("[arena] Post-debate workflow enabled but pattern not available")
        self.post_debate_workflow = post_debate_workflow
        self.enable_post_debate_workflow = enable_post_debate_workflow
        self.post_debate_workflow_threshold = post_debate_workflow_threshold

        # Initialize user participation and roles
        self._init_user_participation()
        self._init_event_bus()
        self._init_roles_and_stances()

        # Initialize convergence detection and caches
        self._init_convergence()
        self._init_caches()

        # Initialize extracted helper classes for lifecycle, events, and checkpoints
        self._init_lifecycle_manager()
        self._init_event_emitter()
        self._init_checkpoint_ops()
        self._init_checkpoint_bridge()

        # Initialize grounded operations helper (uses position_ledger, elo_system)
        self._init_grounded_operations()

        # Initialize agent hierarchy (Gastown pattern)
        self._init_agent_hierarchy(enable_agent_hierarchy, hierarchy_config)

        # Initialize knowledge mound operations
        self._init_knowledge_ops()

        # Initialize RLM cognitive load limiter for context compression
        self._init_rlm_limiter(
            use_rlm_limiter=use_rlm_limiter,
            rlm_limiter=rlm_limiter,
            rlm_compression_threshold=rlm_compression_threshold,
            rlm_max_recent_messages=rlm_max_recent_messages,
            rlm_summary_level=rlm_summary_level,
        )
        self.rlm_compression_round_threshold = rlm_compression_round_threshold

        # Initialize phase classes for orchestrator decomposition
        self._init_phases()

        # Initialize prompt context builder (uses persona_manager, flip_detector, etc.)
        self._init_prompt_context_builder()

        # Initialize context delegator (after phases since it needs evidence_grounder)
        self._init_context_delegator()

        # Initialize termination checker
        self._init_termination_checker()

        # Initialize cross-subscriber bridge for event cross-pollination
        self._init_cross_subscriber_bridge()

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_config(
        cls,
        environment: Environment,
        agents: list[Agent],
        protocol: DebateProtocol = None,
        config: ArenaConfig = None,
    ) -> "Arena":
        """Create an Arena from an ArenaConfig for cleaner dependency injection."""
        from aragora.debate.feature_validator import validate_and_warn

        config = config or ArenaConfig()
        validate_and_warn(config)
        return cls(
            environment=environment,
            agents=agents,
            protocol=protocol,
            **config.to_arena_kwargs(),
        )

    @classmethod
    def from_configs(
        cls,
        environment: Environment,
        agents: list[Agent],
        protocol: Optional[DebateProtocol] = None,
        *,
        debate_config: Optional["DebateConfig"] = None,
        agent_config: Optional["AgentConfig"] = None,
        memory_config: Optional["MemoryConfig"] = None,
        streaming_config: Optional["StreamingConfig"] = None,
        observability_config: Optional["ObservabilityConfig"] = None,
    ) -> "Arena":
        """Create an Arena from grouped config objects.

        This is the preferred factory method for creating Arena instances
        with the new configuration pattern.
        """
        return cls(
            environment=environment,
            agents=agents,
            protocol=protocol,
            debate_config=debate_config,
            agent_config=agent_config,
            memory_config=memory_config,
            streaming_config=streaming_config,
            observability_config=observability_config,
        )

    # =========================================================================
    # Core Component Setup
    # =========================================================================

    def _apply_core_components(self, core) -> None:
        """Unpack CoreComponents dataclass to instance attributes."""
        self.env = core.env
        self.agents = core.agents
        self.protocol = core.protocol
        self.memory = core.memory
        self.hooks = core.hooks
        self.hook_manager = core.hook_manager
        self.event_emitter = core.event_emitter
        self.spectator = core.spectator
        self.debate_embeddings = core.debate_embeddings
        self.insight_store = core.insight_store
        self.recorder = core.recorder
        self.agent_weights = core.agent_weights
        self.loop_id = core.loop_id
        self.strict_loop_scoping = core.strict_loop_scoping
        self.circuit_breaker = core.circuit_breaker
        self.agent_pool = core.agent_pool
        self.immune_system = core.immune_system
        self.chaos_director = core.chaos_director
        self.performance_monitor = core.performance_monitor
        self.prompt_evolver = core.prompt_evolver
        self.autonomic = core.autonomic
        self.initial_messages = core.initial_messages
        self.trending_topic = core.trending_topic
        self.pulse_manager = core.pulse_manager
        self.auto_fetch_trending = core.auto_fetch_trending
        self.population_manager = core.population_manager
        self.auto_evolve = core.auto_evolve
        self.breeding_threshold = core.breeding_threshold
        self.evidence_collector = core.evidence_collector
        self.breakpoint_manager = core.breakpoint_manager
        self.agent_selector = core.agent_selector
        self.use_performance_selection = core.use_performance_selection
        self.checkpoint_manager = core.checkpoint_manager
        self.org_id = core.org_id
        self.user_id = core.user_id
        self._budget_coordinator = BudgetCoordinator(
            org_id=self.org_id,
            user_id=self.user_id,
        )
        self.extensions = core.extensions
        self.cartographer = core.cartographer
        self.event_bridge = core.event_bridge
        # ML Integration
        self.enable_ml_delegation = core.enable_ml_delegation
        self.ml_delegation_weight = core.ml_delegation_weight
        self.enable_quality_gates = core.enable_quality_gates
        self.quality_gate_threshold = core.quality_gate_threshold
        self.enable_consensus_estimation = core.enable_consensus_estimation
        self.consensus_early_termination_threshold = core.consensus_early_termination_threshold
        self._ml_delegation_strategy = core.ml_delegation_strategy
        self._ml_quality_gate = core.ml_quality_gate
        self._ml_consensus_estimator = core.ml_consensus_estimator
        # Event bus initialized later in _init_event_bus() after audience_manager exists
        self.event_bus: EventBus | None = None

    def _apply_tracker_components(self, trackers) -> None:
        """Unpack TrackerComponents dataclass to instance attributes."""
        self.position_tracker = trackers.position_tracker
        self.position_ledger = trackers.position_ledger
        self.elo_system = trackers.elo_system
        self.persona_manager = trackers.persona_manager
        self.dissent_retriever = trackers.dissent_retriever
        self.consensus_memory = trackers.consensus_memory
        self.flip_detector = trackers.flip_detector
        self.calibration_tracker = trackers.calibration_tracker
        self.continuum_memory = trackers.continuum_memory
        self.relationship_tracker = trackers.relationship_tracker
        self.moment_detector = trackers.moment_detector
        self.tier_analytics_tracker = trackers.tier_analytics_tracker
        self.knowledge_mound = trackers.knowledge_mound
        self.enable_knowledge_retrieval = trackers.enable_knowledge_retrieval
        self.enable_knowledge_ingestion = trackers.enable_knowledge_ingestion
        self.enable_knowledge_extraction = trackers.enable_knowledge_extraction
        self.extraction_min_confidence = trackers.extraction_min_confidence
        self.enable_belief_guidance = trackers.enable_belief_guidance
        self._trackers = trackers.coordinator
        self.vertical = trackers.vertical
        self.vertical_persona_manager = trackers.vertical_persona_manager

    def _broadcast_health_event(self, event: dict[str, Any]) -> None:
        """Broadcast health events. Delegates to EventEmitter."""
        self._event_emitter.broadcast_health_event(event)

    def _get_fabric_agents_sync(self, fabric: Any, fabric_config: Any) -> list[Agent]:
        """Get agents from fabric pool. Delegates to orchestrator_agents."""
        return _agents_get_fabric_agents_sync(fabric, fabric_config)

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_user_participation(self) -> None:
        """Initialize user participation tracking and event subscription."""
        _participation_init_user_participation(self)

    def _init_event_bus(self) -> None:
        """Initialize EventBus for pub/sub event handling."""
        _participation_init_event_bus(self)

    @property
    def user_votes(self) -> deque[dict[str, Any]]:
        """Get user votes from AudienceManager (backward compatibility)."""
        return self.audience_manager._votes

    @property
    def user_suggestions(self) -> deque[dict[str, Any]]:
        """Get user suggestions from AudienceManager (backward compatibility)."""
        return self.audience_manager._suggestions

    def _init_roles_and_stances(self) -> None:
        """Initialize cognitive role rotation and agent stances."""
        _roles_init_roles_and_stances(self)

    def _init_convergence(self, debate_id: str | None = None) -> None:
        """Initialize convergence detection if enabled."""
        _conv_init_convergence(self, debate_id)

    def _reinit_convergence_for_debate(self, debate_id: str) -> None:
        """Reinitialize convergence detector with debate-specific cache."""
        _conv_reinit_convergence_for_debate(self, debate_id)

    def _cleanup_convergence_cache(self) -> None:
        """Cleanup embedding cache for the current debate."""
        _conv_cleanup_convergence(self)

    def _init_caches(self) -> None:
        """Initialize caches for computed values."""
        _lifecycle_init_caches(self)

    def _init_lifecycle_manager(self) -> None:
        """Initialize LifecycleManager for cleanup and task cancellation."""
        _lifecycle_init_lifecycle_manager(self)

    def _init_event_emitter(self) -> None:
        """Initialize EventEmitter for spectator/websocket events."""
        _lifecycle_init_event_emitter(self)

    def _init_checkpoint_ops(self) -> None:
        """Initialize CheckpointOperations for checkpoint and memory operations."""
        _lifecycle_init_checkpoint_ops(self)

    def _init_checkpoint_bridge(self) -> None:
        """Initialize optional checkpoint bridge. Delegates to orchestrator_memory."""
        self.molecule_orchestrator, self.checkpoint_bridge = _mem_init_checkpoint_bridge(
            self.protocol, self.checkpoint_manager
        )

    def _init_grounded_operations(self) -> None:
        """Initialize GroundedOperations helper for verdict and relationship management."""
        self._grounded_ops = GroundedOperations(
            position_ledger=self.position_ledger,
            elo_system=self.elo_system,
            evidence_grounder=None,  # Set after _init_phases
        )

    def _init_agent_hierarchy(
        self,
        enable_agent_hierarchy: bool,
        hierarchy_config: HierarchyConfig | None,
    ) -> None:
        """Initialize AgentHierarchy. Delegates to orchestrator_agents."""
        self.enable_agent_hierarchy = enable_agent_hierarchy
        self._hierarchy: AgentHierarchy | None = _agents_init_agent_hierarchy(
            enable_agent_hierarchy, hierarchy_config
        )

    def _assign_hierarchy_roles(
        self,
        ctx: "DebateContext",
        task_type: str | None = None,
    ) -> None:
        """Assign hierarchy roles to agents. Delegates to orchestrator_agents."""
        _agents_assign_hierarchy_roles(ctx, self.enable_agent_hierarchy, self._hierarchy, task_type)

    def _init_knowledge_ops(self) -> None:
        """Initialize ArenaKnowledgeManager for knowledge retrieval and ingestion."""
        self._km_manager = ArenaKnowledgeManager(
            knowledge_mound=self.knowledge_mound,
            enable_retrieval=self.enable_knowledge_retrieval,
            enable_ingestion=self.enable_knowledge_ingestion,
            enable_auto_revalidation=self.enable_auto_revalidation,
            revalidation_staleness_threshold=getattr(self, "revalidation_staleness_threshold", 0.7),
            revalidation_check_interval_seconds=getattr(
                self, "revalidation_check_interval_seconds", 3600
            ),
            notify_callback=self._knowledge_notify_callback,
        )
        self._km_manager.initialize(
            continuum_memory=self.continuum_memory,
            consensus_memory=self.consensus_memory,
            elo_system=self.elo_system,
            cost_tracker=getattr(self, "cost_tracker", None),
            insight_store=self.insight_store,
            flip_detector=self.flip_detector,
            evidence_store=getattr(self, "evidence_store", None),
            pulse_manager=getattr(self, "pulse_manager", None),
            memory=self.memory,
        )
        self._knowledge_ops = self._km_manager._knowledge_ops
        self.knowledge_bridge_hub = self._km_manager.knowledge_bridge_hub
        if self._km_manager.revalidation_scheduler is not None:
            self.revalidation_scheduler = self._km_manager.revalidation_scheduler
        self._km_coordinator = self._km_manager._km_coordinator
        self._km_adapters = self._km_manager._km_adapters

    def _knowledge_notify_callback(self, event_type: str, data: dict[str, Any]) -> None:
        """Callback for knowledge mound notifications."""
        self._notify_spectator(event_type, **data)

    def _init_prompt_context_builder(self) -> None:
        """Initialize PromptContextBuilder for agent prompt context."""
        self._prompt_context = PromptContextBuilder(
            persona_manager=self.persona_manager,
            flip_detector=self.flip_detector,
            protocol=self.protocol,
            prompt_builder=self.prompt_builder,
            audience_manager=self.audience_manager,
            spectator=self.spectator,
            notify_callback=self._notify_spectator,
            vertical=getattr(self, "vertical", None),
            vertical_persona_manager=getattr(self, "vertical_persona_manager", None),
        )

    def _init_context_delegator(self) -> None:
        """Initialize ContextDelegator for context gathering operations."""
        self._context_delegator = ContextDelegator(
            context_gatherer=self.context_gatherer,
            memory_manager=self.memory_manager,
            cache=self._cache,
            evidence_grounder=getattr(self, "evidence_grounder", None),
            continuum_memory=self.continuum_memory,
            env=self.env,
            extract_domain_fn=self._extract_debate_domain,
        )

    def _init_phases(self) -> None:
        """Initialize phase classes for orchestrator decomposition."""
        init_phases(self)
        self.phase_executor = create_phase_executor(self)
        if hasattr(self, "_grounded_ops") and self._grounded_ops:
            self._grounded_ops.evidence_grounder = self.evidence_grounder
        if hasattr(self, "_checkpoint_ops") and self._checkpoint_ops:
            self._checkpoint_ops.memory_manager = self.memory_manager

    def _init_termination_checker(self) -> None:
        """Initialize the termination checker for early debate termination."""

        async def generate_fn(agent: Agent, prompt: str, ctx: list[Message]) -> str:
            return await self.autonomic.generate(agent, prompt, ctx)

        async def select_judge_fn(proposals: dict[str, str], context: list[Message]) -> Agent:
            return await self._select_judge(proposals, context)

        self.termination_checker = TerminationChecker(
            protocol=self.protocol,
            agents=self._require_agents() if self.agents else [],
            generate_fn=generate_fn,
            task=self.env.task if self.env else "",
            select_judge_fn=select_judge_fn,
            hooks=self.hooks,
        )

    def _init_cross_subscriber_bridge(self) -> None:
        """Initialize cross-subscriber bridge. Delegates to orchestrator_memory."""
        self._cross_subscriber_bridge = _mem_init_cross_subscriber_bridge(self.event_bus)

    # =========================================================================
    # Core Instance Helpers
    # =========================================================================

    def _require_agents(self) -> list[Agent]:
        """Return agents list, raising error if empty."""
        if not self.agents:
            raise ValueError("No agents available - Arena requires at least one agent")
        return self.agents

    def _sync_prompt_builder_state(self) -> None:
        """Sync Arena state to PromptBuilder before building prompts."""
        self.prompt_builder.current_role_assignments = self.current_role_assignments
        self.prompt_builder._historical_context_cache = self._cache.historical_context
        self.prompt_builder._continuum_context_cache = self._get_continuum_context()
        self.prompt_builder.user_suggestions = list(self.user_suggestions)

    def _get_continuum_context(self) -> str:
        """Retrieve relevant memories from ContinuumMemory for debate context."""
        return self._context_delegator.get_continuum_context()

    def _extract_debate_domain(self) -> str:
        """Extract domain from the debate task. Cached at instance and module level."""
        if self._cache.has_debate_domain():
            if self._cache.debate_domain is None:
                raise RuntimeError("Cached debate domain is None - cache may be corrupted")
            return self._cache.debate_domain
        domain = _compute_domain_from_task(self.env.task.lower())
        self._cache.debate_domain = domain
        return domain

    def _select_debate_team(self, requested_agents: list[Agent]) -> list[Agent]:
        """Select debate team. Delegates to orchestrator_agents."""
        return _agents_select_debate_team(
            agents=requested_agents,
            env=self.env,
            extract_domain_fn=self._extract_debate_domain,
            enable_ml_delegation=self.enable_ml_delegation,
            ml_delegation_strategy=self._ml_delegation_strategy,
            protocol=self.protocol,
            use_performance_selection=self.use_performance_selection,
            agent_pool=self.agent_pool,
        )

    def _filter_responses_by_quality(
        self, responses: list[tuple[str, str]], context: str = ""
    ) -> list[tuple[str, str]]:
        """Filter responses using ML quality gate. Delegates to orchestrator_agents."""
        return _agents_filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=self.enable_quality_gates,
            ml_quality_gate=self._ml_quality_gate,
            task=self.env.task,
            context=context,
        )

    def _should_terminate_early(self, responses: list[tuple[str, str]], current_round: int) -> bool:
        """Check if debate should terminate early. Delegates to orchestrator_agents."""
        return _agents_should_terminate_early(
            responses=responses,
            current_round=current_round,
            enable_consensus_estimation=self.enable_consensus_estimation,
            ml_consensus_estimator=self._ml_consensus_estimator,
            protocol=self.protocol,
            task=self.env.task,
        )

    def _init_rlm_limiter(
        self,
        use_rlm_limiter: bool,
        rlm_limiter: Optional["RLMCognitiveLoadLimiter"],
        rlm_compression_threshold: int,
        rlm_max_recent_messages: int,
        rlm_summary_level: str,
    ) -> None:
        """Initialize the RLM cognitive load limiter. Delegates to orchestrator_memory."""
        state = _mem_init_rlm_limiter_state(
            use_rlm_limiter=use_rlm_limiter,
            rlm_limiter=rlm_limiter,
            rlm_compression_threshold=rlm_compression_threshold,
            rlm_max_recent_messages=rlm_max_recent_messages,
            rlm_summary_level=rlm_summary_level,
        )
        self.use_rlm_limiter = state["use_rlm_limiter"]
        self.rlm_compression_threshold = state["rlm_compression_threshold"]
        self.rlm_max_recent_messages = state["rlm_max_recent_messages"]
        self.rlm_summary_level = state["rlm_summary_level"]
        self.rlm_limiter = state["rlm_limiter"]

    async def _select_judge(self, proposals: dict[str, str], context: list[Message]) -> Agent:
        """Select judge based on protocol.judge_selection setting. Delegates to JudgeSelector."""

        async def generate_wrapper(agent: Agent, prompt: str, ctx: list[Message]) -> str:
            return await agent.generate(prompt, ctx)

        selector = JudgeSelector(
            agents=self._require_agents(),
            elo_system=self.elo_system,
            judge_selection=self.protocol.judge_selection,
            generate_fn=generate_wrapper,
            build_vote_prompt_fn=lambda candidates,
            props: self.prompt_builder.build_judge_vote_prompt(candidates, props),
            sanitize_fn=OutputSanitizer.sanitize_agent_output,
            consensus_memory=self.consensus_memory,
        )
        return await selector.select_judge(proposals, context)

    # =========================================================================
    # Public Checkpoint API
    # =========================================================================

    async def save_checkpoint(
        self,
        debate_id: str,
        phase: str = "manual",
        messages: Optional[list[Message]] = None,
        critiques: Optional[list[Critique]] = None,
        votes: Optional[list[Vote]] = None,
        current_round: int = 0,
        current_consensus: Optional[str] = None,
    ) -> Optional[str]:
        """Save a checkpoint for the current debate state."""
        return await _cp_save_checkpoint(
            checkpoint_manager=self.checkpoint_manager,
            debate_id=debate_id,
            env=self.env,
            protocol=self.protocol,
            agents=self.agents,
            phase=phase,
            messages=messages,
            critiques=critiques,
            votes=votes,
            current_round=current_round,
            current_consensus=current_consensus,
        )

    async def restore_from_checkpoint(
        self,
        checkpoint_id: str,
        resumed_by: str = "system",
    ) -> Optional["DebateContext"]:
        """Restore debate state from a checkpoint."""
        return await _cp_restore_from_checkpoint(
            checkpoint_manager=self.checkpoint_manager,
            checkpoint_id=checkpoint_id,
            env=self.env,
            agents=self.agents,
            domain=self._extract_debate_domain() if hasattr(self, "_extract_debate_domain") else "",
            hook_manager=self.hook_manager if hasattr(self, "hook_manager") else None,
            org_id=self.org_id if hasattr(self, "org_id") else "",
            resumed_by=resumed_by,
        )

    async def list_checkpoints(
        self,
        debate_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List available checkpoints."""
        return await _cp_list_checkpoints(
            checkpoint_manager=self.checkpoint_manager,
            debate_id=debate_id,
            limit=limit,
        )

    async def cleanup_checkpoints(
        self,
        debate_id: str,
        keep_latest: int = 1,
    ) -> int:
        """Clean up old checkpoints for a completed debate."""
        return await _cp_cleanup_checkpoints(
            checkpoint_manager=self.checkpoint_manager,
            debate_id=debate_id,
            keep_latest=keep_latest,
        )

    # =========================================================================
    # Async Context Manager Protocol
    # =========================================================================

    async def __aenter__(self) -> "Arena":
        """Enter async context - prepare for debate."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context - cleanup resources."""
        await self._cleanup()

    def _track_circuit_breaker_metrics(self) -> None:
        """Track circuit breaker state in metrics. Delegates to LifecycleManager."""
        self._lifecycle.track_circuit_breaker_metrics()

    def _log_phase_failures(self, execution_result: Any) -> None:
        """Log any failed phases. Delegates to LifecycleManager."""
        self._lifecycle.log_phase_failures(execution_result)

    async def _cleanup(self) -> None:
        """Internal cleanup. Delegates to LifecycleManager."""
        await self._lifecycle.cleanup()
        await self._teardown_agent_channels()
        if hasattr(self, "context_gatherer") and self.context_gatherer:
            self.context_gatherer.clear_cache()
        self._cleanup_convergence_cache()

    async def _setup_agent_channels(self, ctx: "DebateContext", debate_id: str) -> None:
        """Initialize agent-to-agent channels for the current debate."""
        if not getattr(self.protocol, "enable_agent_channels", False):
            return
        try:
            from aragora.debate.channel_integration import create_channel_integration

            self._channel_integration = create_channel_integration(
                debate_id=debate_id,
                agents=self.agents,
                protocol=self.protocol,
            )
            if await self._channel_integration.setup():
                ctx.channel_integration = self._channel_integration
            else:
                self._channel_integration = None
        except (ImportError, ConnectionError, OSError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"[channels] Channel setup failed (non-critical): {e}")
            self._channel_integration = None

    async def _teardown_agent_channels(self) -> None:
        """Tear down agent channels after debate completion."""
        if not self._channel_integration:
            return
        try:
            await self._channel_integration.teardown()
        except (ConnectionError, OSError, RuntimeError) as e:
            logger.debug(f"[channels] Channel teardown failed (non-critical): {e}")
        finally:
            self._channel_integration = None

    # =========================================================================
    # Debate Execution
    # =========================================================================

    async def run(self, correlation_id: str = "") -> DebateResult:
        """Run the full debate and return results."""
        if self.protocol.timeout_seconds > 0:
            try:
                return await asyncio.wait_for(
                    self._run_inner(correlation_id=correlation_id),
                    timeout=self.protocol.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"debate_timeout timeout_seconds={self.protocol.timeout_seconds}")
                return DebateResult(
                    task=self.env.task,
                    messages=getattr(self, "_partial_messages", []),
                    critiques=getattr(self, "_partial_critiques", []),
                    votes=[],
                    dissenting_views=[],
                    rounds_used=getattr(self, "_partial_rounds", 0),
                )
        return await self._run_inner(correlation_id=correlation_id)

    async def _run_inner(self, correlation_id: str = "") -> DebateResult:
        """Internal debate execution orchestrator coordinating all phases.

        This method orchestrates the full debate lifecycle:
        1. Initialize debate context (IDs, domain, agents, channels)
        2. Set up infrastructure (logging, trackers, budget, hooks)
        3. Execute debate phases with tracing and error handling
        4. Handle completion (extensions, knowledge ingestion, beads)
        5. Clean up resources (checkpoints, caches, channels)
        """
        # Phase 1: Initialize debate context and execution state
        state = await _runner_initialize_debate_context(self, correlation_id)

        # Phase 2: Set up debate infrastructure
        await _runner_setup_debate_infrastructure(self, state)

        # Track active debates metric
        ACTIVE_DEBATES.inc()

        # Initialize tracing and monitoring
        tracer = get_tracer()
        perf_monitor = get_debate_monitor()
        agent_names = [a.name for a in self.agents]

        # Phase 3: Execute debate phases with tracing context
        with (
            tracer.start_as_current_span("debate") as span,
            perf_monitor.track_debate(state.debate_id, task=self.env.task, agent_names=agent_names),
            n1_detection_scope(f"debate_{state.debate_id}"),
            debate_loader_context(elo_system=self.elo_system) as loaders,
        ):
            # Make loaders available in context for phase handlers
            state.ctx.data_loaders = loaders

            # Add debate attributes to span
            add_span_attributes(
                span,
                {
                    "debate.id": state.debate_id,
                    "debate.correlation_id": state.correlation_id,
                    "debate.domain": state.domain,
                    "debate.complexity": state.task_complexity.value,
                    "debate.agent_count": len(self.agents),
                    "debate.agents": ",".join(a.name for a in self.agents),
                    "debate.task_length": len(self.env.task),
                },
            )

            try:
                await _runner_execute_debate_phases(self, state, span)
            finally:
                _runner_record_debate_metrics(self, state, span)

        # Phase 4: Handle debate completion (trackers, extensions, knowledge)
        await _runner_handle_debate_completion(self, state)

        # Phase 5: Clean up resources and finalize result
        return await _runner_cleanup_debate_resources(self, state)

    # NOTE: Legacy _run_inner code (1,300+ lines) removed after successful phase integration.
    # The debate execution is now handled by phase classes:
    # - ContextInitializer (Phase 0)
    # - ProposalPhase (Phase 1)
    # - DebateRoundsPhase (Phase 2)
    # - ConsensusPhase (Phase 3)
    # - AnalyticsPhase (Phases 4-6)
    # - FeedbackPhase (Phase 7)
    #
    # NOTE: Token usage recording and training export are now handled by
    # ArenaExtensions.on_debate_complete() - see debate/extensions.py
