"""Config merging logic for Arena __init__.

Extracts the config object merging (DebateConfig, AgentConfig, MemoryConfig,
StreamingConfig, ObservabilityConfig) from Arena.__init__ into a standalone
function for readability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.debate.arena_config import (
        AgentConfig,
        DebateConfig,
        MemoryConfig,
        ObservabilityConfig,
        StreamingConfig,
    )
    from aragora.debate.protocol import DebateProtocol


class MergedConfig:
    """Container for all merged configuration values after applying config objects."""

    __slots__ = (
        "enable_adaptive_rounds",
        "debate_strategy",
        "enable_agent_hierarchy",
        "hierarchy_config",
        "agent_weights",
        "agent_selector",
        "use_performance_selection",
        "circuit_breaker",
        "use_airlock",
        "airlock_config",
        "position_tracker",
        "position_ledger",
        "enable_position_ledger",
        "elo_system",
        "calibration_tracker",
        "relationship_tracker",
        "persona_manager",
        "vertical",
        "vertical_persona_manager",
        "auto_detect_vertical",
        "fabric",
        "fabric_config",
        "memory",
        "continuum_memory",
        "consensus_memory",
        "debate_embeddings",
        "insight_store",
        "dissent_retriever",
        "flip_detector",
        "moment_detector",
        "tier_analytics_tracker",
        "cross_debate_memory",
        "enable_cross_debate_memory",
        "knowledge_mound",
        "auto_create_knowledge_mound",
        "enable_knowledge_retrieval",
        "enable_knowledge_ingestion",
        "enable_knowledge_extraction",
        "extraction_min_confidence",
        "enable_belief_guidance",
        "enable_auto_revalidation",
        "revalidation_staleness_threshold",
        "revalidation_check_interval_seconds",
        "revalidation_scheduler",
        "use_rlm_limiter",
        "rlm_limiter",
        "rlm_compression_threshold",
        "rlm_max_recent_messages",
        "rlm_summary_level",
        "rlm_compression_round_threshold",
        "checkpoint_manager",
        "enable_checkpointing",
        "event_hooks",
        "hook_manager",
        "event_emitter",
        "spectator",
        "recorder",
        "loop_id",
        "strict_loop_scoping",
        "skill_registry",
        "enable_skills",
        "propulsion_engine",
        "enable_propulsion",
        "performance_monitor",
        "enable_performance_monitor",
        "enable_telemetry",
        "prompt_evolver",
        "enable_prompt_evolution",
        "breakpoint_manager",
        "trending_topic",
        "pulse_manager",
        "auto_fetch_trending",
        "population_manager",
        "auto_evolve",
        "breeding_threshold",
        "evidence_collector",
        "org_id",
        "user_id",
        "usage_tracker",
        "broadcast_pipeline",
        "auto_broadcast",
        "broadcast_min_confidence",
        "training_exporter",
        "auto_export_training",
        "training_export_min_confidence",
        "enable_ml_delegation",
        "ml_delegation_strategy",
        "ml_delegation_weight",
        "enable_quality_gates",
        "quality_gate_threshold",
        "enable_consensus_estimation",
        "consensus_early_termination_threshold",
        "post_debate_workflow",
        "enable_post_debate_workflow",
        "post_debate_workflow_threshold",
        "initial_messages",
        "protocol",
    )

    # Type annotations for all slots (required by mypy for __slots__ classes)
    protocol: Optional["DebateProtocol"]
    enable_adaptive_rounds: bool
    debate_strategy: Any
    enable_agent_hierarchy: bool
    hierarchy_config: Any
    agent_weights: Any
    agent_selector: Any
    use_performance_selection: bool
    circuit_breaker: Any
    use_airlock: bool
    airlock_config: Any
    position_tracker: Any
    position_ledger: Any
    enable_position_ledger: bool
    elo_system: Any
    calibration_tracker: Any
    relationship_tracker: Any
    persona_manager: Any
    vertical: Any
    vertical_persona_manager: Any
    auto_detect_vertical: bool
    fabric: Any
    fabric_config: Any
    memory: Any
    continuum_memory: Any
    consensus_memory: Any
    debate_embeddings: Any
    insight_store: Any
    dissent_retriever: Any
    flip_detector: Any
    moment_detector: Any
    tier_analytics_tracker: Any
    cross_debate_memory: Any
    enable_cross_debate_memory: bool
    knowledge_mound: Any
    auto_create_knowledge_mound: bool
    enable_knowledge_retrieval: bool
    enable_knowledge_ingestion: bool
    enable_knowledge_extraction: bool
    extraction_min_confidence: float
    enable_belief_guidance: bool
    enable_auto_revalidation: bool
    revalidation_staleness_threshold: float
    revalidation_check_interval_seconds: int
    revalidation_scheduler: Any
    use_rlm_limiter: bool
    rlm_limiter: Any
    rlm_compression_threshold: int
    rlm_max_recent_messages: int
    rlm_summary_level: str
    rlm_compression_round_threshold: int
    checkpoint_manager: Any
    enable_checkpointing: bool
    event_hooks: Any
    hook_manager: Any
    event_emitter: Any
    spectator: Any
    recorder: Any
    loop_id: str
    strict_loop_scoping: bool
    skill_registry: Any
    enable_skills: bool
    propulsion_engine: Any
    enable_propulsion: bool
    performance_monitor: Any
    enable_performance_monitor: bool
    enable_telemetry: bool
    prompt_evolver: Any
    enable_prompt_evolution: bool
    breakpoint_manager: Any
    trending_topic: Any
    pulse_manager: Any
    auto_fetch_trending: bool
    population_manager: Any
    auto_evolve: bool
    breeding_threshold: float
    evidence_collector: Any
    org_id: str
    user_id: str
    usage_tracker: Any
    broadcast_pipeline: Any
    auto_broadcast: bool
    broadcast_min_confidence: float
    training_exporter: Any
    auto_export_training: bool
    training_export_min_confidence: float
    enable_ml_delegation: bool
    ml_delegation_strategy: Any
    ml_delegation_weight: float
    enable_quality_gates: bool
    quality_gate_threshold: float
    enable_consensus_estimation: bool
    consensus_early_termination_threshold: float
    post_debate_workflow: Any
    enable_post_debate_workflow: bool
    post_debate_workflow_threshold: float
    initial_messages: Any


def merge_config_objects(  # noqa: C901 - complexity inherent in config merging
    *,
    # Config objects
    debate_config: Optional["DebateConfig"],
    agent_config: Optional["AgentConfig"],
    memory_config: Optional["MemoryConfig"],
    streaming_config: Optional["StreamingConfig"],
    observability_config: Optional["ObservabilityConfig"],
    # Individual params (defaults from __init__ signature)
    protocol: Optional["DebateProtocol"],
    enable_adaptive_rounds: bool,
    debate_strategy: Any,
    enable_agent_hierarchy: bool,
    hierarchy_config: Any,
    agent_weights: Any,
    agent_selector: Any,
    use_performance_selection: bool,
    circuit_breaker: Any,
    use_airlock: bool,
    airlock_config: Any,
    position_tracker: Any,
    position_ledger: Any,
    enable_position_ledger: bool,
    elo_system: Any,
    calibration_tracker: Any,
    relationship_tracker: Any,
    persona_manager: Any,
    vertical: Any,
    vertical_persona_manager: Any,
    auto_detect_vertical: bool,
    fabric: Any,
    fabric_config: Any,
    memory: Any,
    continuum_memory: Any,
    consensus_memory: Any,
    debate_embeddings: Any,
    insight_store: Any,
    dissent_retriever: Any,
    flip_detector: Any,
    moment_detector: Any,
    tier_analytics_tracker: Any,
    cross_debate_memory: Any,
    enable_cross_debate_memory: bool,
    knowledge_mound: Any,
    auto_create_knowledge_mound: bool,
    enable_knowledge_retrieval: bool,
    enable_knowledge_ingestion: bool,
    enable_knowledge_extraction: bool,
    extraction_min_confidence: float,
    enable_belief_guidance: bool,
    enable_auto_revalidation: bool,
    revalidation_staleness_threshold: float,
    revalidation_check_interval_seconds: int,
    revalidation_scheduler: Any,
    use_rlm_limiter: bool,
    rlm_limiter: Any,
    rlm_compression_threshold: int,
    rlm_max_recent_messages: int,
    rlm_summary_level: str,
    rlm_compression_round_threshold: int,
    checkpoint_manager: Any,
    enable_checkpointing: bool,
    event_hooks: Any,
    hook_manager: Any,
    event_emitter: Any,
    spectator: Any,
    recorder: Any,
    loop_id: str,
    strict_loop_scoping: bool,
    skill_registry: Any,
    enable_skills: bool,
    propulsion_engine: Any,
    enable_propulsion: bool,
    performance_monitor: Any,
    enable_performance_monitor: bool,
    enable_telemetry: bool,
    prompt_evolver: Any,
    enable_prompt_evolution: bool,
    breakpoint_manager: Any,
    trending_topic: Any,
    pulse_manager: Any,
    auto_fetch_trending: bool,
    population_manager: Any,
    auto_evolve: bool,
    breeding_threshold: float,
    evidence_collector: Any,
    org_id: str,
    user_id: str,
    usage_tracker: Any,
    broadcast_pipeline: Any,
    auto_broadcast: bool,
    broadcast_min_confidence: float,
    training_exporter: Any,
    auto_export_training: bool,
    training_export_min_confidence: float,
    enable_ml_delegation: bool,
    ml_delegation_strategy: Any,
    ml_delegation_weight: float,
    enable_quality_gates: bool,
    quality_gate_threshold: float,
    enable_consensus_estimation: bool,
    consensus_early_termination_threshold: float,
    post_debate_workflow: Any,
    enable_post_debate_workflow: bool,
    post_debate_workflow_threshold: float,
    initial_messages: Any,
) -> MergedConfig:
    """Merge config objects with individual parameters.

    Config objects take precedence over individual parameters when provided.
    Returns a MergedConfig with all resolved values.
    """
    cfg = MergedConfig()

    # Merge DebateConfig
    if debate_config is not None:
        enable_adaptive_rounds = debate_config.enable_adaptive_rounds
        debate_strategy = debate_config.debate_strategy
        enable_agent_hierarchy = debate_config.enable_agent_hierarchy
        hierarchy_config = debate_config.hierarchy_config
        # Apply protocol overrides if protocol is provided
        if protocol is not None:
            debate_config.apply_to_protocol(protocol)

    # Merge AgentConfig
    if agent_config is not None:
        agent_weights = agent_config.agent_weights or agent_weights
        agent_selector = agent_config.agent_selector or agent_selector
        use_performance_selection = agent_config.use_performance_selection
        circuit_breaker = agent_config.circuit_breaker or circuit_breaker
        use_airlock = agent_config.use_airlock
        airlock_config = agent_config.airlock_config or airlock_config
        position_tracker = agent_config.position_tracker or position_tracker
        position_ledger = agent_config.position_ledger or position_ledger
        enable_position_ledger = agent_config.enable_position_ledger
        elo_system = agent_config.elo_system or elo_system
        calibration_tracker = agent_config.calibration_tracker or calibration_tracker
        relationship_tracker = agent_config.relationship_tracker or relationship_tracker
        persona_manager = agent_config.persona_manager or persona_manager
        vertical = agent_config.vertical or vertical
        vertical_persona_manager = agent_config.vertical_persona_manager or vertical_persona_manager
        auto_detect_vertical = agent_config.auto_detect_vertical
        fabric = agent_config.fabric or fabric
        fabric_config = agent_config.fabric_config or fabric_config

    # Merge MemoryConfig
    if memory_config is not None:
        memory = memory_config.memory or memory
        continuum_memory = memory_config.continuum_memory or continuum_memory
        consensus_memory = memory_config.consensus_memory or consensus_memory
        debate_embeddings = memory_config.debate_embeddings or debate_embeddings
        insight_store = memory_config.insight_store or insight_store
        dissent_retriever = memory_config.dissent_retriever or dissent_retriever
        flip_detector = memory_config.flip_detector or flip_detector
        moment_detector = memory_config.moment_detector or moment_detector
        tier_analytics_tracker = memory_config.tier_analytics_tracker or tier_analytics_tracker
        cross_debate_memory = memory_config.cross_debate_memory or cross_debate_memory
        enable_cross_debate_memory = memory_config.enable_cross_debate_memory
        knowledge_mound = memory_config.knowledge_mound or knowledge_mound
        auto_create_knowledge_mound = memory_config.auto_create_knowledge_mound
        enable_knowledge_retrieval = memory_config.enable_knowledge_retrieval
        enable_knowledge_ingestion = memory_config.enable_knowledge_ingestion
        enable_knowledge_extraction = memory_config.enable_knowledge_extraction
        extraction_min_confidence = memory_config.extraction_min_confidence
        enable_belief_guidance = memory_config.enable_belief_guidance
        enable_auto_revalidation = memory_config.enable_auto_revalidation
        revalidation_staleness_threshold = memory_config.revalidation_staleness_threshold
        revalidation_check_interval_seconds = memory_config.revalidation_check_interval_seconds
        revalidation_scheduler = memory_config.revalidation_scheduler or revalidation_scheduler
        use_rlm_limiter = memory_config.use_rlm_limiter
        rlm_limiter = memory_config.rlm_limiter or rlm_limiter
        rlm_compression_threshold = memory_config.rlm_compression_threshold
        rlm_max_recent_messages = memory_config.rlm_max_recent_messages
        rlm_summary_level = memory_config.rlm_summary_level
        rlm_compression_round_threshold = memory_config.rlm_compression_round_threshold
        checkpoint_manager = memory_config.checkpoint_manager or checkpoint_manager
        enable_checkpointing = memory_config.enable_checkpointing

    # Merge StreamingConfig
    if streaming_config is not None:
        event_hooks = streaming_config.event_hooks or event_hooks
        hook_manager = streaming_config.hook_manager or hook_manager
        event_emitter = streaming_config.event_emitter or event_emitter
        spectator = streaming_config.spectator or spectator
        recorder = streaming_config.recorder or recorder
        loop_id = streaming_config.loop_id or loop_id
        strict_loop_scoping = streaming_config.strict_loop_scoping
        skill_registry = streaming_config.skill_registry or skill_registry
        enable_skills = streaming_config.enable_skills
        propulsion_engine = streaming_config.propulsion_engine or propulsion_engine
        enable_propulsion = streaming_config.enable_propulsion

    # Merge ObservabilityConfig
    if observability_config is not None:
        performance_monitor = observability_config.performance_monitor or performance_monitor
        enable_performance_monitor = observability_config.enable_performance_monitor
        enable_telemetry = observability_config.enable_telemetry
        prompt_evolver = observability_config.prompt_evolver or prompt_evolver
        enable_prompt_evolution = observability_config.enable_prompt_evolution
        breakpoint_manager = observability_config.breakpoint_manager or breakpoint_manager
        trending_topic = observability_config.trending_topic or trending_topic
        pulse_manager = observability_config.pulse_manager or pulse_manager
        auto_fetch_trending = observability_config.auto_fetch_trending
        population_manager = observability_config.population_manager or population_manager
        auto_evolve = observability_config.auto_evolve
        breeding_threshold = observability_config.breeding_threshold
        evidence_collector = observability_config.evidence_collector or evidence_collector
        org_id = observability_config.org_id or org_id
        user_id = observability_config.user_id or user_id
        usage_tracker = observability_config.usage_tracker or usage_tracker
        broadcast_pipeline = observability_config.broadcast_pipeline or broadcast_pipeline
        auto_broadcast = observability_config.auto_broadcast
        broadcast_min_confidence = observability_config.broadcast_min_confidence
        training_exporter = observability_config.training_exporter or training_exporter
        auto_export_training = observability_config.auto_export_training
        training_export_min_confidence = observability_config.training_export_min_confidence
        enable_ml_delegation = observability_config.enable_ml_delegation
        ml_delegation_strategy = (
            observability_config.ml_delegation_strategy or ml_delegation_strategy
        )
        ml_delegation_weight = observability_config.ml_delegation_weight
        enable_quality_gates = observability_config.enable_quality_gates
        quality_gate_threshold = observability_config.quality_gate_threshold
        enable_consensus_estimation = observability_config.enable_consensus_estimation
        consensus_early_termination_threshold = (
            observability_config.consensus_early_termination_threshold
        )
        post_debate_workflow = observability_config.post_debate_workflow or post_debate_workflow
        enable_post_debate_workflow = observability_config.enable_post_debate_workflow
        post_debate_workflow_threshold = observability_config.post_debate_workflow_threshold
        initial_messages = observability_config.initial_messages or initial_messages

    # Store all resolved values
    cfg.protocol = protocol
    cfg.enable_adaptive_rounds = enable_adaptive_rounds
    cfg.debate_strategy = debate_strategy
    cfg.enable_agent_hierarchy = enable_agent_hierarchy
    cfg.hierarchy_config = hierarchy_config
    cfg.agent_weights = agent_weights
    cfg.agent_selector = agent_selector
    cfg.use_performance_selection = use_performance_selection
    cfg.circuit_breaker = circuit_breaker
    cfg.use_airlock = use_airlock
    cfg.airlock_config = airlock_config
    cfg.position_tracker = position_tracker
    cfg.position_ledger = position_ledger
    cfg.enable_position_ledger = enable_position_ledger
    cfg.elo_system = elo_system
    cfg.calibration_tracker = calibration_tracker
    cfg.relationship_tracker = relationship_tracker
    cfg.persona_manager = persona_manager
    cfg.vertical = vertical
    cfg.vertical_persona_manager = vertical_persona_manager
    cfg.auto_detect_vertical = auto_detect_vertical
    cfg.fabric = fabric
    cfg.fabric_config = fabric_config
    cfg.memory = memory
    cfg.continuum_memory = continuum_memory
    cfg.consensus_memory = consensus_memory
    cfg.debate_embeddings = debate_embeddings
    cfg.insight_store = insight_store
    cfg.dissent_retriever = dissent_retriever
    cfg.flip_detector = flip_detector
    cfg.moment_detector = moment_detector
    cfg.tier_analytics_tracker = tier_analytics_tracker
    cfg.cross_debate_memory = cross_debate_memory
    cfg.enable_cross_debate_memory = enable_cross_debate_memory
    cfg.knowledge_mound = knowledge_mound
    cfg.auto_create_knowledge_mound = auto_create_knowledge_mound
    cfg.enable_knowledge_retrieval = enable_knowledge_retrieval
    cfg.enable_knowledge_ingestion = enable_knowledge_ingestion
    cfg.enable_knowledge_extraction = enable_knowledge_extraction
    cfg.extraction_min_confidence = extraction_min_confidence
    cfg.enable_belief_guidance = enable_belief_guidance
    cfg.enable_auto_revalidation = enable_auto_revalidation
    cfg.revalidation_staleness_threshold = revalidation_staleness_threshold
    cfg.revalidation_check_interval_seconds = revalidation_check_interval_seconds
    cfg.revalidation_scheduler = revalidation_scheduler
    cfg.use_rlm_limiter = use_rlm_limiter
    cfg.rlm_limiter = rlm_limiter
    cfg.rlm_compression_threshold = rlm_compression_threshold
    cfg.rlm_max_recent_messages = rlm_max_recent_messages
    cfg.rlm_summary_level = rlm_summary_level
    cfg.rlm_compression_round_threshold = rlm_compression_round_threshold
    cfg.checkpoint_manager = checkpoint_manager
    cfg.enable_checkpointing = enable_checkpointing
    cfg.event_hooks = event_hooks
    cfg.hook_manager = hook_manager
    cfg.event_emitter = event_emitter
    cfg.spectator = spectator
    cfg.recorder = recorder
    cfg.loop_id = loop_id
    cfg.strict_loop_scoping = strict_loop_scoping
    cfg.skill_registry = skill_registry
    cfg.enable_skills = enable_skills
    cfg.propulsion_engine = propulsion_engine
    cfg.enable_propulsion = enable_propulsion
    cfg.performance_monitor = performance_monitor
    cfg.enable_performance_monitor = enable_performance_monitor
    cfg.enable_telemetry = enable_telemetry
    cfg.prompt_evolver = prompt_evolver
    cfg.enable_prompt_evolution = enable_prompt_evolution
    cfg.breakpoint_manager = breakpoint_manager
    cfg.trending_topic = trending_topic
    cfg.pulse_manager = pulse_manager
    cfg.auto_fetch_trending = auto_fetch_trending
    cfg.population_manager = population_manager
    cfg.auto_evolve = auto_evolve
    cfg.breeding_threshold = breeding_threshold
    cfg.evidence_collector = evidence_collector
    cfg.org_id = org_id
    cfg.user_id = user_id
    cfg.usage_tracker = usage_tracker
    cfg.broadcast_pipeline = broadcast_pipeline
    cfg.auto_broadcast = auto_broadcast
    cfg.broadcast_min_confidence = broadcast_min_confidence
    cfg.training_exporter = training_exporter
    cfg.auto_export_training = auto_export_training
    cfg.training_export_min_confidence = training_export_min_confidence
    cfg.enable_ml_delegation = enable_ml_delegation
    cfg.ml_delegation_strategy = ml_delegation_strategy
    cfg.ml_delegation_weight = ml_delegation_weight
    cfg.enable_quality_gates = enable_quality_gates
    cfg.quality_gate_threshold = quality_gate_threshold
    cfg.enable_consensus_estimation = enable_consensus_estimation
    cfg.consensus_early_termination_threshold = consensus_early_termination_threshold
    cfg.post_debate_workflow = post_debate_workflow
    cfg.enable_post_debate_workflow = enable_post_debate_workflow
    cfg.post_debate_workflow_threshold = post_debate_workflow_threshold
    cfg.initial_messages = initial_messages

    return cfg
