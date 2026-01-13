"""
Arena configuration dataclass.

Extracted from orchestrator.py for modularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aragora.debate.protocol import CircuitBreaker
from aragora.spectate.stream import SpectatorStream


@dataclass
class ArenaConfig:
    """Configuration for Arena debate orchestration.

    Groups optional dependencies and settings that can be passed to Arena.
    This allows for cleaner initialization and easier testing.

    Initialization Flow
    -------------------
    Arena initialization follows a layered architecture pattern:

    1. **Core Configuration** (_init_core):
       - Sets up environment, agents, protocol, spectator
       - Initializes circuit breaker for fault tolerance
       - Configures loop scoping for multi-debate sessions

    2. **Tracking Subsystems** (_init_trackers):
       - Position and belief tracking (PositionTracker, PositionLedger)
       - ELO rating system for agent ranking
       - Persona manager for agent specialization
       - Flip detector for position reversals
       - Relationship tracker for agent interactions
       - Moment detector for significant events

    3. **User Participation** (_init_user_participation):
       - Sets up event queue for user votes/suggestions
       - Subscribes to event emitter for real-time participation

    4. **Roles and Stances** (_init_roles_and_stances):
       - Initializes cognitive role rotation (Heavy3-inspired)
       - Sets up initial agent stances

    5. **Convergence Detection** (_init_convergence):
       - Configures semantic similarity backend
       - Sets up convergence thresholds

    6. **Phase Classes** (_init_phases):
       - ContextInitializer, ProposalPhase, DebateRoundsPhase
       - ConsensusPhase, AnalyticsPhase, FeedbackPhase, VotingPhase

    Dependency Injection
    --------------------
    Most subsystems are optional and can be injected for testing:

    - Memory systems: critique_store, continuum_memory, debate_embeddings
    - Tracking: elo_system, position_ledger, relationship_tracker
    - Events: event_emitter, spectator
    - Recording: recorder, evidence_collector

    Example
    -------
    Basic usage with minimal configuration::

        config = ArenaConfig(loop_id="debate-123")
        arena = Arena.from_config(env, agents, protocol, config)

    Full production setup with all subsystems::

        config = ArenaConfig(
            loop_id="debate-123",
            strict_loop_scoping=True,
            memory=critique_store,
            continuum_memory=continuum,
            elo_system=elo,
            event_emitter=emitter,
            spectator=stream,
        )
        arena = Arena.from_config(env, agents, protocol, config)
    """

    # Identification
    loop_id: str = ""

    # Behavior flags
    strict_loop_scoping: bool = False

    # Core subsystems (typically injected)
    memory: Optional[object] = None  # CritiqueStore
    event_hooks: Optional[dict] = None
    event_emitter: Optional[object] = None
    spectator: Optional[SpectatorStream] = None
    debate_embeddings: Optional[object] = None  # DebateEmbeddingsDatabase
    insight_store: Optional[object] = None  # InsightStore
    recorder: Optional[object] = None  # ReplayRecorder
    circuit_breaker: Optional[CircuitBreaker] = None
    evidence_collector: Optional[object] = None

    # Agent configuration
    agent_weights: Optional[dict] = None

    # Tracking subsystems
    position_tracker: Optional[object] = None
    position_ledger: Optional[object] = None
    enable_position_ledger: bool = False  # Auto-create PositionLedger if not provided
    elo_system: Optional[object] = None
    persona_manager: Optional[object] = None
    dissent_retriever: Optional[object] = None
    consensus_memory: Optional[object] = None  # ConsensusMemory for historical outcomes
    flip_detector: Optional[object] = None
    calibration_tracker: Optional[object] = None
    continuum_memory: Optional[object] = None
    relationship_tracker: Optional[object] = None
    moment_detector: Optional[object] = None
    tier_analytics_tracker: Optional[object] = None  # TierAnalyticsTracker for memory ROI

    # Genesis evolution
    population_manager: Optional[object] = None  # PopulationManager for genome evolution
    auto_evolve: bool = False  # Trigger evolution after high-quality debates
    breeding_threshold: float = 0.8  # Min confidence to trigger evolution

    # Fork/continuation support
    initial_messages: Optional[list] = None
    trending_topic: Optional[object] = None
    pulse_manager: Optional[object] = None  # PulseManager for auto-fetching trending topics
    auto_fetch_trending: bool = False  # Auto-fetch trending topics if none provided

    # Human-in-the-loop breakpoints
    breakpoint_manager: Optional[object] = None  # BreakpointManager

    # Debate checkpointing for resume support
    checkpoint_manager: Optional[object] = None  # CheckpointManager for pause/resume
    enable_checkpointing: bool = False  # Auto-create CheckpointManager if True

    # Performance telemetry
    performance_monitor: Optional[object] = None  # AgentPerformanceMonitor
    enable_performance_monitor: bool = False
    enable_telemetry: bool = False  # Enable Prometheus/Blackbox telemetry emission

    # Agent selection (performance-based team formation)
    agent_selector: Optional[object] = None  # AgentSelector for performance-based selection
    use_performance_selection: bool = False  # Enable ELO/calibration-based agent selection

    # Airlock resilience layer
    use_airlock: bool = False  # Wrap agents with AirlockProxy for timeout/fallback
    airlock_config: Optional[object] = None  # AirlockConfig for customization

    # Prompt evolution for self-improvement
    prompt_evolver: Optional[object] = None  # PromptEvolver for extracting winning patterns
    enable_prompt_evolution: bool = False  # Auto-create PromptEvolver if True

    # Billing/usage tracking (multi-tenancy)
    org_id: str = ""  # Organization ID for multi-tenancy
    user_id: str = ""  # User ID for usage attribution
    usage_tracker: Optional[object] = None  # UsageTracker for token usage

    # Broadcast auto-trigger for high-quality debates
    broadcast_pipeline: Optional[object] = None  # BroadcastPipeline for audio/video generation
    auto_broadcast: bool = False  # Auto-trigger broadcast after high-quality debates
    broadcast_min_confidence: float = 0.8  # Minimum confidence to trigger broadcast
    broadcast_platforms: list = None  # Platforms to publish to (default: ["rss"])

    # Training data export (Tinker integration)
    training_exporter: Optional[object] = None  # TrainingDataExporter for auto-export
    auto_export_training: bool = False  # Auto-export training data after debates
    training_export_min_confidence: float = 0.75  # Min confidence to export as SFT
    training_export_path: str = ""  # Output path for training data (default: data/training/)

    def __post_init__(self):
        """Initialize defaults that can't be set in field definitions."""
        if self.broadcast_platforms is None:
            self.broadcast_platforms = ["rss"]


__all__ = ["ArenaConfig"]
