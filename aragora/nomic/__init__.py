"""
Nomic Loop Module.

Provides the nomic loop self-improvement cycle with two implementations:

1. **State Machine (New)** - Event-driven, robust, checkpoint-resumable
   - NomicStateMachine: Core state machine
   - NomicState: State enum
   - Event, EventType: Event system
   - CheckpointManager: Persistence
   - RecoveryManager: Error recovery

2. **Integration (Legacy)** - Phase-based, integrated with aragora features
   - NomicIntegration: Integration hub
   - Preflight checks

The nomic loop is a 6-phase cycle:
1. Context - Gather codebase understanding
2. Debate - Multi-agent debate on improvements
3. Design - Design the implementation
4. Implement - Write the code changes
5. Verify - Test and validate changes
6. Commit - Commit approved changes

Migration: The state machine is the recommended approach for new deployments.
Legacy integration is preserved for backward compatibility.
"""

# State Machine (New - recommended)
from aragora.nomic.checkpoints import (
    CheckpointManager,
    cleanup_old_checkpoints,
    list_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    save_checkpoint,
)
from aragora.nomic.events import (
    Event,
    EventLog,
    EventType,
    agent_failed_event,
    checkpoint_loaded_event,
    circuit_open_event,
    error_event,
    pause_event,
    phase_complete_event,
    retry_event,
    rollback_event,
    start_event,
    stop_event,
    timeout_event,
)
from aragora.nomic.handlers import (
    create_commit_handler,
    create_context_handler,
    create_debate_handler,
    create_design_handler,
    create_handlers,
    create_implement_handler,
    create_verify_handler,
)

# Phase implementations
from aragora.nomic.phases import (
    BeliefContext,
    CommitPhase,
    ContextPhase,
    DebateConfig,
    DebatePhase,
    DesignConfig,
    DesignPhase,
    ImplementPhase,
    LearningContext,
    PostDebateHooks,
    ScopeLimiter,
    VerifyPhase,
    check_design_scope,
)
from aragora.nomic.metrics import (
    NOMIC_CIRCUIT_BREAKERS_OPEN,
    NOMIC_CURRENT_PHASE,
    NOMIC_CYCLES_IN_PROGRESS,
    NOMIC_CYCLES_TOTAL,
    NOMIC_ERRORS,
    NOMIC_PHASE_DURATION,
    NOMIC_PHASE_LAST_TRANSITION,
    NOMIC_PHASE_TRANSITIONS,
    NOMIC_RECOVERY_DECISIONS,
    NOMIC_RETRIES,
    PHASE_ENCODING,
    check_stuck_phases,
    create_metrics_callback,
    get_nomic_metrics_summary,
    nomic_metrics_callback,
    track_cycle_complete,
    track_cycle_start,
    track_error,
    track_phase_transition,
    track_recovery_decision,
    track_retry,
    update_circuit_breaker_count,
)
from aragora.nomic.recovery import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    RecoveryDecision,
    RecoveryManager,
    RecoveryStrategy,
    calculate_backoff,
    recovery_handler,
)
from aragora.nomic.state_machine import (
    NomicStateMachine,
    StateTimeoutError,
    TransitionError,
    create_nomic_state_machine,
)
from aragora.nomic.states import (
    STATE_CONFIG,
    VALID_TRANSITIONS,
    NomicState,
    StateContext,
    StateMetadata,
    get_state_config,
    is_valid_transition,
)


# Legacy Integration (lazy imports to avoid circular dependencies)
def __getattr__(name):
    """Lazy import legacy integration modules."""
    legacy_integration = {
        "NomicIntegration",
        "BeliefAnalysis",
        "AgentReliability",
        "StalenessReport",
        "PhaseCheckpoint",
        "create_nomic_integration",
    }
    legacy_preflight = {
        "PreflightHealthCheck",
        "PreflightResult",
        "CheckResult",
        "CheckStatus",
        "run_preflight",
    }

    if name in legacy_integration:
        from aragora.nomic import integration

        return getattr(integration, name)
    elif name in legacy_preflight:
        from aragora.nomic import preflight

        return getattr(preflight, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # State Machine (New)
    "NomicStateMachine",
    "NomicState",
    "StateContext",
    "StateMetadata",
    "VALID_TRANSITIONS",
    "STATE_CONFIG",
    "is_valid_transition",
    "get_state_config",
    "create_nomic_state_machine",
    "TransitionError",
    "StateTimeoutError",
    # Events
    "Event",
    "EventType",
    "EventLog",
    "start_event",
    "stop_event",
    "pause_event",
    "error_event",
    "timeout_event",
    "retry_event",
    "phase_complete_event",
    "agent_failed_event",
    "circuit_open_event",
    "rollback_event",
    "checkpoint_loaded_event",
    # Checkpoints
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_checkpoint",
    "list_checkpoints",
    "cleanup_old_checkpoints",
    # Recovery
    "RecoveryStrategy",
    "RecoveryDecision",
    "RecoveryManager",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "calculate_backoff",
    "recovery_handler",
    # Metrics
    "NOMIC_PHASE_TRANSITIONS",
    "NOMIC_CURRENT_PHASE",
    "NOMIC_PHASE_DURATION",
    "NOMIC_CYCLES_TOTAL",
    "NOMIC_CYCLES_IN_PROGRESS",
    "NOMIC_PHASE_LAST_TRANSITION",
    "NOMIC_CIRCUIT_BREAKERS_OPEN",
    "NOMIC_ERRORS",
    "NOMIC_RECOVERY_DECISIONS",
    "NOMIC_RETRIES",
    "PHASE_ENCODING",
    "track_phase_transition",
    "track_cycle_start",
    "track_cycle_complete",
    "track_error",
    "track_recovery_decision",
    "track_retry",
    "update_circuit_breaker_count",
    "nomic_metrics_callback",
    "create_metrics_callback",
    "get_nomic_metrics_summary",
    "check_stuck_phases",
    # Handlers
    "create_handlers",
    "create_context_handler",
    "create_debate_handler",
    "create_design_handler",
    "create_implement_handler",
    "create_verify_handler",
    "create_commit_handler",
    # Legacy Integration
    "NomicIntegration",
    "BeliefAnalysis",
    "AgentReliability",
    "StalenessReport",
    "PhaseCheckpoint",
    "create_nomic_integration",
    # Preflight
    "PreflightHealthCheck",
    "PreflightResult",
    "CheckResult",
    "CheckStatus",
    "run_preflight",
    # Phase implementations
    "ContextPhase",
    "DebatePhase",
    "DesignPhase",
    "ImplementPhase",
    "VerifyPhase",
    "CommitPhase",
    "DebateConfig",
    "DesignConfig",
    "LearningContext",
    "BeliefContext",
    "PostDebateHooks",
    "ScopeLimiter",
    "check_design_scope",
]
