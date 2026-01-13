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
from aragora.nomic.states import (
    NomicState,
    StateContext,
    StateMetadata,
    VALID_TRANSITIONS,
    STATE_CONFIG,
    is_valid_transition,
    get_state_config,
)
from aragora.nomic.events import (
    Event,
    EventType,
    EventLog,
    start_event,
    stop_event,
    pause_event,
    error_event,
    timeout_event,
    retry_event,
    phase_complete_event,
    agent_failed_event,
    circuit_open_event,
    rollback_event,
    checkpoint_loaded_event,
)
from aragora.nomic.state_machine import (
    NomicStateMachine,
    TransitionError,
    StateTimeoutError,
    create_nomic_state_machine,
)
from aragora.nomic.checkpoints import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    list_checkpoints,
    cleanup_old_checkpoints,
)
from aragora.nomic.recovery import (
    RecoveryStrategy,
    RecoveryDecision,
    RecoveryManager,
    CircuitBreaker,
    CircuitBreakerRegistry,
    calculate_backoff,
    recovery_handler,
)
from aragora.nomic.handlers import (
    create_handlers,
    create_context_handler,
    create_debate_handler,
    create_design_handler,
    create_implement_handler,
    create_verify_handler,
    create_commit_handler,
)

# Phase implementations
from aragora.nomic.phases import (
    ContextPhase,
    DebatePhase,
    DesignPhase,
    ImplementPhase,
    VerifyPhase,
    CommitPhase,
    DebateConfig,
    DesignConfig,
    LearningContext,
    BeliefContext,
    PostDebateHooks,
    ScopeLimiter,
    check_design_scope,
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
