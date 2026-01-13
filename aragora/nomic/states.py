"""
Nomic Loop State Definitions.

Defines all possible states and valid transitions for the nomic loop
state machine. Each state is idempotent and can be safely retried.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional, Set


class NomicState(Enum):
    """
    States in the nomic loop state machine.

    State Flow:
        IDLE -> CONTEXT -> DEBATE -> DESIGN -> IMPLEMENT -> VERIFY -> COMMIT -> IDLE
                  |          |         |          |           |         |
                  +----------+-------- +----------+-----------+---------+
                                       |
                                       v
                                   RECOVERY -> IDLE
    """

    # Initial/terminal states
    IDLE = auto()  # Waiting for trigger

    # Active phases
    CONTEXT = auto()  # Gathering codebase context
    DEBATE = auto()  # Multi-agent debate on improvements
    DESIGN = auto()  # Designing the implementation
    IMPLEMENT = auto()  # Writing code changes
    VERIFY = auto()  # Testing and validation
    COMMIT = auto()  # Committing approved changes

    # Error handling
    RECOVERY = auto()  # Handling errors, deciding next action

    # Terminal states
    COMPLETED = auto()  # Cycle completed successfully
    FAILED = auto()  # Cycle failed, cannot recover
    PAUSED = auto()  # Manually paused, awaiting resume


# Valid state transitions
# All active states can transition to PAUSED (for manual pause)
# and RECOVERY (for error handling)
VALID_TRANSITIONS: Dict[NomicState, Set[NomicState]] = {
    NomicState.IDLE: {NomicState.CONTEXT, NomicState.PAUSED},
    NomicState.CONTEXT: {NomicState.DEBATE, NomicState.RECOVERY, NomicState.PAUSED},
    NomicState.DEBATE: {
        NomicState.DESIGN,
        NomicState.RECOVERY,
        NomicState.IDLE,
        NomicState.PAUSED,
        NomicState.COMPLETED,
    },
    NomicState.DESIGN: {
        NomicState.IMPLEMENT,
        NomicState.RECOVERY,
        NomicState.DEBATE,
        NomicState.PAUSED,
        NomicState.COMPLETED,
    },
    NomicState.IMPLEMENT: {
        NomicState.VERIFY,
        NomicState.RECOVERY,
        NomicState.PAUSED,
        NomicState.COMPLETED,
    },
    NomicState.VERIFY: {
        NomicState.COMMIT,
        NomicState.IMPLEMENT,
        NomicState.RECOVERY,
        NomicState.PAUSED,
    },
    NomicState.COMMIT: {NomicState.COMPLETED, NomicState.RECOVERY, NomicState.PAUSED},
    NomicState.RECOVERY: {
        NomicState.IDLE,
        NomicState.CONTEXT,
        NomicState.DEBATE,
        NomicState.DESIGN,
        NomicState.IMPLEMENT,
        NomicState.VERIFY,
        NomicState.FAILED,
        NomicState.PAUSED,
        NomicState.COMPLETED,
    },
    NomicState.COMPLETED: {NomicState.IDLE},
    NomicState.FAILED: {NomicState.IDLE},
    # PAUSED can resume to any active state (stored in previous_state)
    NomicState.PAUSED: {
        NomicState.IDLE,
        NomicState.CONTEXT,
        NomicState.DEBATE,
        NomicState.DESIGN,
        NomicState.IMPLEMENT,
        NomicState.VERIFY,
        NomicState.COMMIT,
    },
}


@dataclass
class StateMetadata:
    """Metadata for a state in the state machine."""

    name: str
    description: str
    timeout_seconds: int
    max_retries: int
    is_critical: bool  # If True, failure goes to FAILED instead of RECOVERY
    requires_checkpoint: bool  # If True, must checkpoint after this state


# State configuration
STATE_CONFIG: Dict[NomicState, StateMetadata] = {
    NomicState.IDLE: StateMetadata(
        name="idle",
        description="Waiting for trigger to start cycle",
        timeout_seconds=0,  # No timeout
        max_retries=0,
        is_critical=False,
        requires_checkpoint=False,
    ),
    NomicState.CONTEXT: StateMetadata(
        name="context",
        description="Gathering codebase context and identifying improvements",
        timeout_seconds=1200,  # 20 minutes
        max_retries=2,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.DEBATE: StateMetadata(
        name="debate",
        description="Multi-agent debate on proposed improvements",
        timeout_seconds=3600,  # 60 minutes
        max_retries=1,
        is_critical=True,
        requires_checkpoint=True,
    ),
    NomicState.DESIGN: StateMetadata(
        name="design",
        description="Designing implementation approach",
        timeout_seconds=1800,  # 30 minutes
        max_retries=2,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.IMPLEMENT: StateMetadata(
        name="implement",
        description="Writing code changes",
        timeout_seconds=2400,  # 40 minutes
        max_retries=1,
        is_critical=True,
        requires_checkpoint=True,
    ),
    NomicState.VERIFY: StateMetadata(
        name="verify",
        description="Running tests and validation",
        timeout_seconds=1800,  # 30 minutes
        max_retries=3,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.COMMIT: StateMetadata(
        name="commit",
        description="Committing approved changes",
        timeout_seconds=300,  # 5 minutes
        max_retries=1,
        is_critical=True,
        requires_checkpoint=True,
    ),
    NomicState.RECOVERY: StateMetadata(
        name="recovery",
        description="Handling errors and deciding recovery strategy",
        timeout_seconds=300,  # 5 minutes
        max_retries=0,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.COMPLETED: StateMetadata(
        name="completed",
        description="Cycle completed successfully",
        timeout_seconds=0,
        max_retries=0,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.FAILED: StateMetadata(
        name="failed",
        description="Cycle failed, cannot recover",
        timeout_seconds=0,
        max_retries=0,
        is_critical=False,
        requires_checkpoint=True,
    ),
    NomicState.PAUSED: StateMetadata(
        name="paused",
        description="Manually paused, awaiting resume",
        timeout_seconds=0,
        max_retries=0,
        is_critical=False,
        requires_checkpoint=True,
    ),
}


@dataclass
class StateContext:
    """
    Context passed between states.

    Contains all data accumulated during the nomic loop cycle.
    This is checkpointed after each state transition.
    """

    # Cycle metadata
    cycle_id: str = ""
    started_at: Optional[datetime] = None
    current_state: NomicState = NomicState.IDLE
    previous_state: Optional[NomicState] = None

    # Phase outputs
    context_result: Optional[Dict[str, Any]] = None
    debate_result: Optional[Dict[str, Any]] = None
    design_result: Optional[Dict[str, Any]] = None
    implement_result: Optional[Dict[str, Any]] = None
    verify_result: Optional[Dict[str, Any]] = None
    commit_result: Optional[Dict[str, Any]] = None

    # Error tracking
    errors: list = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)

    # Agent health
    agent_failures: Dict[str, int] = field(default_factory=dict)
    circuit_breaker_states: Dict[str, bool] = field(default_factory=dict)

    # Metrics
    state_durations: Dict[str, float] = field(default_factory=dict)
    total_tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary for checkpointing."""
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "current_state": self.current_state.name,
            "previous_state": self.previous_state.name if self.previous_state else None,
            "context_result": self.context_result,
            "debate_result": self.debate_result,
            "design_result": self.design_result,
            "implement_result": self.implement_result,
            "verify_result": self.verify_result,
            "commit_result": self.commit_result,
            "errors": self.errors,
            "retry_counts": self.retry_counts,
            "agent_failures": self.agent_failures,
            "circuit_breaker_states": self.circuit_breaker_states,
            "state_durations": self.state_durations,
            "total_tokens_used": self.total_tokens_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateContext":
        """Deserialize context from dictionary."""
        ctx = cls()
        ctx.cycle_id = data.get("cycle_id", "")
        ctx.started_at = (
            datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        )
        ctx.current_state = NomicState[data.get("current_state", "IDLE")]
        ctx.previous_state = (
            NomicState[data["previous_state"]] if data.get("previous_state") else None
        )
        ctx.context_result = data.get("context_result")
        ctx.debate_result = data.get("debate_result")
        ctx.design_result = data.get("design_result")
        ctx.implement_result = data.get("implement_result")
        ctx.verify_result = data.get("verify_result")
        ctx.commit_result = data.get("commit_result")
        ctx.errors = data.get("errors", [])
        ctx.retry_counts = data.get("retry_counts", {})
        ctx.agent_failures = data.get("agent_failures", {})
        ctx.circuit_breaker_states = data.get("circuit_breaker_states", {})
        ctx.state_durations = data.get("state_durations", {})
        ctx.total_tokens_used = data.get("total_tokens_used", 0)
        return ctx


def is_valid_transition(from_state: NomicState, to_state: NomicState) -> bool:
    """Check if a state transition is valid."""
    return to_state in VALID_TRANSITIONS.get(from_state, set())


def get_state_config(state: NomicState) -> StateMetadata:
    """Get configuration for a state."""
    return STATE_CONFIG.get(state, STATE_CONFIG[NomicState.IDLE])
