"""
Debate-specific exception re-exports.

Provides a convenient single import for all debate-related exceptions.
The canonical definitions live in aragora.exceptions; this module
re-exports them for domain-local imports:

    from aragora.debate.exceptions import ConsensusError, DebateTimeoutError
"""

from aragora.exceptions import (
    AragoraError,
    ConsensusError,
    ConsensusTimeoutError,
    DebateBatchError,
    DebateConfigurationError,
    DebateError,
    DebateExecutionError,
    DebateNotFoundError,
    DebateStartError,
    EarlyStopError,
    PhaseExecutionError,
    RoundLimitExceededError,
    VoteProcessingError,
    VoteValidationError,
)

# Additional debate-specific exceptions not yet in the main hierarchy


class DebateTimeoutError(DebateError):
    """A debate exceeded its time limit."""

    def __init__(self, debate_id: str = "", timeout_seconds: float = 0):
        msg = "Debate timed out"
        if debate_id:
            msg = f"Debate {debate_id} timed out"
        if timeout_seconds:
            msg += f" after {timeout_seconds}s"
        super().__init__(msg)
        self.debate_id = debate_id
        self.timeout_seconds = timeout_seconds


class AgentCircuitOpenError(DebateError):
    """An agent's circuit breaker is open, preventing participation."""

    def __init__(self, agent_name: str = ""):
        msg = "Agent circuit breaker open"
        if agent_name:
            msg = f"Agent '{agent_name}' circuit breaker open"
        super().__init__(msg)
        self.agent_name = agent_name


class CheckpointError(AragoraError):
    """Error saving or loading a debate checkpoint."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """A referenced checkpoint does not exist."""

    pass


__all__ = [
    # Re-exported from aragora.exceptions
    "AragoraError",
    "ConsensusError",
    "ConsensusTimeoutError",
    "DebateBatchError",
    "DebateConfigurationError",
    "DebateError",
    "DebateExecutionError",
    "DebateNotFoundError",
    "DebateStartError",
    "EarlyStopError",
    "PhaseExecutionError",
    "RoundLimitExceededError",
    "VoteProcessingError",
    "VoteValidationError",
    # New debate-specific exceptions
    "DebateTimeoutError",
    "AgentCircuitOpenError",
    "CheckpointError",
    "CheckpointNotFoundError",
]
