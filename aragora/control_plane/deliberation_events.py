"""
Deliberation event types for Control Plane streaming.

Defines event types emitted during deliberation execution for real-time
monitoring via ControlPlaneStreamServer.
"""

from enum import Enum, unique


@unique
class DeliberationEventType(str, Enum):
    """Event types for deliberation lifecycle and progress."""

    # Lifecycle events
    DELIBERATION_STARTED = "deliberation.started"
    DELIBERATION_COMPLETED = "deliberation.completed"
    DELIBERATION_FAILED = "deliberation.failed"
    DELIBERATION_CANCELLED = "deliberation.cancelled"

    # Round events
    ROUND_START = "deliberation.round_start"
    ROUND_END = "deliberation.round_end"

    # Agent interaction events
    AGENT_MESSAGE = "deliberation.agent_message"
    AGENT_PROPOSAL = "deliberation.agent_proposal"
    AGENT_CRITIQUE = "deliberation.agent_critique"
    AGENT_REVISION = "deliberation.agent_revision"

    # Voting and consensus events
    VOTE = "deliberation.vote"
    CONSENSUS_CHECK = "deliberation.consensus_check"
    CONSENSUS_REACHED = "deliberation.consensus_reached"
    NO_CONSENSUS = "deliberation.no_consensus"

    # SLA events
    SLA_WARNING = "deliberation.sla_warning"
    SLA_CRITICAL = "deliberation.sla_critical"
    SLA_VIOLATED = "deliberation.sla_violated"

    # Progress events
    PROGRESS_UPDATE = "deliberation.progress"
    CONVERGENCE_UPDATE = "deliberation.convergence"

    # Error events
    AGENT_ERROR = "deliberation.agent_error"
    RECOVERY_ATTEMPTED = "deliberation.recovery_attempted"

    @classmethod
    def lifecycle_events(cls) -> list["DeliberationEventType"]:
        """Return lifecycle-related event types."""
        return [
            cls.DELIBERATION_STARTED,
            cls.DELIBERATION_COMPLETED,
            cls.DELIBERATION_FAILED,
            cls.DELIBERATION_CANCELLED,
        ]

    @classmethod
    def agent_events(cls) -> list["DeliberationEventType"]:
        """Return agent interaction event types."""
        return [
            cls.AGENT_MESSAGE,
            cls.AGENT_PROPOSAL,
            cls.AGENT_CRITIQUE,
            cls.AGENT_REVISION,
        ]

    @classmethod
    def sla_events(cls) -> list["DeliberationEventType"]:
        """Return SLA-related event types."""
        return [cls.SLA_WARNING, cls.SLA_CRITICAL, cls.SLA_VIOLATED]

    @property
    def category(self) -> str:
        """Extract the event category from the value (e.g. 'sla_warning' -> 'sla')."""
        suffix = self.value.removeprefix("deliberation.")
        return suffix.split("_")[0] if "_" in suffix else suffix


__all__ = ["DeliberationEventType"]
