"""
Deliberation event types for Control Plane streaming.

Defines event types emitted during deliberation execution for real-time
monitoring via ControlPlaneStreamServer.
"""

from __future__ import annotations

from enum import Enum, unique


EVENT_PREFIX = "deliberation."

CATEGORY_LIFECYCLE = "lifecycle"
CATEGORY_ROUND = "round"
CATEGORY_AGENT = "agent"
CATEGORY_CONSENSUS = "consensus"
CATEGORY_SLA = "sla"
CATEGORY_PROGRESS = "progress"
CATEGORY_ERROR = "error"


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

    @property
    def category(self) -> str:
        """Return the category string for this event type."""
        return _EVENT_CATEGORIES.get(self, "unknown")

    @property
    def is_terminal(self) -> bool:
        """True if this event signals the end of a deliberation."""
        return self in _TERMINAL_EVENTS

    @classmethod
    def by_category(cls, category: str) -> frozenset[DeliberationEventType]:
        """Return all event types belonging to *category*."""
        return frozenset(e for e, c in _EVENT_CATEGORIES.items() if c == category)

    @classmethod
    def categories(cls) -> frozenset[str]:
        """Return the set of all known category names."""
        return frozenset(_EVENT_CATEGORIES.values())


_EVENT_CATEGORIES: dict["DeliberationEventType", str] = {
    DeliberationEventType.DELIBERATION_STARTED: CATEGORY_LIFECYCLE,
    DeliberationEventType.DELIBERATION_COMPLETED: CATEGORY_LIFECYCLE,
    DeliberationEventType.DELIBERATION_FAILED: CATEGORY_LIFECYCLE,
    DeliberationEventType.DELIBERATION_CANCELLED: CATEGORY_LIFECYCLE,
    DeliberationEventType.ROUND_START: CATEGORY_ROUND,
    DeliberationEventType.ROUND_END: CATEGORY_ROUND,
    DeliberationEventType.AGENT_MESSAGE: CATEGORY_AGENT,
    DeliberationEventType.AGENT_PROPOSAL: CATEGORY_AGENT,
    DeliberationEventType.AGENT_CRITIQUE: CATEGORY_AGENT,
    DeliberationEventType.AGENT_REVISION: CATEGORY_AGENT,
    DeliberationEventType.VOTE: CATEGORY_CONSENSUS,
    DeliberationEventType.CONSENSUS_CHECK: CATEGORY_CONSENSUS,
    DeliberationEventType.CONSENSUS_REACHED: CATEGORY_CONSENSUS,
    DeliberationEventType.NO_CONSENSUS: CATEGORY_CONSENSUS,
    DeliberationEventType.SLA_WARNING: CATEGORY_SLA,
    DeliberationEventType.SLA_CRITICAL: CATEGORY_SLA,
    DeliberationEventType.SLA_VIOLATED: CATEGORY_SLA,
    DeliberationEventType.PROGRESS_UPDATE: CATEGORY_PROGRESS,
    DeliberationEventType.CONVERGENCE_UPDATE: CATEGORY_PROGRESS,
    DeliberationEventType.AGENT_ERROR: CATEGORY_ERROR,
    DeliberationEventType.RECOVERY_ATTEMPTED: CATEGORY_ERROR,
}

_TERMINAL_EVENTS: frozenset[DeliberationEventType] = frozenset(
    {
        DeliberationEventType.DELIBERATION_COMPLETED,
        DeliberationEventType.DELIBERATION_FAILED,
        DeliberationEventType.DELIBERATION_CANCELLED,
    }
)

__all__ = [
    "EVENT_PREFIX",
    "CATEGORY_LIFECYCLE",
    "CATEGORY_ROUND",
    "CATEGORY_AGENT",
    "CATEGORY_CONSENSUS",
    "CATEGORY_SLA",
    "CATEGORY_PROGRESS",
    "CATEGORY_ERROR",
    "DeliberationEventType",
]
