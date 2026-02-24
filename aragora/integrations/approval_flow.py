"""
Approval flow management for channel-delivered decision receipts.

Tracks approval state for debate decisions delivered to Slack/Teams threads,
supporting multi-approver workflows with configurable timeout escalation.

States: PENDING -> APPROVED | REJECTED | ESCALATED | RE_DEBATE

Usage:
    manager = ApprovalFlowManager()

    # Create an approval request
    flow_id = await manager.create_approval(
        debate_id="abc-123",
        receipt_id="rcpt_xyz",
        channel="slack",
        channel_id="C01ABC",
        thread_id="1234567890.123456",
        required_approvers=2,
    )

    # Record a decision
    await manager.record_decision(
        flow_id=flow_id,
        user_id="U01XYZ",
        decision="approved",
    )

    # Check status
    status = manager.get_status(flow_id)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalState(str, Enum):
    """Possible states for an approval flow."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    RE_DEBATE = "re_debate"


class ApprovalDecision(str, Enum):
    """Individual approver decisions."""

    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    RE_DEBATE = "re_debate"


@dataclass
class ApprovalVote:
    """A single approver's decision."""

    user_id: str
    decision: str
    reason: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ApprovalFlow:
    """Tracks the approval workflow for a single debate decision."""

    flow_id: str
    debate_id: str
    receipt_id: str
    channel: str  # "slack" or "teams"
    channel_id: str
    thread_id: str
    state: str = ApprovalState.PENDING.value
    required_approvers: int = 1
    votes: list[ApprovalVote] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    timeout_seconds: int = 86400  # 24 hours
    escalation_target: str = ""  # user/group to escalate to
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def approval_count(self) -> int:
        """Number of approval votes."""
        return sum(1 for v in self.votes if v.decision == ApprovalDecision.APPROVED.value)

    @property
    def rejection_count(self) -> int:
        """Number of rejection votes."""
        return sum(1 for v in self.votes if v.decision == ApprovalDecision.REJECTED.value)

    @property
    def escalation_count(self) -> int:
        """Number of escalation votes."""
        return sum(1 for v in self.votes if v.decision == ApprovalDecision.ESCALATED.value)

    @property
    def re_debate_count(self) -> int:
        """Number of re-debate votes."""
        return sum(1 for v in self.votes if v.decision == ApprovalDecision.RE_DEBATE.value)

    @property
    def is_expired(self) -> bool:
        """Check if the approval flow has timed out."""
        try:
            created = datetime.fromisoformat(self.created_at)
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            return elapsed > self.timeout_seconds
        except (ValueError, TypeError):
            return False

    def has_voted(self, user_id: str) -> bool:
        """Check if a user has already voted."""
        return any(v.user_id == user_id for v in self.votes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "flow_id": self.flow_id,
            "debate_id": self.debate_id,
            "receipt_id": self.receipt_id,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "state": self.state,
            "required_approvers": self.required_approvers,
            "votes": [
                {
                    "user_id": v.user_id,
                    "decision": v.decision,
                    "reason": v.reason,
                    "timestamp": v.timestamp,
                }
                for v in self.votes
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "timeout_seconds": self.timeout_seconds,
            "escalation_target": self.escalation_target,
            "metadata": self.metadata,
        }


class ApprovalFlowManager:
    """Manages approval workflows for debate decisions.

    Supports multi-approver workflows where N of M approvals are required.
    Automatically escalates if no response within the configurable timeout.

    Args:
        default_timeout: Default timeout in seconds (default 24h).
        governance_store: Optional governance store for persistence.
    """

    def __init__(
        self,
        default_timeout: int = 86400,
        governance_store: Any | None = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._governance_store = governance_store
        self._flows: dict[str, ApprovalFlow] = {}
        self._debate_index: dict[str, str] = {}  # debate_id -> flow_id
        self._event_listeners: list[Any] = []

    def add_event_listener(self, listener: Any) -> None:
        """Register a listener for approval state change events.

        The listener must be callable with signature:
        ``listener(event_type: str, flow: ApprovalFlow)``
        """
        self._event_listeners.append(listener)

    def _emit_event(self, event_type: str, flow: ApprovalFlow) -> None:
        """Emit an approval state change event."""
        for listener in self._event_listeners:
            try:
                listener(event_type, flow)
            except (TypeError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("Event listener failed: %s", exc)

    async def create_approval(
        self,
        debate_id: str,
        receipt_id: str,
        channel: str,
        channel_id: str,
        thread_id: str,
        required_approvers: int = 1,
        timeout_seconds: int | None = None,
        escalation_target: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new approval flow for a debate decision.

        Args:
            debate_id: The debate that produced the decision.
            receipt_id: The decision receipt ID.
            channel: Platform name ("slack" or "teams").
            channel_id: Channel/conversation ID.
            thread_id: Thread timestamp or message ID.
            required_approvers: Number of approvals needed (N of M).
            timeout_seconds: Custom timeout (defaults to instance default).
            escalation_target: User/group ID to escalate to on timeout.
            metadata: Additional metadata to store.

        Returns:
            The generated flow_id.
        """
        flow_id = f"af_{uuid.uuid4().hex[:12]}"
        flow = ApprovalFlow(
            flow_id=flow_id,
            debate_id=debate_id,
            receipt_id=receipt_id,
            channel=channel,
            channel_id=channel_id,
            thread_id=thread_id,
            required_approvers=max(1, required_approvers),
            timeout_seconds=timeout_seconds or self._default_timeout,
            escalation_target=escalation_target,
            metadata=metadata or {},
        )

        self._flows[flow_id] = flow
        self._debate_index[debate_id] = flow_id

        # Persist to governance store if available
        self._persist_flow(flow)

        self._emit_event("approval_created", flow)
        logger.info(
            "Created approval flow %s for debate %s (requires %d approvals)",
            flow_id,
            debate_id,
            required_approvers,
        )

        return flow_id

    async def record_decision(
        self,
        flow_id: str,
        user_id: str,
        decision: str,
        reason: str = "",
    ) -> ApprovalFlow | None:
        """Record an approver's decision.

        Args:
            flow_id: The approval flow ID.
            user_id: The user making the decision.
            decision: One of "approved", "rejected", "escalated", "re_debate".
            reason: Optional reason for the decision.

        Returns:
            The updated ApprovalFlow, or None if not found or already decided.
        """
        flow = self._flows.get(flow_id)
        if flow is None:
            logger.warning("Approval flow %s not found", flow_id)
            return None

        if flow.state != ApprovalState.PENDING.value:
            logger.info(
                "Approval flow %s already in state %s, ignoring decision",
                flow_id,
                flow.state,
            )
            return flow

        # Validate decision value
        valid_decisions = {d.value for d in ApprovalDecision}
        if decision not in valid_decisions:
            logger.warning("Invalid decision value: %s", decision)
            return None

        # Check for duplicate votes
        if flow.has_voted(user_id):
            logger.info("User %s already voted on flow %s", user_id, flow_id)
            return flow

        # Record the vote
        vote = ApprovalVote(
            user_id=user_id,
            decision=decision,
            reason=reason,
        )
        flow.votes.append(vote)
        flow.updated_at = datetime.now(timezone.utc).isoformat()

        # Evaluate new state
        self._evaluate_state(flow)

        # Persist
        self._persist_flow(flow)

        self._emit_event("decision_recorded", flow)
        logger.info(
            "Recorded %s from %s on flow %s (state: %s)",
            decision,
            user_id,
            flow_id,
            flow.state,
        )

        return flow

    def _evaluate_state(self, flow: ApprovalFlow) -> None:
        """Evaluate and update the flow state based on current votes."""
        # Any escalation immediately escalates
        if flow.escalation_count > 0:
            flow.state = ApprovalState.ESCALATED.value
            return

        # Any re-debate request triggers re-debate
        if flow.re_debate_count > 0:
            flow.state = ApprovalState.RE_DEBATE.value
            return

        # Check if enough rejections to reject outright
        # A single rejection is sufficient to reject
        if flow.rejection_count > 0:
            flow.state = ApprovalState.REJECTED.value
            return

        # Check if we have enough approvals
        if flow.approval_count >= flow.required_approvers:
            flow.state = ApprovalState.APPROVED.value
            return

        # Otherwise stay pending

    def get_status(self, flow_id: str) -> ApprovalFlow | None:
        """Get the current status of an approval flow.

        Also checks for timeout-based auto-escalation.

        Args:
            flow_id: The approval flow ID.

        Returns:
            The ApprovalFlow, or None if not found.
        """
        flow = self._flows.get(flow_id)
        if flow is None:
            return None

        # Check for timeout escalation
        if flow.state == ApprovalState.PENDING.value and flow.is_expired:
            flow.state = ApprovalState.ESCALATED.value
            flow.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_flow(flow)
            self._emit_event("approval_escalated", flow)
            logger.info("Approval flow %s auto-escalated due to timeout", flow_id)

        return flow

    def get_status_by_debate(self, debate_id: str) -> ApprovalFlow | None:
        """Get the approval flow for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            The ApprovalFlow, or None if not found.
        """
        flow_id = self._debate_index.get(debate_id)
        if flow_id is None:
            return None
        return self.get_status(flow_id)

    def list_pending(self, channel: str | None = None) -> list[ApprovalFlow]:
        """List all pending approval flows.

        Args:
            channel: Optional filter by channel platform.

        Returns:
            List of pending ApprovalFlow instances.
        """
        pending = []
        for flow in self._flows.values():
            if flow.state != ApprovalState.PENDING.value:
                continue
            if channel and flow.channel != channel:
                continue
            # Check for timeout while listing
            if flow.is_expired:
                flow.state = ApprovalState.ESCALATED.value
                flow.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist_flow(flow)
                self._emit_event("approval_escalated", flow)
                continue
            pending.append(flow)
        return pending

    def _persist_flow(self, flow: ApprovalFlow) -> None:
        """Persist an approval flow to the governance store."""
        if self._governance_store is None:
            return

        try:
            from aragora.storage.governance_store import ApprovalRecord

            record = ApprovalRecord(
                approval_id=flow.flow_id,
                title=f"Decision approval for debate {flow.debate_id}",
                description=f"Receipt {flow.receipt_id} requires approval",
                risk_level="medium",
                status=flow.state,
                requested_by=flow.metadata.get("requested_by", "system"),
                requested_at=datetime.fromisoformat(flow.created_at),
                changes_json=_safe_json_dumps(flow.to_dict()),
                timeout_seconds=flow.timeout_seconds,
            )

            if flow.state in (
                ApprovalState.APPROVED.value,
                ApprovalState.REJECTED.value,
            ):
                last_vote = flow.votes[-1] if flow.votes else None
                if last_vote:
                    record.approved_by = last_vote.user_id
                    record.approved_at = datetime.fromisoformat(last_vote.timestamp)
                    if flow.state == ApprovalState.REJECTED.value:
                        record.rejection_reason = last_vote.reason

            self._governance_store.save_approval(record)
        except (ImportError, TypeError, ValueError, RuntimeError, OSError) as exc:
            logger.debug("Failed to persist approval flow: %s", exc)


def _safe_json_dumps(data: Any) -> str:
    """JSON serialize with fallback for non-serializable types."""
    import json

    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return "{}"


__all__ = [
    "ApprovalDecision",
    "ApprovalFlow",
    "ApprovalFlowManager",
    "ApprovalState",
    "ApprovalVote",
]
