"""
Multi-Party Validation Workflow for Knowledge Mound Phase A3.

This module provides N-party validation quorum for disputed claims and
contradiction resolution. It enables multiple agents to vote on knowledge
items with configurable consensus strategies.

Key Components:
- ValidationRequest: Request for multi-party validation
- ValidationVote: Individual vote from a validator
- ValidationResult: Final result after quorum is reached
- MultiPartyValidator: Main workflow orchestrator

Usage:
    from aragora.knowledge.mound.ops.multi_party_validation import (
        MultiPartyValidator,
        ValidationVoteType,
        ValidationConsensusStrategy,
    )

    validator = MultiPartyValidator()
    request = await validator.create_validation_request(
        item_id="km_123",
        validators=["claude", "gpt-4", "gemini"],
        quorum=2,
    )
    await validator.submit_vote(request.request_id, "claude", vote)
    result = await validator.check_consensus(request.request_id)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationVoteType(Enum):
    """Types of validation votes."""

    ACCEPT = "accept"
    """Accept the item/resolution as valid."""

    REJECT = "reject"
    """Reject the item/resolution as invalid."""

    ABSTAIN = "abstain"
    """Abstain from voting (insufficient information)."""

    PROPOSE_ALTERNATIVE = "propose_alternative"
    """Propose an alternative resolution."""

    REQUEST_INFO = "request_info"
    """Request more information before voting."""


class ValidationConsensusStrategy(Enum):
    """Strategies for determining validation consensus."""

    UNANIMOUS = "unanimous"
    """All validators must agree."""

    MAJORITY = "majority"
    """Simple majority (>50%) must agree."""

    SUPERMAJORITY = "supermajority"
    """Two-thirds majority must agree."""

    WEIGHTED = "weighted"
    """Weighted voting based on validator reliability."""

    QUORUM = "quorum"
    """Minimum number of accept votes required."""


class ValidationState(Enum):
    """States of a validation request."""

    PENDING = "pending"
    """Waiting for votes."""

    IN_REVIEW = "in_review"
    """Votes are being collected."""

    CONSENSUS_REACHED = "consensus_reached"
    """Consensus has been reached."""

    DEADLOCKED = "deadlocked"
    """Cannot reach consensus."""

    EXPIRED = "expired"
    """Deadline passed without consensus."""

    ESCALATED = "escalated"
    """Escalated to higher authority."""

    CANCELLED = "cancelled"
    """Validation was cancelled."""


@dataclass
class ValidationVote:
    """A vote from a validator."""

    validator_id: str
    """ID of the validator casting this vote."""

    vote_type: ValidationVoteType
    """Type of vote cast."""

    confidence: float = 0.8
    """Confidence in the vote (0.0 to 1.0)."""

    reasoning: str = ""
    """Explanation for the vote."""

    alternative: Optional[str] = None
    """Alternative proposal if vote_type is PROPOSE_ALTERNATIVE."""

    weight: float = 1.0
    """Weight of this vote for weighted consensus."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the vote was cast."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional vote metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator_id": self.validator_id,
            "vote_type": self.vote_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternative": self.alternative,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ValidationRequest:
    """A request for multi-party validation."""

    request_id: str
    """Unique identifier for this request."""

    item_id: str
    """ID of the knowledge item being validated."""

    contradiction_id: Optional[str] = None
    """ID of the contradiction being resolved (if applicable)."""

    proposer_id: str = ""
    """ID of the user/agent who initiated the request."""

    validators: List[str] = field(default_factory=list)
    """List of validator IDs assigned to this request."""

    required_votes: int = 2
    """Number of votes required for quorum."""

    strategy: ValidationConsensusStrategy = ValidationConsensusStrategy.MAJORITY
    """Strategy for determining consensus."""

    deadline: Optional[datetime] = None
    """Deadline for voting (None = no deadline)."""

    votes: List[ValidationVote] = field(default_factory=list)
    """Votes received so far."""

    state: ValidationState = ValidationState.PENDING
    """Current state of the request."""

    context: Dict[str, Any] = field(default_factory=dict)
    """Additional context for validators."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the request was created."""

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the request was last updated."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional request metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "item_id": self.item_id,
            "contradiction_id": self.contradiction_id,
            "proposer_id": self.proposer_id,
            "validators": self.validators,
            "required_votes": self.required_votes,
            "strategy": self.strategy.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "votes": [v.to_dict() for v in self.votes],
            "state": self.state.value,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def votes_received(self) -> int:
        """Number of votes received."""
        return len(self.votes)

    @property
    def votes_needed(self) -> int:
        """Votes still needed for quorum."""
        return max(0, self.required_votes - self.votes_received)

    @property
    def is_complete(self) -> bool:
        """Check if validation is complete."""
        return self.state in (
            ValidationState.CONSENSUS_REACHED,
            ValidationState.DEADLOCKED,
            ValidationState.EXPIRED,
            ValidationState.ESCALATED,
            ValidationState.CANCELLED,
        )

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline


@dataclass
class ValidationResult:
    """Result of a completed validation."""

    request_id: str
    """ID of the validation request."""

    item_id: str
    """ID of the validated item."""

    outcome: str
    """Outcome: 'accepted', 'rejected', 'deadlocked', 'expired'."""

    final_verdict: ValidationVoteType
    """The winning vote type."""

    accept_count: int = 0
    """Number of accept votes."""

    reject_count: int = 0
    """Number of reject votes."""

    abstain_count: int = 0
    """Number of abstain votes."""

    consensus_strength: float = 0.0
    """Strength of consensus (0.0 to 1.0)."""

    weighted_score: float = 0.0
    """Weighted voting score."""

    alternatives_proposed: List[str] = field(default_factory=list)
    """Alternative resolutions proposed."""

    votes: List[ValidationVote] = field(default_factory=list)
    """All votes cast."""

    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the validation was resolved."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional result metadata."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "item_id": self.item_id,
            "outcome": self.outcome,
            "final_verdict": self.final_verdict.value,
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
            "abstain_count": self.abstain_count,
            "consensus_strength": self.consensus_strength,
            "weighted_score": self.weighted_score,
            "alternatives_proposed": self.alternatives_proposed,
            "votes": [v.to_dict() for v in self.votes],
            "resolved_at": self.resolved_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class EscalationResult:
    """Result of escalating a deadlocked validation."""

    request_id: str
    """ID of the escalated request."""

    escalation_id: str
    """Unique ID for this escalation."""

    escalated_to: str
    """ID of the authority escalated to."""

    reason: str
    """Reason for escalation."""

    original_votes: List[ValidationVote]
    """Votes received before escalation."""

    escalated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the escalation occurred."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "escalation_id": self.escalation_id,
            "escalated_to": self.escalated_to,
            "reason": self.reason,
            "original_votes": [v.to_dict() for v in self.original_votes],
            "escalated_at": self.escalated_at.isoformat(),
        }


@dataclass
class ValidatorConfig:
    """Configuration for the multi-party validator."""

    default_deadline_hours: int = 24
    """Default deadline in hours if not specified."""

    default_quorum: int = 2
    """Default number of required votes."""

    default_strategy: ValidationConsensusStrategy = ValidationConsensusStrategy.MAJORITY
    """Default consensus strategy."""

    supermajority_threshold: float = 0.67
    """Threshold for supermajority (default 2/3)."""

    auto_escalate_on_deadlock: bool = True
    """Whether to auto-escalate when deadlocked."""

    escalation_authority: str = "admin"
    """Default escalation target."""

    max_alternatives: int = 5
    """Maximum alternative proposals allowed."""

    allow_vote_change: bool = False
    """Whether validators can change their votes."""


# Type alias for notification callbacks
NotificationCallback = Callable[[str, str, Dict[str, Any]], None]


class MultiPartyValidator:
    """Orchestrates multi-party validation workflows.

    This class manages the full lifecycle of validation requests:
    creation, vote collection, consensus checking, and resolution.
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        """Initialize the multi-party validator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ValidatorConfig()
        self._requests: Dict[str, ValidationRequest] = {}
        self._results: Dict[str, ValidationResult] = {}
        self._escalations: Dict[str, EscalationResult] = {}
        self._notification_callback: Optional[NotificationCallback] = None

    def set_notification_callback(self, callback: NotificationCallback) -> None:
        """Set callback for sending notifications.

        Args:
            callback: Function(validator_id, event_type, data) for notifications.
        """
        self._notification_callback = callback

    async def create_validation_request(
        self,
        item_id: str,
        validators: List[str],
        quorum: Optional[int] = None,
        strategy: Optional[ValidationConsensusStrategy] = None,
        deadline_hours: Optional[int] = None,
        contradiction_id: Optional[str] = None,
        proposer_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationRequest:
        """Create a new validation request.

        Args:
            item_id: ID of the item to validate.
            validators: List of validator IDs to request votes from.
            quorum: Number of votes required (default from config).
            strategy: Consensus strategy (default from config).
            deadline_hours: Hours until deadline (default from config).
            contradiction_id: Optional contradiction being resolved.
            proposer_id: ID of the proposer.
            context: Additional context for validators.

        Returns:
            The created ValidationRequest.
        """
        request_id = f"val_{uuid.uuid4().hex[:12]}"

        quorum = quorum or self.config.default_quorum
        strategy = strategy or self.config.default_strategy
        deadline_hours = deadline_hours or self.config.default_deadline_hours

        deadline = datetime.now(timezone.utc) + timedelta(hours=deadline_hours)

        request = ValidationRequest(
            request_id=request_id,
            item_id=item_id,
            contradiction_id=contradiction_id,
            proposer_id=proposer_id,
            validators=validators,
            required_votes=min(quorum, len(validators)),
            strategy=strategy,
            deadline=deadline,
            state=ValidationState.PENDING,
            context=context or {},
        )

        self._requests[request_id] = request

        # Notify validators
        await self._notify_validators(request, "validation_requested")

        logger.info(
            f"Created validation request {request_id} for item {item_id} "
            f"with {len(validators)} validators"
        )

        return request

    async def submit_vote(
        self,
        request_id: str,
        validator_id: str,
        vote_type: ValidationVoteType,
        confidence: float = 0.8,
        reasoning: str = "",
        alternative: Optional[str] = None,
        weight: float = 1.0,
    ) -> bool:
        """Submit a vote for a validation request.

        Args:
            request_id: ID of the validation request.
            validator_id: ID of the validator casting the vote.
            vote_type: Type of vote being cast.
            confidence: Confidence in the vote (0.0 to 1.0).
            reasoning: Explanation for the vote.
            alternative: Alternative proposal (for PROPOSE_ALTERNATIVE).
            weight: Weight of this vote (for weighted consensus).

        Returns:
            True if the vote was accepted.
        """
        request = self._requests.get(request_id)
        if not request:
            logger.warning(f"Validation request {request_id} not found")
            return False

        if request.is_complete:
            logger.warning(f"Validation request {request_id} is already complete")
            return False

        if request.is_expired:
            await self._handle_expiration(request)
            return False

        if validator_id not in request.validators:
            logger.warning(f"Validator {validator_id} not authorized for request {request_id}")
            return False

        # Check if already voted
        existing_vote = next((v for v in request.votes if v.validator_id == validator_id), None)
        if existing_vote:
            if not self.config.allow_vote_change:
                logger.warning(f"Validator {validator_id} already voted on request {request_id}")
                return False
            # Remove existing vote to replace
            request.votes = [v for v in request.votes if v.validator_id != validator_id]

        # Create and add vote
        vote = ValidationVote(
            validator_id=validator_id,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            alternative=alternative
            if vote_type == ValidationVoteType.PROPOSE_ALTERNATIVE
            else None,
            weight=weight,
        )

        request.votes.append(vote)
        request.updated_at = datetime.now(timezone.utc)

        if request.state == ValidationState.PENDING:
            request.state = ValidationState.IN_REVIEW

        logger.info(f"Vote {vote_type.value} submitted by {validator_id} for request {request_id}")

        # Check if consensus reached
        result = await self.check_consensus(request_id)
        if result:
            await self._notify_proposer(request, "validation_complete", result)

        return True

    async def check_consensus(self, request_id: str) -> Optional[ValidationResult]:
        """Check if consensus has been reached.

        Args:
            request_id: ID of the validation request.

        Returns:
            ValidationResult if consensus reached, None otherwise.
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        if request.request_id in self._results:
            return self._results[request.request_id]

        if request.is_expired:
            return await self._handle_expiration(request)

        # Count votes by type
        vote_counts = self._count_votes(request.votes)

        # Check if quorum reached
        total_decisive = vote_counts["accept"] + vote_counts["reject"]
        if total_decisive < request.required_votes:
            return None  # Not enough votes yet

        # Determine consensus based on strategy
        result = self._evaluate_consensus(request, vote_counts)

        if result:
            self._results[request.request_id] = result
            request.state = ValidationState.CONSENSUS_REACHED
            request.updated_at = datetime.now(timezone.utc)

        return result

    def _count_votes(self, votes: List[ValidationVote]) -> Dict[str, Any]:
        """Count votes by type.

        Args:
            votes: List of votes.

        Returns:
            Dict with vote counts and weighted scores.
        """
        counts = {
            "accept": 0,
            "reject": 0,
            "abstain": 0,
            "propose_alternative": 0,
            "request_info": 0,
            "weighted_accept": 0.0,
            "weighted_reject": 0.0,
            "total_weight": 0.0,
        }

        for vote in votes:
            if vote.vote_type == ValidationVoteType.ACCEPT:
                counts["accept"] += 1
                counts["weighted_accept"] += vote.weight * vote.confidence
            elif vote.vote_type == ValidationVoteType.REJECT:
                counts["reject"] += 1
                counts["weighted_reject"] += vote.weight * vote.confidence
            elif vote.vote_type == ValidationVoteType.ABSTAIN:
                counts["abstain"] += 1
            elif vote.vote_type == ValidationVoteType.PROPOSE_ALTERNATIVE:
                counts["propose_alternative"] += 1
            elif vote.vote_type == ValidationVoteType.REQUEST_INFO:
                counts["request_info"] += 1

            counts["total_weight"] += vote.weight

        return counts

    def _evaluate_consensus(
        self,
        request: ValidationRequest,
        vote_counts: Dict[str, Any],
    ) -> Optional[ValidationResult]:
        """Evaluate if consensus is reached based on strategy.

        Args:
            request: The validation request.
            vote_counts: Vote counts from _count_votes.

        Returns:
            ValidationResult if consensus reached, None otherwise.
        """
        accept = vote_counts["accept"]
        reject = vote_counts["reject"]
        total = accept + reject

        if total == 0:
            return None

        strategy = request.strategy

        # Determine if consensus reached based on strategy
        consensus_reached = False
        final_verdict = ValidationVoteType.ACCEPT if accept > reject else ValidationVoteType.REJECT

        if strategy == ValidationConsensusStrategy.UNANIMOUS:
            consensus_reached = (accept == total) or (reject == total)

        elif strategy == ValidationConsensusStrategy.MAJORITY:
            consensus_reached = accept > total / 2 or reject > total / 2

        elif strategy == ValidationConsensusStrategy.SUPERMAJORITY:
            threshold = self.config.supermajority_threshold
            consensus_reached = accept / total >= threshold or reject / total >= threshold

        elif strategy == ValidationConsensusStrategy.WEIGHTED:
            wa = vote_counts["weighted_accept"]
            wr = vote_counts["weighted_reject"]
            tw = vote_counts["total_weight"]
            if tw > 0:
                consensus_reached = (wa / tw >= 0.5) or (wr / tw >= 0.5)
                final_verdict = ValidationVoteType.ACCEPT if wa > wr else ValidationVoteType.REJECT

        elif strategy == ValidationConsensusStrategy.QUORUM:
            consensus_reached = accept >= request.required_votes

        if not consensus_reached:
            # Check if all validators have voted (deadlock)
            if len(request.votes) >= len(request.validators):
                return self._create_deadlock_result(request, vote_counts)
            return None

        # Calculate consensus strength
        consensus_strength = max(accept, reject) / total if total > 0 else 0.0

        # Collect alternatives
        alternatives = [
            v.alternative
            for v in request.votes
            if v.alternative and v.vote_type == ValidationVoteType.PROPOSE_ALTERNATIVE
        ]

        return ValidationResult(
            request_id=request.request_id,
            item_id=request.item_id,
            outcome="accepted" if final_verdict == ValidationVoteType.ACCEPT else "rejected",
            final_verdict=final_verdict,
            accept_count=accept,
            reject_count=reject,
            abstain_count=vote_counts["abstain"],
            consensus_strength=consensus_strength,
            weighted_score=(
                vote_counts["weighted_accept"]
                if final_verdict == ValidationVoteType.ACCEPT
                else vote_counts["weighted_reject"]
            ),
            alternatives_proposed=alternatives[: self.config.max_alternatives],
            votes=list(request.votes),
        )

    def _create_deadlock_result(
        self,
        request: ValidationRequest,
        vote_counts: Dict[str, Any],
    ) -> ValidationResult:
        """Create result for a deadlocked validation.

        Args:
            request: The validation request.
            vote_counts: Vote counts.

        Returns:
            ValidationResult indicating deadlock.
        """
        request.state = ValidationState.DEADLOCKED

        # If auto-escalate enabled, escalate
        if self.config.auto_escalate_on_deadlock:
            # Create escalation (fire and forget)
            self._create_escalation(
                request,
                reason=f"Deadlock: {vote_counts['accept']} accept vs {vote_counts['reject']} reject",
            )

        return ValidationResult(
            request_id=request.request_id,
            item_id=request.item_id,
            outcome="deadlocked",
            final_verdict=ValidationVoteType.ABSTAIN,
            accept_count=vote_counts["accept"],
            reject_count=vote_counts["reject"],
            abstain_count=vote_counts["abstain"],
            consensus_strength=0.0,
            votes=list(request.votes),
            metadata={"deadlock_reason": "No consensus reached"},
        )

    async def _handle_expiration(self, request: ValidationRequest) -> ValidationResult:
        """Handle an expired validation request.

        Args:
            request: The expired request.

        Returns:
            ValidationResult indicating expiration.
        """
        request.state = ValidationState.EXPIRED
        request.updated_at = datetime.now(timezone.utc)

        vote_counts = self._count_votes(request.votes)

        result = ValidationResult(
            request_id=request.request_id,
            item_id=request.item_id,
            outcome="expired",
            final_verdict=ValidationVoteType.ABSTAIN,
            accept_count=vote_counts["accept"],
            reject_count=vote_counts["reject"],
            abstain_count=vote_counts["abstain"],
            votes=list(request.votes),
            metadata={
                "deadline": request.deadline.isoformat() if request.deadline else None,
                "votes_received": len(request.votes),
                "votes_needed": request.required_votes,
            },
        )

        self._results[request.request_id] = result
        return result

    def _create_escalation(
        self,
        request: ValidationRequest,
        reason: str,
        escalate_to: Optional[str] = None,
    ) -> EscalationResult:
        """Create an escalation for a request.

        Args:
            request: The request to escalate.
            reason: Reason for escalation.
            escalate_to: Target authority (default from config).

        Returns:
            EscalationResult for the escalation.
        """
        escalation = EscalationResult(
            request_id=request.request_id,
            escalation_id=f"esc_{uuid.uuid4().hex[:12]}",
            escalated_to=escalate_to or self.config.escalation_authority,
            reason=reason,
            original_votes=list(request.votes),
        )

        request.state = ValidationState.ESCALATED
        self._escalations[request.request_id] = escalation

        logger.info(
            f"Escalated request {request.request_id} to {escalation.escalated_to}: {reason}"
        )

        return escalation

    async def escalate_deadlock(
        self,
        request_id: str,
        reason: Optional[str] = None,
        escalate_to: Optional[str] = None,
    ) -> Optional[EscalationResult]:
        """Manually escalate a deadlocked validation.

        Args:
            request_id: ID of the request to escalate.
            reason: Reason for escalation.
            escalate_to: Target authority for escalation.

        Returns:
            EscalationResult if successful, None otherwise.
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        if request.state not in (ValidationState.DEADLOCKED, ValidationState.IN_REVIEW):
            logger.warning(f"Cannot escalate request {request_id} in state {request.state}")
            return None

        return self._create_escalation(
            request,
            reason=reason or "Manual escalation requested",
            escalate_to=escalate_to,
        )

    async def resolve_escalation(
        self,
        request_id: str,
        resolver_id: str,
        verdict: ValidationVoteType,
        reasoning: str = "",
    ) -> Optional[ValidationResult]:
        """Resolve an escalated validation.

        Args:
            request_id: ID of the escalated request.
            resolver_id: ID of the resolver.
            verdict: Final verdict.
            reasoning: Explanation for the resolution.

        Returns:
            ValidationResult if successful, None otherwise.
        """
        request = self._requests.get(request_id)
        if not request or request.state != ValidationState.ESCALATED:
            return None

        escalation = self._escalations.get(request_id)

        result = ValidationResult(
            request_id=request_id,
            item_id=request.item_id,
            outcome="resolved_by_escalation",
            final_verdict=verdict,
            votes=list(request.votes),
            metadata={
                "resolved_by": resolver_id,
                "reasoning": reasoning,
                "escalation_id": escalation.escalation_id if escalation else None,
            },
        )

        request.state = ValidationState.CONSENSUS_REACHED
        self._results[request_id] = result

        return result

    async def cancel_request(self, request_id: str, reason: str = "") -> bool:
        """Cancel a validation request.

        Args:
            request_id: ID of the request to cancel.
            reason: Reason for cancellation.

        Returns:
            True if cancelled successfully.
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.is_complete:
            return False

        request.state = ValidationState.CANCELLED
        request.metadata["cancellation_reason"] = reason
        request.updated_at = datetime.now(timezone.utc)

        logger.info(f"Cancelled validation request {request_id}: {reason}")
        return True

    async def _notify_validators(
        self,
        request: ValidationRequest,
        event_type: str,
    ) -> None:
        """Notify validators of an event.

        Args:
            request: The validation request.
            event_type: Type of event to notify.
        """
        if not self._notification_callback:
            return

        for validator_id in request.validators:
            try:
                self._notification_callback(
                    validator_id,
                    event_type,
                    {
                        "request_id": request.request_id,
                        "item_id": request.item_id,
                        "deadline": request.deadline.isoformat() if request.deadline else None,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to notify {validator_id}: {e}")

    async def _notify_proposer(
        self,
        request: ValidationRequest,
        event_type: str,
        result: Optional[ValidationResult] = None,
    ) -> None:
        """Notify the proposer of an event.

        Args:
            request: The validation request.
            event_type: Type of event to notify.
            result: Optional validation result.
        """
        if not self._notification_callback or not request.proposer_id:
            return

        try:
            self._notification_callback(
                request.proposer_id,
                event_type,
                {
                    "request_id": request.request_id,
                    "item_id": request.item_id,
                    "result": result.to_dict() if result else None,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to notify proposer {request.proposer_id}: {e}")

    def get_request(self, request_id: str) -> Optional[ValidationRequest]:
        """Get a validation request by ID.

        Args:
            request_id: ID of the request.

        Returns:
            ValidationRequest if found, None otherwise.
        """
        return self._requests.get(request_id)

    def get_result(self, request_id: str) -> Optional[ValidationResult]:
        """Get a validation result by request ID.

        Args:
            request_id: ID of the request.

        Returns:
            ValidationResult if found, None otherwise.
        """
        return self._results.get(request_id)

    def get_pending_for_validator(self, validator_id: str) -> List[ValidationRequest]:
        """Get pending validation requests for a validator.

        Args:
            validator_id: ID of the validator.

        Returns:
            List of pending requests assigned to the validator.
        """
        return [
            r
            for r in self._requests.values()
            if validator_id in r.validators
            and not r.is_complete
            and not any(v.validator_id == validator_id for v in r.votes)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics.

        Returns:
            Dict with validation metrics.
        """
        total_requests = len(self._requests)
        completed = len(self._results)
        escalated = sum(1 for r in self._requests.values() if r.state == ValidationState.ESCALATED)
        pending = sum(
            1
            for r in self._requests.values()
            if r.state in (ValidationState.PENDING, ValidationState.IN_REVIEW)
        )

        # Calculate average consensus strength
        avg_strength = 0.0
        if self._results:
            avg_strength = sum(r.consensus_strength for r in self._results.values()) / len(
                self._results
            )

        return {
            "total_requests": total_requests,
            "completed": completed,
            "pending": pending,
            "escalated": escalated,
            "completion_rate": completed / total_requests if total_requests > 0 else 0.0,
            "escalation_rate": escalated / total_requests if total_requests > 0 else 0.0,
            "avg_consensus_strength": avg_strength,
            "by_outcome": self._stats_by_outcome(),
        }

    def _stats_by_outcome(self) -> Dict[str, int]:
        """Get counts by outcome."""
        counts: Dict[str, int] = {}
        for r in self._results.values():
            outcome = r.outcome
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts


# Singleton instance
_multi_party_validator: Optional[MultiPartyValidator] = None


def get_multi_party_validator(
    config: Optional[ValidatorConfig] = None,
) -> MultiPartyValidator:
    """Get or create the singleton multi-party validator.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        MultiPartyValidator instance.
    """
    global _multi_party_validator
    if _multi_party_validator is None:
        _multi_party_validator = MultiPartyValidator(config)
    return _multi_party_validator


__all__ = [
    # Enums
    "ValidationVoteType",
    "ValidationConsensusStrategy",
    "ValidationState",
    # Dataclasses
    "ValidationVote",
    "ValidationRequest",
    "ValidationResult",
    "EscalationResult",
    "ValidatorConfig",
    # Validator
    "MultiPartyValidator",
    "get_multi_party_validator",
]
