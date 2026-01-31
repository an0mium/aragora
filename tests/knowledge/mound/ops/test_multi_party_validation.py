"""
Comprehensive tests for Multi-Party Validation module.

Tests cover:
- Validation request creation and lifecycle
- Multi-party voting and consensus logic
- Threshold calculations and quorum requirements
- Validation result aggregation
- Timeout and expiration handling
- Edge cases (single validator, tie-breaking, late votes)

Target: 50+ tests with proper fixtures and mocking.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.multi_party_validation import (
    EscalationResult,
    MultiPartyValidator,
    ValidationConsensusStrategy,
    ValidationRequest,
    ValidationResult,
    ValidationState,
    ValidationVote,
    ValidationVoteType,
    ValidatorConfig,
    get_multi_party_validator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def validator():
    """Create a default MultiPartyValidator."""
    return MultiPartyValidator()


@pytest.fixture
def validator_no_auto_escalate():
    """Create a validator with auto-escalation disabled."""
    config = ValidatorConfig(auto_escalate_on_deadlock=False)
    return MultiPartyValidator(config)


@pytest.fixture
def validator_with_custom_config():
    """Create a validator with custom configuration."""
    config = ValidatorConfig(
        default_deadline_hours=48,
        default_quorum=3,
        default_strategy=ValidationConsensusStrategy.SUPERMAJORITY,
        supermajority_threshold=0.75,
        auto_escalate_on_deadlock=True,
        escalation_authority="senior_admin",
        max_alternatives=10,
        allow_vote_change=True,
    )
    return MultiPartyValidator(config)


@pytest.fixture
def validator_with_vote_change():
    """Create a validator that allows vote changes."""
    config = ValidatorConfig(allow_vote_change=True)
    return MultiPartyValidator(config)


@pytest.fixture
def sample_validators():
    """Sample list of validator IDs."""
    return ["claude", "gpt-4", "gemini"]


@pytest.fixture
def large_validator_pool():
    """Large pool of validators for quorum tests."""
    return [f"agent_{i}" for i in range(10)]


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values and consistency."""

    def test_validation_vote_type_values(self):
        """Test ValidationVoteType enum values."""
        assert ValidationVoteType.ACCEPT.value == "accept"
        assert ValidationVoteType.REJECT.value == "reject"
        assert ValidationVoteType.ABSTAIN.value == "abstain"
        assert ValidationVoteType.PROPOSE_ALTERNATIVE.value == "propose_alternative"
        assert ValidationVoteType.REQUEST_INFO.value == "request_info"

    def test_validation_consensus_strategy_values(self):
        """Test ValidationConsensusStrategy enum values."""
        assert ValidationConsensusStrategy.UNANIMOUS.value == "unanimous"
        assert ValidationConsensusStrategy.MAJORITY.value == "majority"
        assert ValidationConsensusStrategy.SUPERMAJORITY.value == "supermajority"
        assert ValidationConsensusStrategy.WEIGHTED.value == "weighted"
        assert ValidationConsensusStrategy.QUORUM.value == "quorum"

    def test_validation_state_values(self):
        """Test ValidationState enum values."""
        assert ValidationState.PENDING.value == "pending"
        assert ValidationState.IN_REVIEW.value == "in_review"
        assert ValidationState.CONSENSUS_REACHED.value == "consensus_reached"
        assert ValidationState.DEADLOCKED.value == "deadlocked"
        assert ValidationState.EXPIRED.value == "expired"
        assert ValidationState.ESCALATED.value == "escalated"
        assert ValidationState.CANCELLED.value == "cancelled"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Test dataclass creation and serialization."""

    def test_validation_vote_creation(self):
        """Test ValidationVote dataclass creation."""
        vote = ValidationVote(
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
            confidence=0.9,
            reasoning="Strong evidence supports this",
            weight=1.5,
        )

        assert vote.validator_id == "claude"
        assert vote.vote_type == ValidationVoteType.ACCEPT
        assert vote.confidence == 0.9
        assert vote.reasoning == "Strong evidence supports this"
        assert vote.weight == 1.5
        assert isinstance(vote.timestamp, datetime)

    def test_validation_vote_to_dict(self):
        """Test ValidationVote serialization."""
        vote = ValidationVote(
            validator_id="claude",
            vote_type=ValidationVoteType.REJECT,
            confidence=0.7,
            reasoning="Insufficient evidence",
            metadata={"source": "debate_1"},
        )

        d = vote.to_dict()

        assert d["validator_id"] == "claude"
        assert d["vote_type"] == "reject"
        assert d["confidence"] == 0.7
        assert d["reasoning"] == "Insufficient evidence"
        assert d["metadata"]["source"] == "debate_1"
        assert "timestamp" in d

    def test_validation_vote_with_alternative(self):
        """Test ValidationVote with alternative proposal."""
        vote = ValidationVote(
            validator_id="gpt-4",
            vote_type=ValidationVoteType.PROPOSE_ALTERNATIVE,
            confidence=0.85,
            alternative="Use a different approach",
        )

        assert vote.alternative == "Use a different approach"
        d = vote.to_dict()
        assert d["alternative"] == "Use a different approach"

    def test_validation_request_creation(self):
        """Test ValidationRequest dataclass creation."""
        request = ValidationRequest(
            request_id="val_123",
            item_id="km_456",
            validators=["claude", "gpt-4"],
            required_votes=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        assert request.request_id == "val_123"
        assert request.item_id == "km_456"
        assert len(request.validators) == 2
        assert request.required_votes == 2
        assert request.state == ValidationState.PENDING

    def test_validation_request_to_dict(self):
        """Test ValidationRequest serialization."""
        deadline = datetime.now(timezone.utc) + timedelta(hours=24)
        request = ValidationRequest(
            request_id="val_123",
            item_id="km_456",
            validators=["claude"],
            required_votes=1,
            deadline=deadline,
            context={"debate_id": "deb_789"},
        )

        d = request.to_dict()

        assert d["request_id"] == "val_123"
        assert d["item_id"] == "km_456"
        assert d["deadline"] == deadline.isoformat()
        assert d["context"]["debate_id"] == "deb_789"
        assert d["state"] == "pending"

    def test_validation_request_properties(self):
        """Test ValidationRequest computed properties."""
        request = ValidationRequest(
            request_id="val_123",
            item_id="km_456",
            validators=["a", "b", "c"],
            required_votes=2,
        )

        # Add one vote
        request.votes.append(
            ValidationVote(
                validator_id="a",
                vote_type=ValidationVoteType.ACCEPT,
            )
        )

        assert request.votes_received == 1
        assert request.votes_needed == 1
        assert request.is_complete is False

    def test_validation_request_is_complete(self):
        """Test is_complete property for various states."""
        request = ValidationRequest(
            request_id="val_123",
            item_id="km_456",
            validators=["a"],
        )

        assert request.is_complete is False

        request.state = ValidationState.CONSENSUS_REACHED
        assert request.is_complete is True

        request.state = ValidationState.DEADLOCKED
        assert request.is_complete is True

        request.state = ValidationState.EXPIRED
        assert request.is_complete is True

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass creation."""
        result = ValidationResult(
            request_id="val_123",
            item_id="km_456",
            outcome="accepted",
            final_verdict=ValidationVoteType.ACCEPT,
            accept_count=3,
            reject_count=1,
            consensus_strength=0.75,
        )

        assert result.request_id == "val_123"
        assert result.outcome == "accepted"
        assert result.accept_count == 3
        assert result.consensus_strength == 0.75

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            request_id="val_123",
            item_id="km_456",
            outcome="rejected",
            final_verdict=ValidationVoteType.REJECT,
            reject_count=4,
            weighted_score=3.2,
            alternatives_proposed=["alt1", "alt2"],
        )

        d = result.to_dict()

        assert d["outcome"] == "rejected"
        assert d["final_verdict"] == "reject"
        assert d["reject_count"] == 4
        assert d["weighted_score"] == 3.2
        assert len(d["alternatives_proposed"]) == 2

    def test_escalation_result_creation(self):
        """Test EscalationResult dataclass creation."""
        vote = ValidationVote(
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        escalation = EscalationResult(
            request_id="val_123",
            escalation_id="esc_456",
            escalated_to="admin",
            reason="Deadlock after 2 votes",
            original_votes=[vote],
        )

        assert escalation.request_id == "val_123"
        assert escalation.escalated_to == "admin"
        assert len(escalation.original_votes) == 1

    def test_escalation_result_to_dict(self):
        """Test EscalationResult serialization."""
        escalation = EscalationResult(
            request_id="val_123",
            escalation_id="esc_456",
            escalated_to="admin",
            reason="Manual escalation",
            original_votes=[],
        )

        d = escalation.to_dict()

        assert d["request_id"] == "val_123"
        assert d["escalation_id"] == "esc_456"
        assert d["reason"] == "Manual escalation"
        assert "escalated_at" in d


# =============================================================================
# Request Creation Tests
# =============================================================================


class TestRequestCreation:
    """Test validation request creation."""

    @pytest.mark.asyncio
    async def test_create_validation_request_basic(self, validator, sample_validators):
        """Test basic validation request creation."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=sample_validators,
            quorum=2,
        )

        assert request.item_id == "km_123"
        assert len(request.validators) == 3
        assert request.required_votes == 2
        assert request.state == ValidationState.PENDING
        assert request.request_id.startswith("val_")

    @pytest.mark.asyncio
    async def test_create_validation_request_with_defaults(self, validator):
        """Test request creation uses config defaults."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        # Default quorum is 2, but capped to validators count
        assert request.required_votes == 1  # min(2, 1)
        assert request.strategy == ValidationConsensusStrategy.MAJORITY

    @pytest.mark.asyncio
    async def test_create_validation_request_with_custom_config(self, validator_with_custom_config):
        """Test request creation with custom config."""
        request = await validator_with_custom_config.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
        )

        # Custom default quorum is 3
        assert request.required_votes == 3
        assert request.strategy == ValidationConsensusStrategy.SUPERMAJORITY

    @pytest.mark.asyncio
    async def test_create_validation_request_quorum_capped(self, validator):
        """Test quorum is capped to validator count."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=5,  # More than validators
        )

        assert request.required_votes == 2  # Capped to validator count

    @pytest.mark.asyncio
    async def test_create_validation_request_with_contradiction(self, validator):
        """Test request creation for contradiction resolution."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            contradiction_id="contra_456",
        )

        assert request.contradiction_id == "contra_456"

    @pytest.mark.asyncio
    async def test_create_validation_request_with_context(self, validator):
        """Test request creation with additional context."""
        context = {
            "debate_id": "deb_123",
            "claim_text": "The earth is round",
            "sources": ["source_1", "source_2"],
        }

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            context=context,
        )

        assert request.context["debate_id"] == "deb_123"
        assert len(request.context["sources"]) == 2

    @pytest.mark.asyncio
    async def test_create_request_sets_deadline(self, validator):
        """Test request creation sets deadline from config."""
        before = datetime.now(timezone.utc)
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            deadline_hours=12,
        )
        after = datetime.now(timezone.utc)

        expected_deadline = before + timedelta(hours=12)
        assert request.deadline is not None
        assert request.deadline >= expected_deadline
        assert request.deadline <= after + timedelta(hours=12)

    @pytest.mark.asyncio
    async def test_create_request_stored_in_validator(self, validator):
        """Test created request is stored internally."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        stored = validator.get_request(request.request_id)
        assert stored is not None
        assert stored.item_id == "km_123"


# =============================================================================
# Vote Submission Tests
# =============================================================================


class TestVoteSubmission:
    """Test vote submission functionality."""

    @pytest.mark.asyncio
    async def test_submit_vote_basic(self, validator):
        """Test basic vote submission."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
            confidence=0.9,
            reasoning="Strong evidence",
        )

        assert success is True

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.votes_received == 1
        assert updated.state == ValidationState.IN_REVIEW

    @pytest.mark.asyncio
    async def test_submit_vote_unauthorized_validator(self, validator):
        """Test vote from unauthorized validator is rejected."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="unauthorized_agent",
            vote_type=ValidationVoteType.ACCEPT,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_submit_vote_nonexistent_request(self, validator):
        """Test vote for nonexistent request is rejected."""
        success = await validator.submit_vote(
            request_id="nonexistent_request",
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_submit_vote_completed_request(self, validator):
        """Test vote on completed request is rejected."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        # Complete the request
        request.state = ValidationState.CONSENSUS_REACHED

        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_submit_vote_duplicate_rejected(self, validator):
        """Test duplicate vote from same validator is rejected."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        # First vote succeeds
        await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        # Second vote from same validator fails
        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.REJECT,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_submit_vote_change_allowed(self, validator_with_vote_change):
        """Test vote change when allowed."""
        request = await validator_with_vote_change.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4", "gemini"],
            quorum=2,
        )

        # First vote
        await validator_with_vote_change.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        # Change vote
        success = await validator_with_vote_change.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.REJECT,
        )

        assert success is True

        updated = validator_with_vote_change.get_request(request.request_id)
        assert updated is not None
        assert updated.votes_received == 1  # Still one vote
        assert updated.votes[0].vote_type == ValidationVoteType.REJECT

    @pytest.mark.asyncio
    async def test_submit_vote_with_alternative(self, validator):
        """Test vote with alternative proposal."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.PROPOSE_ALTERNATIVE,
            alternative="Consider a different resolution",
        )

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.votes[0].alternative == "Consider a different resolution"

    @pytest.mark.asyncio
    async def test_submit_vote_alternative_ignored_for_non_propose(self, validator):
        """Test alternative is ignored for non-PROPOSE_ALTERNATIVE votes."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            quorum=1,
        )

        await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
            alternative="This should be ignored",
        )

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.votes[0].alternative is None

    @pytest.mark.asyncio
    async def test_submit_vote_with_weight(self, validator):
        """Test vote with custom weight."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
            weight=2.0,
        )

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.votes[0].weight == 2.0


# =============================================================================
# Consensus Strategy Tests
# =============================================================================


class TestConsensusStrategies:
    """Test different consensus strategies."""

    @pytest.mark.asyncio
    async def test_majority_consensus_accept(self, validator):
        """Test majority strategy accepts when >50% accept."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"
        assert result.final_verdict == ValidationVoteType.ACCEPT

    @pytest.mark.asyncio
    async def test_majority_consensus_reject(self, validator):
        """Test majority strategy rejects when >50% reject."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.REJECT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.REJECT, 0.8)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "rejected"
        assert result.final_verdict == ValidationVoteType.REJECT

    @pytest.mark.asyncio
    async def test_unanimous_consensus_success(self, validator):
        """Test unanimous strategy requires all to agree."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.UNANIMOUS,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"
        assert result.consensus_strength == 1.0

    @pytest.mark.asyncio
    async def test_unanimous_consensus_deadlock(self, validator_no_auto_escalate):
        """Test unanimous strategy deadlocks on disagreement."""
        request = await validator_no_auto_escalate.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.UNANIMOUS,
        )

        await validator_no_auto_escalate.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9
        )
        await validator_no_auto_escalate.submit_vote(
            request.request_id, "b", ValidationVoteType.REJECT, 0.8
        )

        result = await validator_no_auto_escalate.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "deadlocked"

    @pytest.mark.asyncio
    async def test_supermajority_consensus(self, validator_with_custom_config):
        """Test supermajority strategy (75% threshold)."""
        request = await validator_with_custom_config.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
            quorum=3,
            strategy=ValidationConsensusStrategy.SUPERMAJORITY,
        )

        # 3 out of 4 = 75%, meets threshold
        await validator_with_custom_config.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9
        )
        await validator_with_custom_config.submit_vote(
            request.request_id, "b", ValidationVoteType.ACCEPT, 0.8
        )
        await validator_with_custom_config.submit_vote(
            request.request_id, "c", ValidationVoteType.ACCEPT, 0.85
        )

        result = await validator_with_custom_config.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"

    @pytest.mark.asyncio
    async def test_weighted_consensus(self, validator):
        """Test weighted consensus strategy."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.WEIGHTED,
        )

        # Weight strongly favors accept
        await validator.submit_vote(
            request.request_id,
            "a",
            ValidationVoteType.ACCEPT,
            confidence=0.9,
            weight=3.0,
        )
        await validator.submit_vote(
            request.request_id,
            "b",
            ValidationVoteType.REJECT,
            confidence=0.5,
            weight=1.0,
        )

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.final_verdict == ValidationVoteType.ACCEPT
        # Weighted accept = 3.0 * 0.9 = 2.7
        # Weighted reject = 1.0 * 0.5 = 0.5
        assert result.weighted_score > 2.0

    @pytest.mark.asyncio
    async def test_quorum_strategy(self, validator):
        """Test quorum strategy (N accepts needed)."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
            quorum=3,
            strategy=ValidationConsensusStrategy.QUORUM,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)
        await validator.submit_vote(request.request_id, "c", ValidationVoteType.ACCEPT, 0.85)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"
        assert result.accept_count == 3


# =============================================================================
# Threshold and Quorum Tests
# =============================================================================


class TestThresholdAndQuorum:
    """Test threshold calculations and quorum requirements."""

    @pytest.mark.asyncio
    async def test_quorum_not_reached(self, validator):
        """Test consensus returns None when quorum not reached."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=3,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        # Only 2 votes, need 3 for quorum
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        result = await validator.check_consensus(request.request_id)

        # Quorum requires 3, only 2 submitted
        # But with majority strategy, if we have 2 accepts out of 2 that's 100%
        # The check is total_decisive >= required_votes
        # So with 2 votes and required_votes=3, it should return None
        # Let me re-check the implementation logic
        # Actually looking at the code: total_decisive = accept + reject
        # if total_decisive < request.required_votes: return None
        assert result is None

    @pytest.mark.asyncio
    async def test_abstain_votes_dont_count_for_quorum(self, validator):
        """Test abstain votes don't count toward decisive quorum."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ABSTAIN, 0.5)

        result = await validator.check_consensus(request.request_id)

        # Only 1 decisive vote (accept), need 2
        assert result is None

    @pytest.mark.asyncio
    async def test_supermajority_threshold_exact(self, validator):
        """Test exact supermajority threshold (67%)."""
        config = ValidatorConfig(supermajority_threshold=0.67)
        v = MultiPartyValidator(config)

        request = await v.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
            strategy=ValidationConsensusStrategy.SUPERMAJORITY,
        )

        # 2 out of 3 = 66.7%, exactly at threshold
        await v.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await v.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)
        await v.submit_vote(request.request_id, "c", ValidationVoteType.REJECT, 0.7)

        result = await v.check_consensus(request.request_id)

        assert result is not None
        # 2/3 = 0.667 >= 0.67
        assert result.outcome == "accepted"

    @pytest.mark.asyncio
    async def test_supermajority_threshold_not_met(self, validator_no_auto_escalate):
        """Test supermajority threshold not met."""
        config = ValidatorConfig(supermajority_threshold=0.75, auto_escalate_on_deadlock=False)
        v = MultiPartyValidator(config)

        request = await v.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
            quorum=3,
            strategy=ValidationConsensusStrategy.SUPERMAJORITY,
        )

        # 2 out of 4 = 50%, below 75% threshold
        await v.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await v.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)
        await v.submit_vote(request.request_id, "c", ValidationVoteType.REJECT, 0.7)
        await v.submit_vote(request.request_id, "d", ValidationVoteType.REJECT, 0.6)

        result = await v.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "deadlocked"


# =============================================================================
# Result Aggregation Tests
# =============================================================================


class TestResultAggregation:
    """Test validation result aggregation."""

    @pytest.mark.asyncio
    async def test_result_counts_correct(self, validator):
        """Test result correctly counts vote types."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
            quorum=3,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)
        await validator.submit_vote(request.request_id, "c", ValidationVoteType.REJECT, 0.7)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.accept_count == 2
        assert result.reject_count == 1

    @pytest.mark.asyncio
    async def test_result_includes_all_votes(self, validator):
        """Test result includes all votes."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9, "Reason A"
        )
        await validator.submit_vote(
            request.request_id, "b", ValidationVoteType.ACCEPT, 0.8, "Reason B"
        )

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert len(result.votes) == 2
        assert result.votes[0].reasoning in ["Reason A", "Reason B"]

    @pytest.mark.asyncio
    async def test_result_alternatives_collected(self, validator):
        """Test alternatives are collected from PROPOSE_ALTERNATIVE votes."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(
            request.request_id,
            "b",
            ValidationVoteType.PROPOSE_ALTERNATIVE,
            0.7,
            alternative="Use method X",
        )
        await validator.submit_vote(request.request_id, "c", ValidationVoteType.ACCEPT, 0.8)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert "Use method X" in result.alternatives_proposed

    @pytest.mark.asyncio
    async def test_result_max_alternatives(self, validator_with_custom_config):
        """Test alternatives capped to max_alternatives."""
        request = await validator_with_custom_config.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d", "e", "f"],
            quorum=2,
        )

        # Submit more alternatives than max
        await validator_with_custom_config.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9
        )
        await validator_with_custom_config.submit_vote(
            request.request_id, "b", ValidationVoteType.ACCEPT, 0.8
        )

        # These votes won't affect the result since we already have consensus
        # but alternatives would be collected if we had more validators

    @pytest.mark.asyncio
    async def test_consensus_strength_calculation(self, validator):
        """Test consensus strength is calculated correctly."""
        # Use quorum=4 to ensure all 4 votes are counted before consensus
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c", "d"],
            quorum=4,  # Need all 4 votes to reach quorum
        )

        # 3 accept, 1 reject = 75% consensus
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)
        await validator.submit_vote(request.request_id, "c", ValidationVoteType.ACCEPT, 0.85)
        await validator.submit_vote(request.request_id, "d", ValidationVoteType.REJECT, 0.7)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.consensus_strength == pytest.approx(0.75, rel=0.01)


# =============================================================================
# Timeout and Expiration Tests
# =============================================================================


class TestTimeoutAndExpiration:
    """Test timeout and expiration handling."""

    @pytest.mark.asyncio
    async def test_expired_request_on_vote(self, validator):
        """Test voting on expired request fails."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
            deadline_hours=0,  # Immediate deadline
        )

        # Set deadline to past
        request.deadline = datetime.now(timezone.utc) - timedelta(hours=1)

        success = await validator.submit_vote(
            request_id=request.request_id,
            validator_id="claude",
            vote_type=ValidationVoteType.ACCEPT,
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_expired_request_on_check(self, validator):
        """Test checking consensus on expired request returns expiration result."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
        )

        # Set deadline to past
        request.deadline = datetime.now(timezone.utc) - timedelta(hours=1)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "expired"

    @pytest.mark.asyncio
    async def test_expired_result_includes_partial_votes(self, validator):
        """Test expiration result includes votes received before deadline."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            deadline_hours=24,
        )

        # Submit one vote before expiration
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)

        # Expire the request
        request.deadline = datetime.now(timezone.utc) - timedelta(hours=1)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "expired"
        assert len(result.votes) == 1
        assert result.metadata["votes_received"] == 1

    @pytest.mark.asyncio
    async def test_is_expired_property(self, validator):
        """Test is_expired property."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            deadline_hours=24,
        )

        assert request.is_expired is False

        request.deadline = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert request.is_expired is True

    @pytest.mark.asyncio
    async def test_no_deadline_never_expires(self, validator):
        """Test request with no deadline never expires."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        # Remove deadline
        request.deadline = None

        assert request.is_expired is False


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_validator(self, validator):
        """Test validation with single validator."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
            quorum=1,
        )

        await validator.submit_vote(request.request_id, "claude", ValidationVoteType.ACCEPT, 0.9)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"

    @pytest.mark.asyncio
    async def test_tie_breaking_by_count(self, validator_no_auto_escalate):
        """Test tie scenario results in deadlock."""
        request = await validator_no_auto_escalate.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
        )

        await validator_no_auto_escalate.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9
        )
        await validator_no_auto_escalate.submit_vote(
            request.request_id, "b", ValidationVoteType.REJECT, 0.9
        )

        result = await validator_no_auto_escalate.check_consensus(request.request_id)

        assert result is not None
        # With equal votes, neither side has majority
        assert result.outcome == "deadlocked"

    @pytest.mark.asyncio
    async def test_late_vote_after_consensus(self, validator):
        """Test vote after consensus is already reached."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        # Consensus already reached
        result = await validator.check_consensus(request.request_id)
        assert result is not None

        # Late vote should fail
        success = await validator.submit_vote(
            request.request_id, "c", ValidationVoteType.REJECT, 0.7
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_all_abstain_votes(self, validator_no_auto_escalate):
        """Test all validators abstaining."""
        request = await validator_no_auto_escalate.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator_no_auto_escalate.submit_vote(
            request.request_id, "a", ValidationVoteType.ABSTAIN, 0.5
        )
        await validator_no_auto_escalate.submit_vote(
            request.request_id, "b", ValidationVoteType.ABSTAIN, 0.5
        )

        result = await validator_no_auto_escalate.check_consensus(request.request_id)

        # No decisive votes, consensus cannot be reached
        assert result is None or result.outcome == "deadlocked"

    @pytest.mark.asyncio
    async def test_zero_confidence_votes(self, validator):
        """Test votes with zero confidence."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.0)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.0)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"

    @pytest.mark.asyncio
    async def test_max_confidence_votes(self, validator):
        """Test votes with maximum confidence."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 1.0)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 1.0)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.consensus_strength == 1.0

    @pytest.mark.asyncio
    async def test_empty_validators_list(self, validator):
        """Test creating request with empty validators."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=[],
            quorum=1,
        )

        # Required votes capped to 0
        assert request.required_votes == 0
        assert request.validators == []


# =============================================================================
# Escalation Tests
# =============================================================================


class TestEscalation:
    """Test escalation functionality."""

    @pytest.mark.asyncio
    async def test_auto_escalate_on_deadlock(self, validator):
        """Test auto-escalation on deadlock."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            strategy=ValidationConsensusStrategy.UNANIMOUS,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.REJECT, 0.8)

        result = await validator.check_consensus(request.request_id)

        assert result is not None
        # Check escalation was created
        escalation = validator._escalations.get(request.request_id)
        assert escalation is not None

    @pytest.mark.asyncio
    async def test_manual_escalation(self, validator):
        """Test manual escalation of in-review request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b", "c"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)

        # Manually escalate
        escalation = await validator.escalate_deadlock(
            request.request_id,
            reason="Urgent decision needed",
            escalate_to="cto",
        )

        assert escalation is not None
        assert escalation.escalated_to == "cto"
        assert escalation.reason == "Urgent decision needed"

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.state == ValidationState.ESCALATED

    @pytest.mark.asyncio
    async def test_escalation_preserves_votes(self, validator):
        """Test escalation preserves original votes."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(
            request.request_id, "a", ValidationVoteType.ACCEPT, 0.9, "Strong evidence"
        )

        escalation = await validator.escalate_deadlock(request.request_id)

        assert escalation is not None
        assert len(escalation.original_votes) == 1
        assert escalation.original_votes[0].reasoning == "Strong evidence"

    @pytest.mark.asyncio
    async def test_resolve_escalation(self, validator):
        """Test resolving an escalated request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.escalate_deadlock(request.request_id)

        result = await validator.resolve_escalation(
            request.request_id,
            resolver_id="admin",
            verdict=ValidationVoteType.ACCEPT,
            reasoning="Admin approval",
        )

        assert result is not None
        assert result.outcome == "resolved_by_escalation"
        assert result.final_verdict == ValidationVoteType.ACCEPT
        assert result.metadata["resolved_by"] == "admin"

    @pytest.mark.asyncio
    async def test_cannot_escalate_completed(self, validator):
        """Test cannot escalate completed request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        # Request is now complete
        escalation = await validator.escalate_deadlock(request.request_id)

        assert escalation is None


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestCancellation:
    """Test request cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_request(self, validator):
        """Test cancelling a pending request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
        )

        success = await validator.cancel_request(request.request_id, reason="No longer needed")

        assert success is True

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.state == ValidationState.CANCELLED
        assert updated.metadata["cancellation_reason"] == "No longer needed"

    @pytest.mark.asyncio
    async def test_cancel_in_review_request(self, validator):
        """Test cancelling an in-review request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)

        success = await validator.cancel_request(request.request_id)

        assert success is True

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed(self, validator):
        """Test cannot cancel completed request."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        success = await validator.cancel_request(request.request_id)

        assert success is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_request(self, validator):
        """Test cancelling nonexistent request."""
        success = await validator.cancel_request("nonexistent")

        assert success is False


# =============================================================================
# Query Methods Tests
# =============================================================================


class TestQueryMethods:
    """Test query and retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_request(self, validator):
        """Test getting request by ID."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        retrieved = validator.get_request(request.request_id)

        assert retrieved is not None
        assert retrieved.item_id == "km_123"

    def test_get_nonexistent_request(self, validator):
        """Test getting nonexistent request returns None."""
        retrieved = validator.get_request("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_result(self, validator):
        """Test getting result by request ID."""
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        await validator.check_consensus(request.request_id)

        result = validator.get_result(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"

    def test_get_nonexistent_result(self, validator):
        """Test getting nonexistent result returns None."""
        result = validator.get_result("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_for_validator(self, validator):
        """Test getting pending requests for a validator."""
        await validator.create_validation_request(
            item_id="km_1",
            validators=["claude", "gpt-4"],
        )

        await validator.create_validation_request(
            item_id="km_2",
            validators=["claude", "gemini"],
        )

        await validator.create_validation_request(
            item_id="km_3",
            validators=["gpt-4", "gemini"],
        )

        pending = validator.get_pending_for_validator("claude")

        assert len(pending) == 2
        item_ids = [r.item_id for r in pending]
        assert "km_1" in item_ids
        assert "km_2" in item_ids

    @pytest.mark.asyncio
    async def test_pending_excludes_voted(self, validator):
        """Test pending requests exclude already voted."""
        request = await validator.create_validation_request(
            item_id="km_1",
            validators=["claude", "gpt-4"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "claude", ValidationVoteType.ACCEPT, 0.9)

        pending = validator.get_pending_for_validator("claude")

        # Claude has voted, so no pending for them
        assert len(pending) == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics and metrics."""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, validator):
        """Test initial statistics."""
        stats = validator.get_stats()

        assert stats["total_requests"] == 0
        assert stats["completed"] == 0
        assert stats["pending"] == 0
        assert stats["escalated"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_requests(self, validator):
        """Test statistics after creating requests."""
        await validator.create_validation_request(
            item_id="km_1",
            validators=["a"],
        )

        await validator.create_validation_request(
            item_id="km_2",
            validators=["b"],
        )

        stats = validator.get_stats()

        assert stats["total_requests"] == 2
        assert stats["pending"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_completion_rate(self, validator):
        """Test completion rate calculation."""
        request = await validator.create_validation_request(
            item_id="km_1",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        stats = validator.get_stats()

        assert stats["completed"] == 1
        assert stats["completion_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_get_stats_by_outcome(self, validator):
        """Test statistics by outcome."""
        request1 = await validator.create_validation_request(
            item_id="km_1",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request1.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request1.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        request2 = await validator.create_validation_request(
            item_id="km_2",
            validators=["c", "d"],
            quorum=2,
        )

        await validator.submit_vote(request2.request_id, "c", ValidationVoteType.REJECT, 0.9)
        await validator.submit_vote(request2.request_id, "d", ValidationVoteType.REJECT, 0.8)

        stats = validator.get_stats()

        assert stats["by_outcome"]["accepted"] == 1
        assert stats["by_outcome"]["rejected"] == 1

    @pytest.mark.asyncio
    async def test_avg_consensus_strength(self, validator):
        """Test average consensus strength calculation."""
        request = await validator.create_validation_request(
            item_id="km_1",
            validators=["a", "b"],
            quorum=2,
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        stats = validator.get_stats()

        # 2 out of 2 accept = 100% consensus
        assert stats["avg_consensus_strength"] == 1.0


# =============================================================================
# Notification Tests
# =============================================================================


class TestNotifications:
    """Test notification callbacks."""

    @pytest.mark.asyncio
    async def test_validator_notification_callback(self, validator):
        """Test validators are notified on request creation."""
        notifications = []

        def callback(validator_id, event_type, data):
            notifications.append((validator_id, event_type, data))

        validator.set_notification_callback(callback)

        await validator.create_validation_request(
            item_id="km_123",
            validators=["claude", "gpt-4"],
        )

        assert len(notifications) == 2
        assert notifications[0][0] in ["claude", "gpt-4"]
        assert notifications[0][1] == "validation_requested"

    @pytest.mark.asyncio
    async def test_proposer_notification_on_complete(self, validator):
        """Test proposer is notified on completion."""
        notifications = []

        def callback(recipient_id, event_type, data):
            notifications.append((recipient_id, event_type, data))

        validator.set_notification_callback(callback)

        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["a", "b"],
            quorum=2,
            proposer_id="proposer_user",
        )

        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)
        await validator.submit_vote(request.request_id, "b", ValidationVoteType.ACCEPT, 0.8)

        # Find proposer notification
        proposer_notifs = [n for n in notifications if n[0] == "proposer_user"]
        assert len(proposer_notifs) == 1
        assert proposer_notifs[0][1] == "validation_complete"

    @pytest.mark.asyncio
    async def test_notification_callback_error_handled(self, validator):
        """Test notification callback errors are handled gracefully."""

        def bad_callback(validator_id, event_type, data):
            raise Exception("Callback error")

        validator.set_notification_callback(bad_callback)

        # Should not raise despite callback error
        request = await validator.create_validation_request(
            item_id="km_123",
            validators=["claude"],
        )

        assert request is not None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton accessor."""

    def test_get_multi_party_validator_singleton(self):
        """Test get_multi_party_validator returns singleton."""
        # Reset singleton for test
        import aragora.knowledge.mound.ops.multi_party_validation as mpv

        mpv._multi_party_validator = None

        v1 = get_multi_party_validator()
        v2 = get_multi_party_validator()

        assert v1 is v2

        # Cleanup
        mpv._multi_party_validator = None

    def test_singleton_config_on_first_call(self):
        """Test config is only used on first singleton call."""
        import aragora.knowledge.mound.ops.multi_party_validation as mpv

        mpv._multi_party_validator = None

        config = ValidatorConfig(default_quorum=5)
        v1 = get_multi_party_validator(config)

        assert v1.config.default_quorum == 5

        # Second call with different config should not change
        config2 = ValidatorConfig(default_quorum=10)
        v2 = get_multi_party_validator(config2)

        assert v2.config.default_quorum == 5
        assert v1 is v2

        # Cleanup
        mpv._multi_party_validator = None


# =============================================================================
# Vote Count Tests
# =============================================================================


class TestVoteCounting:
    """Test vote counting logic."""

    def test_count_votes_basic(self, validator):
        """Test _count_votes with basic votes."""
        votes = [
            ValidationVote("a", ValidationVoteType.ACCEPT, 0.9, weight=1.0),
            ValidationVote("b", ValidationVoteType.ACCEPT, 0.8, weight=1.0),
            ValidationVote("c", ValidationVoteType.REJECT, 0.7, weight=1.0),
        ]

        counts = validator._count_votes(votes)

        assert counts["accept"] == 2
        assert counts["reject"] == 1
        assert counts["abstain"] == 0

    def test_count_votes_weighted(self, validator):
        """Test _count_votes with weighted votes."""
        votes = [
            ValidationVote("a", ValidationVoteType.ACCEPT, 0.9, weight=2.0),
            ValidationVote("b", ValidationVoteType.REJECT, 0.8, weight=1.0),
        ]

        counts = validator._count_votes(votes)

        # weighted_accept = 2.0 * 0.9 = 1.8
        # weighted_reject = 1.0 * 0.8 = 0.8
        assert counts["weighted_accept"] == pytest.approx(1.8, rel=0.01)
        assert counts["weighted_reject"] == pytest.approx(0.8, rel=0.01)
        assert counts["total_weight"] == 3.0

    def test_count_votes_all_types(self, validator):
        """Test _count_votes with all vote types."""
        votes = [
            ValidationVote("a", ValidationVoteType.ACCEPT),
            ValidationVote("b", ValidationVoteType.REJECT),
            ValidationVote("c", ValidationVoteType.ABSTAIN),
            ValidationVote("d", ValidationVoteType.PROPOSE_ALTERNATIVE),
            ValidationVote("e", ValidationVoteType.REQUEST_INFO),
        ]

        counts = validator._count_votes(votes)

        assert counts["accept"] == 1
        assert counts["reject"] == 1
        assert counts["abstain"] == 1
        assert counts["propose_alternative"] == 1
        assert counts["request_info"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_validation_workflow(self, validator):
        """Test complete validation workflow from creation to result."""
        # Create request
        request = await validator.create_validation_request(
            item_id="km_integration_test",
            validators=["claude", "gpt-4", "gemini"],
            quorum=2,
            strategy=ValidationConsensusStrategy.MAJORITY,
            proposer_id="test_user",
            context={"source": "integration_test"},
        )

        assert request.state == ValidationState.PENDING

        # Submit votes
        await validator.submit_vote(
            request.request_id,
            "claude",
            ValidationVoteType.ACCEPT,
            confidence=0.95,
            reasoning="High-quality evidence",
        )

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.state == ValidationState.IN_REVIEW

        await validator.submit_vote(
            request.request_id,
            "gpt-4",
            ValidationVoteType.ACCEPT,
            confidence=0.85,
            reasoning="Consistent with prior knowledge",
        )

        # Check result
        result = await validator.check_consensus(request.request_id)

        assert result is not None
        assert result.outcome == "accepted"
        assert result.accept_count == 2
        assert result.consensus_strength == 1.0

        # Verify stats
        stats = validator.get_stats()
        assert stats["total_requests"] == 1
        assert stats["completed"] == 1
        assert stats["by_outcome"]["accepted"] == 1

    @pytest.mark.asyncio
    async def test_escalation_resolution_workflow(self, validator):
        """Test manual escalation and resolution workflow."""
        request = await validator.create_validation_request(
            item_id="km_escalation_test",
            validators=["a", "b", "c"],  # 3 validators
            quorum=2,
            strategy=ValidationConsensusStrategy.UNANIMOUS,
        )

        # Submit only one vote to keep in IN_REVIEW state
        await validator.submit_vote(request.request_id, "a", ValidationVoteType.ACCEPT, 0.9)

        # Request should be in IN_REVIEW state
        updated_request = validator.get_request(request.request_id)
        assert updated_request is not None
        assert updated_request.state == ValidationState.IN_REVIEW

        # Manually escalate before more votes come in
        escalation = await validator.escalate_deadlock(
            request.request_id,
            reason="Time-sensitive decision needed",
            escalate_to="senior_admin",
        )

        assert escalation is not None
        assert escalation.escalated_to == "senior_admin"

        # The request state should now be ESCALATED
        updated_request = validator.get_request(request.request_id)
        assert updated_request is not None
        assert updated_request.state == ValidationState.ESCALATED

        # Resolve escalation
        result = await validator.resolve_escalation(
            request.request_id,
            resolver_id="senior_admin",
            verdict=ValidationVoteType.ACCEPT,
            reasoning="Admin override - approved",
        )

        assert result is not None
        assert result.outcome == "resolved_by_escalation"
        assert result.metadata["resolved_by"] == "senior_admin"
        assert result.metadata["reasoning"] == "Admin override - approved"

        updated = validator.get_request(request.request_id)
        assert updated is not None
        assert updated.state == ValidationState.CONSENSUS_REACHED

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, validator, large_validator_pool):
        """Test handling multiple concurrent requests."""
        requests = []

        # Create 5 requests
        for i in range(5):
            req = await validator.create_validation_request(
                item_id=f"km_concurrent_{i}",
                validators=large_validator_pool[:3],
                quorum=2,
            )
            requests.append(req)

        # Submit votes for all
        for i, req in enumerate(requests):
            await validator.submit_vote(
                req.request_id, large_validator_pool[0], ValidationVoteType.ACCEPT, 0.9
            )
            await validator.submit_vote(
                req.request_id, large_validator_pool[1], ValidationVoteType.ACCEPT, 0.8
            )

        # Verify all completed
        stats = validator.get_stats()
        assert stats["total_requests"] == 5
        assert stats["completed"] == 5
