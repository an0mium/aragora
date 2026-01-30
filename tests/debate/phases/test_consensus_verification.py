"""
Tests for consensus verification module.

Tests cover:
- ConsensusVerifier class initialization
- apply_verification_bonuses method
- _update_elo_from_verification method
- adjust_vote_confidence_from_verification method
- _emit_verification_event method
- Vote verification logic
- Bonus calculations
- Consensus detection
- Async behavior
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Vote
from aragora.debate.phases.consensus_verification import ConsensusVerifier


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockProtocol:
    """Mock debate protocol for testing."""

    verify_claims_during_consensus: bool = True
    verification_weight_bonus: float = 0.2
    verification_timeout_seconds: float = 5.0


@dataclass
class MockEloSystem:
    """Mock ELO system for testing."""

    updates: list = field(default_factory=list)
    update_return_value: float = 10.0

    def update_from_verification(
        self,
        agent_name: str,
        domain: str,
        verified_count: int,
        disproven_count: int,
    ) -> float:
        """Record and return an ELO update."""
        self.updates.append(
            {
                "agent_name": agent_name,
                "domain": domain,
                "verified_count": verified_count,
                "disproven_count": disproven_count,
            }
        )
        return self.update_return_value


@dataclass
class MockResult:
    """Mock debate result for testing."""

    verification_results: dict = field(default_factory=dict)
    verification_bonuses: dict = field(default_factory=dict)


@dataclass
class MockEventEmitter:
    """Mock event emitter for testing."""

    events: list = field(default_factory=list)

    def emit(self, event: Any) -> None:
        """Record emitted event."""
        self.events.append(event)


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    result: MockResult = field(default_factory=MockResult)
    event_emitter: MockEventEmitter | None = None
    loop_id: str = "loop-123"
    debate_id: str = "debate-456"


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str = "voter1"
    choice: str = "agent1"
    reasoning: str = "Good proposal"
    confidence: float = 0.8


def create_vote(
    agent: str = "voter1",
    choice: str = "agent1",
    reasoning: str = "Good proposal",
    confidence: float = 0.8,
) -> Vote:
    """Create a real Vote object for testing."""
    return Vote(
        agent=agent,
        choice=choice,
        reasoning=reasoning,
        confidence=confidence,
    )


# =============================================================================
# ConsensusVerifier Initialization Tests
# =============================================================================


class TestConsensusVerifierInit:
    """Tests for ConsensusVerifier initialization."""

    def test_init_with_no_arguments(self):
        """Verifier initializes with defaults when no arguments provided."""
        verifier = ConsensusVerifier()

        assert verifier.protocol is None
        assert verifier.elo_system is None
        assert verifier._verify_claims is None
        assert verifier._extract_debate_domain is None

    def test_init_with_all_arguments(self):
        """Verifier stores all injected dependencies."""
        protocol = MockProtocol()
        elo_system = MockEloSystem()
        verify_cb = AsyncMock()
        domain_cb = MagicMock()

        verifier = ConsensusVerifier(
            protocol=protocol,
            elo_system=elo_system,
            verify_claims=verify_cb,
            extract_debate_domain=domain_cb,
        )

        assert verifier.protocol is protocol
        assert verifier.elo_system is elo_system
        assert verifier._verify_claims is verify_cb
        assert verifier._extract_debate_domain is domain_cb

    def test_init_with_partial_arguments(self):
        """Verifier handles partial argument initialization."""
        protocol = MockProtocol()

        verifier = ConsensusVerifier(protocol=protocol)

        assert verifier.protocol is protocol
        assert verifier.elo_system is None
        assert verifier._verify_claims is None


# =============================================================================
# apply_verification_bonuses Tests
# =============================================================================


class TestApplyVerificationBonuses:
    """Tests for apply_verification_bonuses method."""

    @pytest.mark.asyncio
    async def test_returns_unchanged_when_no_protocol(self):
        """Returns unchanged vote counts when protocol is None."""
        verifier = ConsensusVerifier()
        ctx = MockDebateContext()
        vote_counts = {"agent1": 3.0, "agent2": 2.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal 1"},
            choice_mapping={},
        )

        assert result == {"agent1": 3.0, "agent2": 2.0}

    @pytest.mark.asyncio
    async def test_returns_unchanged_when_verify_disabled(self):
        """Returns unchanged vote counts when verify_claims_during_consensus is False."""
        protocol = MockProtocol(verify_claims_during_consensus=False)
        verifier = ConsensusVerifier(protocol=protocol)
        ctx = MockDebateContext()
        vote_counts = {"agent1": 3.0, "agent2": 2.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal 1"},
            choice_mapping={},
        )

        assert result == {"agent1": 3.0, "agent2": 2.0}

    @pytest.mark.asyncio
    async def test_returns_unchanged_when_no_verify_callback(self):
        """Returns unchanged vote counts when verify_claims callback is None."""
        protocol = MockProtocol(verify_claims_during_consensus=True)
        verifier = ConsensusVerifier(protocol=protocol, verify_claims=None)
        ctx = MockDebateContext()
        vote_counts = {"agent1": 3.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal 1"},
            choice_mapping={},
        )

        assert result == {"agent1": 3.0}

    @pytest.mark.asyncio
    async def test_applies_bonus_for_verified_claims_dict_format(self):
        """Applies bonus when verify_claims returns dict with verified count."""
        verify_cb = AsyncMock(return_value={"verified": 2, "disproven": 0})
        protocol = MockProtocol(verification_weight_bonus=0.2)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Verified proposal"},
            choice_mapping={},
        )

        # Bonus: 10.0 * 0.2 * 2 = 4.0 -> total = 14.0
        assert result["agent1"] == 14.0

    @pytest.mark.asyncio
    async def test_applies_bonus_for_verified_claims_int_format(self):
        """Applies bonus when verify_claims returns int (legacy format)."""
        verify_cb = AsyncMock(return_value=3)
        protocol = MockProtocol(verification_weight_bonus=0.1)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Verified proposal"},
            choice_mapping={},
        )

        # Bonus: 10.0 * 0.1 * 3 = 3.0 -> total = 13.0
        assert result["agent1"] == 13.0

    @pytest.mark.asyncio
    async def test_no_bonus_when_zero_verified(self):
        """No bonus applied when verified count is zero."""
        verify_cb = AsyncMock(return_value={"verified": 0, "disproven": 0})
        protocol = MockProtocol(verification_weight_bonus=0.2)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Unverified proposal"},
            choice_mapping={},
        )

        assert result["agent1"] == 10.0

    @pytest.mark.asyncio
    async def test_uses_choice_mapping_for_canonical_name(self):
        """Uses choice_mapping to map agent names to canonical forms."""
        verify_cb = AsyncMock(return_value={"verified": 1, "disproven": 0})
        protocol = MockProtocol(verification_weight_bonus=0.5)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"proposal_a": 10.0}
        choice_mapping = {"agent1": "proposal_a"}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "My proposal"},
            choice_mapping=choice_mapping,
        )

        # Bonus: 10.0 * 0.5 * 1 = 5.0 -> total = 15.0
        assert result["proposal_a"] == 15.0

    @pytest.mark.asyncio
    async def test_skips_agent_not_in_vote_counts(self):
        """Skips agents not present in vote_counts."""
        verify_cb = AsyncMock(return_value={"verified": 1, "disproven": 0})
        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent2": 5.0}  # agent1 not in vote_counts

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal 1"},
            choice_mapping={},
        )

        assert result == {"agent2": 5.0}
        # verify_claims should not be called since agent1 not in vote_counts
        # Actually, it is called but no bonus applied
        # Let's check verify_cb was called

    @pytest.mark.asyncio
    async def test_stores_verification_results_in_context(self):
        """Stores verification results in ctx.result.verification_results."""
        verify_cb = AsyncMock(return_value={"verified": 2, "disproven": 1})
        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal"},
            choice_mapping={},
        )

        assert "agent1" in ctx.result.verification_results
        assert ctx.result.verification_results["agent1"]["verified"] == 2
        assert ctx.result.verification_results["agent1"]["disproven"] == 1

    @pytest.mark.asyncio
    async def test_stores_verification_bonuses_in_context(self):
        """Stores verification bonuses in ctx.result.verification_bonuses."""
        verify_cb = AsyncMock(return_value={"verified": 2, "disproven": 0})
        protocol = MockProtocol(verification_weight_bonus=0.25)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 8.0}

        await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal"},
            choice_mapping={},
        )

        # Bonus: 8.0 * 0.25 * 2 = 4.0
        assert ctx.result.verification_bonuses["agent1"] == 4.0

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Handles timeout error gracefully and stores -1 indicator."""

        async def slow_verify(proposal_text, limit):
            await asyncio.sleep(10)  # Will timeout
            return {"verified": 1, "disproven": 0}

        protocol = MockProtocol(verification_timeout_seconds=0.01)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=slow_verify,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal"},
            choice_mapping={},
        )

        # Vote count unchanged on timeout
        assert result["agent1"] == 10.0
        # Timeout indicator stored
        assert ctx.result.verification_results["agent1"] == -1

    @pytest.mark.asyncio
    async def test_handles_verify_callback_exception(self):
        """Handles exceptions from verify_claims callback gracefully."""
        verify_cb = AsyncMock(side_effect=RuntimeError("Verification failed"))
        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal"},
            choice_mapping={},
        )

        # Vote count unchanged on error
        assert result["agent1"] == 10.0

    @pytest.mark.asyncio
    async def test_processes_multiple_proposals(self):
        """Processes multiple proposals and applies bonuses correctly."""

        async def verify_cb(proposal_text, limit):
            if "correct" in proposal_text:
                return {"verified": 2, "disproven": 0}
            return {"verified": 0, "disproven": 0}

        protocol = MockProtocol(verification_weight_bonus=0.2)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0, "agent2": 10.0}

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={
                "agent1": "This is a correct proposal",
                "agent2": "This is a wrong proposal",
            },
            choice_mapping={},
        )

        # agent1: 10.0 + (10.0 * 0.2 * 2) = 14.0
        # agent2: 10.0 (no bonus)
        assert result["agent1"] == 14.0
        assert result["agent2"] == 10.0

    @pytest.mark.asyncio
    async def test_calls_verify_with_correct_limit(self):
        """Calls verify_claims with limit=2."""
        verify_cb = AsyncMock(return_value={"verified": 0, "disproven": 0})
        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal text"},
            choice_mapping={},
        )

        verify_cb.assert_called_once_with("Proposal text", limit=2)

    @pytest.mark.asyncio
    async def test_calls_update_elo_from_verification(self):
        """Calls _update_elo_from_verification after processing proposals."""
        verify_cb = AsyncMock(return_value={"verified": 1, "disproven": 0})
        elo_system = MockEloSystem()
        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            elo_system=elo_system,
            verify_claims=verify_cb,
        )
        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}

        await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals={"agent1": "Proposal"},
            choice_mapping={},
        )

        # ELO system should have been called
        assert len(elo_system.updates) == 1
        assert elo_system.updates[0]["agent_name"] == "agent1"


# =============================================================================
# _update_elo_from_verification Tests
# =============================================================================


class TestUpdateEloFromVerification:
    """Tests for _update_elo_from_verification method."""

    @pytest.mark.asyncio
    async def test_no_update_when_no_elo_system(self):
        """Does nothing when elo_system is None."""
        verifier = ConsensusVerifier(elo_system=None)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 1, "disproven": 0}}

        # Should not raise
        await verifier._update_elo_from_verification(ctx)

    @pytest.mark.asyncio
    async def test_no_update_when_no_verification_results(self):
        """Does nothing when verification_results is empty."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {}

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 0

    @pytest.mark.asyncio
    async def test_updates_elo_for_verified_claims_dict_format(self):
        """Updates ELO when verification results use dict format."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 2, "disproven": 1}}

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 1
        assert elo_system.updates[0]["agent_name"] == "agent1"
        assert elo_system.updates[0]["verified_count"] == 2
        assert elo_system.updates[0]["disproven_count"] == 1
        assert elo_system.updates[0]["domain"] == "general"

    @pytest.mark.asyncio
    async def test_updates_elo_for_verified_claims_int_format(self):
        """Updates ELO when verification results use int format (legacy)."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": 3}

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 1
        assert elo_system.updates[0]["verified_count"] == 3
        assert elo_system.updates[0]["disproven_count"] == 0

    @pytest.mark.asyncio
    async def test_skips_timeout_indicator_int_format(self):
        """Skips agents with timeout indicator (-1) in int format."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": -1}  # Timeout

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 0

    @pytest.mark.asyncio
    async def test_skips_zero_verified_and_disproven(self):
        """Skips agents with zero verified and zero disproven."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 0, "disproven": 0}}

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 0

    @pytest.mark.asyncio
    async def test_uses_extract_debate_domain_callback(self):
        """Uses extract_debate_domain callback when provided."""
        elo_system = MockEloSystem()
        domain_cb = MagicMock(return_value="finance")
        verifier = ConsensusVerifier(
            elo_system=elo_system,
            extract_debate_domain=domain_cb,
        )
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 1, "disproven": 0}}

        await verifier._update_elo_from_verification(ctx)

        assert elo_system.updates[0]["domain"] == "finance"

    @pytest.mark.asyncio
    async def test_handles_domain_extraction_error(self):
        """Falls back to 'general' when domain extraction fails."""
        elo_system = MockEloSystem()
        domain_cb = MagicMock(side_effect=RuntimeError("Domain error"))
        verifier = ConsensusVerifier(
            elo_system=elo_system,
            extract_debate_domain=domain_cb,
        )
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 1, "disproven": 0}}

        await verifier._update_elo_from_verification(ctx)

        assert elo_system.updates[0]["domain"] == "general"

    @pytest.mark.asyncio
    async def test_handles_elo_update_error(self):
        """Handles ELO update errors gracefully."""
        elo_system = MagicMock()
        elo_system.update_from_verification.side_effect = RuntimeError("ELO error")
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {"agent1": {"verified": 1, "disproven": 0}}

        # Should not raise
        await verifier._update_elo_from_verification(ctx)

    @pytest.mark.asyncio
    async def test_processes_multiple_agents(self):
        """Processes multiple agents' verification results."""
        elo_system = MockEloSystem()
        verifier = ConsensusVerifier(elo_system=elo_system)
        ctx = MockDebateContext()
        ctx.result.verification_results = {
            "agent1": {"verified": 2, "disproven": 0},
            "agent2": {"verified": 0, "disproven": 1},
            "agent3": {"verified": 1, "disproven": 1},
        }

        await verifier._update_elo_from_verification(ctx)

        assert len(elo_system.updates) == 3


# =============================================================================
# adjust_vote_confidence_from_verification Tests
# =============================================================================


class TestAdjustVoteConfidenceFromVerification:
    """Tests for adjust_vote_confidence_from_verification method."""

    def test_boosts_confidence_for_verified_proposal(self):
        """Boosts vote confidence when supporting a verified proposal."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.5)]
        verification_results = {"agent1": {"verified": 1, "disproven": 0}}
        proposals = {"agent1": "Verified proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence boosted: 0.5 * 1.3 = 0.65
        assert result[0].confidence == pytest.approx(0.65, rel=1e-2)

    def test_reduces_confidence_for_disproven_proposal(self):
        """Reduces vote confidence when supporting a disproven proposal."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.8)]
        verification_results = {"agent1": {"verified": 0, "disproven": 1}}
        proposals = {"agent1": "Disproven proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence reduced: 0.8 * 0.3 = 0.24
        assert result[0].confidence == pytest.approx(0.24, rel=1e-2)

    def test_caps_confidence_at_max(self):
        """Caps boosted confidence at 0.99."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.9)]
        verification_results = {"agent1": {"verified": 2, "disproven": 0}}
        proposals = {"agent1": "Verified proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence boosted: 0.9 * 1.3^2 = 1.521, capped at 0.99
        assert result[0].confidence == 0.99

    def test_floors_confidence_at_min(self):
        """Floors reduced confidence at 0.01."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.1)]
        verification_results = {"agent1": {"verified": 0, "disproven": 3}}
        proposals = {"agent1": "Disproven proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence reduced: 0.1 * 0.3^3 = 0.0027, floored at 0.01
        assert result[0].confidence == 0.01

    def test_no_change_for_unverified_proposal(self):
        """No confidence change when proposal has no verification data."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.7)]
        verification_results = {}  # No verification data
        proposals = {"agent1": "Unverified proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        assert result[0].confidence == 0.7

    def test_handles_mixed_verification_results(self):
        """Handles proposals with both verified and disproven claims."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.6)]
        # When disproven > 0, penalty is applied regardless of verified
        verification_results = {"agent1": {"verified": 2, "disproven": 1}}
        proposals = {"agent1": "Mixed verification proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence reduced (disproven takes precedence): 0.6 * 0.3 = 0.18
        assert result[0].confidence == pytest.approx(0.18, rel=1e-2)

    def test_handles_legacy_int_format(self):
        """Handles legacy int format for verification results."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.5)]
        verification_results = {"agent1": 2}  # Legacy: int = verified count
        proposals = {"agent1": "Verified proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Confidence boosted: 0.5 * 1.3^2 = 0.845
        assert result[0].confidence == pytest.approx(0.845, rel=1e-2)

    def test_skips_timeout_indicator(self):
        """Skips votes for proposals with timeout indicator (-1)."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.5)]
        verification_results = {"agent1": -1}  # Timeout indicator
        proposals = {"agent1": "Timeout proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # No change on timeout
        assert result[0].confidence == 0.5

    def test_handles_vote_without_confidence_attribute(self):
        """Skips votes that don't have confidence attribute."""
        verifier = ConsensusVerifier()

        class VoteWithoutConfidence:
            choice = "agent1"

        votes = [VoteWithoutConfidence()]
        verification_results = {"agent1": {"verified": 1, "disproven": 0}}
        proposals = {"agent1": "Proposal"}

        # Should not raise
        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        assert len(result) == 1

    def test_handles_vote_without_choice_attribute(self):
        """Skips votes that don't have choice attribute."""
        verifier = ConsensusVerifier()

        class VoteWithoutChoice:
            confidence = 0.5

        votes = [VoteWithoutChoice()]
        verification_results = {"agent1": {"verified": 1, "disproven": 0}}
        proposals = {"agent1": "Proposal"}

        # Should not raise
        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        assert len(result) == 1

    def test_matches_agent_name_case_insensitive(self):
        """Matches agent names case-insensitively."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="AGENT1", confidence=0.5)]
        verification_results = {"agent1": {"verified": 1, "disproven": 0}}
        proposals = {"agent1": "Proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # Should match "agent1" with "AGENT1" (case insensitive)
        assert result[0].confidence == pytest.approx(0.65, rel=1e-2)

    def test_processes_multiple_votes(self):
        """Processes multiple votes correctly."""
        verifier = ConsensusVerifier()
        votes = [
            create_vote(agent="voter1", choice="agent1", confidence=0.5),
            create_vote(agent="voter2", choice="agent2", confidence=0.6),
            create_vote(agent="voter3", choice="agent1", confidence=0.7),
        ]
        verification_results = {
            "agent1": {"verified": 1, "disproven": 0},
            "agent2": {"verified": 0, "disproven": 1},
        }
        proposals = {"agent1": "Verified", "agent2": "Disproven"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        # voter1 and voter3 voted for agent1 (verified) - boosted
        assert result[0].confidence == pytest.approx(0.65, rel=1e-2)
        assert result[2].confidence == pytest.approx(0.91, rel=1e-2)
        # voter2 voted for agent2 (disproven) - reduced
        assert result[1].confidence == pytest.approx(0.18, rel=1e-2)

    def test_returns_same_list_modified_in_place(self):
        """Returns the same list (modified in place)."""
        verifier = ConsensusVerifier()
        votes = [create_vote(choice="agent1", confidence=0.5)]
        verification_results = {"agent1": {"verified": 1, "disproven": 0}}
        proposals = {"agent1": "Proposal"}

        result = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=verification_results,
            proposals=proposals,
        )

        assert result is votes


# =============================================================================
# _emit_verification_event Tests
# =============================================================================


class TestEmitVerificationEvent:
    """Tests for _emit_verification_event method."""

    def test_emits_event_with_correct_data(self):
        """Emits verification event with correct data."""
        verifier = ConsensusVerifier()
        event_emitter = MockEventEmitter()
        ctx = MockDebateContext(event_emitter=event_emitter)

        verifier._emit_verification_event(
            ctx=ctx,
            agent_name="agent1",
            verified_count=2,
            bonus=4.0,
            timeout=False,
        )

        # Verify event was emitted
        assert len(event_emitter.events) == 1
        event = event_emitter.events[0]
        assert event.agent == "agent1"
        assert event.loop_id == "loop-123"
        assert event.data["verified_count"] == 2
        assert event.data["bonus_applied"] == 4.0
        assert event.data["timeout"] is False
        assert event.data["agent"] == "agent1"

    def test_no_emit_when_no_event_emitter(self):
        """Does nothing when event_emitter is None."""
        verifier = ConsensusVerifier()
        ctx = MockDebateContext(event_emitter=None)

        # Should not raise
        verifier._emit_verification_event(
            ctx=ctx,
            agent_name="agent1",
            verified_count=2,
            bonus=4.0,
        )

    def test_handles_emit_error_gracefully(self):
        """Handles errors during event emission gracefully."""
        verifier = ConsensusVerifier()
        event_emitter = MagicMock()
        event_emitter.emit.side_effect = RuntimeError("Emit error")
        ctx = MockDebateContext(event_emitter=event_emitter)

        # Should not raise (imports may fail, so we just test it doesn't crash)
        verifier._emit_verification_event(
            ctx=ctx,
            agent_name="agent1",
            verified_count=2,
            bonus=4.0,
        )

    def test_includes_timeout_flag_in_event(self):
        """Includes timeout flag in emitted event."""
        verifier = ConsensusVerifier()
        event_emitter = MockEventEmitter()
        ctx = MockDebateContext(event_emitter=event_emitter)

        verifier._emit_verification_event(
            ctx=ctx,
            agent_name="agent1",
            verified_count=-1,
            bonus=0.0,
            timeout=True,
        )

        assert len(event_emitter.events) == 1
        event = event_emitter.events[0]
        assert event.data["timeout"] is True
        assert event.data["verified_count"] == -1

    def test_includes_debate_id_in_event(self):
        """Includes debate_id in emitted event data."""
        verifier = ConsensusVerifier()
        event_emitter = MockEventEmitter()
        ctx = MockDebateContext(event_emitter=event_emitter)
        ctx.debate_id = "debate-xyz"

        verifier._emit_verification_event(
            ctx=ctx,
            agent_name="agent1",
            verified_count=1,
            bonus=2.0,
        )

        assert len(event_emitter.events) == 1
        event = event_emitter.events[0]
        assert event.data["debate_id"] == "debate-xyz"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConsensusVerifierIntegration:
    """Integration tests for ConsensusVerifier."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self):
        """Tests complete verification flow from proposals to ELO update."""

        # Setup
        async def verify_claims(proposal_text, limit):
            if "correct" in proposal_text.lower():
                return {"verified": 2, "disproven": 0}
            elif "wrong" in proposal_text.lower():
                return {"verified": 0, "disproven": 2}
            return {"verified": 0, "disproven": 0}

        elo_system = MockEloSystem()
        protocol = MockProtocol(verification_weight_bonus=0.2)
        event_emitter = MockEventEmitter()

        verifier = ConsensusVerifier(
            protocol=protocol,
            elo_system=elo_system,
            verify_claims=verify_claims,
        )

        ctx = MockDebateContext(event_emitter=event_emitter)
        vote_counts = {"correct_agent": 10.0, "wrong_agent": 8.0, "neutral_agent": 5.0}
        proposals = {
            "correct_agent": "This correct proposal has good claims",
            "wrong_agent": "This wrong proposal has bad claims",
            "neutral_agent": "This neutral proposal",
        }

        # Execute
        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals=proposals,
            choice_mapping={},
        )

        # Verify vote counts updated
        # correct_agent: 10.0 + (10.0 * 0.2 * 2) = 14.0
        assert result["correct_agent"] == 14.0
        # wrong_agent: no bonus (verified=0)
        assert result["wrong_agent"] == 8.0
        # neutral_agent: no bonus (verified=0)
        assert result["neutral_agent"] == 5.0

        # Verify verification results stored
        assert ctx.result.verification_results["correct_agent"]["verified"] == 2
        assert ctx.result.verification_results["wrong_agent"]["disproven"] == 2

        # Verify verification bonuses stored
        assert ctx.result.verification_bonuses["correct_agent"] == 4.0

        # Verify ELO updates called
        assert len(elo_system.updates) == 2  # Only for non-zero results

    @pytest.mark.asyncio
    async def test_verification_with_vote_confidence_adjustment(self):
        """Tests verification combined with vote confidence adjustment."""

        async def verify_claims(proposal_text, limit):
            return {"verified": 1, "disproven": 0}

        protocol = MockProtocol()
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_claims,
        )

        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0}
        proposals = {"agent1": "Verified proposal"}

        # First apply verification bonuses
        await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals=proposals,
            choice_mapping={},
        )

        # Then adjust vote confidence
        votes = [create_vote(choice="agent1", confidence=0.5)]
        adjusted_votes = verifier.adjust_vote_confidence_from_verification(
            votes=votes,
            verification_results=ctx.result.verification_results,
            proposals=proposals,
        )

        # Confidence should be boosted
        assert adjusted_votes[0].confidence == pytest.approx(0.65, rel=1e-2)

    @pytest.mark.asyncio
    async def test_concurrent_verification_of_multiple_proposals(self):
        """Tests that verification of multiple proposals works correctly."""
        call_order = []

        async def verify_claims(proposal_text, limit):
            agent = proposal_text.split()[0]  # Extract agent name from proposal
            call_order.append(agent)
            await asyncio.sleep(0.01)  # Small delay to simulate async work
            return {"verified": 1, "disproven": 0}

        protocol = MockProtocol(verification_weight_bonus=0.1)
        verifier = ConsensusVerifier(
            protocol=protocol,
            verify_claims=verify_claims,
        )

        ctx = MockDebateContext()
        vote_counts = {"agent1": 10.0, "agent2": 10.0, "agent3": 10.0}
        proposals = {
            "agent1": "agent1 proposal",
            "agent2": "agent2 proposal",
            "agent3": "agent3 proposal",
        }

        result = await verifier.apply_verification_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            proposals=proposals,
            choice_mapping={},
        )

        # All agents should have been verified
        assert len(call_order) == 3
        # All should have received bonus: 10.0 + (10.0 * 0.1 * 1) = 11.0
        assert result["agent1"] == 11.0
        assert result["agent2"] == 11.0
        assert result["agent3"] == 11.0
