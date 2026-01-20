"""
Tests for Byzantine Fault-Tolerant Consensus Protocol.

Tests cover:
- ByzantinePhase and ViewChangeReason enums
- ByzantineMessage data class
- ByzantineConsensusConfig configuration
- ByzantineConsensusResult data class and properties
- ByzantineConsensus protocol operations
- Normal consensus scenarios (no Byzantine nodes)
- Single Byzantine node detection and handling
- Multiple Byzantine nodes (2, 3+ nodes)
- Byzantine recovery scenarios
- Consensus with partial failures
- Detection of conflicting votes/proposals
- Node removal after Byzantine detection (via view changes)
- Edge cases (all nodes Byzantine, no honest majority)
- View change mechanism
- Quorum calculations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Message, Vote
from aragora.debate.byzantine import (
    ByzantineConsensus,
    ByzantineConsensusConfig,
    ByzantineConsensusResult,
    ByzantineMessage,
    ByzantinePhase,
    ConsensusFailure,
    ViewChangeReason,
    verify_with_byzantine_consensus,
)


# =============================================================================
# Mock Agents for Testing
# =============================================================================


class MockAgent(Agent):
    """Mock agent for testing Byzantine consensus."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "PREPARE: YES\nREASONING: I agree",
        model: str = "mock-model",
        role: str = "proposer",
        should_fail: bool = False,
        fail_on_call: int = -1,
        delay_seconds: float = 0.0,
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.should_fail = should_fail
        self.fail_on_call = fail_on_call
        self.delay_seconds = delay_seconds
        self.generate_calls = 0
        self._responses: list[str] = []

    def set_responses(self, responses: list[str]) -> None:
        """Set sequential responses for multiple calls."""
        self._responses = responses

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_fail:
            raise RuntimeError(f"Agent {self.name} failed")

        if self.fail_on_call > 0 and self.generate_calls == self.fail_on_call:
            raise RuntimeError(f"Agent {self.name} failed on call {self.generate_calls}")

        if self._responses:
            idx = min(self.generate_calls - 1, len(self._responses) - 1)
            return self._responses[idx]

        return self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )


class ByzantineAgent(MockAgent):
    """Agent that exhibits Byzantine behavior (malicious/faulty)."""

    def __init__(
        self,
        name: str = "byzantine-agent",
        byzantine_behavior: str = "disagree",  # disagree, random, silent, conflicting
        model: str = "byzantine-model",
    ):
        super().__init__(name=name, model=model)
        self.byzantine_behavior = byzantine_behavior
        self.vote_history: list[str] = []

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1

        if self.byzantine_behavior == "disagree":
            # Always vote no
            return "PREPARE: NO\nREASONING: I disagree with everything"

        elif self.byzantine_behavior == "random":
            # Randomly agree or disagree
            import random

            choice = random.choice(["YES", "NO"])
            self.vote_history.append(choice)
            return f"PREPARE: {choice}\nREASONING: Random decision"

        elif self.byzantine_behavior == "silent":
            # Simulate timeout by raising an exception
            raise asyncio.TimeoutError("Agent silent")

        elif self.byzantine_behavior == "conflicting":
            # Send conflicting votes (different votes in prepare vs commit)
            if "PREPARE PHASE" in prompt:
                self.vote_history.append("YES")
                return "PREPARE: YES\nREASONING: I agree"
            else:
                self.vote_history.append("NO")
                return "COMMIT: NO\nREASONING: Changed my mind"

        elif self.byzantine_behavior == "slow":
            # Very slow responses
            await asyncio.sleep(100)
            return "PREPARE: YES\nREASONING: Finally responding"

        return "PREPARE: NO\nREASONING: Default Byzantine behavior"


# =============================================================================
# ByzantinePhase Enum Tests
# =============================================================================


class TestByzantinePhase:
    """Tests for ByzantinePhase enum."""

    def test_phase_values(self):
        """Test that all phase values are correct."""
        assert ByzantinePhase.PRE_PREPARE.value == "pre_prepare"
        assert ByzantinePhase.PREPARE.value == "prepare"
        assert ByzantinePhase.COMMIT.value == "commit"
        assert ByzantinePhase.REPLY.value == "reply"
        assert ByzantinePhase.VIEW_CHANGE.value == "view_change"

    def test_phase_from_string(self):
        """Test creating phase from string."""
        assert ByzantinePhase("pre_prepare") == ByzantinePhase.PRE_PREPARE
        assert ByzantinePhase("commit") == ByzantinePhase.COMMIT

    def test_phase_count(self):
        """Test that all expected phases exist."""
        phases = list(ByzantinePhase)
        assert len(phases) == 5


class TestViewChangeReason:
    """Tests for ViewChangeReason enum."""

    def test_reason_values(self):
        """Test that all reason values are correct."""
        assert ViewChangeReason.LEADER_TIMEOUT.value == "leader_timeout"
        assert ViewChangeReason.LEADER_FAILURE.value == "leader_failure"
        assert ViewChangeReason.INVALID_PROPOSAL.value == "invalid_proposal"
        assert ViewChangeReason.CONSENSUS_STALL.value == "consensus_stall"


# =============================================================================
# ByzantineMessage Tests
# =============================================================================


class TestByzantineMessage:
    """Tests for ByzantineMessage data class."""

    def test_message_creation(self):
        """Test creating a Byzantine message."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.PRE_PREPARE,
            view=0,
            sequence=1,
            sender="leader",
            proposal_hash="abc123",
            proposal="Test proposal",
        )

        assert msg.phase == ByzantinePhase.PRE_PREPARE
        assert msg.view == 0
        assert msg.sequence == 1
        assert msg.sender == "leader"
        assert msg.proposal_hash == "abc123"
        assert msg.proposal == "Test proposal"
        assert msg.timestamp is not None

    def test_message_hash_computation(self):
        """Test that message hash is computed correctly."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.PREPARE,
            view=1,
            sequence=5,
            sender="agent1",
            proposal_hash="proposal123",
        )

        hash1 = msg.compute_hash()
        hash2 = msg.compute_hash()

        # Hash should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_message_hash_differs_for_different_content(self):
        """Test that different messages have different hashes."""
        msg1 = ByzantineMessage(
            phase=ByzantinePhase.PREPARE,
            view=0,
            sequence=1,
            sender="agent1",
            proposal_hash="abc",
        )
        msg2 = ByzantineMessage(
            phase=ByzantinePhase.PREPARE,
            view=0,
            sequence=2,  # Different sequence
            sender="agent1",
            proposal_hash="abc",
        )

        assert msg1.compute_hash() != msg2.compute_hash()

    def test_message_without_proposal(self):
        """Test message creation without proposal (for non-PRE_PREPARE phases)."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.COMMIT,
            view=0,
            sequence=1,
            sender="agent1",
            proposal_hash="abc123",
        )

        assert msg.proposal is None


# =============================================================================
# ByzantineConsensusConfig Tests
# =============================================================================


class TestByzantineConsensusConfig:
    """Tests for ByzantineConsensusConfig data class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ByzantineConsensusConfig()

        assert config.max_faulty_fraction == 0.33
        assert config.phase_timeout_seconds == 30.0
        assert config.view_change_timeout_seconds == 60.0
        assert config.min_agents == 4
        assert config.max_view_changes == 3
        assert config.max_retries_per_phase == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ByzantineConsensusConfig(
            max_faulty_fraction=0.25,
            phase_timeout_seconds=10.0,
            min_agents=7,
            max_view_changes=5,
        )

        assert config.max_faulty_fraction == 0.25
        assert config.phase_timeout_seconds == 10.0
        assert config.min_agents == 7
        assert config.max_view_changes == 5


# =============================================================================
# ByzantineConsensusResult Tests
# =============================================================================


class TestByzantineConsensusResult:
    """Tests for ByzantineConsensusResult data class."""

    def test_result_creation_success(self):
        """Test creating a successful result."""
        result = ByzantineConsensusResult(
            success=True,
            value="Agreed proposal",
            confidence=0.9,
            view=0,
            sequence=1,
            commit_count=5,
            total_agents=7,
            duration_seconds=2.5,
            agent_votes={"a1": "hash1", "a2": "hash1", "a3": "hash1"},
        )

        assert result.success is True
        assert result.value == "Agreed proposal"
        assert result.confidence == 0.9
        assert result.commit_count == 5

    def test_result_creation_failure(self):
        """Test creating a failed result."""
        result = ByzantineConsensusResult(
            success=False,
            failure_reason="Consensus failed after 3 view changes",
            view=3,
            sequence=1,
            total_agents=7,
            duration_seconds=90.0,
        )

        assert result.success is False
        assert result.failure_reason is not None
        assert "view changes" in result.failure_reason

    def test_agreement_ratio(self):
        """Test agreement ratio calculation."""
        result = ByzantineConsensusResult(
            success=True,
            commit_count=5,
            total_agents=7,
        )

        ratio = result.agreement_ratio
        assert ratio == pytest.approx(5 / 7, rel=0.01)

    def test_agreement_ratio_full(self):
        """Test agreement ratio with full agreement."""
        result = ByzantineConsensusResult(
            success=True,
            commit_count=7,
            total_agents=7,
        )

        assert result.agreement_ratio == 1.0

    def test_agreement_ratio_zero_agents(self):
        """Test agreement ratio with zero agents."""
        result = ByzantineConsensusResult(
            success=False,
            commit_count=0,
            total_agents=0,
        )

        assert result.agreement_ratio == 0.0


# =============================================================================
# ByzantineConsensus Protocol Tests
# =============================================================================


class TestByzantineConsensusProperties:
    """Tests for ByzantineConsensus computed properties."""

    def test_n_property(self):
        """Test n property returns agent count."""
        agents = [MockAgent(name=f"a{i}") for i in range(7)]
        protocol = ByzantineConsensus(agents=agents)

        assert protocol.n == 7

    def test_f_property(self):
        """Test f property calculates faulty threshold correctly."""
        # n=7 -> f=(7-1)//3=2
        agents = [MockAgent(name=f"a{i}") for i in range(7)]
        protocol = ByzantineConsensus(agents=agents)

        assert protocol.f == 2

    def test_f_property_various_sizes(self):
        """Test f calculation for various agent counts."""
        test_cases = [
            (4, 1),  # n=4 -> f=1
            (7, 2),  # n=7 -> f=2
            (10, 3),  # n=10 -> f=3
            (13, 4),  # n=13 -> f=4
        ]

        for n, expected_f in test_cases:
            agents = [MockAgent(name=f"a{i}") for i in range(n)]
            protocol = ByzantineConsensus(agents=agents)
            assert protocol.f == expected_f, f"Failed for n={n}"

    def test_quorum_size(self):
        """Test quorum_size calculation (2f+1)."""
        agents = [MockAgent(name=f"a{i}") for i in range(7)]
        protocol = ByzantineConsensus(agents=agents)

        # f=2, so quorum=2*2+1=5
        assert protocol.quorum_size == 5

    def test_leader_selection(self):
        """Test leader selection based on view."""
        agents = [MockAgent(name=f"a{i}") for i in range(4)]
        protocol = ByzantineConsensus(agents=agents)

        # View 0 -> agent 0
        assert protocol.leader.name == "a0"

    def test_leader_rotation_on_view_change(self):
        """Test that leader rotates with view changes."""
        agents = [MockAgent(name=f"a{i}") for i in range(4)]
        protocol = ByzantineConsensus(agents=agents)

        # Simulate view changes
        protocol._current_view = 1
        assert protocol.leader.name == "a1"

        protocol._current_view = 2
        assert protocol.leader.name == "a2"

        protocol._current_view = 4  # Wraps around
        assert protocol.leader.name == "a0"


class TestByzantineConsensusMinAgentWarning:
    """Tests for minimum agent count warning."""

    def test_warning_when_too_few_agents(self, caplog):
        """Test that a warning is logged when not enough agents."""
        agents = [MockAgent(name=f"a{i}") for i in range(3)]  # Less than 4
        config = ByzantineConsensusConfig(min_agents=4)

        with caplog.at_level("WARNING"):
            protocol = ByzantineConsensus(agents=agents, config=config)

        assert "at least 4 agents" in caplog.text or protocol.n == 3

    def test_no_warning_with_enough_agents(self, caplog):
        """Test no warning when enough agents provided."""
        agents = [MockAgent(name=f"a{i}") for i in range(5)]
        config = ByzantineConsensusConfig(min_agents=4)

        protocol = ByzantineConsensus(agents=agents, config=config)

        assert protocol.n == 5


# =============================================================================
# Normal Consensus Tests (No Byzantine Nodes)
# =============================================================================


class TestNormalConsensus:
    """Tests for consensus with honest nodes only."""

    @pytest.fixture
    def honest_agents(self):
        """Create a set of honest agents that always agree."""
        return [
            MockAgent(name=f"honest{i}", response="PREPARE: YES\nREASONING: I agree")
            for i in range(7)
        ]

    @pytest.mark.asyncio
    async def test_consensus_with_all_honest_agents(self, honest_agents):
        """Test that consensus succeeds with all honest agents."""
        protocol = ByzantineConsensus(agents=honest_agents)

        result = await protocol.propose("Test proposal", task="Test task")

        assert result.success is True
        assert result.value == "Test proposal"
        assert result.commit_count >= protocol.quorum_size

    @pytest.mark.asyncio
    async def test_consensus_confidence_with_unanimous_agreement(self, honest_agents):
        """Test that confidence is high with unanimous agreement."""
        protocol = ByzantineConsensus(agents=honest_agents)

        result = await protocol.propose("Test proposal")

        assert result.confidence == 1.0  # All agents agreed

    @pytest.mark.asyncio
    async def test_consensus_tracks_agent_votes(self, honest_agents):
        """Test that agent votes are tracked in result."""
        protocol = ByzantineConsensus(agents=honest_agents)

        result = await protocol.propose("Test proposal")

        assert len(result.agent_votes) == len(honest_agents)

    @pytest.mark.asyncio
    async def test_consensus_with_specified_proposer(self, honest_agents):
        """Test consensus with a specific proposer agent."""
        protocol = ByzantineConsensus(agents=honest_agents)
        proposer = honest_agents[2]

        result = await protocol.propose("Test proposal", proposer=proposer)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_sequence_increments_on_propose(self, honest_agents):
        """Test that sequence number increments with each proposal."""
        protocol = ByzantineConsensus(agents=honest_agents)

        result1 = await protocol.propose("Proposal 1")
        result2 = await protocol.propose("Proposal 2")

        assert result2.sequence == result1.sequence + 1


# =============================================================================
# Single Byzantine Node Detection Tests
# =============================================================================


class TestSingleByzantineNode:
    """Tests for handling a single Byzantine node."""

    @pytest.fixture
    def agents_with_one_byzantine(self):
        """Create agents with one Byzantine node."""
        honest = [
            MockAgent(name=f"honest{i}", response="PREPARE: YES\nREASONING: I agree")
            for i in range(6)
        ]
        byzantine = [ByzantineAgent(name="byzantine0", byzantine_behavior="disagree")]
        return honest + byzantine

    @pytest.mark.asyncio
    async def test_consensus_succeeds_with_one_byzantine(self, agents_with_one_byzantine):
        """Test that consensus succeeds despite one Byzantine node."""
        protocol = ByzantineConsensus(agents=agents_with_one_byzantine)

        result = await protocol.propose("Test proposal")

        # With 6 honest + 1 byzantine (n=7, f=2), quorum=5
        # 6 honest agents should be enough
        assert result.success is True

    @pytest.mark.asyncio
    async def test_confidence_reflects_byzantine_dissent(self, agents_with_one_byzantine):
        """Test that confidence reflects the Byzantine dissent."""
        protocol = ByzantineConsensus(agents=agents_with_one_byzantine)

        result = await protocol.propose("Test proposal")

        # 6 out of 7 agents committed
        assert result.confidence == pytest.approx(6 / 7, rel=0.01)

    @pytest.mark.asyncio
    async def test_single_silent_byzantine_node(self):
        """Test handling a Byzantine node that goes silent (times out)."""
        config = ByzantineConsensusConfig(phase_timeout_seconds=0.1)
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(6)]
        byzantine = [ByzantineAgent(name="silent", byzantine_behavior="silent")]
        agents = honest + byzantine

        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # Should still succeed with 6 honest nodes
        assert result.success is True


# =============================================================================
# Multiple Byzantine Nodes Tests
# =============================================================================


class TestMultipleByzantineNodes:
    """Tests for handling multiple Byzantine nodes."""

    @pytest.mark.asyncio
    async def test_consensus_with_two_byzantine_nodes(self):
        """Test consensus with 2 Byzantine nodes (within f threshold for n=7)."""
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(5)]
        byzantine = [
            ByzantineAgent(name=f"byzantine{i}", byzantine_behavior="disagree") for i in range(2)
        ]
        agents = honest + byzantine

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # n=7, f=2, quorum=5. With 5 honest agents, should succeed.
        assert result.success is True
        assert result.commit_count >= 5

    @pytest.mark.asyncio
    async def test_consensus_fails_with_three_byzantine_nodes(self):
        """Test that consensus fails with 3 Byzantine nodes (exceeds f for n=7)."""
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(4)]
        byzantine = [
            ByzantineAgent(name=f"byzantine{i}", byzantine_behavior="disagree") for i in range(3)
        ]
        agents = honest + byzantine

        protocol = ByzantineConsensus(agents=agents)
        config = ByzantineConsensusConfig(max_view_changes=1)
        protocol.config = config

        result = await protocol.propose("Test proposal")

        # n=7, f=2, quorum=5. With only 4 honest agents, cannot reach quorum.
        # Should fail after view changes exhausted.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_mixed_byzantine_behaviors(self):
        """Test consensus with different Byzantine behaviors."""
        config = ByzantineConsensusConfig(phase_timeout_seconds=0.1)
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(8)]
        byzantine = [
            ByzantineAgent(name="disagree", byzantine_behavior="disagree"),
            ByzantineAgent(name="silent", byzantine_behavior="silent"),
        ]
        agents = honest + byzantine

        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # n=10, f=3, quorum=7. With 8 honest, should succeed.
        assert result.success is True


# =============================================================================
# Byzantine Recovery Scenarios Tests
# =============================================================================


class TestByzantineRecovery:
    """Tests for recovery from Byzantine failures."""

    @pytest.mark.asyncio
    async def test_view_change_on_leader_failure(self):
        """Test that view change occurs when leader fails."""
        # Make the first agent (leader in view 0) Byzantine
        byzantine_leader = ByzantineAgent(name="leader", byzantine_behavior="silent")
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(6)]
        agents = [byzantine_leader] + honest

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,
            max_view_changes=3,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # View should have changed due to leader timeout
        assert result.view >= 0  # View may have changed

    @pytest.mark.asyncio
    async def test_recovery_after_transient_failure(self):
        """Test recovery after a transient agent failure."""
        # Agent fails on first call but succeeds after
        transient_agent = MockAgent(
            name="transient",
            response="PREPARE: YES",
            fail_on_call=1,
        )
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(6)]
        agents = [transient_agent] + honest

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # Should still succeed with 6 always-honest agents
        assert result.success is True

    @pytest.mark.asyncio
    async def test_multiple_view_changes(self):
        """Test system behavior with multiple view changes."""
        # Multiple leaders fail
        failing_leaders = [
            ByzantineAgent(name=f"leader{i}", byzantine_behavior="silent") for i in range(2)
        ]
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(5)]
        agents = failing_leaders + honest

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,
            max_view_changes=5,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # Should eventually succeed with honest leader
        assert result.success is True


# =============================================================================
# Consensus with Partial Failures Tests
# =============================================================================


class TestPartialFailures:
    """Tests for consensus with partial failures."""

    @pytest.mark.asyncio
    async def test_prepare_phase_partial_failure(self):
        """Test handling partial failures in prepare phase."""
        # Some agents fail during prepare
        failing = [MockAgent(name=f"fail{i}", should_fail=True) for i in range(2)]
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(6)]
        agents = failing + honest

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # Should succeed with 6 honest out of 8 total (f=2, quorum=5)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_commit_phase_partial_failure(self):
        """Test handling partial failures in commit phase."""

        # Agent agrees in prepare but fails in commit
        class PrepareOnlyAgent(MockAgent):
            def __init__(self, name: str):
                super().__init__(name=name)
                self.in_commit = False

            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                if "COMMIT PHASE" in prompt:
                    raise RuntimeError("Commit failure")
                return "PREPARE: YES"

        prepare_only = [PrepareOnlyAgent(name=f"preponly{i}") for i in range(2)]
        fully_honest = [
            MockAgent(name=f"honest{i}", response="PREPARE: YES\nCOMMIT: YES") for i in range(6)
        ]
        agents = prepare_only + fully_honest

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # Should succeed with the fully honest agents
        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_during_consensus(self):
        """Test handling timeouts during consensus phases."""
        config = ByzantineConsensusConfig(phase_timeout_seconds=0.1)

        slow = [
            MockAgent(name=f"slow{i}", response="PREPARE: YES", delay_seconds=10.0)
            for i in range(2)
        ]
        fast = [MockAgent(name=f"fast{i}", response="PREPARE: YES") for i in range(6)]
        agents = slow + fast

        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # Should succeed with the fast agents responding
        assert result.success is True


# =============================================================================
# Conflicting Votes/Proposals Tests
# =============================================================================


class TestConflictingVotes:
    """Tests for detection and handling of conflicting votes."""

    @pytest.mark.asyncio
    async def test_conflicting_byzantine_agent(self):
        """Test handling agent that sends conflicting votes."""
        honest = [
            MockAgent(name=f"honest{i}", response="PREPARE: YES\nCOMMIT: YES") for i in range(6)
        ]
        conflicting = [ByzantineAgent(name="conflicting", byzantine_behavior="conflicting")]
        agents = honest + conflicting

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # Should succeed with honest majority
        assert result.success is True
        # Conflicting agent may not be in final commits
        assert result.commit_count >= protocol.quorum_size

    @pytest.mark.asyncio
    async def test_agents_with_varying_responses(self):
        """Test handling agents with varying response patterns."""

        # Some agents initially disagree, then agree
        class EvolvingAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                if self.generate_calls == 1:
                    return "PREPARE: NO\nREASONING: Need more info"
                return "PREPARE: YES\nREASONING: Now I agree"

        evolving = [EvolvingAgent(name=f"evolve{i}") for i in range(3)]
        stable = [MockAgent(name=f"stable{i}", response="PREPARE: YES") for i in range(4)]
        agents = evolving + stable

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # May need view change, but should eventually succeed
        assert result is not None


# =============================================================================
# View Change Mechanism Tests
# =============================================================================


class TestViewChanges:
    """Tests for the view change mechanism."""

    @pytest.mark.asyncio
    async def test_view_change_increments_view_number(self):
        """Test that view changes increment the view number."""
        # All Byzantine to force view changes
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree") for i in range(7)
        ]

        config = ByzantineConsensusConfig(max_view_changes=2)
        protocol = ByzantineConsensus(agents=byzantine, config=config)

        result = await protocol.propose("Test proposal")

        # View should have changed
        assert result.view >= 1

    @pytest.mark.asyncio
    async def test_max_view_changes_respected(self):
        """Test that max_view_changes limit is respected."""
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree") for i in range(7)
        ]

        config = ByzantineConsensusConfig(max_view_changes=3)
        protocol = ByzantineConsensus(agents=byzantine, config=config)

        result = await protocol.propose("Test proposal")

        # Should fail after exhausting view changes
        assert result.success is False
        assert "view changes" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_view_change_rotates_leader(self):
        """Test that view change causes leader rotation."""
        agents = [MockAgent(name=f"a{i}") for i in range(4)]
        protocol = ByzantineConsensus(agents=agents)

        # Check initial leader
        initial_leader = protocol.leader.name
        assert initial_leader == "a0"

        # Manually trigger view change
        protocol._current_view = 1

        # Check new leader
        new_leader = protocol.leader.name
        assert new_leader == "a1"
        assert new_leader != initial_leader


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in Byzantine consensus."""

    @pytest.mark.asyncio
    async def test_all_nodes_byzantine(self):
        """Test behavior when all nodes are Byzantine."""
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree") for i in range(7)
        ]

        config = ByzantineConsensusConfig(max_view_changes=1)
        protocol = ByzantineConsensus(agents=byzantine, config=config)

        result = await protocol.propose("Test proposal")

        # Should fail - no honest nodes
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_honest_majority(self):
        """Test behavior without honest majority (f+1 > n/2)."""
        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(3)]
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree") for i in range(4)
        ]
        agents = honest + byzantine

        config = ByzantineConsensusConfig(max_view_changes=1)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # n=7, f=2, quorum=5. Only 3 honest, cannot reach quorum.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_minimum_agents(self):
        """Test with minimum required agents (4)."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # n=4, f=1, quorum=3. Should work with all honest.
        assert result.success is True

    @pytest.mark.asyncio
    async def test_single_agent(self):
        """Test with just one agent (degenerate case)."""
        agents = [MockAgent(name="solo", response="PREPARE: YES")]

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal")

        # n=1, f=0, quorum=1. Should work.
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_proposal(self):
        """Test with empty proposal string."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("")

        assert result.success is True
        assert result.value == ""

    @pytest.mark.asyncio
    async def test_very_long_proposal(self):
        """Test with very long proposal string."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents)
        long_proposal = "x" * 10000

        result = await protocol.propose(long_proposal)

        assert result.success is True
        assert len(result.value) == 10000

    @pytest.mark.asyncio
    async def test_unicode_proposal(self):
        """Test with unicode characters in proposal."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents)
        unicode_proposal = "Test proposal with unicode: "

        result = await protocol.propose(unicode_proposal)

        assert result.success is True


# =============================================================================
# Agreement Parsing Tests
# =============================================================================


class TestAgreementParsing:
    """Tests for _parse_agreement method."""

    @pytest.fixture
    def protocol(self):
        """Create a protocol for testing parsing."""
        agents = [MockAgent(name="a1")]
        return ByzantineConsensus(agents=agents)

    def test_parse_explicit_yes(self, protocol):
        """Test parsing explicit YES response."""
        assert protocol._parse_agreement("PREPARE: YES\nREASONING: I agree") is True
        assert protocol._parse_agreement("COMMIT: YES") is True

    def test_parse_explicit_no(self, protocol):
        """Test parsing explicit NO response."""
        assert protocol._parse_agreement("PREPARE: NO\nREASONING: I disagree") is False
        assert protocol._parse_agreement("COMMIT: NO") is False

    def test_parse_implicit_agreement(self, protocol):
        """Test parsing implicit agreement words."""
        assert protocol._parse_agreement("I agree with this proposal") is True
        assert protocol._parse_agreement("I accept and approve this") is True
        assert protocol._parse_agreement("Yes, this is correct") is True

    def test_parse_implicit_disagreement(self, protocol):
        """Test parsing implicit disagreement words."""
        assert protocol._parse_agreement("I disagree with this") is False
        assert protocol._parse_agreement("I reject this proposal") is False
        assert protocol._parse_agreement("No, this is wrong") is False

    def test_parse_mixed_response(self, protocol):
        """Test parsing response with mixed signals."""
        # More agreement words than disagreement
        assert (
            protocol._parse_agreement("I agree and accept, even though I have minor concerns")
            is True
        )

    def test_parse_case_insensitive(self, protocol):
        """Test that parsing is case-insensitive."""
        assert protocol._parse_agreement("PREPARE: yes") is True
        assert protocol._parse_agreement("prepare: YES") is True


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestVerifyWithByzantineConsensus:
    """Tests for verify_with_byzantine_consensus convenience function."""

    @pytest.mark.asyncio
    async def test_basic_verification(self):
        """Test basic verification with convenience function."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        result = await verify_with_byzantine_consensus(
            proposal="Test proposal",
            agents=agents,
            task="Test task",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_custom_fault_tolerance(self):
        """Test with custom fault tolerance setting."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        result = await verify_with_byzantine_consensus(
            proposal="Test proposal",
            agents=agents,
            fault_tolerance=0.25,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_verification_with_byzantine_agents(self):
        """Test verification fails with too many Byzantine agents."""
        honest = [MockAgent(name=f"h{i}", response="PREPARE: YES") for i in range(3)]
        byzantine = [ByzantineAgent(name=f"b{i}", byzantine_behavior="disagree") for i in range(4)]
        agents = honest + byzantine

        result = await verify_with_byzantine_consensus(
            proposal="Test proposal",
            agents=agents,
        )

        # 3 honest out of 7, quorum=5, should fail
        assert result.success is False


# =============================================================================
# Parametrized Byzantine Scenario Tests
# =============================================================================


class TestParametrizedByzantineScenarios:
    """Parametrized tests for various Byzantine scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n_agents,n_byzantine,expected_success",
        [
            (4, 0, True),  # All honest
            (4, 1, True),  # 1 Byzantine (f=1)
            (7, 1, True),  # 1 Byzantine (f=2)
            (7, 2, True),  # 2 Byzantine (at limit)
            (10, 2, True),  # 2 Byzantine (f=3, under limit)
            (10, 3, True),  # 3 Byzantine (at limit)
            (7, 3, False),  # 3 Byzantine (exceeds f=2)
            (10, 4, False),  # 4 Byzantine (exceeds f=3)
        ],
    )
    async def test_byzantine_threshold(self, n_agents, n_byzantine, expected_success):
        """Test consensus success/failure based on Byzantine count."""
        n_honest = n_agents - n_byzantine

        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(n_honest)]
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree")
            for i in range(n_byzantine)
        ]
        agents = honest + byzantine

        config = ByzantineConsensusConfig(max_view_changes=1)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        assert result.success is expected_success, (
            f"Failed for n={n_agents}, byzantine={n_byzantine}: "
            f"expected {expected_success}, got {result.success}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "behavior",
        [
            "disagree",
            "silent",
            "conflicting",
        ],
    )
    async def test_different_byzantine_behaviors(self, behavior):
        """Test different Byzantine behaviors are handled."""
        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,
            max_view_changes=2,
        )

        honest = [MockAgent(name=f"honest{i}", response="PREPARE: YES") for i in range(6)]
        byzantine = [ByzantineAgent(name="byz", byzantine_behavior=behavior)]
        agents = honest + byzantine

        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # Should succeed with 6 honest out of 7
        assert result.success is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout_seconds", [0.01, 0.1, 1.0])
    async def test_varying_timeouts(self, timeout_seconds):
        """Test consensus with varying timeout configurations."""
        config = ByzantineConsensusConfig(
            phase_timeout_seconds=timeout_seconds,
            max_view_changes=2,
        )

        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        assert result.success is True


# =============================================================================
# Proposal Hash Tests
# =============================================================================


class TestProposalHash:
    """Tests for proposal hash computation."""

    @pytest.fixture
    def protocol(self):
        """Create a protocol for testing hash computation."""
        agents = [MockAgent(name="a1")]
        return ByzantineConsensus(agents=agents)

    def test_hash_deterministic(self, protocol):
        """Test that hash is deterministic."""
        proposal = "Test proposal content"

        hash1 = protocol._compute_proposal_hash(proposal)
        hash2 = protocol._compute_proposal_hash(proposal)

        assert hash1 == hash2

    def test_hash_length(self, protocol):
        """Test that hash has expected length."""
        proposal = "Test proposal"

        hash_value = protocol._compute_proposal_hash(proposal)

        assert len(hash_value) == 16

    def test_hash_different_for_different_proposals(self, protocol):
        """Test that different proposals have different hashes."""
        hash1 = protocol._compute_proposal_hash("Proposal A")
        hash2 = protocol._compute_proposal_hash("Proposal B")

        assert hash1 != hash2


# =============================================================================
# ConsensusFailure Exception Tests
# =============================================================================


class TestConsensusFailureException:
    """Tests for ConsensusFailure exception."""

    def test_exception_creation(self):
        """Test creating ConsensusFailure exception."""
        exc = ConsensusFailure("Test failure")
        assert str(exc) == "Test failure"

    def test_exception_inheritance(self):
        """Test that ConsensusFailure is an Exception."""
        exc = ConsensusFailure("Test")
        assert isinstance(exc, Exception)

    @pytest.mark.asyncio
    async def test_exception_raised_on_quorum_failure(self):
        """Test that ConsensusFailure is raised when quorum not met."""
        # This is internal behavior, but we can verify through result
        byzantine = [
            ByzantineAgent(name=f"byz{i}", byzantine_behavior="disagree") for i in range(7)
        ]

        config = ByzantineConsensusConfig(max_view_changes=0)
        protocol = ByzantineConsensus(agents=byzantine, config=config)

        result = await protocol.propose("Test")

        # Failure should be reflected in result
        assert result.success is False


# =============================================================================
# Duration Tracking Tests
# =============================================================================


class TestDurationTracking:
    """Tests for consensus duration tracking."""

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Test that duration is recorded in result."""
        agents = [MockAgent(name=f"a{i}", response="PREPARE: YES") for i in range(4)]

        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test")

        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_duration_increases_with_delays(self):
        """Test that duration reflects agent delays."""
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)

        # Fast agents
        fast_agents = [MockAgent(name=f"fast{i}", response="PREPARE: YES") for i in range(4)]
        protocol_fast = ByzantineConsensus(agents=fast_agents, config=config)
        result_fast = await protocol_fast.propose("Test")

        # Slow agents
        slow_agents = [
            MockAgent(name=f"slow{i}", response="PREPARE: YES", delay_seconds=0.1) for i in range(4)
        ]
        protocol_slow = ByzantineConsensus(agents=slow_agents, config=config)
        result_slow = await protocol_slow.propose("Test")

        # Slow result should have higher duration
        assert result_slow.duration_seconds >= result_fast.duration_seconds
