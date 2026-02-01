"""
Integration tests for consensus with mixed internal and external agents.

Tests consensus behavior when combining internal (trusted) and external
(gateway) agents:
- Internal agents can outvote external proposals
- External proposals are verified by internal agents
- Agent weights are applied correctly
- Slow agents do not block consensus
- Low quality proposals are rejected
- Insufficient votes fail gracefully
"""

import pytest
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from unittest.mock import MagicMock, AsyncMock, patch

from tests.gateway.integration.conftest import (
    MockAgent,
    FailingAgent,
    SlowAgent,
    TenantContext,
)


class AgentType(str, Enum):
    """Type of agent in consensus."""

    INTERNAL = "internal"
    EXTERNAL = "external"


class VoteType(str, Enum):
    """Types of consensus votes."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """A vote from an agent on a proposal."""

    agent_name: str
    agent_type: AgentType
    vote_type: VoteType
    confidence: float
    weight: float = 1.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def weighted_value(self) -> float:
        """Calculate weighted vote value."""
        if self.vote_type == VoteType.APPROVE:
            return self.weight * self.confidence
        elif self.vote_type == VoteType.REJECT:
            return -self.weight * self.confidence
        return 0.0


@dataclass
class Proposal:
    """A proposal from an agent."""

    proposer: str
    proposer_type: AgentType
    content: str
    quality_score: float = 0.5
    evidence: list = field(default_factory=list)
    verified: bool = False
    verified_by: list = field(default_factory=list)


@dataclass
class ConsensusResult:
    """Result of a consensus vote."""

    proposal: Proposal
    votes: list[Vote]
    total_weight: float = 0.0
    approve_weight: float = 0.0
    reject_weight: float = 0.0
    abstain_count: int = 0
    reached_consensus: bool = False
    consensus_threshold: float = 0.5
    quorum_reached: bool = False
    quorum_requirement: int = 3


class MixedAgentConsensus:
    """Consensus engine for mixed internal and external agents."""

    def __init__(
        self,
        consensus_threshold: float = 0.5,
        quorum_requirement: int = 3,
        quality_threshold: float = 0.3,
        verification_required: bool = True,
        timeout_seconds: float = 30.0,
    ):
        self.consensus_threshold = consensus_threshold
        self.quorum_requirement = quorum_requirement
        self.quality_threshold = quality_threshold
        self.verification_required = verification_required
        self.timeout_seconds = timeout_seconds

        self.agents: dict[str, dict] = {}
        self.proposals: list[Proposal] = []
        self.votes: list[Vote] = []

    def register_agent(
        self,
        name: str,
        agent_type: AgentType,
        weight: float = 1.0,
    ) -> None:
        """Register an agent for consensus."""
        self.agents[name] = {
            "name": name,
            "type": agent_type,
            "weight": weight,
        }

    def submit_proposal(self, proposal: Proposal) -> bool:
        """Submit a proposal for consensus.

        Returns False if proposal doesn't meet quality threshold.
        """
        if proposal.quality_score < self.quality_threshold:
            return False
        self.proposals.append(proposal)
        return True

    async def verify_proposal(
        self,
        proposal: Proposal,
        verifying_agents: list[str],
    ) -> bool:
        """Have internal agents verify an external proposal."""
        internal_verifiers = [
            name
            for name in verifying_agents
            if self.agents.get(name, {}).get("type") == AgentType.INTERNAL
        ]

        if not internal_verifiers and self.verification_required:
            return False

        # Simulate verification
        proposal.verified = len(internal_verifiers) > 0 or not self.verification_required
        proposal.verified_by = internal_verifiers
        return proposal.verified

    def cast_vote(
        self,
        agent_name: str,
        proposal: Proposal,
        vote_type: VoteType,
        confidence: float,
        reason: str = "",
    ) -> Vote:
        """Cast a vote on a proposal."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not registered")

        vote = Vote(
            agent_name=agent_name,
            agent_type=agent["type"],
            vote_type=vote_type,
            confidence=confidence,
            weight=agent["weight"],
            reason=reason,
        )
        self.votes.append(vote)
        return vote

    async def collect_votes_with_timeout(
        self,
        proposal: Proposal,
        voting_agents: list[tuple[str, VoteType, float]],
        timeout: float | None = None,
    ) -> list[Vote]:
        """Collect votes with timeout handling for slow agents."""
        timeout = timeout or self.timeout_seconds
        collected_votes = []

        async def cast_with_delay(
            agent_name: str,
            vote_type: VoteType,
            confidence: float,
            delay: float = 0.0,
        ):
            await asyncio.sleep(delay)
            return self.cast_vote(agent_name, proposal, vote_type, confidence)

        # Create vote tasks
        tasks = []
        for agent_name, vote_type, confidence in voting_agents:
            agent = self.agents.get(agent_name, {})
            delay = agent.get("delay", 0.0)
            tasks.append(cast_with_delay(agent_name, vote_type, confidence, delay))

        # Collect with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            for result in results:
                if isinstance(result, Vote):
                    collected_votes.append(result)
        except asyncio.TimeoutError:
            # Return whatever votes we collected before timeout
            pass

        return collected_votes

    def calculate_consensus(
        self,
        proposal: Proposal,
        votes: list[Vote] | None = None,
    ) -> ConsensusResult:
        """Calculate consensus from votes."""
        proposal_votes = votes or [v for v in self.votes if v.agent_name in self.agents]

        total_weight = 0.0
        approve_weight = 0.0
        reject_weight = 0.0
        abstain_count = 0

        for vote in proposal_votes:
            total_weight += vote.weight
            if vote.vote_type == VoteType.APPROVE:
                approve_weight += vote.weighted_value
            elif vote.vote_type == VoteType.REJECT:
                reject_weight += abs(vote.weighted_value)
            else:
                abstain_count += 1

        quorum_reached = len(proposal_votes) >= self.quorum_requirement
        vote_ratio = (
            approve_weight / (approve_weight + reject_weight)
            if (approve_weight + reject_weight) > 0
            else 0
        )
        reached_consensus = quorum_reached and vote_ratio >= self.consensus_threshold

        return ConsensusResult(
            proposal=proposal,
            votes=proposal_votes,
            total_weight=total_weight,
            approve_weight=approve_weight,
            reject_weight=reject_weight,
            abstain_count=abstain_count,
            reached_consensus=reached_consensus,
            consensus_threshold=self.consensus_threshold,
            quorum_reached=quorum_reached,
            quorum_requirement=self.quorum_requirement,
        )


class TestConsensusMixedAgents:
    """Integration tests for consensus with mixed agent types."""

    @pytest.fixture
    def consensus_engine(self) -> MixedAgentConsensus:
        """Create a consensus engine for testing."""
        return MixedAgentConsensus(
            consensus_threshold=0.5,
            quorum_requirement=3,
            quality_threshold=0.3,
        )

    @pytest.mark.asyncio
    async def test_mixed_consensus_internal_majority(
        self,
        mock_agent: MockAgent,
        failing_agent: FailingAgent,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that internal agents can outvote external proposal."""
        # Register agents: 1 external, 2 internal
        consensus_engine.register_agent("external-1", AgentType.EXTERNAL, weight=1.0)
        consensus_engine.register_agent("internal-1", AgentType.INTERNAL, weight=1.0)
        consensus_engine.register_agent("internal-2", AgentType.INTERNAL, weight=1.0)

        # External agent proposes
        proposal = Proposal(
            proposer="external-1",
            proposer_type=AgentType.EXTERNAL,
            content="External proposal content",
            quality_score=0.7,
        )
        assert consensus_engine.submit_proposal(proposal) is True

        # Internal agents reject
        consensus_engine.cast_vote("external-1", proposal, VoteType.APPROVE, confidence=0.9)
        consensus_engine.cast_vote(
            "internal-1", proposal, VoteType.REJECT, confidence=0.8, reason="Proposal lacks detail"
        )
        consensus_engine.cast_vote(
            "internal-2", proposal, VoteType.REJECT, confidence=0.7, reason="Security concerns"
        )

        # Calculate consensus
        result = consensus_engine.calculate_consensus(proposal)

        # Internal majority rejects
        assert result.quorum_reached is True
        assert result.reached_consensus is False
        assert result.reject_weight > result.approve_weight

    @pytest.mark.asyncio
    async def test_mixed_consensus_external_proposal_verified(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that external proposal is verified by internal agents."""
        # Register agents
        consensus_engine.register_agent("external-1", AgentType.EXTERNAL)
        consensus_engine.register_agent("internal-1", AgentType.INTERNAL)
        consensus_engine.register_agent("internal-2", AgentType.INTERNAL)

        # External proposal
        proposal = Proposal(
            proposer="external-1",
            proposer_type=AgentType.EXTERNAL,
            content="External proposal requiring verification",
            quality_score=0.8,
        )

        # Verify by internal agents
        verified = await consensus_engine.verify_proposal(
            proposal,
            verifying_agents=["internal-1", "internal-2"],
        )

        assert verified is True
        assert proposal.verified is True
        assert "internal-1" in proposal.verified_by
        assert "internal-2" in proposal.verified_by

    @pytest.mark.asyncio
    async def test_mixed_consensus_weights(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that agent weights are applied correctly."""
        # Register agents with different weights
        consensus_engine.register_agent("heavy-internal", AgentType.INTERNAL, weight=3.0)
        consensus_engine.register_agent("light-external", AgentType.EXTERNAL, weight=1.0)
        consensus_engine.register_agent("medium-internal", AgentType.INTERNAL, weight=1.5)

        proposal = Proposal(
            proposer="light-external",
            proposer_type=AgentType.EXTERNAL,
            content="Weighted vote test",
            quality_score=0.6,
        )
        consensus_engine.submit_proposal(proposal)

        # Heavy internal approves, others reject
        consensus_engine.cast_vote("heavy-internal", proposal, VoteType.APPROVE, confidence=0.8)
        consensus_engine.cast_vote("light-external", proposal, VoteType.REJECT, confidence=0.9)
        consensus_engine.cast_vote("medium-internal", proposal, VoteType.REJECT, confidence=0.7)

        result = consensus_engine.calculate_consensus(proposal)

        # Despite 2 rejections, heavy weight should tip balance
        # Heavy approve: 3.0 * 0.8 = 2.4
        # Light reject: 1.0 * 0.9 = 0.9
        # Medium reject: 1.5 * 0.7 = 1.05
        # Total reject: 1.95, approve: 2.4 -> consensus reached
        assert result.approve_weight == pytest.approx(2.4, rel=0.01)
        assert result.reject_weight == pytest.approx(1.95, rel=0.01)
        assert result.reached_consensus is True

    @pytest.mark.asyncio
    async def test_mixed_consensus_timeout_handling(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that slow agents do not block consensus."""
        # Register agents with one slow agent
        consensus_engine.register_agent("fast-1", AgentType.INTERNAL)
        consensus_engine.register_agent("fast-2", AgentType.INTERNAL)
        consensus_engine.register_agent("fast-3", AgentType.INTERNAL)
        consensus_engine.register_agent("slow-external", AgentType.EXTERNAL)
        consensus_engine.agents["slow-external"]["delay"] = 5.0  # 5 second delay

        proposal = Proposal(
            proposer="fast-1",
            proposer_type=AgentType.INTERNAL,
            content="Timeout test proposal",
            quality_score=0.7,
        )

        # Collect votes with short timeout
        voting_agents = [
            ("fast-1", VoteType.APPROVE, 0.9),
            ("fast-2", VoteType.APPROVE, 0.8),
            ("fast-3", VoteType.APPROVE, 0.85),
            ("slow-external", VoteType.REJECT, 0.9),
        ]

        votes = await consensus_engine.collect_votes_with_timeout(
            proposal,
            voting_agents,
            timeout=0.5,  # Short timeout
        )

        # Should have 3 fast votes, slow agent timed out
        assert len(votes) == 3
        assert all(v.agent_name.startswith("fast") for v in votes)

        # Consensus should still work with available votes
        result = consensus_engine.calculate_consensus(proposal, votes)
        assert result.quorum_reached is True
        assert result.reached_consensus is True

    @pytest.mark.asyncio
    async def test_mixed_consensus_quality_threshold(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that low quality proposals are rejected."""
        consensus_engine.register_agent("external-1", AgentType.EXTERNAL)
        consensus_engine.register_agent("internal-1", AgentType.INTERNAL)
        consensus_engine.register_agent("internal-2", AgentType.INTERNAL)

        # Low quality proposal
        low_quality = Proposal(
            proposer="external-1",
            proposer_type=AgentType.EXTERNAL,
            content="Vague and incomplete proposal",
            quality_score=0.1,  # Below threshold of 0.3
        )

        # Should be rejected at submission
        accepted = consensus_engine.submit_proposal(low_quality)
        assert accepted is False
        assert low_quality not in consensus_engine.proposals

        # High quality proposal should be accepted
        high_quality = Proposal(
            proposer="external-1",
            proposer_type=AgentType.EXTERNAL,
            content="Well-documented proposal with evidence",
            quality_score=0.8,
        )

        accepted = consensus_engine.submit_proposal(high_quality)
        assert accepted is True
        assert high_quality in consensus_engine.proposals

    @pytest.mark.asyncio
    async def test_mixed_consensus_no_quorum(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that insufficient votes fails gracefully."""
        # Register agents but only 2 will vote (quorum is 3)
        consensus_engine.register_agent("agent-1", AgentType.INTERNAL)
        consensus_engine.register_agent("agent-2", AgentType.EXTERNAL)
        consensus_engine.register_agent("agent-3", AgentType.INTERNAL)

        proposal = Proposal(
            proposer="agent-1",
            proposer_type=AgentType.INTERNAL,
            content="No quorum test",
            quality_score=0.6,
        )
        consensus_engine.submit_proposal(proposal)

        # Only 2 agents vote
        consensus_engine.cast_vote("agent-1", proposal, VoteType.APPROVE, 0.9)
        consensus_engine.cast_vote("agent-2", proposal, VoteType.APPROVE, 0.9)

        result = consensus_engine.calculate_consensus(proposal)

        # Quorum not reached despite all approvals
        assert result.quorum_reached is False
        assert result.reached_consensus is False
        assert len(result.votes) == 2
        assert result.quorum_requirement == 3


class TestConsensusMixedAgentsEdgeCases:
    """Edge case tests for mixed agent consensus."""

    @pytest.fixture
    def consensus_engine(self) -> MixedAgentConsensus:
        """Create a consensus engine for testing."""
        return MixedAgentConsensus(
            consensus_threshold=0.5,
            quorum_requirement=2,
        )

    @pytest.mark.asyncio
    async def test_all_abstain_no_consensus(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that all abstaining votes means no consensus."""
        consensus_engine.register_agent("agent-1", AgentType.INTERNAL)
        consensus_engine.register_agent("agent-2", AgentType.EXTERNAL)
        consensus_engine.register_agent("agent-3", AgentType.INTERNAL)

        proposal = Proposal(
            proposer="agent-1",
            proposer_type=AgentType.INTERNAL,
            content="Abstain test",
            quality_score=0.5,
        )

        consensus_engine.cast_vote("agent-1", proposal, VoteType.ABSTAIN, 0.5)
        consensus_engine.cast_vote("agent-2", proposal, VoteType.ABSTAIN, 0.5)
        consensus_engine.cast_vote("agent-3", proposal, VoteType.ABSTAIN, 0.5)

        result = consensus_engine.calculate_consensus(proposal)

        assert result.abstain_count == 3
        assert result.approve_weight == 0.0
        assert result.reject_weight == 0.0
        assert result.reached_consensus is False

    @pytest.mark.asyncio
    async def test_tie_vote_no_consensus(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that tie votes result in no consensus."""
        consensus_engine.register_agent("agent-1", AgentType.INTERNAL, weight=1.0)
        consensus_engine.register_agent("agent-2", AgentType.EXTERNAL, weight=1.0)

        proposal = Proposal(
            proposer="agent-1",
            proposer_type=AgentType.INTERNAL,
            content="Tie test",
            quality_score=0.5,
        )

        consensus_engine.cast_vote("agent-1", proposal, VoteType.APPROVE, 0.8)
        consensus_engine.cast_vote("agent-2", proposal, VoteType.REJECT, 0.8)

        result = consensus_engine.calculate_consensus(proposal)

        # Exactly 50% is threshold, so tie means no consensus
        assert result.approve_weight == pytest.approx(0.8, rel=0.01)
        assert result.reject_weight == pytest.approx(0.8, rel=0.01)
        assert result.reached_consensus is False

    @pytest.mark.asyncio
    async def test_unregistered_agent_cannot_vote(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test that unregistered agents cannot cast votes."""
        proposal = Proposal(
            proposer="unknown",
            proposer_type=AgentType.EXTERNAL,
            content="Unregistered vote test",
            quality_score=0.5,
        )

        with pytest.raises(ValueError) as exc_info:
            consensus_engine.cast_vote(
                "unregistered-agent",
                proposal,
                VoteType.APPROVE,
                0.9,
            )

        assert "not registered" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_external_proposal_without_verification_fails(
        self,
    ):
        """Test that unverified external proposals fail when verification required."""
        engine = MixedAgentConsensus(verification_required=True)

        engine.register_agent("external-1", AgentType.EXTERNAL)
        # No internal agents to verify

        proposal = Proposal(
            proposer="external-1",
            proposer_type=AgentType.EXTERNAL,
            content="Unverifiable proposal",
            quality_score=0.7,
        )

        # Try to verify with no internal agents
        verified = await engine.verify_proposal(
            proposal,
            verifying_agents=["external-1"],  # External cannot verify
        )

        assert verified is False
        assert proposal.verified is False

    @pytest.mark.asyncio
    async def test_high_confidence_low_weight_vs_low_confidence_high_weight(
        self,
        consensus_engine: MixedAgentConsensus,
    ):
        """Test interaction between confidence and weight."""
        consensus_engine.register_agent("high-conf-low-weight", AgentType.INTERNAL, weight=0.5)
        consensus_engine.register_agent("low-conf-high-weight", AgentType.EXTERNAL, weight=2.0)

        proposal = Proposal(
            proposer="external",
            proposer_type=AgentType.EXTERNAL,
            content="Weight vs confidence test",
            quality_score=0.5,
        )

        # High confidence (0.9) * low weight (0.5) = 0.45
        consensus_engine.cast_vote("high-conf-low-weight", proposal, VoteType.APPROVE, 0.9)
        # Low confidence (0.3) * high weight (2.0) = 0.6
        consensus_engine.cast_vote("low-conf-high-weight", proposal, VoteType.REJECT, 0.3)

        result = consensus_engine.calculate_consensus(proposal)

        # Low confidence but high weight wins
        assert result.reject_weight > result.approve_weight
        assert result.approve_weight == pytest.approx(0.45, rel=0.01)
        assert result.reject_weight == pytest.approx(0.6, rel=0.01)
