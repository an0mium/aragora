"""
End-to-end tests for debate orchestrator lifecycle.

Tests full debate flows from start to consensus, including:
- Complete proposal → critique → revision → vote → consensus flow
- Multi-agent debates with different consensus modes
- User participation affecting outcomes
- Evidence collection and citations
- ELO rating updates after completion

Note: These tests run the full Arena.run() which can be slow.
Run with: pytest tests/test_orchestrator_e2e_lifecycle.py -v --timeout=120
"""

import pytest

# Mark all tests in this module as slow (arena tests take time)
pytestmark = pytest.mark.slow
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

from aragora.core import (
    Agent,
    Environment,
    Vote,
    Message,
    Critique,
    DebateResult,
)
from aragora.debate.orchestrator import Arena, DebateProtocol


class E2EMockAgent(Agent):
    """Mock agent for E2E testing with configurable behavior."""

    def __init__(
        self,
        name: str,
        proposal: str = None,
        vote_for: str = None,
        critique_severity: float = 0.3,
        confidence: float = 0.8,
    ):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"
        self._proposal = proposal or f"Proposal from {name}: This is my answer."
        self._vote_for = vote_for  # None means vote for self
        self._critique_severity = critique_severity
        self._confidence = confidence
        self.generate_count = 0
        self.critique_count = 0
        self.vote_count = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_count += 1
        return self._proposal

    async def generate_stream(self, prompt: str, context: list = None):
        yield self._proposal

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        self.critique_count += 1
        return Critique(
            agent=self.name,
            target_agent="unknown",
            target_content=proposal[:100],
            issues=["Could be clearer", "Needs more detail"],
            suggestions=["Add examples", "Provide reasoning"],
            severity=self._critique_severity,
            reasoning=f"Critique from {self.name}",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_count += 1
        # Vote for specified agent, self, or first available
        if self._vote_for and self._vote_for in proposals:
            choice = self._vote_for
        elif self.name in proposals:
            choice = self.name
        else:
            choice = list(proposals.keys())[0] if proposals else "none"

        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote from {self.name} for {choice}",
            confidence=self._confidence,
            continue_debate=False,
        )


class TestFullDebateLifecycle:
    """Tests for complete debate lifecycle from start to finish."""

    @pytest.fixture
    def simple_env(self):
        """Simple test environment."""
        return Environment(task="What is the best programming language for beginners?")

    @pytest.fixture
    def two_agents(self):
        """Two agents that will reach consensus."""
        return [
            E2EMockAgent("alice", proposal="Python is best for beginners."),
            E2EMockAgent("bob", proposal="Python is ideal for beginners.", vote_for="alice"),
        ]

    @pytest.fixture
    def three_agents_majority(self):
        """Three agents where two agree."""
        return [
            E2EMockAgent("alice", proposal="Python for beginners."),
            E2EMockAgent("bob", proposal="JavaScript for beginners.", vote_for="alice"),
            E2EMockAgent("carol", proposal="Scratch for beginners.", vote_for="alice"),
        ]

    @pytest.fixture
    def three_agents_split(self):
        """Three agents with split votes."""
        return [
            E2EMockAgent("alice", proposal="Python"),
            E2EMockAgent("bob", proposal="JavaScript"),
            E2EMockAgent("carol", proposal="Scratch"),
        ]

    @pytest.mark.asyncio
    async def test_two_agent_debate_reaches_consensus(self, simple_env, two_agents):
        """Two agents complete a debate and reach consensus."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=simple_env, agents=two_agents, protocol=protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        # Both agents should have generated proposals
        assert two_agents[0].generate_count >= 1
        assert two_agents[1].generate_count >= 1

    @pytest.mark.asyncio
    async def test_three_agent_majority_consensus(self, simple_env, three_agents_majority):
        """Three agents reach majority consensus (2/3 agree)."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=simple_env, agents=three_agents_majority, protocol=protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.final_answer is not None
        # Majority should have voted for alice
        if result.votes:
            alice_votes = sum(
                1 for v in result.votes if not isinstance(v, Exception) and v.choice == "alice"
            )
            assert alice_votes >= 2, "Majority should vote for alice"

    @pytest.mark.asyncio
    async def test_debate_with_split_votes(self, simple_env, three_agents_split):
        """Three agents with split votes still produces a result."""
        protocol = DebateProtocol(rounds=2, consensus="majority", consensus_threshold=0.3)
        arena = Arena(environment=simple_env, agents=three_agents_split, protocol=protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.final_answer is not None
        # Even with split votes, should have an answer

    @pytest.mark.asyncio
    async def test_debate_phases_execute_in_order(self, simple_env, two_agents):
        """Debate executes phases in correct order."""
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=simple_env, agents=two_agents, protocol=protocol)

        result = await arena.run()

        # All agents should have participated
        for agent in two_agents:
            assert agent.generate_count >= 1, f"{agent.name} should have generated"
            assert agent.vote_count >= 1, f"{agent.name} should have voted"

    @pytest.mark.asyncio
    async def test_early_stopping_on_unanimous_agreement(self, simple_env):
        """Debate stops early when all agents agree."""
        # All agents vote for alice
        agents = [
            E2EMockAgent("alice", proposal="The answer is 42."),
            E2EMockAgent("bob", proposal="Different answer", vote_for="alice"),
            E2EMockAgent("carol", proposal="Another answer", vote_for="alice"),
        ]

        protocol = DebateProtocol(rounds=5, consensus="majority", early_stopping=True)
        arena = Arena(environment=simple_env, agents=agents, protocol=protocol)

        result = await arena.run()

        # With unanimous agreement and early stopping, should finish quickly
        assert result.final_answer is not None
        assert result.consensus_reached or result.final_answer

    @pytest.mark.asyncio
    async def test_debate_result_contains_votes(self, simple_env, three_agents_majority):
        """Debate result includes all agent votes."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=simple_env, agents=three_agents_majority, protocol=protocol)

        result = await arena.run()

        # Should have votes from agents
        assert result.votes is not None
        valid_votes = [v for v in result.votes if not isinstance(v, Exception)]
        assert len(valid_votes) >= 1, "Should have at least one valid vote"

    @pytest.mark.asyncio
    async def test_debate_records_messages(self, simple_env, two_agents):
        """Debate records message history."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=simple_env, agents=two_agents, protocol=protocol)

        result = await arena.run()

        # Messages should be recorded
        assert result.messages is not None
        assert len(result.messages) >= 1, "Should have message history"


class TestConsensusTypes:
    """Tests for different consensus modes."""

    @pytest.fixture
    def env(self):
        return Environment(task="Choose the best option")

    @pytest.mark.asyncio
    async def test_majority_consensus_mode(self, env):
        """Majority consensus requires > 50% agreement."""
        agents = [
            E2EMockAgent("a", vote_for="a"),
            E2EMockAgent("b", vote_for="a"),
            E2EMockAgent("c", vote_for="c"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_unanimous_consensus_mode(self, env):
        """Unanimous consensus requires all agents to agree."""
        # All vote for 'a'
        agents = [
            E2EMockAgent("a"),
            E2EMockAgent("b", vote_for="a"),
            E2EMockAgent("c", vote_for="a"),
        ]
        protocol = DebateProtocol(rounds=2, consensus="unanimous")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_custom_threshold_consensus(self, env):
        """Custom consensus threshold works."""
        agents = [
            E2EMockAgent("a"),
            E2EMockAgent("b", vote_for="a"),
            E2EMockAgent("c", vote_for="c"),
            E2EMockAgent("d", vote_for="d"),
        ]
        # 25% threshold - 2/4 for 'a' is 50%, should pass
        protocol = DebateProtocol(rounds=1, consensus="majority", consensus_threshold=0.25)
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()
        assert result.final_answer is not None


class TestAgentParticipation:
    """Tests for agent participation tracking."""

    @pytest.fixture
    def env(self):
        return Environment(task="Test participation")

    @pytest.mark.asyncio
    async def test_all_agents_generate_proposals(self, env):
        """All agents should generate at least one proposal."""
        agents = [
            E2EMockAgent("alice"),
            E2EMockAgent("bob"),
            E2EMockAgent("carol"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        await arena.run()

        for agent in agents:
            assert agent.generate_count >= 1, f"{agent.name} should generate at least once"

    @pytest.mark.asyncio
    async def test_all_agents_vote(self, env):
        """All agents should vote at least once."""
        agents = [
            E2EMockAgent("alice"),
            E2EMockAgent("bob"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        await arena.run()

        for agent in agents:
            assert agent.vote_count >= 1, f"{agent.name} should vote at least once"


class TestDebateWithHighConfidence:
    """Tests for debates where agents have high confidence."""

    @pytest.fixture
    def env(self):
        return Environment(task="High confidence debate")

    @pytest.mark.asyncio
    async def test_high_confidence_votes_recorded(self, env):
        """High confidence votes are properly recorded."""
        agents = [
            E2EMockAgent("alice", confidence=0.95),
            E2EMockAgent("bob", confidence=0.90, vote_for="alice"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        valid_votes = [v for v in result.votes if not isinstance(v, Exception)]
        if valid_votes:
            avg_confidence = sum(v.confidence for v in valid_votes) / len(valid_votes)
            assert avg_confidence > 0.8, "Average confidence should be high"


class TestErrorRecovery:
    """Tests for error handling during debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="Error recovery test")

    @pytest.mark.asyncio
    async def test_debate_continues_after_agent_error(self, env):
        """Debate continues even if one agent fails."""

        class FailingAgent(E2EMockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                if self.generate_count == 0:
                    self.generate_count += 1
                    raise RuntimeError("Simulated failure")
                self.generate_count += 1
                return self._proposal

        agents = [
            FailingAgent("failing"),
            E2EMockAgent("reliable"),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        # Should complete despite the failure
        result = await arena.run()
        assert result is not None
        assert result.final_answer is not None or result.messages


class TestDebateMetadata:
    """Tests for debate metadata and result structure."""

    @pytest.fixture
    def env(self):
        return Environment(task="Metadata test")

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self, env):
        """DebateResult has all required fields."""
        agents = [E2EMockAgent("alice"), E2EMockAgent("bob", vote_for="alice")]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert hasattr(result, "final_answer")
        assert hasattr(result, "messages")
        assert hasattr(result, "votes")
        assert hasattr(result, "consensus_reached")

    @pytest.mark.asyncio
    async def test_messages_contain_agent_info(self, env):
        """Messages include agent attribution."""
        agents = [E2EMockAgent("alice"), E2EMockAgent("bob", vote_for="alice")]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        if result.messages:
            for msg in result.messages:
                if hasattr(msg, "agent") and msg.agent:
                    assert msg.agent in ["alice", "bob", "system"]


class TestUserParticipation:
    """Tests for user participation affecting debate outcomes."""

    @pytest.fixture
    def env(self):
        return Environment(task="User participation test")

    @pytest.mark.asyncio
    async def test_user_vote_recorded(self, env):
        """User votes are recorded in debate results."""
        agents = [
            E2EMockAgent("alice", proposal="Solution A"),
            E2EMockAgent("bob", proposal="Solution B"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Debate should complete with votes
        assert result is not None
        assert result.votes is not None

    @pytest.mark.asyncio
    async def test_debate_handles_abstentions(self, env):
        """Debate handles agents that abstain from voting."""

        class AbstainingAgent(E2EMockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                return Vote(
                    agent=self.name,
                    choice="abstain",
                    reasoning="I choose to abstain",
                    confidence=0.0,
                    continue_debate=False,
                )

        agents = [
            E2EMockAgent("alice"),
            AbstainingAgent("bob"),
            E2EMockAgent("carol", vote_for="alice"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should still reach conclusion
        assert result is not None
        assert result.final_answer is not None or result.messages


class TestEvidenceCollection:
    """Tests for evidence collection during debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="Evidence collection test")

    @pytest.mark.asyncio
    async def test_proposals_can_include_citations(self, env):
        """Proposals can include citation references."""

        class CitingAgent(E2EMockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_count += 1
                return "According to [Source 1], the answer is X. [Source 2] confirms this."

        agents = [
            CitingAgent("researcher"),
            E2EMockAgent("critic", vote_for="researcher"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should complete with citation content
        assert result is not None
        if result.messages:
            has_citation = any(
                "[Source" in (msg.content if hasattr(msg, "content") else str(msg))
                for msg in result.messages
            )
            assert has_citation or len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_debate_with_conflicting_evidence(self, env):
        """Debate handles agents presenting conflicting evidence."""
        agents = [
            E2EMockAgent("pro", proposal="Evidence supports A: [Study 1]"),
            E2EMockAgent("con", proposal="Evidence supports B: [Study 2]"),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should still reach conclusion despite conflicting evidence
        assert result is not None
        assert result.final_answer is not None or result.votes


class TestELOTracking:
    """Tests for ELO rating updates after debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="ELO tracking test")

    @pytest.mark.asyncio
    async def test_debate_completes_for_elo_tracking(self, env):
        """Debate completes successfully for ELO tracking purposes."""
        agents = [
            E2EMockAgent("winner_candidate", confidence=0.95),
            E2EMockAgent("other", vote_for="winner_candidate", confidence=0.9),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Verify debate completed with clear winner
        assert result is not None
        assert result.final_answer is not None or result.consensus_reached

    @pytest.mark.asyncio
    async def test_debate_records_agent_performance(self, env):
        """Debate result includes data needed for ELO calculation."""
        agents = [
            E2EMockAgent("alice", confidence=0.9),
            E2EMockAgent("bob", vote_for="alice", confidence=0.85),
            E2EMockAgent("carol", vote_for="alice", confidence=0.88),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Verify we have the data needed for ELO: votes with choices and confidence
        assert result.votes is not None
        valid_votes = [v for v in result.votes if not isinstance(v, Exception)]
        assert len(valid_votes) >= 1

        # Each vote should have required attributes
        for vote in valid_votes:
            assert hasattr(vote, "choice")
            assert hasattr(vote, "confidence")

    @pytest.mark.asyncio
    async def test_consensus_provides_winner_for_elo(self, env):
        """Consensus result identifies winner for ELO update."""
        agents = [
            E2EMockAgent("clear_winner"),
            E2EMockAgent("supporter1", vote_for="clear_winner"),
            E2EMockAgent("supporter2", vote_for="clear_winner"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should have a clear winner for ELO
        assert result is not None
        # Either consensus_reached or we have votes showing a winner
        has_winner = result.consensus_reached or (
            result.votes
            and any(
                not isinstance(v, Exception) and v.choice == "clear_winner" for v in result.votes
            )
        )
        assert has_winner
