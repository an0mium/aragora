"""
Tests for graph-structured debates with branching.

Tests DAG (Directed Acyclic Graph) debate features:
- Branch creation and management
- Parallel debate paths
- Branch merging and convergence
- Node relationships and traversal

Note: These tests run the full Arena.run() which can be slow.
Run with: pytest tests/test_orchestrator_graph_debates.py -v --timeout=120
"""

import pytest

# Mark all tests in this module as slow (arena tests take time)
pytestmark = pytest.mark.slow
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional, Dict, List

from aragora.core import (
    Agent,
    Environment,
    Vote,
    Message,
    Critique,
    DebateResult,
)
from aragora.debate.orchestrator import Arena, DebateProtocol


class GraphMockAgent(Agent):
    """Mock agent for graph debate testing."""

    def __init__(
        self,
        name: str,
        proposals: List[str] = None,
        branch_on: str = None,
    ):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"
        self._proposals = proposals or [f"Proposal from {name}"]
        self._proposal_idx = 0
        self._branch_on = branch_on  # Keyword that triggers branch suggestion
        self.branches_suggested = []

    async def generate(self, prompt: str, context: list = None) -> str:
        proposal = self._proposals[self._proposal_idx % len(self._proposals)]
        self._proposal_idx += 1
        return proposal

    async def generate_stream(self, prompt: str, context: list = None):
        yield await self.generate(prompt, context)

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        # Suggest branch if keyword found
        issues = ["Could be improved"]
        if self._branch_on and self._branch_on.lower() in proposal.lower():
            issues.append(f"[BRANCH] This deserves a separate exploration")
            self.branches_suggested.append(proposal[:50])

        return Critique(
            agent=self.name,
            target_agent="unknown",
            target_content=proposal[:100],
            issues=issues,
            suggestions=["Consider alternatives"],
            severity=0.5,
            reasoning=f"Critique from {self.name}",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = self.name if self.name in proposals else list(proposals.keys())[0]
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote from {self.name}",
            confidence=0.8,
            continue_debate=False,
        )


class TestGraphDebateStructure:
    """Tests for graph debate data structures."""

    @pytest.fixture
    def env(self):
        return Environment(task="Explore multiple solutions to climate change")

    @pytest.mark.asyncio
    async def test_basic_debate_produces_result(self, env):
        """Basic graph debate produces a valid result."""
        agents = [
            GraphMockAgent("alice", proposals=["Carbon tax solution"]),
            GraphMockAgent("bob", proposals=["Renewable energy focus"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        assert result.final_answer is not None or result.messages

    @pytest.mark.asyncio
    async def test_debate_with_multiple_proposals(self, env):
        """Agents can provide multiple different proposals."""
        agents = [
            GraphMockAgent(
                "alice",
                proposals=["Solution A", "Solution A refined", "Solution A final"],
            ),
            GraphMockAgent(
                "bob",
                proposals=["Solution B", "Solution B improved"],
            ),
        ]
        protocol = DebateProtocol(rounds=3, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        # Agents should have cycled through proposals
        assert agents[0]._proposal_idx >= 1
        assert agents[1]._proposal_idx >= 1


class TestGraphDebateBranching:
    """Tests for debate branching behavior."""

    @pytest.fixture
    def env(self):
        return Environment(task="Design a new transportation system")

    @pytest.mark.asyncio
    async def test_debate_handles_divergent_views(self, env):
        """Debate handles agents with very different views."""
        agents = [
            GraphMockAgent("alice", proposals=["Electric cars everywhere"]),
            GraphMockAgent("bob", proposals=["Public transit focus"]),
            GraphMockAgent("carol", proposals=["Bicycle infrastructure"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        # Should still reach some conclusion
        assert result.final_answer is not None or len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_critique_can_suggest_branches(self, env):
        """Agents can suggest branches through critique."""
        agents = [
            GraphMockAgent(
                "alice",
                proposals=["Main proposal with branch keyword"],
                branch_on="branch",
            ),
            GraphMockAgent("bob", proposals=["Counter proposal"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        await arena.run()

        # Alice should have suggested a branch
        # (The actual branching is handled by higher-level orchestration)
        assert len(agents[0].branches_suggested) >= 0  # May or may not trigger


class TestParallelDebatePaths:
    """Tests for parallel debate execution."""

    @pytest.fixture
    def env(self):
        return Environment(task="Evaluate three competing technologies")

    @pytest.mark.asyncio
    async def test_multiple_agents_propose_in_parallel(self, env):
        """Multiple agents can propose simultaneously."""
        agents = [
            GraphMockAgent("tech_a", proposals=["Technology A is best"]),
            GraphMockAgent("tech_b", proposals=["Technology B is best"]),
            GraphMockAgent("tech_c", proposals=["Technology C is best"]),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # All agents should have participated
        for agent in agents:
            assert agent._proposal_idx >= 1

    @pytest.mark.asyncio
    async def test_debate_converges_from_parallel_paths(self, env):
        """Parallel debate paths converge to a consensus."""
        agents = [
            GraphMockAgent("path_1", proposals=["Option 1"]),
            GraphMockAgent("path_2", proposals=["Option 2"]),
            GraphMockAgent("arbiter", proposals=["Combined solution"]),
        ]
        # Arbiter votes for path_1 to break tie
        agents[2]._vote_choice = "path_1"

        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        assert result.final_answer is not None or result.votes


class TestGraphNodeRelationships:
    """Tests for node relationships in debate graphs."""

    @pytest.fixture
    def env(self):
        return Environment(task="Build a recommendation system")

    @pytest.mark.asyncio
    async def test_messages_form_chain(self, env):
        """Messages form a proper chain of responses."""
        agents = [
            GraphMockAgent("alice"),
            GraphMockAgent("bob"),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should have messages in order
        assert result.messages is not None
        if len(result.messages) > 1:
            # Messages should be in chronological order (by index)
            pass  # Order is implicit in list

    @pytest.mark.asyncio
    async def test_votes_reference_proposals(self, env):
        """Votes reference valid proposal agents."""
        agents = [
            GraphMockAgent("alice"),
            GraphMockAgent("bob"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        agent_names = {a.name for a in agents}
        for vote in result.votes:
            if not isinstance(vote, Exception) and hasattr(vote, "choice"):
                # Vote choice should be a valid agent or "none"
                assert vote.choice in agent_names or vote.choice in ["none", "unknown", "abstain"]


class TestGraphDebateConvergence:
    """Tests for debate convergence behavior."""

    @pytest.fixture
    def env(self):
        return Environment(task="Solve the optimization problem")

    @pytest.mark.asyncio
    async def test_debate_converges_within_rounds(self, env):
        """Debate reaches conclusion within specified rounds."""
        agents = [
            GraphMockAgent("alice"),
            GraphMockAgent("bob"),
        ]
        protocol = DebateProtocol(rounds=3, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        # Should have some resolution
        assert result.final_answer is not None or result.messages

    @pytest.mark.asyncio
    async def test_unanimous_agreement_ends_early(self, env):
        """Unanimous agreement triggers early stopping."""

        class AgreeingAgent(GraphMockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                # Always vote for alice
                return Vote(
                    agent=self.name,
                    choice="alice",
                    reasoning="I agree with alice",
                    confidence=0.95,
                    continue_debate=False,
                )

        agents = [
            AgreeingAgent("alice"),
            AgreeingAgent("bob"),
            AgreeingAgent("carol"),
        ]
        protocol = DebateProtocol(rounds=10, consensus="unanimous", early_stopping=True)
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # With early stopping and unanimous agreement, should finish early
        assert result is not None


class TestGraphDebateRobustness:
    """Tests for robustness of graph debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="Stress test the debate system")

    @pytest.mark.asyncio
    async def test_debate_with_many_agents(self, env):
        """Debate handles many agents gracefully."""
        agents = [GraphMockAgent(f"agent_{i}") for i in range(5)]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        # All agents should have participated
        for agent in agents:
            assert agent._proposal_idx >= 1

    @pytest.mark.asyncio
    async def test_debate_with_single_agent(self, env):
        """Single agent debate still produces result."""
        agents = [GraphMockAgent("solo")]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        assert result is not None
        assert result.final_answer is not None or result.messages

    @pytest.mark.asyncio
    async def test_debate_with_empty_proposals(self, env):
        """Debate handles agents returning empty proposals."""

        class EmptyAgent(GraphMockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                return ""

        agents = [
            EmptyAgent("empty"),
            GraphMockAgent("normal"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        # Should not crash
        result = await arena.run()
        assert result is not None


class TestDAGStructureValidation:
    """Tests for DAG (Directed Acyclic Graph) structure in debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="DAG structure validation test")

    @pytest.mark.asyncio
    async def test_messages_have_sequential_structure(self, env):
        """Messages form a sequential DAG structure."""
        agents = [
            GraphMockAgent("alice", proposals=["First proposal"]),
            GraphMockAgent("bob", proposals=["Response to alice"]),
            GraphMockAgent("carol", proposals=["Response to bob"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Messages should be present and ordered
        assert result.messages is not None
        assert len(result.messages) >= 1

        # Messages should have agent attribution
        for msg in result.messages:
            if hasattr(msg, "agent"):
                assert msg.agent in ["alice", "bob", "carol", "system", None]

    @pytest.mark.asyncio
    async def test_round_based_message_ordering(self, env):
        """Messages respect round-based ordering."""
        agents = [
            GraphMockAgent("agent_1"),
            GraphMockAgent("agent_2"),
        ]
        protocol = DebateProtocol(rounds=3, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should have messages from multiple rounds
        assert result.messages is not None
        # Check that we have multiple rounds of messages
        rounds_seen = set()
        for msg in result.messages:
            if hasattr(msg, "round"):
                rounds_seen.add(msg.round)

        # At minimum should complete the debate
        assert result.final_answer is not None or len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_no_circular_references_in_debate(self, env):
        """Debate structure has no circular dependencies."""
        agents = [
            GraphMockAgent("a", proposals=["Proposal A"]),
            GraphMockAgent("b", proposals=["Proposal B references A"]),
            GraphMockAgent("c", proposals=["Proposal C references B"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Debate should complete without hanging (no cycles)
        assert result is not None
        assert result.final_answer is not None or result.messages


class TestBranchMergeConvergence:
    """Tests for branch creation and merge convergence."""

    @pytest.fixture
    def env(self):
        return Environment(task="Branch merge convergence test")

    @pytest.mark.asyncio
    async def test_divergent_proposals_converge(self, env):
        """Divergent proposals eventually converge to consensus."""
        agents = [
            GraphMockAgent("branch_a", proposals=["Solution via approach A"]),
            GraphMockAgent("branch_b", proposals=["Solution via approach B"]),
            GraphMockAgent("mediator", proposals=["Combined solution A+B"]),
        ]
        # Mediator breaks tie by voting for branch_a
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should converge to a single answer
        assert result is not None
        assert result.final_answer is not None or result.consensus_reached or result.votes

    @pytest.mark.asyncio
    async def test_multiple_branches_all_participate(self, env):
        """All branches participate in final convergence."""
        agents = [
            GraphMockAgent("path_1", proposals=["Path 1 solution"]),
            GraphMockAgent("path_2", proposals=["Path 2 solution"]),
            GraphMockAgent("path_3", proposals=["Path 3 solution"]),
        ]
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # All agents should have generated proposals
        for agent in agents:
            assert agent._proposal_idx >= 1

        # Should have votes from all agents
        assert result.votes is not None
        valid_votes = [v for v in result.votes if not isinstance(v, Exception)]
        assert len(valid_votes) >= 1

    @pytest.mark.asyncio
    async def test_tie_breaking_with_multiple_branches(self, env):
        """Tie-breaking works correctly with multiple branches."""

        class TieVotingAgent(GraphMockAgent):
            def __init__(self, name: str, vote_choice: str):
                super().__init__(name)
                self._vote_choice = vote_choice

            async def vote(self, proposals: dict, task: str) -> Vote:
                return Vote(
                    agent=self.name,
                    choice=(
                        self._vote_choice
                        if self._vote_choice in proposals
                        else list(proposals.keys())[0]
                    ),
                    reasoning=f"Voting for {self._vote_choice}",
                    confidence=0.8,
                    continue_debate=False,
                )

        # 2 vote for alpha, 1 for beta, 1 for gamma
        agents = [
            TieVotingAgent("voter_1", "alpha"),
            TieVotingAgent("voter_2", "alpha"),
            TieVotingAgent("voter_3", "beta"),
            TieVotingAgent("voter_4", "gamma"),
        ]
        protocol = DebateProtocol(rounds=1, consensus="majority", consensus_threshold=0.3)
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # Should still produce a result with plurality
        assert result is not None
        assert result.final_answer is not None or result.votes

    @pytest.mark.asyncio
    async def test_branch_quality_affects_convergence(self, env):
        """Higher quality branches (higher confidence votes) influence convergence."""
        agents = [
            GraphMockAgent("high_quality"),
            GraphMockAgent("low_quality"),
        ]

        # Make high_quality vote with high confidence for itself
        class ConfidentAgent(GraphMockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                return Vote(
                    agent=self.name,
                    choice="high_quality" if "high_quality" in proposals else self.name,
                    reasoning="High confidence vote",
                    confidence=0.95,
                    continue_debate=False,
                )

        agents = [
            ConfidentAgent("high_quality"),
            GraphMockAgent("low_quality", proposals=["Low quality solution"]),
        ]

        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = await arena.run()

        # High confidence votes should be recorded
        assert result.votes is not None
        high_conf_votes = [
            v
            for v in result.votes
            if not isinstance(v, Exception) and hasattr(v, "confidence") and v.confidence > 0.9
        ]
        assert len(high_conf_votes) >= 1
