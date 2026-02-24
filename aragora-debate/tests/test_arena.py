"""Tests for aragora_debate.arena."""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.types import (
    Agent,
    Consensus,
    ConsensusMethod,
    Critique,
    DebateConfig,
    Message,
    Vote,
)
from aragora_debate.arena import Arena


class MockAgent(Agent):
    """A deterministic mock agent for testing."""

    def __init__(self, name: str, proposal: str = "My proposal", vote_for: str = ""):
        super().__init__(name=name, model="mock")
        self._proposal = proposal
        self._vote_for = vote_for

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        return self._proposal

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=["needs more evidence"],
            suggestions=["add data sources"],
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        choice = self._vote_for or list(proposals.keys())[0]
        return Vote(agent=self.name, choice=choice, confidence=0.9, reasoning="seems best")


class TestArenaCreation:
    def test_requires_two_agents(self):
        with pytest.raises(ValueError, match="at least 2"):
            Arena(question="test", agents=[MockAgent("solo")])

    def test_default_config(self):
        arena = Arena(
            question="test",
            agents=[MockAgent("a"), MockAgent("b")],
        )
        assert arena.config is not None
        assert arena.config.rounds == 3

    def test_custom_config(self):
        cfg = DebateConfig(rounds=5, consensus_method=ConsensusMethod.UNANIMOUS)
        arena = Arena(
            question="test",
            agents=[MockAgent("a"), MockAgent("b")],
            config=cfg,
        )
        assert arena.config.rounds == 5


class TestArenaRun:
    @pytest.mark.asyncio
    async def test_basic_debate(self):
        agents = [
            MockAgent("claude", proposal="Use microservices", vote_for="claude"),
            MockAgent("gpt4", proposal="Use monolith", vote_for="claude"),
        ]
        arena = Arena(question="Architecture choice?", agents=agents)
        result = await arena.run()

        assert result.task == "Architecture choice?"
        assert result.rounds_used > 0
        assert len(result.participants) == 2
        assert result.receipt is not None
        assert result.verdict is not None

    @pytest.mark.asyncio
    async def test_consensus_reached(self):
        """All agents vote for the same agent -> consensus."""
        agents = [
            MockAgent("a", proposal="Plan A", vote_for="a"),
            MockAgent("b", proposal="Plan B", vote_for="a"),
            MockAgent("c", proposal="Plan C", vote_for="a"),
        ]
        arena = Arena(
            question="Pick a plan",
            agents=agents,
            config=DebateConfig(rounds=2, early_stopping=True, min_rounds=1),
        )
        result = await arena.run()

        assert result.consensus_reached is True
        assert result.confidence > 0.5
        assert result.rounds_used == 1  # early stopping

    @pytest.mark.asyncio
    async def test_no_consensus(self):
        """Each agent votes for itself -> no consensus with supermajority."""
        agents = [
            MockAgent("a", proposal="Plan A", vote_for="a"),
            MockAgent("b", proposal="Plan B", vote_for="b"),
            MockAgent("c", proposal="Plan C", vote_for="c"),
        ]
        arena = Arena(
            question="Pick a plan",
            agents=agents,
            config=DebateConfig(
                rounds=2,
                consensus_method=ConsensusMethod.SUPERMAJORITY,
                early_stopping=False,
            ),
        )
        result = await arena.run()

        assert result.consensus_reached is False
        assert result.rounds_used == 2

    @pytest.mark.asyncio
    async def test_receipt_has_signature(self):
        agents = [
            MockAgent("a", vote_for="a"),
            MockAgent("b", vote_for="a"),
        ]
        arena = Arena(question="test", agents=agents, config=DebateConfig(rounds=1))
        result = await arena.run()

        assert result.receipt is not None
        assert result.receipt.signature is not None
        assert result.receipt.signature_algorithm == "SHA-256-content-hash"

    @pytest.mark.asyncio
    async def test_messages_accumulated(self):
        agents = [
            MockAgent("a", vote_for="a"),
            MockAgent("b", vote_for="a"),
        ]
        arena = Arena(question="test", agents=agents, config=DebateConfig(rounds=1))
        result = await arena.run()

        # Should have proposal + critique + vote messages
        assert len(result.messages) > 0
        roles = {m.role for m in result.messages}
        assert "proposer" in roles
        assert "critic" in roles
        assert "voter" in roles

    @pytest.mark.asyncio
    async def test_context_passed_to_agents(self):
        agents = [
            MockAgent("a", vote_for="a"),
            MockAgent("b", vote_for="a"),
        ]
        arena = Arena(
            question="test",
            agents=agents,
            context="Important background info",
            config=DebateConfig(rounds=1),
        )
        result = await arena.run()
        assert result.status in ("consensus_reached", "completed")


class TestConsensusEvaluation:
    def _make_arena(self):
        return Arena(
            question="test",
            agents=[MockAgent("a"), MockAgent("b"), MockAgent("c")],
        )

    def test_majority(self):
        arena = self._make_arena()
        votes = [
            Vote(agent="a", choice="plan_a", confidence=1.0),
            Vote(agent="b", choice="plan_a", confidence=1.0),
            Vote(agent="c", choice="plan_b", confidence=1.0),
        ]
        result = arena._evaluate_consensus(votes, ["a", "b", "c"])
        assert result.reached is True
        assert "a" in result.supporting_agents
        assert "b" in result.supporting_agents

    def test_no_votes(self):
        arena = self._make_arena()
        result = arena._evaluate_consensus([], ["a", "b", "c"])
        assert result.reached is False
        assert result.confidence == 0.0

    def test_weighted_confidence(self):
        arena = self._make_arena()
        arena.config = DebateConfig(
            consensus_method=ConsensusMethod.WEIGHTED, consensus_threshold=0.7
        )
        votes = [
            Vote(agent="a", choice="plan_a", confidence=0.9),
            Vote(agent="b", choice="plan_a", confidence=0.8),
            Vote(agent="c", choice="plan_b", confidence=0.3),
        ]
        result = arena._evaluate_consensus(votes, ["a", "b", "c"])
        assert result.reached is True
        assert result.confidence > 0.7
