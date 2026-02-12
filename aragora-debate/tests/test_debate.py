"""Tests for the high-level Debate API (aragora_debate.debate)."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate import Debate, MockAgent, create_agent
from aragora_debate.debate import _AGENT_COUNTER
from aragora_debate.types import (
    Agent,
    ConsensusMethod,
    DebateConfig,
    DebateResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Debate class tests
# ---------------------------------------------------------------------------


class TestDebateCreation:
    def test_basic_creation(self):
        d = Debate(topic="Should we use Rust?")
        assert d.topic == "Should we use Rust?"
        assert d.agents == []

    def test_creation_with_options(self):
        d = Debate(
            topic="test",
            context="background info",
            rounds=5,
            consensus="supermajority",
            early_stopping=False,
        )
        assert d.context == "background info"
        assert d._config.rounds == 5
        assert d._config.consensus_method == ConsensusMethod.SUPERMAJORITY
        assert d._config.early_stopping is False

    def test_add_agent_returns_self(self):
        d = Debate(topic="test")
        result = d.add_agent(MockAgent("a"))
        assert result is d

    def test_add_agent_chaining(self):
        d = (
            Debate(topic="test")
            .add_agent(MockAgent("a"))
            .add_agent(MockAgent("b"))
        )
        assert len(d.agents) == 2

    def test_agents_property_is_copy(self):
        d = Debate(topic="test")
        d.add_agent(MockAgent("a"))
        agents = d.agents
        agents.append(MockAgent("b"))
        assert len(d.agents) == 1  # original not modified


class TestDebateRun:
    @pytest.mark.asyncio
    async def test_requires_two_agents(self):
        d = Debate(topic="test")
        d.add_agent(MockAgent("solo"))
        with pytest.raises(ValueError, match="at least 2"):
            await d.run()

    @pytest.mark.asyncio
    async def test_requires_any_agents(self):
        d = Debate(topic="test")
        with pytest.raises(ValueError, match="at least 2"):
            await d.run()

    @pytest.mark.asyncio
    async def test_basic_run(self):
        d = Debate(topic="Should we adopt TypeScript?", rounds=1)
        d.add_agent(MockAgent("pro", proposal="Yes, type safety is worth it", vote_for="pro"))
        d.add_agent(MockAgent("con", proposal="No, too much boilerplate", vote_for="pro"))
        result = await d.run()

        assert isinstance(result, DebateResult)
        assert result.task == "Should we adopt TypeScript?"
        assert result.rounds_used == 1
        assert result.receipt is not None
        assert result.receipt.signature is not None

    @pytest.mark.asyncio
    async def test_consensus_reached(self):
        d = Debate(topic="Pick a framework", rounds=2, consensus="majority")
        d.add_agent(MockAgent("a", proposal="React", vote_for="a"))
        d.add_agent(MockAgent("b", proposal="Vue", vote_for="a"))
        d.add_agent(MockAgent("c", proposal="Svelte", vote_for="a"))
        result = await d.run()

        assert result.consensus_reached is True
        assert result.rounds_used == 1  # early stopping

    @pytest.mark.asyncio
    async def test_no_consensus_with_split_votes(self):
        d = Debate(
            topic="Pick a DB",
            rounds=2,
            consensus="unanimous",
            early_stopping=False,
        )
        d.add_agent(MockAgent("a", vote_for="a"))
        d.add_agent(MockAgent("b", vote_for="b"))
        result = await d.run()

        assert result.consensus_reached is False
        assert result.rounds_used == 2

    @pytest.mark.asyncio
    async def test_with_context(self):
        d = Debate(
            topic="Architecture review",
            context="We have a monolith serving 1M users",
            rounds=1,
        )
        d.add_agent(MockAgent("a", vote_for="a"))
        d.add_agent(MockAgent("b", vote_for="a"))
        result = await d.run()
        assert result.status in ("consensus_reached", "completed")

    @pytest.mark.asyncio
    async def test_result_has_receipt_with_receipt_id(self):
        d = Debate(topic="test", rounds=1)
        d.add_agent(MockAgent("a", vote_for="a"))
        d.add_agent(MockAgent("b", vote_for="a"))
        result = await d.run()

        assert result.receipt is not None
        assert result.receipt.receipt_id.startswith("DR-")
        assert result.verdict is not None

    @pytest.mark.asyncio
    async def test_result_has_messages(self):
        d = Debate(topic="test", rounds=1)
        d.add_agent(MockAgent("a", vote_for="a"))
        d.add_agent(MockAgent("b", vote_for="a"))
        result = await d.run()

        assert len(result.messages) > 0
        roles = {m.role for m in result.messages}
        assert "proposer" in roles
        assert "critic" in roles
        assert "voter" in roles

    @pytest.mark.asyncio
    async def test_result_has_proposals(self):
        d = Debate(topic="test", rounds=1)
        d.add_agent(MockAgent("alpha", proposal="Plan Alpha"))
        d.add_agent(MockAgent("beta", proposal="Plan Beta"))
        result = await d.run()

        assert "alpha" in result.proposals
        assert "beta" in result.proposals
        assert "Plan Alpha" in result.proposals["alpha"]
        assert "Plan Beta" in result.proposals["beta"]

    @pytest.mark.asyncio
    async def test_supermajority_consensus(self):
        d = Debate(topic="test", rounds=1, consensus="supermajority")
        d.add_agent(MockAgent("a", vote_for="a"))
        d.add_agent(MockAgent("b", vote_for="a"))
        d.add_agent(MockAgent("c", vote_for="a"))
        result = await d.run()
        assert result.consensus_reached is True


# ---------------------------------------------------------------------------
# create_agent tests
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def setup_method(self):
        _AGENT_COUNTER.clear()

    def test_mock_agent(self):
        agent = create_agent("mock", name="test-mock")
        assert isinstance(agent, Agent)
        assert agent.name == "test-mock"
        assert agent.model == "mock"

    def test_mock_agent_auto_name(self):
        a1 = create_agent("mock")
        assert a1.name == "mock"
        a2 = create_agent("mock")
        assert a2.name == "mock-2"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_agent("nonexistent")

    def test_anthropic_creates_claude_agent(self):
        """Creating an anthropic agent either succeeds or raises ImportError/auth error."""
        try:
            agent = create_agent("anthropic", name="test", api_key="fake-key")
            assert agent.name == "test"
            assert agent.model == "claude-sonnet-4-5-20250929"
        except ImportError:
            pass  # SDK not installed -- expected in CI

    def test_openai_creates_openai_agent(self):
        """Creating an openai agent either succeeds or raises ImportError/auth error."""
        try:
            agent = create_agent("openai", name="test", api_key="fake-key")
            assert agent.name == "test"
            assert agent.model == "gpt-4o"
        except ImportError:
            pass  # SDK not installed -- expected in CI

    def test_claude_alias(self):
        """'claude' should map to anthropic provider."""
        try:
            agent = create_agent("claude", name="test", api_key="fake-key")
            assert agent.model == "claude-sonnet-4-5-20250929"
        except ImportError:
            pass

    def test_gpt_alias(self):
        """'gpt' should map to openai provider."""
        try:
            agent = create_agent("gpt", name="test", api_key="fake-key")
            assert agent.model == "gpt-4o"
        except ImportError:
            pass

    def test_mock_with_stance(self):
        agent = create_agent("mock", name="advocate", stance="affirmative")
        assert agent.stance == "affirmative"


# ---------------------------------------------------------------------------
# MockAgent tests
# ---------------------------------------------------------------------------


class TestMockAgent:
    @pytest.mark.asyncio
    async def test_generate(self):
        agent = MockAgent("test", proposal="My custom proposal")
        result = await agent.generate("some prompt")
        assert result == "My custom proposal"

    @pytest.mark.asyncio
    async def test_critique(self):
        agent = MockAgent("test")
        critique = await agent.critique(
            proposal="some proposal",
            task="some task",
            target_agent="other",
        )
        assert critique.agent == "test"
        assert critique.target_agent == "other"
        assert len(critique.issues) > 0

    @pytest.mark.asyncio
    async def test_vote_for_specific(self):
        agent = MockAgent("test", vote_for="winner")
        vote = await agent.vote(
            proposals={"winner": "plan A", "loser": "plan B"},
            task="pick one",
        )
        assert vote.choice == "winner"

    @pytest.mark.asyncio
    async def test_vote_default_first(self):
        agent = MockAgent("test")
        vote = await agent.vote(
            proposals={"first": "plan A", "second": "plan B"},
            task="pick one",
        )
        assert vote.choice == "first"

    @pytest.mark.asyncio
    async def test_custom_critique_issues(self):
        agent = MockAgent("test", critique_issues=["issue 1", "issue 2"])
        critique = await agent.critique("prop", "task")
        assert critique.issues == ["issue 1", "issue 2"]


# ---------------------------------------------------------------------------
# Integration: full debate flow with mock agents
# ---------------------------------------------------------------------------


class TestDebateIntegration:
    @pytest.mark.asyncio
    async def test_full_debate_flow(self):
        """End-to-end: create debate, add mocks, run, inspect receipt."""
        debate = Debate(
            topic="Should we open-source our internal tools?",
            context="We have 5 internal tools used by 200 engineers",
            rounds=2,
            consensus="majority",
        )
        debate.add_agent(
            MockAgent("advocate", proposal="Yes, open-source builds community", vote_for="advocate")
        )
        debate.add_agent(
            MockAgent("skeptic", proposal="No, competitive risk", vote_for="advocate")
        )
        debate.add_agent(
            MockAgent("pragmatist", proposal="Partial: open-source non-core tools", vote_for="advocate")
        )

        result = await debate.run()

        # Verify complete result
        assert result.consensus_reached is True
        assert result.receipt is not None
        assert result.receipt.question == "Should we open-source our internal tools?"
        assert len(result.receipt.agents) == 3
        assert result.receipt.rounds_used >= 1

        # Verify receipt exports
        md = result.receipt.to_markdown()
        assert "Decision Receipt" in md
        assert "open-source" in md

        json_str = result.receipt.to_dict()
        assert json_str["verdict"] in ("approved", "approved_with_conditions")

    @pytest.mark.asyncio
    async def test_debate_with_dissent(self):
        """Verify dissent is captured in receipt."""
        debate = Debate(topic="Rewrite in Rust?", rounds=1)
        debate.add_agent(MockAgent("for", proposal="Yes", vote_for="for"))
        debate.add_agent(MockAgent("against", proposal="No", vote_for="against"))
        debate.add_agent(MockAgent("neutral", proposal="Maybe", vote_for="for"))

        result = await debate.run()

        assert result.receipt is not None
        # against voted differently
        assert len(result.receipt.consensus.dissents) > 0 or len(result.receipt.consensus.dissenting_agents) > 0

    @pytest.mark.asyncio
    async def test_chained_api(self):
        """The fluent/chained API should work."""
        result = await (
            Debate(topic="Use tabs or spaces?", rounds=1)
            .add_agent(MockAgent("tabs", vote_for="tabs"))
            .add_agent(MockAgent("spaces", vote_for="tabs"))
        ).run()

        assert result.consensus_reached is True
        assert result.receipt is not None
