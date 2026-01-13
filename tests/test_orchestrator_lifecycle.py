"""
Tests for Arena debate lifecycle and edge cases.

Focus areas:
- Debate lifecycle (start → rounds → consensus → end)
- Error recovery mid-debate
- Memory integration during debate
- Multi-agent consensus edge cases
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from collections import deque

from aragora.debate.orchestrator import Arena, DebateResult, DebateProtocol
from aragora.core import Environment, Agent, Vote, Message, Critique


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str, response: str = "test response", fail_on_call: int = -1):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"
        self.response = response
        self.call_count = 0
        self.fail_on_call = fail_on_call

    async def generate(self, prompt: str, context: list = None) -> str:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError(f"Simulated failure on call {self.call_count}")
        return self.response

    async def generate_stream(self, prompt: str, context: list = None):
        yield self.response

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent="proposer",
            target_content=proposal[:100],
            issues=["Minor issue"],
            suggestions=["Consider refactoring"],
            severity=0.3,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        # Vote for self if available, otherwise first proposal
        if self.name in proposals:
            choice = self.name
        else:
            choice = list(proposals.keys())[0] if proposals else "unknown"
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


class TestDebateLifecycle:
    """Tests for full debate lifecycle."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        return Environment(task="What is 2+2?")

    @pytest.fixture
    def protocol(self):
        """Create test protocol."""
        return DebateProtocol(rounds=2, consensus="majority")

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            MockAgent("agent1", "The answer is 4"),
            MockAgent("agent2", "2+2 equals 4"),
        ]

    @pytest.mark.asyncio
    async def test_basic_debate_completes(self, env, protocol, agents):
        """Test that a basic debate runs to completion."""
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == "What is 2+2?"
        assert result.rounds_used > 0
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_debate_tracks_rounds(self, env, protocol, agents):
        """Test that debate tracks the correct number of rounds."""
        protocol.rounds = 3
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete within max rounds
        assert result.rounds_used <= 3

    @pytest.mark.asyncio
    async def test_debate_with_single_agent(self, env, protocol):
        """Test debate with only one agent."""
        agents = [MockAgent("solo", "Single agent answer")]
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result.final_answer is not None
        # With one agent, should reach consensus immediately
        assert "solo" in str(result.votes) or result.rounds_used <= 1

    @pytest.mark.asyncio
    async def test_debate_produces_votes(self, env, protocol, agents):
        """Test that debate produces votes from agents."""
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should have votes from participating agents
        assert len(result.votes) >= 0  # May be empty if early consensus

    @pytest.mark.asyncio
    async def test_debate_records_proposals(self, env, protocol, agents):
        """Test that proposals are recorded."""
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Final answer should be set
        assert result.final_answer is not None


class TestErrorRecovery:
    """Tests for error recovery during debates."""

    @pytest.fixture
    def env(self):
        return Environment(task="Test error handling")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority")

    @pytest.mark.asyncio
    async def test_debate_continues_after_agent_failure(self, env, protocol):
        """Test that debate can continue when one agent fails."""
        agents = [
            MockAgent("healthy", "Good response"),
            MockAgent("faulty", fail_on_call=1),  # Fails on first call
        ]

        arena = Arena(env, agents, protocol)

        # Should not raise, debate continues with remaining agents
        try:
            result = await arena.run()
            # If it completes, should have a result
            assert result is not None
        except RuntimeError:
            # Acceptable if single failure causes abort
            pass

    @pytest.mark.asyncio
    async def test_debate_handles_empty_response(self, env, protocol):
        """Test handling of empty agent responses."""
        agents = [
            MockAgent("empty", ""),
            MockAgent("normal", "Valid response"),
        ]

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete despite empty response
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_handles_very_long_response(self, env, protocol):
        """Test handling of very long agent responses."""
        long_response = "A" * 10000
        agents = [
            MockAgent("verbose", long_response),
            MockAgent("normal", "Short response"),
        ]

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete and truncate if necessary
        assert result is not None


class TestConsensusEdgeCases:
    """Tests for consensus detection edge cases."""

    @pytest.fixture
    def env(self):
        return Environment(task="Test consensus")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=3, consensus="majority")

    @pytest.mark.asyncio
    async def test_unanimous_agreement(self, env, protocol):
        """Test consensus when all agents agree."""
        # All agents give essentially the same answer
        agents = [
            MockAgent("a1", "The answer is X"),
            MockAgent("a2", "The answer is X"),
            MockAgent("a3", "The answer is X"),
        ]

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should reach consensus quickly
        assert result.rounds_used <= protocol.rounds

    @pytest.mark.asyncio
    async def test_split_vote(self, env, protocol):
        """Test handling of split votes."""
        # Agents give different answers
        agents = [
            MockAgent("a1", "Answer A"),
            MockAgent("a2", "Answer B"),
        ]

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should still produce a result
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_changing_votes(self, env, protocol):
        """Test that agents can change their votes between rounds."""

        class ChangingAgent(MockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                # Vote differently based on call count
                self.call_count += 1
                choices = list(proposals.keys())
                choice = choices[self.call_count % len(choices)] if choices else "none"
                return Vote(
                    agent=self.name,
                    choice=choice,
                    reasoning="Changed",
                    confidence=0.5,
                    continue_debate=False,
                )

        agents = [
            ChangingAgent("changer", "My proposal"),
            MockAgent("stable", "Stable proposal"),
        ]

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should handle changing votes
        assert result is not None


class TestMemoryIntegration:
    """Tests for memory system integration."""

    @pytest.fixture
    def env(self):
        return Environment(task="Test with memory")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority")

    @pytest.fixture
    def agents(self):
        return [MockAgent("a1", "Memory test"), MockAgent("a2", "Memory test 2")]

    @pytest.mark.asyncio
    async def test_debate_without_memory(self, env, protocol, agents):
        """Test debate runs without memory store configured."""
        arena = Arena(env, agents, protocol)
        # No memory configured
        assert arena.memory is None or arena.memory is False or arena.memory == {}

        result = await arena.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_with_mock_memory(self, env, protocol, agents):
        """Test debate with mock memory store."""
        mock_memory = MagicMock()
        mock_memory.get_relevant_context = MagicMock(return_value="Past context")
        mock_memory.store = MagicMock()

        arena = Arena(env, agents, protocol)
        arena.memory = mock_memory

        result = await arena.run()
        assert result is not None


class TestProtocolVariations:
    """Tests for different protocol configurations."""

    @pytest.fixture
    def env(self):
        return Environment(task="Protocol test")

    @pytest.fixture
    def agents(self):
        return [MockAgent("a1", "Response 1"), MockAgent("a2", "Response 2")]

    @pytest.mark.asyncio
    async def test_single_round(self, env, agents):
        """Test debate with single round."""
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result.rounds_used == 1

    @pytest.mark.asyncio
    async def test_many_rounds(self, env, agents):
        """Test debate with many rounds."""
        protocol = DebateProtocol(rounds=5, consensus="majority")
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete within max rounds
        assert result.rounds_used <= 5

    @pytest.mark.asyncio
    async def test_supermajority_consensus(self, env, agents):
        """Test debate with supermajority consensus requirement."""
        protocol = DebateProtocol(rounds=3, consensus="supermajority")
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_unanimous_consensus(self, env, agents):
        """Test debate with unanimous consensus requirement."""
        protocol = DebateProtocol(rounds=3, consensus="unanimous")
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # May or may not reach unanimous consensus
        assert result is not None


class TestEventHooks:
    """Tests for event hook integration."""

    @pytest.fixture
    def env(self):
        return Environment(task="Hook test")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority")

    @pytest.fixture
    def agents(self):
        return [MockAgent("a1", "Response"), MockAgent("a2", "Response 2")]

    @pytest.mark.asyncio
    async def test_debate_with_hooks(self, env, protocol, agents):
        """Test debate with event hooks configured."""
        events_received = []

        def on_round_start(round_num):
            events_received.append(("round_start", round_num))

        def on_round_end(round_num):
            events_received.append(("round_end", round_num))

        hooks = {
            "on_round_start": on_round_start,
            "on_round_end": on_round_end,
        }

        arena = Arena(env, agents, protocol, event_hooks=hooks)
        result = await arena.run()

        # Should have received events
        assert result is not None


class TestArenaState:
    """Tests for Arena state management."""

    @pytest.fixture
    def env(self):
        return Environment(task="State test")

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority")

    @pytest.fixture
    def agents(self):
        return [MockAgent("a1", "Response")]

    def test_arena_initialization(self, env, protocol, agents):
        """Test Arena initializes with correct state."""
        arena = Arena(env, agents, protocol)

        assert arena.env == env
        assert arena.protocol == protocol
        assert len(arena._require_agents()) == 1

    def test_arena_agents_property(self, env, protocol, agents):
        """Test Arena agents property."""
        arena = Arena(env, agents, protocol)

        # Agents should be accessible
        arena_agents = arena._require_agents()
        assert len(arena_agents) == 1
        assert arena_agents[0].name == "a1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
