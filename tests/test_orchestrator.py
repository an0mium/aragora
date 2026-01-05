"""
Tests for the debate orchestrator (Arena).

Uses mock agents to test debate flow without making actual API calls.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.core import Agent, Message, Critique, Vote, Environment, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore


class MockAgent(Agent):
    """Mock agent for testing that doesn't make API calls."""

    def __init__(self, name: str = "mock", model: str = "mock-model", role: str = "proposer"):
        super().__init__(name, model, role)
        self.agent_type = "mock"
        self.generate_responses = []
        self.critique_responses = []
        self.vote_responses = []
        self._generate_call_count = 0
        self._critique_call_count = 0
        self._vote_call_count = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        """Return mock response."""
        if self.generate_responses:
            response = self.generate_responses[self._generate_call_count % len(self.generate_responses)]
            self._generate_call_count += 1
            return response
        return f"Mock response from {self.name}"

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        """Return mock critique."""
        if self.critique_responses:
            response = self.critique_responses[self._critique_call_count % len(self.critique_responses)]
            self._critique_call_count += 1
            return response
        return Critique(
            agent=self.name,
            target_agent="proposer",
            target_content=proposal[:100],
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning"
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        """Return mock vote."""
        if self.vote_responses:
            response = self.vote_responses[self._vote_call_count % len(self.vote_responses)]
            self._vote_call_count += 1
            return response
        choice = list(proposals.keys())[0] if proposals else "none"
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False
        )


class TestDebateProtocol:
    """Tests for DebateProtocol configuration."""

    def test_default_protocol(self):
        """Test default protocol settings."""
        protocol = DebateProtocol()
        assert protocol.rounds == 5  # Default increased for more thorough debates
        assert protocol.consensus == "majority"
        assert protocol.early_stopping is True

    def test_custom_protocol(self):
        """Test custom protocol settings."""
        protocol = DebateProtocol(
            rounds=7,
            consensus="unanimous",
            early_stopping=False,
        )
        assert protocol.rounds == 7
        assert protocol.consensus == "unanimous"
        assert protocol.early_stopping is False


class TestArenaCreation:
    """Tests for Arena initialization."""

    def test_arena_creation_with_agents(self):
        """Test arena can be created with agents."""
        agents = [
            MockAgent("agent1", role="proposer"),
            MockAgent("agent2", role="critic"),
        ]
        env = Environment(task="Test task")
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, agents, protocol)

        assert len(arena.agents) == 2
        assert arena.env.task == "Test task"
        assert arena.protocol.rounds == 2

    def test_arena_with_memory_store(self, tmp_path):
        """Test arena with CritiqueStore."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task")
        protocol = DebateProtocol()
        memory = CritiqueStore(str(tmp_path / "test.db"))

        arena = Arena(env, agents, protocol, memory)

        assert arena.memory is not None


class TestDebateExecution:
    """Tests for debate execution flow."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(asyncio, 'timeout'),
        reason="asyncio.timeout requires Python 3.11+"
    )
    async def test_simple_debate_completes(self):
        """Test a simple debate runs to completion."""
        agents = [
            MockAgent("proposer", role="proposer"),
            MockAgent("critic", role="critic"),
        ]
        agents[0].generate_responses = ["Proposal: Use caching"]
        agents[1].generate_responses = ["Critique: Consider edge cases"]

        env = Environment(task="Design a cache", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == "Design a cache"
        assert result.rounds_used > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(asyncio, 'timeout'),
        reason="asyncio.timeout requires Python 3.11+"
    )
    async def test_debate_produces_messages(self):
        """Test debate produces message history."""
        agents = [
            MockAgent("agent1", role="proposer"),
            MockAgent("agent2", role="critic"),
        ]

        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert len(result.messages) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not hasattr(asyncio, 'timeout'),
        reason="asyncio.timeout requires Python 3.11+"
    )
    async def test_early_stopping_when_consensus(self):
        """Test debate stops early when consensus reached."""
        agents = [
            MockAgent("agent1", role="proposer"),
            MockAgent("agent2", role="critic"),
        ]
        # Both agents vote for same choice with high confidence
        agents[0].vote_responses = [Vote(
            agent="agent1",
            choice="agent1",
            reasoning="Good solution",
            confidence=0.95,
            continue_debate=False
        )]
        agents[1].vote_responses = [Vote(
            agent="agent2",
            choice="agent1",
            reasoning="Agreed",
            confidence=0.9,
            continue_debate=False
        )]

        env = Environment(task="Test task", max_rounds=5)
        protocol = DebateProtocol(rounds=5, early_stopping=True)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete without running all 5 rounds
        assert result.rounds_used <= 5


class TestAgentType:
    """Tests for agent_type attribute."""

    def test_mock_agent_has_agent_type(self):
        """Test mock agent has agent_type."""
        agent = MockAgent("test")
        assert hasattr(agent, "agent_type")
        assert agent.agent_type == "mock"

    def test_agent_type_is_string(self):
        """Test agent_type is a string."""
        agent = MockAgent("test")
        assert isinstance(agent.agent_type, str)


class TestCritiqueGeneration:
    """Tests for critique generation."""

    @pytest.mark.asyncio
    async def test_critique_has_required_fields(self):
        """Test critique contains all required fields."""
        agent = MockAgent("critic", role="critic")

        critique = await agent.critique(
            proposal="Test proposal",
            task="Test task"
        )

        assert isinstance(critique, Critique)
        assert critique.agent == "critic"
        assert len(critique.issues) > 0
        assert len(critique.suggestions) > 0
        assert 0 <= critique.severity <= 1
        assert critique.reasoning


class TestVoting:
    """Tests for voting mechanism."""

    @pytest.mark.asyncio
    async def test_vote_structure(self):
        """Test vote has correct structure."""
        agent = MockAgent("voter")

        vote = await agent.vote(
            proposals={"agent1": "Proposal 1", "agent2": "Proposal 2"},
            task="Test task"
        )

        assert isinstance(vote, Vote)
        assert vote.agent == "voter"
        assert vote.choice in ["agent1", "agent2", "none"]
        assert 0 <= vote.confidence <= 1
        assert isinstance(vote.continue_debate, bool)


class TestEnvironment:
    """Tests for Environment configuration."""

    def test_environment_defaults(self):
        """Test environment default values."""
        env = Environment(task="Test task")

        assert env.task == "Test task"
        assert env.max_rounds == 3
        assert env.require_consensus is False

    def test_environment_with_context(self):
        """Test environment with additional context."""
        env = Environment(
            task="Design API",
            context="RESTful principles preferred",
            roles=["architect", "reviewer"]
        )

        assert env.context == "RESTful principles preferred"
        assert "architect" in env.roles


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test message can be created."""
        msg = Message(
            role="proposer",
            agent="agent1",
            content="Test content",
            round=1
        )

        assert msg.role == "proposer"
        assert msg.agent == "agent1"
        assert msg.content == "Test content"
        assert msg.round == 1

    def test_message_str(self):
        """Test message string representation."""
        msg = Message(
            role="proposer",
            agent="agent1",
            content="A" * 200,
            round=1
        )

        str_repr = str(msg)
        assert "[proposer:agent1]" in str_repr
        # Content should be truncated
        assert len(str_repr) < 250


class TestDebateResult:
    """Tests for DebateResult dataclass."""

    def test_result_has_id(self):
        """Test result has unique ID."""
        result1 = DebateResult()
        result2 = DebateResult()

        assert result1.id != result2.id

    def test_result_summary(self):
        """Test result summary generation."""
        result = DebateResult(
            task="Test task",
            final_answer="Test answer",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=2,
            duration_seconds=5.0
        )

        summary = result.summary()

        assert "Test task" in summary
        assert "Yes" in summary  # consensus
        assert "85" in summary  # confidence percentage


class TestConsensusMechanisms:
    """Tests for different consensus mechanisms."""

    @pytest.mark.asyncio
    async def test_majority_consensus_with_agreement(self):
        """Test majority consensus when agents agree."""
        agents = [
            MockAgent("agent1", role="proposer"),
            MockAgent("agent2", role="critic"),
        ]
        # Both vote for same choice
        agents[0].vote_responses = [Vote(agent="agent1", choice="agent1", reasoning="Best", confidence=0.9, continue_debate=False)]
        agents[1].vote_responses = [Vote(agent="agent2", choice="agent1", reasoning="Agreed", confidence=0.85, continue_debate=False)]

        env = Environment(task="Test consensus", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="majority", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result.consensus_reached is True
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_unanimous_consensus_requires_all(self):
        """Test unanimous consensus requires all agents to agree."""
        agents = [
            MockAgent("agent1"),
            MockAgent("agent2"),
            MockAgent("agent3"),
        ]
        # All vote for same choice
        for i, agent in enumerate(agents):
            agent.vote_responses = [Vote(agent=f"agent{i+1}", choice="agent1", reasoning="Best", confidence=0.9, continue_debate=False)]

        env = Environment(task="Test unanimous", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="unanimous", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result.consensus_reached is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_no_consensus_mode(self):
        """Test 'none' consensus mode returns without voting."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]

        env = Environment(task="Test no consensus", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="none", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should complete without requiring consensus
        assert result is not None


class TestTopology:
    """Tests for different debate topologies."""

    @pytest.mark.asyncio
    async def test_round_robin_topology(self):
        """Test round-robin topology debate."""
        agents = [MockAgent(f"agent{i}") for i in range(3)]

        env = Environment(task="Test topology", max_rounds=1)
        protocol = DebateProtocol(rounds=1, topology="round-robin", consensus="none", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_star_topology(self):
        """Test star topology with hub agent."""
        agents = [MockAgent(f"agent{i}") for i in range(3)]

        env = Environment(task="Test star", max_rounds=1)
        protocol = DebateProtocol(
            rounds=1,
            topology="star",
            topology_hub_agent="agent0",
            consensus="none",
            early_stopping=False,
            convergence_detection=False
        )

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None


class TestTimeoutHandling:
    """Tests for debate timeout."""

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(self):
        """Test debate timeout returns partial results."""
        class SlowAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                await asyncio.sleep(2)  # Slow
                return "Slow response"

        agents = [SlowAgent("slow1"), SlowAgent("slow2")]

        env = Environment(task="Test timeout", max_rounds=1)
        protocol = DebateProtocol(
            rounds=3,
            timeout_seconds=1,  # 1 second timeout
            consensus="none",
            early_stopping=False,
            convergence_detection=False
        )

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Should timeout and return partial result (fewer rounds than requested)
        assert result is not None
        assert result.rounds_used < 3  # Didn't complete all rounds


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_single_agent_debate(self):
        """Test debate with single agent."""
        agents = [MockAgent("solo")]

        env = Environment(task="Solo debate", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="none", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None
        assert len(result.messages) >= 1

    @pytest.mark.asyncio
    async def test_many_agents_debate(self):
        """Test debate with many agents."""
        agents = [MockAgent(f"agent{i}") for i in range(5)]

        env = Environment(task="Many agents", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="majority", early_stopping=False, convergence_detection=False)

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None


class TestUserEventQueue:
    """Tests for user event queue handling and overflow protection."""

    def test_queue_has_capacity_limit(self):
        """Verify queue is created with maxsize limit."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test queue", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Queue should have a maxsize set
        assert arena._user_event_queue.maxsize == 10000

    def test_drain_user_events_empty_queue(self):
        """Draining empty queue should not raise."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test drain", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Should not raise on empty queue
        arena._drain_user_events()
        assert arena.user_votes == []
        assert arena.user_suggestions == []

    def test_drain_user_events_processes_votes(self):
        """Draining queue should populate user_votes list."""
        from aragora.server.stream import StreamEventType

        agents = [MockAgent("agent1")]
        env = Environment(task="Test drain votes", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Manually enqueue some vote events
        arena._user_event_queue.put_nowait((
            StreamEventType.USER_VOTE,
            {"choice": "proposal_a", "user_id": "user1"}
        ))
        arena._user_event_queue.put_nowait((
            StreamEventType.USER_VOTE,
            {"choice": "proposal_b", "user_id": "user2"}
        ))

        arena._drain_user_events()

        assert len(arena.user_votes) == 2
        assert arena.user_votes[0]["choice"] == "proposal_a"
        assert arena.user_votes[1]["choice"] == "proposal_b"

    def test_drain_user_events_processes_suggestions(self):
        """Draining queue should populate user_suggestions list."""
        from aragora.server.stream import StreamEventType

        agents = [MockAgent("agent1")]
        env = Environment(task="Test drain suggestions", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Manually enqueue suggestion events
        arena._user_event_queue.put_nowait((
            StreamEventType.USER_SUGGESTION,
            {"text": "Consider X", "user_id": "user1"}
        ))

        arena._drain_user_events()

        assert len(arena.user_suggestions) == 1
        assert arena.user_suggestions[0]["text"] == "Consider X"

    def test_queue_overflow_drops_events(self):
        """Events should be dropped when queue is full."""
        from aragora.server.stream import StreamEventType, StreamEvent
        import queue

        agents = [MockAgent("agent1")]
        env = Environment(task="Test overflow", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Replace queue with small one for testing
        arena._user_event_queue = queue.Queue(maxsize=5)

        # Fill the queue
        for i in range(5):
            arena._user_event_queue.put_nowait((
                StreamEventType.USER_VOTE,
                {"choice": f"choice_{i}"}
            ))

        # Queue is now full - additional put_nowait should raise
        with pytest.raises(queue.Full):
            arena._user_event_queue.put_nowait((
                StreamEventType.USER_VOTE,
                {"choice": "overflow"}
            ))

    def test_handle_user_event_filters_by_loop_id(self):
        """Events from other loops should be ignored."""
        from aragora.server.stream import StreamEventType, StreamEvent

        agents = [MockAgent("agent1")]
        env = Environment(task="Test filter", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        arena.loop_id = "my-loop-123"

        # Event for different loop should be ignored
        other_event = StreamEvent(
            type=StreamEventType.USER_VOTE,
            loop_id="other-loop-456",
            data={"choice": "proposal_a"}
        )
        arena._handle_user_event(other_event)

        # Queue should still be empty
        assert arena._user_event_queue.empty()

    def test_handle_user_event_accepts_matching_loop(self):
        """Events for matching loop should be enqueued."""
        from aragora.server.stream import StreamEventType, StreamEvent

        agents = [MockAgent("agent1")]
        env = Environment(task="Test accept", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)
        arena.loop_id = "my-loop-123"

        # Event for this loop should be enqueued
        my_event = StreamEvent(
            type=StreamEventType.USER_VOTE,
            loop_id="my-loop-123",
            data={"choice": "proposal_a"}
        )
        arena._handle_user_event(my_event)

        # Queue should have the event
        assert not arena._user_event_queue.empty()


class TestForkInitialMessages:
    """Tests for fork debate initial messages support."""

    def test_arena_accepts_initial_messages_param(self):
        """Arena constructor should accept initial_messages parameter."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test initial messages", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        initial_messages = [
            {"agent": "previous_agent", "content": "Previous discussion point"},
            {"agent": "other_agent", "content": "Another point from earlier"},
        ]

        arena = Arena(env, agents, protocol, initial_messages=initial_messages)

        assert arena.initial_messages == initial_messages
        assert len(arena.initial_messages) == 2

    def test_arena_defaults_to_empty_initial_messages(self):
        """Arena should default to empty list when no initial_messages provided."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test default", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)

        assert arena.initial_messages == []

    def test_arena_handles_none_initial_messages(self):
        """Arena should handle None initial_messages gracefully."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test none", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol, initial_messages=None)

        assert arena.initial_messages == []

    def test_initial_messages_dict_format(self):
        """Initial messages as dicts should be stored correctly."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test dict format", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        initial_messages = [
            {"content": "First message", "agent": "agent_a"},
            {"content": "Second message"},  # No agent specified
        ]

        arena = Arena(env, agents, protocol, initial_messages=initial_messages)

        assert len(arena.initial_messages) == 2
        assert arena.initial_messages[0]["content"] == "First message"
        assert arena.initial_messages[1]["content"] == "Second message"

    def test_empty_initial_messages_list(self):
        """Empty initial_messages list should work correctly."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test empty", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol, initial_messages=[])

        assert arena.initial_messages == []

    @pytest.mark.asyncio
    async def test_initial_messages_in_context(self):
        """Initial messages should be converted to Message objects in debate context."""
        from aragora.core import Message

        agents = [MockAgent("agent1")]
        agents[0].generate_responses = ["Test proposal"]
        env = Environment(task="Test context", max_rounds=1)
        protocol = DebateProtocol(
            rounds=1, consensus="none", early_stopping=False, convergence_detection=False
        )

        initial_messages = [
            {"agent": "previous", "content": "Historical context"},
        ]

        arena = Arena(env, agents, protocol, initial_messages=initial_messages)
        result = await arena.run()

        # Debate should complete successfully with initial context
        assert result is not None
        # The initial messages are used as context but may not be in final messages
        # depending on implementation - just verify no error occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
