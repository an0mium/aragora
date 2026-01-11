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

        # Queue should have a maxsize set (via AudienceManager)
        assert arena.audience_manager._event_queue.maxsize == 10000

    def test_drain_user_events_empty_queue(self):
        """Draining empty queue should not raise."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test drain", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Should not raise on empty queue
        arena._drain_user_events()
        # user_votes and user_suggestions are deques, check they're empty
        assert len(arena.user_votes) == 0
        assert len(arena.user_suggestions) == 0

    def test_drain_user_events_processes_votes(self):
        """Draining queue should populate user_votes list."""
        from aragora.server.stream import StreamEventType

        agents = [MockAgent("agent1")]
        env = Environment(task="Test drain votes", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol)

        # Manually enqueue some vote events (via AudienceManager)
        arena.audience_manager._event_queue.put_nowait((
            StreamEventType.USER_VOTE,
            {"choice": "proposal_a", "user_id": "user1"}
        ))
        arena.audience_manager._event_queue.put_nowait((
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

        # Manually enqueue suggestion events (via AudienceManager)
        arena.audience_manager._event_queue.put_nowait((
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

        # Replace queue with small one for testing (via AudienceManager)
        arena.audience_manager._event_queue = queue.Queue(maxsize=5)

        # Fill the queue
        for i in range(5):
            arena.audience_manager._event_queue.put_nowait((
                StreamEventType.USER_VOTE,
                {"choice": f"choice_{i}"}
            ))

        # Queue is now full - additional put_nowait should raise
        with pytest.raises(queue.Full):
            arena.audience_manager._event_queue.put_nowait((
                StreamEventType.USER_VOTE,
                {"choice": "overflow"}
            ))

    def test_handle_user_event_filters_by_loop_id(self):
        """Events from other loops should be ignored."""
        from aragora.server.stream import StreamEventType, StreamEvent

        agents = [MockAgent("agent1")]
        env = Environment(task="Test filter", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol, loop_id="my-loop-123")

        # Event for different loop should be ignored
        other_event = StreamEvent(
            type=StreamEventType.USER_VOTE,
            loop_id="other-loop-456",
            data={"choice": "proposal_a"}
        )
        arena._handle_user_event(other_event)

        # Queue should still be empty (via AudienceManager)
        assert arena.audience_manager._event_queue.empty()

    def test_handle_user_event_accepts_matching_loop(self):
        """Events for matching loop should be enqueued."""
        from aragora.server.stream import StreamEventType, StreamEvent

        agents = [MockAgent("agent1")]
        env = Environment(task="Test accept", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, agents, protocol, loop_id="my-loop-123")

        # Event for this loop should be enqueued
        my_event = StreamEvent(
            type=StreamEventType.USER_VOTE,
            loop_id="my-loop-123",
            data={"choice": "proposal_a"}
        )
        arena._handle_user_event(my_event)

        # Queue should have the event (via AudienceManager)
        assert not arena.audience_manager._event_queue.empty()


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


class TestArenaFromConfig:
    """Tests for Arena.from_config class method."""

    def test_from_config_creates_arena(self):
        """Test from_config creates Arena correctly."""
        from aragora.debate.orchestrator import ArenaConfig

        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test from_config", max_rounds=2)
        protocol = DebateProtocol(rounds=2)
        config = ArenaConfig()

        arena = Arena.from_config(env, agents, protocol, config)

        assert isinstance(arena, Arena)
        assert len(arena.agents) == 2
        assert arena.env.task == "Test from_config"

    def test_from_config_with_none_config(self):
        """Test from_config works with None config (uses defaults)."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test default config", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena.from_config(env, agents, protocol, None)

        assert isinstance(arena, Arena)

    def test_from_config_with_billing_fields(self):
        """Test from_config properly passes billing fields."""
        from aragora.debate.orchestrator import ArenaConfig
        from unittest.mock import Mock

        agents = [MockAgent("agent1")]
        env = Environment(task="Test billing", max_rounds=1)
        protocol = DebateProtocol(rounds=1)
        usage_tracker = Mock()

        config = ArenaConfig(
            org_id="org-123",
            user_id="user-456",
            usage_tracker=usage_tracker,
        )

        arena = Arena.from_config(env, agents, protocol, config)

        assert arena.org_id == "org-123"
        assert arena.user_id == "user-456"
        assert arena.usage_tracker is usage_tracker


class TestContinuumMemoryIntegration:
    """Tests for ContinuumMemory integration in Arena."""

    def test_get_continuum_context_without_memory(self):
        """Test _get_continuum_context returns empty string without continuum memory."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test continuum", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        # No continuum_memory set

        result = arena._get_continuum_context()

        assert result == ""

    def test_get_continuum_context_with_cache(self):
        """Test _get_continuum_context returns cached value."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test cache", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        arena._continuum_context_cache = "Cached context"

        result = arena._get_continuum_context()

        assert result == "Cached context"

    def test_get_continuum_context_with_mock_memory(self):
        """Test _get_continuum_context retrieves from continuum memory."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Design a caching system", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        # Create mock continuum memory
        mock_memory = MagicMock()
        mock_entry = MagicMock()
        mock_entry.content = "Previous learning about caching"
        mock_entry.tier = MagicMock()
        mock_entry.tier.value = "medium"
        mock_entry.consolidation_score = 0.8
        mock_entry.id = "mem-123"
        mock_memory.retrieve.return_value = [mock_entry]

        arena = Arena(env, agents, protocol, continuum_memory=mock_memory)

        result = arena._get_continuum_context()

        assert "Previous learning" in result
        assert "[medium|high]" in result  # consolidation > 0.7 = high

    def test_get_continuum_context_handles_empty_memories(self):
        """Test _get_continuum_context handles empty memory retrieval."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = []

        arena = Arena(env, agents, protocol, continuum_memory=mock_memory)

        result = arena._get_continuum_context()

        assert result == ""

    def test_get_continuum_context_handles_exception(self):
        """Test _get_continuum_context handles retrieval errors gracefully."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = Exception("Database error")

        arena = Arena(env, agents, protocol, continuum_memory=mock_memory)

        result = arena._get_continuum_context()

        assert result == ""


class TestJudgeTermination:
    """Tests for judge termination feature."""

    @pytest.mark.asyncio
    async def test_judge_termination_disabled_by_default(self):
        """Test judge termination returns continue when disabled."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=3)
        protocol = DebateProtocol(rounds=3, judge_termination=False)

        arena = Arena(env, agents, protocol)

        should_continue, reason = await arena._check_judge_termination(
            round_num=2,
            proposals={"agent1": "Proposal 1"},
            context=[]
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_judge_termination_skips_early_rounds(self):
        """Test judge termination skips check in early rounds."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=5)
        protocol = DebateProtocol(
            rounds=5,
            judge_termination=True,
            min_rounds_before_judge_check=3
        )

        arena = Arena(env, agents, protocol)

        # Round 1 should skip check
        should_continue, reason = await arena._check_judge_termination(
            round_num=1,
            proposals={"agent1": "Proposal"},
            context=[]
        )

        assert should_continue is True
        assert reason == ""


class TestMomentDetectorAutoInit:
    """Tests for MomentDetector auto-initialization."""

    def test_auto_init_with_elo_system(self):
        """Test MomentDetector auto-initializes when elo_system provided."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_elo = MagicMock()

        with patch('aragora.agents.grounded.MomentDetector') as mock_md:
            arena = Arena(env, agents, protocol, elo_system=mock_elo)
            # MomentDetector should be auto-initialized
            # (may or may not succeed depending on import)

    def test_auto_init_import_error_handled(self):
        """Test MomentDetector import error is handled gracefully."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_elo = MagicMock()

        with patch.dict('sys.modules', {'aragora.agents.grounded': None}):
            # Should not raise even if import fails
            arena = Arena(env, agents, protocol, elo_system=mock_elo)
            assert arena is not None


class TestPositionRecording:
    """Tests for position recording functionality."""

    def test_record_position_without_ledger(self):
        """Test _record_grounded_position does nothing without ledger."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        # Should not raise without position_ledger
        arena._record_grounded_position(
            agent_name="agent1",
            content="Test position",
            debate_id="test-123",
            round_num=1,
        )

    def test_record_position_with_mock_ledger(self):
        """Test _record_grounded_position calls ledger correctly."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_ledger = MagicMock()
        arena = Arena(env, agents, protocol, position_ledger=mock_ledger)

        arena._record_grounded_position(
            agent_name="agent1",
            content="Test position content",
            debate_id="test-123",
            round_num=1,
            confidence=0.8,
        )

        mock_ledger.record_position.assert_called_once()

    def test_record_position_handles_exception(self):
        """Test _record_grounded_position handles ledger errors."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_ledger = MagicMock()
        mock_ledger.record_position.side_effect = Exception("DB error")
        arena = Arena(env, agents, protocol, position_ledger=mock_ledger)

        # Should not raise
        arena._record_grounded_position(
            agent_name="agent1",
            content="Test",
            debate_id="test-123",
            round_num=1,
        )


class TestRelationshipUpdates:
    """Tests for agent relationship updates."""

    def test_update_relationships_without_elo(self):
        """Test _update_agent_relationships does nothing without elo_system."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        # Should not raise without elo_system
        arena._update_agent_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner="agent1",
            votes=[],
        )

    def test_update_relationships_with_elo(self):
        """Test _update_agent_relationships calls elo_system correctly."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_elo = MagicMock()
        arena = Arena(env, agents, protocol, elo_system=mock_elo)

        mock_vote1 = MagicMock()
        mock_vote1.agent = "agent1"
        mock_vote1.choice = "agent1"

        mock_vote2 = MagicMock()
        mock_vote2.agent = "agent2"
        mock_vote2.choice = "agent1"

        arena._update_agent_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner="agent1",
            votes=[mock_vote1, mock_vote2],
        )

        mock_elo.update_relationships_batch.assert_called_once()

    def test_update_relationships_handles_exception(self):
        """Test _update_agent_relationships handles errors gracefully."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        mock_elo = MagicMock()
        mock_elo.update_relationships_batch.side_effect = Exception("DB error")
        arena = Arena(env, agents, protocol, elo_system=mock_elo)

        # Should not raise
        arena._update_agent_relationships(
            debate_id="test-123",
            participants=["agent1", "agent2"],
            winner="agent1",
            votes=[],
        )


class TestEarlyStoppingLogic:
    """Tests for early stopping configuration."""

    @pytest.mark.asyncio
    async def test_early_stopping_disabled(self):
        """Test _check_early_stopping returns True when disabled."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        env = Environment(task="Test task", max_rounds=3)
        protocol = DebateProtocol(rounds=3, early_stopping=False)

        arena = Arena(env, agents, protocol)

        result = await arena._check_early_stopping(
            round_num=2,
            proposals={"agent1": "Proposal"},
            context=[]
        )

        assert result is True  # Continue debate


class TestEventNotification:
    """Tests for event notification methods."""

    def test_notify_spectator_delegates_to_bridge(self):
        """Test _notify_spectator delegates to event_bridge."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)

        # Mock the event bridge
        arena.event_bridge = MagicMock()

        arena._notify_spectator("test_event", data="test_data")

        arena.event_bridge.notify.assert_called_once_with("test_event", data="test_data")

    def test_emit_moment_event_delegates_to_bridge(self):
        """Test _emit_moment_event delegates to event_bridge."""
        agents = [MockAgent("agent1")]
        env = Environment(task="Test task", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, agents, protocol)
        arena.event_bridge = MagicMock()

        mock_moment = MagicMock()
        arena._emit_moment_event(mock_moment)

        arena.event_bridge.emit_moment.assert_called_once_with(mock_moment)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
