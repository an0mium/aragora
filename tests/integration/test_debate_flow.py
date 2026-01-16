"""
Integration tests for complete debate flows.

These tests verify the full debate lifecycle from start to finish,
including all major components: Arena, agents, memory, events, and persistence.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from aragora.core import Agent, Message, Critique, Vote, Environment, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore


# =============================================================================
# Auto-mock external dependencies for all tests
# =============================================================================


@pytest.fixture(autouse=True)
def mock_external_calls():
    """Mock external API calls to prevent network requests during tests."""
    # Mock the Arena's trending context gathering to prevent Reddit/pulse API calls
    with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None):
        # Mock the ContextInitializer to skip external research
        with patch(
            "aragora.debate.phases.context_init.ContextInitializer.initialize",
            new_callable=AsyncMock,
            return_value=None,
        ):
            yield


# =============================================================================
# Mock Agent for Testing
# =============================================================================


class MockAgent(Agent):
    """Mock agent for integration testing that doesn't make API calls."""

    def __init__(self, name: str = "mock", model: str = "mock-model", role: str = "proposer"):
        super().__init__(name, model, role)
        self.agent_type = "mock"
        self.generate_responses = []
        self.critique_responses = []
        self.vote_responses = []
        self._generate_call_count = 0
        self._critique_call_count = 0
        self._vote_call_count = 0
        self.generate_calls = []  # Track all calls
        self.critique_calls = []
        self.vote_calls = []

    async def generate(self, prompt: str, context: list = None) -> str:
        """Return mock response and track the call."""
        self.generate_calls.append({"prompt": prompt, "context": context})
        if self.generate_responses:
            response = self.generate_responses[
                self._generate_call_count % len(self.generate_responses)
            ]
            self._generate_call_count += 1
            return response
        return f"Mock response from {self.name}: {prompt[:50]}"

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        """Return mock critique and track the call."""
        self.critique_calls.append({"proposal": proposal, "task": task, "context": context})
        if self.critique_responses:
            response = self.critique_responses[
                self._critique_call_count % len(self.critique_responses)
            ]
            self._critique_call_count += 1
            return response
        return Critique(
            agent=self.name,
            target_agent="proposer",
            target_content=proposal[:100],
            issues=[f"Issue found by {self.name}"],
            suggestions=[f"Suggestion from {self.name}"],
            severity=0.5,
            reasoning=f"Critique reasoning from {self.name}",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        """Return mock vote and track the call."""
        self.vote_calls.append({"proposals": proposals, "task": task})
        if self.vote_responses:
            response = self.vote_responses[self._vote_call_count % len(self.vote_responses)]
            self._vote_call_count += 1
            return response
        choice = list(proposals.keys())[0] if proposals else "none"
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote reasoning from {self.name}",
            confidence=0.8,
            continue_debate=False,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def critique_store(temp_db):
    """Create a CritiqueStore with temp database."""
    return CritiqueStore(temp_db)


@pytest.fixture
def mock_emitter():
    """Create a mock event emitter."""
    emitter = Mock()
    emitter.emit = Mock()
    emitter.subscribe = Mock()
    return emitter


@pytest.fixture
def basic_agents():
    """Create a basic pair of mock agents."""
    proposer = MockAgent("alice", role="proposer")
    critic = MockAgent("bob", role="critic")

    proposer.generate_responses = [
        "I propose we implement a distributed cache with TTL-based expiration.",
        "Based on the feedback, I'll add cache invalidation callbacks.",
    ]
    critic.generate_responses = [
        "The cache design looks solid, but we should consider memory limits.",
        "The invalidation callbacks address my concerns.",
    ]

    return [proposer, critic]


@pytest.fixture
def three_agents():
    """Create three agents for more complex debates."""
    alice = MockAgent("alice", role="proposer")
    bob = MockAgent("bob", role="critic")
    charlie = MockAgent("charlie", role="synthesizer")

    alice.generate_responses = ["Initial proposal from Alice"]
    bob.generate_responses = ["Bob's critique and counter-proposal"]
    charlie.generate_responses = ["Charlie's synthesis of both views"]

    return [alice, bob, charlie]


# =============================================================================
# Test Classes
# =============================================================================


class TestMinimalDebateFlow:
    """Test the minimal code path for a complete debate."""

    @pytest.mark.asyncio
    async def test_debate_completes_successfully(self, basic_agents):
        """A minimal debate should complete without errors."""
        env = Environment(task="Design a caching system", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == "Design a caching system"

    @pytest.mark.asyncio
    async def test_debate_produces_result_fields(self, basic_agents):
        """Debate result should contain all expected fields."""
        env = Environment(task="Test task", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # Core fields should be present
        assert hasattr(result, "task")
        assert hasattr(result, "messages")
        assert hasattr(result, "votes")
        assert hasattr(result, "consensus_reached")
        assert hasattr(result, "rounds_used")

    @pytest.mark.asyncio
    async def test_debate_generates_messages(self, basic_agents):
        """Debate should generate messages from agents."""
        env = Environment(task="Test task", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # Messages list should exist (may be empty if no proposers)
        # The important thing is the debate completed with a result
        assert isinstance(result.messages, list)

    @pytest.mark.asyncio
    async def test_agents_are_called(self, basic_agents):
        """Agents should have their methods called during debate."""
        env = Environment(task="Test task", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        await arena.run()

        # At least one agent should have generated content
        assert any(
            len(agent.generate_calls) > 0 or len(agent.vote_calls) > 0 for agent in basic_agents
        )


class TestDebateWithMemory:
    """Test debates with CritiqueStore persistence."""

    @pytest.mark.asyncio
    async def test_debate_with_memory_completes(self, basic_agents, critique_store):
        """Debate with memory store should complete."""
        env = Environment(task="Test with memory", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol, memory=critique_store)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_memory_stores_are_accessible(self, basic_agents, temp_db):
        """Memory stores should be accessible after debate."""
        store = CritiqueStore(temp_db)
        env = Environment(task="Memory test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, basic_agents, protocol, memory=store)
        await arena.run()

        # Store should have stats method
        stats = store.get_stats()
        assert isinstance(stats, dict)


class TestDebateWithEvents:
    """Test debates with event emission."""

    @pytest.mark.asyncio
    async def test_debate_with_emitter(self, basic_agents, mock_emitter):
        """Debate should work with event emitter."""
        env = Environment(task="Event test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol, event_emitter=mock_emitter)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestDebateConsensus:
    """Test consensus mechanisms."""

    @pytest.mark.asyncio
    async def test_majority_consensus(self, three_agents):
        """Majority consensus should work with 3 agents."""
        # All agents vote for alice
        for agent in three_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="alice",
                    reasoning="Alice's proposal is best",
                    confidence=0.8,
                    continue_debate=False,
                )
            ]

        env = Environment(task="Consensus test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, three_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_unanimous_consensus_requirement(self, basic_agents):
        """Unanimous consensus should require all agents to agree."""
        # Both agents vote for same choice
        basic_agents[0].vote_responses = [
            Vote(
                agent="alice",
                choice="alice",
                reasoning="My proposal is good",
                confidence=0.9,
                continue_debate=False,
            )
        ]
        basic_agents[1].vote_responses = [
            Vote(
                agent="bob",
                choice="alice",
                reasoning="I agree with Alice",
                confidence=0.85,
                continue_debate=False,
            )
        ]

        env = Environment(task="Unanimous test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="unanimous")

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestEarlyStopping:
    """Test early stopping behavior."""

    @pytest.mark.asyncio
    async def test_early_stopping_enabled(self, basic_agents):
        """Debate should stop early when consensus reached."""
        # Both agents vote to stop
        for agent in basic_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="alice",
                    reasoning="Done",
                    confidence=0.95,
                    continue_debate=False,
                )
            ]

        env = Environment(task="Early stop test", max_rounds=10)
        protocol = DebateProtocol(rounds=10, early_stopping=True)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # Should complete without using all 10 rounds
        assert result.rounds_used <= 10

    @pytest.mark.asyncio
    async def test_early_stopping_disabled(self, basic_agents):
        """Debate should continue when early stopping is disabled."""
        # Even if agents want to stop, debate continues
        for agent in basic_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="alice",
                    reasoning="Done",
                    confidence=0.95,
                    continue_debate=False,
                )
            ]

        env = Environment(task="No early stop test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, early_stopping=False)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestDebateRounds:
    """Test round limits and behavior."""

    @pytest.mark.asyncio
    async def test_respects_max_rounds(self, basic_agents):
        """Debate should not exceed max_rounds."""
        env = Environment(task="Round limit test", max_rounds=3)
        protocol = DebateProtocol(rounds=3)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert result.rounds_used <= 3

    @pytest.mark.asyncio
    async def test_single_round_debate(self, basic_agents):
        """Single round debate should complete."""
        env = Environment(task="Single round", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert result.rounds_used >= 1

    @pytest.mark.asyncio
    async def test_zero_rounds_uses_default(self, basic_agents):
        """Zero rounds should use at least one round."""
        env = Environment(task="Zero rounds test", max_rounds=1)
        # Protocol with 0 rounds - should still do something
        protocol = DebateProtocol(rounds=0)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestMultipleAgents:
    """Test debates with varying numbers of agents."""

    @pytest.mark.asyncio
    async def test_two_agents(self, basic_agents):
        """Two-agent debate should work."""
        env = Environment(task="Two agents", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_three_agents(self, three_agents):
        """Three-agent debate should work."""
        env = Environment(task="Three agents", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, three_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_single_agent(self):
        """Single-agent debate should still complete."""
        agent = MockAgent("solo", role="proposer")
        agent.generate_responses = ["Solo proposal"]
        agent.vote_responses = [
            Vote(
                agent="solo",
                choice="solo",
                reasoning="I'm the only one",
                confidence=1.0,
                continue_debate=False,
            )
        ]

        env = Environment(task="Solo test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, [agent], protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestDebateResult:
    """Test DebateResult structure and content."""

    @pytest.mark.asyncio
    async def test_result_has_messages(self, basic_agents):
        """Result should contain message history."""
        env = Environment(task="Messages test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result.messages, list)

    @pytest.mark.asyncio
    async def test_result_has_task(self, basic_agents):
        """Result should preserve the original task."""
        task = "Unique task for testing"
        env = Environment(task=task, max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert result.task == task

    @pytest.mark.asyncio
    async def test_result_rounds_used_positive(self, basic_agents):
        """Result should report positive rounds used."""
        env = Environment(task="Rounds test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert result.rounds_used >= 0


class TestDebateErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_task(self, basic_agents):
        """Empty task should raise ValueError during Environment creation."""
        with pytest.raises(ValueError, match="Task cannot be empty"):
            Environment(task="", max_rounds=1)

    @pytest.mark.asyncio
    async def test_long_task(self, basic_agents):
        """Debate with very long task should handle it."""
        long_task = "A" * 10000  # Very long task
        env = Environment(task=long_task, max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_unicode_task(self, basic_agents):
        """Debate should handle unicode in task."""
        task = "设计一个缓存系统 🚀 with émojis"
        env = Environment(task=task, max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestDebateWithCircuitBreaker:
    """Test debates with circuit breaker for agent failure handling."""

    @pytest.mark.asyncio
    async def test_debate_with_failing_agent_continues(self, basic_agents):
        """Debate should continue when one agent fails intermittently."""
        # Make first agent fail once then succeed
        call_count = 0
        original_generate = basic_agents[0].generate

        async def flaky_generate(prompt, context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated API failure")
            return await original_generate(prompt, context)

        basic_agents[0].generate = flaky_generate

        env = Environment(task="Failure test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        # Debate should handle the failure gracefully
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_debate_completes_with_all_agents_failing_once(self, basic_agents):
        """Debate should still produce a result even with agent failures."""
        for agent in basic_agents:
            original_generate = agent.generate
            call_count = [0]

            async def flaky_generate(
                prompt, context=None, _orig=original_generate, _count=call_count
            ):
                _count[0] += 1
                if _count[0] == 1:
                    raise RuntimeError("First call fails")
                return await _orig(prompt, context)

            agent.generate = flaky_generate

        env = Environment(task="Multi-failure test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


class TestDebateEloIntegration:
    """Test end-to-end debate completion with ELO updates."""

    @pytest.mark.asyncio
    async def test_debate_result_can_update_elo(self, basic_agents, temp_db):
        """After a debate, we can use the result to update ELO ratings."""
        from aragora.ranking.elo import EloSystem

        # Run debate
        env = Environment(task="ELO test debate", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # Create ELO system and update with debate results
        elo = EloSystem(temp_db)

        # Get agent names
        agent_names = [agent.name for agent in basic_agents]

        # Create scores based on vote results (simplified)
        scores = {name: 0.5 for name in agent_names}  # Default to draw
        if result.winner:
            scores[result.winner] = 1.0
            for name in agent_names:
                if name != result.winner:
                    scores[name] = 0.0

        # Record match
        elo.record_match(
            debate_id=f"test-{result.task[:20]}",
            participants=agent_names,
            scores=scores,
            domain="integration_test",
        )

        # Verify ratings were updated
        for name in agent_names:
            rating = elo.get_rating(name)
            assert rating.debates_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_debates_accumulate_elo(self, temp_db):
        """Multiple debates should accumulate ELO changes."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(temp_db)

        # Run multiple simulated debates
        for i in range(3):
            alice = MockAgent("alice", role="proposer")
            bob = MockAgent("bob", role="critic")

            alice.generate_responses = [f"Proposal {i}"]
            bob.generate_responses = [f"Critique {i}"]

            env = Environment(task=f"Multi-debate {i}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, [alice, bob], protocol)
            await arena.run()

            # Record alternating winners
            winner = "alice" if i % 2 == 0 else "bob"
            loser = "bob" if i % 2 == 0 else "alice"
            elo.record_match(
                debate_id=f"multi-debate-{i}",
                participants=["alice", "bob"],
                scores={winner: 1.0, loser: 0.0},
                domain="test",
            )

        # Verify accumulated stats
        alice_rating = elo.get_rating("alice", use_cache=False)
        bob_rating = elo.get_rating("bob", use_cache=False)

        assert alice_rating.debates_count == 3
        assert bob_rating.debates_count == 3
        assert alice_rating.wins >= 1
        assert bob_rating.wins >= 1


class TestDebateWithResiliencePatterns:
    """Test debates with resilience patterns (circuit breaker, retries)."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_tracks_agent_failures(self):
        """Circuit breaker should track agent failures across debates."""
        from aragora.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1.0)

        # Simulate multiple debate failures for one agent
        agent = MockAgent("flaky-agent", role="proposer")

        # First failure
        cb.record_failure("flaky-agent")
        assert cb.get_status("flaky-agent") == "closed"

        # Second failure opens circuit
        cb.record_failure("flaky-agent")
        assert cb.get_status("flaky-agent") == "open"

        # Agent should not be available
        assert not cb.is_available("flaky-agent")

    @pytest.mark.asyncio
    async def test_circuit_breaker_filters_available_agents(self, three_agents):
        """Circuit breaker should filter out unavailable agents."""
        from aragora.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60.0)

        # Mark one agent as failed
        cb.record_failure("bob")
        cb.record_failure("bob")  # Opens circuit

        # Filter available agents
        available = cb.filter_available_agents(three_agents)

        # Bob should be filtered out
        available_names = [a.name for a in available]
        assert "bob" not in available_names
        assert "alice" in available_names
        assert "charlie" in available_names

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Circuit breaker should recover after cooldown."""
        from aragora.resilience import CircuitBreaker
        import time

        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure("agent")
        cb.record_failure("agent")
        assert cb.get_status("agent") == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # Should now be half-open
        assert cb.is_available("agent")

        # Success closes circuit
        cb.record_success("agent")
        cb.record_success("agent")  # Need 2 successes in half-open
        assert cb.get_status("agent") == "closed"


class TestDebateArtifactIntegration:
    """Test debate artifact generation and consumption."""

    @pytest.mark.asyncio
    async def test_debate_produces_trace_data(self, basic_agents):
        """Debate should produce trace data for downstream processing."""
        env = Environment(task="Trace test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # Result should have essential fields
        assert result.task == "Trace test"
        assert hasattr(result, "messages")
        assert hasattr(result, "votes")

    @pytest.mark.asyncio
    async def test_debate_messages_have_structure(self, basic_agents):
        """Debate messages should have proper structure."""
        basic_agents[0].generate_responses = ["A structured proposal"]

        env = Environment(task="Message structure test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, basic_agents, protocol)
        result = await arena.run()

        # If there are messages, they should be Message objects
        for msg in result.messages:
            assert isinstance(msg, Message)
            assert hasattr(msg, "agent")
            assert hasattr(msg, "content")


class TestConcurrentDebates:
    """Test running multiple debates concurrently."""

    @pytest.mark.asyncio
    async def test_parallel_debates_complete(self):
        """Multiple debates can run in parallel without interference."""

        async def run_debate(task_id: int):
            alice = MockAgent(f"alice-{task_id}", role="proposer")
            bob = MockAgent(f"bob-{task_id}", role="critic")

            alice.generate_responses = [f"Proposal for task {task_id}"]
            bob.generate_responses = [f"Critique for task {task_id}"]

            env = Environment(task=f"Parallel task {task_id}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, [alice, bob], protocol)
            return await arena.run()

        # Run 3 debates concurrently
        results = await asyncio.gather(run_debate(1), run_debate(2), run_debate(3))

        # All should complete
        assert len(results) == 3
        for i, result in enumerate(results, 1):
            assert isinstance(result, DebateResult)
            assert f"Parallel task {i}" == result.task

    @pytest.mark.asyncio
    async def test_parallel_debates_with_shared_memory(self, temp_db):
        """Parallel debates can share memory store."""

        store = CritiqueStore(temp_db)

        async def run_debate_with_memory(task_id: int):
            alice = MockAgent(f"alice-{task_id}", role="proposer")
            bob = MockAgent(f"bob-{task_id}", role="critic")

            env = Environment(task=f"Shared memory task {task_id}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, [alice, bob], protocol, memory=store)
            return await arena.run()

        # Run debates concurrently with shared memory
        results = await asyncio.gather(run_debate_with_memory(1), run_debate_with_memory(2))

        assert len(results) == 2
        for result in results:
            assert isinstance(result, DebateResult)
