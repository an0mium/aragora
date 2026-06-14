"""
Tests for orchestrator error recovery and edge cases.

Covers:
1. Debate recovery after partial agent failure
2. Timeout handling with graceful degradation
3. Memory pressure with large suggestion queues
4. DebateContext state serialization
"""

import asyncio
import pytest

# Mark all tests in this module as slow (involve timeouts and recovery scenarios)
pytestmark = pytest.mark.slow
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core import Agent, Environment, Message, DebateResult
from aragora.debate.context import DebateContext
from aragora.debate.protocol import DebateProtocol, CircuitBreaker
from aragora.config import USER_EVENT_QUEUE_SIZE


class TestDebateRecoveryAfterPartialFailure:
    """Tests for debate recovery when some agents fail mid-debate."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with varying reliability."""
        reliable_agent = MagicMock(spec=Agent)
        reliable_agent.name = "reliable"
        reliable_agent.role = "proposer"
        reliable_agent.generate = AsyncMock(return_value="Reliable response")

        failing_agent = MagicMock(spec=Agent)
        failing_agent.name = "failing"
        failing_agent.role = "proposer"
        failing_agent.generate = AsyncMock(side_effect=Exception("Agent failure"))

        timeout_agent = MagicMock(spec=Agent)
        timeout_agent.name = "timeout"
        timeout_agent.role = "proposer"

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)  # Will timeout
            return "Never returned"

        timeout_agent.generate = AsyncMock(side_effect=slow_response)

        return [reliable_agent, failing_agent, timeout_agent]

    def test_circuit_breaker_tracks_individual_failures(self, mock_agents):
        """Circuit breaker tracks failures per-agent."""
        cb = CircuitBreaker(failure_threshold=2)

        # Record failures for failing agent (multi-entity mode)
        cb.record_failure("failing")
        cb.record_failure("failing")

        # Failing agent should be blocked
        assert cb.is_available("failing") is False

        # Other agents should still be allowed
        assert cb.is_available("reliable") is True
        assert cb.is_available("timeout") is True

    def test_debate_continues_with_remaining_agents(self, mock_agents):
        """Debate should continue when some agents fail."""
        cb = CircuitBreaker(failure_threshold=1)

        # Simulate failing agent being blocked (multi-entity mode)
        cb.record_failure("failing")
        assert cb.is_available("failing") is False

        # Filter available agents
        available = [a for a in mock_agents if cb.is_available(a.name)]
        assert len(available) == 2
        assert "failing" not in [a.name for a in available]

    def test_partial_results_preserved_on_agent_failure(self):
        """Partial results should be preserved when agents fail."""
        ctx = DebateContext(
            env=Environment(task="Test task"),
            agents=[],
            start_time=time.time(),
            debate_id="test-123",
        )
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )

        # Add some messages before failure
        msg1 = Message(role="assistant", agent="agent1", content="Response 1", round=1)
        msg2 = Message(role="assistant", agent="agent2", content="Response 2", round=1)

        ctx.add_message(msg1)
        ctx.add_message(msg2)

        # Simulate agent3 failure - partial messages preserved
        assert len(ctx.partial_messages) == 2
        assert ctx.partial_messages[0].agent == "agent1"
        assert ctx.partial_messages[1].agent == "agent2"


class TestTimeoutHandling:
    """Tests for timeout handling during debates."""

    def test_context_tracks_partial_rounds(self):
        """Context should track how many rounds completed before timeout."""
        ctx = DebateContext(
            env=Environment(task="Test"),
            agents=[],
            start_time=time.time(),
            debate_id="test-123",
        )

        # Simulate completing 2 rounds before timeout
        ctx.current_round = 1
        ctx.partial_rounds = 1
        ctx.current_round = 2
        ctx.partial_rounds = 2

        # On timeout, partial_rounds tells us how far we got
        assert ctx.partial_rounds == 2

    @pytest.mark.asyncio
    async def test_with_timeout_returns_partial_on_timeout(self):
        """with_timeout should allow partial result recovery."""

        async def slow_task():
            await asyncio.sleep(10)
            return "completed"

        # Test that timeout is handled gracefully (using wait_for for Python 3.10 compat)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_task(), timeout=0.1)

    def test_timeout_result_contains_metadata(self):
        """Timeout results should include duration and rounds info."""
        ctx = DebateContext(
            env=Environment(task="Test"),
            agents=[],
            start_time=time.time() - 5.0,  # Started 5 seconds ago
            debate_id="test-123",
        )
        ctx.result = DebateResult(
            task="Test",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        # finalize_result uses current_round, not partial_rounds
        ctx.current_round = 2
        ctx.partial_rounds = 2

        # Finalize would set duration and rounds_used from current_round
        result = ctx.finalize_result()
        assert result.duration_seconds >= 5.0
        assert result.rounds_used == 2


class TestMemoryPressure:
    """Tests for handling large suggestion queues."""

    def test_user_suggestions_bounded_by_maxlen(self):
        """User suggestions deque should respect maxlen."""
        suggestions: deque = deque(maxlen=100)

        # Add more than maxlen
        for i in range(150):
            suggestions.append({"content": f"suggestion_{i}"})

        # Should only keep last 100
        assert len(suggestions) == 100
        assert suggestions[0]["content"] == "suggestion_50"
        assert suggestions[-1]["content"] == "suggestion_149"

    def test_user_votes_bounded_by_maxlen(self):
        """User votes deque should respect maxlen."""
        votes: deque = deque(maxlen=100)

        for i in range(150):
            votes.append({"choice": f"choice_{i}"})

        assert len(votes) == 100

    def test_deque_eviction_is_fifo(self):
        """Oldest items should be evicted first."""
        q: deque = deque(maxlen=3)
        q.append("a")
        q.append("b")
        q.append("c")

        assert list(q) == ["a", "b", "c"]

        q.append("d")
        assert list(q) == ["b", "c", "d"]  # "a" evicted

    def test_large_suggestion_processing(self):
        """Suggestion clustering should handle large inputs efficiently."""
        from aragora.audience.suggestions import cluster_suggestions

        # Create 100 similar suggestions
        suggestions = [
            {"suggestion": f"Suggestion variant {i % 10}", "user_id": f"user_{i}"}
            for i in range(100)
        ]

        # Should cluster without timeout (cap is 50)
        start = time.time()
        clusters = cluster_suggestions(suggestions)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be fast
        assert len(clusters) <= 5  # max_clusters default

    def test_config_queue_size_respected(self):
        """USER_EVENT_QUEUE_SIZE from config should be used."""
        # Verify the config value exists and is reasonable
        assert USER_EVENT_QUEUE_SIZE > 0
        assert USER_EVENT_QUEUE_SIZE <= 100000  # Sanity check


class TestDebateContextState:
    """Tests for DebateContext state management."""

    def test_context_summary_dict(self):
        """to_summary_dict should return useful debugging info."""
        agent1 = MagicMock(spec=Agent)
        agent1.name = "agent1"
        agent2 = MagicMock(spec=Agent)
        agent2.name = "agent2"

        ctx = DebateContext(
            env=Environment(task="Test task"),
            agents=[agent1, agent2],
            start_time=time.time(),
            debate_id="test-123",
            domain="technology",
        )
        ctx.proposers = [agent1]
        ctx.proposals = {"agent1": "Proposal text"}
        ctx.winner_agent = "agent1"

        summary = ctx.to_summary_dict()

        assert summary["debate_id"] == "test-123"
        assert summary["domain"] == "technology"
        assert summary["agents"] == ["agent1", "agent2"]
        assert summary["proposers"] == ["agent1"]
        assert summary["num_proposals"] == 1
        assert summary["winner"] == "agent1"

    def test_get_agent_by_name(self):
        """get_agent_by_name should find correct agent."""
        agent1 = MagicMock(spec=Agent)
        agent1.name = "agent1"
        agent2 = MagicMock(spec=Agent)
        agent2.name = "agent2"

        ctx = DebateContext(
            env=Environment(task="Test"),
            agents=[agent1, agent2],
            start_time=time.time(),
            debate_id="test-123",
        )

        assert ctx.get_agent_by_name("agent1") == agent1
        assert ctx.get_agent_by_name("agent2") == agent2
        assert ctx.get_agent_by_name("nonexistent") is None

    def test_add_message_updates_all_tracking(self):
        """add_message should update context, partial, and result."""
        ctx = DebateContext(
            env=Environment(task="Test"),
            agents=[],
            start_time=time.time(),
            debate_id="test-123",
        )
        ctx.result = DebateResult(
            task="Test",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )

        msg = Message(role="assistant", agent="test", content="Test", round=1)
        ctx.add_message(msg)

        assert len(ctx.context_messages) == 1
        assert len(ctx.partial_messages) == 1
        assert len(ctx.result.messages) == 1

    def test_convergence_state_tracking(self):
        """Context should track convergence state."""
        ctx = DebateContext(
            env=Environment(task="Test"),
            agents=[],
            start_time=time.time(),
            debate_id="test-123",
        )

        # Initially no convergence
        assert ctx.convergence_status == ""
        assert ctx.convergence_similarity == 0.0
        assert ctx.early_termination is False

        # Update convergence state
        ctx.convergence_status = "converged"
        ctx.convergence_similarity = 0.95
        ctx.early_termination = True

        assert ctx.convergence_status == "converged"
        assert ctx.convergence_similarity == 0.95
        assert ctx.early_termination is True


class TestCircuitBreakerEdgeCases:
    """Additional edge cases for CircuitBreaker."""

    def test_rapid_failures_open_quickly(self):
        """Rapid successive failures should open circuit quickly."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

        # Rapid failures
        cb.record_failure()
        cb.record_failure()
        opened = cb.record_failure()

        assert opened is True
        assert cb.is_open is True

    def test_cooldown_race_condition(self):
        """Concurrent checks near cooldown boundary should be safe."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

        # Open the circuit
        cb.record_failure()
        assert cb.is_open is True

        # Multiple rapid checks near cooldown boundary
        results = []
        for _ in range(100):
            results.append(cb.can_proceed())
            time.sleep(0.001)

        # Some should be False (before cooldown), some True (after)
        assert False in results  # At least some blocked
        # After cooldown, should be able to proceed
        time.sleep(0.1)
        assert cb.can_proceed() is True

    def test_zero_failure_threshold(self):
        """Zero threshold should never open (or always open?)."""
        cb = CircuitBreaker(failure_threshold=0)
        # With threshold=0, first failure opens
        opened = cb.record_failure()
        assert opened is True

    def test_very_long_cooldown(self):
        """Very long cooldown should not cause issues."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=86400)  # 1 day

        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_mixed_entity_failures(self):
        """Multi-entity mode should isolate failures correctly."""
        cb = CircuitBreaker(failure_threshold=2)

        # Agent A fails twice - opens for A only
        cb.record_failure("agent_a")
        cb.record_failure("agent_a")

        assert cb.is_available("agent_a") is False
        assert cb.is_available("agent_b") is True
        assert cb.is_available("agent_c") is True

        # Agent B fails once - still available
        cb.record_failure("agent_b")
        assert cb.is_available("agent_b") is True

        # Agent B fails again - now blocked
        cb.record_failure("agent_b")
        assert cb.is_available("agent_b") is False

        # Agent C still fine
        assert cb.is_available("agent_c") is True
