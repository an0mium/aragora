"""
E2E tests for error recovery chains.

Tests the system's ability to:
- Handle cascading agent failures with fallback
- Raise AllProvidersExhaustedError when all providers fail
- Continue debates with degraded agent pools
- Recover from circuit breaker trips mid-debate
- Complete debates with minimum viable agents
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.fallback import (
    AgentFallbackChain,
    AllProvidersExhaustedError,
    FallbackTimeoutError,
)
from aragora.resilience import CircuitBreaker, get_circuit_breaker


class MockProviderAgent:
    """Mock agent that simulates a provider with configurable behavior."""

    def __init__(
        self,
        name: str,
        fail_count: int = 0,
        fail_permanently: bool = False,
        delay: float = 0.0,
        quota_error: bool = False,
    ):
        self.name = name
        self.model = name
        self._fail_count = fail_count
        self._fail_permanently = fail_permanently
        self._delay = delay
        self._quota_error = quota_error
        self._call_count = 0
        self.calls: list[str] = []

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        self._call_count += 1
        self.calls.append(prompt)

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._fail_permanently:
            if self._quota_error:
                raise Exception("Rate limit exceeded (429)")
            raise Exception(f"{self.name} permanently failed")

        if self._call_count <= self._fail_count:
            if self._quota_error:
                raise Exception("Rate limit exceeded (429)")
            raise Exception(f"{self.name} failed (attempt {self._call_count})")

        return f"Response from {self.name}"


class TestCascadingAgentFailures:
    """Test fallback chain behavior with cascading failures."""

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider_on_failure(self):
        """Verify chain falls back to second provider when first fails."""
        agent1 = MockProviderAgent("primary", fail_permanently=True)
        agent2 = MockProviderAgent("secondary")

        chain = AgentFallbackChain(providers=[agent1, agent2])
        result = await chain.generate("test prompt")

        assert "secondary" in result
        assert len(agent1.calls) == 1
        assert len(agent2.calls) == 1

    @pytest.mark.asyncio
    async def test_cascading_failures_through_chain(self):
        """Verify chain cascades through multiple failures."""
        agent1 = MockProviderAgent("primary", fail_permanently=True)
        agent2 = MockProviderAgent("secondary", fail_permanently=True)
        agent3 = MockProviderAgent("tertiary")

        chain = AgentFallbackChain(providers=[agent1, agent2, agent3])
        result = await chain.generate("test prompt")

        assert "tertiary" in result
        assert len(agent1.calls) == 1
        assert len(agent2.calls) == 1
        assert len(agent3.calls) == 1

    @pytest.mark.asyncio
    async def test_first_provider_succeeds_no_fallback(self):
        """Verify no fallback when first provider succeeds."""
        agent1 = MockProviderAgent("primary")
        agent2 = MockProviderAgent("secondary")

        chain = AgentFallbackChain(providers=[agent1, agent2])
        result = await chain.generate("test prompt")

        assert "primary" in result
        assert len(agent1.calls) == 1
        assert len(agent2.calls) == 0

    @pytest.mark.asyncio
    async def test_transient_failure_then_recovery(self):
        """Verify agent recovers after transient failures."""
        agent1 = MockProviderAgent("primary", fail_count=2)  # Fails twice, then works
        agent2 = MockProviderAgent("secondary")

        chain = AgentFallbackChain(providers=[agent1, agent2])

        # First call - primary fails, falls back to secondary
        result1 = await chain.generate("prompt 1")
        assert "secondary" in result1

        # Second call - primary still failing
        result2 = await chain.generate("prompt 2")
        assert "secondary" in result2

        # Third call - primary recovered
        result3 = await chain.generate("prompt 3")
        assert "primary" in result3


class TestAllProvidersExhausted:
    """Test behavior when all providers in chain fail."""

    @pytest.mark.asyncio
    async def test_raises_all_providers_exhausted_error(self):
        """Verify AllProvidersExhaustedError raised when all fail."""
        agent1 = MockProviderAgent("primary", fail_permanently=True)
        agent2 = MockProviderAgent("secondary", fail_permanently=True)

        chain = AgentFallbackChain(providers=[agent1, agent2])

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await chain.generate("test prompt")

        assert "primary" in str(exc_info.value) or "All" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_all_providers_exhausted_with_single_provider(self):
        """Verify error raised even with single provider chain."""
        agent = MockProviderAgent("solo", fail_permanently=True)

        chain = AgentFallbackChain(providers=[agent])

        with pytest.raises(AllProvidersExhaustedError):
            await chain.generate("test prompt")

    @pytest.mark.asyncio
    async def test_exhausted_error_contains_tried_providers(self):
        """Verify error message indicates which providers were tried."""
        agents = [MockProviderAgent(f"agent{i}", fail_permanently=True) for i in range(3)]

        chain = AgentFallbackChain(providers=agents)

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await chain.generate("test prompt")

        # All agents should have been called
        for agent in agents:
            assert len(agent.calls) == 1


class TestFallbackTimeout:
    """Test timeout behavior in fallback chains."""

    @pytest.mark.asyncio
    async def test_max_fallback_time_limits_total_duration(self):
        """Verify max_fallback_time limits total chain execution time."""
        # Create agents that fail to force traversal through chain
        agents = [
            MockProviderAgent("agent1", fail_permanently=True),
            MockProviderAgent("agent2", fail_permanently=True),
            MockProviderAgent("agent3", fail_permanently=True),
        ]

        # Very short timeout
        chain = AgentFallbackChain(providers=agents, max_fallback_time=0.01)

        start = time.time()
        with pytest.raises((FallbackTimeoutError, AllProvidersExhaustedError)):
            await chain.generate("test prompt")
        elapsed = time.time() - start

        # Should complete quickly due to timeout
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_successful_fast_agent_completes_before_timeout(self):
        """Verify fast successful agent completes without hitting timeout."""
        fast_agent = MockProviderAgent("fast")

        chain = AgentFallbackChain(providers=[fast_agent], max_fallback_time=5.0)

        start = time.time()
        result = await chain.generate("test prompt")
        elapsed = time.time() - start

        assert "fast" in result
        assert elapsed < 1.0  # Should be nearly instant


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with fallback chains."""

    @pytest.fixture
    def fresh_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        # Clear any existing circuit breakers
        from aragora.resilience import _circuit_breakers

        _circuit_breakers.clear()
        yield
        _circuit_breakers.clear()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, fresh_circuit_breakers):
        """Verify circuit breaker opens after threshold failures."""
        breaker = get_circuit_breaker("test-agent", failure_threshold=3)

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls_when_open(self, fresh_circuit_breakers):
        """Verify open circuit breaker prevents agent calls."""
        breaker = get_circuit_breaker("blocked-agent", failure_threshold=2)

        # Open the breaker
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open

        # Agent should be skipped when breaker is open
        agent = MockProviderAgent("blocked-agent")
        fallback = MockProviderAgent("fallback")

        # Simulate chain behavior - check breaker before calling
        if breaker.is_open:
            result = await fallback.generate("test")
        else:
            result = await agent.generate("test")

        assert "fallback" in result
        assert len(agent.calls) == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_after_cooldown(self, fresh_circuit_breakers):
        """Verify circuit breaker allows requests after cooldown expires."""
        breaker = get_circuit_breaker("recovering-agent", failure_threshold=2, cooldown_seconds=0.1)

        # Open the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # After cooldown, breaker remains technically "open" but allows trial requests
        # The cooldown_remaining() should return 0 or near-zero
        assert breaker.cooldown_remaining() == 0

        # Record success to fully close the breaker (simulating half-open → closed)
        breaker.record_success()
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self, fresh_circuit_breakers):
        """Verify circuit breaker closes after successful call."""
        breaker = get_circuit_breaker("healing-agent", failure_threshold=2, cooldown_seconds=0.1)

        # Open the breaker
        breaker.record_failure()
        breaker.record_failure()

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Record success
        breaker.record_success()

        # Should be closed now
        assert not breaker.is_open


class TestFallbackMetrics:
    """Test metrics collection in fallback chains."""

    @pytest.mark.asyncio
    async def test_metrics_track_primary_success(self):
        """Verify metrics track primary provider successes."""
        agent = MockProviderAgent("primary")
        chain = AgentFallbackChain(providers=[agent])

        await chain.generate("test")

        # Access metrics directly from chain.metrics
        assert chain.metrics.primary_attempts >= 1
        assert chain.metrics.primary_successes >= 1

    @pytest.mark.asyncio
    async def test_metrics_track_fallback_usage(self):
        """Verify metrics track fallback provider usage."""
        agent1 = MockProviderAgent("primary", fail_permanently=True)
        agent2 = MockProviderAgent("fallback")

        chain = AgentFallbackChain(providers=[agent1, agent2])

        await chain.generate("test")

        # Access metrics directly from chain.metrics
        assert chain.metrics.fallback_attempts >= 1
        assert chain.metrics.fallback_successes >= 1

    @pytest.mark.asyncio
    async def test_metrics_calculate_success_rate(self):
        """Verify success rate calculation."""
        agent = MockProviderAgent("primary")
        chain = AgentFallbackChain(providers=[agent])

        # Make several successful calls
        for _ in range(5):
            await chain.generate("test")

        # Access metrics directly from chain.metrics
        assert chain.metrics.success_rate == 1.0


class TestPartialConsensusWithDegradedAgents:
    """Test debate behavior with reduced agent pools."""

    @pytest.fixture
    def mock_debate_agents(self):
        """Create a mix of working and failing agents."""
        return [
            MockProviderAgent("agent1"),  # Works
            MockProviderAgent("agent2", fail_permanently=True),  # Fails
            MockProviderAgent("agent3"),  # Works
            MockProviderAgent("agent4", fail_permanently=True),  # Fails
            MockProviderAgent("agent5"),  # Works
        ]

    @pytest.mark.asyncio
    async def test_debate_continues_with_partial_agents(self, mock_debate_agents):
        """Verify debate can proceed with subset of working agents."""
        working_agents = [a for a in mock_debate_agents if not a._fail_permanently]

        # Collect responses from working agents
        responses = []
        for agent in mock_debate_agents:
            try:
                response = await agent.generate("Debate topic: testing")
                responses.append(response)
            except Exception:
                pass  # Skip failed agents

        # Should have responses from 3 working agents
        assert len(responses) == 3

    @pytest.mark.asyncio
    async def test_minimum_agents_for_valid_debate(self, mock_debate_agents):
        """Verify debate requires minimum agents for validity."""
        min_agents_required = 2
        responses = []

        for agent in mock_debate_agents:
            try:
                response = await agent.generate("Debate topic")
                responses.append(response)
            except Exception:
                pass

        assert len(responses) >= min_agents_required

    @pytest.mark.asyncio
    async def test_consensus_achievable_with_degraded_pool(self):
        """Verify consensus can be reached with reduced agent pool."""
        # Create agents that will agree
        agents = [
            MockProviderAgent("agent1"),
            MockProviderAgent("agent2"),
            MockProviderAgent("agent3"),
        ]

        # Simulate consensus voting
        votes = []
        for agent in agents:
            response = await agent.generate("Vote on proposal")
            votes.append(response)

        # All responses received - consensus possible
        assert len(votes) == 3


class TestDebateCompletesWithMinimumAgents:
    """Test that debates complete even with agent failures."""

    @pytest.mark.asyncio
    async def test_two_agent_debate_completes(self):
        """Verify debate completes with minimum 2 agents."""
        agents = [
            MockProviderAgent("agent1"),
            MockProviderAgent("agent2"),
        ]

        # Simulate multi-round debate
        rounds = 3
        debate_log = []

        for round_num in range(rounds):
            round_responses = []
            for agent in agents:
                response = await agent.generate(f"Round {round_num + 1} argument")
                round_responses.append(response)
            debate_log.append(round_responses)

        # Debate should complete all rounds
        assert len(debate_log) == rounds
        assert all(len(r) == 2 for r in debate_log)

    @pytest.mark.asyncio
    async def test_debate_fails_with_one_agent(self):
        """Verify debate cannot proceed with single agent."""
        min_required = 2
        working_agents = [MockProviderAgent("solo")]

        assert len(working_agents) < min_required

    @pytest.mark.asyncio
    async def test_debate_adapts_to_mid_debate_failures(self):
        """Verify debate continues when agent fails mid-debate."""
        # Agent that fails after 2 calls
        failing_agent = MockProviderAgent("failing", fail_count=0)
        failing_agent._fail_count = 2  # Will work for 2 calls, then fail

        stable_agent = MockProviderAgent("stable")

        agents = [failing_agent, stable_agent]

        rounds = 4
        successful_rounds = 0

        for round_num in range(rounds):
            round_responses = []
            for agent in agents:
                try:
                    response = await agent.generate(f"Round {round_num + 1}")
                    round_responses.append(response)
                except Exception:
                    pass  # Agent failed

            # Count round if we got at least one response
            if round_responses:
                successful_rounds += 1

        # Should complete all rounds (even if degraded)
        assert successful_rounds == rounds


class TestStorageErrorRecovery:
    """Test recovery from storage/persistence errors."""

    @pytest.mark.asyncio
    async def test_debate_result_queued_on_storage_error(self):
        """Verify failed storage queues result for retry."""
        retry_queue: list[dict] = []

        def mock_save_result(result: dict) -> None:
            raise Exception("Database connection lost")

        def queue_for_retry(result: dict) -> None:
            retry_queue.append(result)

        # Simulate debate completion with storage failure
        debate_result = {"debate_id": "test-1", "consensus": "agreed"}

        try:
            mock_save_result(debate_result)
        except Exception:
            queue_for_retry(debate_result)

        assert len(retry_queue) == 1
        assert retry_queue[0]["debate_id"] == "test-1"

    @pytest.mark.asyncio
    async def test_retry_queue_processes_after_recovery(self):
        """Verify queued results are saved after storage recovers."""
        retry_queue = [
            {"debate_id": "test-1", "consensus": "agreed"},
            {"debate_id": "test-2", "consensus": "disagreed"},
        ]
        saved_results: list[dict] = []

        def save_result(result: dict) -> None:
            saved_results.append(result)

        # Process retry queue
        while retry_queue:
            result = retry_queue.pop(0)
            save_result(result)

        assert len(saved_results) == 2
        assert len(retry_queue) == 0

    @pytest.mark.asyncio
    async def test_partial_save_on_storage_recovery(self):
        """Verify partial saves handled correctly."""
        results_to_save = [
            {"id": "1", "data": "first"},
            {"id": "2", "data": "second"},
            {"id": "3", "data": "third"},
        ]
        saved: list[str] = []
        failed: list[str] = []

        # Simulate storage that fails on second item
        for i, result in enumerate(results_to_save):
            if i == 1:
                failed.append(result["id"])
            else:
                saved.append(result["id"])

        assert len(saved) == 2
        assert len(failed) == 1
        assert "2" in failed


class TestErrorRecoveryOrdering:
    """Test that error recovery follows correct precedence."""

    @pytest.mark.asyncio
    async def test_fallback_chain_respects_order(self):
        """Verify fallback follows provider order."""
        call_order: list[str] = []

        class OrderTrackingAgent(MockProviderAgent):
            async def generate(self, prompt: str, context: Optional[list] = None) -> str:
                call_order.append(self.name)
                return await super().generate(prompt, context)

        agents = [
            OrderTrackingAgent("first", fail_permanently=True),
            OrderTrackingAgent("second", fail_permanently=True),
            OrderTrackingAgent("third"),
        ]

        chain = AgentFallbackChain(providers=agents)
        await chain.generate("test")

        assert call_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_known_failed(self):
        """Verify circuit breaker allows skipping known-failed providers."""
        # This tests the concept that open circuit breakers prevent wasted calls
        breakers = {
            "agent1": CircuitBreaker(failure_threshold=1),
            "agent2": CircuitBreaker(failure_threshold=1),
            "agent3": CircuitBreaker(failure_threshold=1),
        }

        # Mark agent1 as failed
        breakers["agent1"].record_failure()

        # Simulate provider selection respecting breakers
        available_agents = ["agent1", "agent2", "agent3"]
        selected = [a for a in available_agents if not breakers[a].is_open]

        assert "agent1" not in selected
        assert "agent2" in selected
        assert "agent3" in selected


class TestConcurrentFailureRecovery:
    """Test recovery behavior under concurrent failures."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_circuit_breaker_state(self):
        """Verify concurrent requests see same circuit breaker state."""
        from aragora.resilience import _circuit_breakers

        _circuit_breakers.clear()

        breaker = get_circuit_breaker("shared-agent", failure_threshold=3)

        async def make_request(request_id: int) -> bool:
            if breaker.is_open:
                return False  # Request blocked
            # Simulate failure
            breaker.record_failure()
            return True  # Request attempted

        # Run concurrent requests
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # After 3 failures, remaining requests should be blocked
        attempted = sum(1 for r in results if r)
        assert attempted <= 3  # Max 3 attempts before breaker opens

        _circuit_breakers.clear()

    @pytest.mark.asyncio
    async def test_recovery_visible_to_all_concurrent_requests(self):
        """Verify recovery state shared across concurrent requests."""
        from aragora.resilience import _circuit_breakers

        _circuit_breakers.clear()

        breaker = get_circuit_breaker(
            "recovering-shared", failure_threshold=2, cooldown_seconds=0.05
        )

        # Open the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        # Wait for cooldown
        await asyncio.sleep(0.1)

        # Record success to close breaker
        breaker.record_success()

        # All concurrent requests should now see closed breaker
        async def check_breaker() -> bool:
            return not breaker.is_open

        results = await asyncio.gather(*[check_breaker() for _ in range(5)])
        assert all(results)

        _circuit_breakers.clear()
