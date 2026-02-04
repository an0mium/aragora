"""
Chaos Engineering Tests: Agent Failures.

Tests agent failure recovery mechanisms:
- Agent unavailability handling
- Fallback to alternate agents
- Recovery after transient failures
- Queue handling during outages

Run with extended timeout:
    pytest tests/chaos/test_agent_failures.py -v --timeout=300
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers
from aragora.resilience.circuit_breaker import CircuitBreaker


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset all circuit breakers before/after each test."""
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture(autouse=True)
def seed_random():
    """Seed random for reproducibility."""
    random.seed(42)
    yield


@dataclass
class MockAgent:
    """Mock agent for chaos testing."""

    name: str
    failure_rate: float = 0.0
    latency_ms: float = 10
    is_available: bool = True

    async def generate(self, prompt: str) -> str:
        if not self.is_available:
            raise ConnectionError(f"Agent {self.name} unavailable")

        await asyncio.sleep(self.latency_ms / 1000)

        if random.random() < self.failure_rate:
            raise RuntimeError(f"Agent {self.name} failed")

        return f"Response from {self.name}"


class AgentPool:
    """Agent pool with circuit breaker protection."""

    def __init__(self, agents: list[MockAgent]):
        self.agents = {agent.name: agent for agent in agents}
        # Create separate circuit breakers for each agent
        self.breakers = {
            agent.name: get_circuit_breaker(
                f"agent-pool-{agent.name}",
                failure_threshold=3,
                cooldown_seconds=1.0,
            )
            for agent in agents
        }

    async def generate(self, prompt: str, preferred_agent: str | None = None) -> tuple[str, str]:
        """Generate response, falling back to available agents."""
        # Try preferred agent first
        if preferred_agent and self.breakers[preferred_agent].can_proceed():
            try:
                result = await self.agents[preferred_agent].generate(prompt)
                self.breakers[preferred_agent].record_success()
                return result, preferred_agent
            except Exception:
                self.breakers[preferred_agent].record_failure()

        # Fall back to other available agents
        for name, agent in self.agents.items():
            if name == preferred_agent:
                continue
            if not self.breakers[name].can_proceed():
                continue
            try:
                result = await agent.generate(prompt)
                self.breakers[name].record_success()
                return result, name
            except Exception:
                self.breakers[name].record_failure()

        raise RuntimeError("All agents unavailable")


class TestAgentFailureRecovery:
    """Tests for agent failure and recovery scenarios."""

    @pytest.fixture
    def agents(self) -> list[MockAgent]:
        """Create a pool of mock agents."""
        return [
            MockAgent(name="primary", failure_rate=0.0),
            MockAgent(name="fallback1", failure_rate=0.0),
            MockAgent(name="fallback2", failure_rate=0.0),
        ]

    @pytest.fixture
    def pool(self, agents: list[MockAgent]) -> AgentPool:
        """Create agent pool."""
        return AgentPool(agents)

    @pytest.mark.asyncio
    async def test_primary_agent_success(self, pool: AgentPool):
        """Test normal operation with primary agent."""
        result, agent_used = await pool.generate("test", preferred_agent="primary")

        assert "primary" in result
        assert agent_used == "primary"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, pool: AgentPool):
        """Test fallback when primary agent fails."""
        # Make primary always fail
        pool.agents["primary"].failure_rate = 1.0

        # Should fall back to fallback1 or fallback2
        result, agent_used = await pool.generate("test", preferred_agent="primary")

        assert agent_used in ["fallback1", "fallback2"]

    @pytest.mark.asyncio
    async def test_circuit_opens_after_repeated_failures(self, pool: AgentPool):
        """Test circuit breaker opens after repeated failures."""
        pool.agents["primary"].failure_rate = 1.0

        # Make multiple requests to trigger circuit
        for _ in range(5):
            try:
                await pool.generate("test", preferred_agent="primary")
            except Exception:
                pass

        # Circuit should be open for primary
        assert pool.breakers["primary"].is_open

    @pytest.mark.asyncio
    async def test_recovery_after_agent_restored(self, pool: AgentPool):
        """Test recovery when failed agent comes back."""
        agent = pool.agents["primary"]
        agent.failure_rate = 1.0

        # Trigger circuit open
        for _ in range(5):
            try:
                await pool.generate("test", preferred_agent="primary")
            except Exception:
                pass

        assert pool.breakers["primary"].is_open

        # Restore agent
        agent.failure_rate = 0.0

        # Wait for cooldown
        await asyncio.sleep(1.5)

        # Should be able to use primary again (half-open state)
        assert pool.breakers["primary"].can_proceed()

        result, agent_used = await pool.generate("test", preferred_agent="primary")
        assert agent_used == "primary"

    @pytest.mark.asyncio
    async def test_cascading_agent_failures(self, pool: AgentPool):
        """Test handling of multiple agents failing."""
        # Fail all agents
        for agent in pool.agents.values():
            agent.failure_rate = 1.0

        # Should eventually exhaust all options
        with pytest.raises(RuntimeError, match="All agents unavailable"):
            for _ in range(20):
                await pool.generate("test")

    @pytest.mark.asyncio
    async def test_partial_recovery(self, pool: AgentPool):
        """Test partial recovery when some agents return."""
        # Fail all agents
        for agent in pool.agents.values():
            agent.failure_rate = 1.0

        # Trigger circuits
        for _ in range(15):
            try:
                await pool.generate("test")
            except Exception:
                pass

        # All circuits should be open
        for breaker in pool.breakers.values():
            assert breaker.is_open

        # Restore one agent
        pool.agents["fallback2"].failure_rate = 0.0

        # Wait for cooldown
        await asyncio.sleep(1.5)

        # Should now work with fallback2
        result, agent_used = await pool.generate("test")
        assert agent_used == "fallback2"


class TestAgentLoadBalancing:
    """Test agent selection under various failure conditions."""

    @pytest.fixture
    def diverse_pool(self) -> AgentPool:
        """Create pool with agents of varying reliability."""
        agents = [
            MockAgent(name="reliable", failure_rate=0.1),
            MockAgent(name="moderate", failure_rate=0.3),
            MockAgent(name="unreliable", failure_rate=0.7),
        ]
        return AgentPool(agents)

    @pytest.mark.asyncio
    async def test_load_distribution_under_failures(self, diverse_pool: AgentPool):
        """Test how load distributes when agents fail."""
        agent_counts: dict[str, int] = {"reliable": 0, "moderate": 0, "unreliable": 0}

        for _ in range(50):
            try:
                _, agent_used = await diverse_pool.generate("test")
                agent_counts[agent_used] += 1
            except Exception:
                pass

        # Reliable agent should handle most requests
        # (since others will fail and circuit will open)
        assert agent_counts["reliable"] >= agent_counts["unreliable"]

    @pytest.mark.asyncio
    async def test_concurrent_requests_during_failures(self, diverse_pool: AgentPool):
        """Test concurrent request handling during agent failures."""
        results = []
        errors = []

        async def make_request():
            try:
                result, agent = await diverse_pool.generate("test")
                results.append((result, agent))
            except Exception as e:
                errors.append(str(e))

        # Send many concurrent requests
        await asyncio.gather(*[make_request() for _ in range(30)])

        # Should have some successes
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_gradual_degradation(self, diverse_pool: AgentPool):
        """Test gradual service degradation as agents fail."""
        # Start with all agents working
        initial_results = []
        for _ in range(10):
            try:
                _, agent = await diverse_pool.generate("test")
                initial_results.append(agent)
            except Exception:
                pass

        # Degrade agents progressively
        diverse_pool.agents["unreliable"].failure_rate = 1.0
        await asyncio.sleep(0.1)

        mid_results = []
        for _ in range(10):
            try:
                _, agent = await diverse_pool.generate("test")
                mid_results.append(agent)
            except Exception:
                pass

        diverse_pool.agents["moderate"].failure_rate = 1.0
        await asyncio.sleep(0.1)

        final_results = []
        for _ in range(10):
            try:
                _, agent = await diverse_pool.generate("test")
                final_results.append(agent)
            except Exception:
                pass

        # Should increasingly use reliable agent
        if final_results:
            assert final_results.count("reliable") >= mid_results.count("reliable") * 0.5


class TestAgentTimeouts:
    """Test handling of agent timeouts."""

    @pytest.fixture
    def slow_pool(self) -> AgentPool:
        """Create pool with agents of varying latency."""
        agents = [
            MockAgent(name="fast", latency_ms=10),
            MockAgent(name="medium", latency_ms=100),
            MockAgent(name="slow", latency_ms=500),
        ]
        return AgentPool(agents)

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self, slow_pool: AgentPool):
        """Test that timeouts trigger fallback behavior."""
        # Make slow agent very slow
        slow_pool.agents["slow"].latency_ms = 5000  # 5 seconds

        async def generate_with_timeout():
            try:
                return await asyncio.wait_for(
                    slow_pool.generate("test", preferred_agent="slow"),
                    timeout=0.5,
                )
            except asyncio.TimeoutError:
                slow_pool.breakers["slow"].record_failure()
                return await slow_pool.generate("test")

        result, agent = await generate_with_timeout()

        # Should have used faster agent
        assert agent in ["fast", "medium"]

    @pytest.mark.asyncio
    async def test_latency_based_selection(self, slow_pool: AgentPool):
        """Test that system naturally prefers faster agents."""
        response_times = []

        for _ in range(10):
            start = asyncio.get_event_loop().time()
            await slow_pool.generate("test", preferred_agent="fast")
            response_times.append(asyncio.get_event_loop().time() - start)

        # Average should be close to fast agent latency
        avg_time = sum(response_times) / len(response_times)
        assert avg_time < 0.1  # 100ms


class TestAgentFailurePatterns:
    """Test specific failure patterns."""

    @pytest.mark.asyncio
    async def test_intermittent_failures(self):
        """Test handling of intermittent failures."""
        agent = MockAgent(name="flaky", failure_rate=0.5)
        breaker = get_circuit_breaker("flaky-agent-test", failure_threshold=5, cooldown_seconds=1.0)

        successes = 0
        failures = 0

        for _ in range(30):
            if not breaker.can_proceed():
                await asyncio.sleep(1.1)  # Wait for cooldown
                continue

            try:
                await agent.generate("test")
                breaker.record_success()
                successes += 1
            except Exception:
                breaker.record_failure()
                failures += 1

        # Should have mix of successes and failures
        assert successes > 0
        assert failures > 0

    @pytest.mark.asyncio
    async def test_burst_failures(self):
        """Test handling of burst failures followed by recovery."""
        agent = MockAgent(name="bursty")
        breaker = get_circuit_breaker(
            "bursty-agent-test", failure_threshold=3, cooldown_seconds=0.5
        )

        # Normal operation
        for _ in range(5):
            await agent.generate("test")
            breaker.record_success()

        assert not breaker.is_open

        # Burst of failures
        agent.failure_rate = 1.0
        for _ in range(5):
            try:
                await agent.generate("test")
            except Exception:
                breaker.record_failure()

        assert breaker.is_open

        # Recovery
        agent.failure_rate = 0.0
        await asyncio.sleep(0.6)

        # Should be able to recover (half-open)
        assert breaker.can_proceed()
        await agent.generate("test")
        breaker.record_success()

    @pytest.mark.asyncio
    async def test_correlated_failures(self):
        """Test handling of correlated failures across agents."""
        agents = [
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
            MockAgent(name="agent3"),
        ]
        pool = AgentPool(agents)

        # Simulate coordinated outage
        for agent in agents:
            agent.is_available = False

        # All requests should fail
        failed_count = 0
        for _ in range(10):
            try:
                await pool.generate("test")
            except Exception:
                failed_count += 1

        assert failed_count == 10

        # Gradual recovery
        agents[0].is_available = True
        agents[0].failure_rate = 0.0

        # Wait for circuit cooldown
        await asyncio.sleep(1.5)

        # Should now work with agent1
        result, agent_used = await pool.generate("test")
        assert agent_used == "agent1"
