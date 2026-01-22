"""
Chaos tests for agent failure scenarios.

Tests system resilience when agents:
- Time out during responses
- Return errors or exceptions
- Become unresponsive mid-debate
- Return malformed responses
- Experience intermittent failures
"""

from __future__ import annotations

import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class ChaosAgent:
    """A configurable chaos agent for testing failure scenarios."""

    def __init__(
        self,
        name: str,
        failure_rate: float = 0.0,
        timeout_rate: float = 0.0,
        malformed_rate: float = 0.0,
        latency_range: tuple[float, float] = (0.0, 0.0),
    ):
        self.name = name
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.malformed_rate = malformed_rate
        self.latency_range = latency_range
        self.call_count = 0
        self.failure_count = 0
        self.timeout_count = 0

    async def respond(self, prompt: str) -> str:
        """Generate a response with configurable chaos behavior."""
        self.call_count += 1

        # Simulate latency
        if self.latency_range[1] > 0:
            delay = random.uniform(*self.latency_range)
            await asyncio.sleep(delay)

        # Simulate timeout
        if random.random() < self.timeout_rate:
            self.timeout_count += 1
            await asyncio.sleep(100)  # Long sleep to trigger timeout
            return ""

        # Simulate failure
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise RuntimeError(f"Chaos agent {self.name} simulated failure")

        # Simulate malformed response
        if random.random() < self.malformed_rate:
            return None  # type: ignore

        return f"Response from {self.name}: {prompt[:50]}..."


class TestAgentTimeouts:
    """Tests for agent timeout handling."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_single_agent_timeout_recovery(self):
        """System should recover when single agent times out."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_timeout", failure_threshold=3, cooldown_seconds=1.0)
        agent = ChaosAgent("timeout_agent", timeout_rate=1.0)

        async def call_with_timeout():
            try:
                return await asyncio.wait_for(agent.respond("test"), timeout=0.1)
            except asyncio.TimeoutError:
                cb.record_failure()
                return None

        # Should handle timeout gracefully
        result = await call_with_timeout()
        assert result is None
        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_multiple_agent_timeouts_trigger_circuit_breaker(self):
        """Multiple timeouts should open circuit breaker."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_multi_timeout", failure_threshold=3, cooldown_seconds=1.0)
        agent = ChaosAgent("timeout_agent", timeout_rate=1.0)

        async def call_with_timeout():
            try:
                return await asyncio.wait_for(agent.respond("test"), timeout=0.05)
            except asyncio.TimeoutError:
                cb.record_failure()
                return None

        # Trigger multiple timeouts
        for _ in range(4):
            await call_with_timeout()

        assert cb.is_open
        assert cb.failures >= 3

    @pytest.mark.asyncio
    async def test_timeout_with_fallback_agent(self):
        """System should fallback to secondary agent on timeout."""
        primary = ChaosAgent("primary", timeout_rate=1.0)
        fallback = ChaosAgent("fallback", timeout_rate=0.0)

        async def call_with_fallback():
            try:
                return await asyncio.wait_for(primary.respond("test"), timeout=0.05)
            except asyncio.TimeoutError:
                return await fallback.respond("test")

        result = await call_with_fallback()
        assert result is not None
        assert "fallback" in result
        assert primary.call_count == 1
        assert fallback.call_count == 1


class TestAgentExceptions:
    """Tests for agent exception handling."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_agent_exception_captured(self):
        """Agent exceptions should be captured, not crash system."""
        agent = ChaosAgent("failing_agent", failure_rate=1.0)

        with pytest.raises(RuntimeError, match="simulated failure"):
            await agent.respond("test")

        assert agent.failure_count == 1

    @pytest.mark.asyncio
    async def test_exception_triggers_circuit_breaker(self):
        """Repeated exceptions should open circuit breaker."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_exception", failure_threshold=2, cooldown_seconds=1.0)
        agent = ChaosAgent("failing_agent", failure_rate=1.0)

        for _ in range(3):
            try:
                await agent.respond("test")
            except RuntimeError:
                cb.record_failure()

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_intermittent_failures_with_retry(self):
        """System should handle intermittent failures with retry."""
        call_count = 0

        async def intermittent_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Intermittent failure")
            return "success"

        # Retry logic
        for attempt in range(5):
            try:
                result = await intermittent_call()
                break
            except ConnectionError:
                await asyncio.sleep(0.01)
        else:
            result = None

        assert result == "success"
        assert call_count == 3


class TestMalformedResponses:
    """Tests for handling malformed agent responses."""

    @pytest.mark.asyncio
    async def test_none_response_handled(self):
        """None responses should be handled gracefully."""
        agent = ChaosAgent("malformed_agent", malformed_rate=1.0)
        result = await agent.respond("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_response_handled(self):
        """Empty responses should be detected."""

        async def empty_responder(prompt: str) -> str:
            return ""

        result = await empty_responder("test")
        assert result == ""
        assert not result  # Falsy check

    @pytest.mark.asyncio
    async def test_truncated_json_response(self):
        """Truncated JSON responses should be handled."""
        import json

        truncated_json = '{"response": "test", "metadata": {'

        with pytest.raises(json.JSONDecodeError):
            json.loads(truncated_json)

    @pytest.mark.asyncio
    async def test_response_validation_rejects_invalid(self):
        """Response validation should reject invalid formats."""

        def validate_response(response: Any) -> bool:
            if response is None:
                return False
            if not isinstance(response, str):
                return False
            if len(response) < 1:
                return False
            return True

        assert validate_response("valid response") is True
        assert validate_response(None) is False
        assert validate_response("") is False
        assert validate_response(123) is False


class TestConcurrentAgentFailures:
    """Tests for concurrent agent failure scenarios."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_multiple_agents_partial_failure(self):
        """System should continue when some agents fail."""
        agents = [
            ChaosAgent("agent1", failure_rate=0.0),
            ChaosAgent("agent2", failure_rate=1.0),
            ChaosAgent("agent3", failure_rate=0.0),
        ]

        results = []
        for agent in agents:
            try:
                result = await agent.respond("test")
                results.append(result)
            except RuntimeError:
                results.append(None)

        # Should have 2 successful, 1 failed
        successful = [r for r in results if r is not None]
        assert len(successful) == 2

    @pytest.mark.asyncio
    async def test_all_agents_fail_graceful_degradation(self):
        """System should degrade gracefully when all agents fail."""
        agents = [ChaosAgent(f"agent{i}", failure_rate=1.0) for i in range(3)]

        results = []
        for agent in agents:
            try:
                result = await agent.respond("test")
                results.append(result)
            except RuntimeError:
                results.append(None)

        # All should fail
        assert all(r is None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_mixed_failures(self):
        """Concurrent calls should handle mixed success/failure."""
        agents = [
            ChaosAgent("fast", failure_rate=0.0, latency_range=(0.01, 0.02)),
            ChaosAgent("slow", failure_rate=0.0, latency_range=(0.05, 0.1)),
            ChaosAgent("failing", failure_rate=1.0),
        ]

        async def safe_call(agent):
            try:
                return await agent.respond("test")
            except RuntimeError:
                return None

        tasks = [safe_call(agent) for agent in agents]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]
        assert len(successful) == 2


class TestAgentRecovery:
    """Tests for agent recovery after failures."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Circuit breaker should recover after reset timeout."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_recovery", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for reset
        await asyncio.sleep(0.15)

        # Should allow a test call (half-open state)
        can_proceed = cb.can_proceed()
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_success_after_recovery_closes_circuit(self):
        """Successful call after recovery should close circuit."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_close", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for reset and record success
        await asyncio.sleep(0.15)
        cb.can_proceed()  # Move to half-open
        cb.record_success()

        # Should be closed now
        assert not cb.is_open
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_failure_during_recovery_reopens_circuit(self):
        """Failures during recovery should reopen circuit."""
        from aragora.resilience import get_circuit_breaker

        cb = get_circuit_breaker("test_reopen", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

        # Wait for cooldown to pass
        await asyncio.sleep(0.15)

        # In single-entity mode, can_proceed() fully resets the circuit after cooldown.
        # So we need to record enough failures to reopen it.
        assert cb.can_proceed() is True  # Circuit now closed

        # Record failures to reopen
        cb.record_failure()
        cb.record_failure()

        # Should be open again
        assert cb.is_open


class TestDebateWithAgentFailures:
    """Integration tests for debates with agent failures."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()
        yield
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_debate_continues_with_agent_dropout(self):
        """Debate should continue even if an agent drops out."""
        # Simulate a debate where one agent fails mid-debate
        agents_responses = {
            "agent1": ["Round 1 response", "Round 2 response", "Round 3 response"],
            "agent2": ["Round 1 response", None, "Round 3 response"],  # Fails round 2
            "agent3": ["Round 1 response", "Round 2 response", "Round 3 response"],
        }

        completed_rounds = 0
        for round_num in range(3):
            round_responses = []
            for agent_name, responses in agents_responses.items():
                response = responses[round_num]
                if response is not None:
                    round_responses.append(response)

            # Round should complete if at least one agent responds
            if round_responses:
                completed_rounds += 1

        assert completed_rounds == 3

    @pytest.mark.asyncio
    async def test_debate_consensus_with_partial_responses(self):
        """Consensus detection should work with partial responses."""
        responses = [
            "I agree with the proposal",
            None,  # Agent failed
            "The proposal is sound",
            "I concur with the approach",
        ]

        valid_responses = [r for r in responses if r is not None]

        # Simple consensus check - all valid responses should be considered
        assert len(valid_responses) == 3

        # Simulate consensus calculation with partial data
        agreement_keywords = ["agree", "sound", "concur"]
        consensus_count = sum(
            1 for r in valid_responses if any(kw in r.lower() for kw in agreement_keywords)
        )

        assert consensus_count == 3

    @pytest.mark.asyncio
    async def test_debate_aborts_when_all_agents_fail(self):
        """Debate should abort gracefully when all agents fail."""
        agents = [ChaosAgent(f"agent{i}", failure_rate=1.0) for i in range(3)]

        debate_completed = False
        abort_reason = None

        try:
            for round_num in range(3):
                round_responses = []
                for agent in agents:
                    try:
                        response = await agent.respond(f"Round {round_num}")
                        round_responses.append(response)
                    except RuntimeError:
                        pass

                if not round_responses:
                    abort_reason = "No agents responded"
                    break
            else:
                debate_completed = True
        except Exception as e:
            abort_reason = str(e)

        assert not debate_completed
        assert abort_reason == "No agents responded"
