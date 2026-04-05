import asyncio
from dataclasses import dataclass

import pytest

from aragora.debate.autonomic_executor import AutonomicExecutor
from aragora.debate.protocol import DebateProtocol
from aragora.debate.termination_checker import TerminationChecker
from aragora.resilience.circuit_breaker import CircuitBreaker


@dataclass
class MockAgent:
    name: str


@pytest.mark.asyncio
async def test_round_timeout_setting_raises_for_slow_agent():
    protocol = DebateProtocol(round_timeout_seconds=0.01)
    circuit_breaker = CircuitBreaker(name="test-breaker", failure_threshold=3, cooldown_seconds=60)
    executor = AutonomicExecutor(circuit_breaker=circuit_breaker, default_timeout=5.0)

    async def slow_response():
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError, match="slow-agent timed out"):
        await executor.with_timeout(
            slow_response(),
            "slow-agent",
            timeout_seconds=protocol.round_timeout_seconds,
        )

    assert circuit_breaker._failures.get("slow-agent", 0) >= 1


@pytest.mark.asyncio
async def test_round_timeout_during_early_stopping_continues_gracefully():
    protocol = DebateProtocol(
        early_stopping=True,
        min_rounds_before_early_stop=1,
        round_timeout_seconds=0.01,
    )
    agents = [MockAgent(name="agent1"), MockAgent(name="agent2")]

    async def slow_generate(agent, prompt, context):
        await asyncio.sleep(1)
        return "STOP"

    checker = TerminationChecker(
        protocol=protocol,
        agents=agents,
        generate_fn=slow_generate,
        task="Test task",
    )

    should_continue = await checker.check_early_stopping(
        round_num=1,
        proposals={"agent1": "proposal"},
        context=[],
    )

    assert should_continue is True
