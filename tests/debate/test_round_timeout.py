"""Tests for round timeout behavior with mocked agent delays."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.termination_checker import TerminationChecker


@dataclass
class _StubProtocol:
    """Minimal protocol stub with round_timeout_seconds."""

    round_timeout_seconds: int = 1
    early_stopping: bool = True
    early_stop_threshold: float = 0.6
    min_rounds_before_early_stop: int = 1
    min_rounds: int = 1
    use_judge: bool = False
    rounds: int = 5


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.agent_id = name
    return agent


class TestRoundTimeoutExceedingDelay:
    """Verify that agent responses exceeding round_timeout_seconds are handled."""

    @pytest.mark.asyncio
    async def test_early_stop_check_returns_continue_on_timeout(self):
        """When agents exceed round_timeout_seconds the early-stop check
        should catch asyncio.TimeoutError and return True (continue debate)."""
        protocol = _StubProtocol(round_timeout_seconds=1)
        agents = [_make_agent("slow-agent")]

        async def slow_generate(agent, prompt, context):
            await asyncio.sleep(5)
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=agents,
            generate_fn=slow_generate,
            task="test task",
        )

        should_continue = await checker.check_early_stopping(round_num=2, proposals={}, context=[])
        # On timeout the checker defaults to continuing the debate
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_fast_agents_allow_normal_stop_vote(self):
        """When agents respond within the timeout, votes are counted normally."""
        protocol = _StubProtocol(round_timeout_seconds=5)
        agents = [_make_agent(f"agent-{i}") for i in range(3)]

        async def fast_generate(agent, prompt, context):
            await asyncio.sleep(0.01)
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=agents,
            generate_fn=fast_generate,
            task="test task",
        )

        should_continue = await checker.check_early_stopping(round_num=2, proposals={}, context=[])
        # All agents voted STOP and threshold is met -> should NOT continue
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_mixed_slow_fast_agents_timeout(self):
        """If any agent is slow enough to cause a timeout, the whole
        gather times out and the checker falls back to continue."""
        protocol = _StubProtocol(round_timeout_seconds=1)
        agents = [_make_agent("fast"), _make_agent("slow")]

        call_count = 0

        async def mixed_generate(agent, prompt, context):
            nonlocal call_count
            call_count += 1
            if agent.name == "slow":
                await asyncio.sleep(5)
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=agents,
            generate_fn=mixed_generate,
            task="test task",
        )

        should_continue = await checker.check_early_stopping(round_num=2, proposals={}, context=[])
        assert should_continue is True
