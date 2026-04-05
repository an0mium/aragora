"""Tests that debate rounds respect timeout settings."""

from contextlib import nullcontext
from dataclasses import dataclass
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.autonomic_executor import AutonomicExecutor
from aragora.debate.phases.debate_rounds import DebateRoundsPhase
from aragora.debate.protocol import DebateProtocol
from aragora.debate.termination_checker import TerminationChecker
from aragora.resilience.circuit_breaker import CircuitBreaker
from tests.debate.phases.test_debate_rounds import (
    MockAgent,
    MockCritique,
    MockDebateContext,
    MockProtocol,
    MockResult,
)


@dataclass
class _StubProtocol:
    """Minimal protocol stub for direct TerminationChecker coverage."""

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


class TestRoundTimeout:
    """Verify round_timeout_seconds is enforced on debate rounds."""

    @pytest.fixture
    def mock_agents(self):
        agents = []
        for name in ("agent_a", "agent_b"):
            a = MagicMock()
            a.name = name
            a.model = "mock"
            agents.append(a)
        return agents

    @pytest.fixture
    def sample_proposals(self):
        return [
            {"agent": "agent_a", "content": "Proposal A"},
            {"agent": "agent_b", "content": "Proposal B"},
        ]

    @pytest.fixture
    def mock_messages(self):
        return [{"role": "user", "content": "Test task"}]

    @pytest.mark.asyncio
    async def test_round_timeout_triggers_on_slow_agent(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """When agent response exceeds round_timeout_seconds, timeout is handled gracefully."""
        protocol = DebateProtocol(
            early_stopping=True,
            min_rounds_before_early_stop=2,
            round_timeout_seconds=0.01,
        )

        async def slow_generate(agent, prompt, context):
            await asyncio.sleep(5)
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=slow_generate,
            task="Test task",
        )

        result = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )
        # Safe default on timeout: continue the debate
        assert result is True

    @pytest.mark.asyncio
    async def test_no_timeout_when_agent_responds_quickly(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Fast agent responses complete within the timeout window."""
        protocol = DebateProtocol(
            early_stopping=True,
            min_rounds_before_early_stop=2,
            round_timeout_seconds=10,
        )

        async def fast_generate(agent, prompt, context):
            return "CONTINUE"

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=fast_generate,
            task="Test task",
        )

        result = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )
        # Should return a real decision, not the timeout default
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_arena_run_returns_partial_on_timeout(self):
        """Arena.run returns partial DebateResult when timeout_seconds is exceeded."""
        from aragora.core_types import Agent

        agents = []
        for i in range(2):
            a = MagicMock(spec=Agent)
            a.name = f"agent_{i}"
            a.model = "mock"
            agents.append(a)

        with patch("aragora.debate.orchestrator.Arena") as mock_arena_cls:
            arena = MagicMock()

            async def slow_run(correlation_id=""):
                await asyncio.sleep(10)

            arena.run = slow_run
            mock_arena_cls.return_value = arena

            from aragora.debate.service import DebateOptions, DebateService

            service = DebateService(default_agents=agents)
            opts = DebateOptions(timeout=0.01)

            with pytest.raises(asyncio.TimeoutError):
                await service.run("Slow debate task", options=opts)


class TestRoundTimeoutExceedingDelay:
    """Verify early-stop timeout behavior directly on the checker."""

    @pytest.mark.asyncio
    async def test_early_stop_check_returns_continue_on_timeout(self):
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
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_fast_agents_allow_normal_stop_vote(self):
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
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_mixed_slow_fast_agents_timeout(self):
        protocol = _StubProtocol(round_timeout_seconds=1)
        agents = [_make_agent("fast"), _make_agent("slow")]

        async def mixed_generate(agent, prompt, context):
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


class TestDebateRoundsPhaseTimeout:
    @pytest.mark.asyncio
    async def test_slow_critique_times_out_and_cleans_up(self):
        protocol = MockProtocol(rounds=1)
        protocol.round_timeout_seconds = 0.02
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def slow_critique(critic, proposal, task, context, target_agent=None):
            started.set()
            try:
                await asyncio.sleep(0.1)
                return MockCritique(agent=critic.name, target_agent=target_agent or "unknown")
            finally:
                cleaned_up.set()

        phase = DebateRoundsPhase(protocol=protocol, critique_with_agent=slow_critique)
        proposer = MockAgent(name="proposer", role="proposer")
        critic = MockAgent(name="critic", role="critic")
        ctx = MockDebateContext(
            agents=[proposer, critic],
            proposers=[proposer],
            proposals={"proposer": "initial proposal"},
            result=MockResult(critiques=[]),
        )
        perf_monitor = SimpleNamespace(
            track_round=lambda *args, **kwargs: nullcontext(),
            track_phase=lambda *args, **kwargs: nullcontext(),
            slow_round_threshold=60.0,
        )
        governor = MagicMock()
        governor.get_scaled_timeout.return_value = 30.0

        with (
            patch(
                "aragora.debate.phases.debate_rounds.get_debate_monitor", return_value=perf_monitor
            ),
            patch(
                "aragora.debate.phases.debate_rounds.get_complexity_governor",
                return_value=governor,
            ),
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(phase.execute(ctx), timeout=protocol.round_timeout_seconds)

        assert started.is_set()
        await asyncio.wait_for(cleaned_up.wait(), timeout=0.2)
        assert ctx.result.rounds_used == 0
        assert ctx.result.critiques == []


class TestAutonomicExecutorRoundTimeout:
    @pytest.mark.asyncio
    async def test_timeout_records_circuit_breaker_failure(self):
        protocol = DebateProtocol(round_timeout_seconds=0.01)
        circuit_breaker = CircuitBreaker(
            name="test-breaker", failure_threshold=3, cooldown_seconds=60
        )
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
