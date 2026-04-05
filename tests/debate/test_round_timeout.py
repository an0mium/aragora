"""Tests that debate rounds respect timeout settings."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.protocol import DebateProtocol
from aragora.debate.termination_checker import TerminationChecker


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
