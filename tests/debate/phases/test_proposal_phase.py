"""
Tests for the ProposalPhase module.

Tests cover:
- ProposalPhase initialization
- Proposal generation with mocked agents
- Circuit breaker filtering
- Hook execution
- Parallel proposal generation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.proposal_phase import ProposalPhase


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, response: str = "Test proposal"):
        self.name = name
        self.response = response
        self.role = "proposer"
        self.model = "mock-model"

    async def generate(self, prompt: str, context=None) -> str:
        return self.response


class MockDebateContext:
    """Mock debate context for testing."""

    def __init__(self, agents=None, task="Test task"):
        self.agents = agents or []
        self.proposers = [a for a in self.agents if getattr(a, "role", "") == "proposer"]
        self.env = MagicMock()
        self.env.task = task
        self.proposals = {}
        self.messages = []
        self.cancellation_token = None
        self.hook_manager = None


class TestProposalPhaseInit:
    """Tests for ProposalPhase initialization."""

    def test_init_minimal(self):
        """ProposalPhase can be initialized with no arguments."""
        phase = ProposalPhase()

        assert phase.circuit_breaker is None
        assert phase.position_tracker is None
        assert phase.hooks == {}

    def test_init_with_circuit_breaker(self):
        """ProposalPhase stores circuit breaker."""
        mock_cb = MagicMock()
        phase = ProposalPhase(circuit_breaker=mock_cb)

        assert phase.circuit_breaker is mock_cb

    def test_init_with_hooks(self):
        """ProposalPhase stores hooks."""
        hooks = {"on_debate_start": MagicMock()}
        phase = ProposalPhase(hooks=hooks)

        assert phase.hooks == hooks

    def test_init_with_callbacks(self):
        """ProposalPhase stores callback functions."""
        build_prompt = MagicMock()
        generate = AsyncMock()

        phase = ProposalPhase(
            build_proposal_prompt=build_prompt,
            generate_with_agent=generate,
        )

        assert phase._build_proposal_prompt is build_prompt
        assert phase._generate_with_agent is generate


class TestProposalPhaseFilterProposers:
    """Tests for _filter_proposers method."""

    def test_filter_without_circuit_breaker(self):
        """Without circuit breaker, all proposers pass through."""
        phase = ProposalPhase()
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents

        result = phase._filter_proposers(ctx)

        assert result == agents

    def test_filter_with_circuit_breaker(self):
        """Circuit breaker filters unavailable agents."""
        mock_cb = MagicMock()
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        mock_cb.filter_available_agents.return_value = [agents[0]]

        phase = ProposalPhase(circuit_breaker=mock_cb)
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents

        result = phase._filter_proposers(ctx)

        assert result == [agents[0]]
        mock_cb.filter_available_agents.assert_called_once_with(agents)

    def test_filter_handles_circuit_breaker_error(self):
        """Circuit breaker errors fall back to all proposers."""
        mock_cb = MagicMock()
        mock_cb.filter_available_agents.side_effect = RuntimeError("CB error")

        agents = [MockAgent("agent1"), MockAgent("agent2")]
        phase = ProposalPhase(circuit_breaker=mock_cb)
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents

        result = phase._filter_proposers(ctx)

        # Should fall back to all proposers on error
        assert result == agents


class TestProposalPhaseEmitDebateStart:
    """Tests for _emit_debate_start method."""

    def test_emit_calls_hook(self):
        """Debate start hook is called with correct arguments."""
        mock_hook = MagicMock()
        hooks = {"on_debate_start": mock_hook}
        phase = ProposalPhase(hooks=hooks)

        agents = [MockAgent("agent1"), MockAgent("agent2")]
        ctx = MockDebateContext(agents=agents, task="Test task")

        phase._emit_debate_start(ctx)

        mock_hook.assert_called_once_with("Test task", ["agent1", "agent2"])

    def test_emit_without_hook(self):
        """No error when hook is not defined."""
        phase = ProposalPhase()
        ctx = MockDebateContext(agents=[MockAgent("a1")])

        # Should not raise
        phase._emit_debate_start(ctx)


class TestProposalPhaseExecute:
    """Tests for execute method."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Execute runs without error with minimal setup."""
        phase = ProposalPhase()
        agents = [MockAgent("agent1", response="Proposal 1")]
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents

        # Mock the parallel generation to avoid complex async setup
        with patch.object(phase, "_generate_proposals_parallel", new_callable=AsyncMock):
            await phase.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_checks_cancellation(self):
        """Execute raises on cancelled token."""
        from aragora.debate.cancellation import DebateCancelled

        phase = ProposalPhase()
        ctx = MockDebateContext(agents=[MockAgent("a1")])
        ctx.cancellation_token = MagicMock()
        ctx.cancellation_token.is_cancelled = True
        ctx.cancellation_token.reason = "User cancelled"

        with pytest.raises(DebateCancelled):
            await phase.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_calls_role_update(self):
        """Execute calls role assignment callback."""
        mock_update = MagicMock()
        phase = ProposalPhase(update_role_assignments=mock_update)
        agents = [MockAgent("agent1")]
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents

        with patch.object(phase, "_generate_proposals_parallel", new_callable=AsyncMock):
            await phase.execute(ctx)

        mock_update.assert_called_once_with(round_num=0)

    @pytest.mark.asyncio
    async def test_execute_notifies_spectator(self):
        """Execute notifies spectator of debate start."""
        mock_notify = MagicMock()
        phase = ProposalPhase(notify_spectator=mock_notify)
        agents = [MockAgent("agent1")]
        ctx = MockDebateContext(agents=agents, task="Test debate task")
        ctx.proposers = agents

        with patch.object(phase, "_generate_proposals_parallel", new_callable=AsyncMock):
            await phase.execute(ctx)

        mock_notify.assert_called()
        # First call should be debate_start
        call_args = mock_notify.call_args_list[0]
        assert call_args[0][0] == "debate_start"

    @pytest.mark.asyncio
    async def test_execute_extracts_citations(self):
        """Execute extracts citation needs after proposals."""
        mock_extract = MagicMock()
        phase = ProposalPhase(extract_citation_needs=mock_extract)
        agents = [MockAgent("agent1")]
        ctx = MockDebateContext(agents=agents)
        ctx.proposers = agents
        ctx.proposals = {"agent1": "Test proposal"}

        with patch.object(phase, "_generate_proposals_parallel", new_callable=AsyncMock):
            await phase.execute(ctx)

        mock_extract.assert_called_once_with(ctx.proposals)


class TestProposalPhaseGenerateProposalsParallel:
    """Tests for _generate_proposals_parallel method."""

    @pytest.mark.asyncio
    async def test_handles_empty_proposers(self):
        """Empty proposers list logs warning and returns."""
        phase = ProposalPhase()
        ctx = MockDebateContext(agents=[])

        # Should not raise, just log warning
        await phase._generate_proposals_parallel(ctx, [])

    @pytest.mark.asyncio
    async def test_creates_tasks_for_each_proposer(self):
        """Creates async tasks for each proposer."""
        phase = ProposalPhase()
        agents = [MockAgent("a1"), MockAgent("a2")]
        ctx = MockDebateContext(agents=agents)

        # Mock _generate_single_proposal to track calls
        with patch.object(phase, "_generate_single_proposal", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = None

            # Use smaller stagger for testing
            with patch(
                "aragora.debate.phases.proposal_phase.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                await phase._generate_proposals_parallel(ctx, agents)

            # Should be called for each agent
            assert mock_gen.call_count == 2


class TestProposalPhaseIntegration:
    """Integration tests for ProposalPhase."""

    @pytest.mark.asyncio
    async def test_full_flow_with_mocks(self):
        """Test complete proposal flow with all mocks."""
        # Setup
        mock_build = MagicMock(return_value="Build prompt for agent")
        mock_generate = AsyncMock(return_value="Generated proposal")
        mock_timeout = AsyncMock(side_effect=lambda coro, **kw: coro)
        mock_notify = MagicMock()
        mock_update = MagicMock()

        phase = ProposalPhase(
            build_proposal_prompt=mock_build,
            generate_with_agent=mock_generate,
            with_timeout=mock_timeout,
            notify_spectator=mock_notify,
            update_role_assignments=mock_update,
        )

        agents = [MockAgent("proposer1", response="My proposal")]
        ctx = MockDebateContext(agents=agents, task="Integration test task")
        ctx.proposers = agents

        # Execute with mocked parallel generation
        with patch.object(phase, "_generate_proposals_parallel", new_callable=AsyncMock):
            await phase.execute(ctx)

        # Verify flow
        mock_update.assert_called_once()
        mock_notify.assert_called()
