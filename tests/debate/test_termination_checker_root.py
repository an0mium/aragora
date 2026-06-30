"""Tests for TerminationChecker class.

Tests judge-based termination and early stopping functionality.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.termination_checker import TerminationChecker
from aragora.debate.protocol import DebateProtocol


@pytest.fixture
def mock_protocol():
    """Create a mock debate protocol."""
    protocol = MagicMock(spec=DebateProtocol)
    protocol.judge_termination = True
    protocol.min_rounds_before_judge_check = 2
    protocol.early_stopping = True
    protocol.min_rounds_before_early_stop = 1
    protocol.early_stop_threshold = 0.5
    protocol.round_timeout_seconds = 30
    return protocol


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    agents = []
    for name in ["claude", "gemini", "grok"]:
        agent = MagicMock()
        agent.name = name
        agents.append(agent)
    return agents


class TestTerminationChecker:
    """Tests for TerminationChecker class."""

    def test_initialization(self, mock_protocol, mock_agents):
        """Test TerminationChecker initialization."""
        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=AsyncMock(),
            task="Test task",
        )
        assert checker.protocol is mock_protocol
        assert checker.agents == mock_agents
        assert checker.task == "Test task"

    @pytest.mark.asyncio
    async def test_check_judge_termination_disabled(self, mock_agents):
        """Test judge termination when disabled."""
        protocol = MagicMock()
        protocol.judge_termination = False

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=AsyncMock(),
            task="Test task",
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=5,
            proposals={"claude": "Proposal A"},
            context=[],
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_check_judge_termination_too_early(self, mock_protocol, mock_agents):
        """Test judge termination before minimum rounds."""
        mock_protocol.min_rounds_before_judge_check = 3

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=AsyncMock(),
            task="Test task",
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=2,  # Less than min_rounds_before_judge_check
            proposals={"claude": "Proposal A"},
            context=[],
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_check_judge_termination_conclusive(self, mock_protocol, mock_agents):
        """Test judge termination when debate is conclusive."""
        judge = MagicMock()
        judge.name = "judge_agent"

        select_judge_fn = AsyncMock(return_value=judge)
        generate_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: All issues resolved")

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals={"claude": "Proposal A"},
            context=[],
        )

        assert should_continue is False
        assert "All issues resolved" in reason

    @pytest.mark.asyncio
    async def test_check_judge_termination_not_conclusive(self, mock_protocol, mock_agents):
        """Test judge termination when debate is not conclusive."""
        judge = MagicMock()
        judge.name = "judge_agent"

        select_judge_fn = AsyncMock(return_value=judge)
        generate_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: More discussion needed")

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals={"claude": "Proposal A"},
            context=[],
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_check_early_stopping_disabled(self, mock_agents):
        """Test early stopping when disabled."""
        protocol = MagicMock()
        protocol.early_stopping = False

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=AsyncMock(),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=5,
            proposals={},
            context=[],
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_check_early_stopping_too_early(self, mock_protocol, mock_agents):
        """Test early stopping before minimum rounds."""
        mock_protocol.min_rounds_before_early_stop = 3

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=AsyncMock(),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=2,  # Less than min_rounds_before_early_stop
            proposals={},
            context=[],
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_check_early_stopping_agents_vote_continue(self, mock_protocol, mock_agents):
        """Test early stopping when agents vote to continue."""
        # All agents say CONTINUE
        generate_fn = AsyncMock(return_value="CONTINUE")

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=5,
            proposals={},
            context=[],
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_check_early_stopping_agents_vote_stop(self, mock_protocol, mock_agents):
        """Test early stopping when agents vote to stop."""
        # All agents say STOP
        generate_fn = AsyncMock(return_value="STOP")
        mock_protocol.early_stop_threshold = 0.5

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=5,
            proposals={},
            context=[],
        )

        assert should_continue is False

    @pytest.mark.asyncio
    async def test_should_terminate_combined(self, mock_protocol, mock_agents):
        """Test combined termination check."""
        judge = MagicMock()
        judge.name = "judge_agent"

        select_judge_fn = AsyncMock(return_value=judge)
        generate_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: Continue")

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_stop, reason = await checker.should_terminate(
            round_num=3,
            proposals={"claude": "Proposal A"},
            context=[],
        )

        # Judge says continue, agents say continue
        assert should_stop is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_hooks_called_on_judge_termination(self, mock_protocol, mock_agents):
        """Test that hooks are called on judge termination."""
        judge = MagicMock()
        judge.name = "judge_agent"

        select_judge_fn = AsyncMock(return_value=judge)
        generate_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: Done")

        hook_called = []
        hooks = {"on_judge_termination": lambda name, reason: hook_called.append((name, reason))}

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
            select_judge_fn=select_judge_fn,
            hooks=hooks,
        )

        await checker.check_judge_termination(
            round_num=3,
            proposals={"claude": "Proposal A"},
            context=[],
        )

        assert len(hook_called) == 1
        assert hook_called[0][0] == "judge_agent"
        assert "Done" in hook_called[0][1]

    @pytest.mark.asyncio
    async def test_hooks_called_on_early_stop(self, mock_protocol, mock_agents):
        """Test that hooks are called on early stop."""
        generate_fn = AsyncMock(return_value="STOP")
        mock_protocol.early_stop_threshold = 0.5

        hook_called = []
        hooks = {"on_early_stop": lambda r, s, t: hook_called.append((r, s, t))}

        checker = TerminationChecker(
            protocol=mock_protocol,
            agents=mock_agents,
            generate_fn=generate_fn,
            task="Test task",
            hooks=hooks,
        )

        await checker.check_early_stopping(
            round_num=5,
            proposals={},
            context=[],
        )

        assert len(hook_called) == 1
        assert hook_called[0][0] == 5  # round_num
        assert hook_called[0][1] == 3  # stop_votes (all 3 agents)
        assert hook_called[0][2] == 3  # total_votes
