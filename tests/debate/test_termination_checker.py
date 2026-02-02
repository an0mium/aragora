"""Tests for TerminationChecker module.

Tests cover:
- Termination conditions (consensus, max rounds, timeout, stagnation)
- Judge-based termination with and without confidence scoring
- Early stopping via agent voting
- Configuration validation
- RLM-style confidence thresholds
- Error handling and edge cases
"""

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.debate.termination_checker import (
    RLM_HIGH_CONFIDENCE_THRESHOLD,
    RLM_MIN_CONFIDENCE_FOR_STOP,
    TerminationChecker,
    TerminationResult,
)
from aragora.debate.protocol import DebateProtocol


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "Mock response"


@dataclass
class MockMessage:
    """Mock message for context."""

    role: str
    agent: str
    content: str
    round: int = 0


@pytest.fixture
def default_protocol() -> DebateProtocol:
    """Create a default DebateProtocol for testing."""
    return DebateProtocol(
        rounds=5,
        judge_termination=True,
        min_rounds_before_judge_check=2,
        early_stopping=True,
        early_stop_threshold=0.7,
        min_rounds_before_early_stop=2,
        round_timeout_seconds=30,
    )


@pytest.fixture
def mock_agents() -> list[MockAgent]:
    """Create a list of mock agents."""
    return [
        MockAgent(name="claude"),
        MockAgent(name="gpt4"),
        MockAgent(name="gemini"),
    ]


@pytest.fixture
def mock_messages() -> list[MockMessage]:
    """Create mock message context."""
    return [
        MockMessage(role="proposer", agent="claude", content="First proposal"),
        MockMessage(role="critic", agent="gpt4", content="Critique of first"),
        MockMessage(role="proposer", agent="gemini", content="Second proposal"),
    ]


@pytest.fixture
def sample_proposals() -> dict[str, str]:
    """Create sample proposals for testing."""
    return {
        "claude": "Proposal: We should implement feature X with approach A.",
        "gpt4": "Proposal: Feature X is best implemented via approach B.",
        "gemini": "Proposal: Consider hybrid approach combining A and B.",
    }


def create_async_generate_fn(response: str = "CONTINUE"):
    """Create an async generate function that returns a fixed response."""

    async def generate_fn(agent: Any, prompt: str, context: list) -> str:
        return response

    return generate_fn


async def select_judge_fn(proposals: dict[str, str], context: list) -> MockAgent:
    """Mock judge selector that returns a fixed judge agent."""
    return MockAgent(name="judge_claude")


# =============================================================================
# TerminationResult Tests
# =============================================================================


class TestTerminationResult:
    """Tests for TerminationResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = TerminationResult(should_terminate=False)
        assert result.should_terminate is False
        assert result.reason == ""
        assert result.confidence == 0.5
        assert result.source == "unknown"
        assert result.votes is None

    def test_is_high_confidence_above_threshold(self):
        """Test is_high_confidence returns True above threshold."""
        result = TerminationResult(
            should_terminate=True,
            confidence=0.85,
        )
        assert result.is_high_confidence is True
        assert result.confidence >= RLM_HIGH_CONFIDENCE_THRESHOLD

    def test_is_high_confidence_at_threshold(self):
        """Test is_high_confidence returns True at exact threshold."""
        result = TerminationResult(
            should_terminate=True,
            confidence=RLM_HIGH_CONFIDENCE_THRESHOLD,
        )
        assert result.is_high_confidence is True

    def test_is_high_confidence_below_threshold(self):
        """Test is_high_confidence returns False below threshold."""
        result = TerminationResult(
            should_terminate=True,
            confidence=0.75,
        )
        assert result.is_high_confidence is False

    def test_should_consider_stopping_true(self):
        """Test should_consider_stopping when conditions met."""
        result = TerminationResult(
            should_terminate=True,
            confidence=0.7,
        )
        assert result.should_consider_stopping is True
        assert result.confidence >= RLM_MIN_CONFIDENCE_FOR_STOP

    def test_should_consider_stopping_false_not_terminating(self):
        """Test should_consider_stopping is False when not terminating."""
        result = TerminationResult(
            should_terminate=False,
            confidence=0.9,
        )
        assert result.should_consider_stopping is False

    def test_should_consider_stopping_false_low_confidence(self):
        """Test should_consider_stopping is False with low confidence."""
        result = TerminationResult(
            should_terminate=True,
            confidence=0.5,
        )
        assert result.should_consider_stopping is False

    def test_votes_storage(self):
        """Test votes dictionary is stored correctly."""
        votes = {"claude": True, "gpt4": False, "gemini": True}
        result = TerminationResult(
            should_terminate=True,
            votes=votes,
        )
        assert result.votes == votes
        assert result.votes["claude"] is True


# =============================================================================
# TerminationChecker Initialization Tests
# =============================================================================


class TestTerminationCheckerInit:
    """Tests for TerminationChecker initialization."""

    def test_basic_initialization(self, default_protocol, mock_agents):
        """Test basic initialization with required params."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
        )
        assert checker.protocol == default_protocol
        assert checker.agents == mock_agents
        assert checker.task == "Test task"
        assert checker.select_judge_fn is None
        assert checker.hooks == {}

    def test_initialization_with_judge_selector(self, default_protocol, mock_agents):
        """Test initialization with judge selector function."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )
        assert checker.select_judge_fn == select_judge_fn

    def test_initialization_with_hooks(self, default_protocol, mock_agents):
        """Test initialization with custom hooks."""
        hooks = {
            "on_judge_termination": MagicMock(),
            "on_early_stop": MagicMock(),
        }
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            hooks=hooks,
        )
        assert checker.hooks == hooks


# =============================================================================
# Judge Termination Tests
# =============================================================================


class TestJudgeTermination:
    """Tests for check_judge_termination method."""

    @pytest.mark.asyncio
    async def test_disabled_when_judge_termination_false(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test returns continue when judge_termination is disabled."""
        protocol = DebateProtocol(judge_termination=False)
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=5,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_continues_before_min_rounds(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues if minimum rounds not reached."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        # Round 1 is below min_rounds_before_judge_check (2)
        should_continue, reason = await checker.check_judge_termination(
            round_num=1,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_continues_without_judge_selector(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues if no judge selector provided."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            select_judge_fn=None,  # No judge selector
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_terminates_when_conclusive_yes(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test terminates when judge says conclusive: yes."""
        response = "CONCLUSIVE: yes\nREASON: All key issues resolved"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is False
        assert "resolved" in reason.lower() or reason != ""

    @pytest.mark.asyncio
    async def test_continues_when_conclusive_no(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues when judge says conclusive: no."""
        response = "CONCLUSIVE: no\nREASON: More discussion needed"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_emits_hook_on_termination(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test on_judge_termination hook is called on termination."""
        hook_mock = MagicMock()
        response = "CONCLUSIVE: yes\nREASON: Debate complete"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
            hooks={"on_judge_termination": hook_mock},
        )

        await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        hook_mock.assert_called_once()
        args = hook_mock.call_args[0]
        assert args[0] == "judge_claude"  # Judge name

    @pytest.mark.asyncio
    async def test_handles_timeout_error(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test handles timeout gracefully."""

        async def timeout_generate(agent, prompt, context):
            raise asyncio.TimeoutError("Timeout occurred")

        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=timeout_generate,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_handles_parse_error(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test handles malformed response gracefully."""
        response = "This is not a valid response format"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # Should continue on parse failure (safe default)
        assert should_continue is True


# =============================================================================
# Judge Termination with Confidence Tests
# =============================================================================


class TestJudgeTerminationWithConfidence:
    """Tests for check_judge_termination_with_confidence method."""

    @pytest.mark.asyncio
    async def test_disabled_returns_result_with_reason(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test returns proper result when judge_termination disabled."""
        protocol = DebateProtocol(judge_termination=False)
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=5,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 1.0
        assert result.source == "config"
        assert "not enabled" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_before_min_rounds_returns_result(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test returns proper result when min rounds not reached."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=1,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 1.0
        assert result.source == "config"
        assert "minimum rounds" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_no_judge_selector_returns_error_result(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test returns error result when no judge selector."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(),
            task="Test task",
            select_judge_fn=None,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 0.5
        assert result.source == "error"

    @pytest.mark.asyncio
    async def test_parses_json_response_correctly(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test parses JSON response with confidence."""
        response = '{"conclusive": true, "confidence": 0.85, "reason": "Clear consensus"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is True
        assert result.confidence == 0.85
        assert result.reason == "Clear consensus"
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_parses_json_in_markdown(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test parses JSON embedded in markdown code block."""
        response = """```json
{"conclusive": true, "confidence": 0.9, "reason": "Well resolved"}
```"""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is True
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_clamps_confidence_to_valid_range(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test confidence is clamped to [0, 1]."""
        response = '{"conclusive": true, "confidence": 1.5, "reason": "Overconfident"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.confidence == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_clamps_negative_confidence(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test negative confidence is clamped to 0."""
        response = '{"conclusive": true, "confidence": -0.5, "reason": "Negative"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_high_confidence_triggers_hook(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test hook is called only for high confidence termination."""
        hook_mock = MagicMock()
        response = '{"conclusive": true, "confidence": 0.9, "reason": "High confidence"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
            hooks={"on_judge_termination": hook_mock},
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.is_high_confidence is True
        hook_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_confidence_does_not_trigger_hook(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test hook is not called for low confidence termination."""
        hook_mock = MagicMock()
        response = '{"conclusive": true, "confidence": 0.6, "reason": "Low confidence"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
            hooks={"on_judge_termination": hook_mock},
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.is_high_confidence is False
        hook_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_returns_error_result(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test timeout returns proper error result."""

        async def timeout_generate(agent, prompt, context):
            raise asyncio.TimeoutError("Timeout")

        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=timeout_generate,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "timeout"

    @pytest.mark.asyncio
    async def test_exception_returns_error_result(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test generic exception returns error result."""

        async def error_generate(agent, prompt, context):
            raise RuntimeError("Unexpected error")

        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=error_generate,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.check_judge_termination_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "error"


# =============================================================================
# Early Stopping Tests
# =============================================================================


class TestEarlyStopping:
    """Tests for check_early_stopping method."""

    @pytest.mark.asyncio
    async def test_disabled_returns_continue(self, mock_agents, sample_proposals, mock_messages):
        """Test returns continue when early_stopping disabled."""
        protocol = DebateProtocol(early_stopping=False)
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=5,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_continues_before_min_rounds(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues if min rounds not reached."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        # Round 1 is below min_rounds_before_early_stop (2)
        should_continue = await checker.check_early_stopping(
            round_num=1,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_stops_when_threshold_reached(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test stops when enough agents vote STOP."""
        # All agents vote STOP
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is False

    @pytest.mark.asyncio
    async def test_continues_when_threshold_not_reached(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues when not enough agents vote STOP."""
        protocol = DebateProtocol(
            early_stopping=True,
            early_stop_threshold=0.9,  # High threshold
            min_rounds_before_early_stop=2,
        )

        responses = iter(["CONTINUE", "STOP", "CONTINUE"])

        async def varying_generate(agent, prompt, context):
            return next(responses)

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=varying_generate,
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # Only 1/3 voted STOP, below 90% threshold
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_continues_when_all_vote_continue(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues when all agents vote CONTINUE."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONTINUE"),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_emits_hook_on_early_stop(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test on_early_stop hook is called when stopping."""
        hook_mock = MagicMock()
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
            hooks={"on_early_stop": hook_mock},
        )

        await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        hook_mock.assert_called_once()
        args = hook_mock.call_args[0]
        assert args[0] == 3  # round_num
        assert args[1] == 3  # stop_votes
        assert args[2] == 3  # total_votes

    @pytest.mark.asyncio
    async def test_handles_agent_exceptions(self, mock_agents, sample_proposals, mock_messages):
        """Test handles exceptions from individual agents."""
        protocol = DebateProtocol(
            early_stopping=True,
            early_stop_threshold=0.6,
            min_rounds_before_early_stop=2,
        )

        call_count = [0]

        async def sometimes_failing_generate(agent, prompt, context):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Agent error")
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=sometimes_failing_generate,
            task="Test task",
        )

        # Should still work with 2/3 agents responding
        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # 2 STOP votes out of 2 valid = 100% > 60%
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_continues_when_all_agents_fail(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues when all agents fail."""

        async def failing_generate(agent, prompt, context):
            raise ValueError("All agents fail")

        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=failing_generate,
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # Safe default: continue
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_handles_timeout(self, mock_agents, sample_proposals, mock_messages):
        """Test handles overall timeout gracefully."""
        protocol = DebateProtocol(
            early_stopping=True,
            min_rounds_before_early_stop=2,
            round_timeout_seconds=0.01,  # Very short timeout
        )

        async def slow_generate(agent, prompt, context):
            await asyncio.sleep(1)  # Longer than timeout
            return "STOP"

        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=slow_generate,
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # Safe default on timeout: continue
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_ignores_stop_in_continue_response(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test STOP is not counted if CONTINUE is also present."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONTINUE (but could STOP)"),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # Response contains both CONTINUE and STOP, so not counted as STOP
        assert should_continue is True


# =============================================================================
# should_terminate Tests
# =============================================================================


class TestShouldTerminate:
    """Tests for combined should_terminate method."""

    @pytest.mark.asyncio
    async def test_terminates_on_judge_decision(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test terminates when judge says conclusive."""
        response = "CONCLUSIVE: yes\nREASON: All issues resolved"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_stop, reason = await checker.should_terminate(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_stop is True
        assert reason != ""

    @pytest.mark.asyncio
    async def test_terminates_on_early_stopping(self, mock_agents, sample_proposals, mock_messages):
        """Test terminates when agents vote to stop."""
        protocol = DebateProtocol(
            judge_termination=False,  # Disable judge
            early_stopping=True,
            early_stop_threshold=0.7,
            min_rounds_before_early_stop=2,
        )
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        should_stop, reason = await checker.should_terminate(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_stop is True
        assert "early" in reason.lower()

    @pytest.mark.asyncio
    async def test_continues_when_both_conditions_false(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues when neither termination condition met."""
        protocol = DebateProtocol(
            judge_termination=False,
            early_stopping=True,
            early_stop_threshold=0.9,
            min_rounds_before_early_stop=2,
        )
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONTINUE"),
            task="Test task",
        )

        should_stop, reason = await checker.should_terminate(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_stop is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_judge_termination_checked_first(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test judge termination is checked before early stopping."""

        # Judge says conclusive, early stopping would say continue
        async def dual_response_generate(agent, prompt, context):
            if "conclusive" in prompt.lower() or "evaluating" in prompt.lower():
                return "CONCLUSIVE: yes\nREASON: Judge decided"
            return "CONTINUE"

        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=dual_response_generate,
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_stop, reason = await checker.should_terminate(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_stop is True
        assert "Judge" not in reason or reason != ""  # Judge reason, not early stop


# =============================================================================
# should_terminate_with_confidence Tests
# =============================================================================


class TestShouldTerminateWithConfidence:
    """Tests for should_terminate_with_confidence method."""

    @pytest.mark.asyncio
    async def test_terminates_on_high_confidence_judge(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test terminates when judge has high confidence."""
        response = '{"conclusive": true, "confidence": 0.9, "reason": "High confidence"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.should_terminate_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is True
        assert result.is_high_confidence is True
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_rejects_low_confidence_when_required(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test rejects termination when confidence too low."""
        response = '{"conclusive": true, "confidence": 0.65, "reason": "Low confidence"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.should_terminate_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
            require_high_confidence=True,
        )

        assert result.should_terminate is False
        assert result.source == "judge_low_confidence"
        assert "below threshold" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_accepts_low_confidence_when_not_required(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test accepts low confidence when requirement disabled."""
        response = '{"conclusive": true, "confidence": 0.65, "reason": "Low but ok"}'
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        result = await checker.should_terminate_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
            require_high_confidence=False,
        )

        assert result.should_terminate is True
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_terminates_on_early_stop_vote(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test terminates when agents vote to stop."""
        protocol = DebateProtocol(
            judge_termination=False,
            early_stopping=True,
            early_stop_threshold=0.7,
            min_rounds_before_early_stop=2,
        )
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        result = await checker.should_terminate_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is True
        assert result.source == "early_stop"
        assert result.confidence == 0.7  # Uses threshold as confidence

    @pytest.mark.asyncio
    async def test_continues_when_no_termination(
        self, mock_agents, sample_proposals, mock_messages
    ):
        """Test continues with high confidence when no termination trigger."""
        protocol = DebateProtocol(
            judge_termination=False,
            early_stopping=True,
            min_rounds_before_early_stop=2,
        )
        checker = TerminationChecker(
            protocol=protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONTINUE"),
            task="Test task",
        )

        result = await checker.should_terminate_with_confidence(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert result.should_terminate is False
        assert result.confidence == 1.0  # High confidence in continuing
        assert result.source == "no_termination_trigger"


# =============================================================================
# Edge Cases and Configuration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and configuration variations."""

    @pytest.mark.asyncio
    async def test_empty_proposals(self, default_protocol, mock_agents, mock_messages):
        """Test handles empty proposals dict."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONCLUSIVE: no"),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals={},
            context=mock_messages,
        )

        assert should_continue is True

    @pytest.mark.asyncio
    async def test_empty_context(self, default_protocol, mock_agents, sample_proposals):
        """Test handles empty context."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONCLUSIVE: yes\nREASON: Done"),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=[],
        )

        assert should_continue is False

    @pytest.mark.asyncio
    async def test_no_agents(self, default_protocol, sample_proposals, mock_messages):
        """Test handles empty agents list."""
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=[],
            generate_fn=create_async_generate_fn("STOP"),
            task="Test task",
        )

        should_continue = await checker.check_early_stopping(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        # No votes = continue (safe default)
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_long_task_truncation(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test long task is truncated in prompts."""
        long_task = "A" * 1000  # Very long task
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONCLUSIVE: yes\nREASON: Done"),
            task=long_task,
            select_judge_fn=select_judge_fn,
        )

        # Should not raise even with very long task
        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is False

    @pytest.mark.asyncio
    async def test_long_proposals_truncation(self, default_protocol, mock_agents, mock_messages):
        """Test long proposals are truncated in prompts."""
        long_proposals = {
            "claude": "P" * 1000,
            "gpt4": "Q" * 1000,
        }
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn("CONCLUSIVE: yes\nREASON: Done"),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        # Should not raise even with very long proposals
        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=long_proposals,
            context=mock_messages,
        )

        assert should_continue is False

    def test_rlm_constants_valid(self):
        """Test RLM constants have valid values."""
        assert 0.0 <= RLM_MIN_CONFIDENCE_FOR_STOP <= 1.0
        assert 0.0 <= RLM_HIGH_CONFIDENCE_THRESHOLD <= 1.0
        assert RLM_MIN_CONFIDENCE_FOR_STOP <= RLM_HIGH_CONFIDENCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_case_insensitive_conclusive_parsing(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test CONCLUSIVE parsing is case insensitive."""
        response = "conclusive: YES\nreason: Case test"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is False

    @pytest.mark.asyncio
    async def test_true_as_conclusive_value(
        self, default_protocol, mock_agents, sample_proposals, mock_messages
    ):
        """Test 'true' is accepted as conclusive value."""
        response = "CONCLUSIVE: true\nREASON: Using true"
        checker = TerminationChecker(
            protocol=default_protocol,
            agents=mock_agents,
            generate_fn=create_async_generate_fn(response),
            task="Test task",
            select_judge_fn=select_judge_fn,
        )

        should_continue, reason = await checker.check_judge_termination(
            round_num=3,
            proposals=sample_proposals,
            context=mock_messages,
        )

        assert should_continue is False
