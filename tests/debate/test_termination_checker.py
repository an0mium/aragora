"""Tests for debate termination checker module.

Covers TerminationResult dataclass, TerminationChecker with judge termination,
early stopping, combined termination, and RLM-style confidence scoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.termination_checker import (
    RLM_HIGH_CONFIDENCE_THRESHOLD,
    RLM_MIN_CONFIDENCE_FOR_STOP,
    TerminationChecker,
    TerminationResult,
)


# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------

@dataclass
class _FakeMessage:
    role: str = "proposer"
    agent: str = "test-agent"
    content: str = "test message"
    round: int = 0


@dataclass
class _FakeAgent:
    name: str = "agent-1"
    model: str = "test-model"


def _make_protocol(**overrides: Any) -> MagicMock:
    """Create a mock DebateProtocol with sensible defaults."""
    defaults = {
        "judge_termination": False,
        "min_rounds_before_judge_check": 2,
        "early_stopping": False,
        "early_stop_threshold": 0.75,
        "min_rounds_before_early_stop": 2,
        "round_timeout_seconds": 30,
    }
    defaults.update(overrides)
    proto = MagicMock()
    for key, value in defaults.items():
        setattr(proto, key, value)
    return proto


def _make_checker(
    protocol: Any | None = None,
    agents: list[Any] | None = None,
    generate_fn: Any | None = None,
    task: str = "Design a rate limiter",
    select_judge_fn: Any | None = None,
    hooks: dict[str, Any] | None = None,
) -> TerminationChecker:
    """Create a TerminationChecker with default test values."""
    return TerminationChecker(
        protocol=protocol or _make_protocol(),
        agents=agents or [_FakeAgent("alice"), _FakeAgent("bob")],
        generate_fn=generate_fn or AsyncMock(return_value="CONTINUE"),
        task=task,
        select_judge_fn=select_judge_fn,
        hooks=hooks,
    )


def _context(n: int = 6) -> list[_FakeMessage]:
    """Return a list of n fake messages for context."""
    return [_FakeMessage(content=f"msg-{i}") for i in range(n)]


PROPOSALS = {"alice": "Use token bucket algorithm", "bob": "Use sliding window"}


# ===========================================================================
# TerminationResult dataclass
# ===========================================================================


class TestTerminationResult:
    """Tests for the TerminationResult dataclass."""

    def test_default_values(self):
        result = TerminationResult(should_terminate=False)
        assert result.should_terminate is False
        assert result.reason == ""
        assert result.confidence == 0.5
        assert result.source == "unknown"
        assert result.votes is None

    def test_custom_values(self):
        result = TerminationResult(
            should_terminate=True,
            reason="Debate is conclusive",
            confidence=0.95,
            source="judge",
            votes={"alice": True, "bob": False},
        )
        assert result.should_terminate is True
        assert result.reason == "Debate is conclusive"
        assert result.confidence == 0.95
        assert result.source == "judge"
        assert result.votes == {"alice": True, "bob": False}

    # -- is_high_confidence property --

    def test_is_high_confidence_at_threshold(self):
        result = TerminationResult(
            should_terminate=True, confidence=RLM_HIGH_CONFIDENCE_THRESHOLD
        )
        assert result.is_high_confidence is True

    def test_is_high_confidence_above_threshold(self):
        result = TerminationResult(should_terminate=True, confidence=0.95)
        assert result.is_high_confidence is True

    def test_is_high_confidence_below_threshold(self):
        result = TerminationResult(should_terminate=True, confidence=0.79)
        assert result.is_high_confidence is False

    def test_is_high_confidence_zero(self):
        result = TerminationResult(should_terminate=True, confidence=0.0)
        assert result.is_high_confidence is False

    # -- should_consider_stopping property --

    def test_should_consider_stopping_true(self):
        result = TerminationResult(
            should_terminate=True, confidence=RLM_MIN_CONFIDENCE_FOR_STOP
        )
        assert result.should_consider_stopping is True

    def test_should_consider_stopping_above_min(self):
        result = TerminationResult(should_terminate=True, confidence=0.9)
        assert result.should_consider_stopping is True

    def test_should_consider_stopping_below_min(self):
        result = TerminationResult(should_terminate=True, confidence=0.4)
        assert result.should_consider_stopping is False

    def test_should_consider_stopping_false_when_not_terminate(self):
        """If should_terminate is False, should_consider_stopping is False."""
        result = TerminationResult(should_terminate=False, confidence=1.0)
        assert result.should_consider_stopping is False

    def test_constants_are_sane(self):
        assert 0.0 < RLM_MIN_CONFIDENCE_FOR_STOP < RLM_HIGH_CONFIDENCE_THRESHOLD <= 1.0


# ===========================================================================
# TerminationChecker.__init__
# ===========================================================================


class TestTerminationCheckerInit:
    """Tests for TerminationChecker initialization."""

    def test_stores_attributes(self):
        proto = _make_protocol()
        agents = [_FakeAgent("a")]
        gen_fn = AsyncMock()
        judge_fn = AsyncMock()
        hooks = {"on_early_stop": MagicMock()}

        checker = TerminationChecker(
            protocol=proto,
            agents=agents,
            generate_fn=gen_fn,
            task="Analyze topic",
            select_judge_fn=judge_fn,
            hooks=hooks,
        )
        assert checker.protocol is proto
        assert checker.agents is agents
        assert checker.generate_fn is gen_fn
        assert checker.task == "Analyze topic"
        assert checker.select_judge_fn is judge_fn
        assert checker.hooks is hooks

    def test_default_hooks_empty_dict(self):
        checker = _make_checker()
        assert checker.hooks == {}

    def test_select_judge_fn_defaults_to_none(self):
        checker = _make_checker()
        assert checker.select_judge_fn is None


# ===========================================================================
# check_judge_termination (traditional, returns tuple)
# ===========================================================================


class TestCheckJudgeTermination:
    """Tests for the traditional judge termination check."""

    @pytest.mark.asyncio
    async def test_disabled_returns_continue(self):
        checker = _make_checker(protocol=_make_protocol(judge_termination=False))
        should_continue, reason = await checker.check_judge_termination(
            5, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_before_min_rounds_returns_continue(self):
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True, min_rounds_before_judge_check=3
            )
        )
        should_continue, reason = await checker.check_judge_termination(
            2, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_no_judge_selector_returns_continue(self):
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=1),
            select_judge_fn=None,
        )
        should_continue, reason = await checker.check_judge_termination(
            3, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_conclusive_yes_terminates(self):
        judge = _FakeAgent("judge-1")
        select_judge = AsyncMock(return_value=judge)
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: All issues resolved")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=1),
            generate_fn=gen_fn,
            select_judge_fn=select_judge,
        )
        should_continue, reason = await checker.check_judge_termination(
            3, PROPOSALS, _context()
        )
        assert should_continue is False
        assert "All issues resolved" in reason

    @pytest.mark.asyncio
    async def test_conclusive_no_continues(self):
        judge = _FakeAgent("judge-1")
        select_judge = AsyncMock(return_value=judge)
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: Needs more discussion")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=1),
            generate_fn=gen_fn,
            select_judge_fn=select_judge,
        )
        should_continue, reason = await checker.check_judge_termination(
            3, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_conclusive_true_terminates(self):
        judge = _FakeAgent("judge-1")
        gen_fn = AsyncMock(return_value="CONCLUSIVE: true\nREASON: Converged")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_continue, _ = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_conclusive_1_terminates(self):
        judge = _FakeAgent("judge-1")
        gen_fn = AsyncMock(return_value="CONCLUSIVE: 1\nREASON: Done")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_continue, _ = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_hook_fired_on_termination(self):
        judge = _FakeAgent("judge-1")
        hook = MagicMock()
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: Done")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
            hooks={"on_judge_termination": hook},
        )
        await checker.check_judge_termination(1, PROPOSALS, _context())
        hook.assert_called_once_with("judge-1", "Done")

    @pytest.mark.asyncio
    async def test_hook_not_fired_on_continue(self):
        judge = _FakeAgent("judge-1")
        hook = MagicMock()
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: Need more")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
            hooks={"on_judge_termination": hook},
        )
        await checker.check_judge_termination(1, PROPOSALS, _context())
        hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_error_continues(self):
        gen_fn = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_cancelled_error_continues(self):
        gen_fn = AsyncMock(side_effect=asyncio.CancelledError())
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_value_error_continues(self):
        gen_fn = AsyncMock(side_effect=ValueError("bad parse"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_unexpected_error_continues(self):
        gen_fn = AsyncMock(side_effect=RuntimeError("boom"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_context_truncated_to_last_five(self):
        """generate_fn should receive at most the last 5 context messages."""
        judge = _FakeAgent("judge-1")
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: nope")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        ctx = _context(10)
        await checker.check_judge_termination(1, PROPOSALS, ctx)
        _, _, passed_ctx = gen_fn.call_args[0]
        assert len(passed_ctx) == 5

    @pytest.mark.asyncio
    async def test_at_exact_min_round_triggers_check(self):
        """When round_num == min_rounds_before_judge_check, check runs."""
        judge = _FakeAgent("j")
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: done")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=3),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_continue, _ = await checker.check_judge_termination(
            3, PROPOSALS, _context()
        )
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_malformed_response_continues(self):
        """If the response has no parseable CONCLUSIVE line, continue."""
        gen_fn = AsyncMock(return_value="I think the debate should keep going.")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is True
        assert reason == ""


# ===========================================================================
# check_judge_termination_with_confidence (RLM-enhanced)
# ===========================================================================


class TestCheckJudgeTerminationWithConfidence:
    """Tests for the RLM-enhanced judge termination check."""

    @pytest.mark.asyncio
    async def test_disabled_returns_not_terminate(self):
        checker = _make_checker(protocol=_make_protocol(judge_termination=False))
        result = await checker.check_judge_termination_with_confidence(
            5, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 1.0
        assert result.source == "config"

    @pytest.mark.asyncio
    async def test_before_min_rounds(self):
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True, min_rounds_before_judge_check=5
            )
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 1.0
        assert result.source == "config"
        assert "Minimum rounds" in result.reason

    @pytest.mark.asyncio
    async def test_no_judge_selector(self):
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            select_judge_fn=None,
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.5
        assert result.source == "error"

    @pytest.mark.asyncio
    async def test_json_conclusive_high_confidence(self):
        judge = _FakeAgent("judge-1")
        json_resp = '{"conclusive": true, "confidence": 0.95, "reason": "Converged well"}'
        gen_fn = AsyncMock(return_value=json_resp)
        hook = MagicMock()
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
            hooks={"on_judge_termination": hook},
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is True
        assert result.confidence == 0.95
        assert result.reason == "Converged well"
        assert result.source == "judge"
        hook.assert_called_once_with("judge-1", "Converged well")

    @pytest.mark.asyncio
    async def test_json_conclusive_low_confidence_no_hook(self):
        """Low-confidence conclusive result should NOT fire hook."""
        judge = _FakeAgent("judge-1")
        json_resp = '{"conclusive": true, "confidence": 0.55, "reason": "Maybe done"}'
        gen_fn = AsyncMock(return_value=json_resp)
        hook = MagicMock()
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
            hooks={"on_judge_termination": hook},
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is True
        assert result.confidence == 0.55
        hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_not_conclusive(self):
        json_resp = '{"conclusive": false, "confidence": 0.3, "reason": "Need more rounds"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.3
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_json_wrapped_in_markdown(self):
        resp = '```json\n{"conclusive": true, "confidence": 0.88, "reason": "Done"}\n```'
        gen_fn = AsyncMock(return_value=resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is True
        assert result.confidence == 0.88

    @pytest.mark.asyncio
    async def test_confidence_clamped_above_one(self):
        json_resp = '{"conclusive": true, "confidence": 1.5, "reason": "Over"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_below_zero(self):
        json_resp = '{"conclusive": false, "confidence": -0.5, "reason": "Under"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_fallback_text_parsing_conclusive(self):
        """When JSON fails, fallback text parsing detects conclusive."""
        resp = "The debate is conclusive: yes\nReason: All settled"
        gen_fn = AsyncMock(return_value=resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is True
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_fallback_text_parsing_not_conclusive(self):
        resp = "conclusive: no\nreason: Keep going"
        gen_fn = AsyncMock(return_value=resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False

    @pytest.mark.asyncio
    async def test_timeout_error_returns_result(self):
        gen_fn = AsyncMock(side_effect=asyncio.TimeoutError("timed out"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "timeout"

    @pytest.mark.asyncio
    async def test_cancelled_error_returns_result(self):
        gen_fn = AsyncMock(side_effect=asyncio.CancelledError())
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "timeout"

    @pytest.mark.asyncio
    async def test_parse_error_returns_result(self):
        gen_fn = AsyncMock(side_effect=TypeError("type error"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "parse_error"

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_result(self):
        gen_fn = AsyncMock(side_effect=RuntimeError("boom"))
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.0
        assert result.source == "error"

    @pytest.mark.asyncio
    async def test_json_missing_fields_uses_defaults(self):
        """Missing JSON fields should use defaults (conclusive=False, confidence=0.5)."""
        json_resp = '{"reason": "partial response"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.check_judge_termination_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.confidence == 0.5
        assert result.reason == "partial response"


# ===========================================================================
# check_early_stopping
# ===========================================================================


class TestCheckEarlyStopping:
    """Tests for the agent-vote early stopping check."""

    @pytest.mark.asyncio
    async def test_disabled_returns_continue(self):
        checker = _make_checker(protocol=_make_protocol(early_stopping=False))
        should_continue = await checker.check_early_stopping(5, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_before_min_rounds_returns_continue(self):
        checker = _make_checker(
            protocol=_make_protocol(early_stopping=True, min_rounds_before_early_stop=4)
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_at_min_round_proceeds(self):
        gen_fn = AsyncMock(return_value="STOP")
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=3,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_all_stop_terminates(self):
        gen_fn = AsyncMock(return_value="STOP")
        agents = [_FakeAgent("a"), _FakeAgent("b"), _FakeAgent("c")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.75,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False  # 3/3 = 1.0 >= 0.75

    @pytest.mark.asyncio
    async def test_all_continue_keeps_going(self):
        gen_fn = AsyncMock(return_value="CONTINUE")
        agents = [_FakeAgent("a"), _FakeAgent("b"), _FakeAgent("c")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.75,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_mixed_votes_below_threshold(self):
        """2 of 4 STOP = 0.5, below threshold 0.75, should continue."""
        responses = iter(["STOP", "CONTINUE", "STOP", "CONTINUE"])
        gen_fn = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        agents = [_FakeAgent(f"a{i}") for i in range(4)]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.75,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_mixed_votes_at_threshold(self):
        """3 of 4 STOP = 0.75, meets threshold 0.75, should stop."""
        responses = iter(["STOP", "STOP", "STOP", "CONTINUE"])
        gen_fn = AsyncMock(side_effect=lambda *a, **kw: next(responses))
        agents = [_FakeAgent(f"a{i}") for i in range(4)]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.75,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_stop_with_extra_text(self):
        """Response containing 'STOP' without 'CONTINUE' should count as stop."""
        gen_fn = AsyncMock(return_value="I think we should STOP the debate now.")
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_response_with_both_stop_and_continue(self):
        """If response contains both STOP and CONTINUE, it should NOT count as stop."""
        gen_fn = AsyncMock(
            return_value="While some say STOP, I prefer CONTINUE."
        )
        agents = [_FakeAgent("a")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        # 0 stop votes out of 1 = 0.0, below 0.5 threshold
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_all_agents_error_continues(self):
        """When all agents raise errors, total_votes == 0 => continue."""
        gen_fn = AsyncMock(side_effect=RuntimeError("all fail"))
        agents = [_FakeAgent("a"), _FakeAgent("b")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_some_agents_error_still_counts_rest(self):
        """One agent errors, two vote STOP => 2/2 = 1.0 >= 0.75 => stop."""
        call_count = 0

        async def mixed_gen(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return "STOP"

        agents = [_FakeAgent("a"), _FakeAgent("b"), _FakeAgent("c")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.75,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=mixed_gen,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_hook_fired_on_early_stop(self):
        gen_fn = AsyncMock(return_value="STOP")
        hook = MagicMock()
        agents = [_FakeAgent("a"), _FakeAgent("b")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
            hooks={"on_early_stop": hook},
        )
        await checker.check_early_stopping(5, PROPOSALS, _context())
        hook.assert_called_once_with(5, 2, 2)

    @pytest.mark.asyncio
    async def test_hook_not_fired_on_continue(self):
        gen_fn = AsyncMock(return_value="CONTINUE")
        hook = MagicMock()
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            hooks={"on_early_stop": hook},
        )
        await checker.check_early_stopping(5, PROPOSALS, _context())
        hook.assert_not_called()

    @pytest.mark.asyncio
    async def test_gather_timeout_continues(self):
        """If asyncio.wait_for times out, debate should continue."""

        async def slow_gen(*a, **kw):
            await asyncio.sleep(100)
            return "STOP"

        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=0.01,  # Very short timeout
            ),
            generate_fn=slow_gen,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True


# ===========================================================================
# should_terminate (combined check)
# ===========================================================================


class TestShouldTerminate:
    """Tests for the combined should_terminate convenience method."""

    @pytest.mark.asyncio
    async def test_both_disabled_returns_false(self):
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=False, early_stopping=False)
        )
        should_stop, reason = await checker.should_terminate(5, PROPOSALS, _context())
        assert should_stop is False
        assert reason == ""

    @pytest.mark.asyncio
    async def test_judge_terminates_first(self):
        """Judge termination is checked first and should short-circuit."""
        judge = _FakeAgent("judge-1")
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: Judge says done")
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_stop, reason = await checker.should_terminate(3, PROPOSALS, _context())
        assert should_stop is True
        assert "Judge says done" in reason

    @pytest.mark.asyncio
    async def test_early_stop_triggers_when_judge_continues(self):
        """If judge says continue, early stopping should be checked."""
        judge = _FakeAgent("judge-1")

        async def gen_fn(agent, prompt, ctx):
            if "evaluating" in prompt.lower():
                return "CONCLUSIVE: no\nREASON: Not yet"
            return "STOP"

        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_stop, reason = await checker.should_terminate(3, PROPOSALS, _context())
        assert should_stop is True
        assert "Agents voted to stop early" in reason

    @pytest.mark.asyncio
    async def test_continues_when_both_say_continue(self):
        judge = _FakeAgent("judge-1")

        async def gen_fn(agent, prompt, ctx):
            if "evaluating" in prompt.lower():
                return "CONCLUSIVE: no\nREASON: Not yet"
            return "CONTINUE"

        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=judge),
        )
        should_stop, reason = await checker.should_terminate(3, PROPOSALS, _context())
        assert should_stop is False
        assert reason == ""


# ===========================================================================
# should_terminate_with_confidence (RLM combined check)
# ===========================================================================


class TestShouldTerminateWithConfidence:
    """Tests for the RLM-enhanced combined should_terminate_with_confidence."""

    @pytest.mark.asyncio
    async def test_high_confidence_terminates(self):
        json_resp = '{"conclusive": true, "confidence": 0.95, "reason": "Clear"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=False,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is True
        assert result.is_high_confidence is True
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_low_confidence_rejected_with_require_high(self):
        """When require_high_confidence=True, low-confidence terminates are rejected."""
        json_resp = '{"conclusive": true, "confidence": 0.55, "reason": "Maybe done"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=False,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context(), require_high_confidence=True
        )
        assert result.should_terminate is False
        assert result.source == "judge_low_confidence"
        assert "0.55" in result.reason

    @pytest.mark.asyncio
    async def test_low_confidence_accepted_without_require_high(self):
        """When require_high_confidence=False, low-confidence terminates are accepted."""
        json_resp = '{"conclusive": true, "confidence": 0.55, "reason": "Maybe done"}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=False,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context(), require_high_confidence=False
        )
        assert result.should_terminate is True
        assert result.source == "judge"

    @pytest.mark.asyncio
    async def test_early_stop_provides_confidence_from_threshold(self):
        """When early stopping triggers, confidence should be the threshold value."""

        async def gen_fn(agent, prompt, ctx):
            if "evaluating" in prompt.lower():
                return '{"conclusive": false, "confidence": 0.3, "reason": "Not done"}'
            return "STOP"

        threshold = 0.8
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=threshold,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context(), require_high_confidence=False
        )
        assert result.should_terminate is True
        assert result.source == "early_stop"
        assert result.confidence == threshold

    @pytest.mark.asyncio
    async def test_no_trigger_continues(self):
        """When neither judge nor early stopping triggers, continue."""

        async def gen_fn(agent, prompt, ctx):
            if "evaluating" in prompt.lower():
                return '{"conclusive": false, "confidence": 0.3, "reason": "Not done"}'
            return "CONTINUE"

        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context()
        )
        assert result.should_terminate is False
        assert result.source == "no_termination_trigger"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_both_disabled_continues(self):
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=False, early_stopping=False)
        )
        result = await checker.should_terminate_with_confidence(
            5, PROPOSALS, _context()
        )
        assert result.should_terminate is False

    @pytest.mark.asyncio
    async def test_confidence_at_exact_threshold_passes(self):
        """Confidence exactly at RLM_HIGH_CONFIDENCE_THRESHOLD should pass."""
        json_resp = f'{{"conclusive": true, "confidence": {RLM_HIGH_CONFIDENCE_THRESHOLD}, "reason": "Exact threshold"}}'
        gen_fn = AsyncMock(return_value=json_resp)
        checker = _make_checker(
            protocol=_make_protocol(
                judge_termination=True,
                min_rounds_before_judge_check=0,
                early_stopping=False,
            ),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        result = await checker.should_terminate_with_confidence(
            3, PROPOSALS, _context(), require_high_confidence=True
        )
        assert result.should_terminate is True
        assert result.is_high_confidence is True


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Additional edge-case tests."""

    @pytest.mark.asyncio
    async def test_empty_proposals(self):
        """Empty proposals dict should still work."""
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: nothing")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, _ = await checker.check_judge_termination(1, {}, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Empty context list should not crash."""
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: done")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(1, PROPOSALS, [])
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_single_agent_early_stopping(self):
        """Early stopping with a single agent voting STOP."""
        gen_fn = AsyncMock(return_value="STOP")
        agents = [_FakeAgent("solo")]
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=1.0,
                round_timeout_seconds=10,
            ),
            agents=agents,
            generate_fn=gen_fn,
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_no_agents_early_stopping(self):
        """No agents means total_votes == 0, should continue."""
        checker = _make_checker(
            protocol=_make_protocol(
                early_stopping=True,
                min_rounds_before_early_stop=0,
                early_stop_threshold=0.5,
                round_timeout_seconds=10,
            ),
            agents=[],
        )
        should_continue = await checker.check_early_stopping(3, PROPOSALS, _context())
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_round_zero(self):
        """Round 0 with min_rounds_before_judge_check=0 should still trigger check."""
        gen_fn = AsyncMock(return_value="CONCLUSIVE: yes\nREASON: immediate")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, _ = await checker.check_judge_termination(0, PROPOSALS, _context())
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_very_long_task_truncated_in_prompt(self):
        """Task text longer than 300 chars should be truncated in the prompt."""
        long_task = "x" * 1000
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: nope")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
            task=long_task,
        )
        await checker.check_judge_termination(1, PROPOSALS, _context())
        prompt_arg = gen_fn.call_args[0][1]
        # The task portion should be truncated to 300 chars (via self.task[:300])
        assert "x" * 300 in prompt_arg
        assert "x" * 301 not in prompt_arg

    @pytest.mark.asyncio
    async def test_very_long_proposal_truncated_in_prompt(self):
        """Proposals longer than 200 chars should be truncated in the prompt."""
        long_proposals = {"alice": "y" * 500, "bob": "z" * 500}
        gen_fn = AsyncMock(return_value="CONCLUSIVE: no\nREASON: nope")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        await checker.check_judge_termination(1, long_proposals, _context())
        prompt_arg = gen_fn.call_args[0][1]
        # Each proposal is truncated to 200 chars (via prop[:200])
        assert "y" * 200 in prompt_arg
        assert "y" * 201 not in prompt_arg

    @pytest.mark.asyncio
    async def test_case_insensitive_conclusive_parsing(self):
        """CONCLUSIVE line should be parsed case-insensitively."""
        gen_fn = AsyncMock(return_value="conclusive: Yes\nreason: Case test")
        checker = _make_checker(
            protocol=_make_protocol(judge_termination=True, min_rounds_before_judge_check=0),
            generate_fn=gen_fn,
            select_judge_fn=AsyncMock(return_value=_FakeAgent("j")),
        )
        should_continue, reason = await checker.check_judge_termination(
            1, PROPOSALS, _context()
        )
        assert should_continue is False
        assert "Case test" in reason

    @pytest.mark.asyncio
    async def test_module_exports(self):
        """Ensure the module exports the expected names."""
        from aragora.debate import termination_checker

        assert "TerminationChecker" in termination_checker.__all__
        assert "TerminationResult" in termination_checker.__all__
