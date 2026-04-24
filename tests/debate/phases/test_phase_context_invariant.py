"""Tests for the ``require_phase_result`` invariant helper.

These tests cover the narrow contract introduced to replace the
``result = ctx.result`` / ``ctx.result.foo`` pattern used throughout the
consensus and winner-selection phases. The helper:

1. Raises ``RuntimeError`` when ``ctx.result`` is ``None`` (phase invariant
   violated — result must be initialized before consensus/winner phases run).
2. Returns the initialized ``DebateResult`` narrowed to the non-``None``
   type when it has been set, so downstream phase code can rely on it
   without redundant ``None`` checks.
3. Is idempotent: repeated calls return the same underlying result.
4. Leaves functional behaviour of the consensus/winner phases unchanged
   compared to the previous ``result = ctx.result`` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.core import DebateResult, Environment
from aragora.debate.debate_state import DebateContext
from aragora.debate.phases._phase_invariant import require_phase_result
from aragora.debate.phases.winner_selector import WinnerSelector


# ---------------------------------------------------------------------------
# require_result property contract
# ---------------------------------------------------------------------------


class TestRequirePhaseResultInvariant:
    """Test the ``require_phase_result`` helper contract."""

    def test_raises_runtime_error_when_result_is_none(self) -> None:
        """The helper must raise when result has not been initialized."""
        ctx = DebateContext()
        assert ctx.result is None

        with pytest.raises(RuntimeError, match="phase invariant violated"):
            require_phase_result(ctx)

    def test_returns_result_when_initialized(self) -> None:
        """The helper returns the underlying DebateResult when set."""
        env = Environment(task="invariant-test")
        result = DebateResult(task=env.task)
        ctx = DebateContext(env=env, result=result)

        narrowed = require_phase_result(ctx)

        assert narrowed is result
        assert isinstance(narrowed, DebateResult)

    def test_idempotent_returns_same_object(self) -> None:
        """Calling the helper multiple times returns the same object."""
        env = Environment(task="idempotent-test")
        result = DebateResult(task=env.task)
        ctx = DebateContext(env=env, result=result)

        first = require_phase_result(ctx)
        second = require_phase_result(ctx)

        assert first is second is result

    def test_mutations_via_helper_alias_are_reflected_on_ctx_result(self) -> None:
        """Writes via the narrowed alias propagate to ctx.result (same object)."""
        env = Environment(task="mutation-test")
        result = DebateResult(task=env.task)
        ctx = DebateContext(env=env, result=result)

        alias = require_phase_result(ctx)
        alias.final_answer = "narrowed-write"
        alias.consensus_reached = True

        assert ctx.result is not None
        assert ctx.result.final_answer == "narrowed-write"
        assert ctx.result.consensus_reached is True

    def test_error_message_is_actionable(self) -> None:
        """The RuntimeError message must identify the invariant violation."""
        ctx = DebateContext()

        with pytest.raises(RuntimeError) as excinfo:
            require_phase_result(ctx)

        message = str(excinfo.value)
        assert "DebateContext.result" in message
        assert "phase invariant violated" in message

    def test_duck_typed_ctx_is_accepted(self) -> None:
        """The helper only reads ctx.result, so duck-typed fakes work.

        The consensus/winner phase tests use lightweight fakes that expose
        a ``result`` attribute. The helper must not require the full
        ``DebateContext`` API so those fakes keep working unchanged.
        """

        class _DuckCtx:
            def __init__(self, result: Any) -> None:
                self.result = result

        env = Environment(task="duck-type-test")
        result = DebateResult(task=env.task)

        narrowed = require_phase_result(_DuckCtx(result))  # type: ignore[arg-type]
        assert narrowed is result

        with pytest.raises(RuntimeError, match="phase invariant violated"):
            require_phase_result(_DuckCtx(None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Behavioural parity: WinnerSelector still works after the refactor
# ---------------------------------------------------------------------------


@dataclass
class _FakeVote:
    agent: str
    choice: str
    confidence: float = 0.8


@dataclass
class _FakeResult:
    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    winner: str = ""
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    status: str = ""
    votes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    id: str = "debate-invariant"


@dataclass
class _FakeAgent:
    name: str


@dataclass
class _FakeEnv:
    task: str = "invariant parity task"


class _FakeCtx:
    """Mimics the subset of DebateContext that WinnerSelector reads.

    The refactor uses ``require_phase_result(ctx)`` which only reads
    ``ctx.result``. Fakes therefore do **not** need to expose any new
    API, which is the whole point of the helper-function approach.
    """

    def __init__(self, *, result: Any, proposals: dict[str, str]) -> None:
        self.result = result
        self.proposals = proposals
        self.agents = [_FakeAgent(name="alice"), _FakeAgent(name="bob")]
        self.env = _FakeEnv()
        self.winner_agent: str | None = None
        self.vote_tally: dict[str, float] = {}


def _normalize_choice(choice: str, agents: list[_FakeAgent], proposals: dict[str, str]) -> str:
    return choice


class TestPhaseParityAfterRefactor:
    """WinnerSelector must produce the same output after the refactor."""

    def test_determine_majority_winner_produces_expected_result(self) -> None:
        """Majority winner path sets winner, confidence, and dissenting views."""
        selector = WinnerSelector(protocol=MagicMock(consensus_threshold=0.5))
        result = _FakeResult()
        ctx = _FakeCtx(result=result, proposals={"alice": "A", "bob": "B"})
        vote_counts = {"alice": 3.0, "bob": 1.0}

        selector.determine_majority_winner(
            ctx,  # type: ignore[arg-type]
            vote_counts=vote_counts,
            total_votes=4.0,
            choice_mapping={"alice": "alice", "bob": "bob"},
            normalize_choice=_normalize_choice,
        )

        assert result.winner == "alice"
        assert ctx.winner_agent == "alice"
        assert result.consensus_reached is True
        assert result.confidence == pytest.approx(0.75)
        assert result.final_answer == "A"
        # Only bob (non-winner) is a dissenting view.
        assert result.dissenting_views == ["[bob]: B"]

    def test_set_unanimous_winner_sets_all_fields(self) -> None:
        """Unanimous path sets consensus_strength='unanimous' and full confidence."""
        selector = WinnerSelector()
        result = _FakeResult()
        ctx = _FakeCtx(result=result, proposals={"alice": "A", "bob": "B"})

        selector.set_unanimous_winner(
            ctx,  # type: ignore[arg-type]
            winner="alice",
            unanimity_ratio=1.0,
            total_voters=2,
            count=2,
        )

        assert result.winner == "alice"
        assert ctx.winner_agent == "alice"
        assert result.consensus_reached is True
        assert result.confidence == 1.0
        assert result.consensus_strength == "unanimous"
        assert result.final_answer == "A"

    def test_set_no_unanimity_records_all_proposals(self) -> None:
        """When unanimity is not reached all proposals appear as dissent."""
        selector = WinnerSelector()
        result = _FakeResult(
            votes=[_FakeVote(agent="alice", choice="alice"), _FakeVote(agent="bob", choice="bob")]
        )
        ctx = _FakeCtx(result=result, proposals={"alice": "A", "bob": "B"})

        selector.set_no_unanimity(
            ctx,  # type: ignore[arg-type]
            winner="alice",
            unanimity_ratio=0.5,
            total_voters=2,
            count=1,
            choice_mapping={"alice": "alice", "bob": "bob"},
        )

        assert result.consensus_reached is False
        assert result.confidence == 0.5
        assert result.consensus_strength == "none"
        # The summary text embeds proposals with per-choice vote counts.
        assert "[alice]" in result.final_answer
        assert "[bob]" in result.final_answer
        assert "A" in result.final_answer
        assert "B" in result.final_answer
        # Both proposals are recorded as dissenting (same format as pre-refactor).
        assert result.dissenting_views == ["[alice]: A", "[bob]: B"]

    def test_refactored_winner_selector_raises_when_result_none(self) -> None:
        """If the phase invariant is violated, the refactored code must raise."""
        selector = WinnerSelector(protocol=MagicMock(consensus_threshold=0.5))

        ctx = _FakeCtx(result=None, proposals={"alice": "A"})

        with pytest.raises(RuntimeError, match="phase invariant violated"):
            selector.determine_majority_winner(
                ctx,  # type: ignore[arg-type]
                vote_counts={"alice": 1.0},
                total_votes=1.0,
                choice_mapping={"alice": "alice"},
                normalize_choice=_normalize_choice,
            )
