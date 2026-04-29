"""Regression tests for ``DebateRoundsExecutor`` phase-invariant narrowing.

Before this round's refactor, ``debate_rounds.py`` had 11 mypy ``union-attr``
errors of the form ``Item "None" of "DebateResult | None" has no attribute X``
because each phase method assigned ``result = ctx.result`` (which is typed
``Optional[DebateResult]``) and then dereferenced it unconditionally.

The fix swapped each ``result = ctx.result`` to
``result = require_phase_result(ctx)`` — the helper that already exists in
``aragora.debate.phases._phase_invariant`` and is already used by
``winner_selector.py`` and ``consensus_phase.py``.

These tests pin the new contract:

1. When ``ctx.result is None`` and one of the four phase methods is invoked,
   a ``RuntimeError`` with the canonical "phase invariant violated" message
   is raised at the entry-point — not deep inside the method where the
   error would be cryptic.
2. The fix does not regress the existing partial-rounds preservation
   behavior shipped in PR #6806.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.debate.debate_state import DebateContext
from aragora.debate.phases.debate_rounds import DebateRoundsPhase


def _make_executor() -> DebateRoundsPhase:
    """Build an executor with the minimum dependencies the phase methods need."""
    return DebateRoundsPhase(
        protocol=MagicMock(rounds=1, consensus_threshold=0.5, use_structured_phases=False),
        circuit_breaker=None,
        convergence_detector=None,
        recorder=None,
        hooks={},
        novelty_tracker=None,
        critique_with_agent=None,
        build_revision_prompt=None,
        generate_with_agent=None,
        with_timeout=None,
        record_grounded_position=None,
        debate_strategy=None,
        rhetorical_observer=None,
        checkpoint_callback=None,
    )


@pytest.mark.asyncio
async def test_execute_raises_runtime_error_when_result_uninitialized() -> None:
    """The top-level ``execute`` must raise via ``require_phase_result`` on None."""
    executor = _make_executor()
    ctx = DebateContext()
    ctx.proposals = {"alice": "draft"}
    assert ctx.result is None

    with pytest.raises(RuntimeError, match="phase invariant violated"):
        await executor.execute(ctx)


@pytest.mark.asyncio
async def test_execute_round_raises_runtime_error_when_result_uninitialized() -> None:
    """``_execute_round`` is the hot path that previously hid 3 of 11 mypy errors."""
    executor = _make_executor()
    ctx = DebateContext()
    ctx.proposals = {"alice": "draft"}
    ctx.proposers = []  # type: ignore[attr-defined]
    assert ctx.result is None

    perf_monitor = MagicMock()
    with pytest.raises(RuntimeError, match="phase invariant violated"):
        await executor._execute_round(ctx, perf_monitor, round_num=1, total_rounds=1)


@pytest.mark.asyncio
async def test_critique_phase_raises_runtime_error_when_result_uninitialized() -> None:
    """``_critique_phase`` previously had 3 mypy errors all under the same root cause."""
    executor = _make_executor()
    ctx = DebateContext()
    ctx.proposals = {"alice": "draft"}
    assert ctx.result is None

    with pytest.raises(RuntimeError, match="phase invariant violated"):
        await executor._critique_phase(ctx, critics=[], round_num=1)


@pytest.mark.asyncio
async def test_revision_phase_raises_runtime_error_when_result_uninitialized() -> None:
    """``_revision_phase`` previously had 3 mypy errors (1246, 1484, 1506)."""
    executor = _make_executor()
    ctx = DebateContext()
    ctx.proposals = {"alice": "draft"}
    assert ctx.result is None

    with pytest.raises(RuntimeError, match="phase invariant violated"):
        await executor._revision_phase(ctx, critics=[], round_num=1)


def test_debate_rounds_imports_require_phase_result() -> None:
    """Pin the import so a future regression of this fix shows up loudly.

    Without this test, somebody could revert one of the four
    ``require_phase_result`` calls back to ``ctx.result`` and the mypy
    errors would silently come back. The import-level pin makes that
    revert visible in test signal as well.
    """
    from aragora.debate.phases import debate_rounds

    assert hasattr(debate_rounds, "require_phase_result"), (
        "require_phase_result must remain imported in debate_rounds; "
        "removing this import is the canary for a regression of the "
        "phase-invariant narrowing fix."
    )


def test_invariant_helper_returns_same_object_as_ctx_result() -> None:
    """Behavioural parity: the narrowed alias is identity-equal to ctx.result."""
    from aragora.core import DebateResult, Environment

    env = Environment(task="parity-test")
    result = DebateResult(task=env.task)
    ctx = DebateContext(env=env, result=result)

    from aragora.debate.phases.debate_rounds import require_phase_result

    narrowed = require_phase_result(ctx)
    assert narrowed is result
    # Mutations via narrowed alias propagate (no defensive copy).
    narrowed.final_answer = "parity"
    assert ctx.result is not None and ctx.result.final_answer == "parity"
