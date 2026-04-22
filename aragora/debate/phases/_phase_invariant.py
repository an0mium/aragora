"""Phase invariant helpers for debate phase code.

Many phases in ``aragora/debate/phases`` construct ``ctx.result`` in an
earlier pipeline step (``context_init``) and then dereference it many
times. Typing-wise ``ctx.result`` is ``DebateResult | None``, which forces
every downstream access to repeat a redundant ``None`` check (and floods
mypy with ``union-attr`` errors when authors legitimately rely on the
invariant).

This module exposes a tiny helper that narrows the type in one place and
makes the invariant violation explicit if it is ever broken at runtime.

The helper is defined at module scope (rather than as a method on
``DebateContext``) so that test fakes which merely expose a ``result``
attribute — without re-implementing the full context API — continue to
work unmodified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.debate.context import DebateContext


def require_phase_result(ctx: "DebateContext") -> "DebateResult":
    """Return ``ctx.result`` after asserting it has been initialized.

    Args:
        ctx: The current :class:`DebateContext`. ``ctx.result`` must have
            been set by an earlier pipeline stage (typically
            ``context_init``).

    Returns:
        The non-``None`` :class:`DebateResult`. Downstream code can
        dereference attributes on the returned value without needing
        further narrowing.

    Raises:
        RuntimeError: If ``ctx.result`` is ``None``. This is a programming
            error — it indicates a phase ran before the context was
            initialized.
    """
    result = ctx.result
    if result is None:
        raise RuntimeError(
            "DebateContext.result has not been initialized; "
            "phase invariant violated (expected a DebateResult)."
        )
    return result


__all__ = ["require_phase_result"]
