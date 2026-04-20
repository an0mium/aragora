"""PDB — PR Decision Brief lifecycle package.

This package hosts the storage + state machine that back the Mode 3
on-demand PR brief generation pipeline described in
``docs/plans/2026-04-20-pdb-brief-generation-mode3-design.md``.

Public surface:

- :class:`BriefLifecycleState` — the six canonical lifecycle states
  (``absent``, ``queued``, ``running``, ``ready``, ``failed``, ``stale``)
- :class:`StateTransitionError` — raised when an illegal transition is
  attempted or when asserting into the same state
- :func:`validate_transition` — assert a transition is legal
- :mod:`aragora.pdb.storage` — flat-file storage layer under
  ``.aragora/review-queue/briefs/`` (see module docstring)

Subsequent PRs layer on executor, panel config, budget enforcement, and
backend endpoints. This package is storage + state only.
"""

from __future__ import annotations

from aragora.pdb.brief_state import (
    BriefLifecycleState,
    LEGAL_TRANSITIONS,
    StateTransitionError,
    validate_transition,
)

__all__ = [
    "BriefLifecycleState",
    "LEGAL_TRANSITIONS",
    "StateTransitionError",
    "validate_transition",
]
