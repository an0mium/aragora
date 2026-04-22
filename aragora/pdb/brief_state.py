"""Back-compat shim for :mod:`aragora.brief_engine.lifecycle`.

The Mode 3 PDB lifecycle state machine moved to
:mod:`aragora.brief_engine.lifecycle` in the Phase 1 brief-engine
extraction. This module re-exports the same public names so external
callers that imported ``BriefLifecycleState`` etc. from
``aragora.pdb.brief_state`` continue to work unchanged.

Do not add new types here — put them in
:mod:`aragora.brief_engine.lifecycle` instead.
"""

from __future__ import annotations

from aragora.brief_engine.lifecycle import (
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
