"""PDB — PR Decision Brief lifecycle package (Mode 3).

This package hosts the Mode-3-specific wiring for PR Decision Briefs:

- :mod:`aragora.pdb.prompts` — PR-review prompt templates
- :mod:`aragora.pdb.input_loader` — ``gh``-backed PR input builder
- :mod:`aragora.pdb.panel_config` — wraps
  :mod:`aragora.brief_engine.panel_config` with the committed PDB
  default yaml path (``aragora/config/pdb_panel.yaml``)
- :mod:`aragora.pdb.protocol` — wraps
  :mod:`aragora.brief_engine.protocol` with the PDB prompt renderer +
  default panel config
- :mod:`aragora.pdb.storage`, :mod:`aragora.pdb.worker`,
  :mod:`aragora.pdb.brief_state`, :mod:`aragora.pdb.budget` — back-
  compat shims that re-export the generic primitives from
  :mod:`aragora.brief_engine` under their legacy ``PDB*`` names.

The generic brief-engine primitives (storage, lifecycle, budget,
worker, executor) were extracted in Phase 1 — see
``docs/plans/2026-04-22-security-report-brief-design.md`` §Phase 1 —
to make future brief variants (SecurityReportBrief, license audit,
compliance review) buildable without copy-paste. Mode 3 PDB remains
the reference consumer; nothing below the surface changed.
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
