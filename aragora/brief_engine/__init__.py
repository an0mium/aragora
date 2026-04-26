"""Brief-engine primitives.

This package hosts the reusable lifecycle + executor + worker + budget
stack behind variant-specific brief pipelines. Mode 3 PDB is the first
in-tree consumer; future variants (SecurityReportBrief, compliance
review, license audit) are configurations of the same primitives.

See ``docs/plans/2026-04-22-security-report-brief-design.md`` §Rollout
plan §Phase 1 for the extraction context.

Public surface
--------------

- :class:`BriefLifecycleState` / :func:`validate_transition` — the six
  canonical lifecycle states and transition table.
- :mod:`aragora.brief_engine.storage` — flat-file storage layer keyed
  on ``(pr_number, head_sha)``.
- :mod:`aragora.brief_engine.budget` — budget reservation + denial.
- :mod:`aragora.brief_engine.panel_config` — typed panel config
  schema and validator.
- :mod:`aragora.brief_engine.protocol` — Protocol B executor
  (:func:`run_brief_protocol_b`).
- :mod:`aragora.brief_engine.worker` — bounded-concurrency in-process
  worker.

The PDB-specific prompts and input loader stay under
:mod:`aragora.pdb`; that package re-exports the generic primitives
under their legacy ``PDB*`` names for backward compatibility.
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
