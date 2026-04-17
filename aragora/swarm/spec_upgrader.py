"""SpecUpgrader: convert weak GitHub-issue specs into dispatchable SwarmSpecs.

Public entry point: ``upgrade_spec()``. See
``docs/plans/2026-04-17-spec-upgrader-design.md`` for the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aragora.swarm.spec import SwarmSpec

UpgradePath = Literal["deterministic", "llm", "deterministic+llm"]
UpgradeStatus = Literal["upgraded", "escalated"]


class SpecUpgraderUnavailable(Exception):
    """Raised for transient infrastructure failure (LLM 5xx, timeout, etc.).

    Callers should treat this as 'skip for this tick, retry next tick'.
    Does NOT consume an attempt in the durable counter.
    """


@dataclass(frozen=True)
class UpgradeFailureContext:
    """Structured input to the upgrader, explaining why the spec needs upgrading."""

    missing_bounds: list[str]
    preflight_diff: dict | None
    prior_attempts: int
    original_issue_body: str
    issue_title: str
    track_tag: str | None


@dataclass(frozen=True)
class UpgradeResult:
    """Outcome of an upgrade attempt. Tagged union via ``status`` field."""

    status: UpgradeStatus
    upgraded_spec: SwarmSpec | None
    audit_markdown: str
    attempt_count: int
    upgrade_path: UpgradePath | None
    failure_context: UpgradeFailureContext
    unresolved_questions: list[str] = field(default_factory=list)


# Labels emitted by ``SwarmSpec.missing_dispatch_bounds()`` mapped to actionable
# enrichment flags. Keep the keys in sync with ``aragora/swarm/spec.py`` -- in
# particular ``"explicit work order"`` matches the label returned by the spec
# (not ``"work order"``).
_BOUND_LABELS = {
    "acceptance criterion": "needs_acceptance",
    "file-scope hint": "needs_file_scope",
    "constraint": "needs_constraint",
    "explicit work order": "needs_work_order",
}


def _classify_missing_bounds(missing_bounds: list[str]) -> dict[str, bool]:
    """Map ``missing_dispatch_bounds()`` labels to actionable flags for enrichment."""
    classified = {flag: False for flag in _BOUND_LABELS.values()}
    for label in missing_bounds:
        flag = _BOUND_LABELS.get(label)
        if flag is not None:
            classified[flag] = True
    return classified
