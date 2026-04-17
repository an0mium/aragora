"""SpecUpgrader: convert weak GitHub-issue specs into dispatchable SwarmSpecs.

Public entry point: ``upgrade_spec()``. See
``docs/plans/2026-04-17-spec-upgrader-design.md`` for the architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
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


# Matches common Python/TS/MD file references. Intentionally narrow to avoid false
# positives.
_PATH_RE = re.compile(r"(?P<path>[a-zA-Z0-9_\-./]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|sh))")


def _extract_file_paths(issue_body: str, *, repo_root: Path) -> list[str]:
    """Extract file paths mentioned in the issue body and validate existence.

    Only paths that actually exist (relative to ``repo_root``) are returned. Paths
    that are hallucinated or merely aspirational are dropped.
    """
    candidates: set[str] = set()
    for match in _PATH_RE.finditer(issue_body):
        candidate = match.group("path").strip("./")
        if "/" in candidate and (repo_root / candidate).is_file():
            candidates.add(candidate)
    return sorted(candidates)


# Low-confidence candidate scopes per track-tag prefix. Must be validated against
# the current repo before merging into a spec.
_TRACK_SCOPE_CANDIDATES: dict[str, list[str]] = {
    "TW": ["aragora/swarm/"],
    "CS": ["aragora/swarm/", "docs/status/"],
    "RS": ["aragora/swarm/"],
}

# Design-heavy tracks must NOT use path inference; fall through to LLM or escalate.
_DESIGN_HEAVY_PREFIXES = frozenset({"AGT", "DIC"})


def _infer_track_scope(track_tag: str | None, *, issue_body: str, repo_root: Path) -> list[str]:
    """Return validated candidate scope hints for ``track_tag``, or ``[]`` to fall through."""
    if not track_tag:
        return []
    prefix = track_tag.split("-", 1)[0].upper()
    if prefix in _DESIGN_HEAVY_PREFIXES:
        return []
    candidates = _TRACK_SCOPE_CANDIDATES.get(prefix)
    if not candidates:
        return []
    validated = [c for c in candidates if (repo_root / c.rstrip("/")).is_dir()]
    return validated


def _drift_to_acceptance_criterion(drift: dict | None) -> str | None:
    """Translate preflight contract drift into an actionable acceptance criterion.

    Returns ``None`` if drift is absent or the expected and actual files match.
    """
    if not drift:
        return None
    expected = drift.get("expected", {}) or {}
    actual = drift.get("actual", {}) or {}
    expected_files = list(expected.get("files", []))
    actual_files = set(actual.get("files", []))
    if not expected_files or set(expected_files) == actual_files:
        return None
    files_str = ", ".join(f"`{f}`" for f in expected_files)
    return (
        f"Worker must scope changes strictly to: {files_str}. "
        "Reject any edits to files outside this list during preflight."
    )
