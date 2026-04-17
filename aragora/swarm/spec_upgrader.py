"""SpecUpgrader: convert weak GitHub-issue specs into dispatchable SwarmSpecs.

Public entry point: ``upgrade_spec()``. See
``docs/plans/2026-04-17-spec-upgrader-design.md`` for the architecture.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

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


def _tier1_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    repo_root: Path,
) -> SwarmSpec | None:
    """Deterministic enrichment from static signals (no LLM).

    Returns the upgraded spec if the enrichment bounds it, otherwise ``None``
    to signal that Tier 2 (LLM) is needed.
    """
    flags = _classify_missing_bounds(ctx.missing_bounds)
    extracted_paths = _extract_file_paths(ctx.original_issue_body, repo_root=repo_root)
    track_hints = _infer_track_scope(
        ctx.track_tag, issue_body=ctx.original_issue_body, repo_root=repo_root
    )
    drift_crit = _drift_to_acceptance_criterion(ctx.preflight_diff)

    new_file_scope = list(spec.file_scope_hints)
    if flags["needs_file_scope"]:
        for path in extracted_paths:
            if path not in new_file_scope:
                new_file_scope.append(path)
        for hint in track_hints:
            if hint not in new_file_scope:
                new_file_scope.append(hint)

    # Always add drift criterion when drift is present -- it conveys a
    # scoping constraint beyond whatever ``missing_bounds`` flags imply.
    new_acceptance = list(spec.acceptance_criteria)
    if drift_crit and drift_crit not in new_acceptance:
        new_acceptance.append(drift_crit)
    if flags["needs_acceptance"] and not new_acceptance and ctx.issue_title and new_file_scope:
        new_acceptance.append(f"Implement the behavior described by: {ctx.issue_title.strip()}")

    new_constraints = list(spec.constraints)
    if flags["needs_constraint"] and new_file_scope:
        constraint = (
            f"Limit modifications to the listed file-scope hints: {', '.join(new_file_scope)}."
        )
        if constraint not in new_constraints:
            new_constraints.append(constraint)

    new_work_orders: list[dict[str, Any]] = list(spec.work_orders)
    if flags["needs_work_order"] and new_acceptance:
        seed_order = {"description": f"Satisfy: {new_acceptance[0]}"}
        if seed_order not in new_work_orders:
            new_work_orders.append(seed_order)

    candidate = replace(
        spec,
        file_scope_hints=new_file_scope,
        acceptance_criteria=new_acceptance,
        constraints=new_constraints,
        work_orders=new_work_orders,
    )
    if candidate.is_dispatch_bounded():
        return candidate
    return None


class _LLMLogicFailure(Exception):
    """Internal: LLM returned malformed / ungrounded output after local retry."""


def _build_tier2_prompt(spec: SwarmSpec, ctx: UpgradeFailureContext, repo_root: Path) -> str:
    """Build the Tier 2 LLM prompt from the current spec + failure context."""
    del repo_root  # reserved for future use (e.g., injecting repo tree)
    return f"""You are upgrading an underspecified GitHub issue into a dispatchable SwarmSpec.

Issue title: {ctx.issue_title}
Issue body:
{ctx.original_issue_body}

Missing bounds: {ctx.missing_bounds}
Preflight drift: {json.dumps(ctx.preflight_diff) if ctx.preflight_diff else "none"}

Current spec state:
- acceptance_criteria: {spec.acceptance_criteria}
- file_scope_hints: {spec.file_scope_hints}
- constraints: {spec.constraints}
- work_orders: {spec.work_orders}

Respond with ONLY a JSON object containing fields to ADD (not replace) to the spec:
{{
  "acceptance_criteria": [...],
  "file_scope_hints": [...],
  "constraints": [...],
  "work_orders": [...]
}}

Rules:
- File paths MUST exist in the repo. Do not invent paths.
- Acceptance criteria must be specific and verifiable.
- Constraints must be enforceable (e.g., "no changes outside listed files").
- Omit any field you cannot responsibly fill.
"""


def _tier2_enrich(
    spec: SwarmSpec,
    ctx: UpgradeFailureContext,
    *,
    client: Any,
    repo_root: Path,
) -> SwarmSpec | None:
    """LLM-backed enrichment.

    Raises :class:`SpecUpgraderUnavailable` on transient infrastructure errors
    (timeouts, connection errors) so the caller can skip-for-this-tick without
    consuming an attempt. Raises :class:`_LLMLogicFailure` on malformed or
    ungrounded output after one local retry.

    Returns the upgraded ``SwarmSpec`` on success, or ``None`` if the upgrade
    still isn't dispatch-bounded (caller escalates).
    """
    prompt = _build_tier2_prompt(spec, ctx, repo_root)
    last_err: Exception | None = None

    for attempt in range(2):
        try:
            raw = client.complete(prompt)
        except (ConnectionError, TimeoutError) as exc:
            raise SpecUpgraderUnavailable(str(exc)) from exc
        except Exception as exc:  # transient infra error surfaced by client
            last_err = exc
            if attempt == 0:
                time.sleep(1)
                continue
            raise SpecUpgraderUnavailable(str(exc)) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            if attempt == 0:
                continue
            raise _LLMLogicFailure(f"LLM output not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise _LLMLogicFailure("LLM output not a JSON object")

        added_acceptance = [str(item) for item in parsed.get("acceptance_criteria", []) if item]
        added_file_scope = [str(item) for item in parsed.get("file_scope_hints", []) if item]
        added_constraints = [str(item) for item in parsed.get("constraints", []) if item]
        added_work_orders: list[dict[str, Any]] = []
        for item in parsed.get("work_orders", []) or []:
            if isinstance(item, dict):
                added_work_orders.append(item)
            elif isinstance(item, str) and item.strip():
                added_work_orders.append({"description": item.strip()})

        candidate = replace(
            spec,
            acceptance_criteria=[*spec.acceptance_criteria, *added_acceptance],
            file_scope_hints=[*spec.file_scope_hints, *added_file_scope],
            constraints=[*spec.constraints, *added_constraints],
            work_orders=[*spec.work_orders, *added_work_orders],
        )
        if candidate.is_dispatch_bounded():
            return candidate
        # Still unbounded even after LLM enrichment -- caller escalates.
        return None

    raise _LLMLogicFailure(f"Exhausted LLM attempts: {last_err}")
