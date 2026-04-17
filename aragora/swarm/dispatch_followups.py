"""Helpers for serialized boss-loop follow-up actions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aragora.swarm.issue_scanner import infer_issue_category_from_title
from aragora.swarm.issue_upgrader import upgrade_issue_heuristic
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.spec_upgrader import (
    SpecUpgraderUnavailable,
    UpgradeFailureContext,
    upgrade_spec,
)

_MARKDOWN_BULLET_RE = re.compile(r"^[-*]\s+(?:\[[ xX]\]\s+)?(?P<text>.+)$")


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _extract_acceptance_criteria(markdown: str) -> list[str]:
    criteria: list[str] = []
    in_acceptance_section = False
    for raw_line in str(markdown or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().rstrip(":").lower()
            in_acceptance_section = heading == "acceptance criteria"
            continue
        if not in_acceptance_section or not stripped:
            continue
        bullet_match = _MARKDOWN_BULLET_RE.match(stripped)
        if bullet_match:
            criteria.append(bullet_match.group("text"))
            continue
        criteria.append(stripped)
    return _ordered_unique(criteria)


def maybe_upgrade_dispatch_spec(
    *,
    issue: Any,
    spec: SwarmSpec,
    sanitized_issue_body: str,
    repo_root: Path,
) -> SwarmSpec:
    """Try upgrading an under-specified issue before blocking dispatch."""
    if spec.is_dispatch_bounded():
        return spec

    category = infer_issue_category_from_title(getattr(issue, "title", None))
    if category is None:
        return spec

    upgraded = upgrade_issue_heuristic(
        str(getattr(issue, "title", "") or ""),
        sanitized_issue_body,
        repo_root=repo_root,
        category=category,
        acceptance_criteria=list(getattr(spec, "acceptance_criteria", []) or []),
    )
    if upgraded is None:
        return spec

    upgraded_spec = SwarmSpec.from_direct_goal(
        f"[Issue #{issue.number}] {issue.title}\n\n{upgraded.upgraded_body}",
        budget_limit_usd=spec.budget_limit_usd,
        requires_approval=spec.requires_approval,
        user_expertise=spec.user_expertise,
        use_llm=False,
    )
    inferred_scope = SwarmSpec.infer_file_scope_hints(upgraded.upgraded_body)
    spec.raw_goal = upgraded_spec.raw_goal
    spec.refined_goal = upgraded_spec.refined_goal or spec.refined_goal
    spec.acceptance_criteria = _ordered_unique(
        [*spec.acceptance_criteria, *_extract_acceptance_criteria(upgraded.upgraded_body)]
    )
    spec.constraints = _ordered_unique([*spec.constraints, *upgraded_spec.constraints])
    spec.track_hints = _ordered_unique([*spec.track_hints, *upgraded_spec.track_hints])
    spec.file_scope_hints = _ordered_unique(
        [*spec.file_scope_hints, *upgraded_spec.file_scope_hints, *inferred_scope]
    )
    spec.estimated_complexity = upgraded_spec.estimated_complexity or spec.estimated_complexity
    return spec


_TRACK_TAG_RE = re.compile(r"^\s*\[([A-Z]+-\d+)\]")


def _extract_track_tag(issue_title: str) -> str | None:
    """Extract ``[TW-02]``-style prefix from an issue title."""
    match = _TRACK_TAG_RE.match(issue_title or "")
    return match.group(1) if match else None


def upgrade_unbounded_spec(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
) -> SwarmSpec | None:
    """Seam A: upgrade an unbounded spec before the contract-gate dispatch.

    Returns the upgraded :class:`SwarmSpec` if dispatch should proceed, or
    ``None`` if the upgrader escalated to ``needs-clarification`` (caller must
    skip dispatch).

    Raises :class:`SpecUpgraderUnavailable` on transient infrastructure
    failure -- caller treats as skip-for-this-tick.
    """
    if spec.is_dispatch_bounded():
        return spec
    ctx = UpgradeFailureContext(
        missing_bounds=spec.missing_dispatch_bounds(),
        preflight_diff=None,
        prior_attempts=0,  # read durably inside ``upgrade_spec``
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )
    result = upgrade_spec(
        spec,
        ctx,
        issue_number=issue_number,
        seam="A",
        repo_root=repo_root,
        metrics_path=metrics_path,
        llm_client=llm_client,
    )
    if result.status == "upgraded":
        return result.upgraded_spec
    return None


def upgrade_on_contract_drift(
    spec: SwarmSpec,
    *,
    issue_number: int,
    issue_title: str,
    issue_body: str,
    preflight_diff: dict,
    repo_root: Path,
    metrics_path: Path,
    llm_client: Any = None,
) -> SwarmSpec | None:
    """Seam B: upgrade a spec after contract-gate reported drift.

    Returns the upgraded spec to retry dispatch, or ``None`` to escalate
    (caller skips). Raises :class:`SpecUpgraderUnavailable` on transient infra
    failure.
    """
    ctx = UpgradeFailureContext(
        missing_bounds=list(spec.missing_dispatch_bounds()),
        preflight_diff=preflight_diff,
        prior_attempts=0,  # read durably inside ``upgrade_spec``
        original_issue_body=issue_body,
        issue_title=issue_title,
        track_tag=_extract_track_tag(issue_title),
    )
    result = upgrade_spec(
        spec,
        ctx,
        issue_number=issue_number,
        seam="B",
        repo_root=repo_root,
        metrics_path=metrics_path,
        llm_client=llm_client,
    )
    if result.status == "upgraded":
        return result.upgraded_spec
    return None


__all__ = [
    "SpecUpgraderUnavailable",
    "annotate_result_with_conductor",
    "maybe_upgrade_dispatch_spec",
    "upgrade_on_contract_drift",
    "upgrade_unbounded_spec",
]


def annotate_result_with_conductor(
    *,
    issue_number: int,
    result: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Attach conductor follow-up hints to non-success dispatch results."""
    if result.get("status") not in {"needs_human", "failed"}:
        return result

    try:
        from aragora.swarm.conductor import Conductor

        step = Conductor(repo_root=repo_root).evaluate_worker_output(issue_number, result)
    except Exception:
        return result

    annotated = dict(result)
    annotated.update(
        {
            "conductor_next_action": step.next_action,
            "conductor_next_prompt": (step.next_prompt or "")[:500],
            "conductor_terminal_class": step.terminal_class.value
            if hasattr(step.terminal_class, "value")
            else str(step.terminal_class),
        }
    )
    return annotated
