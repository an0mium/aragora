"""Helpers for serialized boss-loop follow-up actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aragora.swarm.issue_scanner import infer_issue_category_from_title
from aragora.swarm.issue_upgrader import upgrade_issue_heuristic
from aragora.swarm.spec import SwarmSpec


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
    spec.constraints = _ordered_unique([*spec.constraints, *upgraded_spec.constraints])
    spec.track_hints = _ordered_unique([*spec.track_hints, *upgraded_spec.track_hints])
    spec.file_scope_hints = _ordered_unique(
        [*spec.file_scope_hints, *upgraded_spec.file_scope_hints, *inferred_scope]
    )
    spec.estimated_complexity = upgraded_spec.estimated_complexity or spec.estimated_complexity
    return spec


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
