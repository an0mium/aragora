"""Owner-aware review queue conductor packet builder.

This module is intentionally read-only.  It stitches together the queue surfaces
operators currently poll by hand: open PR metadata, required checks, lane owner
lookup, operator steering, merge-packet status, head-change detection, and
lightweight supersession hints.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aragora.cli.commands.review_queue import (
    _GhError,
    _build_merge_authorization_packet,
    _fetch_required_pr_check_surface,
    _required_pr_check_bucket,
    _summarize_required_pr_checks,
)
from aragora.cli.commands.review_queue_transport import (
    _gh_error_kind,
    _gh_json,
    _gh_json_with_transport_retries,
    _is_github_transport_error,
)
from aragora.worktree.fleet import resolve_repo_root


QUEUE_CONDUCTOR_VERSION = "queue_conductor.v1"
QUEUE_CONDUCTOR_DEFAULT_MODE = "queue"
QUEUE_CONDUCTOR_READY_BOUNDARY_MODE = "ready-boundary"
OWNER_TIMEOUT_CLASSIFICATION = "owner_lookup_timeout_preserve"
TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION = "tier3_or_tier4_evidence_required"
READY_BOUNDARY_MARK_READY_CLASSIFICATION = "ready_boundary_mark_ready_authorization_required"
ACTIVE_OWNER_STATUSES = {
    "active",
    "claimed",
    "running",
    "in_progress",
    "busy",
    "working",
}
NON_ACTIONABLE_CANCELLED_PREFIXES = (
    "build documentation",
    "metrics drift",
    "module tier drift",
    "portability lint",
    "pr admission signal (advisory)",
    "self-hosted shadow",
)
SUPERSEDED_CANCELLED_CHECKS = {
    "lint": {
        "lint-run": "lint",
        "typecheck-run": "typecheck",
    },
}
READY_BOUNDARY_WATCHED_WORKFLOWS = (
    "Tests",
    "Security Gate",
    "Release Readiness",
    "Aragora Code Review",
    "Core Suites",
    "Smoke Tests",
)

PR_LIST_FIELDS = (
    "number,title,url,headRefName,headRefOid,isDraft,state,mergeable,mergeStateStatus,updatedAt"
)
PR_VIEW_FIELDS = (
    "number,title,url,headRefName,headRefOid,isDraft,state,mergeable,mergeStateStatus,"
    "updatedAt,files,statusCheckRollup"
)


@dataclass(frozen=True)
class ConductorProviders:
    """Dependency hooks used by tests and by the CLI builder."""

    gh_json: Callable[[list[str]], Any] = _gh_json
    rest_json: Callable[[list[str]], Any] = _gh_json
    required_surface: Callable[[int, str | None], dict[str, Any]] = _fetch_required_pr_check_surface
    merge_packet: Callable[..., dict[str, Any]] = _build_merge_authorization_packet
    owner_lookup: Callable[[str, float], dict[str, Any]] | None = None
    steering_lookup: Callable[[str, float], dict[str, Any]] | None = None
    origin_main_sha: Callable[[], str] | None = None


def build_queue_conductor_packet(
    *,
    pr_refs: list[str] | None = None,
    limit: int = 30,
    repo_override: str | None = None,
    review_queue_root: str | Path | None = None,
    owner_timeout_seconds: float = 8.0,
    mode: str = QUEUE_CONDUCTOR_DEFAULT_MODE,
    providers: ConductorProviders | None = None,
) -> dict[str, Any]:
    """Build a read-only owner-aware queue conductor packet."""

    active_providers = providers or ConductorProviders()
    conductor_mode = _normalize_mode(mode)
    pr_views = _fetch_pr_views(
        pr_refs=pr_refs or [],
        limit=limit,
        repo_override=repo_override,
        gh_json=active_providers.gh_json,
        rest_json=active_providers.rest_json,
    )
    initial_heads: dict[int, str] = {}
    for view in pr_views:
        pr_number = _safe_int(view.get("number"))
        if pr_number is not None:
            initial_heads[pr_number] = str(view.get("headRefOid") or "")
    file_index = _candidate_file_index(pr_views)

    candidates: list[dict[str, Any]] = []
    for view in pr_views:
        candidate = _build_candidate(
            view=view,
            all_prs=pr_views,
            file_index=file_index,
            initial_head=str(view.get("headRefOid") or ""),
            repo_override=repo_override,
            review_queue_root=review_queue_root,
            owner_timeout_seconds=owner_timeout_seconds,
            providers=active_providers,
        )
        candidates.append(candidate)

    next_prompt = _build_next_prompt(
        candidates,
        repo_override=repo_override,
        mode=conductor_mode,
    )
    return {
        "version": QUEUE_CONDUCTOR_VERSION,
        "mode": conductor_mode,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repo": repo_override or "",
        "origin_main_sha": _safe_origin_main_sha(active_providers),
        "pr_refs": [str(ref) for ref in (pr_refs or [])],
        "limit": limit,
        "initial_heads": initial_heads,
        "candidates": candidates,
        "next_prompt": next_prompt,
        "tooling_notes": [
            "read_only_packet",
            "owner_lookup_timeout_is_preserve_no_mutate",
            "active_owner_blocks_mutation",
        ],
    }


def render_queue_conductor_packet(packet: dict[str, Any]) -> str:
    """Render a compact operator-facing conductor summary."""

    lines = [
        f"Queue conductor {packet.get('version')} generated {packet.get('generated_at')}",
        f"origin/main: {packet.get('origin_main_sha') or 'unknown'}",
        "",
    ]
    for candidate in packet.get("candidates") or []:
        lines.append(
            "#{} {} head={} owner={} class={} mutate={}".format(
                candidate.get("pr_number"),
                candidate.get("title"),
                candidate.get("head_sha") or "unknown",
                candidate.get("owner", {}).get("state", "unknown"),
                candidate.get("classification"),
                candidate.get("mutate_allowed"),
            )
        )
        action = candidate.get("next_action")
        if action:
            lines.append(f"  next: {action}")
        ready_boundary = candidate.get("ready_boundary")
        if isinstance(ready_boundary, dict):
            lines.append(
                "  ready-boundary: {} eligible={}".format(
                    ready_boundary.get("classification"),
                    ready_boundary.get("eligible_for_mark_ready_authorization"),
                )
            )
            watched_rollup = ready_boundary.get("watched_rollup")
            if isinstance(watched_rollup, dict) and watched_rollup.get("summary"):
                lines.append(f"  watched-rollup: {watched_rollup.get('summary')}")
    lines.extend(["", "Best next prompt:", str(packet.get("next_prompt") or "")])
    return "\n".join(lines)


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or QUEUE_CONDUCTOR_DEFAULT_MODE).strip().lower()
    if normalized in {"ready_boundary", "ready"}:
        normalized = QUEUE_CONDUCTOR_READY_BOUNDARY_MODE
    if normalized not in {QUEUE_CONDUCTOR_DEFAULT_MODE, QUEUE_CONDUCTOR_READY_BOUNDARY_MODE}:
        raise ValueError(
            f"unsupported queue conductor mode {mode!r}; expected "
            f"{QUEUE_CONDUCTOR_DEFAULT_MODE!r} or {QUEUE_CONDUCTOR_READY_BOUNDARY_MODE!r}"
        )
    return normalized


def _fetch_pr_views(
    *,
    pr_refs: list[str],
    limit: int,
    repo_override: str | None,
    gh_json: Callable[[list[str]], Any],
    rest_json: Callable[[list[str]], Any],
) -> list[dict[str, Any]]:
    if pr_refs:
        refs = list(dict.fromkeys(str(ref).strip() for ref in pr_refs if str(ref).strip()))
        return [
            _fetch_pr_view(
                str(ref),
                repo_override=repo_override,
                gh_json=gh_json,
                rest_json=rest_json,
            )
            for ref in refs
        ]

    args = [
        "pr",
        "list",
        "--state",
        "open",
        "--limit",
        str(limit),
        "--json",
        PR_LIST_FIELDS,
    ]
    if repo_override:
        args.extend(["--repo", repo_override])
    payload = gh_json(args)
    if not isinstance(payload, list):
        raise _GhError("gh pr list returned a non-list payload")
    refs = [
        str(item.get("number")) for item in payload if isinstance(item, dict) and item.get("number")
    ]
    return [
        _fetch_pr_view(
            ref,
            repo_override=repo_override,
            gh_json=gh_json,
            rest_json=rest_json,
        )
        for ref in refs[: max(limit, 0)]
    ]


def _fetch_pr_view(
    pr_ref: str,
    *,
    repo_override: str | None,
    gh_json: Callable[[list[str]], Any],
    rest_json: Callable[[list[str]], Any],
) -> dict[str, Any]:
    args = ["pr", "view", str(pr_ref), "--json", PR_VIEW_FIELDS]
    if repo_override:
        args.extend(["--repo", repo_override])
    try:
        payload = gh_json(args)
    except _GhError as exc:
        if not _is_github_transport_error(exc):
            raise
        return _fetch_pr_view_rest(
            pr_ref,
            repo_override=repo_override,
            rest_json=rest_json,
            graphql_error=str(exc),
        )
    if not isinstance(payload, dict):
        raise _GhError(f"gh pr view {pr_ref} returned a non-object payload")
    return payload


def _fetch_pr_view_rest(
    pr_ref: str,
    *,
    repo_override: str | None,
    rest_json: Callable[[list[str]], Any],
    graphql_error: str,
) -> dict[str, Any]:
    """Best-effort REST metadata/check-run fallback for conductor reads."""
    pr_number = _safe_int(pr_ref)
    repo_slug = _repo_slug_from_override(repo_override)
    if pr_number is None or not repo_slug:
        raise _GhError(
            f"gh pr view {pr_ref} transport failed and REST fallback is unavailable: "
            f"{graphql_error}"
        )

    try:
        pr_payload = _gh_json_with_transport_retries(
            ["api", f"repos/{repo_slug}/pulls/{pr_number}"],
            gh_json=rest_json,
            attempts=2,
        )
    except _GhError as exc:
        raise _GhError(
            f"gh pr view {pr_ref} transport failed and REST PR fallback failed: "
            f"{graphql_error}; {exc}"
        ) from exc
    if not isinstance(pr_payload, dict):
        raise _GhError(f"REST PR fallback for {pr_ref} returned a non-object payload")

    files_payload, files_error = _try_rest_json(
        ["api", f"repos/{repo_slug}/pulls/{pr_number}/files?per_page=100"],
        rest_json=rest_json,
    )
    files = _rest_files(files_payload)
    head_sha = _nested_str(pr_payload, "head", "sha")
    check_runs_payload, check_runs_error = _try_rest_json(
        ["api", f"repos/{repo_slug}/commits/{head_sha}/check-runs?per_page=100"],
        rest_json=rest_json,
    )
    rollup = _rollup_from_rest_check_runs(check_runs_payload)
    view = _rest_pr_to_view(pr_payload, files=files, rollup=rollup)
    view["_transport_fallback"] = {
        "source": "rest",
        "graphql_error": graphql_error,
        "pr_metadata_available": True,
        "files_available": files_payload is not None,
        "check_runs_available": check_runs_payload is not None,
        "files_error": files_error,
        "check_runs_error": check_runs_error,
    }
    return view


def _try_rest_json(
    args: list[str],
    *,
    rest_json: Callable[[list[str]], Any],
) -> tuple[Any | None, str]:
    try:
        return (
            _gh_json_with_transport_retries(args, gh_json=rest_json, attempts=2),
            "",
        )
    except Exception as exc:  # noqa: BLE001 - fallback metadata should remain best-effort.
        return None, str(exc)


def _repo_slug_from_override(repo_override: str | None) -> str:
    raw = str(repo_override or "").strip()
    if not raw:
        return ""
    raw = raw.removeprefix("repos/").strip("/")
    if raw.startswith("http"):
        match = re.search(r"github\.com[:/]+([^/]+)/([^/.?#]+)", raw)
        return f"{match.group(1)}/{match.group(2)}" if match else ""
    if "/" in raw and not raw.startswith("-"):
        return raw
    return ""


def _rest_pr_to_view(
    payload: dict[str, Any],
    *,
    files: list[dict[str, str]],
    rollup: list[dict[str, str]],
) -> dict[str, Any]:
    state = str(payload.get("state") or "").upper()
    merged_at = str(payload.get("merged_at") or payload.get("mergedAt") or "")
    if merged_at:
        state = "MERGED"
    mergeable_state = str(payload.get("mergeable_state") or "").upper()
    mergeable = _rest_mergeable(payload.get("mergeable"), mergeable_state)
    labels = [
        {"name": str(label.get("name") or "")}
        for label in payload.get("labels") or []
        if isinstance(label, dict) and str(label.get("name") or "").strip()
    ]
    return {
        "number": payload.get("number"),
        "title": str(payload.get("title") or ""),
        "url": str(payload.get("html_url") or payload.get("url") or ""),
        "headRefName": _nested_str(payload, "head", "ref"),
        "headRefOid": _nested_str(payload, "head", "sha"),
        "baseRefName": _nested_str(payload, "base", "ref"),
        "baseRefOid": _nested_str(payload, "base", "sha"),
        "isDraft": bool(payload.get("draft")),
        "state": state,
        "mergeable": mergeable,
        "mergeStateStatus": mergeable_state,
        "updatedAt": str(payload.get("updated_at") or ""),
        "mergedAt": merged_at,
        "author": {"login": _nested_str(payload, "user", "login")},
        "labels": labels,
        "additions": payload.get("additions") or 0,
        "deletions": payload.get("deletions") or 0,
        "changedFiles": payload.get("changed_files") or len(files),
        "body": str(payload.get("body") or ""),
        "files": files,
        "statusCheckRollup": rollup,
    }


def _rest_mergeable(value: Any, mergeable_state: str) -> str:
    if value is True:
        return "MERGEABLE"
    if value is False:
        if mergeable_state in {"DIRTY", "CONFLICTING"}:
            return "CONFLICTING"
        return "UNKNOWN"
    return "UNKNOWN"


def _rest_files(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    files: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = str(item.get("filename") or item.get("path") or "").strip()
        if path:
            files.append({"path": path})
    return files


def _rollup_from_rest_check_runs(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    runs = payload.get("check_runs")
    if not isinstance(runs, list):
        return []
    rollup: list[dict[str, str]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        raw_suite = run.get("check_suite")
        suite: dict[str, Any] = raw_suite if isinstance(raw_suite, dict) else {}
        raw_app = suite.get("app")
        app: dict[str, Any] = raw_app if isinstance(raw_app, dict) else {}
        rollup.append(
            {
                "name": str(run.get("name") or run.get("context") or ""),
                "workflowName": str(run.get("workflowName") or app.get("name") or ""),
                "status": str(run.get("status") or "").upper(),
                "conclusion": str(run.get("conclusion") or "").upper(),
                "detailsUrl": str(run.get("html_url") or run.get("details_url") or ""),
                "source": "rest_check_runs",
            }
        )
    return rollup


def _nested_str(payload: dict[str, Any], *path: str) -> str:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "")


def _build_candidate(
    *,
    view: dict[str, Any],
    all_prs: list[dict[str, Any]],
    file_index: dict[int, set[str]],
    initial_head: str,
    repo_override: str | None,
    review_queue_root: str | Path | None,
    owner_timeout_seconds: float,
    providers: ConductorProviders,
) -> dict[str, Any]:
    pr_number = int(view.get("number") or 0)
    branch = str(view.get("headRefName") or "")
    head_sha = str(view.get("headRefOid") or "")
    required = _required_summary(
        providers.required_surface(pr_number, repo_override),
    )
    owner_lookup = providers.owner_lookup or _default_owner_lookup
    steering_lookup = providers.steering_lookup or _default_steering_lookup
    owner = owner_lookup(branch, owner_timeout_seconds) if branch else _missing_branch_owner()
    steering = (
        steering_lookup(branch, owner_timeout_seconds) if branch else _missing_branch_steering()
    )
    merge_packet = _merge_packet_summary(
        pr_number=pr_number,
        repo_override=repo_override,
        review_queue_root=review_queue_root,
        merge_packet=providers.merge_packet,
    )
    packet_head = str(merge_packet.get("head_sha") or "")
    head_changed = bool(initial_head and packet_head and packet_head != initial_head)
    supersession_hints = _supersession_hints(view, all_prs, file_index)
    rollup = _rollup_summary(view.get("statusCheckRollup") or [])
    classification, mutate_allowed, next_action = _classify_candidate(
        view=view,
        required=required,
        owner=owner,
        merge_packet=merge_packet,
        rollup=rollup,
        head_changed=head_changed,
        supersession_hints=supersession_hints,
    )
    ready_boundary = _ready_boundary_summary(
        view=view,
        required=required,
        owner=owner,
        steering=steering,
        merge_packet=merge_packet,
        rollup=rollup,
        head_changed=head_changed,
        repo_override=repo_override,
    )
    return {
        "pr_number": pr_number,
        "title": str(view.get("title") or ""),
        "url": str(view.get("url") or ""),
        "branch": branch,
        "head_sha": head_sha,
        "initial_head_sha": initial_head,
        "head_changed": head_changed,
        "state": str(view.get("state") or ""),
        "is_draft": bool(view.get("isDraft")),
        "mergeable": str(view.get("mergeable") or ""),
        "merge_state_status": str(view.get("mergeStateStatus") or ""),
        "updated_at": str(view.get("updatedAt") or ""),
        "files": sorted(file_index.get(pr_number, set())),
        "transport_fallback": view.get("_transport_fallback") or {},
        "required_checks": required,
        "rollup": rollup,
        "owner": owner,
        "steering": steering,
        "merge_packet": merge_packet,
        "ready_boundary": ready_boundary,
        "supersession_hints": supersession_hints,
        "classification": classification,
        "mutate_allowed": mutate_allowed,
        "next_action": next_action,
    }


def _required_summary(surface: dict[str, Any]) -> dict[str, Any]:
    checks = [item for item in surface.get("checks") or [] if isinstance(item, dict)]
    summary, has_failures, has_pending = _summarize_required_pr_checks(checks)
    blocking = [
        _check_descriptor(item)
        for item in checks
        if _required_pr_check_bucket(item) not in {"pass", "skipping"}
    ]
    return {
        "available": bool(surface.get("available")),
        "summary": summary,
        "has_failures": bool(has_failures),
        "has_pending": bool(has_pending),
        "blocking": blocking,
        "error": str(surface.get("error") or ""),
        "error_kind": str(surface.get("error_kind") or ""),
        "transport_blocked": bool(surface.get("transport_blocked")),
        "preserve_no_mutate": bool(surface.get("preserve_no_mutate")),
    }


def _rollup_summary(items: list[Any]) -> dict[str, Any]:
    checks = [item for item in items if isinstance(item, dict)]
    descriptors: list[tuple[str, dict[str, str]]] = []
    for item in checks:
        bucket = _rollup_check_bucket(item)
        descriptors.append((bucket, _check_descriptor(item, bucket=bucket)))
    successful = [descriptor for bucket, descriptor in descriptors if bucket == "pass"]

    pending: list[dict[str, str]] = []
    failing: list[dict[str, str]] = []
    cancelled: list[dict[str, str]] = []
    non_actionable: list[dict[str, str]] = []
    for bucket, descriptor in descriptors:
        if bucket == "pending":
            pending.append(descriptor)
        elif bucket == "fail":
            failing.append(descriptor)
        elif bucket == "cancel":
            cancelled.append(descriptor)
            if _is_non_actionable_cancelled_descriptor(descriptor, successful):
                non_actionable.append(descriptor)

    actionable_cancelled = [item for item in cancelled if item not in non_actionable]
    actionable = bool(pending or failing or actionable_cancelled)
    return {
        "total": len(checks),
        "pending": pending,
        "failing": failing,
        "cancelled": cancelled,
        "non_actionable_cancelled": non_actionable,
        "actionable_rows": [*pending, *failing, *actionable_cancelled],
        "actionable_non_green": actionable,
        "watched": _watched_rollup_summary(descriptors),
    }


def _watched_rollup_summary(
    descriptors: list[tuple[str, dict[str, str]]],
) -> dict[str, Any]:
    pending: list[dict[str, str]] = []
    failing: list[dict[str, str]] = []
    cancelled: list[dict[str, str]] = []
    counts_by_workflow: dict[str, dict[str, int]] = {}
    watched_workflows = list(READY_BOUNDARY_WATCHED_WORKFLOWS)

    for bucket, descriptor in descriptors:
        workflow = _watched_workflow_name(descriptor)
        if workflow is None or bucket not in {"pending", "fail", "cancel"}:
            continue
        counts = counts_by_workflow.setdefault(
            workflow,
            {"pending": 0, "fail": 0, "cancel": 0},
        )
        counts[bucket] += 1
        if bucket == "pending":
            pending.append(descriptor)
        elif bucket == "fail":
            failing.append(descriptor)
        elif bucket == "cancel":
            cancelled.append(descriptor)

    actionable_rows = [*pending, *failing, *cancelled]
    return {
        "workflows": watched_workflows,
        "pending": pending,
        "failing": failing,
        "cancelled": cancelled,
        "actionable_rows": actionable_rows,
        "counts_by_workflow": counts_by_workflow,
        "actionable_non_green": bool(actionable_rows),
        "summary": _format_watched_rollup_summary(counts_by_workflow),
    }


def _watched_workflow_name(descriptor: dict[str, str]) -> str | None:
    workflow = str(descriptor.get("workflow") or "").strip().lower()
    name = str(descriptor.get("name") or "").strip().lower()
    for watched in READY_BOUNDARY_WATCHED_WORKFLOWS:
        watched_lower = watched.lower()
        if workflow.startswith(watched_lower) or name.startswith(watched_lower):
            return watched
    return None


def _format_watched_rollup_summary(counts_by_workflow: dict[str, dict[str, int]]) -> str:
    if not counts_by_workflow:
        return "no watched gate rows pending or failing"
    parts: list[str] = []
    for workflow in READY_BOUNDARY_WATCHED_WORKFLOWS:
        counts = counts_by_workflow.get(workflow)
        if not counts:
            continue
        fragments = [
            f"{counts[key]} {label}"
            for key, label in (
                ("pending", "pending"),
                ("fail", "failed"),
                ("cancel", "cancelled"),
            )
            if counts.get(key)
        ]
        if fragments:
            parts.append(f"{workflow}: {', '.join(fragments)}")
    return "; ".join(parts) if parts else "no watched gate rows pending or failing"


def _is_non_actionable_cancelled_descriptor(
    descriptor: dict[str, str],
    successful_descriptors: list[dict[str, str]],
) -> bool:
    names = [
        str(descriptor.get("name") or ""),
        str(descriptor.get("workflow") or ""),
    ]
    if any(name.lower().startswith(NON_ACTIONABLE_CANCELLED_PREFIXES) for name in names if name):
        return True

    workflow = str(descriptor.get("workflow") or "").lower()
    name = str(descriptor.get("name") or "").lower()
    replacement_name = SUPERSEDED_CANCELLED_CHECKS.get(workflow, {}).get(name)
    if not replacement_name:
        return False
    return any(
        str(item.get("workflow") or "").lower() == workflow
        and str(item.get("name") or "").lower() == replacement_name
        for item in successful_descriptors
    )


def _merge_packet_summary(
    *,
    pr_number: int,
    repo_override: str | None,
    review_queue_root: str | Path | None,
    merge_packet: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    try:
        packet = merge_packet(
            pr_refs=[str(pr_number)],
            limit=1,
            repo_override=repo_override,
            review_queue_root=review_queue_root,
            execute_reviewers=False,
            ignore_own_quorum_check=False,
        )
    except Exception as exc:  # noqa: BLE001 - conductor should preserve on helper failures.
        transport_blocked = _is_github_transport_error(exc)
        return {
            "available": False,
            "error": str(exc),
            "error_kind": _gh_error_kind(exc),
            "transport_blocked": transport_blocked,
            "preserve_no_mutate": transport_blocked,
            "not_ready": [pr_number],
            "admin_squash_allowed": False,
        }

    entries = [item for item in packet.get("entries") or [] if isinstance(item, dict)]
    entry = entries[0] if entries else {}
    raw_quorum = entry.get("model_review_quorum")
    quorum: dict[str, Any] = raw_quorum if isinstance(raw_quorum, dict) else {}
    counted_model_families = _packet_value(entry, quorum, "counted_model_families", [])
    reasons = _packet_value(entry, quorum, "reasons", [])
    dogfood_evidence = entry.get("dogfood_evidence") if isinstance(entry, dict) else None
    focused_dogfood_present = bool(
        _packet_value(entry, quorum, "focused_dogfood_present", False)
        or (isinstance(dogfood_evidence, list) and dogfood_evidence)
    )
    return {
        "available": True,
        "error": "",
        "pr_number": entry.get("pr_number", pr_number),
        "head_sha": str(entry.get("head_sha") or ""),
        "tier": _packet_value(entry, quorum, "tier"),
        "verdict": _packet_value(entry, quorum, "verdict"),
        "checks_summary": _packet_value(entry, quorum, "checks_summary"),
        "counted_model_families": counted_model_families
        if isinstance(counted_model_families, list)
        else [],
        "reasons": reasons if isinstance(reasons, list) else [],
        "focused_dogfood_present": focused_dogfood_present,
        "human_preapproval_recorded": bool(
            _packet_value(entry, quorum, "human_preapproval_recorded", False)
        ),
        "requires_human_preapproval": bool(
            _packet_value(entry, quorum, "requires_human_preapproval", False)
        ),
        "requires_human_risk_settlement": bool(
            _packet_value(entry, quorum, "requires_human_risk_settlement", False)
        ),
        "admin_squash_allowed": bool(
            packet.get("admin_squash_allowed") or entry.get("admin_squash_allowed")
        ),
        "admin_squash_order": packet.get("admin_squash_order") or [],
        "not_ready": packet.get("not_ready") or [],
    }


def _ready_boundary_summary(
    *,
    view: dict[str, Any],
    required: dict[str, Any],
    owner: dict[str, Any],
    steering: dict[str, Any],
    merge_packet: dict[str, Any],
    rollup: dict[str, Any],
    head_changed: bool,
    repo_override: str | None,
) -> dict[str, Any]:
    pr_number = int(view.get("number") or 0)
    head_sha = str(view.get("headRefOid") or "")
    state = str(view.get("state") or "").upper()
    is_draft = bool(view.get("isDraft"))
    mergeable = str(view.get("mergeable") or "").upper()
    merge_state = str(view.get("mergeStateStatus") or "").upper()
    tier = _safe_int(merge_packet.get("tier"))
    families = [
        str(family)
        for family in merge_packet.get("counted_model_families") or []
        if str(family).strip()
    ]
    focused_dogfood_present = bool(merge_packet.get("focused_dogfood_present"))
    reasons = [str(reason) for reason in merge_packet.get("reasons") or []]
    reasons_text = " ".join(reason.lower() for reason in reasons)
    actionable_rows = list(rollup.get("actionable_rows") or [])
    watched_rollup = rollup.get("watched") if isinstance(rollup.get("watched"), dict) else {}
    blockers: list[str] = []
    post_ready_blockers: list[str] = []

    if state != "OPEN":
        blockers.append("PR is not open")
    if head_changed:
        blockers.append("head changed during conductor read")
    if not is_draft:
        blockers.append("PR is already non-draft")
    if owner.get("lookup_state") == "timeout" or owner.get("preserve_no_mutate"):
        blockers.append("owner lookup is not mutation-safe")
    if owner.get("active_owner"):
        blockers.append("active owner is present")
    if steering.get("lookup_state") == "timeout" or steering.get("preserve_no_mutate"):
        blockers.append("operator steering lookup is not mutation-safe")
    elif steering.get("has_pending") or int(steering.get("message_count") or 0) > 0:
        blockers.append("operator steering messages are pending")
    if required.get("has_failures") or required.get("has_pending"):
        blockers.append("required checks are not green")
    if required.get("transport_blocked") or required.get("preserve_no_mutate"):
        blockers.append("GitHub required-check transport is blocked")
    if merge_packet.get("transport_blocked") or merge_packet.get("preserve_no_mutate"):
        blockers.append("GitHub merge-packet transport is blocked")
    if rollup.get("actionable_non_green"):
        blockers.append("actionable non-required rollup rows remain")
    if mergeable != "MERGEABLE":
        blockers.append(f"mergeable is {mergeable or 'unknown'}")
    if merge_state != "CLEAN":
        blockers.append(f"mergeStateStatus is {merge_state or 'unknown'}")
    if "model quorum incomplete" in reasons_text:
        blockers.append("model quorum is incomplete")
    if "focused adversarial dogfood evidence is required" in reasons_text:
        blockers.append("focused dogfood evidence is missing")
    if tier is not None and tier >= 3:
        if len({family.lower() for family in families}) < 2:
            blockers.append("Tier 3/4 model quorum has fewer than two families")
        if not focused_dogfood_present:
            blockers.append("Tier 3/4 focused dogfood evidence is missing")
        if not bool(merge_packet.get("human_preapproval_recorded")):
            post_ready_blockers.append(f"Tier {tier} human preapproval/settlement")

    eligible = not blockers
    classification = (
        READY_BOUNDARY_MARK_READY_CLASSIFICATION if eligible else "ready_boundary_blocked"
    )
    authorization_prompt = (
        _build_ready_boundary_authorization_prompt(
            pr_number=pr_number,
            head_sha=head_sha,
            tier=tier,
            repo_override=repo_override,
        )
        if eligible
        else ""
    )
    return {
        "classification": classification,
        "eligible_for_mark_ready_authorization": eligible,
        "blockers": blockers,
        "post_ready_blockers": post_ready_blockers,
        "required_checks_summary": required.get("summary"),
        "actionable_rollup_rows": actionable_rows,
        "watched_rollup": watched_rollup,
        "evidence_status": {
            "tier": tier,
            "counted_model_families": families,
            "focused_dogfood_present": focused_dogfood_present,
            "human_preapproval_recorded": bool(merge_packet.get("human_preapproval_recorded")),
            "reasons": reasons,
        },
        "owner_state": owner.get("state") or owner.get("lookup_state") or "",
        "owner_active": bool(owner.get("active_owner")),
        "steering_lookup_state": steering.get("lookup_state") or "",
        "steering_message_count": int(steering.get("message_count") or 0),
        "mark_ready_command": (
            f"gh pr ready {pr_number}" + (f" --repo {repo_override}" if repo_override else "")
            if eligible
            else ""
        ),
        "authorization_prompt": authorization_prompt,
    }


def _classify_candidate(
    *,
    view: dict[str, Any],
    required: dict[str, Any],
    owner: dict[str, Any],
    merge_packet: dict[str, Any],
    rollup: dict[str, Any],
    head_changed: bool,
    supersession_hints: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    pr_number = int(view.get("number") or 0)
    state = str(view.get("state") or "").upper()
    is_draft = bool(view.get("isDraft"))
    mergeable = str(view.get("mergeable") or "").upper()
    merge_state = str(view.get("mergeStateStatus") or "").upper()
    not_ready = set(
        int(item) for item in merge_packet.get("not_ready") or [] if str(item).isdigit()
    )
    tier = _safe_int(merge_packet.get("tier"))
    verdict = str(merge_packet.get("verdict") or "")

    if state == "MERGED":
        return ("already_merged", False, "post-merge verification or queue cleanup only")
    if state and state != "OPEN":
        return ("closed_or_not_open", False, "do not mutate a non-open PR")
    if head_changed:
        return ("head_changed_preserve", False, "re-read PR state before any mutation")
    if owner.get("lookup_state") == "timeout" or owner.get("preserve_no_mutate"):
        if owner.get("lookup_state") == "timeout":
            return (
                OWNER_TIMEOUT_CLASSIFICATION,
                False,
                "owner lookup timed out; preserve and do not mutate",
            )
        return (
            "owner_lookup_failed_preserve",
            False,
            "owner lookup failed; preserve and do not mutate",
        )
    if owner.get("active_owner"):
        return (
            "active_owned",
            False,
            f"route to active owner {owner.get('owner_session') or owner.get('owner') or ''}".strip(),
        )
    if required.get("transport_blocked") or merge_packet.get("transport_blocked"):
        return (
            "transport_blocked_preserve",
            False,
            "GitHub transport blocked; preserve and do not mutate",
        )
    if required.get("has_failures") or required.get("has_pending"):
        return ("blocked_required_checks", False, "wait for or repair required checks")
    if rollup.get("actionable_non_green"):
        return ("blocked_actionable_rollup", False, "inspect actionable non-required rollup rows")
    if mergeable in {"CONFLICTING", "UNKNOWN"} or merge_state in {"DIRTY"}:
        if supersession_hints:
            return ("superseded_or_stale", False, "analyze supersession before conflict repair")
        return ("unowned_repairable", True, "repair or restack one narrow unowned conflict")
    if is_draft and tier is not None and tier <= 2 and pr_number in not_ready:
        return (
            "unowned_evidence_candidate",
            True,
            "collect exact-head Tier 0-2 evidence, then keep draft/merge boundaries",
        )
    if tier is not None and tier >= 3 and pr_number in not_ready:
        if _missing_tier3_or_tier4_evidence(merge_packet):
            return (
                TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION,
                False,
                f"collect exact-head Tier {tier} model/dogfood evidence before settlement",
            )
        if "human" in verdict or "preapproval" in verdict or "settlement" in verdict:
            return (
                "ready_but_human_gated",
                False,
                "request exact-head Tier 3/4 human settlement authorization",
            )
        return (
            "blocked_tier3_or_tier4",
            False,
            "resolve Tier 3/4 packet blockers without mutation",
        )
    if merge_packet.get("admin_squash_allowed") and not merge_packet.get("not_ready"):
        return (
            "ready_for_final_premerge_verification",
            False,
            "request exact-head final pre-merge authorization",
        )
    if is_draft:
        return (
            "draft_blocked",
            False,
            "verify draft blocker and request mark-ready authorization if appropriate",
        )
    return ("blocked_or_needs_handoff", False, "produce bounded handoff with exact blocker")


def _build_next_prompt(
    candidates: list[dict[str, Any]],
    *,
    repo_override: str | None,
    mode: str = QUEUE_CONDUCTOR_DEFAULT_MODE,
) -> str:
    if mode == QUEUE_CONDUCTOR_READY_BOUNDARY_MODE:
        return _build_ready_boundary_next_prompt(candidates)

    preferred = _first_by_classification(
        candidates,
        [
            "unowned_evidence_candidate",
            TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION,
            "unowned_repairable",
            "ready_but_human_gated",
            "ready_for_final_premerge_verification",
        ],
    )
    if preferred is None:
        return (
            "Do not rely on transcript state. Re-check live GitHub/local state first. "
            "Run review-queue conductor again from a clean current-main checkout and report the "
            "highest-signal owner, head-change, or supersession blocker. If the prompt above "
            "accomplishes no incremental progress, make the next prompt one that does. If any "
            "work can be better automated by improving Aragora tooling at a meta level, include "
            "the concrete tooling improvement plan instead of repeating manual queue checks. "
            "Always include a final summary section with the best next recursive prompt."
        )

    pr_number = preferred["pr_number"]
    head = preferred.get("head_sha") or "UNKNOWN_HEAD"
    repo_text = f" --repo {repo_override}" if repo_override else ""
    classification = preferred.get("classification")
    if classification == "unowned_evidence_candidate":
        primary = (
            f"Primary task: collect fresh exact-head structured model/dogfood evidence for PR "
            f"#{pr_number} only, at current exact head {head}, without marking ready or merging."
        )
    elif classification == TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION:
        tier = _safe_int((preferred.get("merge_packet") or {}).get("tier"))
        tier_text = f"Tier {tier} " if tier is not None else ""
        primary = (
            f"Primary task: collect fresh exact-head {tier_text}structured model/dogfood "
            f"evidence for PR #{pr_number} only, at current exact head {head}, without "
            "marking ready, recording settlement, or merging."
        )
    elif classification == "unowned_repairable":
        primary = (
            f"Primary task: repair or restack PR #{pr_number} only at exact head {head} from a "
            "clean branch worktree after re-checking owner lookup and supersession hints."
        )
    elif classification == "ready_but_human_gated":
        primary = (
            f"Primary task: request/record exact-head Tier 3/4 human settlement for PR "
            f"#{pr_number} only at exact head {head}, without merging."
        )
    else:
        primary = (
            f"Primary task: final pre-merge verification for PR #{pr_number} only at exact head "
            f"{head}, with merge only if separately authorized."
        )
    return (
        "Do not rely on transcript state. Re-check live GitHub/local state first. "
        f"Current environment date is {_current_environment_date()}.\n\n"
        f"{primary}\n\n"
        f"Reconfirm PR #{pr_number} state/head; run gh pr checks {pr_number} --required, "
        f"gh pr view {pr_number} --json statusCheckRollup,mergeStateStatus,headRefOid,"
        "isDraft,state,mergeable, branch owner lookup, operator steering read-only, and "
        f"provider-keys-unset python3 -m aragora.cli.main review-queue merge-packet --limit 1 "
        f"--pr {pr_number}{repo_text} --json from a clean current-main checkout. Respect any "
        "active owner or owner-lookup timeout as preserve/no-mutate. If the prompt above "
        "accomplishes no incremental progress, make the next prompt one that does. If any "
        "work can be better automated by improving Aragora tooling at a meta level, include "
        "the concrete tooling improvement plan instead of repeating manual queue checks. "
        "Always include a final summary section with the best next recursive prompt."
    )


def _build_ready_boundary_next_prompt(candidates: list[dict[str, Any]]) -> str:
    for candidate in candidates:
        ready_boundary = candidate.get("ready_boundary")
        if (
            isinstance(ready_boundary, dict)
            and ready_boundary.get("eligible_for_mark_ready_authorization")
            and ready_boundary.get("authorization_prompt")
        ):
            return str(ready_boundary["authorization_prompt"])
    return (
        "Do not rely on transcript state. Re-check live GitHub/local state first. "
        f"Current environment date is {_current_environment_date()}.\n\n"
        "Primary task: re-run review-queue conductor in ready-boundary mode from a clean "
        "current-main checkout and report the exact owner, steering, required-check, "
        "actionable-rollup, and merge-packet blocker before any mark-ready mutation.\n\n"
        "If the prompt above accomplishes no incremental progress, make the next prompt one "
        "that does. If any work can be better automated by improving Aragora tooling at a meta "
        "level, include the concrete tooling improvement plan instead of repeating manual queue "
        "checks. Always include a final summary section with the best next recursive prompt."
    )


def _build_ready_boundary_authorization_prompt(
    *,
    pr_number: int,
    head_sha: str,
    tier: int | None,
    repo_override: str | None,
) -> str:
    tier_text = f"Tier {tier} " if tier is not None and tier >= 3 else ""
    repo_arg = f" --repo {repo_override}" if repo_override else ""
    settlement_boundary = (
        f" Do not record {tier_text.strip()} settlement or merge."
        if tier_text
        else " Do not merge."
    )
    return (
        "Do not rely on transcript state. Re-check live GitHub/local state first. "
        f"Current environment date is {_current_environment_date()}.\n\n"
        f"Primary task: mark PR #{pr_number} ready for review only if all live gates still "
        "match this authorization.\n\n"
        f"I explicitly authorize marking PR #{pr_number} ready for review at exact head "
        f"{head_sha}.{settlement_boundary}\n\n"
        f"Reconfirm #{pr_number} is open draft at exact head {head_sha}; run gh pr checks "
        f"{pr_number} --required, gh pr view {pr_number} --json "
        "statusCheckRollup,mergeStateStatus,headRefOid,isDraft,state,mergeable, branch "
        "owner lookup, operator steering read-only, and provider-keys-unset python3 -m "
        "aragora.cli.main review-queue merge-packet --limit 1 "
        f"--pr {pr_number}{repo_arg} --json from a clean current-main checkout. Confirm "
        "required checks are green, no actionable non-required rollup item remains, "
        "evidence/quorum remains satisfied, focused dogfood is present, mergeStateStatus "
        "is CLEAN, and the only actionable blockers are draft state plus later human "
        "preapproval/settlement if applicable. If all gates still match, mark the PR ready "
        "for review. Do not merge or record settlement. Re-read PR view and merge-packet. "
        "Report exact head, whether mark-ready was performed, post-ready required-check "
        "state, merge-packet summary, and safest next bounded action.\n\n"
        "If the prompt above accomplishes no incremental progress, make the next prompt one "
        "that does. If any work can be better automated by improving Aragora tooling at a meta "
        "level, include the concrete tooling improvement plan instead of repeating manual queue "
        "checks. Always include a final summary section with the best next recursive prompt."
    )


def _current_environment_date() -> str:
    return datetime.now(UTC).date().isoformat()


def _first_by_classification(
    candidates: list[dict[str, Any]], classes: list[str]
) -> dict[str, Any] | None:
    for wanted in classes:
        for candidate in candidates:
            if candidate.get("classification") == wanted:
                return candidate
    return None


def _missing_tier3_or_tier4_evidence(merge_packet: dict[str, Any]) -> bool:
    families = {
        str(family).strip().lower()
        for family in merge_packet.get("counted_model_families") or []
        if str(family).strip()
    }
    reasons = " ".join(str(reason).lower() for reason in merge_packet.get("reasons") or [])
    model_quorum_missing = len(families) < 2 or "model quorum incomplete" in reasons
    dogfood_missing = (
        not bool(merge_packet.get("focused_dogfood_present"))
        or "focused adversarial dogfood evidence is required" in reasons
    )
    return model_quorum_missing or dogfood_missing


def _candidate_file_index(pr_views: list[dict[str, Any]]) -> dict[int, set[str]]:
    index: dict[int, set[str]] = {}
    for view in pr_views:
        number = int(view.get("number") or 0)
        files = view.get("files") or []
        paths: set[str] = set()
        for item in files:
            if isinstance(item, dict):
                path = str(item.get("path") or "")
                if path:
                    paths.add(path)
            elif isinstance(item, str):
                paths.add(item)
        index[number] = paths
    return index


def _supersession_hints(
    view: dict[str, Any],
    all_prs: list[dict[str, Any]],
    file_index: dict[int, set[str]],
) -> list[dict[str, Any]]:
    number = int(view.get("number") or 0)
    title_tokens = _title_tokens(str(view.get("title") or ""))
    current_files = file_index.get(number, set())
    hints: list[dict[str, Any]] = []
    for other in all_prs:
        other_number = int(other.get("number") or 0)
        if other_number == number:
            continue
        overlap_files = sorted(current_files & file_index.get(other_number, set()))
        token_overlap = sorted(title_tokens & _title_tokens(str(other.get("title") or "")))
        if not overlap_files and len(token_overlap) < 3:
            continue
        hints.append(
            {
                "pr_number": other_number,
                "head_sha": str(other.get("headRefOid") or ""),
                "title": str(other.get("title") or ""),
                "overlap_files": overlap_files,
                "title_token_overlap": token_overlap[:8],
                "newer_pr": other_number > number,
            }
        )
    return hints


def _title_tokens(title: str) -> set[str]:
    stop = {"pr", "fix", "repair", "update", "add", "for", "and", "the", "a", "an"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if len(token) > 2 and token not in stop
    }


def _default_owner_lookup(branch: str, timeout_seconds: float) -> dict[str, Any]:
    result = _run_json_helper(
        "identify_lane_owner.py",
        ["--branch", branch, "--json"],
        timeout_seconds=timeout_seconds,
    )
    if result.get("lookup_state") == "timeout":
        result["preserve_no_mutate"] = True
        result["active_owner"] = False
        return result
    if (
        result.get("lookup_state") == "failed"
        and "no lane matched" in str(result.get("error", "")).lower()
    ):
        return {
            "lookup_state": "no_lane_match",
            "state": "unowned",
            "active_owner": False,
            "preserve_no_mutate": False,
            "owner": "",
            "owner_session": "",
            "lane_id": "",
            "raw": result.get("payload"),
            "error": result.get("error", ""),
        }
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return {
            **result,
            "state": "lookup_failed",
            "active_owner": False,
            "preserve_no_mutate": True,
        }
    owner_state = str(payload.get("state") or payload.get("status") or "").lower()
    active_owner = bool(payload.get("active_owner")) or owner_state in ACTIVE_OWNER_STATUSES
    return {
        "lookup_state": result.get("lookup_state", "ok"),
        "state": owner_state or ("owned" if active_owner else "unowned"),
        "active_owner": active_owner,
        "preserve_no_mutate": bool(result.get("preserve_no_mutate")) or bool(result.get("error")),
        "owner": payload.get("owner") or payload.get("lane_owner") or "",
        "owner_session": payload.get("owner_session") or payload.get("session_id") or "",
        "lane_id": payload.get("lane_id") or payload.get("lane") or "",
        "raw": payload,
        "error": result.get("error", ""),
    }


def _default_steering_lookup(branch: str, timeout_seconds: float) -> dict[str, Any]:
    result = _run_json_helper(
        "read_operator_steering.py",
        ["--branch", branch, "--json", "--no-receipt"],
        timeout_seconds=timeout_seconds,
    )
    if result.get("lookup_state") == "timeout":
        return {
            "lookup_state": "timeout",
            "preserve_no_mutate": True,
            "error": result.get("error", ""),
        }
    if (
        result.get("lookup_state") == "failed"
        and "no lane matched" in str(result.get("error", "")).lower()
    ):
        return {
            "lookup_state": "no_lane_match",
            "message_count": 0,
            "has_pending": False,
            "raw": result.get("payload"),
            "error": result.get("error", ""),
        }
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return {
            "lookup_state": result.get("lookup_state", "failed"),
            "error": result.get("error", ""),
        }
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        messages = raw_messages
    else:
        raw_items = payload.get("items")
        messages = raw_items if isinstance(raw_items, list) else []
    return {
        "lookup_state": result.get("lookup_state", "ok"),
        "message_count": len(messages),
        "has_pending": bool(messages),
        "raw": payload,
        "error": result.get("error", ""),
    }


def _run_json_helper(
    script_name: str, args: list[str], *, timeout_seconds: float
) -> dict[str, Any]:
    repo_root = resolve_repo_root(Path.cwd())
    script_path = repo_root / "scripts" / script_name
    if not script_path.exists():
        return {"lookup_state": "missing_helper", "error": f"{script_path} does not exist"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(timeout_seconds, 0.1),
        )
    except subprocess.TimeoutExpired:
        return {
            "lookup_state": "timeout",
            "error": f"{script_name} timed out after {timeout_seconds}s",
        }
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    payload: Any = None
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return {
                "lookup_state": "malformed_json",
                "error": f"{script_name} returned malformed JSON",
                "stdout": stdout[:1000],
                "stderr": stderr[:1000],
            }
    if proc.returncode != 0:
        return {
            "lookup_state": "failed",
            "error": stderr or stdout or f"{script_name} exited {proc.returncode}",
            "payload": payload,
        }
    return {"lookup_state": "ok", "payload": payload or {}, "error": ""}


def _safe_origin_main_sha(providers: ConductorProviders) -> str:
    if providers.origin_main_sha is not None:
        try:
            return providers.origin_main_sha()
        except Exception:  # noqa: BLE001
            return ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _packet_value(
    entry: dict[str, Any],
    quorum: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in quorum:
        return quorum.get(key)
    return entry.get(key, default)


def _rollup_check_bucket(item: dict[str, Any]) -> str:
    bucket = str(item.get("bucket") or "").strip().lower()
    if bucket:
        return _required_pr_check_bucket(item)

    status = str(item.get("status") or "").strip().upper()
    conclusion = str(item.get("conclusion") or "").strip().upper()
    if status and status != "COMPLETED":
        return "pending"
    if conclusion in {"SUCCESS"}:
        return "pass"
    if conclusion in {"SKIPPED", "NEUTRAL"}:
        return "skipping"
    if conclusion in {"CANCELLED", "CANCELED"}:
        return "cancel"
    if conclusion in {"FAILURE", "FAILED", "ERROR", "TIMED_OUT", "ACTION_REQUIRED"}:
        return "fail"
    return _required_pr_check_bucket(item)


def _check_descriptor(item: dict[str, Any], *, bucket: str | None = None) -> dict[str, str]:
    return {
        "name": str(item.get("name") or item.get("context") or item.get("workflow") or ""),
        "state": str(item.get("state") or item.get("status") or item.get("conclusion") or ""),
        "bucket": bucket or _required_pr_check_bucket(item),
        "workflow": str(item.get("workflow") or item.get("workflowName") or ""),
        "link": str(item.get("link") or item.get("detailsUrl") or ""),
    }


def _missing_branch_owner() -> dict[str, Any]:
    return {
        "lookup_state": "missing_branch",
        "state": "unknown",
        "active_owner": False,
        "preserve_no_mutate": True,
        "error": "PR head branch missing",
    }


def _missing_branch_steering() -> dict[str, Any]:
    return {"lookup_state": "missing_branch", "error": "PR head branch missing"}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
