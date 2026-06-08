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
    _gh_json,
    _required_pr_check_bucket,
    _summarize_required_pr_checks,
)
from aragora.worktree.fleet import resolve_repo_root


QUEUE_CONDUCTOR_VERSION = "queue_conductor.v1"
OWNER_TIMEOUT_CLASSIFICATION = "owner_lookup_timeout_preserve"
TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION = "tier3_or_tier4_evidence_required"
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
    "self-hosted shadow",
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
    providers: ConductorProviders | None = None,
) -> dict[str, Any]:
    """Build a read-only owner-aware queue conductor packet."""

    active_providers = providers or ConductorProviders()
    pr_views = _fetch_pr_views(
        pr_refs=pr_refs or [],
        limit=limit,
        repo_override=repo_override,
        gh_json=active_providers.gh_json,
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

    next_prompt = _build_next_prompt(candidates, repo_override=repo_override)
    return {
        "version": QUEUE_CONDUCTOR_VERSION,
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
    lines.extend(["", "Best next prompt:", str(packet.get("next_prompt") or "")])
    return "\n".join(lines)


def _fetch_pr_views(
    *,
    pr_refs: list[str],
    limit: int,
    repo_override: str | None,
    gh_json: Callable[[list[str]], Any],
) -> list[dict[str, Any]]:
    if pr_refs:
        refs = list(dict.fromkeys(str(ref).strip() for ref in pr_refs if str(ref).strip()))
        return [
            _fetch_pr_view(str(ref), repo_override=repo_override, gh_json=gh_json) for ref in refs
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
        _fetch_pr_view(ref, repo_override=repo_override, gh_json=gh_json)
        for ref in refs[: max(limit, 0)]
    ]


def _fetch_pr_view(
    pr_ref: str,
    *,
    repo_override: str | None,
    gh_json: Callable[[list[str]], Any],
) -> dict[str, Any]:
    args = ["pr", "view", str(pr_ref), "--json", PR_VIEW_FIELDS]
    if repo_override:
        args.extend(["--repo", repo_override])
    payload = gh_json(args)
    if not isinstance(payload, dict):
        raise _GhError(f"gh pr view {pr_ref} returned a non-object payload")
    return payload


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
        "required_checks": required,
        "rollup": rollup,
        "owner": owner,
        "steering": steering,
        "merge_packet": merge_packet,
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
    }


def _rollup_summary(items: list[Any]) -> dict[str, Any]:
    checks = [item for item in items if isinstance(item, dict)]
    pending: list[dict[str, str]] = []
    failing: list[dict[str, str]] = []
    cancelled: list[dict[str, str]] = []
    non_actionable: list[dict[str, str]] = []
    for item in checks:
        bucket = _rollup_check_bucket(item)
        descriptor = _check_descriptor(item, bucket=bucket)
        name = descriptor.get("name", "").lower()
        if bucket == "pending":
            pending.append(descriptor)
        elif bucket == "fail":
            failing.append(descriptor)
        elif bucket == "cancel":
            cancelled.append(descriptor)
            if name.startswith(NON_ACTIONABLE_CANCELLED_PREFIXES):
                non_actionable.append(descriptor)

    actionable_cancelled = [item for item in cancelled if item not in non_actionable]
    actionable = bool(pending or failing or actionable_cancelled)
    return {
        "total": len(checks),
        "pending": pending,
        "failing": failing,
        "cancelled": cancelled,
        "non_actionable_cancelled": non_actionable,
        "actionable_non_green": actionable,
    }


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
        return {
            "available": False,
            "error": str(exc),
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
        "admin_squash_allowed": bool(
            packet.get("admin_squash_allowed") or entry.get("admin_squash_allowed")
        ),
        "admin_squash_order": packet.get("admin_squash_order") or [],
        "not_ready": packet.get("not_ready") or [],
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


def _build_next_prompt(candidates: list[dict[str, Any]], *, repo_override: str | None) -> str:
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
        "Current environment date is 2026-06-06.\n\n"
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
        "workflow": str(item.get("workflow") or ""),
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
