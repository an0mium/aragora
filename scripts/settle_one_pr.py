#!/usr/bin/env python3
"""Dry-run steward for one exact-head PR settlement attempt.

The script is intentionally read-only. It does not approve, comment, mark
ready, rerun workflows, or merge. It gathers the repeated settlement gates into
one report so a follow-up executor can make bounded progress without broad queue
drain. Live mode fails closed when `gh` cannot read branch-protection required
check source metadata; app-pinned required checks must be satisfied by the
expected GitHub App, not by manual status spoofing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, overload
from urllib.parse import quote

CONVERGENCE_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one "
    "that does, include this sentence in all subsequent prompts to ensure they converge "
    "towards prompts that make incremental progress."
)

VERSION = "settle_one_steward.v1"
MERGE_QUORUM = "aragora-merge-quorum"
HUMAN_RISK_EXCLUDES = {7407, 7425, 7438, 7439, 7443}
BROAD_PACKET_NEAR_SELECTED_LOOKAHEAD = 8
OPEN_PR_LIGHT_FIELDS = (
    "number,title,url,headRefName,headRefOid,isDraft,mergeable,mergeStateStatus,"
    "reviewDecision,labels,author,additions,deletions,changedFiles"
)
PR_POLICY_FIELDS = "number,title,headRefName,author,mergeable,mergeStateStatus,files"
SURFACE_EXCLUDE_REASON = (
    "security/auth/RBAC/secrets/deploy/workflow/legal/compliance/destructive/"
    "migration/public-API surface"
)


def _repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return Path.cwd()
    return Path(proc.stdout.strip())


def _state_repo_root(cwd: Path) -> Path:
    env_root = os.environ.get("ARAGORA_STATE_ROOT")
    if env_root:
        return Path(env_root)
    if (cwd / ".aragora").exists():
        return cwd
    canonical = Path.home() / "Development" / "aragora"
    if (canonical / ".aragora").exists():
        return canonical
    return cwd


def _run(args: list[str], *, cwd: Path, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": " ".join(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _run_json(
    args: list[str], *, cwd: Path, timeout: int = 120
) -> tuple[Any | None, dict[str, Any]]:
    result = _run(args, cwd=cwd, timeout=timeout)
    if result["returncode"] != 0 or not result["stdout"]:
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError as exc:
        result["json_error"] = str(exc)
        return None, result


def _with_repo(args: list[str], repo: str | None) -> list[str]:
    if repo:
        return [*args, "--repo", repo]
    return args


def _entry_pr(entry: dict[str, Any]) -> int | None:
    return _coerce_int(entry.get("pr_number"))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _is_green_summary(summary: str) -> bool:
    lower = str(summary).lower()
    return "green" in lower and "failing" not in lower and "pending" not in lower


def _entry_by_pr(packet: dict[str, Any], pr_number: int) -> dict[str, Any] | None:
    for entry in packet.get("entries") or []:
        if isinstance(entry, dict) and _entry_pr(entry) == pr_number:
            return entry
    return None


def _metadata_for_entry(
    entry: dict[str, Any], policy_metadata: dict[int, dict[str, Any]] | None
) -> dict[str, Any]:
    pr_number = _entry_pr(entry)
    if pr_number is None or not policy_metadata:
        return {}
    metadata = policy_metadata.get(pr_number)
    return metadata if isinstance(metadata, dict) else {}


def _title_branch_text(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in ("title", "headRefName", "head_ref_name", "branch"):
        value = metadata.get(key, entry.get(key))
        if value:
            fields.append(str(value))
    return " ".join(fields)


def _branch_values(entry: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("headRefName", "head_ref_name", "branch"):
        value = metadata.get(key, entry.get(key))
        if value:
            values.append(str(value).strip())
    return values


def _file_paths(entry: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for file_item in metadata.get("files") or entry.get("files") or []:
        if isinstance(file_item, dict):
            value = file_item.get("path")
        else:
            value = file_item
        if value:
            paths.append(str(value).strip())
    return paths


def _path_segments(path: str) -> list[str]:
    return [segment for segment in path.replace("\\", "/").strip().lower().split("/") if segment]


def _component_stem(segment: str) -> str:
    # Keep separator-sensitive matching explicit so "authoring" is not treated as "auth".
    return segment.rsplit(".", 1)[0]


def _has_prefixed_component(component: str, stem: str) -> bool:
    return component == stem or component.startswith(f"{stem}_") or component.startswith(f"{stem}-")


def _is_docs_or_tests_path(segments: list[str]) -> bool:
    return bool(segments) and segments[0] in {"doc", "docs", "test", "tests"}


def _is_dependabot_pr(entry: dict[str, Any], metadata: dict[str, Any]) -> bool:
    author = metadata.get("author") or entry.get("author")
    if isinstance(author, dict):
        author_login = str(author.get("login") or "")
    else:
        author_login = str(author or "")
    if author_login.lower().startswith("dependabot"):
        return True
    return any(
        branch.lower().startswith("dependabot/") for branch in _branch_values(entry, metadata)
    )


def _touches_unsafe_surface(entry: dict[str, Any], metadata: dict[str, Any]) -> bool:
    for path in _file_paths(entry, metadata):
        segments = _path_segments(path)
        if _is_docs_or_tests_path(segments):
            continue
        if len(segments) >= 2 and segments[0] == ".github" and segments[1] == "workflows":
            return True
        for index, segment in enumerate(segments):
            component = _component_stem(segment)
            if segment in {"public-api", "public_api"} or (
                segment == "public"
                and index + 1 < len(segments)
                and segments[index + 1].startswith("api")
            ):
                return True
            if component in {"authentication", "authorization", "oauth", "rbac"}:
                return True
            if _has_prefixed_component(component, "auth"):
                return True
            if _has_prefixed_component(component, "security"):
                return True
            if _has_prefixed_component(component, "secret") or _has_prefixed_component(
                component, "secrets"
            ):
                return True
            if _has_prefixed_component(component, "deploy") or component in {
                "deployment",
                "deployments",
            }:
                return True
            if (
                _has_prefixed_component(component, "migrate")
                or _has_prefixed_component(component, "migration")
                or component == "migrations"
            ):
                return True
            if _has_prefixed_component(component, "legal"):
                return True
            if _has_prefixed_component(component, "compliance"):
                return True
            if _has_prefixed_component(component, "destructive"):
                return True
    return False


def policy_exclusion_reasons(
    entry: dict[str, Any],
    *,
    exclude_prs: set[int] | None = None,
    active_owned_prs: set[int] | None = None,
    policy_metadata: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    """Return repo/operator policy reasons that make an entry report-only."""
    pr_number = _entry_pr(entry)
    exclude = set(exclude_prs or set())
    active_owned = set(active_owned_prs or set())
    reasons: list[str] = []
    if pr_number is not None and pr_number in exclude:
        reasons.append("explicitly excluded by steward scope")
    if pr_number is not None and pr_number in active_owned:
        reasons.append("active-owned lane")

    metadata = _metadata_for_entry(entry, policy_metadata)
    title_branch_text = _title_branch_text(entry, metadata)
    if _is_dependabot_pr(entry, metadata):
        reasons.append("Dependabot PR")

    mergeable = str(metadata.get("mergeable") or entry.get("mergeable") or "").upper()
    merge_state = str(
        metadata.get("mergeStateStatus") or entry.get("mergeStateStatus") or ""
    ).upper()
    if mergeable == "CONFLICTING" or merge_state == "DIRTY":
        reasons.append("dirty/conflicting PR")

    if re.search(r"(^|[^a-z0-9])adc([^a-z0-9]|$)", title_branch_text, re.IGNORECASE):
        reasons.append("ADC PR")
    if _touches_unsafe_surface(entry, metadata):
        reasons.append(SURFACE_EXCLUDE_REASON)

    tier = _coerce_int(entry.get("tier"))
    if tier is not None and tier > 2:
        reasons.append(f"Tier {tier}")
    if bool(entry.get("requires_human_risk_settlement")):
        reasons.append("requires_human_risk_settlement=true")
    return list(dict.fromkeys(reasons))


def _exclusion_record(entry: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "pr_number": _entry_pr(entry),
        "title": entry.get("title"),
        "head_sha": entry.get("head_sha"),
        "reasons": reasons,
    }


def _merge_exclusions(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, tuple[str, ...]]] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("pr_number"),
                tuple(str(reason) for reason in item.get("reasons") or []),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


SelectionResult = tuple[dict[str, Any] | None, list[str]]
SelectionResultWithExclusions = tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]


@overload
def select_candidate(
    packet: dict[str, Any],
    *,
    explicit_pr: int | None = None,
    exclude_prs: set[int] | None = None,
    active_owned_prs: set[int] | None = None,
    policy_metadata: dict[int, dict[str, Any]] | None = None,
    return_exclusions: Literal[False] = False,
) -> SelectionResult: ...


@overload
def select_candidate(
    packet: dict[str, Any],
    *,
    explicit_pr: int | None = None,
    exclude_prs: set[int] | None = None,
    active_owned_prs: set[int] | None = None,
    policy_metadata: dict[int, dict[str, Any]] | None = None,
    return_exclusions: Literal[True],
) -> SelectionResultWithExclusions: ...


def select_candidate(
    packet: dict[str, Any],
    *,
    explicit_pr: int | None = None,
    exclude_prs: set[int] | None = None,
    active_owned_prs: set[int] | None = None,
    policy_metadata: dict[int, dict[str, Any]] | None = None,
    return_exclusions: bool = False,
) -> SelectionResult | SelectionResultWithExclusions:
    """Select one dry-run settlement candidate from a merge packet."""
    exclude = set(exclude_prs or set())
    exclusions: list[dict[str, Any]] = []
    if explicit_pr is not None:
        entry = _entry_by_pr(packet, explicit_pr)
        if entry is None:
            blockers = [f"merge-packet has no entry for PR #{explicit_pr}"]
            if return_exclusions:
                return None, blockers, exclusions
            return None, blockers
        policy_reasons = policy_exclusion_reasons(
            entry,
            exclude_prs=exclude,
            active_owned_prs=active_owned_prs,
            policy_metadata=policy_metadata,
        )
        if policy_reasons:
            exclusions.append(_exclusion_record(entry, policy_reasons))
        if return_exclusions:
            return entry, [], exclusions
        return entry, []

    entries = [entry for entry in packet.get("entries") or [] if isinstance(entry, dict)]
    admin_order: list[int] = []
    for raw_pr in packet.get("admin_squash_order") or []:
        pr_number = _coerce_int(raw_pr)
        if pr_number is not None:
            admin_order.append(pr_number)
    for ordered_pr in admin_order:
        entry = _entry_by_pr(packet, ordered_pr)
        if entry is None:
            continue
        policy_reasons = policy_exclusion_reasons(
            entry,
            exclude_prs=exclude,
            active_owned_prs=active_owned_prs,
            policy_metadata=policy_metadata,
        )
        if policy_reasons:
            exclusions.append(_exclusion_record(entry, policy_reasons))
            continue
        if return_exclusions:
            return entry, [], exclusions
        return entry, []

    evidence_candidates: list[dict[str, Any]] = []
    for entry in entries:
        entry_pr_number = _entry_pr(entry)
        if entry_pr_number is None:
            continue
        policy_reasons = policy_exclusion_reasons(
            entry,
            exclude_prs=exclude,
            active_owned_prs=active_owned_prs,
            policy_metadata=policy_metadata,
        )
        if policy_reasons:
            exclusions.append(_exclusion_record(entry, policy_reasons))
            continue
        if bool(entry.get("requires_human_risk_settlement")):
            continue
        if bool(entry.get("unresolved_dissent")):
            continue
        if str(entry.get("machine_recommendation", "")).strip() == "repair_first":
            continue
        tier = _coerce_int(entry.get("tier"))
        if tier is None:
            tier = 99
        if tier > 2:
            continue
        if not _is_green_summary(str(entry.get("checks_summary", ""))):
            continue
        reasons = " ".join(str(reason).lower() for reason in entry.get("reasons") or [])
        if "model quorum incomplete" in reasons or "dogfood" in reasons:
            evidence_candidates.append(entry)
    if evidence_candidates:
        if return_exclusions:
            return evidence_candidates[0], [], exclusions
        return evidence_candidates[0], []

    blockers = ["no Tier 0-2 non-human-risk green PR needs only settlement evidence"]
    if return_exclusions:
        return None, blockers, exclusions
    return None, blockers


def entry_blockers(entry: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    pr_number = _entry_pr(entry)
    tier = _coerce_int(entry.get("tier"))
    if tier is None:
        tier = 99
    if pr_number is not None and pr_number in HUMAN_RISK_EXCLUDES:
        blockers.append(f"PR #{pr_number} is excluded by this steward scope")
    if tier > 2:
        blockers.append(f"Tier {tier} requires report-only handling")
    if bool(entry.get("requires_human_risk_settlement")):
        blockers.append("requires_human_risk_settlement=true")
    if bool(entry.get("unresolved_dissent")):
        blockers.append("unresolved_dissent=true")
    summary = str(entry.get("checks_summary", ""))
    if "failing" in summary.lower():
        blockers.append(f"checks failing: {summary}")
    if "pending" in summary.lower():
        blockers.append(f"checks pending: {summary}")
    return blockers


def evidence_summary(entry: dict[str, Any]) -> dict[str, Any]:
    reasons = [str(reason) for reason in entry.get("reasons") or []]
    missing_model = [reason for reason in reasons if "model quorum incomplete" in reason.lower()]
    missing_dogfood = any("dogfood" in reason.lower() for reason in reasons)
    return {
        "counted_reviewer_ids": entry.get("counted_reviewer_ids") or [],
        "reviewer_signal_count": len(entry.get("reviewer_signals") or []),
        "dogfood_evidence_count": len(entry.get("dogfood_evidence") or []),
        "missing_model_quorum": missing_model,
        "missing_focused_dogfood": missing_dogfood,
    }


def owner_blockers(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    blockers: list[str] = []
    active_records = payload.get("active_owner_records")
    if isinstance(active_records, list) and active_records:
        blockers.append(f"active owner records present: {len(active_records)}")
    owner = payload.get("owner") or payload.get("active_owner") or payload.get("lane")
    if isinstance(owner, dict):
        status = str(owner.get("status", "") or "").lower()
        if status in {"active", "running", "claimed"}:
            blockers.append(
                f"active owner {owner.get('lane_id') or owner.get('owner_session') or 'unknown'}"
            )
    elif owner:
        blockers.append(f"active owner {owner}")
    return blockers


def head_blockers(entry: dict[str, Any], pr_view: Any) -> list[str]:
    if not isinstance(pr_view, dict):
        return ["gh pr view did not return JSON"]
    blockers: list[str] = []
    expected = str(entry.get("head_sha", "") or "")
    actual = str(pr_view.get("headRefOid", "") or "")
    if expected and actual and expected != actual:
        blockers.append(f"head drift: packet {expected} live {actual}")
    if bool(pr_view.get("isDraft")):
        blockers.append("PR is draft")
    mergeable = str(pr_view.get("mergeable", "") or "").upper()
    merge_state = str(pr_view.get("mergeStateStatus", "") or "").upper()
    if mergeable == "CONFLICTING" or merge_state == "DIRTY":
        blockers.append(f"PR is dirty/conflicting: mergeable={mergeable} state={merge_state}")
    return blockers


def _run_id_from_link(link: str) -> str:
    match = re.search(r"/actions/runs/(\d+)", str(link))
    return match.group(1) if match else ""


def required_check_report(checks: Any) -> dict[str, Any]:
    if not isinstance(checks, list):
        return {
            "status": "unknown",
            "blockers": ["required checks JSON unavailable"],
            "suggestions": [],
        }
    blockers: list[str] = []
    suggestions: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or check.get("context") or "")
        workflow = str(check.get("workflow") or check.get("workflowName") or "")
        state = str(
            check.get("state") or check.get("status") or check.get("conclusion") or ""
        ).upper()
        if state in {"SUCCESS", "SKIPPED", "NEUTRAL"}:
            continue
        if state == "CANCELLED" and (name == MERGE_QUORUM or workflow == "Aragora Merge Quorum"):
            run_id = _run_id_from_link(str(check.get("link", "") or check.get("detailsUrl", "")))
            blockers.append("aragora-merge-quorum is cancelled")
            if run_id:
                suggestions.append(f"gh run rerun {run_id} --failed")
            else:
                suggestions.append("rerun the cancelled aragora-merge-quorum workflow")
            continue
        blockers.append(f"{name or workflow or 'required check'} is {state or 'unknown'}")
    status = "pass" if not blockers else "blocked"
    return {"status": status, "blockers": blockers, "suggestions": suggestions}


def _rollup_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("context") or "")


def _rollup_success(item: dict[str, Any]) -> bool:
    conclusion = str(item.get("conclusion") or "").upper()
    if conclusion:
        return conclusion in {"SUCCESS", "SKIPPED", "NEUTRAL"}
    state = str(item.get("state") or item.get("status") or "").upper()
    return state in {"SUCCESS", "SKIPPED", "NEUTRAL"}


def _check_run_app_id(item: dict[str, Any]) -> int | None:
    """Extract the source GitHub App database id from common CheckRun shapes."""
    direct = _coerce_int(item.get("app_id") or item.get("appId") or item.get("appDatabaseId"))
    if direct is not None:
        return direct

    app = item.get("app")
    if isinstance(app, dict):
        app_id = _coerce_int(app.get("id") or app.get("databaseId"))
        if app_id is not None:
            return app_id

    check_suite = item.get("checkSuite")
    if isinstance(check_suite, dict):
        suite_app = check_suite.get("app")
        if isinstance(suite_app, dict):
            return _coerce_int(suite_app.get("id") or suite_app.get("databaseId"))

    return None


def _check_runs_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        check_runs = payload.get("check_runs") or payload.get("nodes") or []
    else:
        check_runs = payload
    if not isinstance(check_runs, list):
        return []
    return [item for item in check_runs if isinstance(item, dict)]


def required_check_source_report(
    protection: Any,
    pr_view: Any,
    check_runs_payload: Any = None,
) -> dict[str, Any]:
    """Fail closed when an app-pinned required check is only a manual status.

    GitHub branch protection may pin a required check context to a specific App.
    In that case a manually posted commit status with the same context can look
    green in `gh pr checks`, but GitHub will reject mergePullRequest because the
    expected App did not set the check. The steward should surface that before
    an executor tries to merge.
    """
    if not isinstance(protection, dict):
        return {
            "status": "unknown",
            "blockers": ["branch protection required_status_checks JSON unavailable"],
            "suggestions": [
                "ensure gh can read branch protection required_status_checks before settlement"
            ],
        }
    if not isinstance(pr_view, dict):
        return {
            "status": "unknown",
            "blockers": ["PR statusCheckRollup JSON unavailable"],
            "suggestions": ["rerun gh pr view with statusCheckRollup before settlement"],
        }

    rollup = pr_view.get("statusCheckRollup") or []
    if not isinstance(rollup, list):
        return {
            "status": "unknown",
            "blockers": ["PR statusCheckRollup is not a list"],
            "suggestions": ["rerun gh pr view with statusCheckRollup before settlement"],
        }

    check_runs = _check_runs_from_payload(check_runs_payload)
    blockers: list[str] = []
    suggestions: list[str] = []
    for required in protection.get("checks") or []:
        if not isinstance(required, dict):
            continue
        context = str(required.get("context") or "")
        app_id = required.get("app_id")
        if not context or app_id in (None, -1):
            continue
        expected_app_id = _coerce_int(app_id)
        if expected_app_id is None:
            blockers.append(f"{context} has unparseable pinned app_id {app_id!r}")
            suggestions.append("inspect branch protection required_status_checks.checks")
            continue

        matching = [
            item for item in rollup if isinstance(item, dict) and _rollup_name(item) == context
        ]
        matching_check_runs = [
            item
            for item in [*matching, *check_runs]
            if _rollup_name(item) == context
            and (item.get("__typename") == "CheckRun" or "app" in item or "checkSuite" in item)
        ]
        successful_check_runs = [item for item in matching_check_runs if _rollup_success(item)]
        successful_expected_app_runs = [
            item for item in successful_check_runs if _check_run_app_id(item) == expected_app_id
        ]
        if successful_expected_app_runs:
            continue

        has_successful_status = any(
            item.get("__typename") == "StatusContext" and _rollup_success(item) for item in matching
        )
        observed_app_ids = sorted(
            {
                observed
                for item in successful_check_runs
                if (observed := _check_run_app_id(item)) is not None
            }
        )
        has_unverified_successful_check_run = bool(successful_check_runs) and not observed_app_ids
        if has_successful_status:
            blockers.append(
                f"{context} is app-pinned to app_id {expected_app_id}, but only a manual "
                "StatusContext is green"
            )
            suggestions.append(
                f"rerun the app-sourced {context} check; do not satisfy it with a manual status"
            )
        elif observed_app_ids:
            blockers.append(
                f"{context} is app-pinned to app_id {expected_app_id}, but successful "
                f"CheckRun app_id(s) were {observed_app_ids}"
            )
            suggestions.append(f"rerun the app-sourced {context} check")
        elif has_unverified_successful_check_run:
            blockers.append(
                f"{context} is app-pinned to app_id {expected_app_id}, but CheckRun "
                "source app could not be verified"
            )
            suggestions.append("fetch commit check-runs with app metadata before settlement")
        else:
            blockers.append(
                f"{context} is app-pinned to app_id {expected_app_id}, but no successful CheckRun is present"
            )
            suggestions.append(f"rerun the app-sourced {context} check")

    status = "pass" if not blockers else "blocked"
    return {"status": status, "blockers": blockers, "suggestions": suggestions}


def validation_report(
    entry: dict[str, Any], *, cwd: Path, run_validation: bool
) -> list[dict[str, Any]]:
    head = str(entry.get("head_sha", "") or "")
    commands = [
        ["git", "diff", "--check", f"origin/main...{head}"],
        ["bash", "scripts/automation_pr_preflight.sh", "origin/main", head],
    ]
    reports: list[dict[str, Any]] = []
    for command in commands:
        if not run_validation:
            reports.append(
                {
                    "command": " ".join(command),
                    "status": "skipped",
                    "reason": "blocked before validation",
                }
            )
            continue
        result = _run(command, cwd=cwd, timeout=300)
        reports.append(
            {
                "command": result["command"],
                "status": "pass" if result["returncode"] == 0 else "blocked",
                "returncode": result["returncode"],
                "stderr": result["stderr"][-1000:],
            }
        )
    return reports


def load_open_pr_metadata(
    cwd: Path, *, limit: int = 200, repo: str | None = None
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload, command = _run_json(
        _with_repo(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                str(limit),
                "--json",
                OPEN_PR_LIGHT_FIELDS,
            ],
            repo,
        ),
        cwd=cwd,
    )
    metadata: dict[int, dict[str, Any]] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            pr_number = _coerce_int(item.get("number"))
            if pr_number is not None:
                metadata[pr_number] = item
    return metadata, command


def load_pr_policy_metadata(
    cwd: Path, pr_number: int, *, repo: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, command = _run_json(
        _with_repo(
            ["gh", "pr", "view", str(pr_number), "--json", PR_POLICY_FIELDS],
            repo,
        ),
        cwd=cwd,
    )
    if isinstance(payload, dict):
        return payload, command
    return {}, command


def load_active_owned_prs(cwd: Path) -> tuple[set[int], dict[str, Any]]:
    payload, command = _run_json(
        ["python3", "scripts/agent_bridge.py", "operator-snapshot", "--json"],
        cwd=cwd,
    )
    active_owned: set[int] = set()
    if isinstance(payload, dict):
        for lane in payload.get("lanes") or []:
            if not isinstance(lane, dict):
                continue
            if str(lane.get("status") or "").lower() != "active":
                continue
            pr_number = _coerce_int(lane.get("pr_number"))
            if pr_number is not None:
                active_owned.add(pr_number)
        command["operator_snapshot_ok"] = True
    else:
        command["operator_snapshot_ok"] = False
        if command.get("returncode") == 0 and not command.get("json_error"):
            command["json_error"] = "operator-snapshot did not return a JSON object"
    return active_owned, command


def active_owned_snapshot_blocker(command: dict[str, Any]) -> str | None:
    if command.get("operator_snapshot_ok"):
        return None
    detail = command.get("stderr") or command.get("json_error") or command.get("stdout") or ""
    if detail:
        return f"operator-snapshot unavailable; active-owned exclusions cannot be trusted: {detail}"
    return "operator-snapshot unavailable; active-owned exclusions cannot be trusted"


def _has_operator_snapshot_load_blocker(blockers: list[str]) -> bool:
    return any(
        blocker.startswith(
            "operator-snapshot unavailable; active-owned exclusions cannot be trusted"
        )
        for blocker in blockers
    )


def _dedupe_strings(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _normalize_repo_slug(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@github.com:"):
        text = text.split(":", 1)[1]
    elif "github.com/" in text:
        text = text.split("github.com/", 1)[1]
    text = text.strip("/")
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts[-2:]).lower()


def cwd_repo_slug(cwd: Path) -> tuple[str | None, dict[str, Any]]:
    command = _run(["git", "remote", "get-url", "origin"], cwd=cwd)
    if command.get("returncode") != 0:
        return None, command
    return _normalize_repo_slug(str(command.get("stdout") or "")), command


def repo_cwd_blocker(
    cwd: Path, repo: str | None
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    expected = _normalize_repo_slug(repo)
    if expected is None:
        return None, None, None
    actual, command = cwd_repo_slug(cwd)
    if actual is None:
        return "cwd origin repo unavailable while --repo is supplied", command, actual
    if actual != expected:
        return f"--repo {repo} does not match cwd origin {actual}", command, actual
    return None, command, actual


def select_candidate_with_lazy_policy_metadata(
    packet: dict[str, Any],
    *,
    cwd: Path,
    repo: str | None,
    explicit_pr: int | None,
    exclude_prs: set[int],
    active_owned_prs: set[int],
    policy_metadata: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a candidate while loading heavy file metadata only as needed."""
    metadata_commands: list[dict[str, Any]] = []
    loaded: set[int] = set()
    unavailable: set[int] = set()
    accumulated_exclusions: list[dict[str, Any]] = []
    max_attempts = len([entry for entry in packet.get("entries") or [] if isinstance(entry, dict)])
    max_attempts = max(1, max_attempts + 1)
    for _ in range(max_attempts):
        effective_exclude_prs = set(exclude_prs) | unavailable
        selected, blockers, exclusions = cast(
            SelectionResultWithExclusions,
            select_candidate(
                packet,
                explicit_pr=explicit_pr,
                exclude_prs=effective_exclude_prs,
                active_owned_prs=active_owned_prs,
                policy_metadata=policy_metadata,
                return_exclusions=True,
            ),
        )
        accumulated_exclusions = _merge_exclusions(accumulated_exclusions, exclusions)
        pr_number = _entry_pr(selected or {})
        if pr_number is None or pr_number in loaded:
            if selected is None and unavailable:
                return (
                    None,
                    ["selected-candidate policy metadata unavailable"],
                    accumulated_exclusions,
                    metadata_commands,
                )
            return selected, blockers, accumulated_exclusions, metadata_commands
        metadata, command = load_pr_policy_metadata(cwd, pr_number, repo=repo)
        metadata_commands.append(command)
        loaded.add(pr_number)
        if command.get("returncode") != 0 or not metadata:
            reason = "selected-candidate policy metadata unavailable"
            failed_exclusion = _exclusion_record(selected or {}, [reason])
            accumulated_exclusions = _merge_exclusions(accumulated_exclusions, [failed_exclusion])
            if explicit_pr is not None:
                return None, [reason], accumulated_exclusions, metadata_commands
            if pr_number is not None:
                unavailable.add(pr_number)
            continue
        merged = dict(policy_metadata.get(pr_number) or {})
        merged.update(metadata)
        policy_metadata[pr_number] = merged
    selected, blockers, exclusions = cast(
        SelectionResultWithExclusions,
        select_candidate(
            packet,
            explicit_pr=explicit_pr,
            exclude_prs=set(exclude_prs) | unavailable,
            active_owned_prs=active_owned_prs,
            policy_metadata=policy_metadata,
            return_exclusions=True,
        ),
    )
    if selected is None and unavailable:
        return (
            None,
            ["selected-candidate policy metadata unavailable"],
            _merge_exclusions(accumulated_exclusions, exclusions),
            metadata_commands,
        )
    return (
        selected,
        blockers,
        _merge_exclusions(accumulated_exclusions, exclusions),
        metadata_commands,
    )


def recursive_prompt(report: dict[str, Any]) -> str:
    pr_number = report.get("selected_pr")
    if pr_number:
        prompt = (
            f"Start from live truth in /Users/armand/Development/aragora. Goal: continue one-PR "
            f"settlement for #{pr_number} only using scripts/settle_one_pr.py as the steward. "
            "Do not broad-drain, do not touch #7407/#7425/#7438/#7439/#7443 unless live owner "
            "checks release them, no branch protection, labels, outbox, harvest, admin merge, or "
            "unscoped PR work. Rerun owner/mailbox checks, exact-head gh pr view/checks, "
            "merge-packet --pr, diff-check, and automation_pr_preflight. If the report says only "
            "model evidence is missing, collect the minimum current-head countable evidence, rerun "
            "packet and aragora-merge-quorum, then merge only by normal protected squash if "
            "status=satisfied and verdict=admin_squash_allowed."
        )
    else:
        prompt = (
            "Start from live truth in /Users/armand/Development/aragora. Goal: make incremental "
            "progress without broad queue drain by selecting exactly one Tier 0-2 non-human-risk PR "
            "or one steward-tooling repair. Run scripts/settle_one_pr.py --json first; if it reports "
            "no candidate, improve the steward's candidate diagnostics or target provider bootstrap "
            "so dogfood evidence collection becomes reliable. Do not touch Tier 3/4 or active-owned "
            "PRs, branch protection, labels, outbox, harvest, or admin merge."
        )
    return f"{prompt}\n{CONVERGENCE_SENTENCE}"


def build_report(
    packet: dict[str, Any],
    *,
    cwd: Path,
    state_root: Path | None = None,
    explicit_pr: int | None,
    exclude_prs: set[int],
    live: bool,
    validate: bool,
    repo: str | None = None,
) -> dict[str, Any]:
    policy_metadata: dict[int, dict[str, Any]] = {}
    active_owned_prs: set[int] = set()
    policy_context: dict[str, Any] = {}
    policy_metadata_commands: list[dict[str, Any]] = []
    preselection_blockers = [str(item) for item in packet.get("load_blockers") or []]
    load_warnings = [str(item) for item in packet.get("load_warnings") or []]
    if live:
        repo_blocker, repo_command, cwd_repo = repo_cwd_blocker(cwd, repo)
        if repo_blocker:
            preselection_blockers.append(repo_blocker)
        if explicit_pr is not None:
            metadata, metadata_command = load_pr_policy_metadata(cwd, explicit_pr, repo=repo)
            policy_metadata = {explicit_pr: metadata} if metadata else {}
        else:
            policy_metadata, metadata_command = load_open_pr_metadata(cwd, repo=repo)
        policy_metadata_commands.append(metadata_command)
        active_owned_command: dict[str, Any] | None = None
        snapshot_preblocked = _has_operator_snapshot_load_blocker(preselection_blockers)
        if not repo_blocker and not snapshot_preblocked:
            active_owned_prs, active_owned_command = load_active_owned_prs(cwd)
            snapshot_blocker = active_owned_snapshot_blocker(active_owned_command)
            if snapshot_blocker:
                preselection_blockers.append(snapshot_blocker)
        elif snapshot_preblocked:
            active_owned_command = {
                "command": "python3 scripts/agent_bridge.py operator-snapshot --json",
                "returncode": None,
                "skipped": True,
                "reason": "operator-snapshot failure already carried by packet load_blockers",
            }
        policy_context = {
            "repo": repo,
            "policy_metadata_commands": policy_metadata_commands,
            "operator_snapshot_command": active_owned_command,
            "active_owned_prs": sorted(active_owned_prs),
        }
        if repo is not None:
            policy_context["cwd_repo_command"] = repo_command
            policy_context["cwd_repo"] = cwd_repo
        if load_warnings:
            policy_context["load_warnings"] = load_warnings

    if live:
        selected, selection_blockers, policy_exclusions, lazy_commands = (
            select_candidate_with_lazy_policy_metadata(
                packet,
                cwd=cwd,
                repo=repo,
                explicit_pr=explicit_pr,
                exclude_prs=exclude_prs,
                active_owned_prs=active_owned_prs,
                policy_metadata=policy_metadata,
            )
        )
        policy_metadata_commands.extend(lazy_commands)
    else:
        selected, selection_blockers, policy_exclusions = cast(
            SelectionResultWithExclusions,
            select_candidate(
                packet,
                explicit_pr=explicit_pr,
                exclude_prs=exclude_prs,
                active_owned_prs=active_owned_prs,
                policy_metadata=policy_metadata,
                return_exclusions=True,
            ),
        )
    has_preselection_blockers = bool(preselection_blockers)
    selection_blockers = _dedupe_strings([*preselection_blockers, *selection_blockers])
    report: dict[str, Any] = {
        "version": VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": True,
        "packet_summary": {
            "entry_count": len(packet.get("entries") or []),
            "admin_squash_order": packet.get("admin_squash_order") or [],
            "not_ready_count": len(packet.get("not_ready") or []),
            "human_risk_settlement_required": packet.get("human_risk_settlement_required") or [],
        },
        "selected_pr": None,
        "head_sha": "",
        "status": "no_candidate",
        "blockers": selection_blockers,
        "evidence": {},
        "checks": {},
        "policy_context": policy_context,
        "load_warnings": load_warnings,
        "policy_exclusions": policy_exclusions,
        "validation": [],
        "suggested_commands": [],
    }
    if selected is None:
        if has_preselection_blockers:
            report["status"] = "blocked"
        report["recursive_best_next_prompt"] = recursive_prompt(report)
        return report

    pr_number = _entry_pr(selected)
    report["selected_pr"] = pr_number
    report["head_sha"] = str(selected.get("head_sha", "") or "")
    blockers = list(selection_blockers)
    blockers.extend(entry_blockers(selected))
    blockers.extend(
        f"excluded_by_policy: {reason}"
        for reason in policy_exclusion_reasons(
            selected,
            exclude_prs=exclude_prs,
            active_owned_prs=active_owned_prs,
            policy_metadata=policy_metadata,
        )
    )
    report["evidence"] = evidence_summary(selected)

    if live and pr_number is not None and not blockers:
        state_root = state_root or _state_repo_root(cwd)
        registry_path = state_root / ".aragora" / "agent-bridge" / "lanes.json"
        steering_root = state_root / ".aragora" / "operator-steering"
        owner_payload, owner_cmd = _run_json(
            [
                "python3",
                "scripts/identify_lane_owner.py",
                "--pr",
                str(pr_number),
                "--json",
                "--registry-path",
                str(registry_path),
                "--steering-inbox-root",
                str(steering_root),
            ],
            cwd=cwd,
        )
        report["owner_check"] = owner_cmd
        blockers.extend(owner_blockers(owner_payload))

        steering_payload, steering_cmd = _run_json(
            [
                "python3",
                "scripts/read_operator_steering.py",
                "--pr",
                str(pr_number),
                "--read-by-session",
                "settle-one-steward",
                "--no-receipt",
                "--json",
                "--quiet-empty",
                "--registry-path",
                str(registry_path),
                "--steering-inbox-root",
                str(steering_root),
            ],
            cwd=cwd,
        )
        report["mailbox_check"] = steering_cmd
        if isinstance(steering_payload, dict) and steering_payload.get("message"):
            blockers.append("operator steering message exists; read and obey before settlement")

        pr_view, view_cmd = _run_json(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--json",
                (
                    "number,state,isDraft,headRefOid,headRefName,baseRefName,mergeable,"
                    "mergeStateStatus,statusCheckRollup"
                ),
            ],
            cwd=cwd,
        )
        report["pr_view_check"] = view_cmd
        blockers.extend(head_blockers(selected, pr_view))

        base_ref = "main"
        if isinstance(pr_view, dict):
            base_ref = str(pr_view.get("baseRefName") or base_ref)
        protected_branch = quote(base_ref, safe="")

        protection, protection_cmd = _run_json(
            [
                "gh",
                "api",
                f"repos/{{owner}}/{{repo}}/branches/{protected_branch}/protection/required_status_checks",
            ],
            cwd=cwd,
        )
        report["required_check_source_command"] = protection_cmd

        head_ref = report["head_sha"]
        if isinstance(pr_view, dict):
            head_ref = str(pr_view.get("headRefOid") or head_ref)
        check_runs_payload, check_runs_cmd = _run_json(
            [
                "gh",
                "api",
                f"repos/{{owner}}/{{repo}}/commits/{head_ref}/check-runs?per_page=100",
                "--jq",
                (
                    "{check_runs: [.check_runs[] | "
                    "{name, status, conclusion, app: {id: .app.id, slug: .app.slug}}]}"
                ),
            ],
            cwd=cwd,
        )
        report["check_run_source_command"] = check_runs_cmd

        required_checks, required_cmd = _run_json(
            [
                "gh",
                "pr",
                "checks",
                str(pr_number),
                "--required",
                "--json",
                "name,state,bucket,workflow,link",
            ],
            cwd=cwd,
        )
        report["required_checks_command"] = required_cmd
        check_report = required_check_report(required_checks)
        report["checks"]["required"] = check_report
        blockers.extend(check_report["blockers"])
        report["suggested_commands"].extend(check_report["suggestions"])

        source_report = required_check_source_report(protection, pr_view, check_runs_payload)
        report["checks"]["required_sources"] = source_report
        blockers.extend(source_report["blockers"])
        report["suggested_commands"].extend(source_report.get("suggestions") or [])

    should_validate = validate and not blockers
    report["validation"] = validation_report(selected, cwd=cwd, run_validation=should_validate)
    for item in report["validation"]:
        if item.get("status") == "blocked":
            blockers.append(f"validation failed: {item.get('command')}")

    if (
        not blockers
        and selected.get("admin_squash_allowed")
        and selected.get("status") == "satisfied"
    ):
        report["status"] = "packet_authorized_dry_run"
        report["suggested_commands"].append(
            f"gh pr merge {pr_number} --squash --match-head-commit {report['head_sha']}"
        )
    elif not blockers and (
        report["evidence"].get("missing_model_quorum")
        or report["evidence"].get("missing_focused_dogfood")
    ):
        report["status"] = "ready_for_minimum_evidence"
        report["suggested_commands"].append(
            f"collect minimum current-head countable model evidence for #{pr_number}"
        )
    elif blockers:
        report["status"] = "blocked"
    else:
        report["status"] = "needs_packet_rerun"

    report["blockers"] = _dedupe_strings(blockers)
    report["recursive_best_next_prompt"] = recursive_prompt(report)
    return report


def _load_single_pr_packet(*, cwd: Path, pr: int, repo: str | None) -> dict[str, Any]:
    command = [
        "python3",
        "-m",
        "aragora.cli.main",
        "review-queue",
        "merge-packet",
        "--json",
        "--pr",
        str(pr),
    ]
    if repo:
        command.extend(["--repo", repo])
    payload, result = _run_json(command, cwd=cwd, timeout=600)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "merge-packet failed")
    if not isinstance(payload, dict):
        raise RuntimeError("merge-packet did not return a JSON object")
    return payload


def _load_broad_packet_bulk(*, cwd: Path, limit: int, repo: str | None) -> dict[str, Any]:
    command = [
        "python3",
        "-m",
        "aragora.cli.main",
        "review-queue",
        "merge-packet",
        "--json",
        "--limit",
        str(limit),
    ]
    if repo:
        command.extend(["--repo", repo])
    payload, result = _run_json(command, cwd=cwd, timeout=600)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "merge-packet failed")
    if not isinstance(payload, dict):
        raise RuntimeError("merge-packet did not return a JSON object")
    return payload


def _combine_packets(packets: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    admin_order: list[int] = []
    human_risk: list[int] = []
    not_ready: list[int] = []
    source_generated_at: list[str] = []
    for packet in packets:
        generated_at = packet.get("generated_at")
        if generated_at:
            source_generated_at.append(str(generated_at))
        for entry in packet.get("entries") or []:
            if isinstance(entry, dict):
                entries.append(entry)
        for raw_pr in packet.get("admin_squash_order") or []:
            pr_number = _coerce_int(raw_pr)
            if pr_number is not None and pr_number not in admin_order:
                admin_order.append(pr_number)
        for raw_pr in packet.get("human_risk_settlement_required") or []:
            pr_number = _coerce_int(raw_pr)
            if pr_number is not None and pr_number not in human_risk:
                human_risk.append(pr_number)
        for raw_pr in packet.get("not_ready") or []:
            pr_number = _coerce_int(raw_pr)
            if pr_number is not None and pr_number not in not_ready:
                not_ready.append(pr_number)
    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": entries,
        "admin_squash_order": admin_order,
        "human_risk_settlement_required": human_risk,
        "not_ready": not_ready,
        "source_generated_at": source_generated_at,
    }


def _empty_packet() -> dict[str, Any]:
    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": [],
        "admin_squash_order": [],
        "human_risk_settlement_required": [],
        "not_ready": [],
    }


def _light_entry_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "pr_number": metadata.get("number"),
        "title": metadata.get("title"),
        "head_sha": metadata.get("headRefOid"),
        "tier": 0,
        "reasons": [],
    }


def load_broad_packet_lazily(*, cwd: Path, limit: int, repo: str | None) -> dict[str, Any]:
    try:
        return _load_broad_packet_bulk(cwd=cwd, limit=limit, repo=repo)
    except RuntimeError as bulk_error:
        load_warnings = [f"bulk merge-packet failed; using fallback: {bulk_error}"]
        metadata: dict[int, dict[str, Any]] = {}
        metadata, command = load_open_pr_metadata(cwd, limit=limit, repo=repo)
        if command.get("returncode") != 0:
            packet = _empty_packet()
            packet["load_blockers"] = [
                str(bulk_error),
                command.get("stderr") or command.get("stdout") or "gh pr list failed",
            ]
            packet["load_warnings"] = load_warnings
            return packet

    active_owned_prs, active_owned_command = load_active_owned_prs(cwd)
    snapshot_blocker = active_owned_snapshot_blocker(active_owned_command)
    if snapshot_blocker:
        packet = _empty_packet()
        packet["load_blockers"] = [snapshot_blocker]
        packet["load_warnings"] = load_warnings
        return packet
    packets: list[dict[str, Any]] = []
    packet_failures: list[str] = []
    packet_attempts = 0
    selected_seen = False
    packets_after_selected = 0
    for pr_number, item in metadata.items():
        light_reasons = policy_exclusion_reasons(
            _light_entry_from_metadata(item),
            exclude_prs=HUMAN_RISK_EXCLUDES,
            active_owned_prs=active_owned_prs,
            policy_metadata=metadata,
        )
        if light_reasons:
            continue
        packet_attempts += 1
        try:
            packets.append(_load_single_pr_packet(cwd=cwd, pr=pr_number, repo=repo))
        except RuntimeError as exc:
            packet_failures.append(f"merge-packet for #{pr_number} failed: {exc}")
            continue
        if selected_seen:
            packets_after_selected += 1
            if packets_after_selected >= BROAD_PACKET_NEAR_SELECTED_LOOKAHEAD:
                break
            continue
        selected, _blockers, _exclusions = cast(
            SelectionResultWithExclusions,
            select_candidate(
                _combine_packets(packets),
                explicit_pr=None,
                exclude_prs=HUMAN_RISK_EXCLUDES,
                active_owned_prs=active_owned_prs,
                policy_metadata=metadata,
                return_exclusions=True,
            ),
        )
        if selected is not None:
            selected_seen = True
    packet = _combine_packets(packets) if packets else _empty_packet()
    if packet_attempts > BROAD_PACKET_NEAR_SELECTED_LOOKAHEAD:
        load_warnings.append(
            "fallback per-PR merge-packet queries: "
            f"{packet_attempts} (light candidates={len(metadata)}, limit={limit})"
        )
    packet["load_warnings"] = load_warnings
    if packet_failures:
        packet["load_blockers"] = packet_failures
    return packet


def load_packet(*, cwd: Path, pr: int | None, limit: int, repo: str | None) -> dict[str, Any]:
    if pr is not None:
        return _load_single_pr_packet(cwd=cwd, pr=pr, repo=repo)
    return load_broad_packet_lazily(cwd=cwd, limit=limit, repo=repo)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, default=None, help="Inspect one PR instead of selecting")
    parser.add_argument("--limit", type=int, default=100, help="Broad packet limit when no --pr")
    parser.add_argument("--repo", default=None, help="GitHub repo slug override")
    parser.add_argument(
        "--exclude-pr",
        action="append",
        type=int,
        default=[],
        help="PR number to exclude from automatic selection. Repeatable.",
    )
    parser.add_argument("--packet-file", default=None, help="Use a saved merge-packet JSON file")
    parser.add_argument("--no-live", action="store_true", help="Skip gh/owner/mailbox probes")
    parser.add_argument("--no-validate", action="store_true", help="Skip diff/preflight validation")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cwd = _repo_root()
    exclude_prs = set(args.exclude_pr or []) | HUMAN_RISK_EXCLUDES
    try:
        if args.packet_file:
            packet = json.loads(Path(args.packet_file).read_text(encoding="utf-8"))
        else:
            packet = load_packet(cwd=cwd, pr=args.pr, limit=args.limit, repo=args.repo)
        report = build_report(
            packet,
            cwd=cwd,
            state_root=_state_repo_root(cwd),
            explicit_pr=args.pr,
            exclude_prs=exclude_prs,
            repo=args.repo,
            live=not args.no_live,
            validate=not args.no_validate,
        )
    except Exception as exc:
        report = {
            "version": VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "dry_run": True,
            "status": "error",
            "blockers": [str(exc)],
            "recursive_best_next_prompt": recursive_prompt({"selected_pr": None}),
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
            print(report["recursive_best_next_prompt"])
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        selected = report.get("selected_pr")
        print(f"settle-one status: {report.get('status')}")
        print(f"selected PR: {('#' + str(selected)) if selected else '(none)'}")
        if report.get("head_sha"):
            print(f"head: {report['head_sha']}")
        for blocker in report.get("blockers") or []:
            print(f"- blocker: {blocker}")
        for command in report.get("suggested_commands") or []:
            print(f"- suggested: {command}")
        print()
        print("recursive best next prompt:")
        print(report["recursive_best_next_prompt"])
    return 0 if report.get("status") not in {"error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
