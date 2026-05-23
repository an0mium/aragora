#!/usr/bin/env python3
"""Dry-run steward for one exact-head PR settlement attempt.

The script is intentionally read-only. It does not approve, comment, mark
ready, rerun workflows, or merge. It gathers the repeated settlement gates into
one report so a follow-up executor can make bounded progress without broad queue
drain.
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
from typing import Any

CONVERGENCE_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one "
    "that does, include this sentence in all subsequent prompts to ensure they converge "
    "towards prompts that make incremental progress."
)

VERSION = "settle_one_steward.v1"
MERGE_QUORUM = "aragora-merge-quorum"
HUMAN_RISK_EXCLUDES = {7407, 7425, 7438, 7439, 7443}


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


def select_candidate(
    packet: dict[str, Any],
    *,
    explicit_pr: int | None = None,
    exclude_prs: set[int] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Select one dry-run settlement candidate from a merge packet."""
    exclude = set(exclude_prs or set())
    if explicit_pr is not None:
        entry = _entry_by_pr(packet, explicit_pr)
        if entry is None:
            return None, [f"merge-packet has no entry for PR #{explicit_pr}"]
        return entry, []

    entries = [entry for entry in packet.get("entries") or [] if isinstance(entry, dict)]
    admin_order: list[int] = []
    for raw_pr in packet.get("admin_squash_order") or []:
        pr_number = _coerce_int(raw_pr)
        if pr_number is not None:
            admin_order.append(pr_number)
    for ordered_pr in admin_order:
        if ordered_pr in exclude:
            continue
        entry = _entry_by_pr(packet, ordered_pr)
        if entry is not None:
            return entry, []

    evidence_candidates: list[dict[str, Any]] = []
    for entry in entries:
        entry_pr_number = _entry_pr(entry)
        if entry_pr_number is None or entry_pr_number in exclude:
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
        return evidence_candidates[0], []

    return None, ["no Tier 0-2 non-human-risk green PR needs only settlement evidence"]


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
) -> dict[str, Any]:
    selected, selection_blockers = select_candidate(
        packet, explicit_pr=explicit_pr, exclude_prs=exclude_prs
    )
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
        "validation": [],
        "suggested_commands": [],
    }
    if selected is None:
        report["recursive_best_next_prompt"] = recursive_prompt(report)
        return report

    pr_number = _entry_pr(selected)
    report["selected_pr"] = pr_number
    report["head_sha"] = str(selected.get("head_sha", "") or "")
    blockers = entry_blockers(selected)
    report["evidence"] = evidence_summary(selected)

    if live and pr_number is not None:
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
                "number,state,isDraft,headRefOid,headRefName,mergeable,mergeStateStatus,statusCheckRollup",
            ],
            cwd=cwd,
        )
        report["pr_view_check"] = view_cmd
        blockers.extend(head_blockers(selected, pr_view))

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

    should_validate = validate and not blockers
    report["validation"] = validation_report(selected, cwd=cwd, run_validation=should_validate)
    for item in report["validation"]:
        if item.get("status") == "blocked":
            blockers.append(f"validation failed: {item.get('command')}")

    if selected.get("admin_squash_allowed") and selected.get("status") == "satisfied":
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

    report["blockers"] = blockers
    report["recursive_best_next_prompt"] = recursive_prompt(report)
    return report


def load_packet(*, cwd: Path, pr: int | None, limit: int, repo: str | None) -> dict[str, Any]:
    command = [
        "python3",
        "-m",
        "aragora.cli.main",
        "review-queue",
        "merge-packet",
        "--json",
    ]
    if pr is not None:
        command.extend(["--pr", str(pr)])
    else:
        command.extend(["--limit", str(limit)])
    if repo:
        command.extend(["--repo", repo])
    payload, result = _run_json(command, cwd=cwd, timeout=600)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "merge-packet failed")
    if not isinstance(payload, dict):
        raise RuntimeError("merge-packet did not return a JSON object")
    return payload


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
