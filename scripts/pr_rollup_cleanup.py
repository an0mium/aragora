#!/usr/bin/env python3
"""Dry-run-first helper for bounded PR rollup cleanup.

This script evaluates one pull request and can rerun a cancelled non-required
Build Documentation workflow exactly once when all live safety gates are green.
It never merges, labels, sets statuses, marks ready, or touches branch
protection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Callable

VERSION = "pr_rollup_cleanup.v1"
BUILD_DOCS_WORKFLOW = "Build Documentation (PR Check)"
BUILD_DOCS_JOB = "build"
SELF_HOSTED_SHADOW_WORKFLOW = "Self-Hosted Shadow CI"
RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": " ".join(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout.strip(),
            "stderr": self.stderr.strip(),
        }


Runner = Callable[[list[str], int], CommandResult]


def _run(args: list[str], timeout: int = 120) -> CommandResult:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=args,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _with_repo(args: list[str], repo: str | None) -> list[str]:
    if repo:
        return [*args, "--repo", repo]
    return args


def _parse_json_result(result: CommandResult) -> tuple[Any | None, str | None]:
    if not result.stdout.strip():
        return None, "empty stdout"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def _run_json(
    args: list[str], *, repo: str | None, runner: Runner
) -> tuple[Any | None, dict[str, Any] | None]:
    result = runner(_with_repo(args, repo), 120)
    payload, error = _parse_json_result(result)
    if error is not None:
        data = result.to_dict()
        data["json_error"] = error
        return None, data
    return payload, result.to_dict()


def _extract_run_id(details_url: str | None) -> str | None:
    if not details_url:
        return None
    match = RUN_ID_RE.search(details_url)
    return match.group(1) if match else None


def _check_name(check: dict[str, Any]) -> str:
    return str(check.get("name") or check.get("context") or "").strip()


def _check_workflow(check: dict[str, Any]) -> str:
    return str(check.get("workflowName") or check.get("workflow") or "").strip()


def _check_status(check: dict[str, Any]) -> str:
    return str(check.get("status") or "").strip().upper()


def _check_conclusion(check: dict[str, Any]) -> str:
    return str(check.get("conclusion") or "").strip().upper()


def _required_bucket(check: dict[str, Any]) -> str:
    bucket = str(check.get("bucket") or "").strip().lower()
    if bucket:
        return bucket
    state = str(check.get("state") or "").strip().upper()
    if state in {"SUCCESS", "PASS"}:
        return "pass"
    if state in {"FAILURE", "ERROR", "FAILED"}:
        return "fail"
    if state in {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED"}:
        return "pending"
    if state in {"CANCELLED", "CANCELED"}:
        return "cancel"
    if state in {"SKIPPED", "NEUTRAL"}:
        return "skipping"
    return state.lower() or "unknown"


def _summarize_required_checks(required_checks: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    non_green: list[dict[str, Any]] = []
    for check in required_checks:
        if not isinstance(check, dict):
            continue
        bucket = _required_bucket(check)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if bucket != "pass":
            non_green.append(
                {
                    "name": _check_name(check),
                    "workflow": _check_workflow(check),
                    "bucket": bucket,
                    "state": check.get("state"),
                    "link": check.get("link"),
                }
            )
    total = len([check for check in required_checks if isinstance(check, dict)])
    return {
        "total": total,
        "green": total > 0 and not non_green,
        "buckets": buckets,
        "non_green": non_green,
        "summary": f"{buckets.get('pass', 0)}/{total} pass",
    }


def _is_pending_self_hosted(check: dict[str, Any]) -> bool:
    return _check_workflow(check) == SELF_HOSTED_SHADOW_WORKFLOW and _check_status(check) in {
        "QUEUED",
        "IN_PROGRESS",
        "PENDING",
        "REQUESTED",
        "WAITING",
    }


def _is_cancelled_check(check: dict[str, Any]) -> bool:
    return _check_status(check) == "COMPLETED" and _check_conclusion(check) in {
        "CANCELLED",
        "CANCELED",
    }


def _is_ignorable_rollup(check: dict[str, Any]) -> bool:
    if check.get("__typename") == "StatusContext":
        return str(check.get("state") or "").upper() == "SUCCESS"
    conclusion = _check_conclusion(check)
    status = _check_status(check)
    if status == "COMPLETED" and conclusion in {"SUCCESS", "SKIPPED", "NEUTRAL"}:
        return True
    return False


def _rollup_item(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _check_name(check),
        "workflow": _check_workflow(check),
        "status": check.get("status"),
        "conclusion": check.get("conclusion"),
        "details_url": check.get("detailsUrl") or check.get("targetUrl"),
        "run_id": _extract_run_id(str(check.get("detailsUrl") or "")),
    }


def _is_build_docs_cancelled(check: dict[str, Any]) -> bool:
    return (
        _is_cancelled_check(check)
        and _check_workflow(check) == BUILD_DOCS_WORKFLOW
        and _check_name(check) == BUILD_DOCS_JOB
    )


def _summarize_rollup(rollup: list[dict[str, Any]]) -> dict[str, Any]:
    pending_self_hosted: list[dict[str, Any]] = []
    cancelled: list[dict[str, Any]] = []
    actionable_build_docs: list[dict[str, Any]] = []
    unexpected_non_green: list[dict[str, Any]] = []
    for raw in rollup:
        if not isinstance(raw, dict):
            continue
        item = _rollup_item(raw)
        if _is_pending_self_hosted(raw):
            pending_self_hosted.append(item)
            unexpected_non_green.append(item)
            continue
        if _is_cancelled_check(raw):
            cancelled.append(item)
            if _is_build_docs_cancelled(raw):
                actionable_build_docs.append(item)
            else:
                unexpected_non_green.append(item)
            continue
        if not _is_ignorable_rollup(raw):
            unexpected_non_green.append(item)
    return {
        "total": len([item for item in rollup if isinstance(item, dict)]),
        "pending_self_hosted": pending_self_hosted,
        "cancelled": cancelled,
        "actionable_build_documentation": actionable_build_docs,
        "unexpected_non_green": unexpected_non_green,
    }


def _base_output(pr_number: int, repo: str | None, expected_head: str | None) -> dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "pr": pr_number,
        "repo": repo,
        "expected_head": expected_head,
        "ok": False,
        "safe_to_apply": False,
        "proposed_action": "stop",
        "blocker": None,
        "rerun_performed": False,
        "rerun_run_id": None,
        "commands": [],
    }


def evaluate_rollup_cleanup(
    *,
    pr_number: int,
    expected_head: str | None,
    repo: str | None = None,
    apply: bool = False,
    runner: Runner = _run,
) -> dict[str, Any]:
    output = _base_output(pr_number, repo, expected_head)
    if apply and not expected_head:
        output["blocker"] = "exact_head_required_for_apply"
        return output

    pr_payload, pr_command = _run_json(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,state,isDraft,headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url",
        ],
        repo=repo,
        runner=runner,
    )
    if pr_command:
        output["commands"].append(pr_command)
    if not isinstance(pr_payload, dict):
        output["blocker"] = "pr_view_unavailable"
        return output

    head = str(pr_payload.get("headRefOid") or "")
    output["head"] = head
    output["state"] = pr_payload.get("state")
    output["is_draft"] = pr_payload.get("isDraft")
    output["mergeable"] = pr_payload.get("mergeable")
    output["merge_state_status"] = pr_payload.get("mergeStateStatus")
    output["url"] = pr_payload.get("url")

    if expected_head and head != expected_head:
        output["blocker"] = "exact_head_mismatch"
        return output
    if pr_payload.get("state") != "OPEN":
        output["blocker"] = "pr_not_open"
        return output
    if bool(pr_payload.get("isDraft")):
        output["blocker"] = "pr_is_draft"
        return output

    required_payload, required_command = _run_json(
        [
            "gh",
            "pr",
            "checks",
            str(pr_number),
            "--required",
            "--json",
            "name,state,bucket,workflow,link,startedAt,completedAt",
        ],
        repo=repo,
        runner=runner,
    )
    if required_command:
        output["commands"].append(required_command)
    if not isinstance(required_payload, list):
        output["blocker"] = "required_checks_unavailable"
        return output

    required_summary = _summarize_required_checks(required_payload)
    output["required_checks"] = required_summary
    if not required_summary["green"]:
        output["blocker"] = "required_checks_not_green"
        return output

    rollup = pr_payload.get("statusCheckRollup") or []
    rollup_summary = _summarize_rollup(rollup if isinstance(rollup, list) else [])
    output["rollup"] = rollup_summary

    if rollup_summary["pending_self_hosted"]:
        output["proposed_action"] = "wait"
        output["blocker"] = "self_hosted_shadow_pending"
        return output

    actionable = rollup_summary["actionable_build_documentation"]
    unexpected = rollup_summary["unexpected_non_green"]
    if not actionable and not unexpected:
        output["ok"] = True
        output["proposed_action"] = "none"
        return output
    if len(actionable) == 1 and not unexpected:
        run_id = actionable[0].get("run_id")
        if not run_id:
            output["blocker"] = "build_documentation_run_id_missing"
            return output
        output["ok"] = True
        output["safe_to_apply"] = True
        output["proposed_action"] = "rerun_build_documentation"
        output["rerun_run_id"] = run_id
        if apply:
            result = runner(_with_repo(["gh", "run", "rerun", str(run_id)], repo), 120)
            output["commands"].append(result.to_dict())
            output["rerun_performed"] = result.returncode == 0
            if result.returncode != 0:
                output["ok"] = False
                output["blocker"] = "rerun_failed"
        return output

    output["blocker"] = "unexpected_non_green_rollup"
    return output


def _render_text(result: dict[str, Any]) -> None:
    print(f"PR #{result['pr']} head={result.get('head') or 'unknown'}")
    print(f"action={result.get('proposed_action')} safe_to_apply={result.get('safe_to_apply')}")
    if result.get("blocker"):
        print(f"blocker={result['blocker']}")
    required = result.get("required_checks") or {}
    if required:
        print(f"required_checks={required.get('summary')}")
    rollup = result.get("rollup") or {}
    if rollup:
        print(f"pending_self_hosted={len(rollup.get('pending_self_hosted') or [])}")
        print(f"cancelled={len(rollup.get('cancelled') or [])}")
    if result.get("rerun_performed"):
        print(f"reran Build Documentation run {result.get('rerun_run_id')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True, help="Pull request number to evaluate")
    parser.add_argument("--head", help="Expected exact head SHA; mismatch fails closed")
    parser.add_argument("--repo", help="GitHub repository, e.g. synaptent/aragora")
    parser.add_argument("--apply", action="store_true", help="Perform the single safe rerun")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_rollup_cleanup(
        pr_number=args.pr,
        expected_head=args.head,
        repo=args.repo,
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _render_text(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
