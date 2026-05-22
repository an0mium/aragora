#!/usr/bin/env python3
"""Check or apply an exact-head Tier 4 PR settlement.

Tier 4 automation may prepare a packet, but merge/protection mutation requires
a repo-visible operator settlement comment naming the exact head and action.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPO = "synaptent/aragora"
AUTHORIZED_MARKER = "Tier-4 Human Settlement Authorization"
AUTHORIZED_ACTION_TOKENS = ("admin_squash_merge", "admin squash")
TRUSTED_OPERATOR_AUTHOR_ASSOCIATIONS = {"OWNER", "MEMBER"}
ALLOWED_TIER4_NOT_READY = {
    "human_risk_settlement",
    "tier4_human_risk_settlement",
    "operator_settlement_required",
}


def _text_items(pr_view: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("comments", "reviews"):
        value = pr_view.get(key)
        if not isinstance(value, list):
            continue
        for entry in value:
            if isinstance(entry, dict):
                body = entry.get("body")
                if isinstance(body, str):
                    items.append(
                        {
                            "body": body,
                            "authorAssociation": entry.get("authorAssociation"),
                            "author": entry.get("author"),
                        }
                    )
    return items


def has_operator_authorization(pr_view: dict[str, Any], *, head: str) -> bool:
    for item in _text_items(pr_view):
        body = str(item.get("body") or "")
        lowered = body.lower()
        if AUTHORIZED_MARKER not in body:
            continue
        association = str(item.get("authorAssociation") or "").upper()
        if association not in TRUSTED_OPERATOR_AUTHOR_ASSOCIATIONS:
            continue
        if head not in body:
            continue
        if any(token in lowered for token in AUTHORIZED_ACTION_TOKENS):
            return True
    return False


def evaluate_tier4_gate(
    *,
    pr: int,
    expected_head: str,
    pr_view: dict[str, Any],
    merge_packet: dict[str, Any],
    required_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    actual_head = str(pr_view.get("headRefOid") or "")
    if actual_head != expected_head:
        blockers.append(f"head mismatch: expected {expected_head}, got {actual_head}")
    if str(pr_view.get("state") or "").upper() != "OPEN":
        blockers.append(f"PR #{pr} is not open")
    if bool(pr_view.get("isDraft")):
        blockers.append(f"PR #{pr} is draft")
    merge_state = str(pr_view.get("mergeStateStatus") or "")
    if merge_state in {"DIRTY", "CONFLICTING"}:
        blockers.append(f"PR #{pr} is {merge_state}")
    for check in required_checks or []:
        name = str(check.get("name") or check.get("workflow") or "required check")
        state = str(check.get("state") or check.get("conclusion") or "UNKNOWN").upper()
        if state not in {"SUCCESS", "PASS", "PASSED", "SKIPPED", "NEUTRAL"}:
            blockers.append(f"required check {name} is {state}")
    not_ready = merge_packet.get("not_ready")
    if isinstance(not_ready, list):
        unexpected = sorted({str(item) for item in not_ready} - ALLOWED_TIER4_NOT_READY)
        if unexpected:
            blockers.append(f"merge-packet has unexpected blockers: {', '.join(unexpected)}")
    if actual_head == expected_head and not has_operator_authorization(pr_view, head=expected_head):
        blockers.append("missing repo-visible Tier 4 operator settlement comment")

    return {
        "ok": not blockers,
        "pr": pr,
        "expected_head": expected_head,
        "actual_head": actual_head,
        "merge_state": merge_state,
        "blockers": blockers,
    }


def _run_json(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{' '.join(command)} did not emit JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{' '.join(command)} emitted non-object JSON")
    return payload


def _run_json_any(command: list[str], *, cwd: Path | None = None) -> Any:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{' '.join(command)} did not emit JSON") from exc


def _load_live_inputs(
    pr: int, *, cwd: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pr_view = _run_json(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--json",
            "headRefOid,state,isDraft,mergeStateStatus,comments,reviews,url",
        ],
        cwd=cwd,
    )
    merge_packet = _run_json(
        [
            sys.executable,
            "-m",
            "aragora.cli.main",
            "review-queue",
            "merge-packet",
            "--pr",
            str(pr),
            "--json",
        ],
        cwd=cwd,
    )
    checks_raw = _run_json_any(
        [
            "gh",
            "pr",
            "checks",
            str(pr),
            "--required",
            "--json",
            "name,state,bucket,workflow,link",
        ],
        cwd=cwd,
    )
    required_checks = (
        [check for check in checks_raw if isinstance(check, dict)]
        if isinstance(checks_raw, list)
        else []
    )
    return pr_view, merge_packet, required_checks


def _required_status_check_patch(*, repo: str, cwd: Path) -> tuple[list[str], str]:
    endpoint = f"repos/{repo}/branches/main/protection/required_status_checks"
    current = _run_json(["gh", "api", endpoint], cwd=cwd)
    contexts = current.get("contexts")
    if not isinstance(contexts, list):
        checks = current.get("checks")
        contexts = (
            [
                str(check.get("context"))
                for check in checks
                if isinstance(check, dict) and check.get("context")
            ]
            if isinstance(checks, list)
            else []
        )
    context_set = {str(context) for context in contexts if str(context)}
    context_set.add("aragora-merge-quorum")
    payload = {"strict": bool(current.get("strict", True)), "contexts": sorted(context_set)}
    return ["gh", "api", "--method", "PATCH", endpoint, "--input", "-"], json.dumps(payload)


def _run_command(command: list[str], *, cwd: Path, input_text: str | None = None) -> None:
    subprocess.run(command, cwd=cwd, input=input_text, text=True, check=True, timeout=180)


def _apply_settlement(*, pr: int, head: str, repo: str, cwd: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    merge_command = [
        "gh",
        "pr",
        "merge",
        str(pr),
        "--squash",
        "--admin",
        "--match-head-commit",
        head,
    ]
    _run_command(merge_command, cwd=cwd)
    commands.append(merge_command)

    reviews_command = [
        "gh",
        "api",
        "--method",
        "PATCH",
        f"repos/{repo}/branches/main/protection/required_pull_request_reviews",
        "--input",
        "-",
    ]
    _run_command(
        reviews_command,
        cwd=cwd,
        input_text=json.dumps(
            {
                "required_approving_review_count": 0,
                "require_code_owner_reviews": False,
            }
        ),
    )
    commands.append(reviews_command)

    checks_command, checks_payload = _required_status_check_patch(repo=repo, cwd=cwd)
    _run_command(checks_command, cwd=cwd, input_text=checks_payload)
    commands.append(checks_command)

    enforce_command = [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{repo}/branches/main/protection/enforce_admins",
    ]
    _run_command(enforce_command, cwd=cwd)
    commands.append(enforce_command)
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pr_view, merge_packet, required_checks = _load_live_inputs(args.pr, cwd=args.cwd)
        gate = evaluate_tier4_gate(
            pr=args.pr,
            expected_head=args.head,
            pr_view=pr_view,
            merge_packet=merge_packet,
            required_checks=required_checks,
        )
        applied_commands: list[list[str]] = []
        if args.apply:
            if not gate["ok"]:
                raise RuntimeError("Tier 4 gate is not satisfied; refusing --apply")
            applied_commands = _apply_settlement(
                pr=args.pr,
                head=args.head,
                repo=args.repo,
                cwd=args.cwd,
            )
        out = {"gate": gate, "applied_commands": applied_commands}
    except RuntimeError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print("ok" if gate["ok"] else "blocked")
        for blocker in gate["blockers"]:
            print(f"- {blocker}")
    return 0 if gate["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
