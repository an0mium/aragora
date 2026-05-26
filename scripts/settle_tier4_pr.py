#!/usr/bin/env python3
"""Check or apply an exact-head Tier 4 PR settlement.

Tier 4 automation may prepare a packet, but merge/protection mutation requires
a repo-visible operator settlement comment naming the exact head and action.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_REPO = "synaptent/aragora"
AUTHORIZED_MARKER = "Tier-4 Human Settlement Authorization"
AUTHORIZED_MERGE_TOKENS = ("admin_squash_merge", "admin squash")
AUTHORIZED_PROTECTION_TOKENS = ("branch_protection_reconcile", "branch protection reconcile")
TRUSTED_OPERATOR_AUTHOR_ASSOCIATIONS = {"OWNER"}
TRUSTED_OPERATOR_MEMBER_ASSOCIATIONS = {"MEMBER"}
TRUSTED_OPERATOR_LOGINS_ENV = "ARAGORA_TIER4_TRUSTED_OPERATORS"
PermissionChecker = Callable[[str], bool]
HUMAN_SETTLEMENT_CONTEXT = "aragora/human-settlement"
SUCCESS_STATES = {"SUCCESS", "PASS", "PASSED", "SKIPPED", "NEUTRAL"}
MIN_TIER4_COUNTED_REVIEWER_IDS = 2
ALLOWED_TIER4_NOT_READY = {
    "human_risk_settlement",
    "tier4_human_risk_settlement",
    "operator_settlement_required",
}
ALLOWED_TIER4_ENTRY_STATUSES = {
    "human_preapproval_required",
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
                            "createdAt": entry.get("createdAt") or entry.get("submittedAt"),
                        }
                    )
    return items


def _parse_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f"{text[:-1]}+00:00" if text.endswith("Z") else text


def _head_committed_at(pr_view: dict[str, Any]) -> str:
    direct = str(pr_view.get("headCommittedDate") or "").strip()
    if direct:
        return direct
    commits = pr_view.get("commits")
    if isinstance(commits, list) and commits:
        latest = commits[-1]
        if isinstance(latest, dict):
            return str(latest.get("committedDate") or "").strip()
    return ""


def _authorization_is_fresh(item: dict[str, Any], *, head_committed_at: str) -> bool:
    if not head_committed_at:
        return False
    created_at = str(item.get("createdAt") or "").strip()
    if not created_at:
        return False
    return _parse_timestamp(created_at) >= _parse_timestamp(head_committed_at)


def _trusted_operator_logins(extra_logins: Sequence[str] | None = None) -> frozenset[str]:
    configured = {
        login.strip().lower()
        for login in os.environ.get(TRUSTED_OPERATOR_LOGINS_ENV, "").split(",")
        if login.strip()
    }
    explicit = {login.strip().lower() for login in extra_logins or () if login.strip()}
    return frozenset(configured | explicit)


def _author_login(item: dict[str, Any]) -> str:
    author = item.get("author")
    if isinstance(author, dict):
        return str(author.get("login") or "").strip().lower()
    if isinstance(author, str):
        return author.strip().lower()
    return ""


def _collaborator_permission_is_admin(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("permission") or "").lower() == "admin"
        or str(payload.get("role_name") or "").lower() == "admin"
    )


def _login_has_admin_permission(login: str, repo: str, cwd: Path | None) -> bool:
    if not login:
        return False
    endpoint = f"repos/{repo}/collaborators/{quote(login, safe='')}/permission"
    try:
        payload = _run_json(["gh", "api", endpoint], cwd=cwd)
    except RuntimeError:
        return False
    return _collaborator_permission_is_admin(payload)


def _is_trusted_operator_author(
    item: dict[str, Any],
    *,
    trusted_operator_logins: frozenset[str],
    permission_checker: PermissionChecker,
) -> bool:
    association = str(item.get("authorAssociation") or "").upper()
    if association in TRUSTED_OPERATOR_AUTHOR_ASSOCIATIONS:
        return True
    if association not in TRUSTED_OPERATOR_MEMBER_ASSOCIATIONS:
        return False
    login = _author_login(item)
    if not login:
        return False
    if trusted_operator_logins and login not in trusted_operator_logins:
        return False
    return permission_checker(login)


def _state_is_success(value: Any) -> bool:
    return str(value or "").upper() in SUCCESS_STATES


def _required_checks_are_green(required_checks: list[dict[str, Any]] | None) -> bool:
    if not required_checks:
        return False
    for check in required_checks:
        state = check.get("state") or check.get("conclusion")
        if not _state_is_success(state):
            return False
    return True


def _human_settlement_status_is_success(pr_view: dict[str, Any]) -> bool:
    rollup = pr_view.get("statusCheckRollup")
    if not isinstance(rollup, list):
        return False
    for item in rollup:
        if not isinstance(item, dict):
            continue
        context = str(item.get("context") or item.get("name") or "")
        if context != HUMAN_SETTLEMENT_CONTEXT:
            continue
        state = item.get("state") or item.get("conclusion")
        return _state_is_success(state)
    return False


def _packet_has_counted_tier4_evidence(merge_packet: dict[str, Any], *, pr: int) -> bool:
    entry = _entry_for_pr(merge_packet, pr=pr)
    if not entry:
        return False
    if bool(entry.get("unresolved_dissent")):
        return False
    counted = entry.get("counted_reviewer_ids")
    counted_ids = (
        {str(item).strip() for item in counted if isinstance(item, str) and str(item).strip()}
        if isinstance(counted, list)
        else set()
    )
    if len(counted_ids) < MIN_TIER4_COUNTED_REVIEWER_IDS:
        return False
    dogfood = entry.get("dogfood_evidence")
    if not isinstance(dogfood, list) or not dogfood:
        return False
    return True


def _comment_authorizes_requested_action(
    body: str, *, require_branch_protection_token: bool
) -> bool:
    actions = _comment_authorized_actions(body)
    if "merge" not in actions:
        return False
    if require_branch_protection_token and "branch_protection" not in actions:
        return False
    return True


def _comment_authorized_actions(body: str) -> set[str]:
    lowered = body.lower()
    actions: set[str] = set()
    if any(token in lowered for token in AUTHORIZED_MERGE_TOKENS):
        actions.add("merge")
    if any(token in lowered for token in AUTHORIZED_PROTECTION_TOKENS):
        actions.add("branch_protection")
    return actions


def _operator_authorized_actions(
    pr_view: dict[str, Any],
    *,
    pr: int,
    head: str,
    merge_packet: dict[str, Any],
    required_checks: list[dict[str, Any]] | None = None,
    require_branch_protection_token: bool = False,
    repo: str = DEFAULT_REPO,
    cwd: Path | None = None,
    trusted_operator_logins: Sequence[str] | None = None,
    permission_checker: PermissionChecker | None = None,
) -> set[str]:
    if not _required_checks_are_green(required_checks):
        return set()
    if not _human_settlement_status_is_success(pr_view):
        return set()
    if not _packet_has_counted_tier4_evidence(merge_packet, pr=pr):
        return set()

    head_committed_at = _head_committed_at(pr_view)
    allowed_logins = _trusted_operator_logins(trusted_operator_logins)
    checker = permission_checker or (lambda login: _login_has_admin_permission(login, repo, cwd))
    for item in _text_items(pr_view):
        body = str(item.get("body") or "")
        if AUTHORIZED_MARKER not in body:
            continue
        if not _is_trusted_operator_author(
            item,
            trusted_operator_logins=allowed_logins,
            permission_checker=checker,
        ):
            continue
        if not _authorization_is_fresh(item, head_committed_at=head_committed_at):
            continue
        if head not in body:
            continue
        if _comment_authorizes_requested_action(
            body, require_branch_protection_token=require_branch_protection_token
        ):
            return _comment_authorized_actions(body)
    return set()


def has_operator_authorization(
    pr_view: dict[str, Any],
    *,
    pr: int,
    head: str,
    merge_packet: dict[str, Any],
    required_checks: list[dict[str, Any]] | None = None,
    require_branch_protection_token: bool = False,
    repo: str = DEFAULT_REPO,
    cwd: Path | None = None,
    trusted_operator_logins: Sequence[str] | None = None,
    permission_checker: PermissionChecker | None = None,
) -> bool:
    return bool(
        _operator_authorized_actions(
            pr_view,
            pr=pr,
            head=head,
            merge_packet=merge_packet,
            required_checks=required_checks,
            require_branch_protection_token=require_branch_protection_token,
            repo=repo,
            cwd=cwd,
            trusted_operator_logins=trusted_operator_logins,
            permission_checker=permission_checker,
        )
    )


def _entry_for_pr(merge_packet: dict[str, Any], *, pr: int) -> dict[str, Any] | None:
    entries = merge_packet.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("pr_number") or "") == str(pr):
            return entry
    return None


def _packet_marks_tier4_human_settlement(merge_packet: dict[str, Any], *, pr: int) -> bool:
    entry = _entry_for_pr(merge_packet, pr=pr)
    if not entry:
        return False
    status = str(entry.get("status") or "")
    if status not in ALLOWED_TIER4_ENTRY_STATUSES:
        return False
    if bool(entry.get("requires_human_risk_settlement")):
        return True
    required = merge_packet.get("human_risk_settlement_required")
    return isinstance(required, list) and str(pr) in {str(item) for item in required}


def evaluate_tier4_gate(
    *,
    pr: int,
    expected_head: str,
    pr_view: dict[str, Any],
    merge_packet: dict[str, Any],
    required_checks: list[dict[str, Any]] | None = None,
    require_branch_protection_token: bool = False,
    repo: str = DEFAULT_REPO,
    cwd: Path | None = None,
    trusted_operator_logins: Sequence[str] | None = None,
    permission_checker: PermissionChecker | None = None,
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
        if not _state_is_success(state):
            blockers.append(f"required check {name} is {state}")
    not_ready = merge_packet.get("not_ready")
    if isinstance(not_ready, list):
        allowed_not_ready = set(ALLOWED_TIER4_NOT_READY)
        if _packet_marks_tier4_human_settlement(merge_packet, pr=pr):
            allowed_not_ready.add(str(pr))
        unexpected = sorted({str(item) for item in not_ready} - allowed_not_ready)
        if unexpected:
            blockers.append(f"merge-packet has unexpected blockers: {', '.join(unexpected)}")
    authorized_actions: set[str] = set()
    if actual_head == expected_head:
        authorized_actions = _operator_authorized_actions(
            pr_view,
            pr=pr,
            head=expected_head,
            merge_packet=merge_packet,
            required_checks=required_checks,
            require_branch_protection_token=require_branch_protection_token,
            repo=repo,
            cwd=cwd,
            trusted_operator_logins=trusted_operator_logins,
            permission_checker=permission_checker,
        )
        if not authorized_actions:
            blockers.append("missing repo-visible Tier 4 operator settlement comment")

    return {
        "ok": not blockers,
        "pr": pr,
        "expected_head": expected_head,
        "actual_head": actual_head,
        "merge_state": merge_state,
        "blockers": blockers,
        "authorized_actions": sorted(authorized_actions),
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
            "headRefOid,state,isDraft,mergeStateStatus,comments,reviews,commits,statusCheckRollup,url",
        ],
        cwd=cwd,
    )
    pr_view["headCommittedDate"] = _head_committed_at(pr_view)
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


def _branch_protection_snapshot(*, repo: str, cwd: Path) -> dict[str, Any]:
    base = f"repos/{repo}/branches/main/protection"
    snapshot: dict[str, Any] = {}
    for key, endpoint in {
        "required_pull_request_reviews": f"{base}/required_pull_request_reviews",
        "required_status_checks": f"{base}/required_status_checks",
        "enforce_admins": f"{base}/enforce_admins",
    }.items():
        try:
            snapshot[key] = _run_json(["gh", "api", endpoint], cwd=cwd)
        except RuntimeError as exc:
            snapshot[key] = {"snapshot_error": str(exc)}
    return snapshot


def _restore_branch_protection(*, repo: str, cwd: Path, snapshot: dict[str, Any]) -> list[str]:
    base = f"repos/{repo}/branches/main/protection"
    errors: list[str] = []
    reviews = snapshot.get("required_pull_request_reviews")
    if isinstance(reviews, dict) and "snapshot_error" not in reviews:
        command = [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"{base}/required_pull_request_reviews",
            "--input",
            "-",
        ]
        try:
            _run_command(command, cwd=cwd, input_text=json.dumps(reviews))
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"restore required_pull_request_reviews failed: {exc}")
    checks = snapshot.get("required_status_checks")
    if isinstance(checks, dict) and "snapshot_error" not in checks:
        command = [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"{base}/required_status_checks",
            "--input",
            "-",
        ]
        try:
            _run_command(command, cwd=cwd, input_text=json.dumps(checks))
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"restore required_status_checks failed: {exc}")
    enforce = snapshot.get("enforce_admins")
    if isinstance(enforce, dict) and "snapshot_error" not in enforce:
        enabled = bool(enforce.get("enabled", False))
        command = [
            "gh",
            "api",
            "--method",
            "POST" if enabled else "DELETE",
            f"{base}/enforce_admins",
        ]
        try:
            _run_command(command, cwd=cwd)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"restore enforce_admins failed: {exc}")
    return errors


def _apply_settlement(
    *,
    pr: int,
    head: str,
    repo: str,
    cwd: Path,
    reconcile_branch_protection: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = []
    snapshot = (
        _branch_protection_snapshot(repo=repo, cwd=cwd) if reconcile_branch_protection else {}
    )
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
    try:
        _run_command(merge_command, cwd=cwd)
        commands.append(merge_command)

        if not reconcile_branch_protection:
            return commands

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
    except (OSError, subprocess.SubprocessError) as exc:
        rollback_errors = _restore_branch_protection(repo=repo, cwd=cwd, snapshot=snapshot)
        raise RuntimeError(
            "Tier 4 apply failed after partial execution; "
            f"completed_commands={len(commands)} rollback_errors={rollback_errors}: {exc}"
        ) from exc
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
    parser.add_argument(
        "--trusted-operator-login",
        action="append",
        default=[],
        help=(
            "Restrict repo-visible MEMBER authorization comments to this admin "
            "login. Repeatable; also reads comma-separated "
            f"{TRUSTED_OPERATOR_LOGINS_ENV}. If omitted, any live admin MEMBER "
            f"may authorize when {HUMAN_SETTLEMENT_CONTEXT} is success."
        ),
    )
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
            require_branch_protection_token=False,
            repo=args.repo,
            cwd=args.cwd,
            trusted_operator_logins=args.trusted_operator_login,
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
                reconcile_branch_protection="branch_protection"
                in set(gate.get("authorized_actions") or []),
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
