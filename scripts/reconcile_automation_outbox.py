#!/usr/bin/env python3
"""Phase 3 of cleanup plan: reconcile automation outbox handoffs against
existing receipts and merged PR state.

Many .aragora/automation-outbox/*.json files are stale: their PR has merged,
or a terminal receipt was written, but the outbox file was never archived.
Each stale entry blocks the corresponding branch from being categorised as
cleanup-eligible by the audit script (because unresolved_outbox_handoff_branches
returns it as protected).

This script:
  1. Reads every outbox file
  2. For each, checks: matching receipt exists? superseded handoff? matching PR merged?
  3. Archives satisfied outbox files to .aragora/automation-outbox-archive/
     and writes a synthetic receipt if needed (so future audits stay correct)
  4. Reports counts before/after

Read-only by default (--dry-run); pass --apply to actually move files.
Dry-run reports are printed to stdout; pass --write-report or --out to persist
a JSON report.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_codex_branch_backlog import (  # noqa: E402
    open_pr_heads,
    run_git,
)
from github_cli_health import check_github_cli_health  # noqa: E402

UTC = timezone.utc


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _list_json(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".json")


def _branch_has_landed_on_main(root: Path, base: str, branch: str) -> bool:
    """Return True if the branch's HEAD or a patch-equivalent commit is on main."""
    proc = run_git(["rev-parse", "--verify", branch], root, timeout=15)
    if proc.returncode != 0:
        return False
    proc = run_git(["merge-base", "--is-ancestor", branch, base], root, timeout=15)
    if proc.returncode == 0:
        return True
    proc = run_git(["cherry", base, branch], root, timeout=120)
    if proc.returncode != 0:
        return False
    statuses = [line.split(" ", 1)[0] for line in proc.stdout.splitlines() if line.strip()]
    return bool(statuses) and all(status == "-" for status in statuses)


def _terminal_receipt_keys(receipt_dir: Path) -> set[str]:
    """Return idempotency keys whose receipts are in a terminal state."""
    return set(_terminal_receipts_by_key(receipt_dir))


def _terminal_receipts_by_key(receipt_dir: Path) -> dict[str, dict[str, Any]]:
    """Return terminal receipts keyed by idempotency key."""
    receipts: dict[str, dict[str, Any]] = {}
    for path in _list_json(receipt_dir):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in ("published", "already_satisfied", "completed", "skipped"):
            key = str(payload.get("idempotency_key") or path.stem).strip()
            if key:
                receipts[key] = payload
    return receipts


def _mapping_from_action(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return parsed

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _local_evidence_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _branch_from_payload(payload: dict[str, Any]) -> str:
    """Extract a branch from outbox payloads with historical shape drift."""
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch

    branch = str(payload.get("branch") or "").strip()
    if branch:
        return branch

    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is not None:
        return str(requested_action.get("branch") or "").strip()
    return ""


def _head_from_payload(payload: dict[str, Any]) -> str:
    """Extract the branch head SHA from outbox payloads when present."""
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        head = str(
            local_evidence.get("head_sha")
            or local_evidence.get("head")
            or local_evidence.get("commit")
            or ""
        ).strip()
        if head:
            return head

    for key in ("head_sha", "head", "commit"):
        head = str(payload.get(key) or "").strip()
        if head:
            return head
    return ""


def _desired_head_from_payload(payload: dict[str, Any]) -> str:
    """Extract the requested branch head SHA from outbox payloads when present."""
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        head = str(
            local_evidence.get("desired_head_sha")
            or local_evidence.get("head_sha")
            or local_evidence.get("head")
            or local_evidence.get("commit")
            or ""
        ).strip()
        if head:
            return head

    for key in ("desired_head_sha", "head_sha", "head", "commit"):
        head = str(payload.get(key) or "").strip()
        if head:
            return head

    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is not None:
        for key in ("desired_head_sha", "head_sha", "head", "commit"):
            head = str(requested_action.get(key) or "").strip()
            if head:
                return head
    return ""


def _heads_match(expected: str, actual: str) -> bool:
    expected_value = expected.strip().lower()
    actual_value = actual.strip().lower()
    if len(expected_value) < 7 or len(actual_value) < 7:
        return False
    return actual_value.startswith(expected_value) or expected_value.startswith(actual_value)


def _git_ref_head(root: Path, ref: str) -> str:
    proc = run_git(["rev-parse", "--verify", ref], root, timeout=10)
    if proc.returncode != 0:
        return ""
    lines = proc.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


def _receipt_handoff_keep_reason(
    root: Path,
    payload: dict[str, Any],
    receipt: Mapping[str, Any],
    branch: str,
) -> str | None:
    """Return why a terminal receipt is not enough to archive this handoff."""

    status = str(receipt.get("status") or "").strip().lower()
    reason = str(receipt.get("reason") or "").strip().lower()
    if status != "already_satisfied" or reason != "target_open_pr":
        return None

    desired_head = _desired_head_from_payload(payload)
    if not desired_head:
        return None

    remote_ref = f"refs/remotes/origin/{branch}"
    remote_head = _git_ref_head(root, remote_ref)
    if remote_head and _heads_match(desired_head, remote_head):
        return None

    local_head = _git_ref_head(root, branch)
    if local_head and _heads_match(desired_head, local_head):
        short_desired = desired_head[:12]
        if remote_head:
            return (
                f"target_open_pr receipt exists, but origin/{branch} is "
                f"{remote_head[:12]}, not desired head {short_desired}"
            )
        return (
            f"target_open_pr receipt exists, but origin/{branch} is unavailable "
            f"and local desired head {short_desired} still needs publication"
        )
    return None


def _superseded_targets(
    outbox_payloads: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Map explicitly superseded branch heads to the active handoff replacing them."""
    targets: dict[tuple[str, str], dict[str, str]] = {}
    for path, payload in outbox_payloads:
        superseder_key = str(payload.get("idempotency_key") or path.stem).strip()
        superseder_branch = _branch_from_payload(payload)
        for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
            branch = str(
                local_evidence.get("supersedes_branch") or local_evidence.get("source_branch") or ""
            ).strip()
            head = str(
                local_evidence.get("supersedes_head_sha")
                or local_evidence.get("source_head_sha")
                or ""
            ).strip()
            if not branch or not head:
                continue
            targets[(branch, head)] = {
                "branch": superseder_branch,
                "idempotency_key": superseder_key,
                "path": str(path),
            }
    return targets


def _write_synthetic_receipt(
    *,
    receipt_dir: Path,
    outbox_payload: dict[str, Any],
    reason: str,
    pr_number: int | None,
    apply: bool,
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    key = str(outbox_payload.get("idempotency_key") or "").strip()
    if not key:
        raise ValueError("outbox payload missing idempotency_key")
    path = receipt_dir / f"{key}.json"
    body = {
        "created_issue_url": None,
        "existing_issue_url": None,
        "existing_pr_url": (
            f"https://github.com/{outbox_payload.get('repo', 'synaptent/aragora')}/pull/{pr_number}"
            if pr_number is not None
            else None
        ),
        "idempotency_key": key,
        "reason": reason,
        "recorded_at": datetime.now(UTC).isoformat(),
        "repo": outbox_payload.get("repo", "synaptent/aragora"),
        "source_file": str(outbox_payload.get("__source_file", "")),
        "status": "already_satisfied",
        "task": outbox_payload.get("task", ""),
        "synthetic": True,
        "synthetic_reason": reason,
    }
    if apply:
        path.write_text(json.dumps(body, indent=2, sort_keys=True))
    return path


def _github_open_pr_state(root: Path, repo_name: str) -> tuple[dict[str, int], bool, str]:
    """Return open codex PR heads when GitHub is healthy enough to trust."""

    try:
        health = check_github_cli_health(root)
    except Exception as exc:
        return {}, False, f"GitHub health check failed ({exc})"

    if not health.ready:
        detail = f"GitHub unavailable [{health.mode}] {health.error}".strip()
        return {}, False, detail

    try:
        open_prs = open_pr_heads(root, repo_name, "codex/")
    except Exception as exc:
        return {}, False, f"open PR fetch failed ({exc})"
    return open_prs, True, f"{len(open_prs)} open codex/* PRs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Main repository root (NOT a worktree). Outbox/receipt state "
            "is read from this path's .aragora/ subdirectory."
        ),
    )
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--repo-name", default="synaptent/aragora")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Move satisfied outbox files (default is dry-run)",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly use the default read-only dry-run mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable reconciliation result instead of human text",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=(
            "Persist a JSON reconciliation report during dry-run. Apply mode always writes "
            "the report."
        ),
    )
    parser.add_argument(
        "--out",
        "--report-path",
        dest="report_path",
        type=Path,
        default=None,
        help=(
            "Explicit JSON report path. Relative paths are resolved from --repo. "
            "Implies --write-report for dry-runs and overrides the default apply report path."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.repo).resolve()
    outbox_dir = root / ".aragora" / "automation-outbox"
    receipt_dir = root / ".aragora" / "automation-receipts"
    archive_dir = root / ".aragora" / "automation-outbox-archive"

    def emit(message: str = "") -> None:
        if not args.json:
            print(message)

    emit(f"outbox_dir: {outbox_dir}")
    emit(f"receipt_dir: {receipt_dir}")
    emit(f"archive_dir: {archive_dir} {'(will create)' if not archive_dir.exists() else ''}")
    emit(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")

    if args.apply:
        archive_dir.mkdir(parents=True, exist_ok=True)

    emit("loading existing terminal receipt keys...")
    receipt_payloads_by_key = _terminal_receipts_by_key(receipt_dir)
    receipt_keys = set(receipt_payloads_by_key)
    emit(f"  {len(receipt_keys)} terminal receipt keys")

    emit("loading outbox files...")
    outbox_files = _list_json(outbox_dir)
    emit(f"  {len(outbox_files)} outbox files\n")

    parsed_outbox_payloads: dict[Path, dict[str, Any]] = {}
    for path in outbox_files:
        payload = _load_json(path)
        if isinstance(payload, dict):
            parsed_outbox_payloads[path] = payload
    superseded_targets = _superseded_targets(list(parsed_outbox_payloads.items()))

    open_prs_cache: dict[str, int] | None = None
    open_pr_state_available = False

    def load_open_pr_state() -> tuple[dict[str, int], bool]:
        nonlocal open_prs_cache, open_pr_state_available
        if open_prs_cache is None:
            emit("loading open PR state from GitHub (one bulk call)...")
            open_prs_cache, open_pr_state_available, message = _github_open_pr_state(
                root, args.repo_name
            )
            if open_pr_state_available:
                emit(f"  {message}\n")
            else:
                emit(f"  WARN: {message}; preserving ambiguous handoffs without open-PR truth\n")
        return open_prs_cache, open_pr_state_available

    counts = {
        "satisfied_by_existing_receipt": 0,
        "blocked_receipt_pr_head_mismatch": 0,
        "satisfied_by_superseded_handoff": 0,
        "satisfied_by_landed_on_main": 0,
        "satisfied_by_open_pr_merged": 0,  # placeholder; we only know open PRs
        "still_protecting_active_work": 0,
        "missing_branch": 0,
        "blocked_missing_branch_open_pr_unknown": 0,
        "skipped_unparseable": 0,
    }

    actions: list[dict[str, Any]] = []
    for path in outbox_files:
        payload = parsed_outbox_payloads.get(path)
        if payload is None:
            counts["skipped_unparseable"] += 1
            continue
        payload["__source_file"] = str(path)
        idem = str(payload.get("idempotency_key") or "").strip()
        branch = _branch_from_payload(payload)

        if not idem or not branch:
            counts["skipped_unparseable"] += 1
            continue

        receipt = receipt_payloads_by_key.get(idem)
        if receipt is not None:
            keep_reason = _receipt_handoff_keep_reason(root, payload, receipt, branch)
            if keep_reason is not None:
                counts["blocked_receipt_pr_head_mismatch"] += 1
                counts["still_protecting_active_work"] += 1
                actions.append(
                    {
                        "path": str(path),
                        "branch": branch,
                        "decision": "keep",
                        "reason": keep_reason,
                        "synthetic_receipt": False,
                    }
                )
                continue
            counts["satisfied_by_existing_receipt"] += 1
            actions.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "decision": "archive",
                    "reason": "matching receipt exists",
                    "synthetic_receipt": False,
                }
            )
            if args.apply:
                shutil.move(str(path), str(archive_dir / path.name))
            continue

        head = _head_from_payload(payload)
        superseder = superseded_targets.get((branch, head)) if head else None
        if superseder is not None and superseder["idempotency_key"] != idem:
            reason = f"superseded by active handoff {superseder['idempotency_key']}"
            counts["satisfied_by_superseded_handoff"] += 1
            actions.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "decision": "archive",
                    "reason": reason,
                    "superseded_by": superseder,
                    "synthetic_receipt": True,
                }
            )
            if args.apply:
                _write_synthetic_receipt(
                    receipt_dir=receipt_dir,
                    outbox_payload=payload,
                    reason=reason,
                    pr_number=None,
                    apply=True,
                )
                shutil.move(str(path), str(archive_dir / path.name))
            continue

        try:
            ref_proc = run_git(["rev-parse", "--verify", branch], root, timeout=10)
        except Exception:
            ref_proc = None
        if ref_proc is None or ref_proc.returncode != 0:
            open_prs, open_pr_state_available = load_open_pr_state()
            if branch in open_prs:
                counts["still_protecting_active_work"] += 1
                actions.append(
                    {
                        "path": str(path),
                        "branch": branch,
                        "decision": "keep",
                        "reason": f"branch missing locally but has open PR #{open_prs[branch]}",
                        "synthetic_receipt": False,
                    }
                )
                continue
            if not open_pr_state_available:
                counts["blocked_missing_branch_open_pr_unknown"] += 1
                actions.append(
                    {
                        "path": str(path),
                        "branch": branch,
                        "decision": "keep",
                        "reason": (
                            "branch no longer exists locally, but open PR state is unavailable"
                        ),
                        "synthetic_receipt": False,
                    }
                )
                continue

            counts["missing_branch"] += 1
            actions.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "decision": "archive",
                    "reason": "branch no longer exists",
                    "synthetic_receipt": True,
                }
            )
            if args.apply:
                _write_synthetic_receipt(
                    receipt_dir=receipt_dir,
                    outbox_payload=payload,
                    reason="branch no longer exists locally",
                    pr_number=None,
                    apply=True,
                )
                shutil.move(str(path), str(archive_dir / path.name))
            continue

        if _branch_has_landed_on_main(root, args.base, branch):
            counts["satisfied_by_landed_on_main"] += 1
            actions.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "decision": "archive",
                    "reason": "branch work landed on main (merge or patch-equivalent)",
                    "synthetic_receipt": True,
                }
            )
            if args.apply:
                _write_synthetic_receipt(
                    receipt_dir=receipt_dir,
                    outbox_payload=payload,
                    reason="branch work landed on main (merge or patch-equivalent)",
                    pr_number=None,
                    apply=True,
                )
                shutil.move(str(path), str(archive_dir / path.name))
            continue

        open_prs, _open_pr_state_available = load_open_pr_state()
        if branch in open_prs:
            counts["still_protecting_active_work"] += 1
            actions.append(
                {
                    "path": str(path),
                    "branch": branch,
                    "decision": "keep",
                    "reason": f"branch has open PR #{open_prs[branch]}",
                    "synthetic_receipt": False,
                }
            )
            continue

        counts["still_protecting_active_work"] += 1
        actions.append(
            {
                "path": str(path),
                "branch": branch,
                "decision": "keep",
                "reason": "branch has unique commits not on main, no open PR — actively protecting",
                "synthetic_receipt": False,
            }
        )

    emit("\n--- summary ---")
    for k, v in counts.items():
        emit(f"  {k:>40}: {v}")
    archived = sum(1 for a in actions if a["decision"] == "archive")
    kept = sum(1 for a in actions if a["decision"] == "keep")
    emit(f"\n  total: {archived} archived, {kept} kept")

    should_write_report = args.apply or args.write_report or args.report_path is not None
    report_path: Path | None = None
    if should_write_report:
        if args.report_path is not None:
            out = root / args.report_path
        else:
            state_dir = root / ".aragora" / "cleanup-state"
            out = (
                state_dir
                / f"outbox-reconciliation-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        report_path = out
        out.write_text(
            json.dumps(
                {"counts": counts, "actions": actions, "applied": args.apply},
                indent=2,
                sort_keys=True,
            )
        )
        emit(f"\n  report: {out}")
    else:
        emit("\n  report: not written in dry-run; pass --write-report to persist one.")
    if not args.apply:
        emit("\n  DRY-RUN — re-run with --apply to actually archive files.")
    if args.json:
        print(
            json.dumps(
                {
                    "actions": actions,
                    "applied": args.apply,
                    "archive_dir": str(archive_dir),
                    "archived": archived,
                    "base": args.base,
                    "counts": counts,
                    "dry_run": not args.apply,
                    "kept": kept,
                    "outbox_count": len(outbox_files),
                    "outbox_dir": str(outbox_dir),
                    "receipt_dir": str(receipt_dir),
                    "repo": str(root),
                    "repo_name": args.repo_name,
                    "report": str(report_path) if report_path is not None else None,
                    "terminal_receipt_count": len(receipt_keys),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
