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
  2. For each, checks: matching receipt exists? matching PR merged?
  3. Archives satisfied outbox files to .aragora/automation-outbox-archive/
     and writes a synthetic receipt if needed (so future audits stay correct)
  4. Reports counts before/after

Read-only by default (--dry-run); pass --apply to actually move files.
Dry-run reports are printed to stdout; pass --write-report to persist a JSON report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_codex_branch_backlog import (  # noqa: E402
    open_pr_heads,
    run_git,
)

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
    keys: set[str] = set()
    for path in _list_json(receipt_dir):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in ("published", "already_satisfied", "completed", "skipped"):
            key = str(payload.get("idempotency_key") or path.stem).strip()
            if key:
                keys.add(key)
    return keys


def _branch_from_payload(payload: dict[str, Any]) -> str:
    """Extract a branch from outbox payloads with historical shape drift."""
    local_evidence = payload.get("local_evidence")
    if isinstance(local_evidence, dict):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch
    return str(payload.get("branch") or "").strip()


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
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move satisfied outbox files (default is dry-run)",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=(
            "Persist a JSON reconciliation report during dry-run. Apply mode always writes "
            "the report."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.repo).resolve()
    outbox_dir = root / ".aragora" / "automation-outbox"
    receipt_dir = root / ".aragora" / "automation-receipts"
    archive_dir = root / ".aragora" / "automation-outbox-archive"

    print(f"outbox_dir: {outbox_dir}")
    print(f"receipt_dir: {receipt_dir}")
    print(f"archive_dir: {archive_dir} {'(will create)' if not archive_dir.exists() else ''}")
    print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")

    if args.apply:
        archive_dir.mkdir(parents=True, exist_ok=True)

    print("loading existing terminal receipt keys...")
    receipt_keys = _terminal_receipt_keys(receipt_dir)
    print(f"  {len(receipt_keys)} terminal receipt keys")

    print("loading outbox files...")
    outbox_files = _list_json(outbox_dir)
    print(f"  {len(outbox_files)} outbox files\n")

    print("loading open PR state from GitHub (one bulk call)...")
    try:
        open_prs = open_pr_heads(root, args.repo_name, "codex/")
        print(f"  {len(open_prs)} open codex/* PRs\n")
    except Exception as exc:
        print(f"  WARN: open PR fetch failed ({exc}); proceeding without open-PR check\n")
        open_prs = {}

    counts = {
        "satisfied_by_existing_receipt": 0,
        "satisfied_by_landed_on_main": 0,
        "satisfied_by_open_pr_merged": 0,  # placeholder; we only know open PRs
        "still_protecting_active_work": 0,
        "missing_branch": 0,
        "skipped_unparseable": 0,
    }

    actions: list[dict[str, Any]] = []
    for path in outbox_files:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            counts["skipped_unparseable"] += 1
            continue
        payload["__source_file"] = str(path)
        idem = str(payload.get("idempotency_key") or "").strip()
        branch = _branch_from_payload(payload)

        if not idem or not branch:
            counts["skipped_unparseable"] += 1
            continue

        if idem in receipt_keys:
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

        try:
            ref_proc = run_git(["rev-parse", "--verify", branch], root, timeout=10)
        except Exception:
            ref_proc = None
        if ref_proc is None or ref_proc.returncode != 0:
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

    print("\n--- summary ---")
    for k, v in counts.items():
        print(f"  {k:>40}: {v}")
    archived = sum(1 for a in actions if a["decision"] == "archive")
    kept = sum(1 for a in actions if a["decision"] == "keep")
    print(f"\n  total: {archived} archived, {kept} kept")

    should_write_report = args.apply or args.write_report
    if should_write_report:
        state_dir = root / ".aragora" / "cleanup-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        out = (
            state_dir / f"outbox-reconciliation-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
        )
        out.write_text(
            json.dumps(
                {"counts": counts, "actions": actions, "applied": args.apply},
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\n  report: {out}")
    else:
        print("\n  report: not written in dry-run; pass --write-report to persist one.")
    if not args.apply:
        print("\n  DRY-RUN — re-run with --apply to actually archive files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
