#!/usr/bin/env python3
"""Fail closed unless the shared root is clean on local origin/main.

This guard reads local git state only. It does not fetch, reset, restore, clean,
stash, switch branches, or mutate local automation configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from root_guarded_queue import _run, _snapshot  # noqa: E402


OK = "ok_clean_on_main"
BLOCKED = "blocked_root_not_clean_on_main"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refuse root use unless shared root is clean on local origin/main.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help=(
            "Shared root checkout path. Defaults to ARAGORA_CANONICAL_ROOT or the "
            "primary worktree from git worktree list."
        ),
    )
    parser.add_argument("--base", default="main", help="Expected branch name.")
    parser.add_argument("--remote", default="origin", help="Remote name for comparison.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    return parser


def _primary_worktree(cwd: Path) -> Path:
    result = _run(["git", "worktree", "list", "--porcelain"], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "git worktree list failed")
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree ")).resolve()
    raise RuntimeError("git worktree list did not report a primary worktree")


def _canonical_root(override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    env_root = os.environ.get("ARAGORA_CANONICAL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _primary_worktree(Path.cwd())


def _ref_exists(cwd: Path, ref: str) -> bool:
    result = _run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=cwd)
    return result.returncode == 0 and bool(result.stdout)


def _git_path_exists(cwd: Path, name: str) -> bool:
    result = _run(["git", "rev-parse", "--git-path", name], cwd=cwd)
    if result.returncode != 0 or not result.stdout:
        return False
    git_path = Path(result.stdout)
    if not git_path.is_absolute():
        git_path = cwd / git_path
    return git_path.exists()


def _operation_state_reasons(cwd: Path) -> list[str]:
    reasons: list[str] = []
    for name in ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD"):
        if _git_path_exists(cwd, name):
            reasons.append(f"merge/rebase/cherry-pick state present: {name}")
    for name in ("rebase-merge", "rebase-apply"):
        if _git_path_exists(cwd, name):
            reasons.append(f"merge/rebase/cherry-pick state present: {name}")
    return reasons


def _rev_parse(cwd: Path, ref: str) -> str | None:
    result = _run(["git", "rev-parse", ref], cwd=cwd)
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _ahead_behind(cwd: Path, left_ref: str, right_ref: str) -> tuple[int, int] | None:
    result = _run(
        ["git", "rev-list", "--left-right", "--count", f"{left_ref}...{right_ref}"],
        cwd=cwd,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    left, right = result.stdout.split()
    return int(left), int(right)


def check_root_clean_on_main(args: argparse.Namespace) -> dict[str, Any]:
    canonical_root = _canonical_root(args.canonical_root)
    snapshot = _snapshot(canonical_root)
    base_ref = f"{args.remote}/{args.base}"
    reasons: list[str] = []

    if snapshot.branch != args.base:
        reasons.append(f"branch is {snapshot.branch}, expected {args.base}")
    if snapshot.dirty_paths:
        reasons.append(f"dirty root: {', '.join(snapshot.dirty_paths)}")
    reasons.extend(_operation_state_reasons(canonical_root))

    base_head: str | None = None
    if not _ref_exists(canonical_root, base_ref):
        reasons.append(f"missing ref {base_ref}")
    else:
        base_head = _rev_parse(canonical_root, base_ref)
        if snapshot.head != base_head:
            counts = _ahead_behind(canonical_root, "HEAD", base_ref)
            if counts is None:
                reasons.append(f"HEAD differs from {base_ref}")
            else:
                ahead, behind = counts
                reasons.append(f"HEAD differs from {base_ref}: ahead={ahead} behind={behind}")

    ok = not reasons
    return {
        "version": "assert_root_clean_on_main.v1",
        "ok": ok,
        "status": OK if ok else BLOCKED,
        "canonical_root": str(canonical_root),
        "branch": snapshot.branch,
        "head": snapshot.head,
        "base_ref": base_ref,
        "base_head": base_head,
        "dirty_paths": snapshot.dirty_paths,
        "status_lines": snapshot.status_lines,
        "reasons": reasons,
    }


def _print_text(report: dict[str, Any]) -> None:
    if report["ok"]:
        print(
            f"OK: {report['canonical_root']} is clean on {report['branch']} == {report['base_ref']}"
        )
        return
    print(f"BLOCKED: {report['canonical_root']} is not clean on {report['base_ref']}")
    for reason in report["reasons"]:
        print(f"- {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = check_root_clean_on_main(args)
    except Exception as exc:  # pragma: no cover - exercised by CLI/system failures
        report = {
            "version": "assert_root_clean_on_main.v1",
            "ok": False,
            "status": "error",
            "canonical_root": str(args.canonical_root.resolve()) if args.canonical_root else None,
            "reasons": [str(exc)],
        }
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
