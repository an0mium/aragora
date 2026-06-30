#!/usr/bin/env python3
"""Fail closed when a git-mutating automation is running from shared root.

This guard is intentionally diagnostic only. It never switches branches, stashes,
cleans, resets, or edits local automation configuration.
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

from root_guarded_queue import _run  # noqa: E402


BLOCKED = "blocked_shared_root"
OK = "ok_linked_worktree"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refuse git-mutating work from the canonical shared root checkout.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Directory to classify; defaults to the current working directory.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help=(
            "Override the shared root checkout path. Defaults to "
            "ARAGORA_CANONICAL_ROOT or the primary worktree from git worktree list."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    return parser


def _git_toplevel(cwd: Path) -> Path:
    result = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr or f"{cwd} is not inside a git worktree")
    return Path(result.stdout).resolve()


def _primary_worktree(cwd: Path) -> Path:
    result = _run(["git", "worktree", "list", "--porcelain"], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "git worktree list failed")
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree ")).resolve()
    raise RuntimeError("git worktree list did not report a primary worktree")


def _canonical_root(cwd: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    env_root = os.environ.get("ARAGORA_CANONICAL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _primary_worktree(cwd)


def check_not_root_checkout(args: argparse.Namespace) -> dict[str, Any]:
    cwd = (args.cwd or Path.cwd()).resolve()
    repo_root = _git_toplevel(cwd)
    canonical_root = _canonical_root(repo_root, args.canonical_root)
    is_shared_root = repo_root == canonical_root
    reasons = ["current git toplevel is the shared root checkout"] if is_shared_root else []
    return {
        "version": "assert_not_root_checkout.v1",
        "ok": not is_shared_root,
        "status": BLOCKED if is_shared_root else OK,
        "cwd": str(cwd),
        "repo_root": str(repo_root),
        "canonical_root": str(canonical_root),
        "reasons": reasons,
    }


def _print_text(report: dict[str, Any]) -> None:
    if report["ok"]:
        print(f"OK: {report['repo_root']} is not the shared root checkout")
        return
    print(f"BLOCKED: {report['repo_root']} is the shared root checkout")
    for reason in report["reasons"]:
        print(f"- {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = check_not_root_checkout(args)
    except Exception as exc:  # pragma: no cover - exercised by CLI/system failures
        report = {
            "version": "assert_not_root_checkout.v1",
            "ok": False,
            "status": "error",
            "cwd": str((args.cwd or Path.cwd()).resolve()),
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
