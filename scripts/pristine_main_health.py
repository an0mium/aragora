#!/usr/bin/env python3
"""Nightly pristine-origin/main health run — the hidden-red countermeasure (epic #9039, issue #9043).

Path-gated CI shards can leave main red without any signal, and session
worktrees inflate failures ~7x, so this script runs the test suite in a
PRISTINE worktree checked out from origin/main (never a session worktree).
On red it writes the same halt marker ``scripts/merge_executor.py`` re-checks
before EVERY merge (``.aragora/merge_executor.halt``) — a human deletes the
marker after verifying main is green. Results append to the throughput ledger.

Read-only vs GitHub. Intended to run nightly under launchd (installation of
the LaunchAgent is an operator action); safe to run by hand:

    python3 scripts/pristine_main_health.py --suite required   # fast lane
    python3 scripts/pristine_main_health.py --suite full       # full shards
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aragora.nomic.throughput import LedgerRecord, ThroughputLedger  # noqa: E402

DEFAULT_PRISTINE_DIR = Path.home() / ".aragora" / "pristine-main"
DEFAULT_HALT_FILE = _REPO_ROOT / ".aragora" / "merge_executor.halt"
OWNER_MARKER_PURPOSE = "aragora.pristine_main_health"

# Suite commands run INSIDE the pristine worktree. "required" mirrors the
# required-check lane; "full" runs the whole suite including path-gated shards
# (the ones that go silently red) minus the known-broken collection module.
SUITES: dict[str, list[list[str]]] = {
    "required": [["make", "ci-required"]],
    "full": [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "no:cacheprovider",
            "--ignore=tests/connectors",
        ]
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _canon(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _owner_marker_path(pristine: Path) -> Path:
    return pristine.with_name(f"{pristine.name}.owner.json")


def _write_owner_marker(repo: Path, pristine: Path) -> None:
    marker = {
        "purpose": OWNER_MARKER_PURPOSE,
        "repo_root": _canon(repo),
        "pristine_dir": _canon(pristine),
        "written_at": _now_iso(),
        "written_by": "scripts/pristine_main_health.py",
    }
    _owner_marker_path(pristine).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _registered_worktree_paths(repo: Path) -> set[str]:
    listed = _run(["git", "worktree", "list", "--porcelain"], cwd=repo, timeout=60)
    if listed.returncode != 0:
        sys.exit(f"git worktree list failed: {listed.stderr.strip()[:300]}")
    paths: set[str] = set()
    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            paths.add(_canon(Path(line.removeprefix("worktree "))))
    return paths


def _verify_script_owned_pristine(repo: Path, pristine: Path) -> None:
    marker_path = _owner_marker_path(pristine)
    if not marker_path.exists():
        sys.exit(
            "refusing to destructively refresh unmarked --pristine-dir "
            f"{pristine}; remove it or recreate it with this script"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"refusing to refresh {pristine}: invalid owner marker {marker_path}: {exc}")

    if (
        marker.get("purpose") != OWNER_MARKER_PURPOSE
        or marker.get("repo_root") != _canon(repo)
        or marker.get("pristine_dir") != _canon(pristine)
    ):
        sys.exit(f"refusing to refresh {pristine}: owner marker {marker_path} does not match")

    if _canon(pristine) not in _registered_worktree_paths(repo):
        sys.exit(f"refusing to refresh {pristine}: not a registered worktree for {repo}")


def refresh_pristine_worktree(repo: Path, pristine: Path) -> str:
    """Create or hard-refresh a detached pristine worktree at origin/main.

    Returns the checked-out commit SHA. This directory is owned by this
    script; it is never a session worktree and holds no uncommitted work.
    """
    # Explicit refspec: plain "fetch origin main" only guarantees FETCH_HEAD,
    # so the later origin/main checkout could test a stale ref and miss a
    # newly red main (#9058 openai [P1]).
    fetch = _run(
        ["git", "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"],
        cwd=repo,
        timeout=300,
    )
    if fetch.returncode != 0:
        sys.exit(f"git fetch failed: {fetch.stderr.strip()[:300]}")
    if not (pristine / ".git").exists():
        pristine.parent.mkdir(parents=True, exist_ok=True)
        add = _run(
            ["git", "worktree", "add", "--detach", str(pristine), "origin/main"],
            cwd=repo,
            timeout=300,
        )
        if add.returncode != 0:
            sys.exit(f"git worktree add failed: {add.stderr.strip()[:300]}")
        _write_owner_marker(repo, pristine)
    else:
        _verify_script_owned_pristine(repo, pristine)
        for cmd in (
            ["git", "checkout", "--detach", "origin/main"],
            ["git", "reset", "--hard", "origin/main"],
            ["git", "clean", "-fdx", "--quiet"],
        ):
            proc = _run(cmd, cwd=pristine, timeout=300)
            if proc.returncode != 0:
                sys.exit(f"pristine refresh failed ({' '.join(cmd)}): {proc.stderr.strip()[:300]}")
    sha = _run(["git", "rev-parse", "HEAD"], cwd=pristine, timeout=60)
    return sha.stdout.strip()


def write_halt_marker(halt_file: Path, *, sha: str, failures: list[str]) -> None:
    """Same marker shape merge_executor honors; human deletes after verifying green."""
    halt_file.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "reason": "main_red",
        "repo": "pristine-main-health",
        "details": [f"pristine origin/main {sha[:12]} red"] + failures[:10],
        "written_at": _now_iso(),
        "written_by": "scripts/pristine_main_health.py",
        "re_arm": "a human deletes this file after verifying main is green",
    }
    halt_file.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--pristine-dir", type=Path, default=DEFAULT_PRISTINE_DIR)
    parser.add_argument("--halt-file", type=Path, default=DEFAULT_HALT_FILE)
    parser.add_argument("--suite", choices=sorted(SUITES), default="full")
    parser.add_argument("--timeout-minutes", type=int, default=180, help="per-command timeout")
    parser.add_argument(
        "--no-halt-file",
        action="store_true",
        help="report only; never write the halt marker (for manual diagnosis)",
    )
    args = parser.parse_args(argv)

    sha = refresh_pristine_worktree(args.repo_root, args.pristine_dir)
    print(f"pristine origin/main at {sha[:12]} in {args.pristine_dir}")

    failures: list[str] = []
    for cmd in SUITES[args.suite]:
        print(f"running: {' '.join(cmd)}")
        try:
            proc = _run(cmd, cwd=args.pristine_dir, timeout=args.timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            failures.append(f"TIMEOUT after {args.timeout_minutes}m: {' '.join(cmd)}")
            continue
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
            failures.append(f"exit {proc.returncode}: {' '.join(cmd)}\n{tail}")

    green = not failures

    # SAFETY ORDER: on red, write the halt marker BEFORE any bookkeeping so an
    # unwritable ledger can never fail open (#9058 openai [P2] round 2).
    if not green and not args.no_halt_file:
        write_halt_marker(args.halt_file, sha=sha, failures=failures)

    try:
        ledger = ThroughputLedger(args.repo_root)
        ledger.append(
            LedgerRecord(
                kind="note",
                timestamp=_now_iso(),
                data={
                    "event": "pristine_main_health",
                    "sha": sha,
                    "suite": args.suite,
                    "green": green,
                    "failures": [f.splitlines()[0] for f in failures],
                },
            )
        )
    except OSError as exc:
        print(f"warning: ledger append failed: {exc}", file=sys.stderr)

    if green:
        print(f"GREEN: pristine main {sha[:12]} passed suite '{args.suite}'")
        return 0

    print(f"RED: pristine main {sha[:12]} failed suite '{args.suite}':", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    if args.no_halt_file:
        print("halt marker NOT written (--no-halt-file)", file=sys.stderr)
    else:
        print(f"halt marker written: {args.halt_file}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
