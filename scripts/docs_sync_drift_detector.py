#!/usr/bin/env python3
"""Docs-site sync drift detector - a bounded, single-shot governed loop iteration.

Root cause being closed (Loop Control Plane follow-up): generated docs-site
mirrors drift on ``main`` when a source-doc PR's advisory "Build Documentation
(PR Check)" run is cancelled by the external canceller documented in
``docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md`` and never re-run (observed
escapes: #7829, #7814). The next doc-touching PR then inherits a red docs check
it did not cause.

Each invocation is one bounded iteration (launchd provides the cadence):

1. Fetch the base ref and regenerate the docs surface inside a throwaway
   detached worktree (``python scripts/doc_stats.py --write`` +
   ``node docs-site/scripts/sync-docs.js`` - the exact commands CI runs).
2. If nothing drifted: report ``clean``.
3. If only generated mirrors under ``docs-site/docs/`` drifted: in ``--apply``
   mode open (at most) one sync PR that settles through the normal
   model-quorum merge gate; otherwise just report ``drift_detected``.
4. If anything *outside* the generated-mirror allowlist drifted (for example
   ``doc_stats`` stamp targets such as protected ``CLAUDE.md``): fail closed
   to report-only (``drift_outside_allowlist``) - never auto-PR those.

Hard bounds and guarantees: single iteration per invocation, per-subprocess
timeout, at most one open sync PR (branch namespace ``bot/docs-site-sync``),
no force-push, and the detector never merges, approves, comments, or reruns
anything. Status is written atomically to ``.aragora/docs_drift_status.json``
for the Loop Control Plane collector (``aragora/swarm/loop_control_io.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_SCHEMA = "docs-drift/v1"
BRANCH_PREFIX = "bot/docs-site-sync"
MIRROR_ALLOWLIST_PREFIX = "docs-site/docs/"
DEFAULT_BASE_REF = "origin/main"
DEFAULT_STATUS_RELPATH = Path(".aragora") / "docs_drift_status.json"
DEFAULT_TIMEOUT_S = 300.0

OUTCOME_CLEAN = "clean"
OUTCOME_DRIFT_DETECTED = "drift_detected"
OUTCOME_DRIFT_PR_OPEN = "drift_pr_open"
OUTCOME_DRIFT_PR_OPENED = "drift_pr_opened"
OUTCOME_OUTSIDE_ALLOWLIST = "drift_outside_allowlist"
OUTCOME_ERROR = "error"

# Outcomes that count toward the no-progress fault streak. Waiting on an open
# sync PR is *waiting*, not a fault (the #7879 fault-vs-waiting distinction).
FAULT_OUTCOMES = frozenset({OUTCOME_ERROR, OUTCOME_OUTSIDE_ALLOWLIST})

EXIT_BY_OUTCOME = {
    OUTCOME_CLEAN: 0,
    OUTCOME_DRIFT_PR_OPEN: 0,
    OUTCOME_DRIFT_PR_OPENED: 0,
    OUTCOME_DRIFT_DETECTED: 1,
    OUTCOME_OUTSIDE_ALLOWLIST: 2,
    OUTCOME_ERROR: 2,
}


class DetectorError(RuntimeError):
    """Operational failure of one iteration (fail-closed to ``error``)."""


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Single subprocess seam (tests fake this to prove the command surface)."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _run_ok(
    cmd: list[str], cwd: Path, timeout: float, what: str
) -> subprocess.CompletedProcess[str]:
    proc = _run(cmd, cwd, timeout)
    if proc is None:
        raise DetectorError(f"{what} failed to execute: {' '.join(cmd)}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else ""
        raise DetectorError(f"{what} failed (rc={proc.returncode}): {tail}")
    return proc


def parse_porcelain(data: str) -> list[str]:
    """Paths from ``git status --porcelain -z`` output (NUL-delimited, deduped).

    The ``-z`` form never quotes or octal-escapes paths, so unicode and
    special characters survive verbatim. Rename/copy entries carry the
    destination in the status token and the *original* path in the next
    NUL token, which is consumed and ignored.
    """
    paths: set[str] = set()
    tokens = data.split("\0")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 4 or token[2] != " ":
            continue
        if token[0] in "RC" and index < len(tokens):
            index += 1
        entry = token[3:]
        if entry:
            paths.add(entry)
    return sorted(paths)


def partition_drift(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split drifted paths into (allowlisted mirrors, everything else)."""
    mirrors = [p for p in paths if p.startswith(MIRROR_ALLOWLIST_PREFIX)]
    other = [p for p in paths if not p.startswith(MIRROR_ALLOWLIST_PREFIX)]
    return mirrors, other


def next_consecutive_errors(previous: int, outcome: str) -> int:
    return previous + 1 if outcome in FAULT_OUTCOMES else 0


def load_previous_status(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_status_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def find_open_sync_pr(repo_root: Path, timeout: float) -> str | None:
    proc = _run_ok(
        ["gh", "pr", "list", "--state", "open", "--limit", "100", "--json", "url,headRefName"],
        repo_root,
        timeout,
        "gh pr list",
    )
    try:
        items = json.loads(proc.stdout or "[]")
    except (json.JSONDecodeError, ValueError) as exc:
        raise DetectorError(f"gh pr list returned invalid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise DetectorError("gh pr list returned non-list JSON")
    for item in items:
        if not isinstance(item, dict):
            continue
        head = str(item.get("headRefName") or "")
        if head.startswith(BRANCH_PREFIX):
            url = str(item.get("url") or "")
            return url or None
    return None


def regenerate_docs(worktree: Path, timeout: float) -> None:
    _run_ok(
        [sys.executable, "scripts/doc_stats.py", "--write"],
        worktree,
        timeout,
        "doc_stats --write",
    )
    _run_ok(
        ["node", "docs-site/scripts/sync-docs.js"],
        worktree,
        timeout,
        "sync-docs",
    )


def open_sync_pr(
    repo_root: Path,
    worktree: Path,
    mirrors: list[str],
    base_sha: str,
    remote: str,
    timeout: float,
) -> str:
    branch = f"{BRANCH_PREFIX}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    _run_ok(["git", "checkout", "-b", branch], worktree, timeout, "git checkout -b")
    _run_ok(["git", "add", "--", *mirrors], worktree, timeout, "git add")
    commit_message = (
        "chore(docs-site): sync generated doc mirrors (drift detector)\n\n"
        "Mechanical regeneration of generated docs-site mirrors that drifted on\n"
        f"main ({base_sha}); produced by the exact commands CI runs\n"
        "(python scripts/doc_stats.py --write && node docs-site/scripts/sync-docs.js).\n"
        "Opened by scripts/docs_sync_drift_detector.py; settles through the normal\n"
        "model-quorum merge gate. See docs/governance/LOOP_CONTROL_PLANE.md and\n"
        "docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md.\n\n"
        "Co-authored-by: claude[bot] <claude[bot]@users.noreply.github.com>"
    )
    _run_ok(["git", "commit", "-m", commit_message], worktree, timeout, "git commit")
    _run_ok(["git", "push", "-u", remote, branch], worktree, timeout, "git push")
    drifted_lines = "\n".join(f"- `{p}`" for p in mirrors)
    body = (
        f"## Summary\n\n"
        f"Generated docs-site mirrors drifted on main (`{base_sha}`). This PR is the\n"
        f"mechanical regeneration produced by the exact commands the\n"
        f"`Build Documentation (PR Check)` workflow runs:\n\n"
        f"```\npython scripts/doc_stats.py --write\nnode docs-site/scripts/sync-docs.js\n```\n\n"
        f"Drifted mirrors:\n\n{drifted_lines}\n\n"
        f"## Why this exists\n\n"
        f"Source-doc PRs whose advisory docs check is cancelled by the external\n"
        f"canceller (see `docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md`; observed\n"
        f"escapes #7829, #7814) can land source changes without regenerated mirrors,\n"
        f"making the *next* doc-touching PR inherit a red docs check. The drift\n"
        f"detector (`scripts/docs_sync_drift_detector.py`, a governed loop in\n"
        f"`docs/governance/LOOP_CONTROL_PLANE.md`) opens at most one such sync PR.\n\n"
        f"## Guarantees\n\n"
        f"- Generated-mirror files only (`docs-site/docs/**`); anything else fails closed.\n"
        f"- The detector never merges, approves, or comments; this PR settles through\n"
        f"  the normal model-quorum merge gate.\n"
    )
    proc = _run_ok(
        [
            "gh",
            "pr",
            "create",
            "--title",
            "chore(docs-site): sync generated doc mirrors (drift detector)",
            "--body",
            body,
        ],
        worktree,
        timeout,
        "gh pr create",
    )
    url = (proc.stdout or "").strip().splitlines()
    return url[-1].strip() if url else ""


def cleanup_worktree(repo_root: Path, worktree: Path, timeout: float) -> None:
    _run(["git", "worktree", "remove", "--force", str(worktree)], repo_root, timeout)
    _run(["git", "worktree", "prune"], repo_root, timeout)


def run_iteration(repo_root: Path, *, apply: bool, base_ref: str, timeout: float) -> dict[str, Any]:
    started = time.time()
    scratch: Path | None = None
    worktree: Path | None = None
    pr_url: str | None = None
    error: str | None = None
    base_sha: str | None = None
    mirrors: list[str] = []
    other: list[str] = []
    outcome = OUTCOME_ERROR
    try:
        remote, _, branch = base_ref.partition("/")
        if not remote or not branch:
            raise DetectorError(f"base ref must look like 'origin/main', got {base_ref!r}")
        _run_ok(["git", "fetch", "--quiet", remote, branch], repo_root, timeout, "git fetch")
        base_sha = _run_ok(
            ["git", "rev-parse", base_ref], repo_root, timeout, "git rev-parse"
        ).stdout.strip()
        scratch = Path(tempfile.mkdtemp(prefix="docs-drift-wt-"))
        worktree = scratch / "worktree"
        _run_ok(
            ["git", "worktree", "add", "--detach", str(worktree), base_ref],
            repo_root,
            timeout,
            "git worktree add",
        )
        regenerate_docs(worktree, timeout)
        drifted = parse_porcelain(
            _run_ok(["git", "status", "--porcelain", "-z"], worktree, timeout, "git status").stdout
        )
        mirrors, other = partition_drift(drifted)
        if other:
            outcome = OUTCOME_OUTSIDE_ALLOWLIST
            shown = ", ".join(other[:10])
            error = f"drift outside generated-mirror allowlist (report-only): {shown}"
        elif not mirrors:
            outcome = OUTCOME_CLEAN
        elif not apply:
            outcome = OUTCOME_DRIFT_DETECTED
        else:
            existing = find_open_sync_pr(repo_root, timeout)
            if existing:
                pr_url = existing
                outcome = OUTCOME_DRIFT_PR_OPEN
            else:
                pr_url = open_sync_pr(repo_root, worktree, mirrors, base_sha, remote, timeout)
                outcome = OUTCOME_DRIFT_PR_OPENED
    except DetectorError as exc:
        outcome = OUTCOME_ERROR
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - one iteration must fail closed, not raise
        outcome = OUTCOME_ERROR
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if worktree is not None:
                cleanup_worktree(repo_root, worktree, timeout)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
    return {
        "outcome": outcome,
        "error": error,
        "base_sha": base_sha,
        "pr_url": pr_url,
        "drifted_mirror_paths": mirrors,
        "drifted_other_paths": other,
        "duration_s": round(time.time() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Open (at most) one sync PR when only generated mirrors drifted",
    )
    parser.add_argument("--json", action="store_true", help="Print the status payload as JSON")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Status JSON path (default: <repo-root>/.aragora/docs_drift_status.json)",
    )
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF, help="Base ref to audit")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-subprocess timeout in seconds",
    )
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    status_path = args.status_path or (repo_root / DEFAULT_STATUS_RELPATH)
    previous = load_previous_status(status_path)
    try:
        previous_errors = int(previous.get("consecutive_errors", 0) or 0)
    except (TypeError, ValueError):
        previous_errors = 0

    result = run_iteration(
        repo_root, apply=args.apply, base_ref=args.base_ref, timeout=args.timeout
    )
    outcome = result["outcome"]
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "apply": bool(args.apply),
        "base_ref": args.base_ref,
        "consecutive_errors": next_consecutive_errors(previous_errors, outcome),
        **result,
    }
    write_status_atomic(status_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"docs-sync drift: {outcome}")
        if payload.get("error"):
            print(f"  error: {payload['error']}")
        if payload.get("drifted_mirror_paths"):
            for path in payload["drifted_mirror_paths"]:
                print(f"  mirror: {path}")
        if payload.get("pr_url"):
            print(f"  pr: {payload['pr_url']}")
    return EXIT_BY_OUTCOME.get(outcome, 2)


if __name__ == "__main__":
    raise SystemExit(main())
