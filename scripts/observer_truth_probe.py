#!/usr/bin/env python3
"""Observer-truth probe.

Asserts that the *observer* (the git checkout from which proof-loop
surfaces -- ``swarm shift-status``, ``swarm status``, benchmark
publication, operator proofs -- are being read) is sitting on a clean
``origin/main`` checkout.

Per ``docs/status/NEXT_STEPS_CANONICAL.md`` (Observer rule):

    "run swarm shift-status, swarm status, benchmark publication, and
    operator proofs from a clean worktree reconciled to current
    origin/main. ... if the observer reports itself as dirty, ahead, or
    behind, fix the observer before widening roadmap scope or restocking
    the live queue."

The probe answers a single question with a JSON payload:

    {
      "clean": true | false,
      "head_sha": "<sha>",
      "origin_main_sha": "<sha>",
      "ahead": <int>,
      "behind": <int>,
      "untracked_count": <int>,
      "uncommitted_modified_count": <int>,
      "submodule_dirty": true | false,
      "reasons": [ "<short string>", ... ],
      "repo_root": "<absolute path>",
      "checked_at": "<RFC3339 UTC>"
    }

Cleanliness definition (all must be true):

* ``untracked_count == 0``
* ``uncommitted_modified_count == 0``
* ``submodule_dirty is False``
* ``ahead == 0``
* ``behind == 0``
* ``head_sha == origin_main_sha``

CLI flags:

  --repo-root <path>      Repo root to probe (default: cwd).
  --strict-mode / ...     Default true: exit non-zero when not clean.
  --no-strict-mode        Always exit 0 (still prints JSON).
  --no-fetch              Skip ``git fetch origin main`` before probing.
  --quiet                 Suppress the human-readable banner; only JSON.
  --json-only             Alias for --quiet.

Design notes:

* Pure stdlib + subprocess, no third-party imports.
* Read-only against the probed working tree -- no mutation. The optional
  ``git fetch origin main`` updates the local origin-tracking ref, which
  is the *only* side effect (and that side effect can be disabled with
  ``--no-fetch``).
* All git subprocess invocations are time-bounded so the probe cannot
  hang on a misbehaving repo / network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


GIT_TIMEOUT_SECONDS = 30
FETCH_TIMEOUT_SECONDS = 60


def _now_iso() -> str:
    """Return current UTC time in RFC3339 with ``Z`` suffix."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(
    args: list[str],
    *,
    repo_root: Path,
    check: bool = True,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run a git subprocess inside ``repo_root`` and return its result.

    When ``check`` is true a non-zero exit code raises CalledProcessError
    so the caller can decide whether to surface it as a probe reason.
    """
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def head_sha(repo_root: Path) -> str:
    """Return the SHA of ``HEAD`` in ``repo_root`` (empty string on failure)."""
    try:
        result = _run_git(["rev-parse", "HEAD"], repo_root=repo_root)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def origin_main_sha(repo_root: Path, *, fetch: bool = True) -> str:
    """Return the SHA of ``origin/main``; optionally fetch first.

    Returns the empty string when no ``origin/main`` ref exists locally
    (or can be fetched), e.g. on a brand-new local repo. The caller is
    responsible for treating an empty value as "missing".
    """
    if fetch:
        try:
            _run_git(
                ["fetch", "origin", "main", "--quiet"],
                repo_root=repo_root,
                check=False,
                timeout=FETCH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            # Network failure should not abort the probe; just continue.
            pass
    try:
        result = _run_git(["rev-parse", "origin/main"], repo_root=repo_root)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def ahead_behind(repo_root: Path) -> tuple[int, int]:
    """Return ``(ahead, behind)`` for ``HEAD`` relative to ``origin/main``.

    ``ahead``  = commits unique to ``HEAD``
    ``behind`` = commits unique to ``origin/main``

    Returns ``(0, 0)`` when the comparison cannot be performed (e.g.
    missing ``origin/main`` ref). Callers should already have detected
    that case via :func:`origin_main_sha`.
    """
    try:
        result = _run_git(
            ["rev-list", "--left-right", "--count", "origin/main...HEAD"],
            repo_root=repo_root,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return (0, 0)
    parts = result.stdout.split()
    if len(parts) != 2:
        return (0, 0)
    try:
        behind = int(parts[0])
        ahead = int(parts[1])
    except ValueError:
        return (0, 0)
    return ahead, behind


def untracked_files(repo_root: Path) -> list[str]:
    """Return the list of untracked, non-ignored files in ``repo_root``."""
    files, _reason = _untracked_files_probe(repo_root)
    return files


def _untracked_files_probe(repo_root: Path) -> tuple[list[str], str | None]:
    try:
        result = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            repo_root=repo_root,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return [], "untracked_files_probe_failed"
    return [line for line in result.stdout.splitlines() if line.strip()], None


def uncommitted_modified_files(repo_root: Path) -> list[str]:
    """Return tracked files with staged or unstaged modifications.

    Combines ``git diff --name-only`` (unstaged changes against the
    index) and ``git diff --name-only --cached`` (staged changes against
    HEAD), de-duplicating paths that appear in both.
    """
    files, _reason = _uncommitted_modified_files_probe(repo_root)
    return files


def _uncommitted_modified_files_probe(repo_root: Path) -> tuple[list[str], str | None]:
    paths: set[str] = set()
    failed = False
    for extra in ([], ["--cached"]):
        try:
            result = _run_git(
                ["diff", "--name-only", *extra],
                repo_root=repo_root,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failed = True
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                paths.add(line)
    reason = "uncommitted_modified_probe_failed" if failed else None
    return sorted(paths), reason


def submodule_dirty(repo_root: Path) -> bool:
    """Return ``True`` when any submodule is uncommitted or out of sync.

    ``git submodule status --recursive`` prefixes each row with one of:

    * ``" "`` -- in sync
    * ``"-"`` -- not initialized
    * ``"+"`` -- different commit checked out
    * ``"U"`` -- merge conflict

    A repo with no submodules emits no output, which is treated as
    clean.
    """
    dirty, _reason = _submodule_dirty_probe(repo_root)
    return dirty


def _submodule_dirty_probe(repo_root: Path) -> tuple[bool, str | None]:
    try:
        result = _run_git(
            ["submodule", "status", "--recursive"],
            repo_root=repo_root,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "submodule_status_probe_failed"
    if result.returncode != 0:
        return False, "submodule_status_probe_failed"
    for line in result.stdout.splitlines():
        if not line:
            continue
        if line[0] in {"+", "-", "U"}:
            return True, None
    return False, None


def probe(
    repo_root: Path,
    *,
    fetch: bool = True,
) -> dict[str, Any]:
    """Run the full observer-truth probe and return a JSON-ready dict."""
    repo_root = repo_root.resolve()
    reasons: list[str] = []

    head = head_sha(repo_root)
    if not head:
        reasons.append("head_sha_unavailable")

    origin = origin_main_sha(repo_root, fetch=fetch)
    if not origin:
        reasons.append("origin_main_unavailable")

    if head and origin:
        ahead, behind = ahead_behind(repo_root)
    else:
        ahead, behind = (0, 0)

    untracked, untracked_probe_reason = _untracked_files_probe(repo_root)
    uncommitted, uncommitted_probe_reason = _uncommitted_modified_files_probe(repo_root)
    sub_dirty, submodule_probe_reason = _submodule_dirty_probe(repo_root)

    if untracked:
        reasons.append(f"untracked_files={len(untracked)}")
    if uncommitted:
        reasons.append(f"uncommitted_modified={len(uncommitted)}")
    if sub_dirty:
        reasons.append("submodule_dirty")
    status_probe_reasons = [
        reason
        for reason in (
            untracked_probe_reason,
            uncommitted_probe_reason,
            submodule_probe_reason,
        )
        if reason
    ]
    reasons.extend(status_probe_reasons)
    if ahead:
        reasons.append(f"ahead_of_origin_main={ahead}")
    if behind:
        reasons.append(f"behind_origin_main={behind}")
    if head and origin and head != origin:
        reasons.append("head_mismatch_origin_main")

    clean = (
        len(untracked) == 0
        and len(uncommitted) == 0
        and not sub_dirty
        and not status_probe_reasons
        and ahead == 0
        and behind == 0
        and bool(head)
        and bool(origin)
        and head == origin
    )

    return {
        "clean": clean,
        "head_sha": head,
        "origin_main_sha": origin,
        "ahead": ahead,
        "behind": behind,
        "untracked_count": len(untracked),
        "uncommitted_modified_count": len(uncommitted),
        "submodule_dirty": sub_dirty,
        "reasons": reasons,
        "repo_root": str(repo_root),
        "checked_at": _now_iso(),
    }


def _render_banner(result: dict[str, Any]) -> str:
    """Human-readable one-paragraph summary of the probe result."""
    state = "CLEAN" if result["clean"] else "DIRTY"
    lines = [
        f"observer_truth_probe: {state}",
        f"  repo_root            : {result['repo_root']}",
        f"  head_sha             : {result['head_sha'] or '<unavailable>'}",
        f"  origin_main_sha      : {result['origin_main_sha'] or '<unavailable>'}",
        f"  ahead / behind       : {result['ahead']} / {result['behind']}",
        f"  untracked_count      : {result['untracked_count']}",
        f"  uncommitted_modified : {result['uncommitted_modified_count']}",
        f"  submodule_dirty      : {result['submodule_dirty']}",
    ]
    if result["reasons"]:
        lines.append("  reasons              : " + ", ".join(result["reasons"]))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root to probe (default: current working directory).",
    )
    parser.add_argument(
        "--strict-mode",
        dest="strict_mode",
        action="store_true",
        default=True,
        help="Exit non-zero when clean=false (default).",
    )
    parser.add_argument(
        "--no-strict-mode",
        dest="strict_mode",
        action="store_false",
        help="Always exit 0 (still prints JSON).",
    )
    parser.add_argument(
        "--no-fetch",
        dest="fetch",
        action="store_false",
        default=True,
        help="Skip 'git fetch origin main --quiet' before probing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable banner; print JSON only.",
    )
    parser.add_argument(
        "--json-only",
        dest="quiet",
        action="store_true",
        help="Alias for --quiet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    result = probe(repo_root, fetch=args.fetch)

    if not args.quiet:
        print(_render_banner(result), file=sys.stderr)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.strict_mode and not result["clean"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
