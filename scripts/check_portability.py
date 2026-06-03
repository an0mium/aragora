#!/usr/bin/env python3
"""Portability/privacy guard for the public repo.

Fails when a tracked file introduces a *new* hardcoded private-machine or
legacy-identity reference that would break a third-party clone:

  - ``users_home``  : a private macOS home dir like ``/Users/<name>`` (the CI
                      runner home ``/Users/runner`` is allowed)
  - ``venv_python`` : ``.venv/bin/python`` captured as a durable interpreter path
  - ``legacy_slug`` : the old GitHub slug ``an0mium/aragora`` (repo is
                      ``synaptent/aragora``)

A baseline (``scripts/portability_baseline.json``) records the violations that
already exist on ``main`` so the guard is **non-breaking**: it only fails on
violations that are *not* baselined. As the Tier A-C cleanup PRs land, prune the
baseline so the surface ratchets down and cannot regress.

Usage::

    python3 scripts/check_portability.py                 # CI: scan all tracked files
    python3 scripts/check_portability.py FILE [FILE ...] # pre-commit: scan given files
    python3 scripts/check_portability.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "scripts" / "portability_baseline.json"

# pattern id -> compiled regex
PATTERNS: dict[str, re.Pattern[str]] = {
    "users_home": re.compile(r"/Users/([A-Za-z0-9._-]+)"),
    "venv_python": re.compile(r"\.venv/bin/python"),
    "legacy_slug": re.compile(r"an0mium/aragora"),
}

# macOS usernames that are legitimate (GitHub-hosted runner home)
ALLOWED_USERS = {"runner"}

# Files never scanned: this guard's own sources (which contain the patterns as
# data) and the audit reports that legitimately quote the patterns.
ALWAYS_SKIP = (
    "scripts/check_portability.py",
    "scripts/portability_baseline.json",
    "tests/scripts/test_check_portability.py",
    # Regression test that asserts installers never bake .venv/bin/python; it
    # holds the pattern as fixture/assertion data, not a real interpreter capture.
    "tests/scripts/test_launchd_installers.py",
    # Machine-local gt store config (gitignored; its `workspace` is a per-clone
    # absolute path). Untracked, so never a portability concern.
    ".gt/config.json",
    "docs/audits/*",
    # Sanctioned runtime interpreter resolver: legitimately references
    # .venv/bin/python3 as an existence check (resolved at runtime, never baked).
    "scripts/aragora_runtime.sh",
)


def _is_skipped(rel_path: str) -> bool:
    p = Path(rel_path)
    return any(p.match(pattern) for pattern in ALWAYS_SKIP)


def find_violations_in_text(text: str) -> set[str]:
    """Return the set of pattern ids that match ``text``."""
    found: set[str] = set()
    for pid, rx in PATTERNS.items():
        for match in rx.finditer(text):
            if pid == "users_home" and match.group(1) in ALLOWED_USERS:
                continue
            found.add(pid)
    return found


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None  # binary or unreadable -> skip


def scan_paths(rel_paths: list[str], *, root: Path | None = None) -> dict[str, list[str]]:
    """Scan ``rel_paths`` (repo-relative) and return ``{path: sorted pattern ids}``."""
    root = root or REPO_ROOT
    violations: dict[str, list[str]] = {}
    for rel in rel_paths:
        rel = Path(rel).as_posix()
        if _is_skipped(rel):
            continue
        text = _read_text(root / rel)
        if text is None:
            continue
        found = find_violations_in_text(text)
        if found:
            violations[rel] = sorted(found)
    return violations


def _git_tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def load_baseline(path: Path | None = None) -> dict[str, list[str]]:
    path = path or BASELINE_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("files", {})


def new_violations(
    found: dict[str, list[str]], baseline: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Violations present in ``found`` but not accepted by ``baseline``."""
    result: dict[str, list[str]] = {}
    for path, pids in found.items():
        allowed = set(baseline.get(path, []))
        delta = sorted(set(pids) - allowed)
        if delta:
            result[path] = delta
    return result


def write_baseline(found: dict[str, list[str]], path: Path | None = None) -> None:
    path = path or BASELINE_PATH
    payload = {
        "_comment": (
            "Baseline of pre-existing portability violations. The guard "
            "(scripts/check_portability.py) only fails on violations NOT listed "
            "here. Prune entries as cleanup PRs remove the underlying paths; do "
            "not add new entries by hand. Regenerate with --update-baseline."
        ),
        "files": dict(sorted(found.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Specific files to scan (default: all tracked).")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline from the current full-tree scan.",
    )
    args = parser.parse_args(argv)

    if args.update_baseline:
        found = scan_paths(_git_tracked_files(REPO_ROOT))
        write_baseline(found)
        print(f"Wrote baseline with {len(found)} files to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    if args.files:
        rel_paths = []
        for f in args.files:
            p = Path(f)
            try:
                rel_paths.append(p.resolve().relative_to(REPO_ROOT).as_posix())
            except ValueError:
                rel_paths.append(p.as_posix())
    else:
        rel_paths = _git_tracked_files(REPO_ROOT)

    found = scan_paths(rel_paths)
    offenders = new_violations(found, load_baseline())

    if offenders:
        print("Portability guard: NEW hardcoded private/legacy references found:\n")
        for path in sorted(offenders):
            print(f"  {path}: {', '.join(offenders[path])}")
        print(
            "\nUse a portable form instead (e.g. $HOME, "
            "git rev-parse --show-toplevel, ${{ github.repository }}, "
            "ARAGORA_PYTHON / synaptent/aragora). If this is a legitimate, "
            "reviewed exception, update scripts/portability_baseline.json via "
            "--update-baseline."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
