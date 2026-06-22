#!/usr/bin/env python3
"""Fail when a tracked repo-root file is not in docs/reference/ROOT_ALLOWLIST.md.

Repo-root hygiene guard (HEALTH-1 #8258): the repository root should hold only a
curated allowlist of files. Any tracked file living directly at the repo root
that is not listed in ``docs/reference/ROOT_ALLOWLIST.md`` trips this check,
naming the offender. Files in subdirectories are out of scope.

Usage::

    python3 scripts/ci/check_root_allowlist.py            # exit 0 when root is clean
    python3 scripts/ci/check_root_allowlist.py            # exit 1 naming any newcomer
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_REL = "docs/reference/ROOT_ALLOWLIST.md"
BEGIN_MARKER = "<!-- ROOT_ALLOWLIST_BEGIN -->"
END_MARKER = "<!-- ROOT_ALLOWLIST_END -->"


def parse_allowlist(text: str) -> set[str]:
    """Extract allowlisted root filenames from the markdown allowlist doc.

    Entries are the non-empty lines between the BEGIN/END HTML-comment markers.
    Code-fence lines, comment lines, surrounding backticks, and any token
    containing ``/`` (i.e. not repo-root scope) are ignored.
    """
    entries: set[str] = set()
    capturing = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == BEGIN_MARKER:
            capturing = True
            continue
        if stripped == END_MARKER:
            break
        if not capturing:
            continue
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        token = stripped.strip("`").strip()
        if not token or "/" in token:
            continue
        entries.add(token)
    return entries


def find_offenders(tracked_root_files: list[str], allowlist: set[str]) -> list[str]:
    """Tracked repo-root files that are not covered by the allowlist."""
    return sorted(f for f in tracked_root_files if f not in allowlist)


def list_tracked_root_files() -> list[str]:
    """Tracked entries living directly at the repo root (no path separator)."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(p for p in out.split("\0") if p and "/" not in p)


def main() -> int:
    doc = REPO_ROOT / ALLOWLIST_REL
    if not doc.is_file():
        print(f"check_root_allowlist: FAIL missing allowlist doc {ALLOWLIST_REL}")
        return 2

    allowlist = parse_allowlist(doc.read_text(encoding="utf-8"))
    if not allowlist:
        print(f"check_root_allowlist: FAIL empty allowlist in {ALLOWLIST_REL}")
        return 2

    tracked = list_tracked_root_files()
    offenders = find_offenders(tracked, allowlist)
    if offenders:
        print("check_root_allowlist: FAIL non-allowlisted tracked repo-root file(s):")
        for offender in offenders:
            print(f"  {offender}")
        print(
            f"Add a legitimate entry to {ALLOWLIST_REL}, or untrack clutter with "
            "'git rm --cached <file>'."
        )
        return 1

    print(f"check_root_allowlist: OK all {len(tracked)} tracked repo-root files are allowlisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
