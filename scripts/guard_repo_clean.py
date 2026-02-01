#!/usr/bin/env python3
"""
Repo hygiene guard.

Fails if tracked files include runtime artifacts (DBs, node_modules, build outputs).
Optionally checks the working tree for untracked artifacts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


TRACKED_PATTERNS = [
    # Directories
    "/node_modules/",
    "/dist/",
    "/output/",
    "/htmlcov/",
    "/.nomic/",
    "/.pytest_cache/",
    "/.mypy_cache/",
    "/.ruff_cache/",
]

TRACKED_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".sqlite",
    ".sqlite3",
    ".log",
)


def _get_data_dir(repo_root: Path) -> Path:
    env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (repo_root / ".nomic").resolve()


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_status_untracked() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:].strip()
        if status in {"??", "A ", "AM", " M"}:
            paths.append(path)
    return paths


def _matches(path: str) -> bool:
    if any(segment in path for segment in TRACKED_PATTERNS):
        return True
    return path.endswith(TRACKED_SUFFIXES)


def _report(heading: str, paths: list[str]) -> None:
    print(heading)
    for path in sorted(paths):
        print(f"  - {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard repo against runtime artifacts.")
    parser.add_argument(
        "--check-working-tree",
        action="store_true",
        help="Also fail if untracked working-tree artifacts are present.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _get_data_dir(repo_root)

    tracked = [path for path in _git_ls_files() if _matches(path)]
    if tracked:
        _report("Tracked runtime artifacts detected:", tracked)
        print("Remove these files from git and keep runtime data under ARAGORA_DATA_DIR.")
        return 1

    if args.check_working_tree:
        untracked = [path for path in _git_status_untracked() if _matches(path)]
        # Ignore artifacts under ARAGORA_DATA_DIR
        untracked = [path for path in untracked if not _is_under(repo_root / path, data_dir)]
        if untracked:
            _report("Untracked runtime artifacts detected:", untracked)
            print("Move runtime data under ARAGORA_DATA_DIR or delete these artifacts.")
            return 1

    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
