#!/usr/bin/env python3
"""
Move root-level runtime SQLite artifacts into ARAGORA_DATA_DIR.

Defaults to scanning the repo root for *.db, *.db-wal, *.db-shm, *.sqlite, *.sqlite3
and moving them under ARAGORA_DATA_DIR (or .nomic/ if unset).

Tracked files are skipped by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from aragora.config import resolve_db_path

SUFFIXES = (".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_data_dir(repo_root: Path) -> Path:
    env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    for candidate in (repo_root / ".nomic", repo_root / "data"):
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / ".nomic").resolve()


def _get_tracked(repo_root: Path) -> set[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return set()
    return {repo_root / line.strip() for line in result.stdout.splitlines() if line.strip()}


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _target_for(src: Path) -> Path:
    name = src.name
    if name.endswith((".db-wal", ".db-shm")):
        base = name[:-4]  # strip -wal / -shm
        target_base = Path(resolve_db_path(base))
        return target_base.with_name(target_base.name + name[len(base) :])
    return Path(resolve_db_path(name))


def _collect_candidates(repo_root: Path, data_dir: Path, include_subdirs: bool) -> list[Path]:
    if include_subdirs:
        paths = [p for p in repo_root.rglob("*") if p.is_file()]
    else:
        paths = [p for p in repo_root.iterdir() if p.is_file()]
    return [p for p in paths if p.name.endswith(SUFFIXES) and not _is_under(p, data_dir)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print moves without modifying files.",
    )
    parser.add_argument(
        "--include-subdirs",
        action="store_true",
        help="Scan subdirectories (default: repo root only).",
    )
    parser.add_argument(
        "--include-tracked",
        action="store_true",
        help="Allow moving tracked files (default: skip tracked).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite targets if they already exist.",
    )
    args = parser.parse_args()

    repo_root = _get_repo_root()
    data_dir = _get_data_dir(repo_root)
    tracked = _get_tracked(repo_root)

    candidates = _collect_candidates(repo_root, data_dir, args.include_subdirs)
    if not candidates:
        print("No runtime DB artifacts found.")
        return 0

    for src in candidates:
        if not args.include_tracked and src in tracked:
            print(f"Skipping tracked file: {src}")
            continue

        target = _target_for(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            print(f"Target exists, skipping: {src} -> {target}")
            continue

        if args.dry_run:
            print(f"[dry-run] {src} -> {target}")
            continue

        shutil.move(str(src), str(target))
        print(f"Moved: {src} -> {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
