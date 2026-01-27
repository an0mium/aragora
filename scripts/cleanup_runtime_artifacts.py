#!/usr/bin/env python3
"""
Move runtime artifacts (SQLite DBs, WALs) out of the repo root into ARAGORA_DATA_DIR.

Default behavior is a dry run. Use --apply to perform moves.
Optional --purge-caches removes common cache/build artifacts in the repo root.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path


DB_PATTERNS = [
    "*.db",
    "*.db-shm",
    "*.db-wal",
    "*.db-journal",
    "*.sqlite",
    "*.sqlite3",
    ":memory:",
    ":memory:*",
]

CACHE_ENTRIES = [
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".test_cache",
    ".benchmarks",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "dist",
    "build",
]


def resolve_data_dir(repo_root: Path) -> Path:
    env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
    return (Path(env_dir) if env_dir else repo_root / ".nomic").resolve()


def collect_db_artifacts(repo_root: Path) -> list[Path]:
    artifacts: list[Path] = []
    for pattern in DB_PATTERNS:
        artifacts.extend(repo_root.glob(pattern))
    return sorted(set(p for p in artifacts if p.is_file()))


def move_artifacts(artifacts: list[Path], data_dir: Path, apply: bool) -> list[str]:
    actions: list[str] = []
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    for src in artifacts:
        dest = data_dir / src.name
        if dest.exists():
            dest = data_dir / f"{src.name}.{timestamp}"
        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
        actions.append(f"{src} -> {dest}")
    return actions


def purge_caches(repo_root: Path, apply: bool) -> list[str]:
    actions: list[str] = []
    for entry in CACHE_ENTRIES:
        target = repo_root / entry
        if not target.exists():
            continue
        if apply:
            if target.is_dir():
                for child in target.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(target.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                target.rmdir()
            else:
                target.unlink()
        actions.append(f"remove {target}")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move runtime artifacts to ARAGORA_DATA_DIR (default: .nomic)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry run).",
    )
    parser.add_argument(
        "--purge-caches",
        action="store_true",
        help="Remove common cache/build artifacts in the repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = resolve_data_dir(repo_root)

    artifacts = collect_db_artifacts(repo_root)
    if not artifacts:
        print("No runtime DB artifacts found in repo root.")
    else:
        print(f"Data dir: {data_dir}")
        print("Planned moves:")
        for action in move_artifacts(artifacts, data_dir, apply=args.apply):
            print(f"  {action}")

    if args.purge_caches:
        print("\nCache cleanup:")
        for action in purge_caches(repo_root, apply=args.apply):
            print(f"  {action}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to make changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
