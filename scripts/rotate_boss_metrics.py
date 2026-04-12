#!/usr/bin/env python3
"""Archive the active boss metrics file and create a fresh empty replacement."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = REPO_ROOT / ".aragora" / "overnight" / "boss_metrics.jsonl"


def archive_path_for(metrics_file: Path, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return metrics_file.parent / "archive" / f"{metrics_file.stem}.{timestamp}{metrics_file.suffix}"


def rotate_metrics_file(
    metrics_file: Path,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    archive_path = archive_path_for(metrics_file, now=now)
    summary = {
        "metrics_file": str(metrics_file),
        "archive_path": str(archive_path),
        "dry_run": dry_run,
        "archived_existing_file": metrics_file.exists(),
        "created_fresh_file": True,
    }

    if dry_run:
        return summary

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if metrics_file.exists():
        metrics_file.replace(archive_path)
    metrics_file.write_text("", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive boss_metrics.jsonl and create a fresh empty file."
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to boss_metrics.jsonl (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions only")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = rotate_metrics_file(args.metrics_file, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        action = "Would archive" if args.dry_run else "Archived"
        print(
            f"{action} {summary['metrics_file']} to {summary['archive_path']} and create a fresh empty file."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
