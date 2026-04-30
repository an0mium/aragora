#!/usr/bin/env python3
"""Measure #6375 invalidation baseline and optionally persist a receipt.

The default mode is read-only and prints the receipt payload that would be
written. Passing ``--write-receipt`` is the only mutation path; it writes one
JSON receipt under ``.aragora/review-queue/thresholds/``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.review import (
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    DEFAULT_SAFETY_MARGIN,
    DEFAULT_THRESHOLD_RECEIPT_DIR,
    ReviewQueueInvalidationEventSource,
    ThresholdRecalibrationScheduler,
    write_recalibration_receipt,
)
from aragora.triage.auto_handle_calibration import AutoHandleCalibrationStore

UTC = timezone.utc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Aragora's empirical invalidation baseline (#6375) and "
            "emit either a threshold_update_receipt.v1 or insufficiency_receipt.v1."
        )
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_BASELINE_WINDOW_DAYS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_BASELINE_SAMPLES)
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN)
    parser.add_argument(
        "--minimum-meaningful-rate",
        type=float,
        default=DEFAULT_MINIMUM_MEANINGFUL_RATE,
    )
    parser.add_argument("--placeholder-value", type=float, default=0.05)
    parser.add_argument("--calibration-db", default=None)
    parser.add_argument("--review-queue-root", default=None)
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=DEFAULT_THRESHOLD_RECEIPT_DIR,
        help="Receipt directory relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--write-receipt",
        action="store_true",
        help="Persist the emitted receipt JSON. Default is dry-run/read-only.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        store = AutoHandleCalibrationStore(db_path=args.calibration_db)
        source = ReviewQueueInvalidationEventSource(
            calibration_store=store,
            review_queue_root=args.review_queue_root,
        )
        scheduler = ThresholdRecalibrationScheduler(
            window_days=args.window_days,
            min_samples=args.min_samples,
            safety_margin=args.safety_margin,
            minimum_meaningful_rate=args.minimum_meaningful_rate,
            placeholder_value=args.placeholder_value,
        )
        receipt = scheduler.run_receipt_from_source(source, now=datetime.now(UTC))
    except (OSError, RuntimeError, sqlite3.Error, ValueError, TypeError) as exc:
        print(f"error: baseline measurement failed: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, Any] = receipt.to_dict()
    path: Path | None = None
    if args.write_receipt:
        repo_root = _resolve_repo_root(Path.cwd())
        receipt_dir = args.receipt_dir
        if receipt_dir.is_absolute():
            try:
                receipt_dir = receipt_dir.relative_to(repo_root)
            except ValueError:
                pass
        path = write_recalibration_receipt(receipt, repo_root=repo_root, receipt_dir=receipt_dir)
        payload["receipt_path"] = str(path)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _render_summary(payload, path=path)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.window_days <= 0:
        raise ValueError("--window-days must be positive")
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    if not 0 < args.safety_margin <= 1:
        raise ValueError("--safety-margin must be in (0, 1]")
    if args.minimum_meaningful_rate <= 0:
        raise ValueError("--minimum-meaningful-rate must be positive")
    if not 0 < args.placeholder_value < 1:
        raise ValueError("--placeholder-value must be in (0, 1)")


def _resolve_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / ".aragora").exists():
            return candidate
    return start


def _render_summary(payload: dict[str, Any], *, path: Path | None) -> None:
    schema = payload["schema_version"]
    sample_count = payload["sample_count"]
    threshold = payload["proposal"]["threshold"]
    placeholder = payload["proposal"]["is_placeholder"]
    print(f"schema: {schema}")
    print(f"receipt_id: {payload['receipt_id']}")
    print(f"sample_count: {sample_count}")
    print(f"threshold: {threshold}")
    print(f"placeholder: {placeholder}")
    if schema == "insufficiency_receipt.v1":
        reasons = ", ".join(payload["insufficiency"]["reasons"])
        needed = payload["insufficiency"]["additional_dispatches_needed"]
        print(f"insufficiency_reasons: {reasons}")
        print(f"additional_dispatches_needed: {needed}")
    if path is None:
        print("receipt_path: (dry-run; pass --write-receipt to persist)")
    else:
        print(f"receipt_path: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
