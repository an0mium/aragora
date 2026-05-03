"""CLI dispatcher + rendering for ``aragora review-queue observe-outcomes``.

The observation pipeline lives in
:mod:`aragora.review.observe_outcomes_cli`. This module is the
``aragora.cli.*`` surface that prints to stdout/stderr — kept here so
the ``T201`` per-file-ignore covers ``print`` cleanly without leaking
into non-CLI modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.review.observe_outcomes_cli import run_observe_outcomes

UTC = timezone.utc


def cmd_observe_outcomes(args: argparse.Namespace) -> int:
    if args.window_days <= 0:
        print("error: --window-days must be positive", file=sys.stderr)
        return 2
    if args.max_receipts <= 0:
        print("error: --max-receipts must be positive", file=sys.stderr)
        return 2
    if args.per_receipt_event_cap <= 0:
        print("error: --per-receipt-event-cap must be positive", file=sys.stderr)
        return 2

    try:
        summary = run_observe_outcomes(
            store_root=args.review_queue_root,
            repo_root=Path.cwd(),
            window_end=datetime.now(UTC),
            window_days=args.window_days,
            max_receipts=args.max_receipts,
            per_receipt_event_cap=args.per_receipt_event_cap,
            write=bool(args.write),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: observe-outcomes failed: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_summary(summary)
    return 0


def _render_summary(summary: Mapping[str, Any]) -> None:
    mode = summary["mode"]
    print(f"observe-outcomes mode={mode}")
    print(
        f"  window_end={summary['window_end_utc']} "
        f"window_days={summary['window_days']} "
        f"max_receipts={summary['max_receipts']}"
    )
    print(
        f"  receipts_examined={summary['receipts_examined']} "
        f"with_signals={summary['receipts_with_signals_fired']} "
        f"written={summary['receipts_written']} "
        f"fetch_errors={summary['github_fetch_errors']}"
    )
    print(f"  v2_now_present_in_window={summary['v2_outcome_fields_now_present_in_window']}")
    if summary["insufficiency_receipt_path"]:
        print(f"  insufficiency_receipt={summary['insufficiency_receipt_path']}")
    if mode == "dry-run" and summary["receipts_examined"] > 0:
        print("  (dry-run; pass --write to mutate receipts in place)")
    for r in summary["results"][:20]:
        skipped = repr(r["skipped_reason"]) if r["skipped_reason"] else "None"
        print(
            f"  pr=#{r['pr_number']} fetched={r['fetched_event_count']} "
            f"skipped={skipped} "
            f"written={r['written']}"
        )
