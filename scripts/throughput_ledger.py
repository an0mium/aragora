#!/usr/bin/env python3
"""Feed and inspect the throughput ledger (epic #9039, issue #9041).

SENSE-phase aggregator for the work-mix governor
(docs/plans/2026-07-08-vision-audit-and-work-mix-governor.md §4.1):
``snapshot`` classifies recently merged PRs (read-only via gh) into
product-core / product-proof / substrate / maintenance and appends merge +
snapshot records to ``.aragora/throughput/ledger.jsonl``; ``show`` prints the
rolling metrics. READ-ONLY vs GitHub — the only writes are ledger appends.
Advisory Phase 1: nothing here changes gate or goal-selection behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aragora.nomic.throughput import ThroughputLedger, compute_metrics
from aragora.nomic.work_mix import classify_paths


def _gh_merged_prs(limit: int) -> list[dict]:
    try:
        out = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--limit",
                str(limit),
                "--json",
                "number,title,mergedAt,labels,files",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout
    except FileNotFoundError:
        sys.exit("gh CLI not found; snapshot requires gh (read-only)")
    except subprocess.TimeoutExpired:
        sys.exit("gh timed out listing merged PRs")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"gh failed listing merged PRs: {exc.stderr.strip()[:200]}")
    return json.loads(out)


@contextlib.contextmanager
def _snapshot_lock(ledger: ThroughputLedger):
    """Exclusive flock across read-seen + append so concurrent snapshot runs
    cannot double-record the same merged PR (#9048 openai [P2])."""
    lock_path = ledger.path.with_suffix(ledger.path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cmd_snapshot(args: argparse.Namespace) -> int:
    ledger = ThroughputLedger(args.repo_root)
    with _snapshot_lock(ledger):
        return _snapshot_locked(args, ledger)


def _snapshot_locked(args: argparse.Namespace, ledger: ThroughputLedger) -> int:
    seen = {record.data.get("identifier") for record in ledger.records() if record.kind == "merge"}
    added = 0
    for pr in _gh_merged_prs(args.limit):
        identifier = str(pr["number"])
        if identifier in seen:
            continue
        files = pr.get("files", [])
        classified = classify_paths(
            [f["path"] for f in files],
            identifier=identifier,
            title=pr.get("title", ""),
            labels=[label["name"] for label in pr.get("labels", [])],
            # Line weights so trivial padding can't flip the classification.
            weights={f["path"]: f.get("additions", 0) + f.get("deletions", 0) for f in files},
        )
        ledger.record_merge(
            classified,
            title=pr.get("title", ""),
            when=datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00")),
        )
        added += 1
    metrics = compute_metrics(ledger.records(), window_days=args.window_days)
    ledger.record_snapshot(
        {
            "merges_total": metrics.merges_total,
            "product_share": round(metrics.product_share, 3),
            "substrate_share": round(metrics.substrate_share, 3),
            "self_repair_ratio": round(metrics.self_repair_ratio, 3),
            "external_artifacts": metrics.external_artifacts,
            "new_merge_records": added,
        }
    )
    print(f"recorded {added} new merges; ledger at {ledger.path}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    ledger = ThroughputLedger(args.repo_root)
    metrics = compute_metrics(ledger.records(), window_days=args.window_days)
    payload = asdict(metrics)
    payload["merges_by_class"] = {
        cls.value: count for cls, count in metrics.merges_by_class.items()
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="repo checkout root")
    parser.add_argument("--window-days", type=int, default=7)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="classify recent merges into the ledger")
    snapshot.add_argument("--limit", type=int, default=30, help="merged PRs to scan")
    snapshot.set_defaults(func=cmd_snapshot)
    show = sub.add_parser("show", help="print rolling metrics as JSON")
    show.set_defaults(func=cmd_show)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
