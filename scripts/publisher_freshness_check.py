#!/usr/bin/env python3
"""Diagnostic surface for the codex automation publisher.

Joins three independent signals into a single readiness verdict:

1. launchd job loaded? (via ``launchctl print``)
2. status cache age (mtime of ``.aragora/automation-github-status/latest.json``)
3. outbox-vs-cache drift (does the cache's ``outbox_count`` agree with the
   real file count under ``.aragora/automation-outbox``?)

The previous-task contract requires all three to be healthy before opening
new product lanes. Until this script existed, an operator had to run three
different commands and join the results manually.

Examples
--------
$ python3 scripts/publisher_freshness_check.py
publisher: degraded (launchd: not loaded; cache: 48.5h stale; drift: outbox=18 cache=18)
$ python3 scripts/publisher_freshness_check.py --json
{"verdict": "degraded", ...}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAUNCHD_LABEL = "com.aragora.codex-automation-publisher"


@dataclass
class FreshnessReport:
    verdict: str
    summary: str
    launchd_loaded: bool
    launchd_detail: str
    cache_path: str
    cache_present: bool
    cache_age_seconds: float | None
    cache_age_human: str
    cache_stale: bool
    cache_stale_threshold_seconds: int
    outbox_dir: str
    outbox_real_count: int
    outbox_cache_count: int | None
    outbox_drift: bool
    drift_detail: str
    blockers: list[str] = field(default_factory=list)


def _launchd_loaded(label: str) -> tuple[bool, str]:
    try:
        domain = f"gui/{os.getuid()}"
    except AttributeError:
        return False, "non-posix-platform"
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, "launchctl-not-found"
    except subprocess.TimeoutExpired:
        return False, "launchctl-timeout"
    if proc.returncode == 0:
        return True, "loaded"
    stderr = (proc.stderr or "").strip().splitlines()
    detail = stderr[-1] if stderr else f"rc={proc.returncode}"
    return False, detail


def _human_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _count_outbox_files(outbox_dir: Path) -> int:
    if not outbox_dir.is_dir():
        return 0
    return sum(1 for p in outbox_dir.iterdir() if p.is_file() and p.suffix == ".json")


def _read_cache_outbox_count(cache_path: Path) -> int | None:
    try:
        with cache_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    local_queue = payload.get("local_queue")
    if not isinstance(local_queue, dict):
        return None
    raw = local_queue.get("outbox_count")
    return int(raw) if isinstance(raw, int) else None


def evaluate(
    repo_root: Path,
    *,
    cache_path: Path | None = None,
    outbox_dir: Path | None = None,
    stale_threshold_seconds: int = 1800,
    now: float | None = None,
) -> FreshnessReport:
    if cache_path is None:
        cache_path = repo_root / ".aragora" / "automation-github-status" / "latest.json"
    if outbox_dir is None:
        outbox_dir = repo_root / ".aragora" / "automation-outbox"
    now = now if now is not None else time.time()

    loaded, detail = _launchd_loaded(LAUNCHD_LABEL)

    cache_present = cache_path.is_file()
    cache_age: float | None
    if cache_present:
        cache_age = max(0.0, now - cache_path.stat().st_mtime)
    else:
        cache_age = None
    cache_stale = (cache_age is None) or (cache_age > stale_threshold_seconds)

    outbox_real = _count_outbox_files(outbox_dir)
    outbox_cache = _read_cache_outbox_count(cache_path) if cache_present else None
    drift = outbox_cache is not None and outbox_cache != outbox_real
    if outbox_cache is None:
        drift_detail = f"outbox={outbox_real} cache=missing"
    elif drift:
        drift_detail = f"outbox={outbox_real} cache={outbox_cache}"
    else:
        drift_detail = f"outbox={outbox_real} cache={outbox_cache}"

    blockers: list[str] = []
    if not loaded:
        blockers.append(f"launchd: {detail}")
    if not cache_present:
        blockers.append("cache: missing")
    elif cache_stale:
        blockers.append(f"cache: {_human_age(cache_age)} stale")
    if drift:
        blockers.append(f"drift: {drift_detail}")

    if not blockers:
        verdict = "ready"
    elif loaded and not drift:
        verdict = "warming"
    else:
        verdict = "degraded"

    summary_parts = []
    summary_parts.append(f"launchd: {'loaded' if loaded else detail}")
    summary_parts.append(f"cache: {_human_age(cache_age) if cache_present else 'missing'}")
    summary_parts.append(f"drift: {drift_detail}")
    summary = f"publisher: {verdict} ({'; '.join(summary_parts)})"

    return FreshnessReport(
        verdict=verdict,
        summary=summary,
        launchd_loaded=loaded,
        launchd_detail=detail,
        cache_path=str(cache_path),
        cache_present=cache_present,
        cache_age_seconds=cache_age,
        cache_age_human=_human_age(cache_age),
        cache_stale=cache_stale,
        cache_stale_threshold_seconds=stale_threshold_seconds,
        outbox_dir=str(outbox_dir),
        outbox_real_count=outbox_real,
        outbox_cache_count=outbox_cache,
        outbox_drift=drift,
        drift_detail=drift_detail,
        blockers=blockers,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=os.getcwd(), help="Path inside the target repository")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--outbox-dir", default=None)
    parser.add_argument(
        "--stale-threshold-seconds",
        type=int,
        default=1800,
        help="Cache mtime older than this is considered stale (default: 1800s = 30min)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON to stdout instead of the 1-line summary"
    )
    parser.add_argument(
        "--exit-nonzero-on-degraded",
        action="store_true",
        help="Exit 1 when verdict is degraded (useful in CI/launchd post-flight)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(args.repo).resolve()
    cache_path = Path(args.cache_path).resolve() if args.cache_path else None
    outbox_dir = Path(args.outbox_dir).resolve() if args.outbox_dir else None
    report = evaluate(
        repo_root,
        cache_path=cache_path,
        outbox_dir=outbox_dir,
        stale_threshold_seconds=args.stale_threshold_seconds,
    )
    if args.json:
        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            **asdict(report),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.summary)
        if report.blockers and not args.json:
            for blocker in report.blockers:
                print(f"  - {blocker}", file=sys.stderr)
    if args.exit_nonzero_on_degraded and report.verdict == "degraded":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
