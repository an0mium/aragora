#!/usr/bin/env python3
"""PR value-composition classifier for the autonomous PR pipeline (READ-ONLY).

This script makes self-maintenance drift visible by classifying open PRs as
``maintenance``, ``product``, ``infra``, or ``unknown`` using the shared
``aragora.swarm.pr_value`` heuristic. It never mutates GitHub; the only optional
write is the ``--out`` JSON snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.pr_value import (  # noqa: E402
    DEFAULT_MAX_MAINTENANCE_RATIO,
    DEFAULT_STALE_DAYS,
    build_value_report,
    classify_value_record,
    summary_line,
)

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LIMIT = 300
GH_TIMEOUT_SECONDS = 120

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_BREACH = 3

_GH_FIELDS = "number,title,labels,isDraft,createdAt,updatedAt"


def classify_pr(pr: dict[str, Any]) -> str:
    """Backward-compatible script-level alias for tests and ad-hoc callers."""
    return classify_value_record(pr)


def default_list_prs(repo: str, limit: int) -> list[dict[str, Any]]:
    """One ``gh pr list`` call for open PRs (read-only, capped at ``limit``)."""
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        _GH_FIELDS,
        "--limit",
        str(limit),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=GH_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed (exit {result.returncode}): {result.stderr.strip()}")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list returned unexpected payload (expected a list)")
    return payload


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write temp + ``os.replace`` so readers never observe a partial file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def run_classifier(
    *,
    list_prs: Callable[[], list[dict[str, Any]]],
    limit: int = DEFAULT_LIMIT,
    stale_days: int = DEFAULT_STALE_DAYS,
    max_maintenance_ratio: float = DEFAULT_MAX_MAINTENANCE_RATIO,
    out_file: str | None = None,
    summary: bool = False,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Build one value-composition report; return 0 / 3 (breach) / 1."""
    if now is None:
        now = datetime.now(timezone.utc)
    annotations: list[str] = []

    try:
        raw = list_prs()
        if not isinstance(raw, list):
            raise RuntimeError("list_prs returned unexpected payload (expected a list)")
        prs = [pr for pr in raw if isinstance(pr, dict)]
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"action": "error", "error": str(exc)[:500]}), file=sys.stderr)
        return EXIT_FAILURE

    if len(prs) >= limit:
        annotations.append(f"list_truncated:>={limit}")

    report = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **build_value_report(
            prs,
            stale_days=stale_days,
            max_maintenance_ratio=max_maintenance_ratio,
            now=now,
            annotations=annotations,
        ),
    }

    log(summary_line(report) if summary else json.dumps(report, sort_keys=True))

    if out_file:
        try:
            atomic_write_json(out_file, report)
        except OSError as exc:
            print(
                json.dumps({"action": "out_write_failed", "error": str(exc)[:300]}),
                file=sys.stderr,
            )
            return EXIT_FAILURE

    if report["maintenance_ratio"] > max_maintenance_ratio:
        return EXIT_BREACH
    return EXIT_OK


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_arg(value: str) -> str:
    """Argparse type: validate ``owner/name`` at parse time."""
    if not _REPO_RE.match(value):
        raise argparse.ArgumentTypeError(f"--repo must look like owner/name, got {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only PR value-composition classifier. Exits 0 normally, "
            "3 when maintenance_ratio exceeds the threshold, 1 on failure."
        )
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, type=repo_arg, help="GitHub repo (owner/name)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Per-run cap on open PRs fetched (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Open PRs older than this many days count as stale (default {DEFAULT_STALE_DAYS})",
    )
    parser.add_argument(
        "--max-maintenance-ratio",
        type=float,
        default=DEFAULT_MAX_MAINTENANCE_RATIO,
        help=f"Breach (exit 3) when maintenance_ratio exceeds this "
        f"(default {DEFAULT_MAX_MAINTENANCE_RATIO})",
    )
    parser.add_argument("--summary", action="store_true", help="Emit one summary line")
    parser.add_argument("--out", default=None, help="Optional atomic JSON output path")
    args = parser.parse_args(argv)

    return run_classifier(
        list_prs=lambda: default_list_prs(args.repo, args.limit),
        limit=args.limit,
        stale_days=args.stale_days,
        max_maintenance_ratio=args.max_maintenance_ratio,
        out_file=args.out,
        summary=args.summary,
    )


if __name__ == "__main__":
    sys.exit(main())
