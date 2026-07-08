#!/usr/bin/env python3
"""Report whether parked PRs have moved to a new head SHA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_LEDGER = Path(".aragora/parked_prs.json")
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class ParkedPr:
    pr: int
    head_sha: str
    parked_at: str
    blocker_class: str
    source: str


@dataclass(frozen=True)
class RecheckResult:
    pr: int
    parked_head: str
    live_head: str | None
    changed: bool | None
    recommendation: str
    blocker_class: str
    source: str
    error: str | None = None


RunGh = Callable[..., subprocess.CompletedProcess[str]]


def load_ledger(path: Path) -> list[ParkedPr]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"ledger not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ledger is not valid JSON: {path}: {exc}") from exc

    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("ledger must be a list or an object with a records list")

    parked: list[ParkedPr] = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"ledger record {index} must be an object")
        try:
            pr = int(record["pr"])
            head_sha = str(record["head_sha"])
            parked_at = str(record["parked_at"])
            blocker_class = str(record["blocker_class"])
            source = str(record["source"])
        except KeyError as exc:
            raise ValueError(f"ledger record {index} missing field: {exc.args[0]}") from exc
        if pr <= 0:
            raise ValueError(f"ledger record {index} has invalid pr: {pr}")
        if not head_sha:
            raise ValueError(f"ledger record {index} has empty head_sha")
        parked.append(
            ParkedPr(
                pr=pr,
                head_sha=head_sha,
                parked_at=parked_at,
                blocker_class=blocker_class,
                source=source,
            )
        )
    return parked


def fetch_live_head(
    pr: int,
    *,
    repo: str,
    gh_bin: str,
    timeout: int,
    run: RunGh = subprocess.run,
) -> str:
    proc = run(
        [
            gh_bin,
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "headRefOid",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or f"gh exited {proc.returncode}"
        raise RuntimeError(stderr)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh returned invalid JSON for PR #{pr}: {exc}") from exc
    head = payload.get("headRefOid")
    if not isinstance(head, str) or not head:
        raise RuntimeError(f"gh response for PR #{pr} did not include headRefOid")
    return head


def recheck_records(
    records: list[ParkedPr],
    *,
    repo: str = DEFAULT_REPO,
    gh_bin: str = "gh",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    run: RunGh = subprocess.run,
) -> tuple[list[RecheckResult], bool]:
    results: list[RecheckResult] = []
    ok = True
    for record in records:
        try:
            live_head = fetch_live_head(
                record.pr,
                repo=repo,
                gh_bin=gh_bin,
                timeout=timeout,
                run=run,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            ok = False
            results.append(
                RecheckResult(
                    pr=record.pr,
                    parked_head=record.head_sha,
                    live_head=None,
                    changed=None,
                    recommendation="lookup failed; preserve parked state",
                    blocker_class=record.blocker_class,
                    source=record.source,
                    error=str(exc),
                )
            )
            continue

        changed = live_head != record.head_sha
        recommendation = "requeue candidate (head changed)" if changed else "skip parked head"
        results.append(
            RecheckResult(
                pr=record.pr,
                parked_head=record.head_sha,
                live_head=live_head,
                changed=changed,
                recommendation=recommendation,
                blocker_class=record.blocker_class,
                source=record.source,
            )
        )
    return results, ok


def _short_sha(value: str | None) -> str:
    if not value:
        return "-"
    return value[:12]


def render_table(results: list[RecheckResult]) -> str:
    headers = ("pr", "parked_head", "live_head", "changed", "recommendation")
    rows = [
        (
            f"#{result.pr}",
            _short_sha(result.parked_head),
            _short_sha(result.live_head),
            "unknown" if result.changed is None else str(result.changed).lower(),
            result.recommendation,
        )
        for result in results
    ]
    widths = [
        max(len(str(item)) for item in column) for column in zip(headers, *rows, strict=False)
    ]
    lines = [
        " | ".join(str(item).ljust(width) for item, width in zip(headers, widths, strict=False)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            " | ".join(str(item).ljust(width) for item, width in zip(row, widths, strict=False))
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recheck parked PRs and flag heads that changed since parking."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--gh-bin", default="gh")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = load_ledger(args.ledger)
    except ValueError as exc:
        print(f"parked_pr_recheck: {exc}", file=sys.stderr)
        return 2

    results, ok = recheck_records(
        records,
        repo=args.repo,
        gh_bin=args.gh_bin,
        timeout=args.timeout,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "ledger": str(args.ledger),
                    "repo": args.repo,
                    "results": [asdict(result) for result in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_table(results))
        errors = [result for result in results if result.error]
        if errors:
            print("\nlookup errors:", file=sys.stderr)
            for result in errors:
                print(f"- PR #{result.pr}: {result.error}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
