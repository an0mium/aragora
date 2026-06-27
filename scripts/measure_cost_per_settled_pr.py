#!/usr/bin/env python3
"""Measure recorded model cost per settled PR (#8233 phase 1 instrument).

cost_per_settled_pr = total RECORDED model cost in a window / PRs merged
("settled") in the same window.

Honesty contract (mirrors ``scripts/measure_leverage_ratio.py``):

- Only *recorded* cost is summed. Cost sources are routing-rationale records
  (``scripts/auto_evidence_cycle.py`` writes them per applied collect run,
  schema ``aragora.routing_rationale/v1``) whose ``cost.recorded`` is true,
  plus DecisionReceipt JSON files carrying a ``cost_summary.total_cost_usd``.
  Nothing is ever estimated or back-filled: a settled PR without any cost
  record stays uncovered and is disclosed as such.
- The published number is therefore a **lower bound** on true cost. Coverage
  (settled PRs with at least one attributed cost record) is published next to
  the ratio so the gap is visible, not hidden.
- Double-count guard: every counted artifact is deduplicated by resolved path
  across all scanned directories, a file contributes through exactly one
  source category (routing record first, receipt otherwise), and every counted
  artifact is listed with its amount in ``cost_sources`` so an auditor can
  re-add the total.

``--publish`` renders/updates a dedicated marker-delimited region in
``docs/status/LEVERAGE.md`` (outside the existing ``leverage-managed`` region,
so neither publisher ever rewrites the other's section).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

METHODOLOGY_VERSION = 1
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_ROUTING_RECORDS_DIRS = (".aragora/automation-receipts/routing",)
DEFAULT_RECEIPTS_DIRS = (".aragora/run-20260610/receipts",)
DEFAULT_STATUS_DOC = "docs/status/LEVERAGE.md"

COST_BEGIN = "<!-- cost-per-settled-pr:begin -->"
COST_END = "<!-- cost-per-settled-pr:end -->"

COVERAGE_CAVEAT = """\
**Coverage caveat (required reading).** Only *recorded* model cost is summed —
routing-rationale records with `cost.recorded: true` and receipts carrying a
`cost_summary`. Settled PRs without any cost record are NOT estimated; they are
counted in the denominator and disclosed as uncovered, so the ratio is a lower
bound on true cost. Unattributed receipt costs (receipts do not carry PR
numbers) are included in the total and disclosed separately. Recording starts
with #8233 phase 1; coverage is expected to grow from near zero.\
"""


# ---------------------------------------------------------------------------
# Subprocess boundary (mockable)
# ---------------------------------------------------------------------------


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=True, timeout=300)
    return proc.stdout


def fetch_merged_prs(repo: str, since: datetime) -> list[dict]:
    """Fetch PRs merged at/after ``since`` (same query as measure_leverage_ratio)."""
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = f"repo:{repo} is:pr is:merged merged:>={since_str}"
    out = _run_gh(
        [
            "api",
            "--paginate",
            "-X",
            "GET",
            "search/issues",
            "-f",
            f"q={query}",
            "--jq",
            ".items[] | {number: .number, merged_at: .pull_request.merged_at}",
        ]
    )
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_cost(value: object) -> float | None:
    """Parse a recorded cost amount; None when absent/unparseable/negative."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(str(value))
    except (TypeError, ValueError):
        return None
    if amount < 0:
        return None
    return amount


def load_json_artifacts(dirs: Sequence[Path]) -> list[tuple[Path, dict]]:
    """Load ``*.json`` objects from dirs, deduplicated by resolved path."""
    seen: set[Path] = set()
    out: list[tuple[Path, dict]] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                out.append((resolved, {"_unreadable": True}))
                continue
            if isinstance(payload, dict):
                out.append((resolved, payload))
    return out


def compute_cost_per_settled_pr(
    *,
    merged_prs: list[dict],
    routing_artifacts: list[tuple[Path, dict]],
    receipt_artifacts: list[tuple[Path, dict]],
    window_start: datetime,
    window_end: datetime,
    window_days: int,
    repo: str,
) -> dict:
    """Aggregate recorded cost over the window against settled PRs.

    Every artifact contributes through exactly one source category: a file
    that parses as a routing-rationale record is consumed there and never
    re-scanned as a receipt (the same resolved path is skipped), so an
    artifact's cost can be counted at most once.
    """
    settled_numbers: set[int] = set()
    for pr in merged_prs:
        try:
            settled_numbers.add(int(pr["number"]))
        except (KeyError, TypeError, ValueError):
            continue
    settled_total = len(settled_numbers)

    cost_sources: list[dict] = []
    consumed_paths: set[Path] = set()

    records_in_window = 0
    records_with_cost = 0
    records_without_cost = 0
    records_malformed_cost: list[str] = []
    records_outside_window = 0
    attributed_cost = 0.0
    covered_prs: set[int] = set()

    for path, payload in routing_artifacts:
        if payload.get("_unreadable"):
            continue
        if str(payload.get("record_type") or "") != "routing_rationale":
            continue
        consumed_paths.add(path)
        if str(payload.get("repo") or "") not in ("", repo):
            continue
        when = _parse_ts(str(payload.get("generated_at") or ""))
        if when is None or not (window_start <= when <= window_end):
            records_outside_window += 1
            continue
        records_in_window += 1
        raw_cost = payload.get("cost")
        cost: dict = raw_cost if isinstance(raw_cost, dict) else {}
        if not cost.get("recorded"):
            records_without_cost += 1
            continue
        amount = _parse_cost(cost.get("total_usd"))
        if amount is None:
            # recorded=true but unusable amount: disclosed, never guessed.
            records_malformed_cost.append(str(path))
            continue
        records_with_cost += 1
        attributed_cost += amount
        cost_sources.append({"path": str(path), "kind": "routing_record", "usd": amount})
        raw_pr = payload.get("pr")
        try:
            pr_number = int(raw_pr) if raw_pr is not None else None
        except (TypeError, ValueError):
            pr_number = None
        if pr_number is not None and pr_number in settled_numbers:
            covered_prs.add(pr_number)

    receipts_scanned = 0
    receipts_with_cost = 0
    receipts_without_cost = 0
    receipts_skipped_no_timestamp = 0
    receipts_outside_window = 0
    receipts_unreadable = 0
    unattributed_cost = 0.0

    for path, payload in receipt_artifacts:
        if path in consumed_paths:
            continue  # already counted as a routing record; never twice
        if payload.get("_unreadable"):
            receipts_unreadable += 1
            continue
        if str(payload.get("record_type") or "") == "routing_rationale":
            continue  # routing record found via a receipts dir: routing pass owns it
        receipts_scanned += 1
        summary = payload.get("cost_summary")
        amount = _parse_cost(summary.get("total_cost_usd")) if isinstance(summary, dict) else None
        if amount is None:
            receipts_without_cost += 1
            continue
        when = _parse_ts(str(payload.get("timestamp") or ""))
        if when is None:
            receipts_skipped_no_timestamp += 1
            continue
        if not (window_start <= when <= window_end):
            receipts_outside_window += 1
            continue
        receipts_with_cost += 1
        unattributed_cost += amount
        cost_sources.append({"path": str(path), "kind": "receipt_cost_summary", "usd": amount})

    total_recorded = attributed_cost + unattributed_cost
    return {
        "methodology_version": METHODOLOGY_VERSION,
        "repo": repo,
        "window_days": window_days,
        "window_start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_end": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settled_prs_total": settled_total,
        "settled_prs_with_cost_record": len(covered_prs),
        "covered_pr_numbers": sorted(covered_prs),
        "coverage_ratio": (len(covered_prs) / settled_total) if settled_total else None,
        "attributed_recorded_cost_usd": round(attributed_cost, 6),
        "unattributed_recorded_cost_usd": round(unattributed_cost, 6),
        "total_recorded_cost_usd": round(total_recorded, 6),
        "cost_per_settled_pr_usd": (
            round(total_recorded / settled_total, 6) if settled_total else None
        ),
        "cost_is_lower_bound": True,
        "routing_records_in_window": records_in_window,
        "routing_records_with_cost": records_with_cost,
        "routing_records_without_cost": records_without_cost,
        "routing_records_outside_window": records_outside_window,
        "routing_records_malformed_cost": records_malformed_cost,
        "receipts_scanned": receipts_scanned,
        "receipts_with_cost": receipts_with_cost,
        "receipts_without_cost": receipts_without_cost,
        "receipts_skipped_no_timestamp": receipts_skipped_no_timestamp,
        "receipts_outside_window": receipts_outside_window,
        "receipts_unreadable": receipts_unreadable,
        "cost_sources": cost_sources,
    }


# ---------------------------------------------------------------------------
# LEVERAGE.md publication (own managed region, outside leverage-managed)
# ---------------------------------------------------------------------------


def render_cost_block(result: dict, now_iso: str) -> str:
    coverage = result["coverage_ratio"]
    coverage_text = f"{coverage:.0%}" if coverage is not None else "n/a (0 settled PRs)"
    ratio = result["cost_per_settled_pr_usd"]
    ratio_text = (
        f"${ratio:.4f} (lower bound; see coverage)"
        if ratio is not None
        else "n/a (0 settled PRs in window)"
    )
    rows = [
        (
            "Window",
            f"{result['window_start']} -> {result['window_end']} ({result['window_days']}d)",
        ),
        ("Settled (merged) PRs in window", result["settled_prs_total"]),
        (
            "Settled PRs with attributed cost record",
            f"{result['settled_prs_with_cost_record']} ({coverage_text} coverage)",
        ),
        ("Attributed recorded cost (USD)", f"{result['attributed_recorded_cost_usd']:.4f}"),
        (
            "Unattributed recorded cost (USD, receipts)",
            f"{result['unattributed_recorded_cost_usd']:.4f}",
        ),
        ("Total recorded model cost (USD)", f"{result['total_recorded_cost_usd']:.4f}"),
        ("Recorded cost per settled PR", ratio_text),
        (
            "Routing records in window (with / without cost)",
            f"{result['routing_records_in_window']} "
            f"({result['routing_records_with_cost']} / "
            f"{result['routing_records_without_cost']})",
        ),
        (
            "Receipts scanned (with cost / without / no timestamp)",
            f"{result['receipts_scanned']} ({result['receipts_with_cost']} / "
            f"{result['receipts_without_cost']} / "
            f"{result['receipts_skipped_no_timestamp']})",
        ),
        ("Methodology version", result["methodology_version"]),
    ]
    lines = [
        "## Cost per settled PR (#8233 phase 1)",
        "",
        f"Last updated: {now_iso}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in rows)
    lines.append("")
    lines.append(COVERAGE_CAVEAT)
    if result["routing_records_malformed_cost"]:
        lines.append("")
        lines.append(
            "Routing records with recorded-but-unparseable cost (excluded, never guessed):"
        )
        lines.extend(f"- `{p}`" for p in result["routing_records_malformed_cost"])
    return "\n".join(lines)


def update_status_doc(path: Path, block: str) -> str:
    """Create/update only the cost-per-settled-pr region of the status doc."""
    region = f"{COST_BEGIN}\n{block}\n{COST_END}"
    existing = path.read_text() if path.exists() else ""
    if COST_BEGIN in existing and COST_END in existing:
        start = existing.index(COST_BEGIN)
        stop = existing.index(COST_END) + len(COST_END)
        new_text = existing[:start] + region + existing[stop:]
    elif existing:
        new_text = existing.rstrip("\n") + "\n\n" + region + "\n"
    else:
        new_text = "# Leverage & Waste Status\n\n" + region + "\n"
    if not new_text.endswith("\n"):
        new_text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    return new_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--since",
        default=None,
        help="ISO8601 UTC window start override (default: now - window-days).",
    )
    parser.add_argument(
        "--routing-records-dir",
        action="append",
        default=None,
        help="Routing-rationale record dir(s) "
        f"(repeatable; default: {DEFAULT_ROUTING_RECORDS_DIRS[0]}).",
    )
    parser.add_argument(
        "--receipts-dir",
        action="append",
        default=None,
        help="DecisionReceipt JSON dir(s) scanned for cost_summary "
        f"(repeatable; default: {DEFAULT_RECEIPTS_DIRS[0]}).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--status-doc", default=DEFAULT_STATUS_DOC)
    args = parser.parse_args(argv)

    window_end = datetime.now(timezone.utc)
    if args.since:
        window_start = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        if window_start.tzinfo is None:
            window_start = window_start.replace(tzinfo=timezone.utc)
    else:
        window_start = window_end - timedelta(days=args.window_days)

    routing_dirs = [Path(d) for d in (args.routing_records_dir or DEFAULT_ROUTING_RECORDS_DIRS)]
    receipts_dirs = [Path(d) for d in (args.receipts_dir or DEFAULT_RECEIPTS_DIRS)]

    merged_prs = fetch_merged_prs(args.repo, window_start)
    result = compute_cost_per_settled_pr(
        merged_prs=merged_prs,
        routing_artifacts=load_json_artifacts(routing_dirs),
        receipt_artifacts=load_json_artifacts(receipts_dirs),
        window_start=window_start,
        window_end=window_end,
        window_days=args.window_days,
        repo=args.repo,
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        ratio = result["cost_per_settled_pr_usd"]
        ratio_text = f"${ratio:.4f}" if ratio is not None else "n/a"
        print(
            f"cost_per_settled_pr={ratio_text} (lower bound; "
            f"{result['total_recorded_cost_usd']:.4f} USD recorded over "
            f"{result['settled_prs_total']} settled PRs; coverage "
            f"{result['settled_prs_with_cost_record']}/{result['settled_prs_total']})"
        )

    if args.publish:
        doc = Path(args.status_doc)
        now_iso = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        update_status_doc(doc, render_cost_block(result, now_iso))
        print(f"published cost-per-settled-PR section to {doc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
