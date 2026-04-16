#!/usr/bin/env python3
"""Week-over-week B0 scorecards from benchmark truth reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CORPUS = Path("docs/benchmarks/corpus.json")


# fmt: off
def _pct(summary: dict[str, Any], key: str) -> float:
    """Coerce a percentage-like field to float.

    >>> _pct({"rate": "0.4"}, "rate")
    0.4
    """
    try:
        return float(summary.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(summary: dict[str, Any], key: str) -> int:
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("summary"), dict):
        return payload["summary"]

    coverage = _nested_dict(payload, "coverage")
    primary_metrics = _nested_dict(payload, "primary_metrics")
    if coverage or primary_metrics:
        return {
            "attempted_issue_count": coverage.get("attempted_issue_count"),
            "truth_success_rate": primary_metrics.get("truth_success_rate"),
            "no_rescue_truth_success_rate": primary_metrics.get(
                "no_rescue_truth_success_rate"
            ),
            "merged_issue_rate": primary_metrics.get("merged_issue_rate")
            if "merged_issue_rate" in primary_metrics
            else primary_metrics.get("merged_only_rate"),
        }

    return payload


def _rescue_count(payload: dict[str, Any]) -> int:
    issues = payload.get("issues")
    if isinstance(issues, list):
        return sum(
            1
            for issue in issues
            if isinstance(issue, dict) and issue.get("had_rescue")
        )

    rescue_counts = _nested_dict(payload, "rescue_counts_by_type")
    return sum(_int(rescue_counts, key) for key in rescue_counts)

def _run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _run_summary(payload)
    stamp = str(payload.get("generated_at") or payload.get("recorded_on") or "").strip()
    if not stamp:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
    return {
        "date": stamp[:10],
        "attempted_issue_count": _int(summary, "attempted_issue_count"),
        "truth_success_rate": _pct(summary, "truth_success_rate"),
        "no_rescue_truth_success_rate": _pct(summary, "no_rescue_truth_success_rate"),
        "merged_issue_rate": _pct(summary, "merged_issue_rate"),
        "rescue_count": _rescue_count(payload),
    }

def build_scorecard(report_paths: list[Path], corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8")) if corpus_path.exists() else {}
    runs = sorted((_run(path) for path in report_paths), key=lambda run: run["date"])
    previous = None
    for run in runs:
        delta = None if previous is None else round((run["no_rescue_truth_success_rate"] - previous) * 100, 1)
        run["delta_no_rescue_pp"] = delta
        run["trend"] = "baseline" if delta is None else "improving" if delta > 0 else "degrading" if delta < 0 else "stable"
        previous = run["no_rescue_truth_success_rate"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {"corpus_id": corpus.get("corpus_id"), "revision": corpus.get("revision")},
        "latest_trend": runs[-1]["trend"] if runs else "no_data",
        "runs": runs,
    }

def render_scorecard(scorecard: dict[str, Any]) -> str:
    corpus = scorecard.get("corpus", {})
    lines = [f"Corpus: {corpus.get('corpus_id')} r{corpus.get('revision')}", "Date       Truth  No rescue  Merged  Rescue  Trend"]
    for run in scorecard.get("runs", []):
        trend = run["trend"] if run["delta_no_rescue_pp"] is None else f"{run['trend']} ({run['delta_no_rescue_pp']:+.1f}pp)"
        lines.append(f"{run['date']}  {run['truth_success_rate']:>5.0%}    {run['no_rescue_truth_success_rate']:>5.0%}   {run['merged_issue_rate']:>5.0%}   {run['rescue_count']:>6}  {trend}")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="JSON reports from reconcile_b0_pr_truth.py --json")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON scorecard")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text table")
    args = parser.parse_args(argv)
    scorecard = build_scorecard(args.reports, args.corpus)
    payload = json.dumps(scorecard, indent=2, sort_keys=True)
    print(payload if args.json else render_scorecard(scorecard))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0
# fmt: on

if __name__ == "__main__":
    raise SystemExit(main())
