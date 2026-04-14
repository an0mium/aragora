#!/usr/bin/env python3
"""Build a corpus-linked benchmark truth artifact for TW-01/TW-02."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.terminal_truth import TerminalClass  # noqa: E402
from scripts.reconcile_b0_pr_truth import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    GitHubTruthClient,
    IssueMetricsAggregate,
    reconcile_issue_truth,
    report_to_json as _unused_report_to_json,
    resolve_metrics_path,
    resolve_terminal_class,
    load_metrics_rows,
)

DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "benchmarks" / "corpus.json"


def load_corpus(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Corpus at {path} must be a JSON object")
    issues = payload.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"Corpus at {path} must contain a non-empty 'issues' list")
    return payload


def aggregate_corpus_issues(
    rows: list[dict[str, Any]],
    corpus: dict[str, Any],
) -> list[IssueMetricsAggregate]:
    by_issue: dict[int, IssueMetricsAggregate] = {}
    corpus_issues = [item for item in list(corpus.get("issues") or []) if isinstance(item, dict)]
    for item in corpus_issues:
        issue_number = int(item.get("issue_id", 0) or 0)
        if issue_number <= 0:
            continue
        by_issue[issue_number] = IssueMetricsAggregate(
            issue_number=issue_number,
            title=str(item.get("title") or "").strip(),
            row_count=0,
            proxy_pr_signal=False,
            had_rescue=False,
        )

    corpus_issue_numbers = set(by_issue)
    for row in rows:
        issue_number = row.get("issue_number")
        if not isinstance(issue_number, int) or issue_number not in corpus_issue_numbers:
            continue
        aggregate = by_issue[issue_number]
        aggregate.row_count += 1
        terminal_class = resolve_terminal_class(row)
        publish_action = str(row.get("publish_action", "") or "").strip().lower()
        worker_outcome = str(row.get("worker_outcome", "") or "").strip().lower()
        if (
            terminal_class is TerminalClass.DELIVERABLE_PR_CREATED
            or publish_action in {"pr_created", "existing_pr", "discovered_after_push"}
            or worker_outcome in {"pr_adopted"}
        ):
            aggregate.proxy_pr_signal = True
        if terminal_class.value.startswith("rescue_"):
            aggregate.had_rescue = True
    return [by_issue[number] for number in sorted(by_issue)]


def _failure_distributions(
    rows: list[dict[str, Any]],
    *,
    corpus_issue_numbers: set[int],
) -> tuple[dict[str, int], dict[str, int]]:
    failure_counts: Counter[str] = Counter()
    rescue_counts: Counter[str] = Counter()
    for row in rows:
        issue_number = row.get("issue_number")
        if not isinstance(issue_number, int) or issue_number not in corpus_issue_numbers:
            continue
        terminal_class = resolve_terminal_class(row).value
        if terminal_class.startswith("deliverable_") or terminal_class == "issue_already_resolved":
            continue
        failure_counts[terminal_class] += 1
        if terminal_class.startswith("rescue_"):
            rescue_counts[terminal_class] += 1
    return dict(sorted(failure_counts.items())), dict(sorted(rescue_counts.items()))


def _missing_corpus_issue_numbers(aggregates: list[IssueMetricsAggregate]) -> list[int]:
    return [
        aggregate.issue_number
        for aggregate in aggregates
        if aggregate.issue_number > 0 and aggregate.row_count <= 0
    ]


def build_benchmark_truth_artifact(
    *,
    repo: str,
    metrics_file: Path,
    corpus_path: Path,
    client: GitHubTruthClient | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = load_metrics_rows(metrics_file)
    corpus = load_corpus(corpus_path)
    aggregates = aggregate_corpus_issues(rows, corpus)
    truth_client = client or GitHubTruthClient()
    records = [reconcile_issue_truth(repo, aggregate, truth_client) for aggregate in aggregates]
    corpus_issue_count = len(aggregates)
    attempted_issue_count = sum(1 for aggregate in aggregates if aggregate.row_count > 0)
    truth_success_issue_count = sum(1 for record in records if record.truth_success)
    no_rescue_truth_success_issue_count = sum(
        1 for record in records if record.no_rescue_truth_success
    )
    merged_issue_count = sum(1 for record in records if record.truth_state == "merged_pr")
    proxy_pr_signal_issue_count = sum(1 for aggregate in aggregates if aggregate.proxy_pr_signal)
    missing_issue_numbers = _missing_corpus_issue_numbers(aggregates)
    run_complete = not missing_issue_numbers
    failure_class_distribution, rescue_counts_by_type = _failure_distributions(
        rows,
        corpus_issue_numbers={aggregate.issue_number for aggregate in aggregates},
    )
    return {
        "generated_at": generated_at or dt.datetime.now(dt.UTC).isoformat(),
        "repo": repo,
        "metrics_file": str(metrics_file),
        "corpus": {
            "path": str(corpus_path),
            "corpus_id": str(corpus.get("corpus_id") or "").strip(),
            "revision": int(corpus.get("revision", 0) or 0),
            "recorded_on": str(corpus.get("recorded_on") or "").strip(),
            "success_contract": str(corpus.get("success_contract") or "").strip(),
            "issue_count": corpus_issue_count,
        },
        "run_status": "complete" if run_complete else "incomplete",
        "coverage": {
            "attempted_issue_count": attempted_issue_count,
            "missing_issue_count": len(missing_issue_numbers),
            "missing_issue_numbers": missing_issue_numbers,
            "is_complete": run_complete,
            "status": "complete" if run_complete else "incomplete",
        },
        "primary_metrics": {
            "truth_success_rate": round(truth_success_issue_count / corpus_issue_count, 4)
            if corpus_issue_count
            else 0.0,
            "no_rescue_truth_success_rate": round(
                no_rescue_truth_success_issue_count / corpus_issue_count,
                4,
            )
            if corpus_issue_count
            else 0.0,
            "merged_only_rate": round(merged_issue_count / corpus_issue_count, 4)
            if corpus_issue_count
            else 0.0,
        },
        "proxy_metrics": {
            "attempted_issue_count": attempted_issue_count,
            "proxy_pr_signal_issue_count": proxy_pr_signal_issue_count,
            "proxy_pr_signal_issue_rate": round(
                proxy_pr_signal_issue_count / corpus_issue_count,
                4,
            )
            if corpus_issue_count
            else 0.0,
            "note": "Proxy metrics are secondary. Truth metrics remain mergeable_pr OR merged_pr.",
        },
        "failure_class_distribution": failure_class_distribution,
        "rescue_counts_by_type": rescue_counts_by_type,
        "issues": [record.to_dict() for record in records],
    }


def write_artifact(path: Path, artifact: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora", help="GitHub repo owner/name")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSONL file (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Benchmark corpus manifest (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional artifact output path")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument(
        "--fail-incomplete",
        action="store_true",
        help="Exit non-zero when the artifact does not cover every corpus issue",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_file = resolve_metrics_path(args.metrics_file)
    corpus_path = args.corpus.resolve()
    if not metrics_file.exists():
        raise SystemExit(f"metrics file not found: {metrics_file}")
    if not corpus_path.exists():
        raise SystemExit(f"corpus file not found: {corpus_path}")
    artifact = build_benchmark_truth_artifact(
        repo=str(args.repo),
        metrics_file=metrics_file,
        corpus_path=corpus_path,
    )
    if args.output is not None:
        output_path = write_artifact(args.output.resolve(), artifact)
        print(str(output_path))
    if args.json or args.output is None:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    if args.fail_incomplete and artifact.get("run_status") != "complete":
        missing_issue_numbers = list(
            (artifact.get("coverage") or {}).get("missing_issue_numbers") or []
        )
        missing_suffix = ", ".join(str(item) for item in missing_issue_numbers) or "unknown"
        print(
            f"incomplete corpus coverage: missing issue numbers {missing_suffix}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
