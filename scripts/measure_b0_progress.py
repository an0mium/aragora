#!/usr/bin/env python3
"""Measure B0 progress from boss metrics JSONL using issue-level cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.terminal_truth import TerminalClass, classify_from_metrics  # noqa: E402

DEFAULT_METRICS_PATH = REPO_ROOT / ".aragora" / "overnight" / "boss_metrics.jsonl"
COHORT_TAG = "[b0-cohort]"
COHORT_TAG_PLAIN = "b0-cohort"
PR_SIGNAL_ACTIONS = frozenset({"pr_created", "existing_pr", "discovered_after_push"})
PR_SIGNAL_OUTCOMES = frozenset({"pr_adopted"})
COMPLETED_STATUS = "completed"


@dataclass(frozen=True)
class CohortSummary:
    name: str
    rows: int
    unique_issues_attempted: int
    unique_issues_with_proxy_pr_signal: int
    unique_issues_with_completed_iteration: int
    deferred_publish_issue_count: int
    deferred_publish_event_count: int
    deferred_publish_issue_rate: float
    average_iterations_per_issue: float
    proxy_pr_signal_issue_rate: float
    completed_issue_rate: float
    terminal_class_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "unique_issues_attempted": self.unique_issues_attempted,
            "unique_issues_with_proxy_pr_signal": self.unique_issues_with_proxy_pr_signal,
            "unique_issues_with_completed_iteration": self.unique_issues_with_completed_iteration,
            "deferred_publish_issue_count": self.deferred_publish_issue_count,
            "deferred_publish_event_count": self.deferred_publish_event_count,
            "deferred_publish_issue_rate": round(self.deferred_publish_issue_rate, 4),
            "average_iterations_per_issue": round(self.average_iterations_per_issue, 4),
            "proxy_pr_signal_issue_rate": round(self.proxy_pr_signal_issue_rate, 4),
            "completed_issue_rate": round(self.completed_issue_rate, 4),
            "terminal_class_distribution": self.terminal_class_distribution,
        }


@dataclass
class IssueAggregate:
    iterations: int = 0
    has_proxy_pr_signal: bool = False
    has_completed_iteration: bool = False
    has_deferred_publish: bool = False


def load_metrics_rows(metrics_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _normalize_issue_key(row: dict[str, Any]) -> str | None:
    value = row.get("issue_number")
    if value is None or value == "":
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_cohort_tag(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.lower()
        return COHORT_TAG in normalized or COHORT_TAG_PLAIN in normalized
    if isinstance(value, dict):
        return any(_contains_cohort_tag(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_cohort_tag(item) for item in value)
    return False


def is_b0_cohort_row(row: dict[str, Any]) -> bool:
    if _contains_cohort_tag(row.get("cohort_tag")):
        return True
    for key in ("issue_title", "title"):
        if _contains_cohort_tag(row.get(key)):
            return True
    for key in ("metadata", "issue_metadata"):
        if _contains_cohort_tag(row.get(key)):
            return True
    return False


def is_v2_row(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("prompt_chars", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def is_decomposed_row(row: dict[str, Any]) -> bool:
    return row.get("is_decomposed_issue") is True


def resolve_terminal_class(row: dict[str, Any]) -> TerminalClass:
    existing = row.get("terminal_class")
    if isinstance(existing, str) and existing.strip():
        try:
            return TerminalClass(existing.strip())
        except ValueError:
            pass
    return classify_from_metrics(row)


def has_proxy_pr_signal(row: dict[str, Any], terminal_class: TerminalClass) -> bool:
    if terminal_class is TerminalClass.DELIVERABLE_PR_CREATED:
        return True
    publish_action = _normalize_text(row.get("publish_action"))
    worker_outcome = _normalize_text(row.get("worker_outcome"))
    return publish_action in PR_SIGNAL_ACTIONS or worker_outcome in PR_SIGNAL_OUTCOMES


def has_completed_iteration(row: dict[str, Any]) -> bool:
    return _normalize_text(row.get("worker_status")) == COMPLETED_STATUS


def is_deferred_publish(row: dict[str, Any], terminal_class: TerminalClass) -> bool:
    publish_action = _normalize_text(row.get("publish_action"))
    return "deferred" in publish_action or terminal_class is TerminalClass.RESCUE_PUBLISH_DEFERRED


def summarize_cohort(name: str, rows: list[dict[str, Any]]) -> CohortSummary:
    terminal_counts: Counter[str] = Counter()
    issues: dict[str, IssueAggregate] = {}

    for row in rows:
        terminal_class = resolve_terminal_class(row)
        terminal_counts[terminal_class.value] += 1

        issue_key = _normalize_issue_key(row)
        if issue_key is None:
            continue

        aggregate = issues.setdefault(issue_key, IssueAggregate())
        aggregate.iterations += 1
        if has_proxy_pr_signal(row, terminal_class):
            aggregate.has_proxy_pr_signal = True
        if has_completed_iteration(row):
            aggregate.has_completed_iteration = True
        if is_deferred_publish(row, terminal_class):
            aggregate.has_deferred_publish = True

    issue_count = len(issues)
    proxy_signal_count = sum(1 for issue in issues.values() if issue.has_proxy_pr_signal)
    completed_count = sum(1 for issue in issues.values() if issue.has_completed_iteration)
    deferred_issue_count = sum(1 for issue in issues.values() if issue.has_deferred_publish)
    deferred_event_count = sum(
        1 for row in rows if is_deferred_publish(row, resolve_terminal_class(row))
    )
    total_iterations = sum(issue.iterations for issue in issues.values())

    return CohortSummary(
        name=name,
        rows=len(rows),
        unique_issues_attempted=issue_count,
        unique_issues_with_proxy_pr_signal=proxy_signal_count,
        unique_issues_with_completed_iteration=completed_count,
        deferred_publish_issue_count=deferred_issue_count,
        deferred_publish_event_count=deferred_event_count,
        deferred_publish_issue_rate=(deferred_issue_count / issue_count if issue_count else 0.0),
        average_iterations_per_issue=(total_iterations / issue_count if issue_count else 0.0),
        proxy_pr_signal_issue_rate=(proxy_signal_count / issue_count if issue_count else 0.0),
        completed_issue_rate=(completed_count / issue_count if issue_count else 0.0),
        terminal_class_distribution=dict(sorted(terminal_counts.items())),
    )


def measure_b0_progress(rows: list[dict[str, Any]]) -> dict[str, CohortSummary]:
    cohorts = {
        "all": rows,
        "v2": [row for row in rows if is_v2_row(row)],
        "decomposed": [row for row in rows if is_decomposed_row(row)],
        "b0_tagged": [row for row in rows if is_b0_cohort_row(row)],
    }
    return {name: summarize_cohort(name, cohort_rows) for name, cohort_rows in cohorts.items()}


def render_table(report: dict[str, CohortSummary]) -> str:
    headers = [
        "Cohort",
        "Rows",
        "Issues",
        "PR signal (proxy)",
        "Proxy rate",
        "Completed",
        "Deferred",
        "Avg iters",
    ]
    body_rows: list[list[str]] = []
    for name, summary in report.items():
        body_rows.append(
            [
                name,
                str(summary.rows),
                str(summary.unique_issues_attempted),
                str(summary.unique_issues_with_proxy_pr_signal),
                f"{summary.proxy_pr_signal_issue_rate:.1%}",
                str(summary.unique_issues_with_completed_iteration),
                f"{summary.deferred_publish_issue_count} ({summary.deferred_publish_issue_rate:.1%})",
                f"{summary.average_iterations_per_issue:.2f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in body_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    lines = [_fmt(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(_fmt(row) for row in body_rows)
    lines.append("")
    lines.append("Terminal class distribution:")
    for name, summary in report.items():
        lines.append(f"  {name}:")
        if summary.terminal_class_distribution:
            for terminal_class, count in summary.terminal_class_distribution.items():
                lines.append(f"    - {terminal_class}: {count}")
        else:
            lines.append("    - none")
    return "\n".join(lines)


def report_to_json(metrics_file: Path, report: dict[str, CohortSummary]) -> str:
    payload = {
        "metrics_file": str(metrics_file),
        "proxy_metric_note": (
            "PR signal is a proxy from metrics, not GitHub truth or merged-PR truth."
        ),
        "cohorts": {name: summary.to_dict() for name, summary in report.items()},
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure B0 progress from boss metrics JSONL.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to boss_metrics.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.metrics_file.exists():
        print(f"Metrics file not found: {args.metrics_file}", file=sys.stderr)
        return 1

    rows = load_metrics_rows(args.metrics_file)
    report = measure_b0_progress(rows)
    if args.json:
        print(report_to_json(args.metrics_file, report))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
