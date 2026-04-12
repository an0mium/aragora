#!/usr/bin/env python3
"""Analyze boss loop metrics and outcome signals.

Outputs a JSON summary followed by a readable text report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from aragora.swarm.outcome_signals import OutcomeSignal, snapshot_outcome_signals
from aragora.swarm.terminal_truth import score_benchmark


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_signals(path: Path) -> list[OutcomeSignal]:
    signals: list[OutcomeSignal] = []
    for payload in _load_jsonl(path):
        data = {k: v for k, v in payload.items() if k in OutcomeSignal.__dataclass_fields__}
        try:
            signals.append(OutcomeSignal(**data))
        except TypeError:
            continue
    return signals


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def analyze_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars = 0
    enriched_context_chars = 0
    has_deliverable_count = 0
    publish_actions: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    worker_statuses: Counter[str] = Counter()
    decomposed_counts: Counter[str] = Counter()

    total = 0
    for record in records:
        total += 1
        prompt_chars += _safe_int(record.get("prompt_chars"))
        enriched_context_chars += _safe_int(record.get("enriched_context_chars"))
        if record.get("has_deliverable") is True:
            has_deliverable_count += 1

        publish_action = record.get("publish_action")
        if isinstance(publish_action, str) and publish_action.strip():
            publish_actions[publish_action.strip()] += 1

        worker_outcome = record.get("worker_outcome")
        if isinstance(worker_outcome, str) and worker_outcome.strip():
            outcomes[worker_outcome.strip()] += 1

        worker_status = record.get("worker_status")
        if isinstance(worker_status, str) and worker_status.strip():
            worker_statuses[worker_status.strip()] += 1

        if record.get("is_decomposed_issue") is True:
            decomposed_counts["decomposed"] += 1
        else:
            decomposed_counts["non_decomposed"] += 1

    avg_prompt = prompt_chars / total if total else 0.0
    avg_context = enriched_context_chars / total if total else 0.0

    failure_taxonomy: Counter[str] = Counter()
    for status, count in worker_statuses.items():
        if status != "completed":
            failure_taxonomy[f"worker_status:{status}"] += count
    for outcome, count in outcomes.items():
        if outcome and outcome != "deliverable_created":
            failure_taxonomy[f"worker_outcome:{outcome}"] += count

    return {
        "totals": {"records": total},
        "prompt_chars": {
            "total": prompt_chars,
            "avg": avg_prompt,
        },
        "enriched_context_chars": {
            "total": enriched_context_chars,
            "avg": avg_context,
        },
        "deliverables": {
            "count": has_deliverable_count,
            "rate": has_deliverable_count / total if total else 0.0,
        },
        "publish_actions": dict(sorted(publish_actions.items())),
        "worker_outcomes": dict(sorted(outcomes.items())),
        "worker_statuses": dict(sorted(worker_statuses.items())),
        "decomposed": dict(sorted(decomposed_counts.items())),
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
    }


def analyze_boss_metrics(*, metrics_path: Path, signals_path: Path | None) -> dict[str, Any]:
    metrics_records = _load_jsonl(metrics_path)
    metrics_summary = analyze_metrics(metrics_records)
    terminal_truth_benchmark = score_benchmark(metrics_records)
    signals_summary: dict[str, Any] | None = None
    if signals_path:
        signals = _load_signals(signals_path)
        signals_summary = snapshot_outcome_signals(signals).to_dict()
    return {
        "metrics_summary": metrics_summary,
        "terminal_truth_benchmark": terminal_truth_benchmark,
        "signals_summary": signals_summary,
    }


def render_text(report: dict[str, Any]) -> str:
    metrics = report.get("metrics_summary") or {}
    totals = metrics.get("totals", {})
    prompt = metrics.get("prompt_chars", {})
    context = metrics.get("enriched_context_chars", {})
    deliverables = metrics.get("deliverables", {})
    publish_actions = metrics.get("publish_actions", {})
    failure_taxonomy = metrics.get("failure_taxonomy", {})
    terminal_truth = report.get("terminal_truth_benchmark") or {}
    no_rescue_rate = float(terminal_truth.get("no_rescue_rate", 0.0) or 0.0)
    meets_target = bool(terminal_truth.get("meets_30d_target", False))
    actionable_failures = int(terminal_truth.get("actionable_failures", 0) or 0)
    terminal_families = terminal_truth.get("families", {})
    terminal_classes = terminal_truth.get("classes", {})

    lines = [
        "Boss Metrics Summary",
        f"  records: {totals.get('records', 0)}",
        f"  prompt_chars avg: {prompt.get('avg', 0):.1f}",
        f"  enriched_context_chars avg: {context.get('avg', 0):.1f}",
        f"  deliverable rate: {deliverables.get('rate', 0):.0%}",
        f"  terminal-truth no-rescue rate: {no_rescue_rate:.0%}",
        f"  terminal-truth meets 30d target: {meets_target}",
        f"  terminal-truth actionable failures: {actionable_failures}",
        "  publish actions:",
    ]
    for action, count in publish_actions.items():
        lines.append(f"    - {action}: {count}")
    if not publish_actions:
        lines.append("    - none")

    if failure_taxonomy:
        lines.append("  failure taxonomy:")
        for reason, count in failure_taxonomy.items():
            lines.append(f"    - {reason}: {count}")
    if terminal_families:
        lines.append("  terminal-truth families:")
        for family, count in sorted(terminal_families.items()):
            lines.append(f"    - {family}: {count}")
    if terminal_classes:
        lines.append("  terminal-truth classes:")
        for terminal_class, count in sorted(terminal_classes.items()):
            lines.append(f"    - {terminal_class}: {count}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze boss metrics JSONL.")
    parser.add_argument(
        "--metrics-file",
        required=True,
        type=Path,
        help="Path to boss_metrics.jsonl file",
    )
    parser.add_argument(
        "--signals-file",
        type=Path,
        default=None,
        help="Optional outcome_signals.jsonl file",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = analyze_boss_metrics(
        metrics_path=args.metrics_file,
        signals_path=args.signals_file,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
