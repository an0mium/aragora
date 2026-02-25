#!/usr/bin/env python3
"""Epistemic compliance regression checker.

Runs deterministic fixture cases through ``score_response`` and enforces
threshold-based regression gates for:
- per-model aggregate compliance metrics
- targeted adversarial case ceilings

Usage:
    python scripts/check_epistemic_compliance_regression.py
    python scripts/check_epistemic_compliance_regression.py --strict
    python scripts/check_epistemic_compliance_regression.py --strict \
      --fixtures scripts/fixtures/epistemic_compliance_fixtures.json \
      --baseline scripts/baselines/epistemic_compliance_regression.json \
      --json-out /tmp/epistemic.json --md-out /tmp/epistemic.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FIXTURES = PROJECT_ROOT / "scripts" / "fixtures" / "epistemic_compliance_fixtures.json"
DEFAULT_BASELINE = PROJECT_ROOT / "scripts" / "baselines" / "epistemic_compliance_regression.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def _compute_metrics(cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    from aragora.debate.epistemic_hygiene import score_response

    case_results: dict[str, Any] = {}
    model_buckets: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        case_id = str(case.get("id", "")).strip()
        model = str(case.get("model", "unknown")).strip() or "unknown"
        text = str(case.get("text", ""))
        if not case_id:
            continue

        score = score_response(text, agent=model)
        result = {
            "id": case_id,
            "model": model,
            "score": round(score.score, 4),
            "has_alternatives": bool(score.has_alternatives),
            "has_falsifiers": bool(score.has_falsifiers),
            "has_confidence": bool(score.has_confidence),
            "has_unknowns": bool(score.has_unknowns),
            "tags": case.get("tags", []),
        }
        case_results[case_id] = result
        model_buckets.setdefault(model, []).append(result)

    model_metrics: dict[str, Any] = {}
    for model, rows in model_buckets.items():
        n = len(rows)
        if n == 0:
            continue
        model_metrics[model] = {
            "samples": n,
            "avg_score": round(sum(r["score"] for r in rows) / n, 4),
            "alternatives_rate": round(sum(1 for r in rows if r["has_alternatives"]) / n, 4),
            "falsifiers_rate": round(sum(1 for r in rows if r["has_falsifiers"]) / n, 4),
            "confidence_rate": round(sum(1 for r in rows if r["has_confidence"]) / n, 4),
            "unknowns_rate": round(sum(1 for r in rows if r["has_unknowns"]) / n, 4),
        }

    return model_metrics, case_results


def _check_thresholds(
    model_metrics: dict[str, Any],
    case_results: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    regressions: list[str] = []

    metric_key_map = {
        "avg_score": "avg_score",
        "alternatives_rate": "alternatives_rate",
        "falsifiers_rate": "falsifiers_rate",
        "confidence_rate": "confidence_rate",
        "unknowns_rate": "unknowns_rate",
    }

    global_cfg = baseline.get("global", {})
    min_total_cases = int(global_cfg.get("min_total_cases", 0))
    if len(case_results) < min_total_cases:
        regressions.append(
            f"global.min_total_cases violated: expected >= {min_total_cases}, got {len(case_results)}"
        )

    for model_name, thresholds in baseline.get("models", {}).items():
        metrics = model_metrics.get(model_name)
        if metrics is None:
            regressions.append(f"missing model metrics for baseline model: {model_name}")
            continue

        for key, threshold in thresholds.items():
            if not isinstance(threshold, (int, float)):
                continue
            if key.startswith("min_"):
                metric_name = key.removeprefix("min_")
                mapped = metric_key_map.get(metric_name)
                if mapped is None:
                    continue
                value = float(metrics.get(mapped, 0.0))
                if value < float(threshold):
                    regressions.append(
                        f"{model_name}.{mapped} below threshold: {value:.4f} < {float(threshold):.4f}"
                    )
            elif key.startswith("max_"):
                metric_name = key.removeprefix("max_")
                mapped = metric_key_map.get(metric_name)
                if mapped is None:
                    continue
                value = float(metrics.get(mapped, 0.0))
                if value > float(threshold):
                    regressions.append(
                        f"{model_name}.{mapped} above threshold: {value:.4f} > {float(threshold):.4f}"
                    )

    for case_id, thresholds in baseline.get("cases", {}).items():
        case_metric = case_results.get(case_id)
        if case_metric is None:
            regressions.append(f"missing case metrics for baseline case: {case_id}")
            continue
        score = float(case_metric.get("score", 0.0))
        max_score = thresholds.get("max_score")
        min_score = thresholds.get("min_score")
        if isinstance(max_score, (int, float)) and score > float(max_score):
            regressions.append(
                f"{case_id}.score above threshold: {score:.4f} > {float(max_score):.4f}"
            )
        if isinstance(min_score, (int, float)) and score < float(min_score):
            regressions.append(
                f"{case_id}.score below threshold: {score:.4f} < {float(min_score):.4f}"
            )

    return regressions


def _to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Epistemic Compliance Summary")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Fixture cases: {summary['fixture_case_count']}")
    lines.append(f"- Baseline path: `{summary['baseline_path']}`")
    lines.append(f"- Result: {'PASS' if summary['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("## Per-Model Metrics")
    lines.append("")
    lines.append(
        "| Model | Samples | Avg Score | Alt Rate | Falsifier Rate | Confidence Rate | Unknowns Rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, metrics in sorted(summary["models"].items()):
        lines.append(
            "| "
            f"{model} | {metrics['samples']} | {metrics['avg_score']:.4f} | "
            f"{metrics['alternatives_rate']:.4f} | {metrics['falsifiers_rate']:.4f} | "
            f"{metrics['confidence_rate']:.4f} | {metrics['unknowns_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Regressions")
    lines.append("")
    if summary["regressions"]:
        for item in summary["regressions"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check epistemic compliance regression thresholds."
    )
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on regressions.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    fixture_doc = _load_json(args.fixtures)
    baseline_doc = _load_json(args.baseline)
    cases = fixture_doc.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError("fixtures JSON must contain a list field named 'cases'")

    model_metrics, case_results = _compute_metrics(cases)
    regressions = _check_thresholds(model_metrics, case_results, baseline_doc)

    summary: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "fixtures_path": str(args.fixtures),
        "baseline_path": str(args.baseline),
        "fixture_case_count": len(case_results),
        "models": model_metrics,
        "cases": case_results,
        "regressions": regressions,
        "passed": len(regressions) == 0,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_content = _to_markdown(summary)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md_content, encoding="utf-8")

    print(md_content)

    if args.strict and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
