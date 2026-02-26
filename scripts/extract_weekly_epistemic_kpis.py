#!/usr/bin/env python3
"""Extract weekly runtime KPIs for settlement + oracle quality signals.

Pulls data from the observability dashboard endpoint (or an input JSON fixture),
normalizes key metrics, and emits machine- and human-readable reports.

Usage:
    python scripts/extract_weekly_epistemic_kpis.py \
      --dashboard-url https://api.aragora.ai/api/v1/observability/dashboard \
      --token "$ARAGORA_API_TOKEN" \
      --json-out /tmp/weekly_kpis.json \
      --md-out /tmp/weekly_kpis.md

    python scripts/extract_weekly_epistemic_kpis.py \
      --input-json /tmp/dashboard.json --strict
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DASHBOARD_URL = "https://api.aragora.ai/api/v1/observability/dashboard"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_dashboard_json(url: str, *, token: str | None, timeout_seconds: int) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dashboard request failed ({exc.code}): {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Dashboard request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dashboard response is not valid JSON: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Input JSON is invalid: {path} ({exc})") from exc


def _metric_status(
    *,
    value: float | int | None,
    min_allowed: float | int | None = None,
    max_allowed: float | int | None = None,
) -> str:
    if value is None:
        return "unknown"
    if min_allowed is not None and value < min_allowed:
        return "fail"
    if max_allowed is not None and value > max_allowed:
        return "fail"
    return "pass"


def _extract_summary(
    payload: dict[str, Any],
    *,
    source: str,
    min_settlement_success_rate: float,
    max_oracle_stall_rate: float,
    min_calibration_updates: int,
) -> dict[str, Any]:
    settlement = payload.get("settlement_review")
    settlement = settlement if isinstance(settlement, dict) else {}
    settlement_stats = settlement.get("stats")
    settlement_stats = settlement_stats if isinstance(settlement_stats, dict) else {}
    settlement_last = settlement_stats.get("last_result")
    settlement_last = settlement_last if isinstance(settlement_last, dict) else {}

    calibration = settlement.get("calibration_outcomes")
    calibration = calibration if isinstance(calibration, dict) else {}

    oracle = payload.get("oracle_stream")
    oracle = oracle if isinstance(oracle, dict) else {}

    settlement_success_rate = _as_float(settlement_stats.get("success_rate"))
    settlement_unresolved_due = _as_int(settlement_last.get("unresolved_due"))
    settlement_total_runs = _as_int(settlement_stats.get("total_runs"))

    calibration_correct = _as_int(calibration.get("correct")) or 0
    calibration_incorrect = _as_int(calibration.get("incorrect")) or 0
    calibration_updates_realized = calibration_correct + calibration_incorrect

    oracle_sessions_started = _as_int(oracle.get("sessions_started")) or 0
    oracle_stalls_total = _as_int(oracle.get("stalls_total")) or 0
    oracle_stall_rate = (
        round(oracle_stalls_total / oracle_sessions_started, 4)
        if oracle_sessions_started > 0
        else None
    )

    metric_rows: dict[str, dict[str, Any]] = {
        "settlement_success_rate": {
            "label": "Settlement review success rate",
            "value": settlement_success_rate,
            "target_min": min_settlement_success_rate,
            "target_max": None,
            "status": _metric_status(
                value=settlement_success_rate, min_allowed=min_settlement_success_rate
            ),
        },
        "settlement_unresolved_due": {
            "label": "Unresolved due settlements (last run)",
            "value": settlement_unresolved_due,
            "target_min": None,
            "target_max": None,
            "status": _metric_status(value=settlement_unresolved_due),
        },
        "calibration_updates_realized": {
            "label": "Calibration updates (correct + incorrect)",
            "value": calibration_updates_realized,
            "target_min": min_calibration_updates,
            "target_max": None,
            "status": _metric_status(
                value=calibration_updates_realized, min_allowed=min_calibration_updates
            ),
        },
        "oracle_stall_rate": {
            "label": "Oracle stream stall rate",
            "value": oracle_stall_rate,
            "target_min": None,
            "target_max": max_oracle_stall_rate,
            "status": _metric_status(value=oracle_stall_rate, max_allowed=max_oracle_stall_rate),
        },
    }

    blocking_failures = [
        key
        for key, row in metric_rows.items()
        if row["status"] == "fail"
        or (row["status"] == "unknown" and key in {"settlement_success_rate", "oracle_stall_rate"})
    ]

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": source,
        "dashboard_timestamp": payload.get("timestamp"),
        "collection_time_ms": payload.get("collection_time_ms"),
        "kpis": metric_rows,
        "raw": {
            "settlement_review": {
                "available": bool(settlement.get("available", False)),
                "running": settlement.get("running"),
                "last_run": settlement_stats.get("last_run"),
                "total_runs": settlement_total_runs,
                "last_result": settlement_last,
                "calibration_outcomes": calibration,
            },
            "oracle_stream": {
                "available": bool(oracle.get("available", False)),
                "sessions_started": oracle_sessions_started,
                "sessions_completed": _as_int(oracle.get("sessions_completed")),
                "stalls_total": oracle_stalls_total,
                "ttft_avg_ms": _as_float(oracle.get("ttft_avg_ms")),
                "ttft_samples": _as_int(oracle.get("ttft_samples")),
            },
            "alerts": payload.get("alerts"),
        },
        "blocking_failures": blocking_failures,
        "passed": len(blocking_failures) == 0,
    }


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly Epistemic Runtime KPI Snapshot")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Source: `{summary['source']}`")
    lines.append(f"- Dashboard timestamp: {_fmt_value(summary.get('dashboard_timestamp'))}")
    lines.append(f"- Collection time (ms): {_fmt_value(summary.get('collection_time_ms'))}")
    lines.append(f"- Overall result: {'PASS' if summary.get('passed') else 'FAIL'}")
    lines.append("")
    lines.append("| KPI | Value | Target | Status |")
    lines.append("|---|---:|---:|---|")
    for row in summary["kpis"].values():
        target = "n/a"
        if row.get("target_min") is not None and row.get("target_max") is not None:
            target = f"[{row['target_min']}, {row['target_max']}]"
        elif row.get("target_min") is not None:
            target = f">= {row['target_min']}"
        elif row.get("target_max") is not None:
            target = f"<= {row['target_max']}"
        lines.append(
            f"| {row['label']} | {_fmt_value(row.get('value'))} | {target} | {row['status']} |"
        )
    lines.append("")
    lines.append("## Blocking Failures")
    lines.append("")
    if summary.get("blocking_failures"):
        for item in summary["blocking_failures"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract weekly runtime KPIs from observability dashboard."
    )
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--token", default=None, help="Bearer token for dashboard endpoint.")
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on KPI failures.")
    parser.add_argument("--min-settlement-success-rate", type=float, default=0.99)
    parser.add_argument("--max-oracle-stall-rate", type=float, default=0.02)
    parser.add_argument("--min-calibration-updates", type=int, default=1)
    args = parser.parse_args()

    if args.input_json is not None:
        payload = _load_json(args.input_json)
        source = f"file:{args.input_json}"
    else:
        payload = _fetch_dashboard_json(
            args.dashboard_url,
            token=(args.token or None),
            timeout_seconds=max(1, args.timeout_seconds),
        )
        source = args.dashboard_url

    summary = _extract_summary(
        payload,
        source=source,
        min_settlement_success_rate=args.min_settlement_success_rate,
        max_oracle_stall_rate=args.max_oracle_stall_rate,
        min_calibration_updates=max(0, args.min_calibration_updates),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_to_markdown(summary), encoding="utf-8")

    print(
        json.dumps({"passed": summary["passed"], "blocking_failures": summary["blocking_failures"]})
    )

    if args.strict and not summary["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
