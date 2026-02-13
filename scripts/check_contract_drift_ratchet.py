#!/usr/bin/env python3
"""Enforce week-over-week contract drift ratchet targets."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _count_current_total(
    verify_baseline: Path,
    routes_baseline: Path,
    parity_baseline: Path,
) -> dict[str, int]:
    verify = _load_json(verify_baseline)
    routes = _load_json(routes_baseline)
    parity = _load_json(parity_baseline)

    counts = {
        "verify_python_sdk_drift": len(verify.get("python_sdk_drift", [])),
        "verify_typescript_sdk_drift": len(verify.get("typescript_sdk_drift", [])),
        "routes_missing_in_spec": len(routes.get("missing_in_spec", [])),
        "routes_orphaned_in_spec": len(routes.get("orphaned_in_spec", [])),
        "sdk_missing_from_both": len(parity.get("missing_from_both_sdks", [])),
    }
    counts["total_items"] = sum(counts.values())
    return counts


def _target_after_weeks(start_total: int, weekly_reduction: float, weeks: int) -> int:
    current = start_total
    for _ in range(max(0, weeks)):
        current = max(0, int(round(current * (1.0 - weekly_reduction))))
    return current


def build_ratchet_result(
    *,
    program_baseline: Path,
    verify_baseline: Path,
    routes_baseline: Path,
    parity_baseline: Path,
    as_of: date,
) -> dict[str, Any]:
    program = _load_json(program_baseline)
    if not program:
        raise ValueError(
            f"Program baseline missing or empty: {program_baseline}. "
            "Create scripts/baselines/contract_drift_program.json first."
        )

    start_date_raw = program.get("start_date")
    start_total = int(program.get("start_total_items", 0))
    weekly_reduction = float(program.get("weekly_reduction", 0.1))
    grace_weeks = int(program.get("grace_weeks", 0))

    if not start_date_raw:
        raise ValueError("Program baseline must include 'start_date'")
    if start_total < 0:
        raise ValueError("Program baseline has invalid 'start_total_items'")
    if not (0.0 < weekly_reduction < 1.0):
        raise ValueError("Program baseline 'weekly_reduction' must be between 0 and 1")

    start_date = date.fromisoformat(start_date_raw)
    days_elapsed = max(0, (as_of - start_date).days)
    weeks_elapsed = days_elapsed // 7
    effective_weeks = max(0, weeks_elapsed - grace_weeks)

    counts = _count_current_total(verify_baseline, routes_baseline, parity_baseline)
    target_max = _target_after_weeks(start_total, weekly_reduction, effective_weeks)
    current_total = counts["total_items"]

    result = {
        "program": {
            "start_date": start_date.isoformat(),
            "as_of": as_of.isoformat(),
            "days_elapsed": days_elapsed,
            "weeks_elapsed": weeks_elapsed,
            "effective_weeks": effective_weeks,
            "grace_weeks": grace_weeks,
            "weekly_reduction": weekly_reduction,
            "start_total_items": start_total,
        },
        "current": counts,
        "target": {
            "max_open_items": target_max,
        },
        "delta_to_target": current_total - target_max,
        "passing": current_total <= target_max,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check contract drift weekly ratchet target")
    parser.add_argument(
        "--program-baseline",
        type=Path,
        default=Path("scripts/baselines/contract_drift_program.json"),
        help="Program baseline config path",
    )
    parser.add_argument(
        "--verify-baseline",
        type=Path,
        default=Path("scripts/baselines/verify_sdk_contracts.json"),
        help="verify_sdk_contracts baseline path",
    )
    parser.add_argument(
        "--routes-baseline",
        type=Path,
        default=Path("scripts/baselines/validate_openapi_routes.json"),
        help="validate_openapi_routes baseline path",
    )
    parser.add_argument(
        "--parity-baseline",
        type=Path,
        default=Path("scripts/baselines/check_sdk_parity.json"),
        help="check_sdk_parity baseline path",
    )
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Date for ratchet evaluation (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 when above ratchet target")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    result = build_ratchet_result(
        program_baseline=args.program_baseline,
        verify_baseline=args.verify_baseline,
        routes_baseline=args.routes_baseline,
        parity_baseline=args.parity_baseline,
        as_of=as_of,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        program = result["program"]
        current = result["current"]
        target = result["target"]
        print("Contract Drift Ratchet")
        print("=" * 60)
        print(
            f"As of: {program['as_of']}  |  Start: {program['start_date']}  |  "
            f"Weeks elapsed: {program['weeks_elapsed']} (effective: {program['effective_weeks']})"
        )
        print(
            f"Start total: {program['start_total_items']}  |  "
            f"Current total: {current['total_items']}  |  "
            f"Target max: {target['max_open_items']}"
        )
        print("-" * 60)
        print(
            "Source counts: "
            f"py={current['verify_python_sdk_drift']} "
            f"ts={current['verify_typescript_sdk_drift']} "
            f"missing={current['routes_missing_in_spec']} "
            f"orphaned={current['routes_orphaned_in_spec']} "
            f"both={current['sdk_missing_from_both']}"
        )
        print(f"Delta to target: {result['delta_to_target']:+d}")
        print("PASS" if result["passing"] else "FAIL")

    if args.strict and not result["passing"]:
        print("\nFAIL: Contract drift is above ratchet target.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
