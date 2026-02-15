#!/usr/bin/env python3
"""Namespace-focused SDK parity gate.

Tracks missing-from-both-SDK coverage by API namespace and blocks regressions
for the highest-impact namespaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import check_sdk_parity as sdk_parity


def route_namespace(route: str) -> str:
    parts = [p for p in route.strip("/").split("/") if p]
    if len(parts) >= 2 and parts[0] == "api":
        return parts[1]
    return parts[0] if parts else "root"


def build_namespace_counts(report: dict[str, Any]) -> dict[str, int]:
    routes = report.get("gaps", {}).get("missing_from_both_sdks", [])
    counter: Counter[str] = Counter()
    for route in routes:
        if isinstance(route, str):
            counter[route_namespace(route)] += 1
    return dict(sorted(counter.items()))


def load_baseline(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    raw = data.get("namespaces", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, int):
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Check namespace-focused SDK parity regressions")
    parser.add_argument(
        "--strict", action="store_true", help="Fail when focused namespace counts regress"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("scripts/baselines/check_sdk_namespace_parity.json"),
        help="Path to namespace parity baseline (default: scripts/baselines/check_sdk_namespace_parity.json)",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="Comma-separated namespace list to track (defaults to baseline namespaces)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    report = sdk_parity.build_parity_report(
        sdk_parity.extract_handler_routes(),
        sdk_parity.extract_sdk_paths_python(),
        sdk_parity.extract_sdk_paths_typescript(),
        sdk_parity.extract_openapi_routes(),
    )
    namespace_counts = build_namespace_counts(report)
    baseline = load_baseline(args.baseline)

    if args.focus.strip():
        focus = [x.strip() for x in args.focus.split(",") if x.strip()]
    else:
        focus = sorted(baseline.keys())
    if not focus:
        # safe default: top namespaces by current missing count
        focus = [
            k for k, _ in sorted(namespace_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ]

    regressions: dict[str, dict[str, int]] = {}
    summary_rows: list[dict[str, int | str]] = []
    for ns in focus:
        current = namespace_counts.get(ns, 0)
        expected = baseline.get(ns, 0)
        delta = current - expected
        summary_rows.append(
            {
                "namespace": ns,
                "baseline": expected,
                "current": current,
                "delta": delta,
            }
        )
        if delta > 0:
            regressions[ns] = {"baseline": expected, "current": current, "delta": delta}

    result = {
        "focused_namespaces": focus,
        "regressions": regressions,
        "regression_count": len(regressions),
        "rows": summary_rows,
        "total_missing_from_both": report["summary"]["routes_missing_from_both_sdks"],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Namespace SDK Parity")
        print("=" * 60)
        for row in summary_rows:
            print(
                f"{row['namespace']:20} baseline={row['baseline']:>4} "
                f"current={row['current']:>4} delta={row['delta']:>+4}"
            )
        print("-" * 60)
        print(f"Total missing-from-both routes: {result['total_missing_from_both']}")
        print(f"Regressions: {result['regression_count']}")

    if args.strict and regressions:
        print("\nFAIL: Namespace parity regressions detected.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
