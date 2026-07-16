#!/usr/bin/env python3
"""Generate contract drift observability summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import check_sdk_parity
import check_sdk_namespace_parity
from contract_drift_inventory import InventoryError, build_live_inventory
import contract_drift_inventory as drift

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BaselineJsonError(RuntimeError):
    """Raised when a required contract-drift baseline cannot be trusted."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BaselineJsonError(f"contract drift baseline missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineJsonError(f"cannot load contract drift baseline {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineJsonError(f"contract drift baseline must be a JSON object: {path}")
    return payload


def _inventory_counts(inventory: dict[str, Any] | None = None) -> dict[str, int]:
    measured = inventory if inventory is not None else build_live_inventory(PROJECT_ROOT)
    raw = measured["summary"]["raw_category_counts"]
    return {key: int(value) for key, value in raw.items()}


def _verify_counts(inventory: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/verify_sdk_contracts.json"
    baseline = _load_json(baseline_path)
    counts = _inventory_counts(inventory)
    current = {
        "python_sdk_drift": counts["python_sdk_drift"],
        "typescript_sdk_drift": counts["typescript_sdk_drift"],
        "missing_stable": 0,
    }
    base = {
        "python_sdk_drift": len(baseline.get("python_sdk_drift", [])),
        "typescript_sdk_drift": len(baseline.get("typescript_sdk_drift", [])),
        "missing_stable": len(baseline.get("missing_stable", [])),
    }
    return base, current


def _route_counts(inventory: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/validate_openapi_routes.json"
    baseline = _load_json(baseline_path)
    counts = _inventory_counts(inventory)
    base = {
        "missing_in_spec": len(baseline.get("missing_in_spec", [])),
        "orphaned_in_spec": len(baseline.get("orphaned_in_spec", [])),
    }
    current = {
        "missing_in_spec": counts["routes_missing_in_spec"],
        "orphaned_in_spec": counts["routes_orphaned_in_spec"],
    }
    return base, current


def _parity_counts(inventory: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/check_sdk_parity.json"
    baseline = _load_json(baseline_path)
    counts = _inventory_counts(inventory)
    base = {
        "missing_from_both_sdks": len(baseline.get("missing_from_both_sdks", [])),
    }
    current = {"missing_from_both_sdks": counts["sdk_missing_from_both"]}
    return base, current


def _namespace_counts() -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/check_sdk_namespace_parity.json"
    baseline = _load_json(baseline_path).get("namespaces", {})
    sdk_root = PROJECT_ROOT / "sdk"
    py = drift._scan(
        sdk_root / "python/aragora_sdk/namespaces", drift.extract_python_endpoints, "py"
    )
    ts, _ = drift.scan_typescript_sdk_by_namespace(sdk_root / "typescript/src/namespaces")
    report = check_sdk_parity.build_parity_report(
        check_sdk_parity.extract_handler_routes(),
        {name: {path for _, path in endpoints} for name, endpoints in py.items()},
        {name: {path for _, path in endpoints} for name, endpoints in ts.items()},
        check_sdk_parity.extract_openapi_routes(),
    )
    current = check_sdk_namespace_parity.build_namespace_counts(report)
    # limit to tracked namespaces
    current_tracked = {k: current.get(k, 0) for k in baseline.keys()}
    return baseline, current_tracked


def _delta(base: int, current: int) -> int:
    return current - base


def build_summary() -> dict[str, Any]:
    inventory = build_live_inventory(PROJECT_ROOT)
    verify_base, verify_current = _verify_counts(inventory)
    route_base, route_current = _route_counts(inventory)
    parity_base, parity_current = _parity_counts(inventory)
    ns_base, ns_current = _namespace_counts()

    sections = {
        "verify_sdk_contracts": {
            key: {
                "baseline": verify_base[key],
                "current": verify_current[key],
                "delta": _delta(verify_base[key], verify_current[key]),
            }
            for key in verify_base
        },
        "validate_openapi_routes": {
            key: {
                "baseline": route_base[key],
                "current": route_current[key],
                "delta": _delta(route_base[key], route_current[key]),
            }
            for key in route_base
        },
        "check_sdk_parity": {
            key: {
                "baseline": parity_base[key],
                "current": parity_current[key],
                "delta": _delta(parity_base[key], parity_current[key]),
            }
            for key in parity_base
        },
        "sdk_namespace_parity": {
            key: {
                "baseline": ns_base.get(key, 0),
                "current": ns_current.get(key, 0),
                "delta": _delta(ns_base.get(key, 0), ns_current.get(key, 0)),
            }
            for key in sorted(ns_base.keys())
        },
    }
    return sections


def to_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Contract Drift Summary", ""]
    for section, values in summary.items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Delta |")
        lines.append("|---|---:|---:|---:|")
        for key, row in values.items():
            lines.append(f"| `{key}` | {row['baseline']} | {row['current']} | {row['delta']:+d} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate contract drift summary report")
    parser.add_argument(
        "--json-out",
        default="artifacts/contract-drift-summary.json",
        help="Path to JSON output file",
    )
    parser.add_argument(
        "--md-out",
        default="artifacts/contract-drift-summary.md",
        help="Path to Markdown output file",
    )
    parser.add_argument(
        "--print-md",
        action="store_true",
        help="Print markdown summary to stdout",
    )
    args = parser.parse_args()

    try:
        summary = build_summary()
    except (BaselineJsonError, InventoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    md = to_markdown(summary)

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    md_path.write_text(md)

    if args.print_md:
        print(md)

    if "GITHUB_STEP_SUMMARY" in os.environ:
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
