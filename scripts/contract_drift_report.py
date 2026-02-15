#!/usr/bin/env python3
"""Generate contract drift observability summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import check_sdk_parity
import check_sdk_namespace_parity
import validate_openapi_routes
import verify_sdk_contracts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _verify_counts() -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/verify_sdk_contracts.json"
    baseline = _load_json(baseline_path)

    spec_paths = [PROJECT_ROOT / "docs/api/openapi.json"]
    generated = PROJECT_ROOT / "docs/api/openapi_generated.json"
    if generated.exists():
        spec_paths.append(generated)
    openapi_eps = verify_sdk_contracts._load_openapi_endpoints_multi(spec_paths)

    py_drift: set[str] = set()
    for p in sorted((PROJECT_ROOT / "sdk/python/aragora_sdk/namespaces").glob("*.py")):
        if p.stem.startswith("_"):
            continue
        for method, path in sorted(verify_sdk_contracts._extract_py(p.read_text()) - openapi_eps):
            py_drift.add(f"{method.upper()} {path}")

    ts_drift: set[str] = set()
    for p in sorted((PROJECT_ROOT / "sdk/typescript/src/namespaces").glob("*.ts")):
        if p.stem.startswith("_"):
            continue
        for method, path in sorted(verify_sdk_contracts._extract_ts(p.read_text()) - openapi_eps):
            ts_drift.add(f"{method.upper()} {path}")

    current = {
        "python_sdk_drift": len(py_drift),
        "typescript_sdk_drift": len(ts_drift),
        "missing_stable": 0,
    }
    base = {
        "python_sdk_drift": len(baseline.get("python_sdk_drift", [])),
        "typescript_sdk_drift": len(baseline.get("typescript_sdk_drift", [])),
        "missing_stable": len(baseline.get("missing_stable", [])),
    }
    return base, current


def _route_counts() -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/validate_openapi_routes.json"
    baseline = _load_json(baseline_path)
    result = validate_openapi_routes.validate_coverage(
        "docs/api/openapi.json",
        fail_on_missing=False,
        output_json=False,
        baseline_path=str(baseline_path),
        include_internal=False,
        internal_prefixes_path="scripts/baselines/internal_route_prefixes.json",
    )
    base = {
        "missing_in_spec": len(baseline.get("missing_in_spec", [])),
        "orphaned_in_spec": len(baseline.get("orphaned_in_spec", [])),
    }
    current = {
        "missing_in_spec": result["missing_in_spec_count"],
        "orphaned_in_spec": result["orphaned_in_spec_count"],
    }
    return base, current


def _parity_counts() -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/check_sdk_parity.json"
    baseline = _load_json(baseline_path)
    report = check_sdk_parity.build_parity_report(
        check_sdk_parity.extract_handler_routes(),
        check_sdk_parity.extract_sdk_paths_python(),
        check_sdk_parity.extract_sdk_paths_typescript(),
        check_sdk_parity.extract_openapi_routes(),
    )
    base = {
        "missing_from_both_sdks": len(baseline.get("missing_from_both_sdks", [])),
    }
    current = {"missing_from_both_sdks": report["summary"]["routes_missing_from_both_sdks"]}
    return base, current


def _namespace_counts() -> tuple[dict[str, int], dict[str, int]]:
    baseline_path = PROJECT_ROOT / "scripts/baselines/check_sdk_namespace_parity.json"
    baseline = _load_json(baseline_path).get("namespaces", {})
    report = check_sdk_parity.build_parity_report(
        check_sdk_parity.extract_handler_routes(),
        check_sdk_parity.extract_sdk_paths_python(),
        check_sdk_parity.extract_sdk_paths_typescript(),
        check_sdk_parity.extract_openapi_routes(),
    )
    current = check_sdk_namespace_parity.build_namespace_counts(report)
    # limit to tracked namespaces
    current_tracked = {k: current.get(k, 0) for k in baseline.keys()}
    return baseline, current_tracked


def _delta(base: int, current: int) -> int:
    return current - base


def build_summary() -> dict[str, Any]:
    verify_base, verify_current = _verify_counts()
    route_base, route_current = _route_counts()
    parity_base, parity_current = _parity_counts()
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

    summary = build_summary()
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
