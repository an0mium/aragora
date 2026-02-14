#!/usr/bin/env python3
"""SDK Parity Checker.

Compares handler endpoints against SDK namespace methods to detect
coverage drift. Intended for CI integration to fail when new handler
routes lack corresponding SDK bindings.

Usage:
    python scripts/check_sdk_parity.py             # Report only
    python scripts/check_sdk_parity.py --strict    # Exit 1 if gaps found
    python scripts/check_sdk_parity.py --strict --allow-missing  # transitional override
    python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json
    python scripts/check_sdk_parity.py --json       # JSON output
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Extract handler routes from ROUTES class variables
# ---------------------------------------------------------------------------

# Known internal/system routes that don't need SDK coverage
INTERNAL_ROUTES = {
    "/api/v1/webhooks/stripe",
    "/api/health",
    "/api/v1/health",
    "/health",
    "/api/v1/system",
    "/api/v1/system/mode",
    "/api/v1/docs",
    "/api/v1/docs/openapi.json",
    "/api/v1/status",
}


def extract_handler_routes() -> dict[str, list[str]]:
    """Extract ROUTES from all handler classes.

    Returns:
        Dict mapping handler class name -> list of route paths
    """
    from aragora.server.handlers._lazy_imports import ALL_HANDLER_NAMES, HANDLER_MODULES

    handler_routes: dict[str, list[str]] = {}

    for name in ALL_HANDLER_NAMES:
        module_path = HANDLER_MODULES.get(name)
        if not module_path:
            continue

        try:
            module = importlib.import_module(module_path)
            handler_cls = getattr(module, name, None)
            if handler_cls is None:
                continue

            routes = getattr(handler_cls, "ROUTES", None)
            if routes and isinstance(routes, (list, tuple)):
                handler_routes[name] = list(routes)
        except (ImportError, AttributeError, ModuleNotFoundError):
            continue

    return handler_routes


def extract_openapi_routes(spec_path: Path | None = None) -> set[str]:
    """Extract normalized route paths documented in OpenAPI."""
    if spec_path is None:
        spec_path = PROJECT_ROOT / "docs" / "api" / "openapi.json"
    if not spec_path.exists():
        return set()

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    documented: set[str] = set()
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return documented

    for path, methods in paths.items():
        if not isinstance(path, str) or not isinstance(methods, dict):
            continue
        # Keep path only if at least one HTTP operation is present.
        has_http_op = any(
            isinstance(method, str)
            and method.lower() in {"get", "post", "put", "patch", "delete", "options", "head"}
            for method in methods
        )
        if has_http_op:
            documented.add(normalize_route(path))

    return documented


def normalize_route(route: str) -> str:
    """Normalize a route for comparison.

    Strips version prefix, converts all path parameters to {param} form,
    and lowercases.
    """
    # Strip version prefix
    route = re.sub(r"^/api/v\d+/", "/api/", route)
    # Convert * wildcards to {param}
    route = route.replace("/*", "/{param}")
    # Convert Express-style :param segments to {param}
    route = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", "{param}", route)
    # Normalize all named path parameters ({session_id}, {finding_id}, etc.) to {param}
    route = re.sub(r"\{[^}]+\}", "{param}", route)
    return route.lower().rstrip("/")


# ---------------------------------------------------------------------------
# 2. Extract SDK endpoint paths from Python namespace files
# ---------------------------------------------------------------------------


def extract_sdk_paths_python() -> dict[str, set[str]]:
    """Extract HTTP paths from Python SDK namespace files.

    Parses self._client.request("METHOD", "/api/v1/...") calls.

    Returns:
        Dict mapping namespace name -> set of endpoint paths
    """
    sdk_dir = PROJECT_ROOT / "sdk" / "python" / "aragora_sdk" / "namespaces"
    if not sdk_dir.exists():
        return {}

    path_pattern = re.compile(r'self\._client\.request\(\s*"[A-Z]+"\s*,\s*[f"]([^"]+)"')
    fstring_pattern = re.compile(r'self\._client\.request\(\s*"[A-Z]+"\s*,\s*f"([^"]+)"')

    namespace_paths: dict[str, set[str]] = {}

    for py_file in sorted(sdk_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        ns_name = py_file.stem
        paths: set[str] = set()

        try:
            content = py_file.read_text(encoding="utf-8")
        except OSError:
            continue

        # Match both regular strings and f-strings
        for match in path_pattern.finditer(content):
            raw_path = match.group(1)
            # Replace f-string expressions with {param}
            cleaned = re.sub(r"\{[^}]+\}", "{param}", raw_path)
            paths.add(cleaned)

        for match in fstring_pattern.finditer(content):
            raw_path = match.group(1)
            cleaned = re.sub(r"\{[^}]+\}", "{param}", raw_path)
            paths.add(cleaned)

        if paths:
            namespace_paths[ns_name] = paths

    return namespace_paths


def extract_sdk_paths_typescript() -> dict[str, set[str]]:
    """Extract HTTP paths from TypeScript SDK namespace files.

    Returns:
        Dict mapping namespace name -> set of endpoint paths
    """
    sdk_dir = PROJECT_ROOT / "sdk" / "typescript" / "src" / "namespaces"
    if not sdk_dir.exists():
        return {}

    # Match patterns like: request("GET", "/api/v1/...") or request('GET', '/api/v1/...')
    # Also match template literals: `...`
    path_pattern = re.compile(r'request(?:<[^>]*>)?\(\s*["\'][A-Z]+["\']\s*,\s*[`"\']([^`"\']+)[`"\']')

    namespace_paths: dict[str, set[str]] = {}

    for ts_file in sorted(sdk_dir.glob("*.ts")):
        if ts_file.name.startswith("_") or ts_file.name == "index.ts":
            continue

        ns_name = ts_file.stem
        paths: set[str] = set()

        try:
            content = ts_file.read_text(encoding="utf-8")
        except OSError:
            continue

        for match in path_pattern.finditer(content):
            raw_path = match.group(1)
            # Replace template expressions ${...} with {param}
            cleaned = re.sub(r"\$\{[^}]+\}", "{param}", raw_path)
            paths.add(cleaned)

        if paths:
            namespace_paths[ns_name] = paths

    return namespace_paths


# ---------------------------------------------------------------------------
# 3. Compare and report
# ---------------------------------------------------------------------------


def build_parity_report(
    handler_routes: dict[str, list[str]],
    python_sdk: dict[str, set[str]],
    typescript_sdk: dict[str, set[str]],
    documented_routes: set[str] | None = None,
) -> dict[str, Any]:
    """Build a parity report comparing handlers vs SDKs.

    Returns structured report with coverage stats and gaps.
    """
    # Flatten all SDK paths (normalized)
    all_py_paths = set()
    for paths in python_sdk.values():
        for p in paths:
            all_py_paths.add(normalize_route(p))

    all_ts_paths = set()
    for paths in typescript_sdk.values():
        for p in paths:
            all_ts_paths.add(normalize_route(p))

    # Flatten all handler routes (normalized)
    all_handler_paths = set()
    handler_to_routes: dict[str, list[str]] = {}
    for handler_name, routes in handler_routes.items():
        normalized = [normalize_route(r) for r in routes]
        handler_to_routes[handler_name] = normalized
        all_handler_paths.update(normalized)

    # Filter out internal routes
    internal_normalized = {normalize_route(r) for r in INTERNAL_ROUTES}
    public_handler_paths = all_handler_paths - internal_normalized
    if documented_routes is not None:
        # SDK coverage should be enforced for documented API routes.
        public_handler_paths = public_handler_paths & documented_routes

    # Find gaps (wildcard-aware: handler routes ending with /{param}
    # are considered covered if any SDK path shares the prefix)
    def _is_covered(route: str, sdk_paths: set[str]) -> bool:
        if route in sdk_paths:
            return True
        if route.endswith("/{param}"):
            prefix = route[: -len("/{param}")]
            return any(sp.startswith(prefix + "/") for sp in sdk_paths)
        return False

    missing_from_py_sdk = {r for r in public_handler_paths if not _is_covered(r, all_py_paths)}
    missing_from_ts_sdk = {r for r in public_handler_paths if not _is_covered(r, all_ts_paths)}
    missing_from_both = missing_from_py_sdk & missing_from_ts_sdk

    # SDK paths not in handlers (potential stale SDK methods)
    # Build wildcard prefixes from handler routes ending with {param}
    # to match SDK sub-paths like /api/integrations/stats against /api/integrations/{param}
    wildcard_prefixes = set()
    for p in all_handler_paths:
        if p.endswith("/{param}"):
            wildcard_prefixes.add(p[: -len("/{param}")])

    def _covered_by_handler(sdk_path: str) -> bool:
        if sdk_path in all_handler_paths:
            return True
        # Check if any wildcard handler covers this path
        for prefix in wildcard_prefixes:
            if sdk_path.startswith(prefix + "/"):
                return True
        return False

    stale_py = {p for p in all_py_paths if not _covered_by_handler(p)}
    stale_ts = {p for p in all_ts_paths if not _covered_by_handler(p)}

    # Helper: check if an SDK path set covers a handler route.
    # For wildcard routes ending with /{param}, any SDK path starting with
    # the prefix counts as coverage (mirrors _covered_by_handler logic).
    def _sdk_covers_route(route: str, sdk_paths: set[str]) -> bool:
        if route in sdk_paths:
            return True
        if route.endswith("/{param}"):
            prefix = route[: -len("/{param}")]
            return any(sp.startswith(prefix + "/") for sp in sdk_paths)
        return False

    # Per-handler coverage
    handler_coverage: list[dict[str, Any]] = []
    for handler_name, normalized_routes in sorted(handler_to_routes.items()):
        public_routes = [r for r in normalized_routes if r not in internal_normalized]
        if not public_routes:
            continue

        py_covered = sum(1 for r in public_routes if _sdk_covers_route(r, all_py_paths))
        ts_covered = sum(1 for r in public_routes if _sdk_covers_route(r, all_ts_paths))

        handler_coverage.append(
            {
                "handler": handler_name,
                "total_routes": len(public_routes),
                "python_sdk_covered": py_covered,
                "typescript_sdk_covered": ts_covered,
                "missing_python": [r for r in public_routes if not _sdk_covers_route(r, all_py_paths)],
                "missing_typescript": [r for r in public_routes if not _sdk_covers_route(r, all_ts_paths)],
            }
        )

    # Summary stats
    total_public = len(public_handler_paths)
    py_coverage = (
        (total_public - len(missing_from_py_sdk)) / total_public * 100 if total_public else 0
    )
    ts_coverage = (
        (total_public - len(missing_from_ts_sdk)) / total_public * 100 if total_public else 0
    )

    return {
        "summary": {
            "total_handlers": len(handler_routes),
            "total_public_routes": total_public,
            "python_sdk_namespaces": len(python_sdk),
            "typescript_sdk_namespaces": len(typescript_sdk),
            "python_sdk_paths": len(all_py_paths),
            "typescript_sdk_paths": len(all_ts_paths),
            "python_sdk_coverage_pct": round(py_coverage, 1),
            "typescript_sdk_coverage_pct": round(ts_coverage, 1),
            "routes_missing_from_both_sdks": len(missing_from_both),
        },
        "gaps": {
            "missing_from_python_sdk": sorted(missing_from_py_sdk),
            "missing_from_typescript_sdk": sorted(missing_from_ts_sdk),
            "missing_from_both_sdks": sorted(missing_from_both),
            "stale_python_sdk_paths": sorted(stale_py),
            "stale_typescript_sdk_paths": sorted(stale_ts),
        },
        "handler_coverage": handler_coverage,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print human-readable parity report."""
    s = report["summary"]

    print("=" * 70)
    print("SDK Parity Report")
    print("=" * 70)
    print(f"Handlers scanned:           {s['total_handlers']}")
    print(f"Public routes found:        {s['total_public_routes']}")
    print(f"Python SDK namespaces:      {s['python_sdk_namespaces']}")
    print(f"TypeScript SDK namespaces:  {s['typescript_sdk_namespaces']}")
    print(f"Python SDK coverage:        {s['python_sdk_coverage_pct']}%")
    print(f"TypeScript SDK coverage:    {s['typescript_sdk_coverage_pct']}%")
    print(f"Missing from BOTH SDKs:     {s['routes_missing_from_both_sdks']}")
    print()

    gaps = report["gaps"]

    if gaps["missing_from_both_sdks"]:
        print("-" * 70)
        print(f"Routes missing from BOTH SDKs ({len(gaps['missing_from_both_sdks'])}):")
        for route in gaps["missing_from_both_sdks"][:30]:
            print(f"  {route}")
        if len(gaps["missing_from_both_sdks"]) > 30:
            print(f"  ... and {len(gaps['missing_from_both_sdks']) - 30} more")
        print()

    # Per-handler gaps (only show handlers with missing coverage)
    uncovered = [
        h for h in report["handler_coverage"] if h["missing_python"] or h["missing_typescript"]
    ]
    if uncovered:
        print("-" * 70)
        print(f"Handlers with SDK gaps ({len(uncovered)}):")
        for h in uncovered[:20]:
            py_gap = len(h["missing_python"])
            ts_gap = len(h["missing_typescript"])
            print(f"  {h['handler']}: {h['total_routes']} routes (py: -{py_gap}, ts: -{ts_gap})")
        if len(uncovered) > 20:
            print(f"  ... and {len(uncovered) - 20} more handlers")
        print()

    if gaps["stale_python_sdk_paths"]:
        print("-" * 70)
        print(f"Stale Python SDK paths ({len(gaps['stale_python_sdk_paths'])}):")
        for path in gaps["stale_python_sdk_paths"][:10]:
            print(f"  {path}")
        if len(gaps["stale_python_sdk_paths"]) > 10:
            print(f"  ... and {len(gaps['stale_python_sdk_paths']) - 10} more")
        print()

    print("=" * 70)
    if s["routes_missing_from_both_sdks"] == 0:
        print("PASS: All public routes have SDK coverage in at least one SDK.")
    else:
        print(f"WARN: {s['routes_missing_from_both_sdks']} routes lack SDK coverage.")


def _expected_budget_max(
    *,
    initial: int,
    weekly_reduction: int,
    start_date: dt.date,
    today: dt.date,
) -> int:
    """Compute expected maximum debt after weekly reduction cadence."""
    if weekly_reduction <= 0 or today <= start_date:
        return initial
    weeks_elapsed = (today - start_date).days // 7
    return max(0, initial - (weeks_elapsed * weekly_reduction))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK parity with handler endpoints")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any gaps found")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="When used with --strict, allow routes missing from both SDKs (transitional override)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("scripts/baselines/check_sdk_parity.json"),
        help="Path to parity drift baseline file (default: scripts/baselines/check_sdk_parity.json)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument(
        "--include-undocumented",
        action="store_true",
        help="Include handler routes not present in docs/api/openapi.json (default: documented routes only)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum coverage %% for --strict mode (default: 0)",
    )
    parser.add_argument(
        "--budget",
        type=Path,
        default=Path("scripts/baselines/check_sdk_parity_budget.json"),
        help="Optional budget file for progressive parity debt reduction",
    )
    parser.add_argument(
        "--today",
        type=str,
        default=None,
        help="Override current date (YYYY-MM-DD) for deterministic budget checks",
    )
    args = parser.parse_args()

    # Extract data
    handler_routes = extract_handler_routes()
    python_sdk = extract_sdk_paths_python()
    typescript_sdk = extract_sdk_paths_typescript()
    documented_routes = None if args.include_undocumented else extract_openapi_routes()

    # Build report
    report = build_parity_report(handler_routes, python_sdk, typescript_sdk, documented_routes)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    baseline_missing: set[str] = set()
    if args.baseline and args.baseline.exists():
        baseline_data = json.loads(args.baseline.read_text())
        baseline_missing = set(baseline_data.get("missing_from_both_sdks", []))
    missing_set = set(report["gaps"]["missing_from_both_sdks"])
    new_missing = missing_set - baseline_missing
    if args.baseline and not args.json:
        print(f"\nBaseline regressions: missing_from_both={len(new_missing)}")
        for route in sorted(new_missing)[:20]:
            print(f"  NEW: {route}")
        if len(new_missing) > 20:
            print(f"  ... and {len(new_missing) - 20} more")

    budget_status: dict[str, int] | None = None
    if args.budget and args.budget.exists():
        try:
            budget_data = json.loads(args.budget.read_text(encoding="utf-8"))
            start_date_str = str(budget_data.get("start_date", "")).strip()
            if not start_date_str:
                raise ValueError("budget.start_date is required")
            start_date = dt.date.fromisoformat(start_date_str)
            today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

            initial_missing = int(
                budget_data.get(
                    "initial_missing_from_both_sdks",
                    report["summary"]["routes_missing_from_both_sdks"],
                )
            )
            weekly_missing = int(budget_data.get("weekly_reduction_missing_from_both_sdks", 0))
            expected_missing = _expected_budget_max(
                initial=initial_missing,
                weekly_reduction=weekly_missing,
                start_date=start_date,
                today=today,
            )

            stale_current = len(report["gaps"]["stale_python_sdk_paths"])
            initial_stale = int(
                budget_data.get("initial_stale_python_sdk_paths", stale_current)
            )
            weekly_stale = int(budget_data.get("weekly_reduction_stale_python_sdk_paths", 0))
            expected_stale = _expected_budget_max(
                initial=initial_stale,
                weekly_reduction=weekly_stale,
                start_date=start_date,
                today=today,
            )

            budget_status = {
                "expected_missing_max": expected_missing,
                "current_missing": report["summary"]["routes_missing_from_both_sdks"],
                "expected_stale_python_max": expected_stale,
                "current_stale_python": stale_current,
            }
            if not args.json:
                print(
                    "\nBudget status: "
                    f"missing_from_both {budget_status['current_missing']}/{budget_status['expected_missing_max']} "
                    f"| stale_python {budget_status['current_stale_python']}/{budget_status['expected_stale_python_max']}"
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"\nFAIL: Invalid SDK parity budget file ({args.budget}): {exc}")
            return 2

    # Strict mode: fail if gaps exceed threshold
    if args.strict:
        py_cov = report["summary"]["python_sdk_coverage_pct"]
        ts_cov = report["summary"]["typescript_sdk_coverage_pct"]
        if py_cov < args.threshold or ts_cov < args.threshold:
            print(f"\nFAIL: Coverage below threshold ({args.threshold}%)")
            return 1
        missing = report["summary"]["routes_missing_from_both_sdks"]
        if len(new_missing) > 0 and not args.allow_missing:
            print(
                f"\nFAIL: {len(new_missing)} new routes lack SDK coverage "
                f"(total missing: {missing})."
            )
            print("Run with --allow-missing only as a temporary migration override.")
            return 1
        if budget_status:
            if budget_status["current_missing"] > budget_status["expected_missing_max"]:
                print(
                    "\nFAIL: Missing-from-both debt exceeds budget "
                    f"({budget_status['current_missing']} > {budget_status['expected_missing_max']})."
                )
                return 1
            if budget_status["current_stale_python"] > budget_status["expected_stale_python_max"]:
                print(
                    "\nFAIL: Stale Python SDK debt exceeds budget "
                    f"({budget_status['current_stale_python']} > "
                    f"{budget_status['expected_stale_python_max']})."
                )
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
