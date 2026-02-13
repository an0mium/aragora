#!/usr/bin/env python3
"""SDK Parity Checker.

Compares handler endpoints against SDK namespace methods to detect
coverage drift. Intended for CI integration to fail when new handler
routes lack corresponding SDK bindings.

Usage:
    python scripts/check_sdk_parity.py             # Report only
    python scripts/check_sdk_parity.py --strict    # Exit 1 if gaps found
    python scripts/check_sdk_parity.py --strict --allow-missing  # transitional override
    python scripts/check_sdk_parity.py --json       # JSON output
"""

from __future__ import annotations

import argparse
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


def normalize_route(route: str) -> str:
    """Normalize a route for comparison.

    Strips version prefix, converts wildcards to {param} form,
    and lowercases.
    """
    # Strip version prefix
    route = re.sub(r"^/api/v\d+/", "/api/", route)
    # Convert * wildcards to {param}
    route = route.replace("/*", "/{param}")
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
    path_pattern = re.compile(r'request\(\s*["\'][A-Z]+["\']\s*,\s*[`"\']([^`"\']+)[`"\']')

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

    # Find gaps
    missing_from_py_sdk = public_handler_paths - all_py_paths
    missing_from_ts_sdk = public_handler_paths - all_ts_paths
    missing_from_both = missing_from_py_sdk & missing_from_ts_sdk

    # SDK paths not in handlers (potential stale SDK methods)
    stale_py = all_py_paths - all_handler_paths
    stale_ts = all_ts_paths - all_handler_paths

    # Per-handler coverage
    handler_coverage: list[dict[str, Any]] = []
    for handler_name, normalized_routes in sorted(handler_to_routes.items()):
        public_routes = [r for r in normalized_routes if r not in internal_normalized]
        if not public_routes:
            continue

        py_covered = sum(1 for r in public_routes if r in all_py_paths)
        ts_covered = sum(1 for r in public_routes if r in all_ts_paths)

        handler_coverage.append(
            {
                "handler": handler_name,
                "total_routes": len(public_routes),
                "python_sdk_covered": py_covered,
                "typescript_sdk_covered": ts_covered,
                "missing_python": [r for r in public_routes if r not in all_py_paths],
                "missing_typescript": [r for r in public_routes if r not in all_ts_paths],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK parity with handler endpoints")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any gaps found")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="When used with --strict, allow routes missing from both SDKs (transitional override)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum coverage %% for --strict mode (default: 0)",
    )
    args = parser.parse_args()

    # Extract data
    handler_routes = extract_handler_routes()
    python_sdk = extract_sdk_paths_python()
    typescript_sdk = extract_sdk_paths_typescript()

    # Build report
    report = build_parity_report(handler_routes, python_sdk, typescript_sdk)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    # Strict mode: fail if gaps exceed threshold
    if args.strict:
        py_cov = report["summary"]["python_sdk_coverage_pct"]
        ts_cov = report["summary"]["typescript_sdk_coverage_pct"]
        if py_cov < args.threshold or ts_cov < args.threshold:
            print(f"\nFAIL: Coverage below threshold ({args.threshold}%)")
            return 1
        missing = report["summary"]["routes_missing_from_both_sdks"]
        if missing > 0 and not args.allow_missing:
            print(f"\nFAIL: {missing} routes lack SDK coverage.")
            print("Run with --allow-missing only as a temporary migration override.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
