#!/usr/bin/env python
"""Validate that OpenAPI spec routes match handler registry.

This script ensures the OpenAPI specification stays in sync with actual
handler implementations by comparing:
1. Routes defined in handler ROUTES attributes
2. Routes in the OpenAPI spec paths

Usage:
    python scripts/validate_openapi_routes.py
    python scripts/validate_openapi_routes.py --spec docs/api/openapi.json
    python scripts/validate_openapi_routes.py --fail-on-missing  # Exit 1 if routes missing
    python scripts/validate_openapi_routes.py --fail-on-missing --baseline scripts/baselines/validate_openapi_routes.json
    python scripts/validate_openapi_routes.py --json  # Output as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def get_handler_routes() -> set[str]:
    """Extract all routes from handler ROUTES attributes.

    Returns:
        Set of route paths defined across all handlers.
    """
    routes: set[str] = set()

    try:
        from aragora.server.handler_registry import HANDLER_REGISTRY
    except ImportError:
        print("Error: Cannot import handler_registry. Ensure aragora is installed.")
        sys.exit(1)

    for attr_name, handler_class in HANDLER_REGISTRY:
        if handler_class is None:
            continue

        # Collect ROUTES attribute
        if hasattr(handler_class, "ROUTES"):
            handler_routes = handler_class.ROUTES
            if isinstance(handler_routes, (list, tuple)):
                routes.update(handler_routes)

        # Collect method-specific routes
        for method_attr in (
            "GET_ROUTES",
            "POST_ROUTES",
            "PUT_ROUTES",
            "PATCH_ROUTES",
            "DELETE_ROUTES",
        ):
            if hasattr(handler_class, method_attr):
                method_routes = getattr(handler_class, method_attr)
                if isinstance(method_routes, (list, tuple)):
                    routes.update(method_routes)

    return routes


def get_openapi_routes(spec_path: str) -> set[str]:
    """Extract all paths from OpenAPI spec.

    Args:
        spec_path: Path to OpenAPI JSON file.

    Returns:
        Set of API paths from the spec.
    """
    path = Path(spec_path)
    if not path.exists():
        print(f"Error: OpenAPI spec not found at {spec_path}")
        sys.exit(1)

    with open(path) as f:
        spec = json.load(f)

    return set(spec.get("paths", {}).keys())


def normalize_route(route: str | tuple) -> str:
    """Normalize a route for comparison.

    Handles:
    - Tuple routes like (method, path) or (method, path, handler)
    - Version prefixes (/api/v1/ vs /api/)
    - Trailing slashes
    - Wildcard patterns (* to {param})

    Args:
        route: Raw route string or tuple containing route path.

    Returns:
        Normalized route for comparison.
    """
    import re

    # Handle tuple routes - extract path (second element for (method, path, ...) format)
    if isinstance(route, tuple):
        route = route[1] if len(route) > 1 else str(route[0])

    # Strip trailing slash
    route = route.rstrip("/")

    # Normalize version prefix
    if route.startswith("/api/") and not route.startswith("/api/v"):
        route = route.replace("/api/", "/api/v1/", 1)

    # Convert wildcard * to generic {param} for comparison
    # This matches both /debates/* and /debates/{id}
    route = re.sub(r"/\*(/|$)", r"/{param}\1", route)

    # Also normalize common OpenAPI param names to generic {param}
    # so /debates/{id} matches /debates/{param}
    route = re.sub(r"/\{[^}]+\}", "/{param}", route)

    return route


def validate_coverage(
    spec_path: str,
    fail_on_missing: bool = False,
    output_json: bool = False,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    """Compare handler routes against OpenAPI spec.

    Args:
        spec_path: Path to OpenAPI spec.
        fail_on_missing: Whether to exit with error if routes missing.
        output_json: Whether to output as JSON.

    Returns:
        Validation results dict.
    """
    handler_routes = get_handler_routes()
    openapi_routes = get_openapi_routes(spec_path)

    # Normalize routes for comparison
    normalized_handler = {normalize_route(r) for r in handler_routes}
    normalized_openapi = {normalize_route(r) for r in openapi_routes}

    # Find discrepancies
    # Routes in handlers but not in OpenAPI (these need to be documented)
    missing_in_spec = normalized_handler - normalized_openapi

    # Routes in OpenAPI but not in handlers (may be deprecated or generated)
    missing_handlers = normalized_openapi - normalized_handler

    # Filter out known patterns that may not have explicit ROUTES
    # (e.g., dynamic routes handled by can_handle())
    known_dynamic_patterns = {
        r
        for r in missing_handlers
        if any(
            p in r
            for p in [
                "{",
                "*",
                "debate_id",
                "agent",
                "workspace_id",
                "org_id",
            ]
        )
    }

    # Routes that are truly orphaned in OpenAPI
    orphaned_in_spec = missing_handlers - known_dynamic_patterns

    baseline_missing: set[str] = set()
    baseline_orphaned: set[str] = set()
    if baseline_path:
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            baseline_data = json.loads(baseline_file.read_text())
            baseline_missing = set(baseline_data.get("missing_in_spec", []))
            baseline_orphaned = set(baseline_data.get("orphaned_in_spec", []))

    new_missing_in_spec = sorted(set(missing_in_spec) - baseline_missing)
    new_orphaned_in_spec = sorted(set(orphaned_in_spec) - baseline_orphaned)

    results = {
        "handler_routes_count": len(handler_routes),
        "openapi_routes_count": len(openapi_routes),
        "missing_in_spec": sorted(missing_in_spec),
        "missing_in_spec_count": len(missing_in_spec),
        "new_missing_in_spec": new_missing_in_spec,
        "new_missing_in_spec_count": len(new_missing_in_spec),
        "orphaned_in_spec": sorted(orphaned_in_spec),
        "orphaned_in_spec_count": len(orphaned_in_spec),
        "new_orphaned_in_spec": new_orphaned_in_spec,
        "new_orphaned_in_spec_count": len(new_orphaned_in_spec),
        "dynamic_routes_skipped": len(known_dynamic_patterns),
        "coverage_percentage": round(
            (1 - len(missing_in_spec) / max(len(handler_routes), 1)) * 100, 1
        ),
    }

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 60)
        print("OpenAPI Route Validation Report")
        print("=" * 60)
        print(f"Handler routes:  {results['handler_routes_count']}")
        print(f"OpenAPI routes:  {results['openapi_routes_count']}")
        print(f"Coverage:        {results['coverage_percentage']}%")
        print()
        if baseline_path:
            print(
                f"New missing in spec vs baseline: {results['new_missing_in_spec_count']}"
            )
            print(
                f"New orphaned in spec vs baseline: {results['new_orphaned_in_spec_count']}"
            )
            print()

        if missing_in_spec:
            print(f"Routes missing from OpenAPI spec ({len(missing_in_spec)}):")
            for route in sorted(missing_in_spec)[:20]:
                print(f"  - {route}")
            if len(missing_in_spec) > 20:
                print(f"  ... and {len(missing_in_spec) - 20} more")
            print()

        if orphaned_in_spec:
            print(f"OpenAPI routes without handlers ({len(orphaned_in_spec)}):")
            for route in sorted(orphaned_in_spec)[:10]:
                print(f"  - {route}")
            if len(orphaned_in_spec) > 10:
                print(f"  ... and {len(orphaned_in_spec) - 10} more")
            print()

        if not missing_in_spec and not orphaned_in_spec:
            print("All routes properly documented!")

    if fail_on_missing:
        if baseline_path:
            if new_missing_in_spec or new_orphaned_in_spec:
                sys.exit(1)
        elif missing_in_spec:
            sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate OpenAPI spec routes match handler registry"
    )
    parser.add_argument(
        "--spec",
        default="docs/api/openapi.json",
        help="Path to OpenAPI spec (default: docs/api/openapi.json)",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with error code 1 if routes are missing from spec",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--baseline",
        default="scripts/baselines/validate_openapi_routes.json",
        help="Path to baseline drift file (default: scripts/baselines/validate_openapi_routes.json)",
    )

    args = parser.parse_args()
    validate_coverage(args.spec, args.fail_on_missing, args.json, args.baseline)


if __name__ == "__main__":
    main()
