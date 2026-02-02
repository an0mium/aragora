#!/usr/bin/env python3
"""
SDK-Handler Parity Audit Script

Analyzes SDK namespace files and handler implementations to identify
endpoints that are defined in SDKs but not implemented server-side.

Usage:
    python scripts/sdk_handler_audit.py           # Full report
    python scripts/sdk_handler_audit.py --verify  # Quick parity check
    python scripts/sdk_handler_audit.py --json    # JSON output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Endpoint:
    """Represents an API endpoint."""

    method: str
    path: str
    source_file: str
    line_number: int
    namespace: str = ""


@dataclass
class ParityReport:
    """Report of SDK-Handler parity analysis."""

    ts_endpoints: list[Endpoint] = field(default_factory=list)
    py_endpoints: list[Endpoint] = field(default_factory=list)
    handler_routes: list[str] = field(default_factory=list)
    missing_handlers: list[Endpoint] = field(default_factory=list)
    coverage_ts: float = 0.0
    coverage_py: float = 0.0


def extract_ts_endpoints(sdk_path: Path) -> list[Endpoint]:
    """Extract API endpoints from TypeScript SDK namespace files."""
    endpoints: list[Endpoint] = []
    namespaces_dir = sdk_path / "src" / "namespaces"

    if not namespaces_dir.exists():
        print(f"Warning: TypeScript namespaces dir not found: {namespaces_dir}")
        return endpoints

    # Patterns to match endpoint calls
    patterns = [
        # this.client.request<T>('METHOD', '/api/...')
        re.compile(
            r"this\.client\.request\s*<[^>]*>\s*\(\s*['\"](\w+)['\"]\s*,\s*['\"`]([^'\"`]+)['\"`]"
        ),
        # this.client.get('/api/...')
        re.compile(r"this\.client\.get\s*\(\s*['\"`]([^'\"`]+)['\"`]"),
        # this.client.post('/api/...')
        re.compile(r"this\.client\.post\s*\(\s*['\"`]([^'\"`]+)['\"`]"),
        # this.client.put('/api/...')
        re.compile(r"this\.client\.put\s*\(\s*['\"`]([^'\"`]+)['\"`]"),
        # this.client.delete('/api/...')
        re.compile(r"this\.client\.delete\s*\(\s*['\"`]([^'\"`]+)['\"`]"),
        # this.client.patch('/api/...')
        re.compile(r"this\.client\.patch\s*\(\s*['\"`]([^'\"`]+)['\"`]"),
    ]

    for ts_file in namespaces_dir.glob("*.ts"):
        if ts_file.name.startswith("__"):
            continue

        namespace = ts_file.stem
        content = ts_file.read_text(errors="ignore")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments and test files
            if line.strip().startswith("//") or ".test." in ts_file.name:
                continue

            for pattern in patterns:
                for match in pattern.finditer(line):
                    groups = match.groups()
                    if len(groups) == 2:
                        method, path = groups
                    else:
                        # Extract method from pattern name (get, post, put, delete, patch)
                        if "get\\s" in pattern.pattern or "get(" in pattern.pattern:
                            method = "GET"
                        elif "post\\s" in pattern.pattern or "post(" in pattern.pattern:
                            method = "POST"
                        elif "put\\s" in pattern.pattern or "put(" in pattern.pattern:
                            method = "PUT"
                        elif "delete\\s" in pattern.pattern or "delete(" in pattern.pattern:
                            method = "DELETE"
                        elif "patch\\s" in pattern.pattern or "patch(" in pattern.pattern:
                            method = "PATCH"
                        else:
                            method = "GET"
                        path = groups[0]

                    # Normalize path (handle template literals)
                    path = re.sub(r"\$\{[^}]+\}", "*", path)
                    path = re.sub(r"`", "", path)

                    if path.startswith("/api"):
                        endpoints.append(
                            Endpoint(
                                method=method.upper() if method else "GET",
                                path=path,
                                source_file=str(ts_file.relative_to(sdk_path)),
                                line_number=line_num,
                                namespace=namespace,
                            )
                        )

    return endpoints


def extract_py_endpoints(sdk_path: Path) -> list[Endpoint]:
    """Extract API endpoints from Python SDK namespace files."""
    endpoints: list[Endpoint] = []

    # Check multiple possible locations
    possible_dirs = [
        sdk_path / "aragora_sdk" / "namespaces",
        sdk_path / "aragora_sdk" / "resources",
        sdk_path / "src" / "aragora_sdk" / "namespaces",
    ]

    namespaces_dir = None
    for d in possible_dirs:
        if d.exists():
            namespaces_dir = d
            break

    if not namespaces_dir:
        print(f"Warning: Python SDK namespaces dir not found in {sdk_path}")
        return endpoints

    # Patterns to match endpoint calls
    patterns = [
        # self._client.request("METHOD", "/api/...")
        re.compile(r'self\._client\.request\s*\(\s*["\'](\w+)["\']\s*,\s*[f]?["\']([^"\']+)["\']'),
        # self._client.get("/api/...")
        re.compile(r'self\._client\.get\s*\(\s*[f]?["\']([^"\']+)["\']'),
        # self._client.post("/api/...")
        re.compile(r'self\._client\.post\s*\(\s*[f]?["\']([^"\']+)["\']'),
    ]

    for py_file in namespaces_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue

        namespace = py_file.stem
        content = py_file.read_text(errors="ignore")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue

            for pattern in patterns:
                for match in pattern.finditer(line):
                    groups = match.groups()
                    if len(groups) == 2:
                        method, path = groups
                    else:
                        method_match = re.search(r"_client\.(\w+)", line)
                        method = method_match.group(1).upper() if method_match else "GET"
                        path = groups[0]

                    # Normalize path (handle f-strings)
                    path = re.sub(r"\{[^}]+\}", "*", path)

                    if path.startswith("/api"):
                        endpoints.append(
                            Endpoint(
                                method=method.upper() if isinstance(method, str) else "GET",
                                path=path,
                                source_file=str(py_file.relative_to(sdk_path)),
                                line_number=line_num,
                                namespace=namespace,
                            )
                        )

    return endpoints


def extract_handler_routes(aragora_path: Path) -> list[str]:
    """Extract registered routes from handler files."""
    routes: list[str] = []
    handlers_dir = aragora_path / "server" / "handlers"

    if not handlers_dir.exists():
        print(f"Warning: Handlers dir not found: {handlers_dir}")
        return routes

    # Pattern to match ROUTES arrays
    routes_pattern = re.compile(r"ROUTES\s*=\s*\[(.*?)\]", re.DOTALL)
    route_string_pattern = re.compile(r'["\']([^"\']+)["\']')

    for py_file in handlers_dir.rglob("*.py"):
        content = py_file.read_text(errors="ignore")

        for routes_match in routes_pattern.finditer(content):
            routes_block = routes_match.group(1)
            for route_match in route_string_pattern.finditer(routes_block):
                route = route_match.group(1)
                if route.startswith("/api"):
                    routes.append(route)

    # Also check handler_registry for PREFIX_PATTERNS
    registry_dir = aragora_path / "server" / "handler_registry"
    if registry_dir.exists():
        for py_file in registry_dir.glob("*.py"):
            content = py_file.read_text(errors="ignore")
            # Look for prefix patterns that indicate route handling
            prefix_pattern = re.compile(r'"/api/v1/([^"]+)"')
            for match in prefix_pattern.finditer(content):
                routes.append(f"/api/v1/{match.group(1)}")

    return list(set(routes))


def strip_version_prefix(path: str) -> str:
    """Strip API version prefix from path (e.g., /api/v1/foo -> /api/foo)."""
    # Match /api/v1/, /api/v2/, etc.
    return re.sub(r"^/api/v\d+/", "/api/", path)


def route_matches(endpoint_path: str, handler_routes: list[str]) -> bool:
    """Check if an endpoint path matches any handler route."""
    # Normalize the endpoint path
    normalized_endpoint = endpoint_path.rstrip("/")
    # Also try version-stripped path (handlers often omit version)
    stripped_endpoint = strip_version_prefix(normalized_endpoint)

    for route in handler_routes:
        normalized_route = route.rstrip("/")
        stripped_route = strip_version_prefix(normalized_route)

        # Try matching both versioned and unversioned paths
        for ep in [normalized_endpoint, stripped_endpoint]:
            for rt in [normalized_route, stripped_route]:
                # Exact match
                if ep == rt:
                    return True

                # Wildcard match (route ends with /* or contains /*)
                if "*" in rt:
                    # Convert route pattern to regex
                    pattern = rt.replace("**", ".*").replace("*", "[^/]+")
                    if re.match(f"^{pattern}$", ep):
                        return True

                # Wildcard in endpoint (from f-string interpolation)
                if "*" in ep:
                    pattern = ep.replace("**", ".*").replace("*", "[^/]+")
                    if re.match(f"^{pattern}$", rt):
                        return True

        # Prefix match for handlers using startswith
        endpoint_parts = stripped_endpoint.split("/")
        route_parts = stripped_route.split("/")
        if len(route_parts) >= 3 and len(endpoint_parts) >= 3:
            # Match on /api/prefix (after stripping version)
            if route_parts[:3] == endpoint_parts[:3]:
                return True

    return False


def analyze_parity(
    ts_endpoints: list[Endpoint],
    py_endpoints: list[Endpoint],
    handler_routes: list[str],
) -> ParityReport:
    """Analyze SDK-Handler parity and generate report."""
    report = ParityReport(
        ts_endpoints=ts_endpoints,
        py_endpoints=py_endpoints,
        handler_routes=handler_routes,
    )

    # Find missing handlers for TypeScript endpoints
    ts_matched = 0
    for endpoint in ts_endpoints:
        if route_matches(endpoint.path, handler_routes):
            ts_matched += 1
        else:
            report.missing_handlers.append(endpoint)

    # Find missing handlers for Python endpoints
    py_matched = 0
    for endpoint in py_endpoints:
        if route_matches(endpoint.path, handler_routes):
            py_matched += 1
        else:
            # Check if not already in missing (by path)
            if not any(e.path == endpoint.path for e in report.missing_handlers):
                report.missing_handlers.append(endpoint)

    # Calculate coverage
    if ts_endpoints:
        report.coverage_ts = (ts_matched / len(ts_endpoints)) * 100
    if py_endpoints:
        report.coverage_py = (py_matched / len(py_endpoints)) * 100

    return report


def print_report(report: ParityReport, verbose: bool = False) -> None:
    """Print the parity report to stdout."""
    print("=" * 70)
    print("SDK-Handler Parity Report")
    print("=" * 70)
    print()

    print(f"TypeScript SDK Endpoints: {len(report.ts_endpoints)}")
    print(f"Python SDK Endpoints:     {len(report.py_endpoints)}")
    print(f"Handler Routes:           {len(report.handler_routes)}")
    print()

    print(f"TypeScript Coverage: {report.coverage_ts:.1f}%")
    print(f"Python Coverage:     {report.coverage_py:.1f}%")
    print()

    print(f"Missing Handlers: {len(report.missing_handlers)}")
    print()

    if report.missing_handlers:
        # Group by namespace
        by_namespace: dict[str, list[Endpoint]] = defaultdict(list)
        for endpoint in report.missing_handlers:
            by_namespace[endpoint.namespace].append(endpoint)

        print("Missing by Namespace:")
        print("-" * 50)
        for namespace in sorted(by_namespace.keys()):
            endpoints = by_namespace[namespace]
            print(f"\n{namespace}: {len(endpoints)} missing")
            if verbose:
                for ep in endpoints[:10]:  # Limit output
                    print(f"  {ep.method:6} {ep.path}")
                if len(endpoints) > 10:
                    print(f"  ... and {len(endpoints) - 10} more")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SDK-Handler Parity Audit")
    parser.add_argument("--verify", action="store_true", help="Quick parity check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    ts_sdk_path = project_root / "sdk" / "typescript"
    py_sdk_path = project_root / "sdk" / "python"
    aragora_path = project_root / "aragora"

    # Extract endpoints
    print("Extracting TypeScript SDK endpoints...", file=sys.stderr)
    ts_endpoints = extract_ts_endpoints(ts_sdk_path)

    print("Extracting Python SDK endpoints...", file=sys.stderr)
    py_endpoints = extract_py_endpoints(py_sdk_path)

    print("Extracting handler routes...", file=sys.stderr)
    handler_routes = extract_handler_routes(aragora_path)

    # Analyze parity
    print("Analyzing parity...", file=sys.stderr)
    report = analyze_parity(ts_endpoints, py_endpoints, handler_routes)

    if args.json:
        output = {
            "ts_endpoints": len(report.ts_endpoints),
            "py_endpoints": len(report.py_endpoints),
            "handler_routes": len(report.handler_routes),
            "coverage_ts": report.coverage_ts,
            "coverage_py": report.coverage_py,
            "missing_count": len(report.missing_handlers),
            "missing": [
                {
                    "method": e.method,
                    "path": e.path,
                    "namespace": e.namespace,
                    "source": e.source_file,
                }
                for e in report.missing_handlers
            ],
        }
        print(json.dumps(output, indent=2))
    elif args.verify:
        # Quick check - just print coverage
        ts_ok = report.coverage_ts >= 95.0
        py_ok = report.coverage_py >= 95.0
        status = "PASS" if (ts_ok and py_ok) else "FAIL"
        print(f"{status}: TS={report.coverage_ts:.1f}% PY={report.coverage_py:.1f}%")
        return 0 if (ts_ok and py_ok) else 1
    else:
        print_report(report, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
