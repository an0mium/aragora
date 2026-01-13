#!/usr/bin/env python3
"""
Audit API endpoints for usage.

Compares documented endpoints in handlers with frontend usage to identify
potentially unused endpoints.

Usage:
    python scripts/audit_endpoints.py

Output:
    - List of endpoints documented but not used in frontend
    - Note: Some endpoints may be used by external clients or tests
"""

import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict


def get_documented_endpoints(handlers_dir: Path) -> set[str]:
    """Extract endpoints from handler docstrings."""
    endpoints = set()
    pattern = re.compile(r"(GET|POST|PUT|DELETE|PATCH)\s+(/api/[^\s]+)")

    for handler_file in handlers_dir.glob("*.py"):
        try:
            content = handler_file.read_text()
            for match in pattern.finditer(content):
                method, path = match.groups()
                # Normalize path parameters
                normalized = re.sub(r"\{[^}]+\}", "*", path)
                normalized = re.sub(r":[a-z_]+", "*", normalized)
                endpoints.add((method, normalized))
        except Exception as e:
            print(f"Error reading {handler_file}: {e}")

    return endpoints


def get_frontend_usage(frontend_dir: Path) -> set[str]:
    """Extract API paths used in frontend code."""
    paths = set()
    pattern = re.compile(r'["\'](/api/[^"\']+)["\']')

    for src_file in frontend_dir.rglob("*.ts"):
        try:
            content = src_file.read_text()
            for match in pattern.finditer(content):
                path = match.group(1)
                # Normalize path parameters
                normalized = re.sub(r"\$\{[^}]+\}", "*", path)
                paths.add(normalized)
        except Exception as e:
            pass  # Skip unreadable files

    for src_file in frontend_dir.rglob("*.tsx"):
        try:
            content = src_file.read_text()
            for match in pattern.finditer(content):
                path = match.group(1)
                normalized = re.sub(r"\$\{[^}]+\}", "*", path)
                paths.add(normalized)
        except Exception as e:
            pass

    return paths


def categorize_unused(unused: list[tuple[str, str]]) -> dict[str, list]:
    """Categorize unused endpoints by domain."""
    categories = defaultdict(list)

    for method, path in unused:
        parts = path.split("/")
        if len(parts) >= 3:
            category = parts[2]  # /api/{category}/...
        else:
            category = "root"
        categories[category].append((method, path))

    return dict(categories)


def main():
    root = Path(__file__).parent.parent
    handlers_dir = root / "aragora" / "server" / "handlers"
    frontend_dir = root / "aragora" / "live" / "src"

    print("API Endpoint Audit")
    print("=" * 60)

    # Get endpoints
    documented = get_documented_endpoints(handlers_dir)
    frontend_paths = get_frontend_usage(frontend_dir)

    print(f"\nDocumented endpoints: {len(documented)}")
    print(f"Frontend API calls: {len(frontend_paths)}")

    # Find unused
    unused = []
    for method, path in documented:
        if path not in frontend_paths:
            unused.append((method, path))

    print(f"Potentially unused: {len(unused)}")

    # Categorize
    categories = categorize_unused(unused)

    print("\n" + "-" * 60)
    print("Potentially Unused Endpoints by Category")
    print("-" * 60)

    # Known external APIs (don't deprecate)
    external_apis = {
        "auth",
        "billing",
        "webhook",
        "oauth",
        "api-key",
        "health",
        "status",
        "metrics",
        "admin",
    }

    for category in sorted(categories.keys()):
        endpoints = categories[category]
        is_external = category in external_apis
        marker = " (likely external API)" if is_external else ""
        print(f"\n{category}{marker}:")
        for method, path in sorted(endpoints):
            print(f"  {method:6} {path}")

    print("\n" + "=" * 60)
    print("Notes:")
    print("- Endpoints marked 'external API' are likely used by:")
    print("  - External integrations")
    print("  - CLI tools")
    print("  - Test suites")
    print("- Review before deprecating any endpoint")
    print("=" * 60)


if __name__ == "__main__":
    main()
