#!/usr/bin/env python3
"""Audit API endpoints for frontend usage.

Compares documented endpoints in handlers with frontend usage to identify
potentially unused endpoints.

Usage:
    python scripts/audit_endpoints.py

Output:
    - List of endpoints documented but not used in frontend
    - Note: Some endpoints may be used by external clients or tests
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def get_documented_endpoints(handlers_dir: Path) -> set[tuple[str, str]]:
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


def categorize_unused(unused: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Categorize unused endpoints by domain."""
    categories: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    for method, path in unused:
        parts = path.split("/")
        if len(parts) >= 3:
            category = parts[2]  # /api/{category}/...
        else:
            category = "root"
        categories[category].append((method, path))

    return dict(categories)


def _mute_stdout_after_broken_pipe() -> None:
    close = getattr(sys.stdout, "close", None)
    if callable(close):
        try:
            close()
        except OSError:
            pass
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _emit_output(output: str) -> None:
    try:
        sys.stdout.write(output)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()


def render_report(
    documented: set[tuple[str, str]],
    frontend_paths: set[str],
    categories: dict[str, list[tuple[str, str]]],
    unused_count: int,
) -> str:
    """Render a human-readable endpoint audit report."""
    lines = [
        "API Endpoint Audit",
        "=" * 60,
        "",
        f"Documented endpoints: {len(documented)}",
        f"Frontend API calls: {len(frontend_paths)}",
        f"Potentially unused: {unused_count}",
        "",
        "-" * 60,
        "Potentially Unused Endpoints by Category",
        "-" * 60,
    ]

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
        lines.extend(("", f"{category}{marker}:"))
        for method, path in sorted(endpoints):
            lines.append(f"  {method:6} {path}")

    lines.extend(
        (
            "",
            "=" * 60,
            "Notes:",
            "- Endpoints marked 'external API' are likely used by:",
            "  - External integrations",
            "  - CLI tools",
            "  - Test suites",
            "- Review before deprecating any endpoint",
            "=" * 60,
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root used to resolve handler and frontend directories.",
    )
    parser.add_argument(
        "--handlers-dir",
        type=Path,
        default=None,
        help="Override the server handlers directory.",
    )
    parser.add_argument(
        "--frontend-dir",
        type=Path,
        default=None,
        help="Override the frontend source directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    handlers_dir = args.handlers_dir or root / "aragora" / "server" / "handlers"
    frontend_dir = args.frontend_dir or root / "aragora" / "live" / "src"

    documented = get_documented_endpoints(handlers_dir)
    frontend_paths = get_frontend_usage(frontend_dir)
    unused = [(method, path) for method, path in documented if path not in frontend_paths]
    categories = categorize_unused(unused)

    _emit_output(render_report(documented, frontend_paths, categories, len(unused)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
