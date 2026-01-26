#!/usr/bin/env python
"""Analyze OpenAPI schema coverage and identify gaps.

This script compares the OpenAPI spec against handler implementations
to identify endpoints missing proper schema definitions.

Usage:
    python scripts/analyze_openapi_coverage.py
    python scripts/analyze_openapi_coverage.py --json  # Output as JSON
    python scripts/analyze_openapi_coverage.py --verbose  # Show all endpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_openapi_spec() -> dict[str, Any]:
    """Load the OpenAPI specification."""
    spec_path = Path("docs/api/openapi.json")
    if not spec_path.exists():
        print("Error: docs/api/openapi.json not found. Run scripts/export_openapi.py first.")
        sys.exit(1)

    with open(spec_path) as f:
        return json.load(f)


def analyze_endpoint(method: str, path: str, operation: dict[str, Any]) -> dict[str, Any]:
    """Analyze a single endpoint for schema coverage."""
    result = {
        "method": method.upper(),
        "path": path,
        "operation_id": operation.get("operationId", "unknown"),
        "tags": operation.get("tags", []),
        "has_request_schema": False,
        "has_response_schema": False,
        "request_schema_ref": None,
        "response_schema_ref": None,
        "issues": [],
    }

    # Check request body schema
    if "requestBody" in operation:
        content = operation["requestBody"].get("content", {})
        for content_type, details in content.items():
            schema = details.get("schema", {})
            if "$ref" in schema:
                result["has_request_schema"] = True
                result["request_schema_ref"] = schema["$ref"]
            elif schema.get("type") not in (None, "object") or schema.get("properties"):
                result["has_request_schema"] = True
                result["request_schema_ref"] = "inline"

        if not result["has_request_schema"]:
            result["issues"].append("Request body without schema definition")

    # Check response schema
    responses = operation.get("responses", {})
    for code, response in responses.items():
        if code.startswith("2"):  # Success responses
            content = response.get("content", {})
            for content_type, details in content.items():
                schema = details.get("schema", {})
                if "$ref" in schema:
                    result["has_response_schema"] = True
                    result["response_schema_ref"] = schema["$ref"]
                elif schema.get("type") not in (None, "object") or schema.get("properties"):
                    result["has_response_schema"] = True
                    result["response_schema_ref"] = "inline"

            if not result["has_response_schema"] and content:
                result["issues"].append(f"Response {code} without schema definition")
            break

    return result


def analyze_coverage(spec: dict[str, Any]) -> dict[str, Any]:
    """Analyze OpenAPI schema coverage."""
    endpoints = []

    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method in ("get", "post", "put", "patch", "delete"):
                analysis = analyze_endpoint(method, path, operation)
                endpoints.append(analysis)

    # Compute statistics
    total = len(endpoints)
    with_request = sum(1 for e in endpoints if e["has_request_schema"])
    with_response = sum(1 for e in endpoints if e["has_response_schema"])
    with_issues = sum(1 for e in endpoints if e["issues"])

    # Group by tag
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for endpoint in endpoints:
        tags = endpoint["tags"] or ["Untagged"]
        for tag in tags:
            by_tag.setdefault(tag, []).append(endpoint)

    # Find highest priority gaps (endpoints with request body but no schema)
    priority_gaps = [
        e
        for e in endpoints
        if "requestBody" in spec["paths"].get(e["path"], {}).get(e["method"].lower(), {})
        and not e["has_request_schema"]
    ]

    return {
        "summary": {
            "total_operations": total,
            "with_request_schema": with_request,
            "with_response_schema": with_response,
            "with_issues": with_issues,
            "request_coverage_pct": round(100 * with_request / max(total, 1), 1),
            "response_coverage_pct": round(100 * with_response / max(total, 1), 1),
        },
        "by_tag": {
            tag: {
                "total": len(eps),
                "with_response_schema": sum(1 for e in eps if e["has_response_schema"]),
                "coverage_pct": round(
                    100 * sum(1 for e in eps if e["has_response_schema"]) / len(eps), 1
                ),
            }
            for tag, eps in sorted(by_tag.items())
        },
        "priority_gaps": priority_gaps[:20],  # Top 20 priority gaps
        "endpoints": endpoints,
    }


def print_report(analysis: dict[str, Any], verbose: bool = False) -> None:
    """Print a human-readable coverage report."""
    summary = analysis["summary"]

    print("=" * 60)
    print("OpenAPI Schema Coverage Report")
    print("=" * 60)
    print()
    print(f"Total operations: {summary['total_operations']}")
    print(
        f"Request schema coverage: {summary['with_request_schema']} ({summary['request_coverage_pct']}%)"
    )
    print(
        f"Response schema coverage: {summary['with_response_schema']} ({summary['response_coverage_pct']}%)"
    )
    print(f"Operations with issues: {summary['with_issues']}")
    print()

    print("Coverage by Tag:")
    print("-" * 50)
    for tag, stats in sorted(analysis["by_tag"].items(), key=lambda x: x[1]["coverage_pct"]):
        bar_len = int(stats["coverage_pct"] / 5)  # 20 char max
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  {tag:25} {bar} {stats['coverage_pct']:5.1f}% ({stats['with_response_schema']}/{stats['total']})"
        )
    print()

    if analysis["priority_gaps"]:
        print("Priority Gaps (endpoints with request body but no schema):")
        print("-" * 50)
        for gap in analysis["priority_gaps"][:10]:
            print(f"  {gap['method']:6} {gap['path']}")
        if len(analysis["priority_gaps"]) > 10:
            print(f"  ... and {len(analysis['priority_gaps']) - 10} more")
        print()

    if verbose:
        print("All Endpoints Missing Response Schema:")
        print("-" * 50)
        missing = [e for e in analysis["endpoints"] if not e["has_response_schema"]]
        for endpoint in sorted(missing, key=lambda x: (x["tags"] or ["Z"])[0] + x["path"]):
            tags = ", ".join(endpoint["tags"]) if endpoint["tags"] else "Untagged"
            print(f"  [{tags}] {endpoint['method']:6} {endpoint['path']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OpenAPI schema coverage")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all missing endpoints")
    args = parser.parse_args()

    spec = load_openapi_spec()
    analysis = analyze_coverage(spec)

    if args.json:
        # Remove full endpoint list for cleaner JSON output
        output = {k: v for k, v in analysis.items() if k != "endpoints"}
        print(json.dumps(output, indent=2))
    else:
        print_report(analysis, verbose=args.verbose)


if __name__ == "__main__":
    main()
