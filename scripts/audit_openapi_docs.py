#!/usr/bin/env python3
"""
OpenAPI Documentation Audit Script for Aragora.

Analyzes OpenAPI endpoint definitions and reports documentation coverage.
Helps identify endpoints missing descriptions, operationIds, and other required fields.

Usage:
    python scripts/audit_openapi_docs.py [--min-coverage 80] [--fail-on-missing] [--json]

Exit codes:
    0: Success (coverage meets threshold)
    1: Coverage below threshold or missing critical fields
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class EndpointDoc:
    """Documentation status for a single endpoint operation."""

    path: str
    method: str
    has_description: bool = False
    has_operation_id: bool = False
    has_summary: bool = False
    has_parameters_docs: bool = True  # True if no params or all params documented
    has_responses_docs: bool = False
    has_security: bool = False
    has_examples: bool = False
    tags: List[str] = field(default_factory=list)
    file: str = ""

    @property
    def doc_score(self) -> float:
        """Calculate documentation completeness score (0-100)."""
        checks = [
            self.has_description,
            self.has_operation_id,
            self.has_summary,
            self.has_parameters_docs,
            self.has_responses_docs,
        ]
        return (sum(checks) / len(checks)) * 100

    @property
    def is_fully_documented(self) -> bool:
        """Check if endpoint meets minimum documentation standards."""
        return (
            self.has_description
            and self.has_operation_id
            and self.has_summary
            and self.has_parameters_docs
        )


@dataclass
class AuditReport:
    """Complete audit report."""

    endpoints: List[EndpointDoc] = field(default_factory=list)
    files_scanned: int = 0
    total_endpoints: int = 0

    @property
    def coverage_percent(self) -> float:
        """Calculate overall documentation coverage."""
        if not self.endpoints:
            return 0.0
        scores = [e.doc_score for e in self.endpoints]
        return sum(scores) / len(scores)

    @property
    def fully_documented_count(self) -> int:
        """Count of fully documented endpoints."""
        return sum(1 for e in self.endpoints if e.is_fully_documented)

    @property
    def missing_description(self) -> List[EndpointDoc]:
        """Endpoints missing descriptions."""
        return [e for e in self.endpoints if not e.has_description]

    @property
    def missing_operation_id(self) -> List[EndpointDoc]:
        """Endpoints missing operationId."""
        return [e for e in self.endpoints if not e.has_operation_id]

    @property
    def by_file(self) -> Dict[str, List[EndpointDoc]]:
        """Group endpoints by source file."""
        result: Dict[str, List[EndpointDoc]] = {}
        for e in self.endpoints:
            if e.file not in result:
                result[e.file] = []
            result[e.file].append(e)
        return result

    @property
    def by_tag(self) -> Dict[str, List[EndpointDoc]]:
        """Group endpoints by tag."""
        result: Dict[str, List[EndpointDoc]] = {}
        for e in self.endpoints:
            for tag in e.tags or ["Untagged"]:
                if tag not in result:
                    result[tag] = []
                result[tag].append(e)
        return result


def analyze_endpoint_operation(
    path: str, method: str, spec: Dict[str, Any], file: str
) -> EndpointDoc:
    """Analyze a single endpoint operation for documentation completeness."""
    doc = EndpointDoc(
        path=path,
        method=method.upper(),
        file=file,
        tags=spec.get("tags", []),
    )

    # Check basic fields
    doc.has_description = bool(spec.get("description", "").strip())
    doc.has_operation_id = bool(spec.get("operationId", "").strip())
    doc.has_summary = bool(spec.get("summary", "").strip())
    doc.has_security = bool(spec.get("security"))
    doc.has_examples = bool(
        spec.get("requestBody", {}).get("content", {}).get("application/json", {}).get("examples")
    )

    # Check parameters documentation
    params = spec.get("parameters", [])
    if params:
        params_with_desc = sum(1 for p in params if p.get("description"))
        doc.has_parameters_docs = params_with_desc == len(params)
    else:
        doc.has_parameters_docs = True  # No params to document

    # Check responses documentation
    responses = spec.get("responses", {})
    if responses:
        # Check if at least 200 response has description
        ok_response = responses.get("200", responses.get("201", {}))
        doc.has_responses_docs = bool(ok_response.get("description"))
    else:
        doc.has_responses_docs = False

    return doc


def scan_endpoint_file(file_path: Path) -> List[EndpointDoc]:
    """Scan a single endpoint definition file."""
    endpoints = []

    try:
        # Read and exec the file to get endpoint definitions
        file_globals: Dict[str, Any] = {}
        exec(file_path.read_text(), file_globals)

        # Find endpoint dictionaries (named *_ENDPOINTS)
        for name, value in file_globals.items():
            if name.endswith("_ENDPOINTS") and isinstance(value, dict):
                for path, methods in value.items():
                    if not isinstance(methods, dict):
                        continue
                    for method, spec in methods.items():
                        if method.lower() in ("get", "post", "put", "delete", "patch"):
                            if isinstance(spec, dict):
                                doc = analyze_endpoint_operation(path, method, spec, file_path.name)
                                endpoints.append(doc)

    except Exception as e:
        print(f"Warning: Failed to parse {file_path.name}: {e}", file=sys.stderr)

    return endpoints


def scan_all_endpoints() -> AuditReport:
    """Scan all endpoint definition files."""
    endpoints_dir = project_root / "aragora" / "server" / "openapi" / "endpoints"
    report = AuditReport()

    if not endpoints_dir.exists():
        print(f"Endpoints directory not found: {endpoints_dir}", file=sys.stderr)
        return report

    for file_path in sorted(endpoints_dir.glob("*.py")):
        if file_path.name.startswith("_") or file_path.name == "__init__.py":
            continue

        report.files_scanned += 1
        file_endpoints = scan_endpoint_file(file_path)
        report.endpoints.extend(file_endpoints)

    report.total_endpoints = len(report.endpoints)
    return report


def print_summary(report: AuditReport) -> None:
    """Print summary report to stdout."""
    print("=" * 70)
    print("OpenAPI Documentation Audit Report")
    print("=" * 70)
    print()
    print(f"Files scanned:        {report.files_scanned}")
    print(f"Total endpoints:      {report.total_endpoints}")
    print(
        f"Fully documented:     {report.fully_documented_count} ({report.fully_documented_count / max(report.total_endpoints, 1) * 100:.1f}%)"
    )
    print(f"Coverage score:       {report.coverage_percent:.1f}%")
    print()

    # Missing descriptions
    missing_desc = report.missing_description
    if missing_desc:
        print(f"Missing description ({len(missing_desc)}):")
        for e in missing_desc[:10]:  # Show first 10
            print(f"  - {e.method} {e.path} ({e.file})")
        if len(missing_desc) > 10:
            print(f"  ... and {len(missing_desc) - 10} more")
        print()

    # Missing operationId
    missing_op_id = report.missing_operation_id
    if missing_op_id:
        print(f"Missing operationId ({len(missing_op_id)}):")
        for e in missing_op_id[:10]:
            print(f"  - {e.method} {e.path} ({e.file})")
        if len(missing_op_id) > 10:
            print(f"  ... and {len(missing_op_id) - 10} more")
        print()

    # Coverage by file
    print("Coverage by file:")
    by_file = report.by_file
    file_scores = []
    for file, endpoints in sorted(by_file.items()):
        avg_score = sum(e.doc_score for e in endpoints) / len(endpoints)
        file_scores.append((file, avg_score, len(endpoints)))

    # Sort by score (worst first)
    file_scores.sort(key=lambda x: x[1])
    for file, score, count in file_scores[:15]:
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"  {bar} {score:5.1f}%  {file} ({count} endpoints)")

    if len(file_scores) > 15:
        print(f"  ... and {len(file_scores) - 15} more files")


def print_json(report: AuditReport) -> None:
    """Print JSON report to stdout."""
    data = {
        "summary": {
            "files_scanned": report.files_scanned,
            "total_endpoints": report.total_endpoints,
            "fully_documented": report.fully_documented_count,
            "coverage_percent": round(report.coverage_percent, 2),
        },
        "by_file": {
            file: {
                "count": len(endpoints),
                "coverage": round(sum(e.doc_score for e in endpoints) / len(endpoints), 2),
                "missing_description": [
                    f"{e.method} {e.path}" for e in endpoints if not e.has_description
                ],
            }
            for file, endpoints in report.by_file.items()
        },
        "priority_list": [
            {
                "path": e.path,
                "method": e.method,
                "file": e.file,
                "score": round(e.doc_score, 2),
                "missing": [
                    field
                    for field, has in [
                        ("description", e.has_description),
                        ("operationId", e.has_operation_id),
                        ("summary", e.has_summary),
                        ("parameters", e.has_parameters_docs),
                        ("responses", e.has_responses_docs),
                    ]
                    if not has
                ],
            }
            for e in sorted(report.endpoints, key=lambda x: x.doc_score)[:50]
        ],
    }
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Audit OpenAPI endpoint documentation coverage")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0,
        help="Minimum coverage percentage required (default: 0, no threshold)",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with error if any endpoint missing description",
    )
    parser.add_argument(
        "--fail-on-missing-operation-id",
        action="store_true",
        help="Exit with error if any endpoint missing operationId",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of text",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Audit only a specific file (e.g., debates.py)",
    )
    args = parser.parse_args()

    # Run audit
    report = scan_all_endpoints()

    # Filter by file if specified
    if args.file:
        report.endpoints = [e for e in report.endpoints if e.file == args.file]
        report.total_endpoints = len(report.endpoints)

    # Output report
    if args.json:
        print_json(report)
    else:
        print_summary(report)

    # Check thresholds
    exit_code = 0

    if args.min_coverage > 0 and report.coverage_percent < args.min_coverage:
        print(
            f"\nERROR: Coverage {report.coverage_percent:.1f}% is below threshold {args.min_coverage}%",
            file=sys.stderr,
        )
        exit_code = 1

    if args.fail_on_missing and report.missing_description:
        print(
            f"\nERROR: {len(report.missing_description)} endpoints missing description",
            file=sys.stderr,
        )
        exit_code = 1

    if args.fail_on_missing_operation_id and report.missing_operation_id:
        print(
            f"\nERROR: {len(report.missing_operation_id)} endpoints missing operationId",
            file=sys.stderr,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
