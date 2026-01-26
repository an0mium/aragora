#!/usr/bin/env python3
"""
V1 API Deprecation Tool.

Generates reports on V1 API endpoints that need deprecation decorators
to prepare for sunset on 2026-06-01.

Usage:
    python scripts/add_v1_deprecation.py --report
    python scripts/add_v1_deprecation.py --report --json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

SUNSET_DATE = "2026-06-01"
V1_PATTERN = re.compile(r'["\']\/api\/v1\/([^"\']+)["\']')
DEPRECATED_PATTERN = re.compile(r"@deprecated_endpoint\s*\(")


@dataclass
class EndpointInfo:
    file_path: str
    v1_path: str
    line_number: int = 0
    already_deprecated: bool = False


@dataclass
class DeprecationReport:
    total_files: int = 0
    total_endpoints: int = 0
    already_deprecated: int = 0
    needs_deprecation: int = 0
    endpoints: List[EndpointInfo] = field(default_factory=list)
    files_by_count: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_files": self.total_files,
                "total_endpoints": self.total_endpoints,
                "already_deprecated": self.already_deprecated,
                "needs_deprecation": self.needs_deprecation,
                "coverage": f"{(self.already_deprecated / max(self.total_endpoints, 1)) * 100:.1f}%",
                "sunset_date": SUNSET_DATE,
            },
            "top_files": dict(sorted(self.files_by_count.items(), key=lambda x: -x[1])[:20]),
        }


def find_v1_handlers(handlers_dir: Path) -> List[Path]:
    v1_files = []
    for py_file in handlers_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
        try:
            content = py_file.read_text()
            if "/api/v1/" in content:
                v1_files.append(py_file)
        except Exception:
            pass
    return sorted(v1_files)


def analyze_file(file_path: Path) -> List[EndpointInfo]:
    try:
        content = file_path.read_text()
    except Exception:
        return []

    endpoints = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        matches = V1_PATTERN.findall(line)
        for match in matches:
            v1_path = f"/api/v1/{match}"
            context_start = max(0, i - 10)
            context = "\n".join(lines[context_start:i])
            already_deprecated = bool(DEPRECATED_PATTERN.search(context))
            endpoints.append(
                EndpointInfo(
                    file_path=str(file_path),
                    v1_path=v1_path,
                    line_number=i + 1,
                    already_deprecated=already_deprecated,
                )
            )

    return endpoints


def generate_report(handlers_dir: Path) -> DeprecationReport:
    report = DeprecationReport()
    v1_files = find_v1_handlers(handlers_dir)
    report.total_files = len(v1_files)

    for file_path in v1_files:
        endpoints = analyze_file(file_path)
        if endpoints:
            rel_path = str(file_path.relative_to(handlers_dir.parent.parent))
            report.files_by_count[rel_path] = len(endpoints)

        report.endpoints.extend(endpoints)
        report.total_endpoints += len(endpoints)

        for ep in endpoints:
            if ep.already_deprecated:
                report.already_deprecated += 1
            else:
                report.needs_deprecation += 1

    return report


def main():
    parser = argparse.ArgumentParser(description="V1 API Deprecation Tool")
    parser.add_argument("--report", action="store_true", help="Generate deprecation report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    handlers_dir = project_root / "aragora" / "server" / "handlers"

    if not handlers_dir.exists():
        print(f"Error: Handlers directory not found: {handlers_dir}")
        sys.exit(1)

    if args.report:
        report = generate_report(handlers_dir)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("V1 API DEPRECATION REPORT")
            print("=" * 60)
            print(f"Total files with V1 endpoints: {report.total_files}")
            print(f"Total V1 endpoint references: {report.total_endpoints}")
            print(f"Already deprecated: {report.already_deprecated}")
            print(f"Needs deprecation: {report.needs_deprecation}")
            print(
                f"Coverage: {(report.already_deprecated / max(report.total_endpoints, 1)) * 100:.1f}%"
            )
            print(f"Sunset date: {SUNSET_DATE}")
            print()
            print("Top 20 files by V1 endpoint count:")
            print("-" * 60)
            for path, count in sorted(report.files_by_count.items(), key=lambda x: -x[1])[:20]:
                print(f"  {count:3d}  {path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
