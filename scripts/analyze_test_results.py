#!/usr/bin/env python3
"""
Test Results Analyzer for CI/CD.

Parses pytest output to identify:
- Flaky tests (passed on rerun)
- Consistently failing tests
- Skip/xfail patterns
- Test duration outliers

Usage:
    # Parse JUnit XML report
    python scripts/analyze_test_results.py --junit report.xml

    # Parse pytest output directly
    pytest tests/ --tb=short 2>&1 | python scripts/analyze_test_results.py --stdin

    # Generate markdown summary for PR comment
    python scripts/analyze_test_results.py --junit report.xml --format markdown
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Single test result."""

    name: str
    file: str
    outcome: str  # passed, failed, skipped, xfailed, xpassed
    duration: float = 0.0
    error_message: str = ""
    reruns: int = 0
    is_flaky: bool = False


@dataclass
class TestAnalysis:
    """Aggregated test analysis."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    flaky: int = 0
    slowest: list[TestResult] = field(default_factory=list)
    flaky_tests: list[TestResult] = field(default_factory=list)
    failed_tests: list[TestResult] = field(default_factory=list)
    skip_reasons: dict[str, int] = field(default_factory=dict)
    by_module: dict[str, dict[str, int]] = field(default_factory=dict)


def parse_junit_xml(xml_path: Path) -> list[TestResult]:
    """Parse JUnit XML report."""
    results = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "")
            classname = testcase.get("classname", "")
            time_str = testcase.get("time", "0")
            duration = float(time_str) if time_str else 0.0

            # Determine outcome
            outcome = "passed"
            error_msg = ""

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None:
                outcome = "failed"
                error_msg = failure.get("message", "") or failure.text or ""
            elif error is not None:
                outcome = "failed"
                error_msg = error.get("message", "") or error.text or ""
            elif skipped is not None:
                outcome = "skipped"
                error_msg = skipped.get("message", "")

            # Check for reruns (pytest-rerunfailures adds these)
            reruns = 0
            for prop in testcase.iter("property"):
                if prop.get("name") == "reruns":
                    reruns = int(prop.get("value", 0))

            results.append(
                TestResult(
                    name=name,
                    file=classname,
                    outcome=outcome,
                    duration=duration,
                    error_message=error_msg[:500],  # Truncate
                    reruns=reruns,
                    is_flaky=reruns > 0 and outcome == "passed",
                )
            )

    return results


def parse_pytest_output(content: str) -> list[TestResult]:
    """Parse pytest stdout output."""
    results = []

    # Match test result lines: PASSED/FAILED/SKIPPED/etc
    test_pattern = re.compile(r"(tests/\S+\.py)::(\S+)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS)")

    # Match rerun patterns
    rerun_pattern = re.compile(r"RERUN\s+(\d+)")

    current_file = ""
    for line in content.split("\n"):
        match = test_pattern.search(line)
        if match:
            file_path, test_name, outcome = match.groups()
            current_file = file_path

            # Check for reruns
            reruns = 0
            rerun_match = rerun_pattern.search(line)
            if rerun_match:
                reruns = int(rerun_match.group(1))

            outcome_map = {
                "PASSED": "passed",
                "FAILED": "failed",
                "SKIPPED": "skipped",
                "XFAIL": "xfailed",
                "XPASS": "xpassed",
            }

            results.append(
                TestResult(
                    name=test_name,
                    file=file_path,
                    outcome=outcome_map.get(outcome, "unknown"),
                    reruns=reruns,
                    is_flaky=reruns > 0 and outcome == "PASSED",
                )
            )

    return results


def analyze_results(results: list[TestResult]) -> TestAnalysis:
    """Analyze test results."""
    analysis = TestAnalysis()
    analysis.total = len(results)

    for r in results:
        # Count by outcome
        if r.outcome == "passed":
            analysis.passed += 1
        elif r.outcome == "failed":
            analysis.failed += 1
            analysis.failed_tests.append(r)
        elif r.outcome == "skipped":
            analysis.skipped += 1
            # Track skip reasons
            reason = r.error_message or "No reason given"
            reason = reason.split("\n")[0][:100]  # First line, truncated
            analysis.skip_reasons[reason] = analysis.skip_reasons.get(reason, 0) + 1
        elif r.outcome == "xfailed":
            analysis.xfailed += 1
        elif r.outcome == "xpassed":
            analysis.xpassed += 1

        # Track flaky
        if r.is_flaky:
            analysis.flaky += 1
            analysis.flaky_tests.append(r)

        # Track by module
        module = r.file.split("/")[1] if "/" in r.file else r.file
        if module not in analysis.by_module:
            analysis.by_module[module] = defaultdict(int)
        analysis.by_module[module][r.outcome] += 1

    # Find slowest tests
    sorted_by_time = sorted(results, key=lambda x: x.duration, reverse=True)
    analysis.slowest = sorted_by_time[:10]

    return analysis


def format_json(analysis: TestAnalysis) -> str:
    """Format analysis as JSON."""
    return json.dumps(
        {
            "summary": {
                "total": analysis.total,
                "passed": analysis.passed,
                "failed": analysis.failed,
                "skipped": analysis.skipped,
                "xfailed": analysis.xfailed,
                "xpassed": analysis.xpassed,
                "flaky": analysis.flaky,
                "pass_rate": round(analysis.passed / analysis.total * 100, 1)
                if analysis.total
                else 0,
            },
            "flaky_tests": [
                {"name": t.name, "file": t.file, "reruns": t.reruns} for t in analysis.flaky_tests
            ],
            "failed_tests": [
                {"name": t.name, "file": t.file, "error": t.error_message[:200]}
                for t in analysis.failed_tests
            ],
            "slowest_tests": [
                {"name": t.name, "file": t.file, "duration": round(t.duration, 2)}
                for t in analysis.slowest
            ],
            "skip_reasons": dict(sorted(analysis.skip_reasons.items(), key=lambda x: -x[1])[:10]),
            "by_module": {k: dict(v) for k, v in analysis.by_module.items()},
            "timestamp": datetime.utcnow().isoformat(),
        },
        indent=2,
    )


def format_markdown(analysis: TestAnalysis) -> str:
    """Format analysis as Markdown for PR comment."""
    lines = []
    lines.append("## Test Results Summary\n")

    # Summary badge-style
    pass_rate = round(analysis.passed / analysis.total * 100, 1) if analysis.total else 0
    status = "✅" if analysis.failed == 0 else "❌"
    lines.append(f"{status} **{pass_rate}% pass rate** ({analysis.passed}/{analysis.total})\n")

    # Quick stats
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Passed | {analysis.passed} |")
    lines.append(f"| Failed | {analysis.failed} |")
    lines.append(f"| Skipped | {analysis.skipped} |")
    lines.append(f"| Flaky | {analysis.flaky} |")
    lines.append("")

    # Flaky tests
    if analysis.flaky_tests:
        lines.append("### ⚠️ Flaky Tests\n")
        lines.append("Tests that passed on rerun:\n")
        for t in analysis.flaky_tests[:5]:
            lines.append(f"- `{t.file}::{t.name}` (reruns: {t.reruns})")
        if len(analysis.flaky_tests) > 5:
            lines.append(f"- ... and {len(analysis.flaky_tests) - 5} more")
        lines.append("")

    # Failed tests
    if analysis.failed_tests:
        lines.append("### ❌ Failed Tests\n")
        for t in analysis.failed_tests[:10]:
            lines.append(f"- `{t.file}::{t.name}`")
            if t.error_message:
                lines.append(f"  > {t.error_message[:100]}...")
        if len(analysis.failed_tests) > 10:
            lines.append(f"- ... and {len(analysis.failed_tests) - 10} more")
        lines.append("")

    # Slowest tests (if >1s)
    slow_tests = [t for t in analysis.slowest if t.duration > 1.0]
    if slow_tests:
        lines.append("### 🐢 Slowest Tests\n")
        for t in slow_tests[:5]:
            lines.append(f"- `{t.name}` - {t.duration:.1f}s")
        lines.append("")

    return "\n".join(lines)


def format_text(analysis: TestAnalysis) -> str:
    """Format analysis as plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("TEST RESULTS ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    pass_rate = round(analysis.passed / analysis.total * 100, 1) if analysis.total else 0
    lines.append(f"Pass Rate: {pass_rate}%")
    lines.append(f"Total: {analysis.total}")
    lines.append(f"Passed: {analysis.passed}")
    lines.append(f"Failed: {analysis.failed}")
    lines.append(f"Skipped: {analysis.skipped}")
    lines.append(f"Flaky: {analysis.flaky}")
    lines.append("")

    if analysis.flaky_tests:
        lines.append("FLAKY TESTS (passed on rerun)")
        lines.append("-" * 40)
        for t in analysis.flaky_tests[:10]:
            lines.append(f"  {t.file}::{t.name} (reruns: {t.reruns})")
        lines.append("")

    if analysis.failed_tests:
        lines.append("FAILED TESTS")
        lines.append("-" * 40)
        for t in analysis.failed_tests[:10]:
            lines.append(f"  {t.file}::{t.name}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze test results")
    parser.add_argument("--junit", type=Path, help="JUnit XML report file")
    parser.add_argument("--stdin", action="store_true", help="Read pytest output from stdin")
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--output", type=Path, help="Output file (default: stdout)")
    args = parser.parse_args()

    # Parse results
    results = []
    if args.junit and args.junit.exists():
        results = parse_junit_xml(args.junit)
    elif args.stdin:
        content = sys.stdin.read()
        results = parse_pytest_output(content)
    else:
        print("Error: Provide --junit file or --stdin", file=sys.stderr)
        return 1

    if not results:
        print("No test results found", file=sys.stderr)
        return 1

    # Analyze
    analysis = analyze_results(results)

    # Format output
    if args.format == "json":
        output = format_json(analysis)
    elif args.format == "markdown":
        output = format_markdown(analysis)
    else:
        output = format_text(analysis)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Wrote analysis to {args.output}")
    else:
        print(output)

    # Exit with error if tests failed
    return 1 if analysis.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
