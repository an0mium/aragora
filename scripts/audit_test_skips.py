#!/usr/bin/env python3
"""
Audit test skip markers and generate categorized report.

This script parses all test files for @pytest.mark.skip, @pytest.mark.skipif,
and pytest.skip() calls, categorizes them by reason, and generates reports.

Usage:
    python scripts/audit_test_skips.py                    # Full report
    python scripts/audit_test_skips.py --count-only       # Just total count
    python scripts/audit_test_skips.py --json             # JSON output
    python scripts/audit_test_skips.py --update-docs      # Update SKIP_AUDIT.md
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SkipMarker:
    """Represents a single skip marker found in test code."""

    file: str
    line: int
    marker_type: str  # 'skip', 'skipif', 'pytest.skip'
    reason: str
    category: str
    condition: Optional[str] = None


# Category patterns - order matters (first match wins)
CATEGORY_PATTERNS = {
    "optional_dependency": [
        r"not.*installed",
        r"import.*error",
        r"importerror",
        r"HAS_\w+",
        r"requires_\w+",
        r"module.*not.*found",
        r"no module",
        r"cannot import",
        r"pip install",
        r"\w+\s+required$",  # "MCP required", "asyncpg required"
        r"^\w+\s+required",  # "aiosqlite required for..."
        r"is installed",  # "python3-saml is installed" (inverted check)
        r"binary.*not.*PATH",  # "whisper.cpp binary not in PATH"
    ],
    "missing_feature": [
        r"not.*available",
        r"not.*implemented",
        r"not yet",
        r"TODO",
        r"WIP",
        r"pending",
        r"placeholder",
        r"not found",  # "Encryption service not found"
        r"unavailable",  # "TournamentManager unavailable"
        r"renamed to",  # "method renamed to..."
    ],
    "integration_dependency": [
        r"redis",
        r"postgres",
        r"database",
        r"supabase",
        r"CI",
        r"server.*not.*running",
        r"requires.*service",
        r"external.*service",
        r"requires running server",
        r"requires.*server",
        r"API.*KEY",  # "Requires ANTHROPIC_API_KEY"
        r"ARAGORA_TEST_",  # "ARAGORA_TEST_SERVER_URL not set"
        r"_URL not set",
        r"_TOKEN not set",
    ],
    "platform_specific": [
        r"platform",
        r"darwin",
        r"windows",
        r"linux",
        r"macos",
        r"win32",
        r"symlink",
        r"GIL",  # Python GIL limitations
        r"main thread",  # "signal.alarm requires main thread"
        r"/etc/",  # "Test requires /etc/hostname"
    ],
    "flaky_test": [
        r"flaky",
        r"unstable",
        r"intermittent",
        r"timing",
        r"race.*condition",
        r"sporadic",
        r"rate.*limited",  # "Rate limited during refresh test"
    ],
    "known_bug": [
        r"bug",
        r"issue",
        r"known",
        r"broken",
        r"fails",
        r"regression",
        r"needs.*fix",
        r"investigation",
        r"API changed",  # "OIDCConfig API changed"
        r"need.*update",
        r"does not support",  # "Handler does not support..."
        r"incompatibility",  # "aiohttp event loop incompatibility"
        r"complex setup",  # "requires complex setup"
        r"complex.*mocking",  # "Requires complex httpx mocking"
        r"import failed",  # "Subprocess import failed"
    ],
    "performance": [
        r"slow",
        r"performance",
        r"timeout",
        r"long.*running",
        r"memory",
    ],
}


def categorize_reason(reason: str) -> str:
    """Categorize a skip reason based on pattern matching."""
    reason_lower = reason.lower()

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, reason_lower, re.IGNORECASE):
                return category

    return "uncategorized"


def extract_string_value(node: ast.expr) -> str:
    """Extract string value from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    elif isinstance(node, ast.JoinedStr):
        # f-string - extract parts
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return ""


def parse_decorator(decorator: ast.expr, file_path: str, line: int) -> Optional[SkipMarker]:
    """Parse a decorator node to extract skip information."""
    # Handle @pytest.mark.skip and @pytest.mark.skipif
    if isinstance(decorator, ast.Call):
        func = decorator.func

        # Check for pytest.mark.skip(reason=...)
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Attribute):
                if (
                    hasattr(func.value, "attr")
                    and func.value.attr == "mark"
                    and func.attr in ("skip", "skipif")
                ):
                    reason = ""
                    condition = ""

                    # Extract reason from keyword args
                    for keyword in decorator.keywords:
                        if keyword.arg == "reason":
                            reason = extract_string_value(keyword.value)

                    # For skipif, extract condition
                    if func.attr == "skipif" and decorator.args:
                        condition = ast.unparse(decorator.args[0])
                        # If no explicit reason, use condition as reason
                        if not reason:
                            reason = f"skipif: {condition}"

                    if not reason and func.attr == "skip":
                        # Check for positional reason
                        if decorator.args:
                            reason = extract_string_value(decorator.args[0])
                        else:
                            reason = "No reason provided"

                    return SkipMarker(
                        file=file_path,
                        line=line,
                        marker_type=func.attr,
                        reason=reason,
                        category=categorize_reason(reason),
                        condition=condition if condition else None,
                    )

    # Handle @pytest.mark.skip without call (bare decorator)
    elif isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Attribute):
            if (
                hasattr(decorator.value, "attr")
                and decorator.value.attr == "mark"
                and decorator.attr == "skip"
            ):
                return SkipMarker(
                    file=file_path,
                    line=line,
                    marker_type="skip",
                    reason="No reason provided",
                    category="uncategorized",
                )

    return None


def find_pytest_skip_calls(tree: ast.AST, file_path: str) -> list[SkipMarker]:
    """Find pytest.skip() calls in the AST."""
    markers = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for pytest.skip(...)
            if isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pytest"
                    and node.func.attr == "skip"
                ):
                    reason = ""
                    if node.args:
                        reason = extract_string_value(node.args[0])
                    for keyword in node.keywords:
                        if keyword.arg == "reason":
                            reason = extract_string_value(keyword.value)

                    if not reason:
                        reason = "No reason provided"

                    markers.append(
                        SkipMarker(
                            file=file_path,
                            line=node.lineno,
                            marker_type="pytest.skip",
                            reason=reason,
                            category=categorize_reason(reason),
                        )
                    )

    return markers


def parse_test_file(file_path: Path) -> list[SkipMarker]:
    """Parse a single test file and extract all skip markers."""
    markers = []

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return markers

    rel_path = str(file_path)

    # Find decorators on functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                marker = parse_decorator(decorator, rel_path, decorator.lineno)
                if marker:
                    markers.append(marker)

    # Find pytest.skip() calls
    markers.extend(find_pytest_skip_calls(tree, rel_path))

    return markers


def audit_skips(tests_dir: Path) -> list[SkipMarker]:
    """Parse all test files and collect skip markers."""
    all_markers = []

    for test_file in tests_dir.rglob("test_*.py"):
        markers = parse_test_file(test_file)
        all_markers.extend(markers)

    # Also check conftest.py files
    for conftest in tests_dir.rglob("conftest.py"):
        markers = parse_test_file(conftest)
        all_markers.extend(markers)

    return all_markers


def generate_report(markers: list[SkipMarker]) -> dict:
    """Generate a structured report from skip markers."""
    by_category = defaultdict(list)
    by_file = defaultdict(list)
    by_type = defaultdict(int)

    for marker in markers:
        by_category[marker.category].append(marker)
        by_file[marker.file].append(marker)
        by_type[marker.marker_type] += 1

    return {
        "total": len(markers),
        "by_category": {k: len(v) for k, v in sorted(by_category.items())},
        "by_type": dict(by_type),
        "by_file": {k: len(v) for k, v in sorted(by_file.items(), key=lambda x: -len(x[1]))},
        "high_skip_files": [
            {"file": k, "count": len(v)}
            for k, v in sorted(by_file.items(), key=lambda x: -len(x[1]))[:10]
        ],
        "markers": [asdict(m) for m in markers],
        "generated_at": datetime.now().isoformat(),
    }


def generate_markdown(report: dict) -> str:
    """Generate markdown documentation from report."""
    lines = [
        "# Test Skip Marker Audit",
        "",
        f"**Generated**: {report['generated_at'][:10]}",
        f"**Total Skip Markers**: {report['total']}",
        "",
        "---",
        "",
        "## Summary by Category",
        "",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ]

    total = report["total"]
    for category, count in sorted(report["by_category"].items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {category} | {count} | {pct:.1f}% |")

    lines.extend(
        [
            "",
            "## Summary by Marker Type",
            "",
            "| Type | Count |",
            "|------|-------|",
        ]
    )

    for marker_type, count in sorted(report["by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{marker_type}` | {count} |")

    lines.extend(
        [
            "",
            "## High-Skip Files (Top 10)",
            "",
            "| File | Skip Count |",
            "|------|------------|",
        ]
    )

    for item in report["high_skip_files"]:
        short_path = item["file"].replace("/Users/armand/Development/aragora/", "")
        lines.append(f"| `{short_path}` | {item['count']} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Category Definitions",
            "",
            "| Category | Description |",
            "|----------|-------------|",
            "| optional_dependency | Missing optional Python package |",
            "| missing_feature | Feature not yet implemented |",
            "| integration_dependency | Requires external service (Redis, Postgres) |",
            "| platform_specific | OS-specific limitation |",
            "| flaky_test | Test has intermittent failures |",
            "| known_bug | Known issue being tracked |",
            "| performance | Too slow or resource-intensive |",
            "| uncategorized | Reason did not match any pattern |",
            "",
            "---",
            "",
            "## Remediation Guidelines",
            "",
            "1. **optional_dependency**: Add to `[project.optional-dependencies.test]` in pyproject.toml",
            "2. **missing_feature**: Create GitHub issue and link in skip reason",
            "3. **integration_dependency**: Ensure CI runs integration tests with services",
            "4. **flaky_test**: Fix root cause or add retry mechanism",
            "5. **known_bug**: Link to GitHub issue in skip reason",
            "6. **uncategorized**: Review and add appropriate category pattern",
            "",
            "---",
            "",
            "## Skip Count Baseline",
            "",
            f"Current baseline: **{report['total']}** skips",
            "",
            "CI will warn if skip count exceeds this baseline.",
            "Update `tests/.skip_baseline` when intentionally adding skips.",
            "",
        ]
    )

    return "\n".join(lines)


def generate_github_issue_template(marker: SkipMarker, category_labels: dict) -> dict:
    """Generate a GitHub issue template for a skip marker."""
    # Map categories to labels
    labels = category_labels.get(marker.category, ["test-skip"])

    # Generate title
    short_file = marker.file.split("/")[-1] if "/" in marker.file else marker.file
    title = f"[Test Skip] {short_file}:{marker.line} - {marker.reason[:50]}"
    if len(marker.reason) > 50:
        title += "..."

    # Generate body
    body = f"""## Test Skip Report

**File**: `{marker.file}`
**Line**: {marker.line}
**Skip Reason**: {marker.reason}
**Category**: {marker.category}
**Marker Type**: `{marker.marker_type}`

### Context

This test is currently skipped and needs attention.

### Acceptance Criteria

- [ ] Root cause identified
- [ ] Fix implemented or skip reason updated
- [ ] Test passes consistently
- [ ] Skip marker removed or linked to this issue

### Related

<!-- Link related PRs or issues here -->

---
_Generated by `scripts/audit_test_skips.py`_
"""

    return {
        "title": title,
        "body": body,
        "labels": labels,
        "file": marker.file,
        "line": marker.line,
        "category": marker.category,
    }


def generate_issues_batch(markers: list[SkipMarker], category_filter: str = None) -> list[dict]:
    """Generate GitHub issue templates for markers."""
    category_labels = {
        "optional_dependency": ["test-skip", "dependencies"],
        "missing_feature": ["test-skip", "feature-request"],
        "integration_dependency": ["test-skip", "ci-infrastructure"],
        "platform_specific": ["test-skip", "platform"],
        "flaky_test": ["test-skip", "flaky-test"],
        "known_bug": ["test-skip", "bug"],
        "performance": ["test-skip", "performance"],
        "uncategorized": ["test-skip", "needs-triage"],
    }

    issues = []
    for marker in markers:
        if category_filter and marker.category != category_filter:
            continue
        issue = generate_github_issue_template(marker, category_labels)
        issues.append(issue)

    return issues


def generate_gh_commands(issues: list[dict], dry_run: bool = True) -> list[str]:
    """Generate gh CLI commands to create issues."""
    commands = []
    for issue in issues:
        labels = ",".join(issue["labels"])
        # Escape special characters in title and body
        title = issue["title"].replace('"', '\\"')
        body = issue["body"].replace('"', '\\"').replace("\n", "\\n")

        cmd = f'gh issue create --title "{title}" --body "{body}" --label "{labels}"'
        if dry_run:
            commands.append(f"# {cmd}")
        else:
            commands.append(cmd)

    return commands


def main():
    parser = argparse.ArgumentParser(description="Audit test skip markers")
    parser.add_argument("--count-only", action="store_true", help="Only output total count")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--update-docs", action="store_true", help="Update tests/SKIP_AUDIT.md")
    parser.add_argument(
        "--generate-issues",
        action="store_true",
        help="Generate GitHub issue templates for skipped tests",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter issues by category (e.g., missing_feature)",
    )
    parser.add_argument(
        "--gh-commands",
        action="store_true",
        help="Output gh CLI commands to create issues",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of issues to generate (default: 10)",
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests"),
        help="Tests directory to scan",
    )
    args = parser.parse_args()

    # Find tests directory
    tests_dir = args.tests_dir
    if not tests_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        tests_dir = script_dir / "tests"

    if not tests_dir.exists():
        print(f"Error: Tests directory not found: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    markers = audit_skips(tests_dir)
    report = generate_report(markers)

    if args.count_only:
        print(report["total"])
        return

    if args.json:
        print(json.dumps(report, indent=2))
        return

    if args.update_docs:
        # Update SKIP_AUDIT.md
        docs_path = tests_dir / "SKIP_AUDIT.md"
        markdown = generate_markdown(report)
        docs_path.write_text(markdown)
        print(f"Updated {docs_path}")

        # Update baseline
        baseline_path = tests_dir / ".skip_baseline"
        baseline_path.write_text(str(report["total"]))
        print(f"Updated {baseline_path} with baseline: {report['total']}")
        return

    if args.generate_issues or args.gh_commands:
        # Generate GitHub issue templates
        issues = generate_issues_batch(markers, args.category)

        # Limit number of issues
        if args.limit and len(issues) > args.limit:
            issues = issues[: args.limit]
            print(f"Limited to {args.limit} issues (use --limit to change)", file=sys.stderr)

        if args.gh_commands:
            # Output gh CLI commands
            commands = generate_gh_commands(issues, dry_run=True)
            print("# GitHub CLI commands to create issues")
            print("# Remove the leading # to execute")
            print(f"# Total issues: {len(issues)}")
            print()
            for cmd in commands:
                print(cmd)
        else:
            # Output JSON issue templates
            print(json.dumps(issues, indent=2))

        return

    # Default: print summary
    print(f"Total skip markers: {report['total']}")
    print("\nBy category:")
    for category, count in sorted(report["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    print("\nBy type:")
    for marker_type, count in sorted(report["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {marker_type}: {count}")

    print("\nTop 5 high-skip files:")
    for item in report["high_skip_files"][:5]:
        short_path = item["file"].replace("/Users/armand/Development/aragora/", "")
        print(f"  {short_path}: {item['count']}")


if __name__ == "__main__":
    main()
