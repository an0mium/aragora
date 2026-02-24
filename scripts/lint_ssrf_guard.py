#!/usr/bin/env python3
"""
CI lint script: detect outbound HTTP calls missing SSRF protection.

Scans Python source files for outbound HTTP call patterns and checks whether
the calling module imports from ``aragora.security.ssrf_protection``. Files that
make HTTP calls without importing SSRF protection are flagged.

Usage:
    python scripts/lint_ssrf_guard.py                  # Scan aragora/
    python scripts/lint_ssrf_guard.py --path aragora/connectors
    python scripts/lint_ssrf_guard.py --strict          # Exit 1 on violations
    python scripts/lint_ssrf_guard.py --json             # Machine-readable output

Exit codes:
    0 - No violations (or --strict not set)
    1 - Violations found (--strict mode)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns that indicate outbound HTTP calls
HTTP_CALL_PATTERNS = [
    # requests library
    re.compile(r"\brequests\.(get|post|put|delete|patch|head|request)\s*\("),
    # httpx sync/async
    re.compile(r"\bhttpx\.(get|post|put|delete|patch|head|request)\s*\("),
    re.compile(r"\bhttpx\.(AsyncClient|Client)\s*\("),
    # aiohttp
    re.compile(r"\baiohttp\.ClientSession\s*\("),
    # urllib
    re.compile(r"\burllib\.request\.(urlopen|Request)\s*\("),
]

# Patterns that indicate SSRF protection is imported/used
SSRF_GUARD_PATTERNS = [
    re.compile(r"from\s+aragora\.security\.ssrf_protection\s+import"),
    re.compile(r"import\s+aragora\.security\.ssrf_protection"),
    re.compile(r"\bvalidate_url\s*\("),
    re.compile(r"\bis_url_safe\s*\("),
    re.compile(r"\bvalidate_webhook_url\s*\("),
    re.compile(r"\bvalidate_slack_url\s*\("),
    re.compile(r"\bvalidate_discord_url\s*\("),
    re.compile(r"\bvalidate_github_url\s*\("),
    re.compile(r"\bvalidate_microsoft_url\s*\("),
    re.compile(r"\bssrf_protection\b"),
]

# Files/directories to skip (tests, scripts, the SSRF module itself)
SKIP_PATTERNS = [
    "tests/",
    "scripts/",
    "__pycache__/",
    ".pyc",
    "ssrf_protection.py",
    "aragora/security/ssrf_protection.py",
    # HTTP client infrastructure (provides its own guards)
    "aragora/http_client.py",
    "aragora/server/http_client_pool.py",
    # Config/timeout modules that reference httpx but don't make calls
    "aragora/config/timeouts.py",
]

# Allowlist: files that are known-safe (internal-only URLs, test infrastructure, etc.)
ALLOWLIST_COMMENT = "# ssrf-safe:"


def _should_skip(path: Path, base: Path) -> bool:
    """Check if a file should be skipped."""
    rel = str(path.relative_to(base))
    return any(skip in rel for skip in SKIP_PATTERNS)


def _has_ssrf_guard(content: str) -> bool:
    """Check if file content imports/uses SSRF protection."""
    return any(p.search(content) for p in SSRF_GUARD_PATTERNS)


def _has_allowlist_comment(content: str) -> bool:
    """Check if file has an ssrf-safe allowlist comment."""
    return ALLOWLIST_COMMENT in content


def _find_http_calls(content: str) -> list[tuple[int, str, str]]:
    """Find HTTP call sites in file content.

    Returns:
        List of (line_number, matched_text, pattern_name) tuples.
    """
    results = []
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        for pattern in HTTP_CALL_PATTERNS:
            match = pattern.search(line)
            if match:
                results.append((i, match.group(0).strip(), pattern.pattern))
                break  # One match per line is enough
    return results


def scan_directory(
    base_path: Path,
) -> list[dict]:
    """Scan a directory for unguarded HTTP calls.

    Returns:
        List of violation dicts with file, line, call, and pattern fields.
    """
    violations = []

    for py_file in sorted(base_path.rglob("*.py")):
        if _should_skip(py_file, base_path):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        # Skip if file has SSRF guard or allowlist
        if _has_ssrf_guard(content) or _has_allowlist_comment(content):
            continue

        calls = _find_http_calls(content)
        if calls:
            rel_path = str(py_file.relative_to(base_path))
            for line_no, matched, pattern in calls:
                violations.append(
                    {
                        "file": rel_path,
                        "line": line_no,
                        "call": matched,
                        "pattern": pattern,
                    }
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint for unguarded outbound HTTP calls (SSRF protection)",
    )
    parser.add_argument(
        "--path",
        default="aragora",
        help="Directory to scan (default: aragora/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if violations found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Resolve path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    scan_path = project_root / args.path

    if not scan_path.is_dir():
        print(f"Error: {scan_path} is not a directory", file=sys.stderr)
        return 1

    violations = scan_directory(scan_path)

    if args.json_output:
        print(json.dumps({"violations": violations, "count": len(violations)}, indent=2))
    else:
        if violations:
            print(f"\nSSRF Guard Lint: {len(violations)} unguarded HTTP call(s) found\n")
            print(f"{'File':<60} {'Line':<6} {'Call'}")
            print("-" * 100)
            for v in violations:
                print(f"{v['file']:<60} {v['line']:<6} {v['call']}")
            print("\nTo fix: import from aragora.security.ssrf_protection and validate URLs")
            print("To allowlist: add a comment '# ssrf-safe: <reason>' to the file")
        else:
            print("SSRF Guard Lint: All outbound HTTP calls are guarded. OK")

    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
