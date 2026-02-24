#!/usr/bin/env python3
"""
Audit SSRF protection coverage across outbound HTTP call sites.

Scans the codebase for outbound HTTP calls (httpx, aiohttp, requests) and
identifies which call sites use SSRF validation (validate_url, is_url_safe,
SSRFValidationError) from aragora.security.ssrf_protection.

Usage:
    python scripts/audit_ssrf_coverage.py              # Full report
    python scripts/audit_ssrf_coverage.py --json       # JSON output
    python scripts/audit_ssrf_coverage.py --summary    # Summary only
    python scripts/audit_ssrf_coverage.py --ci         # CI mode (exit 1 if regression)

Exit codes:
    0 - All checks pass (or --json/--summary mode)
    1 - Unprotected call sites exceed baseline (--ci mode only)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Root of the aragora package
ARAGORA_ROOT = Path(__file__).resolve().parent.parent / "aragora"

# ── Patterns: outbound HTTP calls ──────────────────────────────────────────

# httpx patterns
HTTPX_CLIENT = re.compile(r"httpx\.(AsyncClient|Client)\s*\(")
HTTPX_DIRECT = re.compile(r"httpx\.(get|post|put|patch|delete|head|options|request)\s*\(")
HTTPX_METHOD = re.compile(r"\.(get|post|put|patch|delete|head|options|request|send)\s*\(")

# aiohttp patterns
AIOHTTP_SESSION = re.compile(r"aiohttp\.ClientSession\s*\(")
AIOHTTP_METHOD = re.compile(
    r"session\.(get|post|put|patch|delete|head|options|request|ws_connect)\s*\("
)

# requests patterns
REQUESTS_CALL = re.compile(
    r"requests\.(get|post|put|patch|delete|head|options|request|Session)\s*\("
)

# urllib (rare but worth tracking)
URLLIB_CALL = re.compile(r"urllib\.request\.(urlopen|urlretrieve|Request)\s*\(")

# All outbound patterns grouped
OUTBOUND_PATTERNS = {
    "httpx.Client": HTTPX_CLIENT,
    "httpx.direct": HTTPX_DIRECT,
    "aiohttp.ClientSession": AIOHTTP_SESSION,
    "requests": REQUESTS_CALL,
    "urllib": URLLIB_CALL,
}

# ── Patterns: SSRF protection ─────────────────────────────────────────────

SSRF_IMPORT = re.compile(
    r"from\s+aragora\.security\.ssrf_protection\s+import|"
    r"from\s+aragora\.security\s+import.*ssrf_protection|"
    r"import\s+aragora\.security\.ssrf_protection"
)
SSRF_CALL = re.compile(r"validate_url\s*\(|is_url_safe\s*\(|SSRFValidationError")
SAFE_HTTP_IMPORT = re.compile(
    r"from\s+aragora\.security\.safe_http\s+import|"
    r"from\s+aragora\.security\s+import.*safe_http"
)
SAFE_HTTP_CALL = re.compile(r"safe_get\s*\(|safe_post\s*\(|safe_request\s*\(|SafeHTTPClient")

# ── Known safe categories (no user-controlled URLs) ───────────────────────
# Files whose URLs are entirely from config/env/hardcoded constants
KNOWN_SAFE_FILES: frozenset[str] = frozenset(
    {
        # Internal infra — URLs from env vars / config only
        "observability/tracing.py",
        "observability/load_test.py",
        "caching/decorators.py",
        "config/timeouts.py",
        "storage/teams_workspace_store.py",
        "storage/slack_workspace_store.py",
        # Agent API clients — model provider endpoints from config
        "agents/api_agents/common.py",
        "agents/api_agents/external_framework.py",
        "agents/errors/decorators.py",
        "core/embeddings/backends/openai.py",
        "core/embeddings/backends/ollama.py",
        "core/embeddings/backends/gemini.py",
        "memory/embeddings.py",
        # Self-referencing internal clients
        "client/client.py",
        "http_client.py",
        "server/http_client_pool.py",
        # CLI tools (local execution context)
        "cli/commands/rbac_ops.py",
        "cli/commands/skills.py",
        "cli/commands/memory_ops.py",
        "cli/commands/computer_use.py",
        "cli/commands/connectors.py",
        "cli/commands/billing_ops.py",
        "cli/commands/knowledge.py",
        # Test / training utilities
        "training/tinker_client.py",
    }
)


@dataclass
class CallSite:
    """A single outbound HTTP call site."""

    file: str
    line: int
    pattern_type: str
    code_snippet: str


@dataclass
class FileReport:
    """SSRF protection status for a single file."""

    file: str
    has_ssrf_import: bool = False
    has_ssrf_call: bool = False
    has_safe_http: bool = False
    is_known_safe: bool = False
    call_sites: list[CallSite] = field(default_factory=list)

    @property
    def is_protected(self) -> bool:
        return (
            self.has_ssrf_import or self.has_ssrf_call or self.has_safe_http or self.is_known_safe
        )

    @property
    def protection_type(self) -> str:
        if self.has_ssrf_import or self.has_ssrf_call:
            return "ssrf_validation"
        if self.has_safe_http:
            return "safe_http_client"
        if self.is_known_safe:
            return "known_safe"
        return "unprotected"


def scan_file(filepath: Path, rel_path: str) -> FileReport | None:
    """Scan a single Python file for outbound HTTP calls and SSRF protection."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = content.splitlines()
    report = FileReport(file=rel_path)

    # Check for SSRF protection
    report.has_ssrf_import = bool(SSRF_IMPORT.search(content))
    report.has_ssrf_call = bool(SSRF_CALL.search(content))
    report.has_safe_http = bool(SAFE_HTTP_IMPORT.search(content)) or bool(
        SAFE_HTTP_CALL.search(content)
    )
    report.is_known_safe = rel_path in KNOWN_SAFE_FILES

    # Find all outbound call sites
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        for pattern_type, pattern in OUTBOUND_PATTERNS.items():
            if pattern.search(line):
                report.call_sites.append(
                    CallSite(
                        file=rel_path,
                        line=i,
                        pattern_type=pattern_type,
                        code_snippet=stripped[:120],
                    )
                )

    # Only return files that actually have outbound calls
    return report if report.call_sites else None


def scan_codebase(root: Path) -> list[FileReport]:
    """Scan all Python files under the aragora package."""
    reports = []
    for filepath in sorted(root.rglob("*.py")):
        # Skip test files, __pycache__, and the ssrf_protection module itself
        rel = str(filepath.relative_to(root))
        if "__pycache__" in rel:
            continue
        if rel == "security/ssrf_protection.py":
            continue
        if rel == "security/safe_http.py":
            continue

        report = scan_file(filepath, rel)
        if report:
            reports.append(report)

    return reports


def print_report(reports: list[FileReport], summary_only: bool = False) -> None:
    """Print a human-readable report."""
    protected = [r for r in reports if r.is_protected]
    unprotected = [r for r in reports if not r.is_protected]

    total_call_sites = sum(len(r.call_sites) for r in reports)
    protected_sites = sum(len(r.call_sites) for r in protected)
    unprotected_sites = sum(len(r.call_sites) for r in unprotected)

    # Group unprotected by directory
    by_dir: dict[str, list[FileReport]] = defaultdict(list)
    for r in unprotected:
        d = r.file.rsplit("/", 1)[0] if "/" in r.file else "."
        by_dir[d].append(r)

    print("=" * 72)
    print("SSRF Protection Coverage Audit")
    print("=" * 72)
    print()
    print(f"  Files with outbound HTTP calls: {len(reports)}")
    print(f"  Total outbound call sites:      {total_call_sites}")
    print()
    print(
        f"  Protected files:    {len(protected):>4}  ({len(protected) * 100 // max(len(reports), 1)}%)"
    )
    print(
        f"  Unprotected files:  {len(unprotected):>4}  ({len(unprotected) * 100 // max(len(reports), 1)}%)"
    )
    print()
    print(f"  Protected call sites:   {protected_sites:>4}")
    print(f"  Unprotected call sites: {unprotected_sites:>4}")
    print()

    # Protection breakdown
    by_type: dict[str, int] = defaultdict(int)
    for r in protected:
        by_type[r.protection_type] += 1
    if by_type:
        print("  Protection breakdown:")
        for ptype, count in sorted(by_type.items()):
            print(f"    {ptype}: {count} files")
        print()

    if summary_only:
        return

    if unprotected:
        print("-" * 72)
        print("UNPROTECTED FILES (need SSRF validation)")
        print("-" * 72)
        for dirname in sorted(by_dir):
            print(f"\n  {dirname}/")
            for r in sorted(by_dir[dirname], key=lambda x: x.file):
                sites = len(r.call_sites)
                patterns = {s.pattern_type for s in r.call_sites}
                print(
                    f"    {r.file.rsplit('/', 1)[-1]}  ({sites} call{'s' if sites > 1 else ''}: {', '.join(sorted(patterns))})"
                )
                for s in r.call_sites[:3]:  # Show first 3 call sites
                    print(f"      L{s.line}: {s.code_snippet[:90]}")
                if len(r.call_sites) > 3:
                    print(f"      ... and {len(r.call_sites) - 3} more")
        print()

    if protected:
        print("-" * 72)
        print("PROTECTED FILES")
        print("-" * 72)
        for r in sorted(protected, key=lambda x: x.file):
            print(f"  {r.file}  [{r.protection_type}]")
        print()


def print_json(reports: list[FileReport]) -> None:
    """Print a JSON report."""
    protected = [r for r in reports if r.is_protected]
    unprotected = [r for r in reports if not r.is_protected]

    output = {
        "summary": {
            "total_files": len(reports),
            "total_call_sites": sum(len(r.call_sites) for r in reports),
            "protected_files": len(protected),
            "unprotected_files": len(unprotected),
            "protected_call_sites": sum(len(r.call_sites) for r in protected),
            "unprotected_call_sites": sum(len(r.call_sites) for r in unprotected),
            "coverage_pct": round(len(protected) * 100 / max(len(reports), 1), 1),
        },
        "unprotected": [
            {
                "file": r.file,
                "call_sites": [asdict(s) for s in r.call_sites],
            }
            for r in sorted(unprotected, key=lambda x: x.file)
        ],
        "protected": [
            {
                "file": r.file,
                "protection_type": r.protection_type,
            }
            for r in sorted(protected, key=lambda x: x.file)
        ],
    }
    print(json.dumps(output, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SSRF protection coverage")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Summary only")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit 1 if unprotected files exceed baseline",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=163,
        help="Maximum unprotected files before CI fails (default: 163)",
    )
    args = parser.parse_args()

    reports = scan_codebase(ARAGORA_ROOT)

    if args.json:
        print_json(reports)
        return 0

    print_report(reports, summary_only=args.summary)

    if args.ci:
        unprotected_count = sum(1 for r in reports if not r.is_protected)
        if unprotected_count > args.baseline:
            print(
                f"\nCI FAIL: {unprotected_count} unprotected files exceed "
                f"baseline of {args.baseline}"
            )
            return 1
        print(
            f"\nCI PASS: {unprotected_count} unprotected files within baseline of {args.baseline}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
