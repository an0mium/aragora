#!/usr/bin/env python3
"""
Pre-release validation script for Aragora.

Runs all release gates locally before pushing a release tag.
Each gate is run independently and reports pass/fail status.

Usage:
    python scripts/pre_release_check.py              # Run all gates
    python scripts/pre_release_check.py --gate secrets-only   # Run single gate
    python scripts/pre_release_check.py --gate version-tag    # Version check
    python scripts/pre_release_check.py --gate status-doc     # STATUS.md check
    python scripts/pre_release_check.py --verbose             # Detailed output

Exit codes:
    0 - All gates passed
    1 - One or more gates failed
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []
_verbose = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def _gate(name: str, passed: bool, detail: str = "") -> bool:
    """Record and print gate result."""
    _results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    line = f"  {marker} {name}: {status}"
    if detail and (_verbose or not passed):
        line += f"  ({detail})"
    _log(line)
    return passed


def _run_cmd(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(PROJECT_ROOT),
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


# ---------------------------------------------------------------------------
# Gate: Hardcoded secrets pattern scan
# ---------------------------------------------------------------------------

# Patterns that indicate hardcoded secrets in source code.
# Each entry: (pattern_name, regex, description)
SECRET_PATTERNS: list[tuple[str, str, str]] = [
    (
        "AWS Access Key",
        r"(?<![A-Z0-9])(AKIA[0-9A-Z]{16})(?![A-Z0-9])",
        "AWS access key ID found in source",
    ),
    (
        "AWS Secret Key",
        r'(?:aws_secret_access_key|secret_key)\s*[=:]\s*["\'][A-Za-z0-9/+=]{40}["\']',
        "AWS secret access key found in source",
    ),
    (
        "Private Key Header",
        r"-----BEGIN\s+(RSA|EC|DSA|OPENSSH)?\s*PRIVATE KEY-----",
        "Private key embedded in source",
    ),
    (
        "Generic API Key Assignment",
        r'(?:api_key|apikey|api_secret|secret_key)\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
        "Hardcoded API key assignment",
    ),
    (
        "Generic Token Assignment",
        r'(?:token|auth_token|access_token|bearer)\s*=\s*["\'][A-Za-z0-9_\-\.]{20,}["\']',
        "Hardcoded token assignment",
    ),
    (
        "Password Assignment",
        r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
        "Hardcoded password assignment",
    ),
]

# Files and directories to exclude from secret scanning
SECRET_SCAN_EXCLUDES = {
    "tests/",
    "test_",
    ".pyc",
    "__pycache__",
    "node_modules/",
    ".git/",
    "dist/",
    "build/",
    ".venv/",
    "venv/",
    ".env.example",
    "docs/",
    "CLAUDE.md",
    "MEMORY.md",
    # Config files that define patterns (not actual secrets)
    "scripts/pre_release_check.py",
    "scripts/security_scan.py",
    ".gitleaks.toml",
    # Migration/fixture files with example data
    "migrations/",
    "fixtures/",
    "examples/",
}


def _should_scan_file(filepath: Path) -> bool:
    """Check if a file should be scanned for secrets."""
    rel = str(filepath.relative_to(PROJECT_ROOT))
    for exclude in SECRET_SCAN_EXCLUDES:
        if exclude in rel:
            return False
    # Only scan Python and config files
    return filepath.suffix in {".py", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".env"}


def gate_secrets_scan() -> bool:
    """Scan source code for hardcoded secret patterns."""
    findings: list[tuple[str, str, int, str]] = []

    # Scan aragora/ directory (main source)
    source_dirs = [PROJECT_ROOT / "aragora", PROJECT_ROOT / "scripts"]

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for filepath in source_dir.rglob("*"):
            if not filepath.is_file():
                continue
            if not _should_scan_file(filepath):
                continue

            try:
                content = filepath.read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            for pattern_name, regex, description in SECRET_PATTERNS:
                for line_num, line in enumerate(content.splitlines(), 1):
                    # Skip comment lines and docstrings
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("//"):
                        continue
                    # Skip regex pattern definitions (e.g. secret scanners)
                    if "re.compile" in line or 'r"' in stripped[:15] or "r'" in stripped[:15]:
                        continue
                    # Skip lines that look like test fixtures, examples, or env lookups
                    if any(
                        marker in line.lower()
                        for marker in [
                            "example",
                            "placeholder",
                            "dummy",
                            "test",
                            "mock",
                            "fake",
                            "sample",
                            "xxx",
                            "changeme",
                            "your_",
                            "your-",
                            "<your",
                            "todo",
                            "fixme",
                            "os.environ",
                            "os.getenv",
                            "getenv",
                            "get_secret",
                            "env.get",
                            "nosec",
                            "noqa",
                        ]
                    ):
                        continue

                    if re.search(regex, line):
                        rel_path = str(filepath.relative_to(PROJECT_ROOT))
                        findings.append((pattern_name, rel_path, line_num, description))

    if findings:
        detail_lines = []
        for pattern_name, path, line_num, desc in findings[:10]:
            detail_lines.append(f"  {path}:{line_num} [{pattern_name}]")
        if len(findings) > 10:
            detail_lines.append(f"  ... and {len(findings) - 10} more")
        detail = "\n".join(detail_lines)
        return _gate(
            "secrets_scan",
            False,
            f"{len(findings)} potential hardcoded secret(s) found:\n{detail}",
        )

    return _gate("secrets_scan", True, "no hardcoded secret patterns detected")


# ---------------------------------------------------------------------------
# Gate: Bandit security scan
# ---------------------------------------------------------------------------


def gate_bandit() -> bool:
    """Run bandit static analysis (HIGH severity only)."""
    code, output = _run_cmd(
        [sys.executable, "-m", "bandit", "-r", "aragora/", "-c", "pyproject.toml", "-lll"],
        timeout=120,
    )
    if code != 0:
        # Parse output for finding count
        lines = output.splitlines()
        issue_lines = [l for l in lines if "Issue:" in l or "Severity:" in l]
        detail = f"{len(issue_lines)} HIGH severity finding(s)"
        if _verbose and issue_lines:
            detail += "\n" + "\n".join(f"  {l.strip()}" for l in issue_lines[:5])
        return _gate("bandit", False, detail)
    return _gate("bandit", True, "no HIGH severity findings")


# ---------------------------------------------------------------------------
# Gate: pip-audit dependency check
# ---------------------------------------------------------------------------


def gate_pip_audit() -> bool:
    """Check installed packages for known vulnerabilities."""
    code, output = _run_cmd(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--vulnerability-service",
            "osv",
            "--ignore-vuln",
            "CVE-2025-14009",
        ],
        timeout=120,
    )
    if code != 0:
        vuln_lines = [l for l in output.splitlines() if "PYSEC" in l or "CVE" in l or "GHSA" in l]
        detail = f"{len(vuln_lines)} vulnerability/ies detected"
        if _verbose and vuln_lines:
            detail += "\n" + "\n".join(f"  {l.strip()}" for l in vuln_lines[:5])
        return _gate("pip_audit", False, detail)
    return _gate("pip_audit", True, "no known vulnerabilities")


# ---------------------------------------------------------------------------
# Gate: Smoke test (--skip-server)
# ---------------------------------------------------------------------------


def gate_smoke_test() -> bool:
    """Run the smoke test harness (skipping server startup)."""
    code, output = _run_cmd(
        [sys.executable, "scripts/smoke_test.py", "--skip-server"],
        timeout=120,
    )
    if code != 0:
        fail_lines = [l for l in output.splitlines() if "FAIL" in l]
        detail = "; ".join(l.strip() for l in fail_lines[:3]) if fail_lines else output[-200:]
        return _gate("smoke_test", False, detail)
    return _gate("smoke_test", True, "all smoke checks passed")


# ---------------------------------------------------------------------------
# Gate: Integration smoke tests (pytest)
# ---------------------------------------------------------------------------


def gate_integration_tests() -> bool:
    """Run pytest integration smoke tests."""
    code, output = _run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_smoke.py",
            "-q",
            "--timeout=60",
            "--tb=short",
        ],
        timeout=120,
    )
    if code != 0:
        # Extract summary line
        lines = output.splitlines()
        summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
        detail = summary[-1].strip() if summary else output[-200:]
        return _gate("integration_tests", False, detail)
    return _gate("integration_tests", True, "all integration tests passed")


# ---------------------------------------------------------------------------
# Gate: OpenAPI contract sync
# ---------------------------------------------------------------------------


def gate_openapi_sync() -> bool:
    """Check that OpenAPI spec is in sync with server endpoints."""
    code, output = _run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/sdk/test_openapi_sync.py",
            "-q",
            "--timeout=60",
            "--tb=short",
        ],
        timeout=120,
    )
    if code != 0:
        lines = output.splitlines()
        summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
        detail = summary[-1].strip() if summary else "OpenAPI spec out of sync"
        return _gate("openapi_sync", False, detail)
    return _gate("openapi_sync", True, "OpenAPI spec in sync")


# ---------------------------------------------------------------------------
# Gate: SDK contract parity
# ---------------------------------------------------------------------------


def gate_contract_parity() -> bool:
    """Check Python/TypeScript SDK contract parity."""
    code, output = _run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/sdk/test_contract_parity.py",
            "-q",
            "--timeout=60",
            "--tb=short",
        ],
        timeout=120,
    )
    if code != 0:
        lines = output.splitlines()
        summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
        detail = summary[-1].strip() if summary else "contract parity check failed"
        return _gate("contract_parity", False, detail)
    return _gate("contract_parity", True, "SDK contract parity verified")


# ---------------------------------------------------------------------------
# Gate: VERSION-TAG consistency
# ---------------------------------------------------------------------------


def gate_version_tag() -> bool:
    """Check that pyproject.toml version matches the release version."""
    release_version = os.environ.get("RELEASE_VERSION", "")

    # Read pyproject.toml version
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return _gate("version_tag", False, "pyproject.toml not found")

    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        return _gate("version_tag", False, "could not parse version from pyproject.toml")

    pyproject_version = match.group(1)

    if not release_version:
        # No release version set -- just validate pyproject.toml has a valid version
        if not re.match(r"^\d+\.\d+\.\d+", pyproject_version):
            return _gate("version_tag", False, f"invalid version format: {pyproject_version}")
        return _gate(
            "version_tag",
            True,
            f"pyproject.toml version={pyproject_version} (no release tag to compare)",
        )

    if pyproject_version != release_version:
        return _gate(
            "version_tag",
            False,
            f"version mismatch: pyproject.toml={pyproject_version} release={release_version}",
        )

    return _gate("version_tag", True, f"version={pyproject_version}")


# ---------------------------------------------------------------------------
# Gate: STATUS.md validation
# ---------------------------------------------------------------------------


def gate_status_doc() -> bool:
    """Validate that docs/STATUS.md exists and is parseable."""
    status_path = PROJECT_ROOT / "docs" / "STATUS.md"

    if not status_path.exists():
        return _gate("status_doc", False, "docs/STATUS.md does not exist")

    try:
        content = status_path.read_text()
    except OSError as exc:
        return _gate("status_doc", False, f"could not read docs/STATUS.md: {exc}")

    if not content.strip():
        return _gate("status_doc", False, "docs/STATUS.md is empty")

    lines = content.splitlines()
    has_heading = any(line.startswith("# ") for line in lines)
    has_sections = sum(1 for line in lines if line.startswith("## ")) >= 1

    if not has_heading:
        return _gate("status_doc", False, "docs/STATUS.md has no top-level heading")

    if not has_sections:
        return _gate("status_doc", False, "docs/STATUS.md has no sections (## headings)")

    return _gate("status_doc", True, f"{len(lines)} lines, well-formed")


# ---------------------------------------------------------------------------
# Gate: Pentest findings
# ---------------------------------------------------------------------------


def gate_pentest_findings() -> bool:
    """Check for unresolved pentest findings."""
    script = PROJECT_ROOT / "scripts" / "check_pentest_findings.py"
    if not script.exists():
        return _gate("pentest_findings", True, "no pentest script found (skipped)")

    code, output = _run_cmd(
        [sys.executable, str(script)],
        timeout=30,
    )
    if code != 0:
        return _gate("pentest_findings", False, output[-200:])
    return _gate("pentest_findings", True, "no unresolved findings")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# Map of gate names to functions
ALL_GATES: dict[str, callable] = {
    "secrets-only": gate_secrets_scan,
    "bandit": gate_bandit,
    "pip-audit": gate_pip_audit,
    "smoke-test": gate_smoke_test,
    "integration-tests": gate_integration_tests,
    "openapi-sync": gate_openapi_sync,
    "contract-parity": gate_contract_parity,
    "version-tag": gate_version_tag,
    "status-doc": gate_status_doc,
    "pentest-findings": gate_pentest_findings,
}

# Gates grouped by category for full runs
SECURITY_GATES = ["secrets-only", "bandit", "pip-audit"]
INTEGRATION_GATES = ["smoke-test", "integration-tests", "openapi-sync", "contract-parity"]
RELEASE_GATES = ["version-tag", "status-doc", "pentest-findings"]


def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(
        description="Aragora pre-release gate checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available gates: {', '.join(ALL_GATES.keys())}",
    )
    parser.add_argument(
        "--gate",
        choices=list(ALL_GATES.keys()),
        help="Run a single gate (default: run all)",
    )
    parser.add_argument(
        "--category",
        choices=["security", "integration", "release", "all"],
        default="all",
        help="Run gates by category (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for all gates",
    )
    args = parser.parse_args()
    _verbose = args.verbose

    # Single gate mode
    if args.gate:
        _log(f"Running gate: {args.gate}")
        _log("-" * 40)
        gate_fn = ALL_GATES[args.gate]
        gate_fn()
        return 0 if all(ok for _, ok, _ in _results) else 1

    # Category or full run
    gates_to_run: list[str] = []
    if args.category == "security":
        gates_to_run = SECURITY_GATES
    elif args.category == "integration":
        gates_to_run = INTEGRATION_GATES
    elif args.category == "release":
        gates_to_run = RELEASE_GATES
    else:
        gates_to_run = SECURITY_GATES + INTEGRATION_GATES + RELEASE_GATES

    _log("=" * 60)
    _log("ARAGORA PRE-RELEASE GATE CHECK")
    _log("=" * 60)
    _log("")

    # Security gates
    if any(g in gates_to_run for g in SECURITY_GATES):
        _log("[Security Gates]")
        for gate_name in SECURITY_GATES:
            if gate_name in gates_to_run:
                ALL_GATES[gate_name]()
        _log("")

    # Integration gates
    if any(g in gates_to_run for g in INTEGRATION_GATES):
        _log("[Integration Gates]")
        for gate_name in INTEGRATION_GATES:
            if gate_name in gates_to_run:
                ALL_GATES[gate_name]()
        _log("")

    # Release gates
    if any(g in gates_to_run for g in RELEASE_GATES):
        _log("[Release Gates]")
        for gate_name in RELEASE_GATES:
            if gate_name in gates_to_run:
                ALL_GATES[gate_name]()
        _log("")

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed = total - passed

    _log("=" * 60)
    if failed == 0:
        _log(f"ALL {total} GATES PASSED")
    else:
        _log(f"{failed}/{total} GATES FAILED:")
        for name, ok, detail in _results:
            if not ok:
                _log(f"  - {name}: {detail}")
    _log("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
