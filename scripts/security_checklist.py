#!/usr/bin/env python3
"""
Security Checklist Validator.

Validates that security best practices are followed in the codebase.
Designed to run as a pre-commit hook or CI check.

Usage:
    # Run all checks
    python scripts/security_checklist.py

    # CI mode (exit code 1 on failures)
    python scripts/security_checklist.py --ci

    # Check specific categories
    python scripts/security_checklist.py --categories auth,secrets
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

# Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class CheckResult:
    """Result of a security check."""

    name: str
    passed: bool
    message: str
    category: str
    severity: str = "high"  # critical, high, medium, low


def check_env_gitignore() -> CheckResult:
    """Check that .env is in .gitignore."""
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return CheckResult(
            name=".env in .gitignore",
            passed=False,
            message=".gitignore file not found",
            category="secrets",
            severity="critical",
        )

    content = gitignore.read_text()
    patterns = [".env", "*.env", ".env.*"]
    found = any(p in content for p in patterns)

    return CheckResult(
        name=".env in .gitignore",
        passed=found,
        message=".env is properly gitignored"
        if found
        else ".env not found in .gitignore - credentials may be committed!",
        category="secrets",
        severity="critical",
    )


def check_no_hardcoded_secrets() -> CheckResult:
    """Check for hardcoded API keys in Python files."""
    secret_patterns = [
        r'(?:api[_-]?key|apikey)\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
        r"sk-ant-[a-zA-Z0-9-]{20,}",  # Anthropic keys
        r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
    ]

    violations = []
    for py_file in Path("aragora").rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Skip test files and examples
                    if "test" not in str(py_file) and "example" not in str(py_file):
                        violations.append(str(py_file))
                        break
        except Exception:
            pass

    return CheckResult(
        name="No hardcoded secrets",
        passed=len(violations) == 0,
        message=f"Found {len(violations)} files with potential hardcoded secrets: {violations[:3]}"
        if violations
        else "No hardcoded secrets found",
        category="secrets",
        severity="critical",
    )


def check_sql_parameterization() -> CheckResult:
    """Check for SQL string formatting (potential injection)."""
    unsafe_patterns = [
        r"execute\([^)]*%[^)]*\)",  # execute with % formatting
        r"execute\([^)]*\.format\([^)]*\)",  # execute with .format()
        r'execute\([^)]*f["\'][^"\']*{[^}]+}',  # execute with f-strings
    ]

    violations = []
    for py_file in Path("aragora").rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern in unsafe_patterns:
                if re.search(pattern, content):
                    violations.append(str(py_file))
                    break
        except Exception:
            pass

    return CheckResult(
        name="SQL parameterization",
        passed=len(violations) == 0,
        message=f"Found {len(violations)} files with potential SQL injection: {violations[:3]}"
        if violations
        else "SQL queries appear to use parameterization",
        category="injection",
        severity="critical",
    )


def check_jwt_secret_length() -> CheckResult:
    """Check JWT secret configuration."""
    min_length = 32

    # Check environment variable
    jwt_secret = os.environ.get("ARAGORA_JWT_SECRET", "")
    if not jwt_secret:
        return CheckResult(
            name="JWT secret configured",
            passed=False,
            message="ARAGORA_JWT_SECRET not set in environment",
            category="auth",
            severity="high",
        )

    if len(jwt_secret) < min_length:
        return CheckResult(
            name="JWT secret strength",
            passed=False,
            message=f"ARAGORA_JWT_SECRET is {len(jwt_secret)} chars, minimum is {min_length}",
            category="auth",
            severity="critical",
        )

    return CheckResult(
        name="JWT secret configured",
        passed=True,
        message="JWT secret is properly configured",
        category="auth",
    )


def check_cors_not_wildcard() -> CheckResult:
    """Check CORS configuration."""
    allowed_origins = os.environ.get("ARAGORA_ALLOWED_ORIGINS", "")
    env = os.environ.get("ARAGORA_ENV", "development")

    if env == "production":
        if allowed_origins == "*" or not allowed_origins:
            return CheckResult(
                name="CORS configuration",
                passed=False,
                message="CORS allows all origins (*) in production - set ARAGORA_ALLOWED_ORIGINS",
                category="config",
                severity="high",
            )

    return CheckResult(
        name="CORS configuration",
        passed=True,
        message="CORS is properly configured",
        category="config",
    )


def check_debug_disabled() -> CheckResult:
    """Check debug mode is disabled in production."""
    env = os.environ.get("ARAGORA_ENV", "development")
    debug = os.environ.get("ARAGORA_DEBUG", "").lower()

    if env == "production" and debug in ("true", "1", "yes"):
        return CheckResult(
            name="Debug disabled in production",
            passed=False,
            message="ARAGORA_DEBUG is enabled in production!",
            category="config",
            severity="high",
        )

    return CheckResult(
        name="Debug mode",
        passed=True,
        message="Debug mode is appropriately configured",
        category="config",
    )


def check_rate_limiting() -> CheckResult:
    """Check rate limiting configuration."""
    backend = os.environ.get("ARAGORA_RATE_LIMIT_BACKEND", "memory")
    multi_instance = os.environ.get("ARAGORA_MULTI_INSTANCE", "").lower() in ("true", "1")

    if multi_instance and backend == "memory":
        return CheckResult(
            name="Distributed rate limiting",
            passed=False,
            message="Using in-memory rate limiting with multi-instance deployment - use Redis",
            category="config",
            severity="high",
        )

    return CheckResult(
        name="Rate limiting",
        passed=True,
        message=f"Rate limiting backend: {backend}",
        category="config",
    )


def check_no_pickle_loads() -> CheckResult:
    """Check for unsafe pickle usage."""
    unsafe_patterns = [
        r"pickle\.loads?\(",
        r"cPickle\.loads?\(",
        r"_pickle\.loads?\(",
    ]

    violations = []
    for py_file in Path("aragora").rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern in unsafe_patterns:
                if re.search(pattern, content):
                    violations.append(str(py_file))
                    break
        except Exception:
            pass

    return CheckResult(
        name="No unsafe pickle",
        passed=len(violations) == 0,
        message=f"Found {len(violations)} files with pickle.load: {violations[:3]}"
        if violations
        else "No unsafe pickle usage found",
        category="injection",
        severity="high",
    )


def check_no_eval_exec() -> CheckResult:
    """Check for eval/exec usage."""
    unsafe_patterns = [
        r"\beval\s*\(",
        r"\bexec\s*\(",
    ]

    violations = []
    for py_file in Path("aragora").rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern in unsafe_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Allow in specific safe contexts (e.g., test files)
                    if "test" not in str(py_file):
                        violations.append(str(py_file))
                        break
        except Exception:
            pass

    return CheckResult(
        name="No eval/exec",
        passed=len(violations) == 0,
        message=f"Found {len(violations)} files with eval/exec: {violations[:3]}"
        if violations
        else "No unsafe eval/exec found",
        category="injection",
        severity="high",
    )


# All checks
CHECKS: list[Callable[[], CheckResult]] = [
    check_env_gitignore,
    check_no_hardcoded_secrets,
    check_sql_parameterization,
    check_jwt_secret_length,
    check_cors_not_wildcard,
    check_debug_disabled,
    check_rate_limiting,
    check_no_pickle_loads,
    check_no_eval_exec,
]


def main() -> int:
    """Run security checklist."""
    parser = argparse.ArgumentParser(description="Security checklist validator")
    parser.add_argument("--ci", action="store_true", help="CI mode (exit 1 on failure)")
    parser.add_argument("--categories", help="Comma-separated categories to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else None

    print(f"{BOLD}Security Checklist{RESET}")
    print("=" * 50)

    results: list[CheckResult] = []
    for check_fn in CHECKS:
        try:
            result = check_fn()
            if categories and result.category not in categories:
                continue
            results.append(result)
        except Exception as e:
            results.append(
                CheckResult(
                    name=check_fn.__name__,
                    passed=False,
                    message=f"Check failed with error: {e}",
                    category="error",
                    severity="medium",
                )
            )

    # Print results
    passed = 0
    failed = 0
    for r in results:
        if r.passed:
            status = f"{GREEN}PASS{RESET}"
            passed += 1
        else:
            status = f"{RED}FAIL{RESET}"
            failed += 1

        print(f"[{status}] {r.name}")
        if not r.passed or args.verbose:
            print(f"       {r.message}")

    print()
    print(f"Results: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")

    if failed > 0 and args.ci:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
