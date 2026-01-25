#!/usr/bin/env python3
"""
Integration Test Baseline Runner for Aragora.

Provides tiered test execution for CI/CD pipelines and local development.

Test Tiers:
-----------
1. SMOKE (< 2 min)
   - Critical path tests only
   - No external dependencies
   - Run on every commit

2. OFFLINE_INTEGRATION (< 10 min)
   - Integration tests with mocked dependencies
   - Uses `integration_minimal` marker
   - Run on every PR

3. FULL_INTEGRATION (< 30 min)
   - All integration tests
   - May require external services
   - Run nightly

4. E2E (< 60 min)
   - End-to-end tests requiring running server
   - Full system tests
   - Run before release

Usage:
------
    # Run smoke tests (fastest)
    python scripts/run_integration_baseline.py smoke

    # Run offline integration (PR-safe)
    python scripts/run_integration_baseline.py offline

    # Run all integration tests
    python scripts/run_integration_baseline.py full

    # Run with coverage
    python scripts/run_integration_baseline.py offline --coverage

    # Dry run (show what would run)
    python scripts/run_integration_baseline.py offline --dry-run

Environment Variables:
----------------------
    ARAGORA_BASELINE_PARALLEL: Number of parallel workers (default: auto)
    ARAGORA_BASELINE_TIMEOUT: Test timeout in seconds (default: 60)
    DATABASE_URL: PostgreSQL URL for full integration tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestTier:
    """Configuration for a test tier."""

    name: str
    description: str
    pytest_args: List[str]
    timeout_minutes: int
    parallel: bool = True
    requires_server: bool = False
    requires_database: bool = False


# =============================================================================
# Test Tier Definitions
# =============================================================================

TIERS = {
    "smoke": TestTier(
        name="smoke",
        description="Quick sanity tests for critical paths",
        pytest_args=[
            "-m",
            "smoke or (unit and not slow)",
            "--tb=short",
            "-q",
        ],
        timeout_minutes=2,
        parallel=True,
    ),
    "offline": TestTier(
        name="offline",
        description="Integration tests with mocked dependencies (PR-safe)",
        pytest_args=[
            "-m",
            "integration_minimal or (integration and not external)",
            "--tb=short",
            "tests/integration/",
        ],
        timeout_minutes=10,
        parallel=True,
    ),
    "knowledge": TestTier(
        name="knowledge",
        description="KnowledgeMound and CDC integration tests",
        pytest_args=[
            "-m",
            "knowledge or integration_minimal",
            "--tb=short",
            "tests/knowledge/",
            "tests/integration/test_cdc*.py",
        ],
        timeout_minutes=15,
        parallel=True,
    ),
    "full": TestTier(
        name="full",
        description="All integration tests (may require external services)",
        pytest_args=[
            "-m",
            "integration",
            "--tb=short",
            "tests/integration/",
        ],
        timeout_minutes=30,
        parallel=True,
        requires_database=True,
    ),
    "e2e": TestTier(
        name="e2e",
        description="End-to-end tests requiring running server",
        pytest_args=[
            "-m",
            "e2e",
            "--tb=short",
            "tests/e2e/",
        ],
        timeout_minutes=60,
        parallel=False,  # E2E tests often share state
        requires_server=True,
    ),
    "debate": TestTier(
        name="debate",
        description="Debate orchestration and flow tests",
        pytest_args=[
            "--tb=short",
            "tests/debate/",
            "tests/integration/test_debate*.py",
        ],
        timeout_minutes=20,
        parallel=True,
    ),
    "handlers": TestTier(
        name="handlers",
        description="HTTP handler tests",
        pytest_args=[
            "--tb=short",
            "tests/handlers/",
        ],
        timeout_minutes=15,
        parallel=True,
    ),
}


def get_pytest_command(
    tier: TestTier,
    coverage: bool = False,
    parallel_workers: Optional[int] = None,
    timeout: Optional[int] = None,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build pytest command for the given tier."""
    cmd = ["python", "-m", "pytest"]

    # Add tier-specific args
    cmd.extend(tier.pytest_args)

    # Parallelism
    if tier.parallel and parallel_workers != 0:
        workers = parallel_workers or os.environ.get("ARAGORA_BASELINE_PARALLEL", "auto")
        cmd.extend(["-n", str(workers)])
    elif not tier.parallel:
        cmd.extend(["-n", "0"])  # Explicitly disable parallel

    # Timeout
    test_timeout = timeout or int(os.environ.get("ARAGORA_BASELINE_TIMEOUT", "60"))
    cmd.extend(["--timeout", str(test_timeout)])

    # Coverage
    if coverage:
        cmd.extend(
            [
                "--cov=aragora",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-fail-under=0",  # Don't fail on coverage threshold
            ]
        )

    # Verbosity
    if verbose:
        cmd.append("-v")

    # Extra args
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def check_prerequisites(tier: TestTier) -> List[str]:
    """Check if tier prerequisites are met. Returns list of warnings."""
    warnings = []

    if tier.requires_database:
        if not os.environ.get("DATABASE_URL"):
            warnings.append(
                "DATABASE_URL not set. PostgreSQL tests will be skipped. "
                "Set DATABASE_URL=postgresql://user:pass@host/db for full coverage."
            )

    if tier.requires_server:
        # Check if server is running
        try:
            import httpx

            response = httpx.get("http://localhost:8080/health", timeout=2.0)
            if response.status_code != 200:
                warnings.append(
                    "Aragora server not responding. Start with: "
                    "python -m aragora.server.unified_server --port 8080"
                )
        except Exception:
            warnings.append(
                "Aragora server not running. Start with: "
                "python -m aragora.server.unified_server --port 8080"
            )

    return warnings


def print_tier_info(tier: TestTier):
    """Print information about the test tier."""
    print(f"\n{'=' * 60}")
    print(f"Test Tier: {tier.name.upper()}")
    print(f"{'=' * 60}")
    print(f"Description: {tier.description}")
    print(f"Timeout: {tier.timeout_minutes} minutes")
    print(f"Parallel: {'Yes' if tier.parallel else 'No'}")
    if tier.requires_database:
        print("Requires: Database (DATABASE_URL)")
    if tier.requires_server:
        print("Requires: Running server")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run Aragora integration test baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s smoke              # Quick sanity tests
  %(prog)s offline            # PR-safe integration tests
  %(prog)s full --coverage    # Full integration with coverage
  %(prog)s --list             # List available tiers
        """,
    )

    parser.add_argument(
        "tier",
        nargs="?",
        choices=list(TIERS.keys()),
        help="Test tier to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test tiers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show command without running",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel workers (0 to disable)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Test timeout in seconds",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional pytest arguments",
    )

    args = parser.parse_args()

    # List tiers
    if args.list:
        print("\nAvailable Test Tiers:")
        print("-" * 60)
        for name, tier in TIERS.items():
            print(f"\n  {name:12} - {tier.description}")
            print(
                f"               Timeout: {tier.timeout_minutes} min, "
                f"Parallel: {'Yes' if tier.parallel else 'No'}"
            )
        print()
        return 0

    # Require tier if not listing
    if not args.tier:
        parser.print_help()
        return 1

    tier = TIERS[args.tier]
    print_tier_info(tier)

    # Check prerequisites
    warnings = check_prerequisites(tier)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if warnings:
        print()

    # Build command
    cmd = get_pytest_command(
        tier,
        coverage=args.coverage,
        parallel_workers=args.parallel,
        timeout=args.timeout,
        verbose=args.verbose,
        extra_args=args.extra_args,
    )

    print(f"Command: {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("(Dry run - not executing)")
        return 0

    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
