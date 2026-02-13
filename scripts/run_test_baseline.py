#!/usr/bin/env python3
"""Deterministic baseline test command used by local and CI workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_MARKERS = (
    "not slow and not load and not e2e and not integration "
    "and not integration_minimal and not benchmark and not performance"
)
DEFAULT_PATHS = ["tests/"]


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic baseline pytest command")
    parser.add_argument(
        "--no-clean-check",
        action="store_true",
        help="Skip repo hygiene working-tree check",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute tests (default is --collect-only)",
    )
    parser.add_argument(
        "--markers",
        default=DEFAULT_MARKERS,
        help="Pytest marker expression for baseline run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-test timeout in seconds",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=1,
        help="Stop after this many failures",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbose pytest output",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Test paths (default: tests/)",
    )
    args = parser.parse_args()

    if not args.no_clean_check:
        _run([sys.executable, "scripts/guard_repo_clean.py", "--check-working-tree"])

    _run([sys.executable, "scripts/check_test_dependencies.py"])

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        *args.paths,
        "-m",
        args.markers,
        "--timeout",
        str(args.timeout),
        "--tb=short",
        "--maxfail",
        str(args.maxfail),
    ]
    if not args.run:
        pytest_cmd.append("--collect-only")
    if not args.verbose:
        pytest_cmd.append("-q")

    _run(pytest_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
