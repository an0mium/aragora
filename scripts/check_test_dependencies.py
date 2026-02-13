#!/usr/bin/env python3
"""Verify pinned test dependencies and pytest plugin availability."""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from importlib import metadata


REQUIRED_DISTS: dict[str, str] = {
    "pytest": "7.0",
    "pytest-rerunfailures": "14.0",
    "PyJWT": "2.8",
}


def _normalize_version(raw: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", raw)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts[:4])


def _version_at_least(current: str, minimum: str) -> bool:
    return _normalize_version(current) >= _normalize_version(minimum)


def _check_distribution(name: str, minimum: str) -> tuple[bool, str]:
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        return False, f"{name} not installed (required >= {minimum})"
    if not _version_at_least(installed, minimum):
        return False, f"{name} version {installed} < required {minimum}"
    return True, f"{name} {installed} (ok)"


def _check_module_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return False, f"Import failed for {module_name}: {exc}"
    return True, f"{module_name} importable"


def _check_pytest_reruns_flag() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, "pytest --help failed"
    if "--reruns" not in proc.stdout:
        return False, "pytest-rerunfailures plugin not active (--reruns flag missing)"
    return True, "pytest --reruns flag available"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify required test dependencies")
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()

    checks: list[tuple[bool, str]] = []
    for name, minimum in REQUIRED_DISTS.items():
        checks.append(_check_distribution(name, minimum))

    checks.append(_check_module_import("jwt"))
    checks.append(_check_module_import("pytest_rerunfailures"))
    checks.append(_check_pytest_reruns_flag())

    failures = [msg for ok, msg in checks if not ok]

    if not args.quiet:
        print("Test Dependency Verification")
        print("=" * 40)
        for ok, msg in checks:
            prefix = "OK" if ok else "FAIL"
            print(f"[{prefix}] {msg}")

    if failures:
        print("\nDependency verification failed.")
        for msg in failures:
            print(f"- {msg}")
        return 1

    if not args.quiet:
        print("\nAll required test dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
