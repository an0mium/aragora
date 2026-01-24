#!/usr/bin/env python3
"""
Check version alignment across all package manifests.

This script validates that all version sources in the repository are aligned.
It fails with exit code 1 if any version mismatch is detected.

Usage:
    python scripts/check_version_alignment.py
    python scripts/check_version_alignment.py --fix  # Auto-fix mismatches
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def get_canonical_version() -> str:
    """Get the canonical version from aragora/__version__.py."""
    version_file = Path("aragora/__version__.py")
    if not version_file.exists():
        raise FileNotFoundError("aragora/__version__.py not found")

    content = version_file.read_text()

    # Extract version components
    major = re.search(r"VERSION_MAJOR\s*=\s*(\d+)", content)
    minor = re.search(r"VERSION_MINOR\s*=\s*(\d+)", content)
    patch = re.search(r"VERSION_PATCH\s*=\s*(\d+)", content)

    if not all([major, minor, patch]):
        raise ValueError("Could not parse version from aragora/__version__.py")

    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def get_pyproject_version(path: Path) -> str | None:
    """Extract version from a pyproject.toml file."""
    if not path.exists():
        return None

    content = path.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def get_package_json_version(path: Path) -> str | None:
    """Extract version from a package.json file."""
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return data.get("version")
    except json.JSONDecodeError:
        return None


def fix_pyproject_version(path: Path, new_version: str) -> bool:
    """Update version in a pyproject.toml file."""
    if not path.exists():
        return False

    content = path.read_text()
    new_content = re.sub(
        r'^(version\s*=\s*["\'])([^"\']+)(["\'])',
        rf"\g<1>{new_version}\g<3>",
        content,
        flags=re.MULTILINE,
    )

    if new_content != content:
        path.write_text(new_content)
        return True
    return False


def fix_package_json_version(path: Path, new_version: str) -> bool:
    """Update version in a package.json file."""
    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text())
        if data.get("version") != new_version:
            data["version"] = new_version
            path.write_text(json.dumps(data, indent=2) + "\n")
            return True
    except json.JSONDecodeError:
        pass
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check version alignment across packages")
    parser.add_argument("--fix", action="store_true", help="Auto-fix version mismatches")
    args = parser.parse_args()

    # Get canonical version
    try:
        canonical = get_canonical_version()
        print(f"Canonical version (aragora/__version__.py): {canonical}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Define all version sources
    version_sources: list[tuple[str, Path, str]] = [
        ("pyproject.toml", Path("pyproject.toml"), "pyproject"),
        ("aragora-py/pyproject.toml", Path("aragora-py/pyproject.toml"), "pyproject"),
        ("aragora-sdk/pyproject.toml", Path("aragora-sdk/pyproject.toml"), "pyproject"),
        ("sdk/python/pyproject.toml", Path("sdk/python/pyproject.toml"), "pyproject"),
        ("aragora-js/package.json", Path("aragora-js/package.json"), "package"),
        ("sdk/typescript/package.json", Path("sdk/typescript/package.json"), "package"),
    ]

    mismatches: list[tuple[str, str | None]] = []
    fixed: list[str] = []

    print("\nChecking version alignment:")
    print("-" * 50)

    for name, path, file_type in version_sources:
        if file_type == "pyproject":
            version = get_pyproject_version(path)
        else:
            version = get_package_json_version(path)

        if version is None:
            print(f"  {name}: (not found)")
            continue

        status = "OK" if version == canonical else "MISMATCH"
        print(f"  {name}: {version} [{status}]")

        if version != canonical:
            mismatches.append((name, version))

            if args.fix:
                if file_type == "pyproject":
                    if fix_pyproject_version(path, canonical):
                        fixed.append(name)
                else:
                    if fix_package_json_version(path, canonical):
                        fixed.append(name)

    print("-" * 50)

    if fixed:
        print(f"\nFixed {len(fixed)} file(s):")
        for name in fixed:
            print(f"  - {name} -> {canonical}")

    if mismatches and not args.fix:
        print(f"\nERROR: {len(mismatches)} version mismatch(es) found!")
        print("Run with --fix to auto-fix, or manually update the files.")
        return 1

    if mismatches and fixed:
        remaining = len(mismatches) - len(fixed)
        if remaining > 0:
            print(f"\nWARNING: {remaining} mismatch(es) could not be fixed.")
            return 1

    print("\nAll versions aligned!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
