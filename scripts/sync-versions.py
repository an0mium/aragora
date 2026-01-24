#!/usr/bin/env python3
"""
Version Sync Script

Synchronizes version numbers across all packages in the monorepo.
Source of truth: pyproject.toml in the root directory.

Usage:
    python scripts/sync-versions.py          # Dry run (show what would change)
    python scripts/sync-versions.py --apply  # Apply changes
    python scripts/sync-versions.py --set 2.3.0 --apply  # Set specific version
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Packages to sync (relative to root)
PACKAGES = {
    "pyproject.toml": "toml",  # Source of truth
    "sdk/typescript/package.json": "json",
    "aragora-py/pyproject.toml": "toml",
    "aragora-js/package.json": "json",
    "aragora/live/package.json": "json",
}

# Packages to exclude from sync (independent versioning)
EXCLUDED = {
    "docs-site/package.json",  # Internal tooling, independent version
}


def read_version_from_toml(path: Path) -> str:
    """Extract version from pyproject.toml."""
    content = path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {path}")
    return match.group(1)


def read_version_from_json(path: Path) -> str:
    """Extract version from package.json."""
    data = json.loads(path.read_text())
    return data.get("version", "0.0.0")


def update_version_in_toml(path: Path, new_version: str) -> str:
    """Update version in pyproject.toml, return new content."""
    content = path.read_text()
    return re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )


def update_version_in_json(path: Path, new_version: str) -> str:
    """Update version in package.json, return new content."""
    data = json.loads(path.read_text())
    data["version"] = new_version
    return json.dumps(data, indent=2) + "\n"


def get_current_versions() -> dict[str, str]:
    """Get current versions of all packages."""
    versions = {}
    for pkg_path, pkg_type in PACKAGES.items():
        full_path = ROOT / pkg_path
        if not full_path.exists():
            versions[pkg_path] = "NOT FOUND"
            continue

        if pkg_type == "toml":
            versions[pkg_path] = read_version_from_toml(full_path)
        else:
            versions[pkg_path] = read_version_from_json(full_path)

    return versions


def sync_versions(target_version: str, apply: bool = False) -> list[tuple[str, str, str]]:
    """
    Sync all packages to target version.

    Returns list of (package, old_version, new_version) tuples.
    """
    changes = []

    for pkg_path, pkg_type in PACKAGES.items():
        if pkg_path == "pyproject.toml":
            continue  # Skip source of truth

        full_path = ROOT / pkg_path
        if not full_path.exists():
            print(f"  SKIP {pkg_path} (not found)")
            continue

        if pkg_type == "toml":
            old_version = read_version_from_toml(full_path)
            new_content = update_version_in_toml(full_path, target_version)
        else:
            old_version = read_version_from_json(full_path)
            new_content = update_version_in_json(full_path, target_version)

        if old_version == target_version:
            print(f"  OK   {pkg_path} ({old_version})")
            continue

        changes.append((pkg_path, old_version, target_version))

        if apply:
            full_path.write_text(new_content)
            print(f"  UPDATED {pkg_path}: {old_version} -> {target_version}")
        else:
            print(f"  WOULD UPDATE {pkg_path}: {old_version} -> {target_version}")

    return changes


def main():
    parser = argparse.ArgumentParser(description="Sync version numbers across packages")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument(
        "--set", metavar="VERSION", help="Set specific version (default: read from pyproject.toml)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if all versions match (exit 1 if not)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Aragora Version Sync")
    print("=" * 60)

    # Get source version
    source_path = ROOT / "pyproject.toml"
    if not source_path.exists():
        print(f"ERROR: Source of truth not found: {source_path}")
        sys.exit(1)

    source_version = read_version_from_toml(source_path)
    target_version = args.set if args.set else source_version

    if args.set and args.apply:
        # Also update the source of truth
        new_content = update_version_in_toml(source_path, target_version)
        source_path.write_text(new_content)
        print(f"\nSource of truth: pyproject.toml -> {target_version}")
    else:
        print(f"\nSource of truth: pyproject.toml = {source_version}")

    if args.set:
        print(f"Target version: {target_version}")

    # Check mode
    if args.check:
        print("\nChecking version alignment...")
        versions = get_current_versions()
        all_match = all(v == source_version for v in versions.values() if v != "NOT FOUND")

        for pkg, ver in versions.items():
            status = "OK" if ver == source_version else "MISMATCH"
            print(f"  {status:8} {pkg}: {ver}")

        if all_match:
            print("\nAll versions match!")
            sys.exit(0)
        else:
            print("\nVersions out of sync! Run: python scripts/sync-versions.py --apply")
            sys.exit(1)

    # Sync mode
    print("\nSyncing packages...")
    changes = sync_versions(target_version, apply=args.apply)

    print()
    if not args.apply and changes:
        print(f"Dry run complete. {len(changes)} package(s) would be updated.")
        print("Run with --apply to make changes.")
    elif args.apply and changes:
        print(f"Done! Updated {len(changes)} package(s) to {target_version}")
    else:
        print("All packages already in sync!")


if __name__ == "__main__":
    main()
