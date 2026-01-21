#!/usr/bin/env python3
"""
Migrate API routes from /api/ to /api/v1/.

This script updates handler ROUTES arrays and path comparisons to use
versioned API paths. The routing infrastructure already supports both
versioned and legacy paths, but this migration establishes consistency.

Usage:
    python scripts/migrate_api_versioning.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be changed without modifying files
    --verbose   Show detailed changes
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Paths that should NOT be versioned (internal/system endpoints)
EXCLUDE_PATTERNS = [
    "/healthz",
    "/readyz",
    "/metrics",
    "/.well-known/",
    "/audio/",  # Static file serving
    "/api/openapi",  # OpenAPI spec endpoint
    "/api/docs",  # Documentation
    "/api/redoc",
    "/api/postman",
]

# Paths that are already versioned
ALREADY_VERSIONED_PATTERN = re.compile(r'"/api/v\d+/')


def should_version_path(path: str) -> bool:
    """Check if a path should be versioned."""
    # Already versioned
    if re.match(r"/api/v\d+/", path):
        return False

    # Check exclusions
    for pattern in EXCLUDE_PATTERNS:
        if path.startswith(pattern) or pattern in path:
            return False

    # Must start with /api/
    return path.startswith("/api/")


def migrate_file(
    file_path: Path, dry_run: bool = False, verbose: bool = False
) -> Tuple[int, List[str]]:
    """
    Migrate a single file to use versioned API paths.

    Returns:
        Tuple of (changes_count, list_of_changes)
    """
    content = file_path.read_text()
    original_content = content
    changes: List[str] = []

    # Pattern 1: ROUTES array entries
    # Match: "/api/foo" or '/api/foo'
    route_pattern = re.compile(r'(["\'])(/api/)([^v][^"\']*)\1')

    def replace_route(match):
        quote = match.group(1)
        prefix = match.group(2)  # /api/
        rest = match.group(3)  # foo/bar

        path = prefix + rest
        if not should_version_path(path):
            return match.group(0)

        new_path = f"/api/v1/{rest}"
        changes.append(f"  Route: {path} -> {new_path}")
        return f"{quote}{new_path}{quote}"

    content = route_pattern.sub(replace_route, content)

    # Pattern 2: Path comparisons like `path == "/api/foo"`
    # Be careful not to match f-strings or complex expressions
    comparison_pattern = re.compile(r'(path\s*==\s*)(["\'])(/api/)([^v][^"\']*)\2')

    def replace_comparison(match):
        prefix_part = match.group(1)  # path ==
        quote = match.group(2)
        api_prefix = match.group(3)  # /api/
        rest = match.group(4)  # foo

        path = api_prefix + rest
        if not should_version_path(path):
            return match.group(0)

        new_path = f"/api/v1/{rest}"
        changes.append(f"  Comparison: {path} -> {new_path}")
        return f"{prefix_part}{quote}{new_path}{quote}"

    content = comparison_pattern.sub(replace_comparison, content)

    # Pattern 3: path.startswith("/api/foo")
    startswith_pattern = re.compile(
        r'(path\.startswith\s*\(\s*)(["\'])(/api/)([^v][^"\']*)\2(\s*\))'
    )

    def replace_startswith(match):
        prefix_part = match.group(1)
        quote = match.group(2)
        api_prefix = match.group(3)
        rest = match.group(4)
        suffix = match.group(5)

        path = api_prefix + rest
        if not should_version_path(path):
            return match.group(0)

        new_path = f"/api/v1/{rest}"
        changes.append(f"  startswith: {path} -> {new_path}")
        return f"{prefix_part}{quote}{new_path}{quote}{suffix}"

    content = startswith_pattern.sub(replace_startswith, content)

    # Count actual changes
    changes_count = len(changes)

    if changes_count > 0 and not dry_run:
        file_path.write_text(content)

    return changes_count, changes


def migrate_handlers(handlers_dir: Path, dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Migrate all handler files.

    Returns:
        Summary dict with counts
    """
    results = {
        "total_files": 0,
        "files_changed": 0,
        "total_changes": 0,
        "files": {},
    }

    handler_files = list(handlers_dir.glob("*.py"))
    results["total_files"] = len(handler_files)

    for file_path in sorted(handler_files):
        if file_path.name == "__init__.py":
            continue

        changes_count, changes = migrate_file(file_path, dry_run, verbose)

        if changes_count > 0:
            results["files_changed"] += 1
            results["total_changes"] += changes_count
            results["files"][file_path.name] = changes

            if verbose or dry_run:
                action = "Would update" if dry_run else "Updated"
                print(f"\n{action} {file_path.name} ({changes_count} changes):")
                for change in changes[:10]:  # Limit output
                    print(change)
                if len(changes) > 10:
                    print(f"  ... and {len(changes) - 10} more")

    return results


def update_prefix_patterns(registry_path: Path, dry_run: bool = False) -> int:
    """
    Update PREFIX_PATTERNS in handler_registry.py to include v1 variants.

    Returns:
        Number of patterns updated
    """
    content = registry_path.read_text()
    original = content

    # Find PREFIX_PATTERNS dict and add v1 variants
    # This is complex - for safety, we'll just add a note
    # The routing already handles both via strip_version_prefix

    # Count unversioned prefixes
    unversioned_count = len(re.findall(r'"/api/[^v][^"]*"', content))

    if not dry_run:
        # The routing infrastructure already handles versioned paths
        # by stripping the version prefix, so no changes needed here
        pass

    return 0  # No changes needed - routing handles this


def main():
    parser = argparse.ArgumentParser(description="Migrate API routes to versioned paths")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without modifying")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    handlers_dir = project_root / "aragora" / "server" / "handlers"

    if not handlers_dir.exists():
        print(f"Error: Handlers directory not found: {handlers_dir}")
        sys.exit(1)

    print("=" * 60)
    print("API Versioning Migration")
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN - No files will be modified\n")

    # Migrate handlers
    print(f"\nMigrating handlers in {handlers_dir}...")
    results = migrate_handlers(handlers_dir, args.dry_run, args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total files scanned: {results['total_files']}")
    print(f"Files changed: {results['files_changed']}")
    print(f"Total route changes: {results['total_changes']}")

    if results["files_changed"] > 0:
        print("\nModified files:")
        for fname in results["files"]:
            print(f"  - {fname}")

    # Verification command
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    print("Run these commands to verify the migration:")
    print(f"  grep -r '\"/api/[^v]' {handlers_dir} | wc -l  # Should decrease")
    print(f"  grep -r '\"/api/v1/' {handlers_dir} | wc -l  # Should increase")
    print("  pytest tests/server/ -v -k 'api'  # Run API tests")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")

    return 0 if results["files_changed"] == 0 or not args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
