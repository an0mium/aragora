#!/usr/bin/env python3
"""
Add operationIds to endpoint definition files that don't have them.

This script directly modifies the source files in aragora/server/openapi/endpoints/
to add operationId fields to each endpoint operation.

Usage:
    python scripts/add_operation_ids_to_sources.py
    python scripts/add_operation_ids_to_sources.py --dry-run
"""

import argparse
import re
from pathlib import Path


def path_to_operation_id(method: str, path: str) -> str:
    """Convert HTTP method + path to camelCase operationId."""
    # Remove /api/ prefix and version prefix
    clean_path = re.sub(r"^/api/(v\d+/)?", "", path)

    # Split into segments
    segments = [s for s in clean_path.split("/") if s and not s.startswith("{")]

    # Handle path parameters
    has_param = "{" in path

    # Map HTTP methods to verbs
    method_prefix = {
        "get": "get" if has_param else "list",
        "post": "create",
        "put": "update",
        "patch": "patch",
        "delete": "delete",
    }

    prefix = method_prefix.get(method.lower(), method.lower())

    # Build operation name from segments
    if segments:
        last = segments[-1]
        if has_param and last.endswith("s") and len(last) > 2:
            last = last[:-1]

        parts = []
        for i, seg in enumerate(segments):
            if i == len(segments) - 1:
                continue
            parts.append(seg.title().replace("-", "").replace("_", ""))

        suffix = last.title().replace("-", "").replace("_", "")
        parts.append(suffix)

        return prefix + "".join(parts)

    return prefix


def add_operation_ids_to_file(filepath: Path, dry_run: bool = False) -> tuple[int, int]:
    """Add operationIds to a single endpoint file.

    Returns:
        Tuple of (operations_updated, operations_skipped)
    """
    content = filepath.read_text()

    # Track statistics
    updated = 0
    skipped = 0

    lines = content.split("\n")
    new_lines = []
    current_path = None
    current_method = None
    in_method_block = False
    has_operation_id = False

    for line in lines:
        # Check for path definition
        path_match = re.match(r'\s*"(/api[^"]+)":\s*\{', line)
        if path_match:
            current_path = path_match.group(1)

        # Check for method definition
        method_match = re.match(r'\s*"(get|post|put|patch|delete)":\s*\{', line, re.IGNORECASE)
        if method_match:
            current_method = method_match.group(1)
            in_method_block = True
            has_operation_id = False

        # Check if this block already has operationId
        if in_method_block and '"operationId"' in line:
            has_operation_id = True
            skipped += 1

        # Look for summary line to insert operationId after it
        if in_method_block and not has_operation_id and '"summary"' in line and current_path:
            # Insert operationId after this line
            new_lines.append(line)

            # Generate operationId
            op_id = path_to_operation_id(current_method, current_path)

            # Get indentation
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent

            # Add operationId line
            op_line = f'{indent_str}"operationId": "{op_id}",'
            new_lines.append(op_line)

            has_operation_id = True
            updated += 1
            continue

        # Check for end of method block
        if in_method_block and re.match(r"\s*\}", line):
            in_method_block = False

        new_lines.append(line)

    new_content = "\n".join(new_lines)

    if not dry_run and updated > 0:
        filepath.write_text(new_content)

    return updated, skipped


def main():
    parser = argparse.ArgumentParser(description="Add operationIds to endpoint sources")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    endpoints_dir = Path("aragora/server/openapi/endpoints")
    if not endpoints_dir.exists():
        print(f"Error: {endpoints_dir} not found")
        return 1

    # Files that already have complete operationId coverage - skip them
    skip_files = {
        "agents.py",
        "budgets.py",
        "consensus.py",
        "debates.py",
        "integrations.py",
        "system.py",
        "teams.py",
        "workspace.py",
    }

    total_updated = 0
    total_skipped = 0

    for filepath in sorted(endpoints_dir.glob("*.py")):
        if filepath.name == "__init__.py":
            continue

        # Skip files that already have operationIds
        if filepath.name in skip_files:
            print(f"  {filepath.name}: skipped (already has operationIds)")
            continue

        updated, skipped = add_operation_ids_to_file(filepath, args.dry_run)

        if updated > 0:
            status = "(dry run)" if args.dry_run else ""
            print(f"  {filepath.name}: +{updated} operationIds {status}")

        total_updated += updated
        total_skipped += skipped

    print("\nSummary:")
    print(f"  Added: {total_updated}")
    print(f"  Already present: {total_skipped}")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

    return 0


if __name__ == "__main__":
    exit(main())
