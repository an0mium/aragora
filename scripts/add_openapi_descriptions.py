#!/usr/bin/env python3
"""
Add descriptions to OpenAPI operations that lack them.

Uses summary or operationId as a fallback to ensure descriptions exist.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _humanize_operation_id(operation_id: str) -> str:
    """Convert an operationId into a human-friendly sentence."""
    if not operation_id:
        return ""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", operation_id)
    spaced = spaced.replace("_", " ").replace("-", " ").strip()
    if not spaced:
        return ""
    return spaced[0].upper() + spaced[1:]


def add_descriptions(spec: dict) -> tuple[dict, int, int]:
    """Add descriptions to operations missing them.

    Returns: (updated_spec, added_count, existing_count)
    """
    added = 0
    existing = 0

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue

            description = details.get("description", "")
            if isinstance(description, str) and description.strip():
                existing += 1
                continue

            summary = details.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                details["description"] = summary.strip()
                added += 1
                continue

            operation_id = details.get("operationId", "")
            humanized = _humanize_operation_id(operation_id)
            if humanized:
                details["description"] = f"{humanized}."
            else:
                details["description"] = f"{method.upper()} {path}"
            added += 1

    return spec, added, existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Add operation descriptions to OpenAPI spec")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/api/openapi.json"),
        help="Path to OpenAPI JSON spec",
    )
    args = parser.parse_args()

    spec_path = args.spec
    if not spec_path.exists():
        print(f"Error: {spec_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path, "r") as f:
        spec = json.load(f)

    print("Adding descriptions...")
    spec, added, existing = add_descriptions(spec)

    print(f"Writing {spec_path}...")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print("\nResults:")
    print(f"  - Already had description: {existing}")
    print(f"  - Added description: {added}")
    print(f"  - Total operations: {added + existing}")


if __name__ == "__main__":
    main()
