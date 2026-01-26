#!/usr/bin/env python3
"""
Add operationIds to OpenAPI spec endpoints.

Generates operationIds from HTTP method + path using camelCase convention.
Example: GET /api/v1/debates/{id} -> getDebateById
"""

import argparse
import json
import re
import sys
from pathlib import Path


def path_to_camel_case(path: str) -> str:
    """Convert path segments to camelCase."""
    # Remove /api/v1/ or /api/ prefix
    path = re.sub(r"^/api(/v\d+)?/", "", path)

    # Handle path parameters like {id}, {debate_id}, etc.
    # Replace {param} with ByParam
    path = re.sub(r"\{([^}]+)\}", lambda m: "By" + m.group(1).title().replace("_", ""), path)

    # Split by / and _
    parts = re.split(r"[/_-]", path)

    # Filter empty parts and convert to title case (except first)
    result = []
    for i, part in enumerate(parts):
        if part:
            if i == 0 or result:  # Title case for all parts
                result.append(part.title())
            else:
                result.append(part)

    return "".join(result)


def generate_operation_id(method: str, path: str) -> str:
    """Generate an operationId from method and path."""
    method = method.lower()

    # Map HTTP methods to verbs
    verb_map = {
        "get": "get",
        "post": "create",
        "put": "update",
        "patch": "patch",
        "delete": "delete",
        "head": "head",
        "options": "options",
    }

    verb = verb_map.get(method, method)

    # Special cases for common patterns
    path_lower = path.lower()

    # List operations (GET on collection)
    if method == "get" and not re.search(r"\{[^}]+\}$", path):
        # If path ends with a noun (not a parameter), it's likely a list
        if not path.endswith("/health") and not path.endswith("/metrics"):
            verb = "list"

    # Get single item (GET with path parameter at end)
    if method == "get" and re.search(r"\{[^}]+\}$", path):
        verb = "get"

    # Convert path to camelCase
    path_part = path_to_camel_case(path)

    # Combine verb + path
    operation_id = verb + path_part

    # Handle edge cases
    if not path_part:
        operation_id = verb + "Root"

    # Ensure it starts with lowercase
    if operation_id:
        operation_id = operation_id[0].lower() + operation_id[1:]

    return operation_id


def add_operation_ids(spec: dict) -> tuple[dict, int, int, int]:
    """Add or dedupe operationIds for all endpoints.

    Returns: (updated_spec, added_count, existing_count, updated_count)
    """
    added = 0
    existing = 0
    updated = 0
    seen_ids = set()

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue

            base_id = details.get("operationId") or generate_operation_id(method, path)
            operation_id = base_id

            # Handle duplicates by appending a numeric suffix
            original_id = operation_id
            counter = 1
            while operation_id in seen_ids:
                operation_id = f"{original_id}{counter}"
                counter += 1

            if "operationId" in details:
                if operation_id != details["operationId"]:
                    details["operationId"] = operation_id
                    updated += 1
                else:
                    existing += 1
            else:
                details["operationId"] = operation_id
                added += 1

            seen_ids.add(operation_id)

    return spec, added, existing, updated


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add operationIds to OpenAPI spec")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/api/openapi.json"),
        help="Path to OpenAPI JSON spec",
    )
    args = parser.parse_args()

    spec_path = args.spec

    if not spec_path.exists():
        print(f"Error: {spec_path} not found")
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path, "r") as f:
        spec = json.load(f)

    print("Adding operationIds...")
    spec, added, existing, updated = add_operation_ids(spec)

    print(f"Writing {spec_path}...")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print("\nResults:")
    print(f"  - Already had operationId: {existing}")
    print(f"  - Added operationId: {added}")
    print(f"  - Updated duplicate operationId: {updated}")
    print(f"  - Total endpoints: {added + existing}")


if __name__ == "__main__":
    main()
