#!/usr/bin/env python3
"""
Add request body schemas to OpenAPI operations that lack them.

Introspects handler source files to infer request body fields from
validate_body decorators, request_body dicts, and json body reads.
For endpoints where no schema can be inferred, generates a minimal
placeholder schema with "additionalProperties: true".

Usage:
    python scripts/add_openapi_request_schemas.py
    python scripts/add_openapi_request_schemas.py --spec docs/api/openapi.json --dry-run
    python scripts/add_openapi_request_schemas.py --verbose
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path


# Known request body schemas for common endpoint patterns
KNOWN_SCHEMAS: dict[str, dict] = {
    "POST /api/debates": {
        "type": "object",
        "required": ["question"],
        "properties": {
            "question": {"type": "string", "description": "The question or topic to debate"},
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agent identifiers to include in the debate",
            },
            "rounds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 3,
                "description": "Number of debate rounds",
            },
            "consensus_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.7,
                "description": "Threshold for consensus detection",
            },
            "protocol": {
                "type": "string",
                "enum": ["structured", "free", "adversarial", "collaborative"],
                "default": "structured",
                "description": "Debate protocol to use",
            },
            "context": {"type": "string", "description": "Additional context for the debate"},
            "workspace_id": {"type": "string", "description": "Workspace ID"},
        },
    },
    "POST /api/knowledge/entries": {
        "type": "object",
        "required": ["content"],
        "properties": {
            "content": {"type": "string", "description": "Knowledge entry content"},
            "title": {"type": "string", "description": "Entry title"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization",
            },
            "visibility": {
                "type": "string",
                "enum": ["private", "workspace", "organization", "public"],
                "default": "workspace",
            },
        },
    },
    "POST /api/workflows": {
        "type": "object",
        "required": ["name", "steps"],
        "properties": {
            "name": {"type": "string", "description": "Workflow name"},
            "description": {"type": "string", "description": "Workflow description"},
            "steps": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Workflow step definitions",
            },
            "trigger": {"type": "object", "description": "Trigger configuration"},
        },
    },
}

# HTTP methods that can have request bodies
BODY_METHODS = {"post", "put", "patch"}


def _infer_schema_from_handler(handler_dir: Path, operation_id: str) -> dict | None:
    """Try to infer request body schema from handler source code."""
    if not handler_dir.exists():
        return None

    # Search handler files for validate_body or request_body patterns
    for py_file in handler_dir.rglob("*.py"):
        try:
            source = py_file.read_text()
        except Exception:
            continue

        # Look for @validate_body(required_fields=[...]) decorator
        validate_match = re.search(r"@validate_body\(\s*required_fields\s*=\s*\[([^\]]+)\]", source)
        if validate_match and operation_id and operation_id.lower() in source.lower():
            fields_str = validate_match.group(1)
            fields = [f.strip().strip("'\"") for f in fields_str.split(",")]
            properties = {}
            for field in fields:
                if field:
                    properties[field] = {"type": "string"}
            if properties:
                return {
                    "type": "object",
                    "required": [f for f in fields if f],
                    "properties": properties,
                }

    return None


def add_request_schemas(
    spec: dict,
    handler_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[dict, int, int, int]:
    """Add request body schemas to operations missing them.

    Returns: (updated_spec, added_known, added_inferred, added_generic)
    """
    added_known = 0
    added_inferred = 0
    added_generic = 0

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in BODY_METHODS:
                continue

            # Skip if already has requestBody with schema
            request_body = details.get("requestBody", {})
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    json_content = content.get("application/json", {})
                    if isinstance(json_content, dict) and json_content.get("schema"):
                        continue  # Already has schema

            # Try known schemas first
            key = f"{method.upper()} {path}"
            if key in KNOWN_SCHEMAS:
                details["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": KNOWN_SCHEMAS[key],
                        }
                    },
                }
                added_known += 1
                if verbose:
                    print(f"  [known] {key}")
                continue

            # Try to infer from handler source
            if handler_dir:
                operation_id = details.get("operationId", "")
                inferred = _infer_schema_from_handler(handler_dir, operation_id)
                if inferred:
                    details["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": inferred,
                            }
                        },
                    }
                    added_inferred += 1
                    if verbose:
                        print(f"  [inferred] {key}")
                    continue

            # Add generic schema as placeholder
            details["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": f"Request body for {method.upper()} {path}",
                        }
                    }
                },
            }
            added_generic += 1
            if verbose:
                print(f"  [generic] {key}")

    return spec, added_known, added_inferred, added_generic


def main() -> None:
    parser = argparse.ArgumentParser(description="Add request body schemas to OpenAPI spec")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/api/openapi.json"),
        help="Path to OpenAPI JSON spec",
    )
    parser.add_argument(
        "--handlers",
        type=Path,
        default=Path("aragora/server/handlers"),
        help="Path to handler source directory for schema inference",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--verbose", action="store_true", help="Show each operation")
    args = parser.parse_args()

    spec_path = args.spec
    if not spec_path.exists():
        print(f"Error: {spec_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path) as f:
        spec = json.load(f)

    handler_dir = args.handlers if args.handlers.exists() else None

    spec, known, inferred, generic = add_request_schemas(
        spec, handler_dir=handler_dir, verbose=args.verbose
    )

    total = known + inferred + generic
    print(f"\nRequest schemas added: {total}")
    print(f"  Known schemas:    {known}")
    print(f"  Inferred schemas: {inferred}")
    print(f"  Generic schemas:  {generic}")

    if args.dry_run:
        print("\n(dry-run: no changes written)")
    else:
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"\nWrote updated spec to {spec_path}")


if __name__ == "__main__":
    main()
