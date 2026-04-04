#!/usr/bin/env python3
"""
Generate TypeScript SDK OpenAPI types using openapi-typescript.

Usage:
  python scripts/generate_sdk_types.py
  python scripts/generate_sdk_types.py --check
  python scripts/generate_sdk_types.py --output sdk/typescript/src/openapi-types.ts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


DEFAULT_OPENAPI = Path("docs/api/openapi.json")
DEFAULT_OUTPUT = Path("sdk/typescript/src/openapi-types.ts")
OPENAPI_TYPESCRIPT_VERSION = "7.10.1"


def load_openapi_spec(openapi_path: Path) -> dict:
    """Load an OpenAPI spec from JSON or YAML."""
    if openapi_path.suffix.lower() == ".json":
        return json.loads(openapi_path.read_text())

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - YAML is optional for this script
        raise RuntimeError(
            "PyYAML is required to validate non-JSON OpenAPI specs before SDK generation."
        ) from exc

    return yaml.safe_load(openapi_path.read_text())


def find_duplicate_operation_ids(spec: dict) -> dict[str, list[str]]:
    """Return duplicate operationIds mapped to the operations that use them."""
    seen: dict[str, list[str]] = defaultdict(list)

    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue

        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                continue

            operation_id = details.get("operationId")
            if not operation_id:
                continue

            seen[str(operation_id)].append(f"{method.upper()} {path}")

    return {operation_id: uses for operation_id, uses in seen.items() if len(uses) > 1}


def ensure_unique_operation_ids(openapi_path: Path) -> None:
    """Fail fast with a clear error when the spec has duplicate operationIds."""
    duplicates = find_duplicate_operation_ids(load_openapi_spec(openapi_path))
    if not duplicates:
        return

    lines = [
        "OpenAPI spec has duplicate operationIds. Run the operationId postprocessor and regenerate SDK types:"
    ]
    for operation_id, uses in sorted(duplicates.items()):
        joined_uses = "; ".join(uses)
        lines.append(f"  - {operation_id}: {joined_uses}")
    raise ValueError("\n".join(lines))


def resolve_generator() -> list[str]:
    """Resolve the openapi-typescript binary to invoke."""
    candidates = [
        Path("sdk/typescript/node_modules/.bin/openapi-typescript"),
        Path("aragora/live/node_modules/.bin/openapi-typescript"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return ["npx", f"openapi-typescript@{OPENAPI_TYPESCRIPT_VERSION}"]


def generate_types(openapi_path: Path, output_path: Path) -> int:
    """Generate TypeScript types from the OpenAPI spec."""
    cmd = resolve_generator() + [str(openapi_path), "-o", str(output_path)]
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDK types from OpenAPI")
    parser.add_argument("--openapi", type=Path, default=DEFAULT_OPENAPI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if generated output differs")
    args = parser.parse_args()

    if not args.openapi.exists():
        print(f"OpenAPI spec not found: {args.openapi}", file=sys.stderr)
        sys.exit(1)

    try:
        ensure_unique_operation_ids(args.openapi)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.check:
        if not args.output.exists():
            print(f"Expected output not found: {args.output}", file=sys.stderr)
            sys.exit(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_out = Path(tmpdir) / args.output.name
            code = generate_types(args.openapi, temp_out)
            if code != 0:
                sys.exit(code)
            if temp_out.read_text() != args.output.read_text():
                print("Generated SDK types are out of date.", file=sys.stderr)
                sys.exit(1)
        print("SDK types are up to date.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.exit(generate_types(args.openapi, args.output))


if __name__ == "__main__":
    main()
