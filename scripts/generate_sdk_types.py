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
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_OPENAPI = Path("docs/api/openapi.json")
DEFAULT_OUTPUT = Path("sdk/typescript/src/openapi-types.ts")
OPENAPI_TYPESCRIPT_VERSION = "7.10.1"


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
