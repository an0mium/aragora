#!/usr/bin/env python3
"""
Export the OpenAPI schema to docs/api/openapi.json and docs/api/openapi.yaml.

The .yaml file is JSON-formatted for consistency with current docs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aragora.server.openapi import generate_openapi_schema


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema to docs/api.")
    parser.add_argument(
        "--output-dir",
        default="docs/api",
        help="Output directory for openapi.json/openapi.yaml (default: docs/api)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = generate_openapi_schema()

    write_json(output_dir / "openapi.json", schema)
    write_json(output_dir / "openapi.yaml", schema)

    print(f"Wrote OpenAPI schema to {output_dir}/openapi.json and {output_dir}/openapi.yaml")


if __name__ == "__main__":
    main()
