#!/usr/bin/env python3
"""
Export the OpenAPI schema to docs/api.

The .yaml files are JSON-formatted for consistency with current docs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Ensure the local checkout wins over any globally installed Aragora package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# OpenAPI export is an offline docs task; do not reach out to Secrets Manager.
os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")

from scripts.add_openapi_descriptions import add_descriptions
from scripts.add_openapi_operation_ids import add_operation_ids
from scripts.add_openapi_param_descriptions import add_param_descriptions
from scripts.generate_openapi import generate_schema as generate_discovered_schema
from scripts.generate_openapi import save_schema as save_discovered_schema

from aragora.server.openapi import generate_openapi_schema


PRIMARY_ARTIFACT_NAMES = ("openapi.json", "openapi.yaml")
GENERATED_JSON_NAME = "openapi_generated.json"
GENERATED_YAML_NAME = "openapi_generated.yaml"
ARTIFACT_NAMES = PRIMARY_ARTIFACT_NAMES + (GENERATED_JSON_NAME, GENERATED_YAML_NAME)


def write_json(path: Path, data: dict, *, trailing_newline: bool = True) -> None:
    content = json.dumps(data, indent=2, sort_keys=False)
    if trailing_newline:
        content += "\n"
    path.write_text(content)


def export_schema(output_dir: Path, schema: dict) -> list[Path]:
    written: list[Path] = []
    for name in PRIMARY_ARTIFACT_NAMES:
        path = output_dir / name
        write_json(path, schema)
        written.append(path)

    discovered_schema = generate_discovered_schema()
    discovered_yaml_path = output_dir / GENERATED_YAML_NAME
    save_discovered_schema(discovered_schema, str(discovered_yaml_path), fmt="yaml")
    written.append(discovered_yaml_path)

    generated_json_schema = copy.deepcopy(discovered_schema)
    generated_json_schema, _, _, _ = add_operation_ids(generated_json_schema)
    generated_json_schema, _, _ = add_param_descriptions(generated_json_schema)
    generated_json_schema, _, _ = add_descriptions(generated_json_schema)

    generated_json_path = output_dir / GENERATED_JSON_NAME
    write_json(generated_json_path, generated_json_schema, trailing_newline=False)
    written.append(generated_json_path)

    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema to docs/api.")
    parser.add_argument(
        "--output-dir",
        default="docs/api",
        help="Output directory for exported OpenAPI artifacts (default: docs/api)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = generate_openapi_schema()

    written = export_schema(output_dir, schema)

    artifact_list = ", ".join(path.name for path in written)
    print(f"Wrote OpenAPI schema artifacts to {output_dir}: {artifact_list}")


if __name__ == "__main__":
    main()
