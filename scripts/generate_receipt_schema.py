#!/usr/bin/env python3
"""Generate JSON Schema files for Aragora receipt models.

Produces two schema files under ``docs/schemas/``:
- ``decision_receipt.v1.json`` -- export/decision_receipt.DecisionReceipt
- ``gauntlet_receipt.v1.json`` -- gauntlet/receipt_models.DecisionReceipt

Usage::

    python scripts/generate_receipt_schema.py            # write to docs/schemas/
    python scripts/generate_receipt_schema.py --check     # verify schemas are up to date
    python scripts/generate_receipt_schema.py --stdout    # print to stdout instead
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, get_type_hints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON_TYPE: dict[str, str | list[str]] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _python_type_to_json(annotation: str) -> str | list[str]:
    """Map a simplified Python type annotation to a JSON Schema type string."""
    clean = annotation.replace("typing.", "").strip()
    # Handle Optional / union with None
    if "None" in clean:
        base = clean.replace("| None", "").replace("None |", "").replace("Optional[", "").rstrip("]").strip()
        json_type = _PYTHON_TO_JSON_TYPE.get(base, "string")
        if isinstance(json_type, list):
            return [*json_type, "null"]
        return [json_type, "null"]
    return _PYTHON_TO_JSON_TYPE.get(clean, "string")


def _field_schema(f: dataclasses.Field[Any]) -> dict[str, Any]:
    """Build a JSON Schema property dict from a dataclass field."""
    schema: dict[str, Any] = {}
    type_str = str(f.type) if f.type else "str"
    json_type = _python_type_to_json(type_str)
    schema["type"] = json_type

    # Add default if present
    if f.default is not dataclasses.MISSING:
        schema["default"] = f.default
    elif f.default_factory is not dataclasses.MISSING:  # type: ignore[arg-type]
        try:
            schema["default"] = f.default_factory()  # type: ignore[misc]
        except Exception:
            pass

    return schema


def _generate_schema_from_dataclass(
    cls: type,
    *,
    schema_id: str,
    title: str,
    description: str,
    required_fields: list[str] | None = None,
    extra_defs: dict[str, Any] | None = None,
    verdict_enum: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a JSON Schema dict from a dataclass type.

    Parameters
    ----------
    cls:
        The dataclass to introspect.
    schema_id:
        The ``$id`` URI for the schema.
    title:
        Human-readable title.
    description:
        Human-readable description.
    required_fields:
        Explicit list of required property names.  Defaults to fields
        without defaults.
    extra_defs:
        Additional ``$defs`` to include.
    verdict_enum:
        If provided, sets the ``enum`` constraint on the ``verdict`` property.
    """
    fields = dataclasses.fields(cls)

    properties: dict[str, Any] = {}
    auto_required: list[str] = []

    for f in fields:
        prop = _field_schema(f)
        properties[f.name] = prop

        # Track required fields (no default)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[arg-type]
            auto_required.append(f.name)

    # Override verdict enum if provided
    if verdict_enum and "verdict" in properties:
        properties["verdict"]["enum"] = verdict_enum

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "title": title,
        "description": description,
        "type": "object",
        "required": required_fields if required_fields is not None else auto_required,
        "properties": properties,
    }

    if extra_defs:
        schema["$defs"] = extra_defs

    return schema


# ---------------------------------------------------------------------------
# Receipt-specific schemas
# ---------------------------------------------------------------------------


def generate_decision_receipt_schema() -> dict[str, Any]:
    """Generate schema for aragora.export.decision_receipt.DecisionReceipt."""
    from aragora.core_types import Verdict
    from aragora.export.decision_receipt import DecisionReceipt

    verdict_values = [v.value for v in Verdict]

    return _generate_schema_from_dataclass(
        DecisionReceipt,
        schema_id="https://aragora.ai/schemas/decision_receipt.v1.json",
        title="Aragora Decision Receipt (Export)",
        description=(
            "Audit-ready decision receipt from Gauntlet stress-tests or debate results. "
            "Defined in aragora.export.decision_receipt.DecisionReceipt."
        ),
        required_fields=["receipt_id", "gauntlet_id", "timestamp", "verdict", "confidence", "schema_version"],
        verdict_enum=verdict_values,
    )


def generate_gauntlet_receipt_schema() -> dict[str, Any]:
    """Generate schema for aragora.gauntlet.receipt_models.DecisionReceipt."""
    from aragora.core_types import Verdict
    from aragora.gauntlet.receipt_models import DecisionReceipt

    # Gauntlet receipts accept both legacy PASS/CONDITIONAL/FAIL and canonical Verdict values
    verdict_values = ["PASS", "CONDITIONAL", "FAIL"] + [v.value for v in Verdict]

    return _generate_schema_from_dataclass(
        DecisionReceipt,
        schema_id="https://aragora.ai/schemas/gauntlet_receipt.v1.json",
        title="Aragora Gauntlet Decision Receipt",
        description=(
            "Cryptographic audit-ready receipt from gauntlet adversarial validation. "
            "Defined in aragora.gauntlet.receipt_models.DecisionReceipt."
        ),
        required_fields=[
            "receipt_id", "gauntlet_id", "timestamp",
            "input_summary", "input_hash",
            "risk_summary", "attacks_attempted", "attacks_successful",
            "probes_run", "vulnerabilities_found",
            "verdict", "confidence", "robustness_score",
        ],
        verdict_enum=verdict_values,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSON Schema files for Aragora receipt models."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing schemas are up to date (exit 1 if stale).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print schemas to stdout instead of writing files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "schemas",
        help="Directory to write schema files (default: docs/schemas/).",
    )
    args = parser.parse_args()

    schemas = {
        "decision_receipt.v1.json": generate_decision_receipt_schema(),
        "gauntlet_receipt.v1.json": generate_gauntlet_receipt_schema(),
    }

    if args.stdout:
        for name, schema in schemas.items():
            print(f"--- {name} ---")
            print(json.dumps(schema, indent=2))
            print()
        return

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        stale = False
        for name, schema in schemas.items():
            path = output_dir / name
            generated = json.dumps(schema, indent=2) + "\n"
            if not path.exists():
                print(f"MISSING: {path}")
                stale = True
            elif path.read_text() != generated:
                print(f"STALE:   {path}")
                stale = True
            else:
                print(f"OK:      {path}")
        if stale:
            print("\nRun 'python scripts/generate_receipt_schema.py' to regenerate.")
            sys.exit(1)
        print("\nAll schemas up to date.")
        return

    for name, schema in schemas.items():
        path = output_dir / name
        path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"Wrote {path}")

    print("Done.")


if __name__ == "__main__":
    main()
