"""Generate connector registry JSON for docs and tooling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_build_registry():
    """Load connector registry builder without importing aragora.connectors package."""
    registry_path = REPO_ROOT / "aragora" / "connectors" / "registry.py"
    spec = importlib.util.spec_from_file_location("connector_registry_module", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load connector registry module from {registry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_registry


build_registry = _load_build_registry()


def render_catalog(registry) -> str:
    lines: list[str] = []
    lines.append("# Connector Catalog")
    lines.append("")
    lines.append("This catalog is generated from the codebase. Do not edit by hand.")
    lines.append("")
    lines.append(f"_Generated: {registry.generated_at}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total connectors: {registry.total}")
    for kind, count in registry.by_kind.items():
        lines.append(f"- {kind.title()} connectors: {count}")

    grouped: dict[str, dict[str, list]] = {}
    for connector in registry.connectors:
        grouped.setdefault(connector.kind, {}).setdefault(connector.category, []).append(connector)

    for kind in sorted(grouped.keys()):
        lines.append("")
        lines.append(f"## {kind.title()} connectors")
        categories = grouped[kind]
        for category in sorted(categories.keys()):
            lines.append("")
            lines.append(f"### {category.replace('.', ' / ')}")
            lines.append("")
            for connector in categories[category]:
                lines.append(f"- `{connector.name}` — `{connector.module}`")

    lines.append("")
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _normalize_generated_timestamp(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.startswith("_Generated:")]
    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Update connector registry artifacts.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if registry artifacts are out of date.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    connectors_root = repo_root / "aragora" / "connectors"
    output_dir = repo_root / "docs" / "connectors"
    output_path = output_dir / "CONNECTOR_REGISTRY.json"
    catalog_path = output_dir / "CONNECTOR_CATALOG.md"

    registry = build_registry(connectors_root)
    payload = registry.to_dict()
    catalog = render_catalog(registry)

    expected_json = json.dumps(payload, indent=2, sort_keys=True)
    expected_md = catalog

    if args.check:
        current_json = _read_text(output_path)
        current_md = _read_text(catalog_path)
        if not current_json or not current_md:
            print("Connector registry artifacts are missing.")
            print("Run:")
            print("  python scripts/update_connector_registry.py")
            return 1

        try:
            current_payload = json.loads(current_json)
        except json.JSONDecodeError:
            print("Connector registry JSON is invalid.")
            print("Run:")
            print("  python scripts/update_connector_registry.py")
            return 1

        expected_payload = dict(payload)
        expected_payload.pop("generated_at", None)
        current_payload.pop("generated_at", None)

        normalized_current_md = _normalize_generated_timestamp(current_md)
        normalized_expected_md = _normalize_generated_timestamp(expected_md)

        json_match = current_payload == expected_payload
        md_match = normalized_current_md == normalized_expected_md

        if not json_match or not md_match:
            print("Connector registry artifacts are out of date.")
            if not json_match:
                exp_connectors = {
                    (c["name"], c["module"]) for c in expected_payload.get("connectors", [])
                }
                cur_connectors = {
                    (c["name"], c["module"]) for c in current_payload.get("connectors", [])
                }
                added = sorted(exp_connectors - cur_connectors)
                removed = sorted(cur_connectors - exp_connectors)
                if added:
                    print(f"  Added (expected but not committed): {added}")
                if removed:
                    print(f"  Removed (committed but not expected): {removed}")
                if expected_payload.get("total") != current_payload.get("total"):
                    print(
                        f"  Total: expected={expected_payload.get('total')} committed={current_payload.get('total')}"
                    )
                if expected_payload.get("by_kind") != current_payload.get("by_kind"):
                    print(
                        f"  by_kind: expected={expected_payload.get('by_kind')} committed={current_payload.get('by_kind')}"
                    )
                if expected_payload.get("by_category") != current_payload.get("by_category"):
                    print(
                        f"  by_category: expected={expected_payload.get('by_category')} committed={current_payload.get('by_category')}"
                    )
                if not added and not removed:
                    # Same connectors but different field values
                    exp_by_key = {
                        (c["name"], c["module"]): c for c in expected_payload.get("connectors", [])
                    }
                    cur_by_key = {
                        (c["name"], c["module"]): c for c in current_payload.get("connectors", [])
                    }
                    for key in sorted(exp_by_key.keys() & cur_by_key.keys()):
                        if exp_by_key[key] != cur_by_key[key]:
                            print(
                                f"  Field diff {key}: expected={exp_by_key[key]} committed={cur_by_key[key]}"
                            )
            if not md_match:
                exp_lines = normalized_expected_md.splitlines()
                cur_lines = normalized_current_md.splitlines()
                print(f"  Markdown lines: expected={len(exp_lines)} committed={len(cur_lines)}")
                for i, (e, c) in enumerate(zip(exp_lines, cur_lines)):
                    if e != c:
                        print(f"  First MD diff at line {i}: expected={e!r} committed={c!r}")
                        break
            print("Run:")
            print("  python scripts/update_connector_registry.py")
            return 1
        print("Connector registry artifacts are up to date.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(expected_json, encoding="utf-8")
    catalog_path.write_text(expected_md, encoding="utf-8")

    summary = [
        f"Total connectors: {registry.total}",
        "By kind:",
    ]
    for kind, count in registry.by_kind.items():
        summary.append(f"  - {kind}: {count}")
    print("\n".join(summary))
    print(f"Wrote {output_path}")
    print(f"Wrote {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
