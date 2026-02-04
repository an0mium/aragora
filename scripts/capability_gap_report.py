#!/usr/bin/env python3
"""Generate a capability surface gap report.

Uses:
- aragora/capabilities.yaml (capability catalog)
- aragora/capability_surfaces.yaml (surface exposure map)

Outputs Markdown (default) or JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SurfaceCoverage:
    cli: list[str]
    api: list[str]
    sdk_python: list[str]
    sdk_typescript: list[str]
    ui: list[str]
    channels: list[str]

    @property
    def has_cli(self) -> bool:
        return bool(self.cli)

    @property
    def has_api(self) -> bool:
        return bool(self.api)

    @property
    def has_sdk(self) -> bool:
        return bool(self.sdk_python or self.sdk_typescript)

    @property
    def has_ui(self) -> bool:
        return bool(self.ui)

    @property
    def has_channels(self) -> bool:
        return bool(self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cli": self.cli,
            "api": self.api,
            "sdk": {
                "python": self.sdk_python,
                "typescript": self.sdk_typescript,
            },
            "ui": self.ui,
            "channels": self.channels,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if str(v).strip()]
    return [str(value)]


def _parse_surface_entry(entry: dict[str, Any]) -> SurfaceCoverage:
    sdk = entry.get("sdk") or {}
    if not isinstance(sdk, dict):
        sdk = {}
    return SurfaceCoverage(
        cli=_as_list(entry.get("cli")),
        api=_as_list(entry.get("api")),
        sdk_python=_as_list(sdk.get("python")),
        sdk_typescript=_as_list(sdk.get("typescript")),
        ui=_as_list(entry.get("ui")),
        channels=_as_list(entry.get("channels")),
    )


def build_report(repo_root: Path, *, include_unmapped: bool = True) -> dict[str, Any]:
    base_path = repo_root / "aragora" / "capabilities.yaml"
    surfaces_path = repo_root / "aragora" / "capability_surfaces.yaml"

    base_data = _load_yaml(base_path)
    surface_data = _load_yaml(surfaces_path)

    base_caps = base_data.get("capabilities") or {}
    if not isinstance(base_caps, dict):
        base_caps = {}

    surface_caps = surface_data.get("capabilities") or {}
    if not isinstance(surface_caps, dict):
        surface_caps = {}

    report_items: dict[str, Any] = {}
    unmapped: list[str] = []

    for key, payload in base_caps.items():
        if not isinstance(payload, dict):
            payload = {}
        name = payload.get("name", key)
        category = payload.get("category", "unknown")
        status = payload.get("status", "unknown")

        surfaces_entry = surface_caps.get(key)
        if surfaces_entry is None:
            unmapped.append(key)
            surfaces = None
        else:
            surfaces = _parse_surface_entry(surfaces_entry)

        report_items[key] = {
            "name": name,
            "category": category,
            "status": status,
            "surfaces": surfaces.to_dict() if surfaces else None,
        }

    # Include capabilities defined only in surface map
    for key, surfaces_entry in surface_caps.items():
        if key in report_items:
            continue
        surfaces = _parse_surface_entry(surfaces_entry)
        report_items[key] = {
            "name": key,
            "category": "unclassified",
            "status": "unknown",
            "surfaces": surfaces.to_dict(),
        }

    gaps = {
        "cli": [],
        "api": [],
        "sdk": [],
        "ui": [],
        "channels": [],
    }

    mapped_count = 0
    for key, info in report_items.items():
        surfaces = info.get("surfaces")
        if surfaces is None:
            continue
        mapped_count += 1
        coverage = _parse_surface_entry(surfaces)
        if not coverage.has_cli:
            gaps["cli"].append(key)
        if not coverage.has_api:
            gaps["api"].append(key)
        if not coverage.has_sdk:
            gaps["sdk"].append(key)
        if not coverage.has_ui:
            gaps["ui"].append(key)
        if not coverage.has_channels:
            gaps["channels"].append(key)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_capabilities": len(report_items),
        "mapped_capabilities": mapped_count,
        "unmapped_capabilities": sorted(unmapped) if include_unmapped else [],
        "gaps": {k: sorted(v) for k, v in gaps.items()},
        "items": report_items,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Capability Surface Gap Report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total capabilities: {report['total_capabilities']}")
    lines.append(f"- Mapped capabilities: {report['mapped_capabilities']}")
    unmapped = report.get("unmapped_capabilities") or []
    lines.append(f"- Unmapped capabilities: {len(unmapped)}")

    lines.append("")
    lines.append("## Gaps By Surface")
    lines.append("")
    for surface in ["cli", "api", "sdk", "ui", "channels"]:
        lines.append(f"### Missing {surface.upper()}")
        lines.append("")
        items = report["gaps"].get(surface) or []
        if not items:
            lines.append("- None")
        else:
            for item in items:
                lines.append(f"- {item}")
        lines.append("")

    if unmapped:
        lines.append("## Unmapped Capabilities")
        lines.append("")
        for item in unmapped:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate capability surface gap report")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root (default: repo root)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output path (default: stdout)",
    )
    args = parser.parse_args()

    report = build_report(Path(args.root))
    if args.format == "json":
        output = json.dumps(report, indent=2)
    else:
        output = render_markdown(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
