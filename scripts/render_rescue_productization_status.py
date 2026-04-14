#!/usr/bin/env python3
"""Render a repo-tracked TW-03 rescue productization status summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_ROOT = REPO_ROOT / "docs" / "status" / "generated" / "rescue_productization"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "status" / "TW03_RESCUE_PRODUCTIZATION_STATUS.md"


def _repo_stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _issue_numbers_cell(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    return ", ".join(f"#{int(item)}" for item in value if isinstance(item, int))


def _render_repeated_classes(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No repeated rescue classes found in the current ledger window."]

    lines = [
        "| Rescue class | Count | Productization | Target | Example issues |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{str(row.get('class') or '').strip()}` | "
            f"{int(row.get('count', 0) or 0)} | "
            f"`{str(row.get('productization_status') or 'unlinked').strip() or 'unlinked'}` | "
            f"`{str(row.get('productization_target') or '-').strip() or '-'}` | "
            f"{_issue_numbers_cell(row.get('issue_numbers'))} |"
        )
    return lines


def _render_linkage_actions(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    lines: list[str] = []
    for row in rows:
        target = str(row.get("target") or "").strip()
        url = str(row.get("url") or "").strip()
        error = str(row.get("error") or "").strip()
        action = str(row.get("action") or "").strip()
        class_name = str(row.get("class") or "").strip()
        if url:
            lines.append(f"- `{action}` `{class_name}` -> `{target}` ([link]({url}))")
        elif error:
            lines.append(f"- `{action}` `{class_name}`: `{error}`")
        else:
            lines.append(f"- `{action}` `{class_name}` -> `{target or 'n/a'}`")
    return lines


def _render_issue_drafts(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [
        f"- `{str(row.get('class') or '').strip()}` -> `{str(row.get('title') or '').strip()}`"
        for row in rows
    ]


def render_status_markdown(*, report_path: Path, payload: dict[str, Any]) -> str:
    summary = dict(payload.get("summary") or {})
    repeated_classes = list(payload.get("repeated_classes") or [])
    one_off_classes = list(payload.get("one_off_classes") or [])
    below_threshold_classes = list(payload.get("below_threshold_classes") or [])
    issue_linkage_results = list(payload.get("issue_linkage_results") or [])
    issue_drafts = list(payload.get("issue_drafts") or [])
    generated_at = str(payload.get("generated_at") or "").strip() or "unknown"

    lines = [
        "# TW-03 Rescue Productization Status",
        "",
        f"Last updated: {generated_at}",
        "",
        "This is the repo-tracked recurring `TW-03` publication surface for repeated rescue-class harvest and conversion.",
        "",
        "## Summary",
        "",
        f"- Latest report: `{_repo_stable_path(report_path)}`",
        f"- Rescue ledger path: `{_format_value(payload.get('ledger_path'))}`",
        f"- Productization map: `{_format_value(payload.get('productization_map_path'))}`",
        f"- Repeated rescue classes: `{_format_value(summary.get('repeated_class_count'))}`",
        f"- Linked repeated classes: `{_format_value(summary.get('linked_fixture_count', 0) + summary.get('linked_issue_count', 0) + summary.get('linked_other_count', 0))}`",
        f"- Unlinked repeated classes: `{_format_value(summary.get('unlinked_repeated_class_count'))}`",
        f"- One-off classes: `{_format_value(summary.get('one_off_class_count'))}`",
        f"- Below-threshold classes: `{_format_value(summary.get('below_threshold_class_count'))}`",
        f"- Issue drafts remaining: `{len(issue_drafts)}`",
        "",
        "## Repeated Rescue Classes",
        "",
        *_render_repeated_classes(repeated_classes),
        "",
        "## Issue Linkage Actions",
        "",
        *_render_linkage_actions(issue_linkage_results),
        "",
        "## Remaining Issue Drafts",
        "",
        *_render_issue_drafts(issue_drafts),
        "",
        "## One-Off Rescue Classes",
        "",
    ]
    if one_off_classes:
        lines.extend(
            f"- `{str(row.get('class') or '').strip()}` ({int(row.get('count', 0) or 0)}x)"
            for row in one_off_classes
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Below-Threshold Rescue Classes", ""])
    if below_threshold_classes:
        lines.extend(
            f"- `{str(row.get('class') or '').strip()}` ({int(row.get('count', 0) or 0)}x)"
            for row in below_threshold_classes
        )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_output(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help=f"Tracked rescue-productization report root (default: {DEFAULT_REPORT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown status output path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_root = args.report_root.resolve()
    output_path = args.output.resolve()
    report_path = report_root / "latest.json"
    if not report_path.exists():
        raise SystemExit(f"rescue productization report not found: {report_path}")

    content = render_status_markdown(report_path=report_path, payload=_load_json(report_path))
    written = write_output(output_path, content)
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
