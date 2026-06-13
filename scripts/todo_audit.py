#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MARKER_RE = re.compile(r"^\s*#\s*(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)


def iter_markers(root: Path) -> list[tuple[Path, int, str]]:
    markers: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, 1):
            if MARKER_RE.search(line):
                markers.append((path, idx, line.strip()))
    return markers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count or list Python comment TODO/FIXME/HACK/XXX markers."
    )
    parser.add_argument(
        "--mode",
        choices=("count", "list"),
        required=True,
        help="Whether to print the marker count or matching lines.",
    )
    parser.add_argument(
        "--root",
        default="aragora",
        help="Root directory to scan for Python files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of list results to print.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _marker_tag(content: str) -> str:
    match = MARKER_RE.search(content)
    return match.group(1).upper() if match else ""


def build_json_payload(
    root: Path, mode: str, markers: list[tuple[Path, int, str]], *, limit: int
) -> dict[str, object]:
    payload: dict[str, object] = {
        "root": str(root),
        "mode": mode,
        "count": len(markers),
    }
    if mode == "list":
        payload["limit"] = limit
        payload["markers"] = [
            {
                "path": _relative_path(path, root),
                "line": line_no,
                "tag": _marker_tag(content),
                "content": content,
            }
            for path, line_no, content in markers[:limit]
        ]
    return payload


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    markers = iter_markers(root)
    if args.json:
        payload = build_json_payload(root, args.mode, markers, limit=args.limit)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.mode == "count":
        print(len(markers))
        return 0
    for path, line_no, content in markers[: args.limit]:
        print(f"{path}:{line_no}:{content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
