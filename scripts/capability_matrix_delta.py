#!/usr/bin/env python3
"""Generate a compact capability-matrix PR summary artifact."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = Path("docs/CAPABILITY_MATRIX.md")


def _extract(pattern: str, text: str) -> tuple[int, ...] | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def _parse_matrix(text: str) -> dict[str, tuple[int, ...]]:
    parsed: dict[str, tuple[int, ...]] = {}
    patterns = {
        "http": r"^\|\s*\*\*HTTP API\*\*\s*\|\s*(\d+)\s+paths\s*/\s*(\d+)\s+operations\s*\|",
        "cli": r"^\|\s*\*\*CLI\*\*\s*\|\s*(\d+)\s+commands\s*\|",
        "py_sdk": r"^\|\s*\*\*SDK \(Python\)\*\*\s*\|\s*(\d+)\s+namespaces\s*\|",
        "ts_sdk": r"^\|\s*\*\*SDK \(TypeScript\)\*\*\s*\|\s*(\d+)\s+namespaces\s*\|",
        "catalog": r"^\|\s*\*\*Capability Catalog\*\*\s*\|\s*(\d+)\/(\d+)\s+mapped\s*\|",
    }

    for key, pattern in patterns.items():
        vals = _extract(pattern, text)
        if vals is None:
            raise ValueError(f"Could not parse metric '{key}' from capability matrix")
        parsed[key] = vals

    return parsed


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_base_text(repo_root: Path, base_ref: str) -> str | None:
    target = f"{base_ref}:{MATRIX_PATH.as_posix()}"
    proc = subprocess.run(
        ["git", "show", target],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _delta(new: int, old: int | None) -> str:
    if old is None:
        return "n/a"
    diff = new - old
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff}"


def _metric_line(name: str, head: int, base: int | None) -> str:
    base_val = str(base) if base is not None else "n/a"
    return f"| {name} | {base_val} | {head} | {_delta(head, base)} |"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate capability matrix delta summary")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repo root")
    parser.add_argument("--base-ref", help="Git ref to compare against (e.g. origin/main)")
    parser.add_argument(
        "--out",
        default="/tmp/capability-matrix-summary.md",
        help="Output markdown file",
    )
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    current_path = repo_root / MATRIX_PATH
    out_path = Path(args.out)

    current_text = _load_text(current_path)
    current = _parse_matrix(current_text)

    base: dict[str, tuple[int, ...]] | None = None
    base_note = ""

    if args.base_ref:
        base_text = _load_base_text(repo_root, args.base_ref)
        if base_text:
            base = _parse_matrix(base_text)
        else:
            base_note = f"Base matrix unavailable for ref `{args.base_ref}`; showing head snapshot only."

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("# Capability Matrix Delta Summary")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Source: `{MATRIX_PATH.as_posix()}`")
    if args.base_ref:
        lines.append(f"Base ref: `{args.base_ref}`")
    if base_note:
        lines.append("")
        lines.append(f"> {base_note}")
    lines.append("")

    lines.append("| Metric | Base | Head | Delta |")
    lines.append("|---|---:|---:|---:|")

    base_http_paths = base["http"][0] if base else None
    base_http_ops = base["http"][1] if base else None
    base_cli = base["cli"][0] if base else None
    base_py = base["py_sdk"][0] if base else None
    base_ts = base["ts_sdk"][0] if base else None
    base_mapped = base["catalog"][0] if base else None
    base_total = base["catalog"][1] if base else None

    lines.append(_metric_line("HTTP paths", current["http"][0], base_http_paths))
    lines.append(_metric_line("HTTP operations", current["http"][1], base_http_ops))
    lines.append(_metric_line("CLI commands", current["cli"][0], base_cli))
    lines.append(_metric_line("Python SDK namespaces", current["py_sdk"][0], base_py))
    lines.append(_metric_line("TypeScript SDK namespaces", current["ts_sdk"][0], base_ts))
    lines.append(_metric_line("Mapped capabilities", current["catalog"][0], base_mapped))
    lines.append(_metric_line("Total capabilities", current["catalog"][1], base_total))

    current_cov = (current["catalog"][0] / current["catalog"][1] * 100) if current["catalog"][1] else 0.0
    if base and base["catalog"][1]:
        base_cov = base["catalog"][0] / base["catalog"][1] * 100
        cov_delta = current_cov - base_cov
        lines.append("")
        lines.append(f"Coverage: {base_cov:.1f}% -> {current_cov:.1f}% ({cov_delta:+.1f}pp)")
    else:
        lines.append("")
        lines.append(f"Coverage: {current_cov:.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
