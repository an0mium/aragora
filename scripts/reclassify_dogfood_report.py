#!/usr/bin/env python3
"""Reclassify dogfood report stderr signals into blockers vs warning-only noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aragora.debate.runtime_blockers import classify_stderr_signals


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def reclassify_report(data: dict[str, Any]) -> dict[str, Any]:
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Report must contain runs list.")

    warning_only_runs = 0
    blocker_runs = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        stderr_excerpt = str(run.get("stderr_excerpt") or "")
        classified = classify_stderr_signals(stderr_excerpt)
        run["runtime_blockers"] = classified["runtime_blockers"]
        run["warning_signals"] = classified["warning_signals"]
        run["warning_only"] = classified["warning_only"]
        if classified["runtime_blockers"]:
            blocker_runs += 1
        elif classified["warning_only"]:
            warning_only_runs += 1

    data["runtime_blockers_zero"] = blocker_runs == 0
    data["warning_only_runs"] = warning_only_runs
    data["blocker_runs"] = blocker_runs
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_report", help="Path to existing dogfood report JSON")
    parser.add_argument(
        "--output",
        help="Output path (defaults to overwriting input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_report).expanduser()
    output_path = Path(args.output).expanduser() if args.output else input_path

    report = _load_json(input_path)
    updated = reclassify_report(report)
    _write_json(output_path, updated)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "runtime_blockers_zero": updated.get("runtime_blockers_zero"),
                "warning_only_runs": updated.get("warning_only_runs"),
                "blocker_runs": updated.get("blocker_runs"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
