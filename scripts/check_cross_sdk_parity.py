#!/usr/bin/env python3
"""Cross-language SDK parity check.

Extracts normalized endpoint paths from both the Python and TypeScript SDKs
and reports which paths exist in only one language.

Usage:
    python scripts/check_cross_sdk_parity.py
    python scripts/check_cross_sdk_parity.py --strict --baseline scripts/baselines/cross_sdk_parity.json
    python scripts/check_cross_sdk_parity.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from sdk_path_normalize import normalize_sdk_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Extraction helpers (mirroring check_sdk_parity.py patterns)
# ---------------------------------------------------------------------------

_PY_PATH_RE = re.compile(r'self\._client\.(?:_request|request)\(\s*"[A-Z]+"\s*,\s*[f"]([^"]+)"')
_PY_FSTR_RE = re.compile(r'self\._client\.(?:_request|request)\(\s*"[A-Z]+"\s*,\s*f"([^"]+)"')
_TS_REQUEST_RE = re.compile(
    r"request(?:<[^(]*>)?\(\s*['\"](?:[A-Z]+)['\"]\s*,"
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
_TS_DIRECT_RE = re.compile(
    r"this\.client\.(?:get|post|put|delete|patch)\("
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)


def _extract_python_paths() -> set[str]:
    sdk_dir = PROJECT_ROOT / "sdk" / "python" / "aragora_sdk" / "namespaces"
    if not sdk_dir.exists():
        return set()

    paths: set[str] = set()
    for py_file in sdk_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _PY_PATH_RE.finditer(content):
            paths.add(normalize_sdk_path(m.group(1)))
        for m in _PY_FSTR_RE.finditer(content):
            paths.add(normalize_sdk_path(m.group(1)))
    return paths


def _extract_typescript_paths() -> set[str]:
    sdk_dir = PROJECT_ROOT / "sdk" / "typescript" / "src" / "namespaces"
    if not sdk_dir.exists():
        return set()

    paths: set[str] = set()
    for ts_file in sdk_dir.glob("*.ts"):
        if ts_file.name.startswith("_") or ts_file.name == "index.ts":
            continue
        try:
            content = ts_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _TS_REQUEST_RE.finditer(content):
            raw = m.group("path")[1:-1]  # strip quotes/backticks
            paths.add(normalize_sdk_path(raw))
        for m in _TS_DIRECT_RE.finditer(content):
            raw = m.group("path")[1:-1]
            paths.add(normalize_sdk_path(raw))
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _mute_stdout_after_broken_pipe() -> None:
    try:
        sys.stdout.close()
    except OSError:
        pass
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _emit_text(output: str) -> bool:
    try:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-language SDK parity check")
    parser.add_argument("--strict", action="store_true", help="Fail on regressions beyond baseline")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON file for regression gating",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    py_paths = _extract_python_paths()
    ts_paths = _extract_typescript_paths()

    python_only = sorted(py_paths - ts_paths)
    typescript_only = sorted(ts_paths - py_paths)
    common = sorted(py_paths & ts_paths)

    report = {
        "python_endpoint_count": len(py_paths),
        "typescript_endpoint_count": len(ts_paths),
        "common_count": len(common),
        "python_only": python_only,
        "typescript_only": typescript_only,
    }

    lines: list[str] = []
    if args.json:
        lines.append(json.dumps(report, indent=2))
    else:
        lines.extend(
            [
                f"Python SDK paths:     {len(py_paths)}",
                f"TypeScript SDK paths: {len(ts_paths)}",
                f"Common:               {len(common)}",
                f"Python-only:          {len(python_only)}",
                f"TypeScript-only:      {len(typescript_only)}",
            ]
        )

        if python_only:
            lines.append("")
            lines.append(f"Python-only endpoints ({len(python_only)}):")
            lines.extend(f"  {p}" for p in python_only[:20])
            if len(python_only) > 20:
                lines.append(f"  ... and {len(python_only) - 20} more")

        if typescript_only:
            lines.append("")
            lines.append(f"TypeScript-only endpoints ({len(typescript_only)}):")
            lines.extend(f"  {p}" for p in typescript_only[:20])
            if len(typescript_only) > 20:
                lines.append(f"  ... and {len(typescript_only) - 20} more")

    # Baseline regression check
    baseline_py_only: set[str] = set()
    baseline_ts_only: set[str] = set()
    if args.baseline and args.baseline.exists():
        data = json.loads(args.baseline.read_text())
        baseline_py_only = set(data.get("python_only", []))
        baseline_ts_only = set(data.get("typescript_only", []))

    new_py_only = set(python_only) - baseline_py_only
    new_ts_only = set(typescript_only) - baseline_ts_only

    if not args.json:
        if args.baseline:
            lines.append("")
            lines.append(
                f"Baseline regressions: python_only={len(new_py_only)} typescript_only={len(new_ts_only)}"
            )
            lines.extend(f"  NEW PY-ONLY: {p}" for p in sorted(new_py_only)[:10])
            lines.extend(f"  NEW TS-ONLY: {p}" for p in sorted(new_ts_only)[:10])

    exit_code = 0
    if args.strict:
        if new_py_only or new_ts_only:
            lines.append("")
            lines.append("FAILED: Cross-SDK parity regression (--strict mode)")
            exit_code = 1
        elif not args.json:
            lines.append("")
            lines.append("PASS: No new cross-SDK parity regressions")

    if not _emit_text("\n".join(lines)):
        return 0
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
