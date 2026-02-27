#!/usr/bin/env python3
"""
Compute documentation stats and optionally patch key docs.

Usage:
  python scripts/doc_stats.py            # print metrics only
  python scripts/doc_stats.py --write    # update key docs in-place
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Stats:
    python_modules: int
    test_count: int
    test_files: int
    api_paths: int
    api_operations: int
    ws_event_types: int
    km_adapters_registered: int
    workflow_templates: int
    ts_namespaces: int
    agent_types_allowlisted: int


def _run_rg_count(pattern: str, globs: Iterable[str], exclude_globs: Iterable[str]) -> int:
    cmd = ["rg", pattern]
    for glob in globs:
        cmd.extend(["-g", glob])
    for glob in exclude_globs:
        cmd.extend(["-g", f"!{glob}"])
    cmd.append(str(ROOT))
    try:
        out = subprocess.check_output(cmd, cwd=ROOT)
        return len(out.splitlines())
    except FileNotFoundError:
        return -1


def _count_py_files(path: Path) -> int:
    if not path.exists():
        return 0
    rel = path.relative_to(ROOT).as_posix()
    tracked = _tracked_python_files()
    if tracked is not None:
        return sum(1 for p in tracked if p.startswith(f"{rel}/"))
    return sum(
        1
        for p in path.rglob("*.py")
        if "__pycache__" not in p.parts
        and ".venv" not in p.parts
        and "venv" not in p.parts
        and "node_modules" not in p.parts
        and ".worktrees" not in p.parts
    )


@lru_cache(maxsize=1)
def _tracked_python_files() -> list[str] | None:
    try:
        out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def _count_tests() -> int:
    tracked = _tracked_python_files()
    if tracked is not None:
        pattern = re.compile(r"^\s*def test_", re.MULTILINE)
        total = 0
        for rel_path in tracked:
            path = ROOT / rel_path
            try:
                total += len(pattern.findall(path.read_text(errors="ignore")))
            except OSError:
                continue
        return total

    count = _run_rg_count(
        "def test_",
        globs=["*.py"],
        exclude_globs=[
            "**/node_modules/**",
            "**/.nomic/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.worktrees/**",
        ],
    )
    if count >= 0:
        return count
    # Fallback: scan tests/ directory only
    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        return 0
    pattern = re.compile(r"^\s*def test_", re.MULTILINE)
    total = 0
    for p in tests_dir.rglob("*.py"):
        total += len(pattern.findall(p.read_text(errors="ignore")))
    return total


def _count_api_ops() -> tuple[int, int]:
    candidates = [
        ROOT / "docs/api/openapi.json",
        ROOT / "docs/api/openapi_generated.json",
        ROOT / "docs/api/openapi.yaml",
    ]
    spec_path = next((p for p in candidates if p.exists()), None)
    if not spec_path:
        return 0, 0
    data: dict
    if spec_path.suffix == ".json":
        data = json.loads(spec_path.read_text())
    else:
        # YAML file may be JSON-formatted; try JSON parse first
        raw = spec_path.read_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return 0, 0
    paths = data.get("paths", {})
    ops = 0
    for _, methods in paths.items():
        for method in methods:
            if method.lower() in {"get", "post", "put", "patch", "delete", "head", "options"}:
                ops += 1
    return len(paths), ops


def _count_ws_events() -> int:
    path = ROOT / "aragora/events/types.py"
    if not path.exists():
        return 0
    text = path.read_text()
    in_enum = False
    count = 0
    for line in text.splitlines():
        if line.startswith("class StreamEventType"):
            in_enum = True
            continue
        if in_enum and line.startswith("class ") and not line.startswith("class StreamEventType"):
            break
        if not in_enum:
            continue
        if re.match(r"\s*[A-Z0-9_]+\s*=\s*\"[a-z0-9_]+\"", line):
            count += 1
    return count


def _count_km_adapters() -> int:
    path = ROOT / "aragora/knowledge/mound/adapters/factory.py"
    if not path.exists():
        return 0
    text = path.read_text()
    return len(re.findall(r'name="[^"]+"', text))


def _count_templates() -> int:
    base = ROOT / "aragora/workflow/templates"
    if not base.exists():
        return 0
    exts = {".yaml", ".yml", ".py"}
    return sum(
        1
        for p in base.rglob("*")
        if p.is_file() and p.suffix in exts and "__pycache__" not in p.parts
    )


def _count_ts_namespaces() -> int:
    base = ROOT / "sdk/typescript/src/namespaces"
    if not base.exists():
        return 0
    return sum(1 for p in base.glob("*.ts") if p.is_file())


def _count_allowlisted_agents() -> int:
    path = ROOT / "aragora/config/settings.py"
    if not path.exists():
        return 0
    text = path.read_text()
    m = re.search(r"ALLOWED_AGENT_TYPES:.*?=\s*frozenset\((\s*\{.*?\}\s*)\)", text, re.S)
    if not m:
        return 0
    return len(re.findall(r"\"([^\"]+)\"", m.group(1)))


def _approx(value: int, step: int) -> str:
    if value <= 0:
        return "0"
    rounded = max(step, (value // step) * step)
    return f"{rounded:,}+"


def compute_stats() -> Stats:
    python_modules = _count_py_files(ROOT / "aragora")
    test_count = _count_tests()
    test_files = _count_py_files(ROOT / "tests")
    api_paths, api_operations = _count_api_ops()
    ws_event_types = _count_ws_events()
    km_adapters_registered = _count_km_adapters()
    workflow_templates = _count_templates()
    ts_namespaces = _count_ts_namespaces()
    agent_types_allowlisted = _count_allowlisted_agents()
    return Stats(
        python_modules=python_modules,
        test_count=test_count,
        test_files=test_files,
        api_paths=api_paths,
        api_operations=api_operations,
        ws_event_types=ws_event_types,
        km_adapters_registered=km_adapters_registered,
        workflow_templates=workflow_templates,
        ts_namespaces=ts_namespaces,
        agent_types_allowlisted=agent_types_allowlisted,
    )


def _apply_patterns(
    text: str, patterns: list[tuple[str, str | Callable[[re.Match], str], int]]
) -> tuple[str, int]:
    total = 0
    for pattern, repl, flags in patterns:
        text, n = re.subn(pattern, repl, text, flags=flags)
        total += n
    return text, total


def patch_docs(stats: Stats, write: bool) -> int:
    modules_approx = _approx(stats.python_modules, 1000)
    tests_approx = _approx(stats.test_count, 1000)
    test_files_approx = _approx(stats.test_files, 1000)
    api_ops_approx = _approx(stats.api_operations, 1000)
    api_paths_approx = _approx(stats.api_paths, 100)
    ws_events_approx = _approx(stats.ws_event_types, 10)
    templates_approx = _approx(stats.workflow_templates, 10)
    agent_types_approx = _approx(stats.agent_types_allowlisted, 10)

    replacements = {
        "README.md": [
            (
                r"orchestrates\s+\d[\d,]*\+\s+agent types",
                f"orchestrates {agent_types_approx} agent types",
                0,
            ),
            (
                r"Knowledge Mound with\s+\d+\s+registered adapters",
                f"Knowledge Mound with {stats.km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*\+\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*\+\s+paths", f"{api_paths_approx} paths", 0),
            (
                r"\d[\d,]*\+\s+WebSocket event types",
                f"{ws_events_approx} WebSocket event types",
                0,
            ),
            (r"\d[\d,]*\+\s+templates", f"{templates_approx} templates", 0),
            (r"\d[\d,]*\+\s+Python modules", f"{modules_approx} Python modules", 0),
            (r"\d[\d,]*\+\s+tests", f"{tests_approx} tests", 0),
            (r"\(\d[\d,]*\s+namespaces\)", f"({stats.ts_namespaces} namespaces)", 0),
        ],
        "docs/EXTENDED_README.md": [
            (
                r"AGENT LAYER \(\d[\d,]*\+\s+Agent Types\)",
                f"AGENT LAYER ({agent_types_approx} Agent Types)",
                0,
            ),
            (r"\d[\d,]*\+\s+agent types", f"{agent_types_approx} agent types", 0),
            (
                r"\d+\s+registered adapters",
                f"{stats.km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*\+\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*\+\s+paths", f"{api_paths_approx} paths", 0),
            (
                r"\d[\d,]*\+\s+WebSocket event types",
                f"{ws_events_approx} WebSocket event types",
                0,
            ),
            (r"\d[\d,]*\+\s+templates", f"{templates_approx} templates", 0),
            (r"\d[\d,]*\+\s+Python modules", f"{modules_approx} Python modules", 0),
            (
                r"(\*\*Scale:\*\*[^\n]*?)\d[\d,]*\+\s+tests",
                lambda m, value=tests_approx: f"{m.group(1)}{value} tests",
                0,
            ),
            (r"\d[\d,]*\+\s+test files", f"{test_files_approx} test files", 0),
            (
                r"\d[\d,]*\s+TypeScript SDK namespaces",
                f"{stats.ts_namespaces} TypeScript SDK namespaces",
                0,
            ),
        ],
        "docs/COMMERCIAL_OVERVIEW.md": [
            (
                r"orchestrating\s+\d[\d,]*\+\s+agent types",
                f"orchestrating {agent_types_approx} agent types",
                0,
            ),
            (
                r"\d+\s+registered adapters",
                f"{stats.km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*\+\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*\+\s+agent types", f"{agent_types_approx} agent types", 0),
        ],
        "docs/FEATURE_DISCOVERY.md": [
            (r"\d[\d,]*\+\s+Python modules", f"{modules_approx} Python modules", 0),
            (
                r"(\*\*Total\*\*:[^\n]*?)\d[\d,]*\+\s+tests",
                lambda m, value=tests_approx: f"{m.group(1)}{value} tests",
                0,
            ),
            (r"\d[\d,]*\+\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*\+\s+pre-built templates", f"{templates_approx} pre-built templates", 0),
            (
                r"Supported Providers \(\d[\d,]*\+\s+agent types\)",
                f"Supported Providers ({agent_types_approx} agent types)",
                0,
            ),
        ],
        "docs/FEATURE_PARITY_MATRIX.md": [
            (r"\d[\d,]*\+\s+operations", f"{api_ops_approx} operations", 0),
        ],
        "docs/WEBSOCKET_EVENTS.md": [
            (r"\(\d+ event types", f"({stats.ws_event_types} event types", 0),
        ],
        "docs/KNOWLEDGE_MOUND.md": [
            (
                r"\d+\s+registered adapters",
                f"{stats.km_adapters_registered} registered adapters",
                0,
            ),
        ],
        "docs/DOCUMENTATION_HUB.md": [
            (
                r"\d+\s+registered adapters",
                f"{stats.km_adapters_registered} registered adapters",
                0,
            ),
        ],
        "CLAUDE.md": [
            (r"\d[\d,]*\+\s+Python modules", f"{modules_approx} Python modules", 0),
            (
                r"(\*\*Codebase Scale:\*\*[^\n]*?)\d[\d,]*\+\s+tests",
                lambda m, value=tests_approx: f"{m.group(1)}{value} tests",
                0,
            ),
            (
                r"(\*\*Codebase Scale:\*\*[^\n]*?)\d[\d,]*\+\s+test files",
                lambda m, value=test_files_approx: f"{m.group(1)}{value} test files",
                0,
            ),
            (
                r"\*\*Test Suite:\*\*\s*\d[\d,]*\+\s+tests\s+across\s+\d[\d,]*\+\s+test files",
                f"**Test Suite:** {tests_approx} tests across {test_files_approx} test files",
                0,
            ),
            (r"\d[\d,]*\+\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*\+\s+paths", f"{api_paths_approx} paths", 0),
            (r"\d+\s+KM adapters", f"{stats.km_adapters_registered} KM adapters", 0),
            (r"\d[\d,]*\s+SDK namespaces", f"{stats.ts_namespaces} SDK namespaces", 0),
        ],
        "docs/architecture/system-overview.md": [
            (
                r"Agents Layer \(\d[\d,]*\+\s+Agent Types\)",
                f"Agents Layer ({agent_types_approx} Agent Types)",
                0,
            ),
            (
                r"\d[\d,]*\+\s+agent-type integrations",
                f"{agent_types_approx} agent-type integrations",
                0,
            ),
        ],
        "docs/landing/hero.md": [
            (
                r"\*\*\d[\d,]*\+\s+agent types\*\*",
                f"**{agent_types_approx} agent types**",
                0,
            ),
        ],
    }

    updated_files = 0
    for rel_path, patterns in replacements.items():
        path = ROOT / rel_path
        if not path.exists():
            continue
        original = path.read_text()
        updated, total = _apply_patterns(original, patterns)
        if total > 0:
            updated_files += 1
            if write:
                path.write_text(updated)
        elif write:
            # Keep silent on missing patterns to avoid noise in CI
            pass
    return updated_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Patch key docs in-place")
    args = parser.parse_args()

    stats = compute_stats()
    print("Doc stats:")
    print(f"- Python modules (aragora/): {stats.python_modules}")
    print(f"- Tests (def test_ across repo): {stats.test_count}")
    print(f"- Test files (tests/): {stats.test_files}")
    print(f"- API paths: {stats.api_paths}")
    print(f"- API operations: {stats.api_operations}")
    print(f"- WebSocket event types: {stats.ws_event_types}")
    print(f"- KM adapters registered: {stats.km_adapters_registered}")
    print(f"- Workflow templates: {stats.workflow_templates}")
    print(f"- TypeScript namespaces: {stats.ts_namespaces}")
    print(f"- Allowlisted agent types: {stats.agent_types_allowlisted}")

    if args.write:
        updated = patch_docs(stats, write=True)
        print(f"\\nUpdated {updated} documentation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
