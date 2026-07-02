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


@dataclass(frozen=True)
class CanonicalMetric:
    value: int
    has_plus: bool


def _canonical_metrics() -> dict[str, CanonicalMetric]:
    """Read public baseline metric floors from the canonical goals table."""
    path = ROOT / "docs" / "CANONICAL_GOALS.md"
    if not path.exists():
        return {}

    key_for = {
        "python modules": "modules",
        "automated tests": "tests",
        "test files": "test_files",
        "api operations": "api_operations",
        "api paths": "api_paths",
        "knowledge mound adapters": "adapters",
        "agent types": "agent_types",
    }
    metrics: dict[str, CanonicalMetric] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = key_for.get(cells[0].lower())
        if not key:
            continue
        num = re.search(r"\d+(?:,\d+)*", cells[1])
        if not num:
            continue
        first_token = cells[1].split()[0] if cells[1].split() else ""
        metrics[key] = CanonicalMetric(
            value=int(num.group(0).replace(",", "")),
            has_plus="+" in first_token,
        )
    return metrics


def _metrics_doc_values() -> dict[str, CanonicalMetric]:
    """Read exact generated values from docs/METRICS.md."""
    path = ROOT / "docs" / "METRICS.md"
    if not path.exists():
        return {}

    key_for = {
        "python files under aragora/": "python_files",
        "python lines of code under aragora/": "python_lines",
        "top-level modules under aragora/": "top_level_modules",
        "test files (test_*.py under tests/)": "test_files",
        "test functions (class + module level)": "tests",
        "openapi paths": "api_paths",
        "openapi operations (http verbs)": "api_operations",
        "allowlisted agent types": "allowlisted_agent_types",
        "knowledge mound adapter specs": "adapter_specs",
        "knowledge mound adapter files": "adapter_files",
    }
    metrics: dict[str, CanonicalMetric] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = key_for.get(cells[0].lower())
        if not key:
            continue
        value_text = cells[1].strip().strip("`")
        num = re.search(r"\d+(?:,\d+)*", value_text)
        if not num:
            continue
        metrics[key] = CanonicalMetric(
            value=int(num.group(0).replace(",", "")),
            has_plus="+" in value_text,
        )
    return metrics


def _canonical_count(canonical: dict[str, CanonicalMetric], key: str, measured: str) -> str:
    metric = canonical.get(key)
    if not metric:
        return measured
    suffix = "+" if metric.has_plus else ""
    return f"{metric.value:,}{suffix}"


def _canonical_int(canonical: dict[str, CanonicalMetric], key: str, measured: int) -> int:
    metric = canonical.get(key)
    return metric.value if metric else measured


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
    return sum(
        1
        for p in path.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts and "node_modules" not in p.parts
    )


def _count_tests() -> int:
    # Keep the docs baseline stable across platforms and CI environments by
    # counting only tracked test definitions under tests/.
    try:
        out = subprocess.check_output(
            ["git", "grep", "-E", r"^[[:space:]]*(async )?def test_", "--", "tests"],
            cwd=ROOT,
            text=True,
        )
        return len(out.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError):
        tests_dir = ROOT / "tests"
        if not tests_dir.exists():
            return 0
        pattern = re.compile(r"^\s*(?:async\s+)?def test_", re.MULTILINE)
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
    return len(re.findall(r'"\.[a-z_]+_adapter"', text))


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
    rounded = (value // step) * step
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
    canonical = _canonical_metrics()
    metrics_doc = _metrics_doc_values()
    modules_approx = _canonical_count(canonical, "modules", _approx(stats.python_modules, 1000))
    tests_approx = _canonical_count(canonical, "tests", _approx(stats.test_count, 1000))
    test_files_approx = _canonical_count(
        canonical,
        "test_files",
        _approx(stats.test_files, 1000),
    )
    api_ops_approx = _canonical_count(
        canonical, "api_operations", _approx(stats.api_operations, 1000)
    )
    api_paths_approx = _canonical_count(
        canonical,
        "api_paths",
        _approx(stats.api_paths, 100),
    )
    ws_events_approx = _approx(stats.ws_event_types, 10)
    templates_approx = _approx(stats.workflow_templates, 10)
    agent_types_approx = _canonical_count(
        canonical,
        "agent_types",
        _approx(stats.agent_types_allowlisted, 10),
    )
    km_adapters_registered = _canonical_int(
        metrics_doc,
        "adapter_specs",
        stats.km_adapters_registered,
    )
    km_adapter_files = _canonical_count(metrics_doc, "adapter_files", "missing")
    exact_python_files = _canonical_count(
        metrics_doc,
        "python_files",
        f"{stats.python_modules:,}",
    )
    exact_python_lines = _canonical_count(metrics_doc, "python_lines", "missing")
    top_level_modules_fallback = _canonical_count(canonical, "modules", modules_approx)
    exact_top_level_modules = _canonical_count(
        metrics_doc,
        "top_level_modules",
        top_level_modules_fallback,
    )
    exact_tests = _canonical_count(metrics_doc, "tests", tests_approx)
    exact_test_files = _canonical_count(metrics_doc, "test_files", test_files_approx)
    exact_api_ops = _canonical_count(metrics_doc, "api_operations", api_ops_approx)
    exact_api_paths = _canonical_count(metrics_doc, "api_paths", api_paths_approx)
    claude_codebase_scale = (
        f"**Codebase Scale:** {exact_python_files} tracked Python files | "
        f"{exact_top_level_modules} top-level modules | {exact_tests} test functions | "
        f"{exact_test_files} test files | {exact_api_ops} API operations across "
        f"{exact_api_paths} paths | canonical counts in `docs/METRICS.md`"
    )
    extended_readme_scale = (
        f"**Scale:** {exact_python_files} tracked Python files | "
        f"{exact_top_level_modules} top-level modules | {exact_tests} test functions "
        f"across {exact_test_files} test files | canonical counts in [METRICS.md](METRICS.md)"
    )
    protected_metrics_keys = {
        "python_files",
        "python_lines",
        "top_level_modules",
        "tests",
        "test_files",
        "api_operations",
        "api_paths",
        "adapter_specs",
        "adapter_files",
    }
    missing_protected_metrics = sorted(protected_metrics_keys - set(metrics_doc))
    if write and missing_protected_metrics:
        raise RuntimeError(
            "refusing to update protected generated metric claims because "
            "docs/METRICS.md is missing rows: " + ", ".join(missing_protected_metrics)
        )
    claude_metrics_keys = protected_metrics_keys - {"python_lines"}
    missing_claude_metrics = sorted(claude_metrics_keys - set(metrics_doc))
    claude_patterns: list[tuple[str, str | Callable[[re.Match], str], int]] = []
    if not missing_claude_metrics:
        claude_patterns.extend(
            [
                (
                    r"(unified_server\.py\s+# Main server \()\d[\d,]*(?:\+)?\s+API operations(?=\))",
                    lambda m, value=exact_api_ops: f"{m.group(1)}{value} API operations",
                    0,
                ),
                (
                    r"\*\*Codebase Scale:\*\*[^\n]*canonical counts in `docs/METRICS\.md`",
                    claude_codebase_scale,
                    0,
                ),
                (
                    r"\*\*Test Suite:\*\*[^\n]*canonical counts in `docs/METRICS\.md`[^\n]*",
                    f"**Test Suite:** {exact_tests} test functions across "
                    f"{exact_test_files} test files (canonical counts in `docs/METRICS.md`)",
                    0,
                ),
            ]
        )
    claude_patterns.extend(
        [
            (r"\d+\s+KM adapters", f"{km_adapters_registered} KM adapters", 0),
            (
                r"(adapters/\s+# KM adapters \()\d+\s+registered(?=\))",
                lambda m, value=km_adapters_registered: f"{m.group(1)}{value} registered",
                0,
            ),
            (r"\d[\d,]*\s+SDK namespaces", f"{stats.ts_namespaces} SDK namespaces", 0),
        ]
    )

    replacements = {
        "README.md": [
            (
                r"orchestrates\s+\d[\d,]*(?:\+)?\s+agent types",
                f"orchestrates {agent_types_approx} agent types",
                0,
            ),
            (
                r"Knowledge Mound with\s+\d+\s+registered adapters",
                f"Knowledge Mound with {km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*(?:\+)?\s+paths", f"{api_paths_approx} paths", 0),
            (
                r"\d[\d,]*(?:\+)?\s+WebSocket event types",
                f"{ws_events_approx} WebSocket event types",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+Python modules", f"{modules_approx} Python modules", 0),
            (r"\(\d[\d,]*\s+namespaces\)", f"({stats.ts_namespaces} namespaces)", 0),
        ],
        "docs/EXTENDED_README.md": [
            (
                r"AGENT LAYER \(\d[\d,]*(?:\+)?\s+Agent Types\)",
                f"AGENT LAYER ({agent_types_approx} Agent Types)",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+agent types", f"{agent_types_approx} agent types", 0),
            (
                r"\d+\s+registered adapters",
                f"{km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*(?:\+)?\s+paths", f"{api_paths_approx} paths", 0),
            (
                r"\d[\d,]*(?:\+)?\s+WebSocket event types",
                f"{ws_events_approx} WebSocket event types",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+templates", f"{templates_approx} templates", 0),
            (r"\d[\d,]*(?:\+)?\s+Python modules", f"{modules_approx} Python modules", 0),
            (
                r"(\*\*Scale:\*\*[^\n]*?)\d[\d,]*(?:\+)?\s+tests",
                lambda m, value=tests_approx: f"{m.group(1)}{value} tests",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+test files", f"{test_files_approx} test files", 0),
            (
                r"\d[\d,]*\s+TypeScript SDK namespaces",
                f"{stats.ts_namespaces} TypeScript SDK namespaces",
                0,
            ),
            (
                r"\*\*Scale:\*\*[^\n]*canonical counts in \[METRICS\.md\]\(METRICS\.md\)",
                extended_readme_scale,
                0,
            ),
        ],
        "docs/COMMERCIAL_OVERVIEW.md": [
            (
                r"orchestrating\s+\d[\d,]*(?:\+)?\s+agent types",
                f"orchestrating {agent_types_approx} agent types",
                0,
            ),
            (
                r"\d+\s+registered adapters",
                f"{km_adapters_registered} registered adapters",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+API operations", f"{api_ops_approx} API operations", 0),
            (r"\d[\d,]*(?:\+)?\s+agent types", f"{agent_types_approx} agent types", 0),
        ],
        "docs/FEATURE_DISCOVERY.md": [
            (r"\d[\d,]*(?:\+)?\s+Python modules", f"{modules_approx} Python modules", 0),
            (
                r"(\*\*Total\*\*:[^\n]*?)\d[\d,]*(?:\+)?\s+tests",
                lambda m, value=tests_approx: f"{m.group(1)}{value} tests",
                0,
            ),
            (r"\d[\d,]*(?:\+)?\s+API operations", f"{api_ops_approx} API operations", 0),
            (
                r"\d[\d,]*(?:\+)?\s+pre-built templates",
                f"{templates_approx} pre-built templates",
                0,
            ),
            (
                r"Supported Providers \(\d[\d,]*(?:\+)?\s+agent types\)",
                f"Supported Providers ({agent_types_approx} agent types)",
                0,
            ),
        ],
        "docs/FEATURE_PARITY_MATRIX.md": [
            (r"\d[\d,]*(?:\+)?\s+operations", f"{api_ops_approx} operations", 0),
        ],
        "docs/WEBSOCKET_EVENTS.md": [
            (r"\(\d+ event types", f"({stats.ws_event_types} event types", 0),
        ],
        "docs/KNOWLEDGE_MOUND.md": [
            (
                r"\d+\s+registered adapters",
                f"{km_adapters_registered} registered adapters",
                0,
            ),
        ],
        "docs/DOCUMENTATION_HUB.md": [
            (
                r"\d+\s+registered adapters",
                f"{km_adapters_registered} registered adapters",
                0,
            ),
        ],
        "docs/CANONICAL_GOALS.md": [
            (
                r"(\| Python files under `aragora/` \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, value=exact_python_files: f"{m.group(1)}{value}",
                0,
            ),
            (
                r"(\| Python modules \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m,
                value=exact_top_level_modules: f"{m.group(1)}{value} top-level package directories",
                0,
            ),
            (
                r"(\| Lines of code under `aragora/` \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, value=exact_python_lines: f"{m.group(1)}{value}",
                0,
            ),
            (
                r"(\| Automated tests \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, value=exact_tests: f"{m.group(1)}{value} test functions",
                0,
            ),
            (
                r"(\| Test files \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, value=exact_test_files: f"{m.group(1)}{value}",
                0,
            ),
            (
                r"(\| API operations \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, ops=exact_api_ops, paths=exact_api_paths: (
                    f"{m.group(1)}{ops} across {paths} paths"
                ),
                0,
            ),
            (
                r"(\| API paths \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, value=exact_api_paths: f"{m.group(1)}{value}",
                0,
            ),
            (
                r"(\| Knowledge Mound adapters \| )[^|]+(?= \| `docs/METRICS\.md` \|)",
                lambda m, files=km_adapter_files, specs=km_adapters_registered: (
                    f"{m.group(1)}{files} adapter files / {specs} registered specs"
                ),
                0,
            ),
        ],
        "CLAUDE.md": claude_patterns,
        "docs-site/docs/contributing/claude.md": claude_patterns,
        "docs/architecture/system-overview.md": [
            (
                r"Agents Layer \(\d[\d,]*(?:\+)?\s+Agent Types\)",
                f"Agents Layer ({agent_types_approx} Agent Types)",
                0,
            ),
            (
                r"\d[\d,]*(?:\+)?\s+agent-type integrations",
                f"{agent_types_approx} agent-type integrations",
                0,
            ),
        ],
        "docs/landing/hero.md": [
            (
                r"\*\*\d[\d,]*(?:\+)?\s+agent types\*\*",
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
    metrics_doc = _metrics_doc_values()
    print("Live doc stats from repository scan:")
    print(f"- Python modules (aragora/, live scan): {stats.python_modules}")
    print(f"- Tests (def test_ across repo, live scan): {stats.test_count}")
    print(f"- Test files (tests/, live scan): {stats.test_files}")
    print(f"- API paths (docs/api/openapi.json): {stats.api_paths}")
    print(f"- API operations (docs/api/openapi.json): {stats.api_operations}")
    print(f"- WebSocket event types (live scan): {stats.ws_event_types}")
    print(f"- KM adapters registered (live scan): {stats.km_adapters_registered}")
    print(f"- Workflow templates (live scan): {stats.workflow_templates}")
    print(f"- TypeScript namespaces (live scan): {stats.ts_namespaces}")
    print(f"- Allowlisted agent types (settings allowlist): {stats.agent_types_allowlisted}")

    if metrics_doc:
        print("\nExact protected metrics from docs/METRICS.md:")
        print(f"- Python files: {_canonical_count(metrics_doc, 'python_files', 'missing')}")
        print(
            f"- Top-level modules: {_canonical_count(metrics_doc, 'top_level_modules', 'missing')}"
        )
        print(f"- Tests: {_canonical_count(metrics_doc, 'tests', 'missing')}")
        print(f"- Test files: {_canonical_count(metrics_doc, 'test_files', 'missing')}")
        print(f"- API paths: {_canonical_count(metrics_doc, 'api_paths', 'missing')}")
        print(f"- API operations: {_canonical_count(metrics_doc, 'api_operations', 'missing')}")
        print(
            "- Knowledge Mound adapter specs: "
            f"{_canonical_count(metrics_doc, 'adapter_specs', 'missing')}"
        )
        print(
            "- Knowledge Mound adapter files: "
            f"{_canonical_count(metrics_doc, 'adapter_files', 'missing')}"
        )

    if args.write:
        updated = patch_docs(stats, write=True)
        print(f"\nUpdated {updated} documentation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
