from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "doc_stats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("doc_stats_under_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metrics_doc_values_parse_exact_generated_metrics(tmp_path, monkeypatch):
    mod = _load_module()
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "METRICS.md").write_text(
        "\n".join(
            [
                "| Metric | Value | Source | Command |",
                "|---|---|---|---|",
                "| Python files under aragora/ | `4219` | `aragora/` | `cmd` |",
                "| Python lines of code under aragora/ | `1972052` | `aragora/` | `cmd` |",
                "| Top-level modules under aragora/ | `144` | `aragora/` | `cmd` |",
                "| Test files (test_*.py under tests/) | `5402` | `tests/` | `cmd` |",
                "| Test functions (class + module level) | `222659` | `tests/` | `cmd` |",
                "| OpenAPI paths | `2870` | `docs/api/openapi.json` | `cmd` |",
                "| OpenAPI operations (HTTP verbs) | `3297` | `docs/api/openapi.json` | `cmd` |",
                "| Allowlisted agent types | `35` | `aragora/config/settings.py` | `cmd` |",
                "| Knowledge Mound adapter specs | `41` | `aragora/knowledge/mound/adapters/factory.py` | `cmd` |",
                "| Knowledge Mound adapter files | `46` | `aragora/knowledge/mound/adapters/` | `cmd` |",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "ROOT", root)

    values = mod._metrics_doc_values()

    assert values["python_files"].value == 4219
    assert values["python_lines"].value == 1972052
    assert values["top_level_modules"].value == 144
    assert values["test_files"].value == 5402
    assert values["tests"].value == 222659
    assert values["api_paths"].value == 2870
    assert values["api_operations"].value == 3297
    assert values["allowlisted_agent_types"].value == 35
    assert values["adapter_specs"].value == 41
    assert values["adapter_files"].value == 46


def test_patch_docs_uses_metrics_doc_for_claude_and_preserves_readme_scope(tmp_path, monkeypatch):
    mod = _load_module()
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "METRICS.md").write_text(
        "\n".join(
            [
                "| Metric | Value | Source | Command |",
                "|---|---|---|---|",
                "| Python files under aragora/ | `4219` | `aragora/` | `cmd` |",
                "| Python lines of code under aragora/ | `1972052` | `aragora/` | `cmd` |",
                "| Top-level modules under aragora/ | `144` | `aragora/` | `cmd` |",
                "| Test files (test_*.py under tests/) | `5402` | `tests/` | `cmd` |",
                "| Test functions (class + module level) | `222659` | `tests/` | `cmd` |",
                "| OpenAPI paths | `2870` | `docs/api/openapi.json` | `cmd` |",
                "| OpenAPI operations (HTTP verbs) | `3297` | `docs/api/openapi.json` | `cmd` |",
                "| Allowlisted agent types | `35` | `aragora/config/settings.py` | `cmd` |",
                "| Knowledge Mound adapter specs | `41` | `aragora/knowledge/mound/adapters/factory.py` | `cmd` |",
                "| Knowledge Mound adapter files | `46` | `aragora/knowledge/mound/adapters/` | `cmd` |",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "CANONICAL_GOALS.md").write_text(
        "\n".join(
            [
                "| Metric | Value | Source |",
                "|--------|-------|--------|",
                "| Python files under `aragora/` | 4,069 | `docs/METRICS.md` |",
                "| Python modules | 135 top-level package directories | `docs/METRICS.md` |",
                "| Lines of code under `aragora/` | 1,915,420 | `docs/METRICS.md` |",
                "| Automated tests | 216,016 test functions | `docs/METRICS.md` |",
                "| Test files | 5,078 | `docs/METRICS.md` |",
                "| API operations | 3,297 across 2,870 paths | `docs/METRICS.md` |",
                "| API paths | 2,870 | `docs/METRICS.md` |",
                "| Knowledge Mound adapters | 46 adapter files / 41 registered specs | `docs/METRICS.md` |",
                "| Agent types | 43 across 6+ LLM providers | agent registry |",
                "| Workflow templates | 50+ across 6 categories | template registry |",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "EXTENDED_README.md").write_text(
        "**Scale:** 4,069 tracked Python files | 135 top-level modules | "
        "216,000+ test functions across 5,078 test files | canonical counts in [METRICS.md](METRICS.md)",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text(
        "\n".join(
            [
                "**Codebase Scale:** 4,069 tracked Python files | 135 top-level modules | 216,000+ test functions | 5,078 test files | 3,386 API operations across 2,928 paths | canonical counts in `docs/METRICS.md`",
                "**Test Suite:** 216,000+ test functions across 5,078 test files (canonical counts in `docs/METRICS.md`)",
                "│       └── adapters/       # KM adapters (42 registered)",
            ]
        ),
        encoding="utf-8",
    )
    docs_site_contributing = root / "docs-site" / "docs" / "contributing"
    docs_site_contributing.mkdir(parents=True)
    (docs_site_contributing / "claude.md").write_text(
        "\n".join(
            [
                "**Codebase Scale:** 4,069 tracked Python files | 135 top-level modules | 216,000+ test functions | 5,078 test files | 3,386 API operations across 2,928 paths | canonical counts in `docs/METRICS.md`",
                "**Test Suite:** 216,000+ test functions across 5,078 test files (canonical counts in `docs/METRICS.md`)",
                "│   ├── unified_server.py   # Main server (3,386 API operations)",
                "│       └── adapters/       # KM adapters (42 registered)",
            ]
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "\n".join(
            [
                "loop (✅); Workflow Engine — DAG automation with 50+ templates (✅); Prompt Engine —",
                "**The Nomic Loop (✅, 233+ tests).** A five-phase autonomous self-improvement cycle:",
                "> 3,386 API operations across 2,928 paths",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "ROOT", root)
    stats = mod.Stats(
        python_modules=4256,
        test_count=163898,
        test_files=5916,
        api_paths=2870,
        api_operations=3297,
        ws_event_types=272,
        km_adapters_registered=0,
        workflow_templates=62,
        ts_namespaces=191,
        agent_types_allowlisted=35,
    )

    mod.patch_docs(stats, write=True)

    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "4,219 tracked Python files | 144 top-level modules" in claude
    assert "222,659 test functions | 5,402 test files" in claude
    assert "3,297 API operations across 2,870 paths" in claude
    assert "**Test Suite:** 222,659 test functions across 5,402 test files" in claude
    assert "│       └── adapters/       # KM adapters (41 registered)" in claude

    docs_site_claude = (docs_site_contributing / "claude.md").read_text(encoding="utf-8")
    assert "222,659 test functions | 5,402 test files" in docs_site_claude
    assert "│   ├── unified_server.py   # Main server (3,297 API operations)" in docs_site_claude
    assert "│       └── adapters/       # KM adapters (41 registered)" in docs_site_claude

    canonical_goals = (docs / "CANONICAL_GOALS.md").read_text(encoding="utf-8")
    assert "| Python files under `aragora/` | 4,219 | `docs/METRICS.md` |" in canonical_goals
    assert "| Lines of code under `aragora/` | 1,972,052 | `docs/METRICS.md` |" in canonical_goals
    assert "| Automated tests | 222,659 test functions | `docs/METRICS.md` |" in canonical_goals
    assert "| Test files | 5,402 | `docs/METRICS.md` |" in canonical_goals
    assert (
        "| Knowledge Mound adapters | 46 adapter files / 41 registered specs | `docs/METRICS.md` |"
        in canonical_goals
    )

    extended_readme = (docs / "EXTENDED_README.md").read_text(encoding="utf-8")
    assert "4,219 tracked Python files | 144 top-level modules" in extended_readme
    assert "222,659 test functions across 5,402 test files" in extended_readme

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Workflow Engine — DAG automation with 50+ templates" in readme
    assert "**The Nomic Loop (✅, 233+ tests)." in readme
    assert "3,297 API operations across 2,870 paths" in readme


def test_patch_docs_refuses_protected_writes_when_metrics_doc_is_partial(tmp_path, monkeypatch):
    mod = _load_module()
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "METRICS.md").write_text(
        "\n".join(
            [
                "| Metric | Value | Source | Command |",
                "|---|---|---|---|",
                "| Python files under aragora/ | `4219` | `aragora/` | `cmd` |",
                "| OpenAPI operations (HTTP verbs) | `3297` | `docs/api/openapi.json` | `cmd` |",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "CANONICAL_GOALS.md").write_text(
        "\n".join(
            [
                "| Metric | Value | Source |",
                "|--------|-------|--------|",
                "| API operations | 3,297 across 2,870 paths | `docs/METRICS.md` |",
                "| API paths | 2,870 | `docs/METRICS.md` |",
                "| Knowledge Mound adapters | 46 adapter files / 41 registered specs | `docs/METRICS.md` |",
            ]
        ),
        encoding="utf-8",
    )
    old_codebase_scale = (
        "**Codebase Scale:** 4,069 tracked Python files | 135 top-level modules | "
        "216,000+ test functions | 5,078 test files | 3,386 API operations across "
        "2,928 paths | canonical counts in `docs/METRICS.md`"
    )
    old_test_suite = (
        "**Test Suite:** 216,000+ test functions across 5,078 test files "
        "(canonical counts in `docs/METRICS.md`)"
    )
    (root / "CLAUDE.md").write_text(
        "\n".join(
            [
                old_codebase_scale,
                "│       └── adapters/       # KM adapters (42 registered)",
                "│   ├── unified_server.py   # Main server (3,386 API operations)",
                old_test_suite,
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "ROOT", root)
    stats = mod.Stats(
        python_modules=4256,
        test_count=163898,
        test_files=5916,
        api_paths=2870,
        api_operations=3297,
        ws_event_types=272,
        km_adapters_registered=0,
        workflow_templates=62,
        ts_namespaces=191,
        agent_types_allowlisted=35,
    )

    docs_site_contributing = root / "docs-site" / "docs" / "contributing"
    docs_site_contributing.mkdir(parents=True)
    (docs_site_contributing / "claude.md").write_text(old_codebase_scale, encoding="utf-8")

    with pytest.raises(RuntimeError, match="docs/METRICS.md is missing rows"):
        mod.patch_docs(stats, write=True)

    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert old_codebase_scale in claude
    assert old_test_suite in claude
    assert "3,386 API operations" in claude
    docs_site_claude = (docs_site_contributing / "claude.md").read_text(encoding="utf-8")
    assert docs_site_claude == old_codebase_scale
