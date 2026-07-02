from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
                "| Top-level modules under aragora/ | `144` | `aragora/` | `cmd` |",
                "| Test files (test_*.py under tests/) | `5402` | `tests/` | `cmd` |",
                "| Test functions (class + module level) | `222659` | `tests/` | `cmd` |",
                "| OpenAPI paths | `2870` | `docs/api/openapi.json` | `cmd` |",
                "| OpenAPI operations (HTTP verbs) | `3297` | `docs/api/openapi.json` | `cmd` |",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "ROOT", root)

    values = mod._metrics_doc_values()

    assert values["python_files"].value == 4219
    assert values["top_level_modules"].value == 144
    assert values["test_files"].value == 5402
    assert values["tests"].value == 222659
    assert values["api_paths"].value == 2870
    assert values["api_operations"].value == 3297


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
                "| Top-level modules under aragora/ | `144` | `aragora/` | `cmd` |",
                "| Test files (test_*.py under tests/) | `5402` | `tests/` | `cmd` |",
                "| Test functions (class + module level) | `222659` | `tests/` | `cmd` |",
                "| OpenAPI paths | `2870` | `docs/api/openapi.json` | `cmd` |",
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
                "| Python modules | 135 top-level package directories | `docs/METRICS.md` |",
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
    (root / "CLAUDE.md").write_text(
        "\n".join(
            [
                "**Codebase Scale:** 4,069 tracked Python files | 135 top-level modules | 216,000+ test functions | 5,078 test files | 3,386 API operations across 2,928 paths | canonical counts in `docs/METRICS.md`",
                "**Test Suite:** 216,000+ test functions across 5,078 test files (canonical counts in `docs/METRICS.md`)",
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

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Workflow Engine — DAG automation with 50+ templates" in readme
    assert "**The Nomic Loop (✅, 233+ tests)." in readme
    assert "3,297 API operations across 2,870 paths" in readme
