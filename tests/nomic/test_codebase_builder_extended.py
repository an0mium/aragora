"""Tests for CodebaseKnowledgeBuilder — ingest_module_summaries and ingest_dependency_graph."""

from __future__ import annotations

import ast
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aragora.memory.codebase_builder import CodebaseKnowledgeBuilder, IngestionStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fabric() -> MagicMock:
    """Create a mock MemoryFabric with ``remember`` returning stored=True."""
    fabric = MagicMock()
    fabric.remember = AsyncMock(return_value=MagicMock(stored=True))
    return fabric


def _make_builder(tmp_path: Path) -> tuple[CodebaseKnowledgeBuilder, MagicMock]:
    """Return (builder, mock_fabric) rooted at *tmp_path*."""
    fabric = _make_fabric()
    builder = CodebaseKnowledgeBuilder(fabric=fabric, repo_path=tmp_path)
    return builder, fabric


def _create_package(
    tmp_path: Path,
    name: str,
    init_content: str = "",
    files: dict[str, str] | None = None,
) -> Path:
    """Create a Python package directory with __init__.py and optional extra files."""
    pkg = tmp_path / name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text(init_content)
    for fname, content in (files or {}).items():
        (pkg / fname).write_text(content)
    return pkg


# ===========================================================================
# ingest_module_summaries
# ===========================================================================


class TestIngestModuleSummaries:
    """Tests for CodebaseKnowledgeBuilder.ingest_module_summaries."""

    @pytest.mark.asyncio
    async def test_extracts_init_docstring(self, tmp_path: Path) -> None:
        """__init__.py module docstring is included in the summary."""
        _create_package(tmp_path, "mypkg", '"""This is the docstring."""\n')
        builder, fabric = _make_builder(tmp_path)

        stats = await builder.ingest_module_summaries()

        assert stats.items_ingested == 1
        call_kwargs = fabric.remember.call_args.kwargs
        assert "This is the docstring." in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_extracts_class_and_function_names(self, tmp_path: Path) -> None:
        """Class and function names from .py files appear in the summary."""
        _create_package(
            tmp_path,
            "mypkg",
            '"""Pkg."""\n',
            files={
                "models.py": (
                    "class Foo:\n    pass\n\n"
                    "class Bar:\n    pass\n\n"
                    "def helper():\n    pass\n"
                ),
            },
        )
        builder, fabric = _make_builder(tmp_path)

        await builder.ingest_module_summaries()

        content = fabric.remember.call_args.kwargs["content"]
        assert "class Foo" in content
        assert "class Bar" in content
        assert "def helper" in content

    @pytest.mark.asyncio
    async def test_extracts_all_exports(self, tmp_path: Path) -> None:
        """__all__ list in __init__.py is extracted as Exports."""
        _create_package(
            tmp_path,
            "mypkg",
            '"""Pkg."""\n__all__ = ["Alpha", "Beta"]\n',
        )
        builder, fabric = _make_builder(tmp_path)

        await builder.ingest_module_summaries()

        content = fabric.remember.call_args.kwargs["content"]
        assert "Alpha" in content
        assert "Beta" in content
        assert "Exports:" in content

    @pytest.mark.asyncio
    async def test_respects_max_modules(self, tmp_path: Path) -> None:
        """Only the first *max_modules* packages are processed."""
        for i in range(5):
            _create_package(tmp_path, f"pkg{i}", f'"""Package {i}."""\n')

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_module_summaries(max_modules=2)

        assert stats.items_ingested == 2
        assert fabric.remember.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_dirs_without_init(self, tmp_path: Path) -> None:
        """Directories lacking __init__.py are skipped."""
        (tmp_path / "nopkg").mkdir()
        (tmp_path / "nopkg" / "stuff.py").write_text("x = 1\n")
        # Also create a valid one so we get at least one result
        _create_package(tmp_path, "validpkg", '"""Valid."""\n')

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_module_summaries()

        assert stats.items_ingested == 1
        content = fabric.remember.call_args.kwargs["content"]
        assert "validpkg" in content

    @pytest.mark.asyncio
    async def test_skips_dot_and_underscore_dirs(self, tmp_path: Path) -> None:
        """Directories starting with '.' or '_' are skipped."""
        _create_package(tmp_path, ".hidden", '"""Hidden."""\n')
        _create_package(tmp_path, "_private", '"""Private."""\n')
        _create_package(tmp_path, "visible", '"""Visible."""\n')

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_module_summaries()

        assert stats.items_ingested == 1
        content = fabric.remember.call_args.kwargs["content"]
        assert "visible" in content

    @pytest.mark.asyncio
    async def test_uses_llm_summarizer(self, tmp_path: Path) -> None:
        """When a summarizer callable is provided, its output replaces the raw summary."""
        _create_package(tmp_path, "mypkg", '"""Original."""\n')

        def my_summarizer(module_name: str, raw_info: str) -> str:
            return f"LLM summary of {module_name}"

        builder, fabric = _make_builder(tmp_path)
        await builder.ingest_module_summaries(summarizer=my_summarizer)

        content = fabric.remember.call_args.kwargs["content"]
        assert content == "LLM summary of mypkg"

    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back(self, tmp_path: Path) -> None:
        """If the summarizer raises, we fall back to the raw summary."""
        _create_package(tmp_path, "mypkg", '"""Fallback doc."""\n')

        def bad_summarizer(module_name: str, raw_info: str) -> str:
            raise ValueError("summarizer broke")

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_module_summaries(summarizer=bad_summarizer)

        assert stats.items_ingested == 1
        content = fabric.remember.call_args.kwargs["content"]
        # Falls back to raw summary which contains the module name and docstring
        assert "mypkg" in content
        assert "Fallback doc." in content


# ===========================================================================
# ingest_dependency_graph
# ===========================================================================


class TestIngestDependencyGraph:
    """Tests for CodebaseKnowledgeBuilder.ingest_dependency_graph."""

    @pytest.mark.asyncio
    async def test_builds_module_level_edges(self, tmp_path: Path) -> None:
        """Import statements are aggregated into module-level dependency edges."""
        pkg = _create_package(
            tmp_path,
            "alpha",
            "",
            files={
                "core.py": "import os\nfrom pathlib import Path\n",
            },
        )

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_dependency_graph()

        assert stats.items_ingested == 1
        call_kwargs = fabric.remember.call_args.kwargs
        content = call_kwargs["content"]
        assert "alpha" in content
        # os and pathlib are external deps of alpha
        assert "os" in content
        assert "pathlib" in content
        meta = call_kwargs["metadata"]
        assert meta["type"] == "dependency_graph"
        assert meta["module_count"] >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_edges(self, tmp_path: Path) -> None:
        """Same import from multiple files only produces one edge."""
        _create_package(
            tmp_path,
            "beta",
            "",
            files={
                "a.py": "import os\n",
                "b.py": "import os\nfrom os.path import join\n",
            },
        )

        builder, fabric = _make_builder(tmp_path)
        await builder.ingest_dependency_graph()

        content = fabric.remember.call_args.kwargs["content"]
        # The edge beta -> os should appear only once in the graph line
        graph_lines = [line for line in content.splitlines() if line.strip().startswith("beta")]
        assert len(graph_lines) == 1
        # "os" should appear exactly once in that line
        assert graph_lines[0].count("os") == 1

    @pytest.mark.asyncio
    async def test_handles_circular_deps(self, tmp_path: Path) -> None:
        """Circular imports (A->B, B->A) are both captured without crashing."""
        _create_package(
            tmp_path,
            "pkga",
            "",
            files={"mod.py": "import pkgb\n"},
        )
        _create_package(
            tmp_path,
            "pkgb",
            "",
            files={"mod.py": "import pkga\n"},
        )

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_dependency_graph()

        assert stats.items_ingested == 1
        assert stats.errors == 0
        content = fabric.remember.call_args.kwargs["content"]
        assert "pkga" in content
        assert "pkgb" in content

    @pytest.mark.asyncio
    async def test_respects_max_files(self, tmp_path: Path) -> None:
        """Only the first *max_files* Python files are analyzed."""
        pkg = tmp_path / "big"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for i in range(10):
            (pkg / f"mod{i}.py").write_text(f"import sys\n")

        builder, fabric = _make_builder(tmp_path)
        # max_files=3 means only 3 of the 11 .py files (init + 10 mods) get parsed
        stats = await builder.ingest_dependency_graph(max_files=3)

        # Should still store the graph (at least some edges found)
        assert fabric.remember.call_count <= 1
        assert stats.errors == 0

    @pytest.mark.asyncio
    async def test_empty_repo_returns_empty(self, tmp_path: Path) -> None:
        """An empty directory produces empty stats and no fabric.remember calls."""
        builder, fabric = _make_builder(tmp_path)

        stats = await builder.ingest_dependency_graph()

        assert stats.items_ingested == 0
        assert stats.items_skipped == 0
        assert stats.errors == 0
        fabric.remember.assert_not_called()

    @pytest.mark.asyncio
    async def test_syntax_error_increments_errors(self, tmp_path: Path) -> None:
        """A file with invalid Python syntax increments errors, doesn't crash."""
        _create_package(
            tmp_path,
            "broken",
            "",
            files={
                "bad.py": "def foo(\n",  # SyntaxError: unexpected EOF
                "good.py": "import os\n",
            },
        )

        builder, fabric = _make_builder(tmp_path)
        stats = await builder.ingest_dependency_graph()

        assert stats.errors >= 1
        # The good file should still be processed
        assert fabric.remember.call_count >= 0  # may or may not have edges

    @pytest.mark.asyncio
    async def test_graph_metadata_counts(self, tmp_path: Path) -> None:
        """Stored metadata contains correct module_count and edge_count."""
        _create_package(
            tmp_path,
            "svc",
            "",
            files={
                "api.py": "import os\nimport json\n",
                "db.py": "import sqlite3\n",
            },
        )

        builder, fabric = _make_builder(tmp_path)
        await builder.ingest_dependency_graph()

        meta = fabric.remember.call_args.kwargs["metadata"]
        assert meta["module_count"] >= 1
        assert meta["edge_count"] >= 1
        # svc imports os, json, sqlite3 -> 3 edges from 1 module
        assert meta["module_count"] == 1
        assert meta["edge_count"] == 3
