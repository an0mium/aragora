"""Tests for MCP codebase tools."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symbol(name="Arena", kind="class", line=50, docstring="The Arena class", source="class Arena: ..."):
    sym = MagicMock()
    sym.name = name
    sym.kind = kind
    sym.line = line
    sym.docstring = docstring
    sym.source = source
    return sym


def _make_crawled_file(relative_path="aragora/debate/arena.py", symbols=None, line_count=200):
    f = MagicMock()
    f.relative_path = relative_path
    f.path = relative_path
    f.symbols = symbols or []
    f.line_count = line_count
    return f


def _make_crawl_result(files=None, dependency_graph=None, file_type_counts=None, symbol_counts=None):
    cr = MagicMock()
    cr.files = files or []
    cr.total_files = len(files or [])
    cr.total_lines = sum(getattr(f, "line_count", 0) for f in (files or []))
    cr.dependency_graph = dependency_graph or {}
    cr.file_type_counts = file_type_counts or {".py": 10}
    cr.symbol_counts = symbol_counts or {"class": 5, "function": 20}
    return cr


# ---------------------------------------------------------------------------
# search_codebase_tool
# ---------------------------------------------------------------------------


class TestSearchCodebaseTool:
    @pytest.mark.asyncio
    async def test_search_by_symbol(self):
        from aragora.mcp.tools_module.codebase import search_codebase_tool

        sym = _make_symbol("Arena", "class")
        files = [_make_crawled_file(symbols=[sym])]
        crawl = _make_crawl_result(files=files)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await search_codebase_tool("Arena")
            assert result["count"] >= 1
            assert result["results"][0]["symbol"] == "Arena"

    @pytest.mark.asyncio
    async def test_search_by_path(self):
        from aragora.mcp.tools_module.codebase import search_codebase_tool

        files = [_make_crawled_file(relative_path="aragora/debate/arena.py")]
        crawl = _make_crawl_result(files=files)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await search_codebase_tool("arena")
            assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_with_file_type_filter(self):
        from aragora.mcp.tools_module.codebase import search_codebase_tool

        sym = _make_symbol("setup", "function")
        py_file = _make_crawled_file("setup.py", symbols=[sym])
        ts_file = _make_crawled_file("client.ts", symbols=[_make_symbol("setup", "function")])
        crawl = _make_crawl_result(files=[py_file, ts_file])

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await search_codebase_tool("setup", file_types="py")
            # Only .py file should match
            for r in result["results"]:
                if r.get("match_type") == "symbol":
                    assert r["file"].endswith(".py")

    @pytest.mark.asyncio
    async def test_search_import_error(self):
        from aragora.mcp.tools_module.codebase import search_codebase_tool

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            side_effect=ImportError("not available"),
        ):
            result = await search_codebase_tool("anything")
            assert "error" in result


# ---------------------------------------------------------------------------
# get_symbol_tool
# ---------------------------------------------------------------------------


class TestGetSymbolTool:
    @pytest.mark.asyncio
    async def test_find_symbol(self):
        from aragora.mcp.tools_module.codebase import get_symbol_tool

        sym = _make_symbol("DebateProtocol", "class", 132)
        files = [_make_crawled_file(symbols=[sym])]
        crawl = _make_crawl_result(files=files)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_symbol_tool("DebateProtocol")
            assert result["count"] == 1
            assert result["matches"][0]["name"] == "DebateProtocol"

    @pytest.mark.asyncio
    async def test_symbol_not_found(self):
        from aragora.mcp.tools_module.codebase import get_symbol_tool

        crawl = _make_crawl_result(files=[_make_crawled_file()])

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_symbol_tool("NonExistent")
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_symbol_with_source(self):
        from aragora.mcp.tools_module.codebase import get_symbol_tool

        sym = _make_symbol("Arena", "class", source="class Arena:\n    pass")
        files = [_make_crawled_file(symbols=[sym])]
        crawl = _make_crawl_result(files=files)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_symbol_tool("Arena", include_source=True)
            assert "source" in result["matches"][0]


# ---------------------------------------------------------------------------
# get_dependencies_tool
# ---------------------------------------------------------------------------


class TestGetDependenciesTool:
    @pytest.mark.asyncio
    async def test_outgoing_dependencies(self):
        from aragora.mcp.tools_module.codebase import get_dependencies_tool

        dep_graph = {"arena.py": ["protocol.py", "core.py"]}
        crawl = _make_crawl_result(dependency_graph=dep_graph)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_dependencies_tool("arena.py")
            assert result["direction"] == "outgoing"
            assert result["count"] == 2
            assert "protocol.py" in result["dependencies"]

    @pytest.mark.asyncio
    async def test_incoming_dependencies(self):
        from aragora.mcp.tools_module.codebase import get_dependencies_tool

        dep_graph = {
            "orchestrator.py": ["arena.py"],
            "phases/proposal.py": ["arena.py"],
        }
        crawl = _make_crawl_result(dependency_graph=dep_graph)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_dependencies_tool("arena.py", direction="incoming")
            assert result["direction"] == "incoming"
            assert result["count"] == 2


# ---------------------------------------------------------------------------
# get_codebase_structure_tool
# ---------------------------------------------------------------------------


class TestGetCodebaseStructureTool:
    @pytest.mark.asyncio
    async def test_structure_overview(self):
        from aragora.mcp.tools_module.codebase import get_codebase_structure_tool

        files = [
            _make_crawled_file("aragora/debate/arena.py", symbols=[_make_symbol()]),
            _make_crawled_file("aragora/debate/protocol.py"),
            _make_crawled_file("tests/test_arena.py"),
        ]
        crawl = _make_crawl_result(files=files)

        mock_crawler = AsyncMock()
        mock_crawler.crawl = AsyncMock(return_value=crawl)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            return_value=mock_crawler,
        ):
            result = await get_codebase_structure_tool()
            assert result["total_files"] == 3
            assert "structure" in result
            assert "aragora" in result["structure"]
