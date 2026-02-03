"""Tests for LocalDocsConnector - init, filtering, search, fetch."""

import os
from pathlib import Path

import pytest

from aragora.connectors.local_docs import LocalDocsConnector


class TestLocalDocsInit:
    """Initialization and configuration."""

    def test_default_init(self, tmp_path):
        connector = LocalDocsConnector(root_path=str(tmp_path))
        assert connector.root_path == tmp_path.resolve()
        assert connector.file_types == "all"
        assert connector.max_file_size == 1_000_000

    def test_custom_file_types(self, tmp_path):
        connector = LocalDocsConnector(root_path=str(tmp_path), file_types="docs")
        assert connector.file_types == "docs"

    def test_name_and_source_type(self, tmp_path):
        connector = LocalDocsConnector(root_path=str(tmp_path))
        assert connector.name == "Local Documentation"
        assert connector.source_type.value  # Should be a valid SourceType


class TestFileFiltering:
    """Tests for _should_search_file."""

    def setup_method(self, tmp_path=None):
        self._tmp = None

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self._tmp = tmp_path
        self.connector = LocalDocsConnector(root_path=str(tmp_path))

    def test_skips_hidden_files(self):
        hidden = self._tmp / ".hidden" / "file.py"
        hidden.parent.mkdir(parents=True)
        hidden.write_text("content")
        assert not self.connector._should_search_file(hidden)

    def test_skips_node_modules(self):
        nm = self._tmp / "node_modules" / "pkg" / "index.js"
        nm.parent.mkdir(parents=True)
        nm.write_text("content")
        assert not self.connector._should_search_file(nm)

    def test_skips_pycache(self):
        pc = self._tmp / "__pycache__" / "mod.pyc"
        pc.parent.mkdir(parents=True)
        pc.write_text("content")
        assert not self.connector._should_search_file(pc)

    def test_accepts_normal_file(self):
        f = self._tmp / "readme.md"
        f.write_text("hello")
        assert self.connector._should_search_file(f)

    def test_rejects_oversized_file(self):
        connector = LocalDocsConnector(root_path=str(self._tmp), max_file_size=10)
        f = self._tmp / "big.txt"
        f.write_text("x" * 100)
        assert not connector._should_search_file(f)

    def test_extension_filter_docs(self):
        connector = LocalDocsConnector(root_path=str(self._tmp), file_types="docs")
        md = self._tmp / "readme.md"
        md.write_text("hello")
        py = self._tmp / "main.py"
        py.write_text("print()")
        assert connector._should_search_file(md)
        assert not connector._should_search_file(py)

    def test_symlink_rejected(self):
        target = self._tmp / "real.txt"
        target.write_text("real content")
        link = self._tmp / "link.txt"
        link.symlink_to(target)
        assert not self.connector._should_search_file(link)


class TestSearch:
    """Tests for search method."""

    @pytest.mark.asyncio
    async def test_search_finds_content(self, tmp_path):
        (tmp_path / "doc.md").write_text("The quick brown fox jumps over the lazy dog")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        results = await connector.search("brown fox")
        assert len(results) == 1
        assert "brown fox" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"doc{i}.txt").write_text(f"needle in haystack {i}")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        results = await connector.search("needle", limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_empty_results(self, tmp_path):
        (tmp_path / "doc.md").write_text("nothing relevant here")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        results = await connector.search("xyznonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_regex_search(self, tmp_path):
        (tmp_path / "code.py").write_text("def calculate_total(items):\n    pass")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        results = await connector.search(r"def \w+_total", regex=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dangerous_regex_falls_back(self, tmp_path):
        """ReDoS patterns should fall back to literal search."""
        (tmp_path / "doc.txt").write_text("some text")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        # This pattern triggers ReDoS detection
        results = await connector.search(r"(a+)+", regex=True)
        # Should not raise, just return empty or literal match
        assert isinstance(results, list)


class TestFetch:
    """Tests for fetch method."""

    @pytest.mark.asyncio
    async def test_fetch_by_path(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Hello World")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        result = await connector.fetch("readme.md")
        assert result is not None
        assert "Hello World" in result.content

    @pytest.mark.asyncio
    async def test_fetch_nonexistent(self, tmp_path):
        connector = LocalDocsConnector(root_path=str(tmp_path))
        result = await connector.fetch("does_not_exist.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_blocks_path_traversal(self, tmp_path):
        """Directory traversal should be blocked."""
        connector = LocalDocsConnector(root_path=str(tmp_path))
        result = await connector.fetch("../../etc/passwd")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        connector = LocalDocsConnector(root_path=str(tmp_path))
        files = await connector.list_files("*.md")
        assert "a.md" in files
        assert "b.txt" not in files
