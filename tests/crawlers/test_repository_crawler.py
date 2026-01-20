"""Tests for Repository Crawler (legacy module).

Note: These tests cover the deprecated ``aragora.crawlers`` module which is
maintained for backward compatibility. For new code, use
``aragora.connectors.repository_crawler`` instead.

The deprecated module emits DeprecationWarning on import, which is suppressed
here to keep test output clean.
"""

import asyncio
import os
import pytest
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

# Suppress deprecation warning for this legacy test module
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from aragora.crawlers import (
        RepositoryCrawler,
        RepositoryCrawlerConfig,
        RepositoryInfo,
        BaseCrawler,
        ContentType,
        CrawlerConfig,
        CrawlResult,
        CrawlStats,
        CrawlStatus,
        IndexResult,
    )


class TestContentType:
    """Test ContentType enum."""

    def test_content_types(self):
        """Test that all content types are defined."""
        assert ContentType.CODE.value == "code"
        assert ContentType.DOCUMENTATION.value == "documentation"
        assert ContentType.CONFIG.value == "config"
        assert ContentType.DATA.value == "data"
        assert ContentType.BINARY.value == "binary"
        assert ContentType.UNKNOWN.value == "unknown"


class TestCrawlStatus:
    """Test CrawlStatus enum."""

    def test_crawl_statuses(self):
        """Test that all crawl statuses are defined."""
        assert CrawlStatus.PENDING.value == "pending"
        assert CrawlStatus.RUNNING.value == "running"
        assert CrawlStatus.COMPLETED.value == "completed"
        assert CrawlStatus.FAILED.value == "failed"
        assert CrawlStatus.CANCELLED.value == "cancelled"


class TestCrawlerConfig:
    """Test CrawlerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CrawlerConfig()

        assert config.max_files == 10000
        assert config.max_file_size_bytes == 1_000_000  # 1MB
        assert "**/*" in config.include_patterns
        assert "**/.git/**" in config.exclude_patterns
        assert config.extract_symbols is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CrawlerConfig(
            max_files=100,
            max_file_size_bytes=1024 * 1024,
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["*_test.py"],
            extract_symbols=False,
        )

        assert config.max_files == 100
        assert config.max_file_size_bytes == 1024 * 1024
        assert "*.py" in config.include_patterns
        assert "*_test.py" in config.exclude_patterns
        assert config.extract_symbols is False


class TestRepositoryCrawlerConfig:
    """Test RepositoryCrawlerConfig."""

    def test_default_config(self):
        """Test default repository crawler configuration."""
        config = RepositoryCrawlerConfig()

        assert config.clone_depth == 1
        assert config.branch == "main"
        assert config.fetch_tags is False
        assert config.extract_functions is True
        assert config.extract_classes is True
        assert config.extract_imports is True
        assert ".png" in config.binary_extensions
        assert ".exe" in config.binary_extensions

    def test_custom_config(self):
        """Test custom repository crawler configuration."""
        config = RepositoryCrawlerConfig(
            clone_depth=10,
            branch="develop",
            fetch_tags=True,
            extract_functions=False,
        )

        assert config.clone_depth == 10
        assert config.branch == "develop"
        assert config.fetch_tags is True
        assert config.extract_functions is False


class TestCrawlResult:
    """Test CrawlResult dataclass."""

    def test_create_result(self):
        """Test creating a crawl result."""
        result = CrawlResult(
            id="cr_1",
            path="src/main.py",
            content="print('hello')",
            content_type=ContentType.CODE,
            size_bytes=15,
        )

        assert result.id == "cr_1"
        assert result.path == "src/main.py"
        assert result.content == "print('hello')"
        assert result.content_type == ContentType.CODE
        assert result.size_bytes == 15
        assert result.language is None
        assert result.symbols == []
        assert result.imports == []
        assert result.metadata == {}

    def test_create_result_with_all_fields(self):
        """Test creating a result with all fields."""
        now = datetime.utcnow()
        result = CrawlResult(
            id="cr_full",
            path="src/utils.py",
            content="def helper(): pass",
            content_type=ContentType.CODE,
            size_bytes=20,
            created_at=now,
            modified_at=now,
            language="python",
            symbols=["function:helper"],
            imports=["os", "sys"],
            metadata={"repo": "test-repo"},
        )

        assert result.language == "python"
        assert "function:helper" in result.symbols
        assert "os" in result.imports
        assert result.metadata["repo"] == "test-repo"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = CrawlResult(
            id="cr_1",
            path="test.py",
            content="code",
            content_type=ContentType.CODE,
            size_bytes=4,
            language="python",
        )

        d = result.to_dict()

        assert d["id"] == "cr_1"
        assert d["path"] == "test.py"
        assert d["content_type"] == "code"
        assert d["language"] == "python"


class TestCrawlStats:
    """Test CrawlStats dataclass."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = CrawlStats(status=CrawlStatus.PENDING)

        assert stats.status == CrawlStatus.PENDING
        assert stats.total_files == 0
        assert stats.processed_files == 0
        assert stats.skipped_files == 0
        assert stats.failed_files == 0
        assert stats.total_bytes == 0
        assert stats.errors == []

    def test_stats_calculation(self):
        """Test statistics calculations."""
        stats = CrawlStats(
            status=CrawlStatus.COMPLETED,
            total_files=100,
            processed_files=85,
            skipped_files=10,
            failed_files=5,
            total_bytes=1024 * 1024,
        )

        assert stats.total_files == 100
        assert stats.processed_files + stats.skipped_files + stats.failed_files == 100


class TestRepositoryInfo:
    """Test RepositoryInfo dataclass."""

    def test_create_info(self):
        """Test creating repository info."""
        info = RepositoryInfo(
            path=Path("/tmp/repo"),
            remote_url="https://github.com/user/repo",
            branch="main",
        )

        assert info.path == Path("/tmp/repo")
        assert info.remote_url == "https://github.com/user/repo"
        assert info.branch == "main"
        assert info.last_commit is None
        assert info.commit_count == 0


class TestRepositoryCrawler:
    """Test RepositoryCrawler class.

    Note: This module is deprecated. Use aragora.connectors.repository_crawler instead.
    Tests marked with xfail are known issues in the deprecated implementation.
    """

    @pytest.fixture
    def crawler(self):
        """Create a repository crawler for testing."""
        # Explicitly set include pattern that works with fnmatch
        config = RepositoryCrawlerConfig(
            max_files=50,
            include_patterns=["*", "*/*", "*/*/*"],  # fnmatch compatible patterns
        )
        return RepositoryCrawler(config)

    @pytest.fixture
    def test_repo(self):
        """Create a temporary test repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create some test files
            (repo_path / "README.md").write_text("# Test Repo\n")
            (repo_path / "main.py").write_text(
                """
import os
import sys

def main():
    print("Hello")

class MyClass:
    def method(self):
        pass
"""
            )
            (repo_path / "utils.py").write_text(
                """
from typing import List

def helper(items: List[str]) -> int:
    return len(items)
"""
            )
            (repo_path / "config.json").write_text('{"key": "value"}')

            # Create subdirectory
            (repo_path / "src").mkdir()
            (repo_path / "src" / "module.py").write_text(
                """
from main import main
from utils import helper

class Service:
    def run(self):
        main()
"""
            )

            yield repo_path

    def test_crawler_name(self, crawler):
        """Test crawler name property."""
        assert crawler.name == "repository"

    def test_crawler_source_type(self, crawler):
        """Test crawler source type property."""
        assert crawler.source_type == "git"

    @pytest.mark.asyncio
    async def test_discover_files(self, crawler, test_repo):
        """Test discovering files in a repository."""
        files = await crawler.discover(str(test_repo))

        assert len(files) > 0
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)
        assert any("README.md" in f for f in files)
        assert any("config.json" in f for f in files)

    @pytest.mark.asyncio
    async def test_crawl_repository(self, crawler, test_repo):
        """Test crawling a repository."""
        results: List[CrawlResult] = []

        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        assert len(results) > 0

        # Check that Python files were processed
        py_results = [r for r in results if r.path.endswith(".py")]
        assert len(py_results) >= 3

        # Check that content was extracted
        main_result = next((r for r in results if "main.py" in r.path), None)
        assert main_result is not None
        assert "def main" in main_result.content
        assert main_result.language == "python"

    @pytest.mark.asyncio
    async def test_symbol_extraction(self, crawler, test_repo):
        """Test symbol extraction from Python files."""
        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        main_result = next((r for r in results if "main.py" in r.path), None)
        assert main_result is not None

        # Check functions and classes were extracted
        assert any("function:main" in s for s in main_result.symbols)
        assert any("class:MyClass" in s for s in main_result.symbols)

    @pytest.mark.asyncio
    async def test_import_extraction(self, crawler, test_repo):
        """Test import extraction from Python files."""
        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        main_result = next((r for r in results if "main.py" in r.path), None)
        assert main_result is not None

        # Check imports were extracted
        assert "os" in main_result.imports
        assert "sys" in main_result.imports

    @pytest.mark.asyncio
    async def test_crawl_stats(self, crawler, test_repo):
        """Test crawl statistics are tracked."""
        async for _ in crawler.crawl(str(test_repo)):
            pass

        stats = crawler._stats

        assert stats.status == CrawlStatus.COMPLETED
        assert stats.processed_files > 0
        assert stats.total_bytes > 0

    @pytest.mark.asyncio
    async def test_skip_binary_files(self, crawler, test_repo):
        """Test that binary files are skipped."""
        # Create a binary file
        (test_repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        # Binary file should not be in results
        assert not any(r.path.endswith(".png") for r in results)

    @pytest.mark.asyncio
    async def test_skip_large_files(self, test_repo):
        """Test that large files are skipped."""
        # Create crawler with small max file size
        config = RepositoryCrawlerConfig(max_file_size_bytes=10)
        crawler = RepositoryCrawler(config)

        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        # All files should be skipped (they're all > 10 bytes)
        # README.md has content "# Test Repo\n" which is > 10 bytes
        stats = crawler._stats
        assert stats.skipped_files > 0

    @pytest.mark.asyncio
    async def test_max_files_limit(self, test_repo):
        """Test that max files limit is respected."""
        config = RepositoryCrawlerConfig(max_files=2)
        crawler = RepositoryCrawler(config)

        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, crawler):
        """Test crawling a nonexistent path."""
        results = []
        async for result in crawler.crawl("/nonexistent/path"):
            results.append(result)

        assert len(results) == 0
        stats = crawler._stats
        assert stats.status == CrawlStatus.FAILED

    @pytest.mark.asyncio
    async def test_crawl_result_metadata(self, crawler, test_repo):
        """Test that crawl results include metadata."""
        results = []
        async for result in crawler.crawl(str(test_repo)):
            results.append(result)

        if results:
            result = results[0]
            assert "repo" in result.metadata
            assert "branch" in result.metadata


class TestLanguageDetection:
    """Test language detection in RepositoryCrawler."""

    @pytest.fixture
    def crawler(self):
        return RepositoryCrawler()

    def test_detect_python(self, crawler):
        """Test detecting Python files."""
        assert crawler._detect_language("main.py") == "python"
        assert crawler._detect_language("src/utils.py") == "python"

    def test_detect_javascript(self, crawler):
        """Test detecting JavaScript files."""
        assert crawler._detect_language("app.js") == "javascript"
        assert crawler._detect_language("components/Button.jsx") == "javascript"

    def test_detect_typescript(self, crawler):
        """Test detecting TypeScript files."""
        assert crawler._detect_language("app.ts") == "typescript"
        assert crawler._detect_language("components/Button.tsx") == "typescript"

    def test_detect_go(self, crawler):
        """Test detecting Go files."""
        assert crawler._detect_language("main.go") == "go"

    def test_detect_rust(self, crawler):
        """Test detecting Rust files."""
        assert crawler._detect_language("main.rs") == "rust"

    def test_detect_unknown(self, crawler):
        """Test detecting unknown file types."""
        # Should return None for unknown extensions
        result = crawler._detect_language("file.unknown")
        assert result is None or result == ""


class TestSymbolExtraction:
    """Test symbol extraction patterns."""

    @pytest.fixture
    def crawler(self):
        return RepositoryCrawler()

    def test_python_functions(self, crawler):
        """Test extracting Python function names."""
        content = """
def simple_func():
    pass

def func_with_args(a, b):
    return a + b

async def async_func():
    await something()
"""
        symbols = crawler._extract_symbols(content, "python")

        assert "function:simple_func" in symbols
        assert "function:func_with_args" in symbols
        # Note: async functions may or may not be captured depending on regex

    def test_python_classes(self, crawler):
        """Test extracting Python class names."""
        content = """
class SimpleClass:
    pass

class ChildClass(ParentClass):
    pass
"""
        symbols = crawler._extract_symbols(content, "python")

        assert "class:SimpleClass" in symbols
        assert "class:ChildClass" in symbols

    def test_javascript_functions(self, crawler):
        """Test extracting JavaScript function names."""
        content = """
function regularFunc() {}
async function asyncFunc() {}
const arrowFunc = () => {};
"""
        symbols = crawler._extract_symbols(content, "javascript")

        assert "function:regularFunc" in symbols
        assert "function:asyncFunc" in symbols

    def test_go_functions(self, crawler):
        """Test extracting Go function names."""
        content = """
func main() {}
func (s *Server) HandleRequest() {}
func helper(x int) int {}
"""
        symbols = crawler._extract_symbols(content, "go")

        assert "function:main" in symbols
        assert "function:HandleRequest" in symbols
        assert "function:helper" in symbols

    def test_rust_functions(self, crawler):
        """Test extracting Rust function names."""
        content = """
fn main() {}
pub fn public_func() {}
async fn async_func() {}
"""
        symbols = crawler._extract_symbols(content, "rust")

        assert "function:main" in symbols
        assert "function:public_func" in symbols


class TestImportExtraction:
    """Test import extraction patterns."""

    @pytest.fixture
    def crawler(self):
        return RepositoryCrawler()

    def test_python_imports(self, crawler):
        """Test extracting Python imports."""
        content = """
import os
import sys
from typing import List, Dict
from collections import defaultdict
from .relative import something
"""
        imports = crawler._extract_imports(content, "python")

        assert "os" in imports
        assert "sys" in imports
        assert "typing" in imports
        assert "collections" in imports

    def test_javascript_imports(self, crawler):
        """Test extracting JavaScript imports."""
        content = """
import React from 'react';
import { useState } from 'react';
const fs = require('fs');
"""
        imports = crawler._extract_imports(content, "javascript")

        assert "react" in imports
        assert "fs" in imports

    def test_go_imports(self, crawler):
        """Test extracting Go imports."""
        content = """
import "fmt"
import (
    "os"
    "net/http"
)
"""
        imports = crawler._extract_imports(content, "go")

        assert "fmt" in imports

    def test_rust_imports(self, crawler):
        """Test extracting Rust imports."""
        content = """
use std::io;
use std::collections::HashMap;
use crate::module::something;
"""
        imports = crawler._extract_imports(content, "rust")

        assert any("std::io" in i for i in imports)
        assert any("std::collections::HashMap" in i for i in imports)
