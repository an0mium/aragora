"""
Tests for Local Documentation Connector.

Tests cover:
- File filtering (_should_search_file)
- Search operations (literal, regex, limits)
- Fetch operations
- Security (hidden files, path boundaries)
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from aragora.connectors.local_docs import LocalDocsConnector
from aragora.reasoning.provenance import SourceType


class TestLocalDocsConnectorInit:
    """Tests for LocalDocsConnector initialization."""

    def test_connector_creation(self, temp_dir):
        """Connector should be created with root path."""
        connector = LocalDocsConnector(root_path=str(temp_dir))
        assert connector.root_path == temp_dir.resolve()

    def test_source_type_is_document(self, temp_dir):
        """source_type should be DOCUMENT."""
        connector = LocalDocsConnector(root_path=str(temp_dir))
        assert connector.source_type == SourceType.DOCUMENT

    def test_name_is_local_documentation(self, temp_dir):
        """name should be 'Local Documentation'."""
        connector = LocalDocsConnector(root_path=str(temp_dir))
        assert connector.name == "Local Documentation"

    def test_default_file_types(self, temp_dir):
        """Default file_types should be 'all'."""
        connector = LocalDocsConnector(root_path=str(temp_dir))
        assert connector.file_types == "all"

    def test_custom_file_types(self, temp_dir):
        """Custom file_types should be stored."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="docs")
        assert connector.file_types == "docs"


class TestExtensionsDictionary:
    """Tests for EXTENSIONS dictionary."""

    def test_docs_extensions(self, temp_dir):
        """docs category should include markdown and text files."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="docs")
        extensions = connector._get_extensions()
        assert ".md" in extensions
        assert ".rst" in extensions
        assert ".txt" in extensions

    def test_code_extensions(self, temp_dir):
        """code category should include programming languages."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="code")
        extensions = connector._get_extensions()
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions

    def test_config_extensions(self, temp_dir):
        """config category should include config files."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="config")
        extensions = connector._get_extensions()
        assert ".yaml" in extensions
        assert ".json" in extensions
        assert ".toml" in extensions

    def test_all_returns_none(self, temp_dir):
        """'all' file_types should return None (search all)."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="all")
        extensions = connector._get_extensions()
        assert extensions is None


class TestFileFiltering:
    """Tests for _should_search_file method."""

    @pytest.fixture
    def connector(self, temp_dir):
        return LocalDocsConnector(root_path=str(temp_dir))

    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample file structure."""
        # Regular files
        (temp_dir / "doc.md").write_text("Documentation")
        (temp_dir / "code.py").write_text("print('hello')")
        (temp_dir / "config.yaml").write_text("key: value")

        # Hidden file
        (temp_dir / ".hidden").write_text("Secret")

        # Hidden directory
        hidden_dir = temp_dir / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config").write_text("git config")

        # node_modules
        node_dir = temp_dir / "node_modules"
        node_dir.mkdir()
        (node_dir / "package.json").write_text("{}")

        # __pycache__
        cache_dir = temp_dir / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.pyc").write_bytes(b"bytecode")

        return temp_dir

    def test_skips_hidden_files(self, connector, sample_files):
        """Hidden files (starting with .) should be skipped."""
        hidden_file = sample_files / ".hidden"
        assert connector._should_search_file(hidden_file) is False

    def test_skips_hidden_directories(self, connector, sample_files):
        """Files in hidden directories should be skipped."""
        git_file = sample_files / ".git" / "config"
        assert connector._should_search_file(git_file) is False

    def test_skips_node_modules(self, connector, sample_files):
        """Files in node_modules should be skipped."""
        node_file = sample_files / "node_modules" / "package.json"
        assert connector._should_search_file(node_file) is False

    def test_skips_pycache(self, connector, sample_files):
        """Files in __pycache__ should be skipped."""
        cache_file = sample_files / "__pycache__" / "module.pyc"
        assert connector._should_search_file(cache_file) is False

    def test_accepts_regular_files(self, connector, sample_files):
        """Regular files should be accepted."""
        doc_file = sample_files / "doc.md"
        assert connector._should_search_file(doc_file) is True

    def test_respects_extension_filter(self, temp_dir):
        """When file_types is set, only matching extensions should pass."""
        connector = LocalDocsConnector(root_path=str(temp_dir), file_types="docs")
        (temp_dir / "doc.md").write_text("Doc")
        (temp_dir / "code.py").write_text("Code")

        assert connector._should_search_file(temp_dir / "doc.md") is True
        assert connector._should_search_file(temp_dir / "code.py") is False

    def test_respects_max_file_size(self, temp_dir):
        """Files exceeding max_file_size should be skipped."""
        connector = LocalDocsConnector(
            root_path=str(temp_dir),
            max_file_size=100,  # 100 bytes
        )
        # Create large file
        (temp_dir / "large.txt").write_text("x" * 200)
        # Create small file
        (temp_dir / "small.txt").write_text("x" * 50)

        assert connector._should_search_file(temp_dir / "large.txt") is False
        assert connector._should_search_file(temp_dir / "small.txt") is True


class TestSearchOperations:
    """Tests for search functionality."""

    @pytest.fixture
    def sample_project(self, temp_dir):
        """Create sample project structure."""
        # Root files
        (temp_dir / "README.md").write_text("# Project\nThis is a test project")
        (temp_dir / "main.py").write_text("def main():\n    print('Hello')")

        # Subdirectory
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "utils.py").write_text("def helper():\n    return 'world'")
        (src_dir / "config.yaml").write_text("debug: true\nport: 8080")

        return temp_dir

    @pytest.fixture
    def connector(self, sample_project):
        return LocalDocsConnector(root_path=str(sample_project))

    @pytest.mark.asyncio
    async def test_search_finds_literal_match(self, connector):
        """search should find literal string matches."""
        results = await connector.search("Project")

        assert len(results) >= 1
        assert any("README.md" in r.source_id for r in results)

    @pytest.mark.asyncio
    async def test_search_with_regex(self, connector):
        """search with regex=True should match patterns."""
        results = await connector.search(r"def \w+\(\)", regex=True)

        assert len(results) >= 2  # main.py and utils.py

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, connector):
        """search should respect limit parameter."""
        results = await connector.search("e", limit=1)  # 'e' matches many files
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_includes_context(self, connector):
        """search should include context lines."""
        results = await connector.search("print", context_lines=2)

        assert len(results) >= 1
        # Should have context around the match
        assert "def main" in results[0].content or "Hello" in results[0].content

    @pytest.mark.asyncio
    async def test_search_returns_relative_paths(self, connector):
        """search results should have relative paths as source_id."""
        results = await connector.search("Project")

        assert len(results) >= 1
        # source_id should be relative path
        assert not results[0].source_id.startswith("/")

    @pytest.mark.asyncio
    async def test_search_creates_correct_evidence_id(self, connector):
        """search should create hash-based evidence IDs."""
        results = await connector.search("Project")

        assert len(results) >= 1
        assert results[0].id.startswith("local:")

    @pytest.mark.asyncio
    async def test_search_records_match_count(self, connector):
        """search should record match count in metadata."""
        results = await connector.search("def")

        assert len(results) >= 1
        assert "match_count" in results[0].metadata
        assert results[0].metadata["match_count"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_matches(self, connector):
        """search with no matches should return empty list."""
        results = await connector.search("xyznonexistent123")
        assert results == []


class TestFetchOperations:
    """Tests for fetch functionality."""

    @pytest.fixture
    def sample_project(self, temp_dir):
        """Create sample project."""
        (temp_dir / "doc.md").write_text("# Documentation\nThis is content")
        return temp_dir

    @pytest.fixture
    def connector(self, sample_project):
        return LocalDocsConnector(root_path=str(sample_project))

    @pytest.mark.asyncio
    async def test_fetch_by_relative_path(self, connector):
        """fetch should load file by relative path."""
        result = await connector.fetch("doc.md")

        assert result is not None
        assert "Documentation" in result.content
        assert result.source_id == "doc.md"

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_returns_none(self, connector):
        """fetch nonexistent file should return None."""
        result = await connector.fetch("nonexistent.md")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_cache(self, connector, sample_project):
        """fetch should return cached evidence."""
        # First, search to populate cache
        results = await connector.search("Documentation")
        if results:
            evidence_id = results[0].id
            # Add to cache using proper method
            connector._cache_put(evidence_id, results[0])

            # Fetch from cache
            cached = await connector.fetch(evidence_id)
            assert cached is not None

    @pytest.mark.asyncio
    async def test_fetch_truncates_large_content(self, temp_dir):
        """fetch should truncate very large files."""
        # Create large file
        large_content = "x" * 20000  # Larger than 10000 limit
        (temp_dir / "large.txt").write_text(large_content)

        connector = LocalDocsConnector(root_path=str(temp_dir))
        result = await connector.fetch("large.txt")

        assert result is not None
        assert len(result.content) <= 10000
        assert result.metadata["full_content"] is False


class TestListFiles:
    """Tests for list_files functionality."""

    @pytest.fixture
    def sample_project(self, temp_dir):
        """Create sample project."""
        (temp_dir / "file1.md").write_text("Doc 1")
        (temp_dir / "file2.md").write_text("Doc 2")
        (temp_dir / "code.py").write_text("Code")

        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("Nested")

        return temp_dir

    @pytest.fixture
    def connector(self, sample_project):
        return LocalDocsConnector(root_path=str(sample_project))

    @pytest.mark.asyncio
    async def test_list_all_files(self, connector):
        """list_files should return all matching files."""
        files = await connector.list_files("*")
        # Should find at least the files we created
        assert len(files) >= 3

    @pytest.mark.asyncio
    async def test_list_files_with_pattern(self, connector):
        """list_files should filter by pattern."""
        files = await connector.list_files("*.md")
        assert all(f.endswith(".md") for f in files)

    @pytest.mark.asyncio
    async def test_list_files_respects_limit(self, connector):
        """list_files should respect limit parameter."""
        files = await connector.list_files("*", limit=2)
        assert len(files) <= 2


class TestSecurityBehavior:
    """Tests for security-related behavior."""

    @pytest.fixture
    def sample_project(self, temp_dir):
        """Create sample project."""
        (temp_dir / "public.md").write_text("Public content")

        # Hidden directory
        hidden = temp_dir / ".secret"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("Secret content")

        return temp_dir

    @pytest.fixture
    def connector(self, sample_project):
        return LocalDocsConnector(root_path=str(sample_project))

    @pytest.mark.asyncio
    async def test_hidden_directory_excluded_from_search(self, connector):
        """Search should not find content in hidden directories."""
        results = await connector.search("Secret")
        # Should not find the secret content
        assert not any(".secret" in r.source_id for r in results)

    @pytest.mark.asyncio
    async def test_list_files_excludes_hidden(self, connector):
        """list_files should not include hidden files/directories."""
        files = await connector.list_files("*")
        assert not any(".secret" in f for f in files)
        assert not any(f.startswith(".") for f in files)

    def test_root_path_resolved(self, temp_dir):
        """root_path should be resolved to absolute path."""
        connector = LocalDocsConnector(root_path=".")
        assert connector.root_path.is_absolute()


class TestEncodingHandling:
    """Tests for file encoding handling."""

    @pytest.fixture
    def connector(self, temp_dir):
        return LocalDocsConnector(root_path=str(temp_dir))

    @pytest.mark.asyncio
    async def test_handles_utf8_content(self, connector, temp_dir):
        """search should handle UTF-8 content."""
        (temp_dir / "unicode.md").write_text("Hello 世界 🌍", encoding="utf-8")

        results = await connector.search("世界")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_handles_encoding_errors_gracefully(self, connector, temp_dir):
        """search should handle files with encoding errors."""
        # Write binary content that's not valid UTF-8
        (temp_dir / "binary.txt").write_bytes(b"\xff\xfe\x00\x01")

        # Should not raise, should skip the file
        results = await connector.search("test")
        # Should complete without error
        assert isinstance(results, list)
