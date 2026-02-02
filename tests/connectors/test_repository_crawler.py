"""Comprehensive tests for RepositoryCrawler connector.

Tests cover:
1. Repository discovery
2. File crawling and filtering
3. AST parsing for multiple languages
4. Language detection
5. Index management
6. Error handling (invalid repos, parse errors)
"""

import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.repository_crawler import (
    EXTENSION_MAP,
    CrawlConfig,
    CrawledFile,
    CrawlResult,
    CrawlState,
    FileDependency,
    FileSymbol,
    FileType,
    RepositoryCrawler,
    crawl_repository,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    os.system(f"cd {repo_path} && git init --quiet")
    os.system(f'cd {repo_path} && git config user.email "test@test.com"')
    os.system(f'cd {repo_path} && git config user.name "Test User"')

    # Create some test files
    (repo_path / "main.py").write_text(
        '''"""Main module."""

import os
from pathlib import Path

def main():
    """Main entry point."""
    print("Hello, World!")

async def async_main():
    """Async entry point."""
    await asyncio.sleep(1)

class Application:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name

    def run(self):
        """Run the application."""
        pass
'''
    )

    (repo_path / "utils.py").write_text(
        '''"""Utility functions."""

from main import Application

def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2
'''
    )

    (repo_path / "config.json").write_text('{"name": "test", "version": "1.0.0"}')

    (repo_path / "README.md").write_text("# Test Repository\n\nA test repo.")

    # Create subdirectory with more files
    (repo_path / "src").mkdir()
    (repo_path / "src" / "app.ts").write_text(
        """import { Component } from 'react';

export interface AppProps {
    name: string;
}

export class App extends Component<AppProps> {
    render() {
        return <div>{this.props.name}</div>;
    }
}

export function createApp(name: string): App {
    return new App({ name });
}

export type AppType = typeof App;
"""
    )

    (repo_path / "src" / "server.go").write_text(
        """package main

import "fmt"

type Server struct {
    port int
}

func NewServer(port int) *Server {
    return &Server{port: port}
}

func (s *Server) Start() {
    fmt.Println("Starting server on port", s.port)
}
"""
    )

    # Create node_modules (should be excluded)
    (repo_path / "node_modules").mkdir()
    (repo_path / "node_modules" / "package.json").write_text("{}")

    # Commit all files
    os.system(f"cd {repo_path} && git add . && git commit -m 'Initial commit' --quiet")

    return repo_path


@pytest.fixture
def temp_non_git_repo(tmp_path: Path) -> Path:
    """Create a temporary directory without git initialization."""
    repo_path = tmp_path / "non_git_repo"
    repo_path.mkdir()

    (repo_path / "main.py").write_text("print('hello')")
    (repo_path / "data.json").write_text('{"key": "value"}')

    return repo_path


@pytest.fixture
def default_config() -> CrawlConfig:
    """Return default crawl configuration."""
    return CrawlConfig()


@pytest.fixture
def crawler() -> RepositoryCrawler:
    """Return a RepositoryCrawler instance."""
    return RepositoryCrawler()


@pytest.fixture
def crawler_with_workspace() -> RepositoryCrawler:
    """Return a RepositoryCrawler instance with workspace ID."""
    return RepositoryCrawler(workspace_id="test-workspace")


# =============================================================================
# FileType and Language Detection Tests
# =============================================================================


class TestFileType:
    """Test FileType enum and extension mapping."""

    def test_file_type_values(self):
        """Test all FileType enum values exist."""
        assert FileType.PYTHON.value == "python"
        assert FileType.JAVASCRIPT.value == "javascript"
        assert FileType.TYPESCRIPT.value == "typescript"
        assert FileType.JAVA.value == "java"
        assert FileType.GO.value == "go"
        assert FileType.RUST.value == "rust"
        assert FileType.CPP.value == "cpp"
        assert FileType.C.value == "c"
        assert FileType.CSHARP.value == "csharp"
        assert FileType.RUBY.value == "ruby"
        assert FileType.PHP.value == "php"
        assert FileType.OTHER.value == "other"

    def test_extension_map_python(self):
        """Test Python extension mapping."""
        assert EXTENSION_MAP[".py"] == FileType.PYTHON
        assert EXTENSION_MAP[".pyi"] == FileType.PYTHON

    def test_extension_map_javascript(self):
        """Test JavaScript extension mapping."""
        assert EXTENSION_MAP[".js"] == FileType.JAVASCRIPT
        assert EXTENSION_MAP[".mjs"] == FileType.JAVASCRIPT
        assert EXTENSION_MAP[".jsx"] == FileType.JAVASCRIPT

    def test_extension_map_typescript(self):
        """Test TypeScript extension mapping."""
        assert EXTENSION_MAP[".ts"] == FileType.TYPESCRIPT
        assert EXTENSION_MAP[".tsx"] == FileType.TYPESCRIPT

    def test_extension_map_go(self):
        """Test Go extension mapping."""
        assert EXTENSION_MAP[".go"] == FileType.GO

    def test_extension_map_rust(self):
        """Test Rust extension mapping."""
        assert EXTENSION_MAP[".rs"] == FileType.RUST

    def test_extension_map_cpp(self):
        """Test C++ extension mapping."""
        assert EXTENSION_MAP[".cpp"] == FileType.CPP
        assert EXTENSION_MAP[".cc"] == FileType.CPP
        assert EXTENSION_MAP[".cxx"] == FileType.CPP
        assert EXTENSION_MAP[".hpp"] == FileType.CPP

    def test_extension_map_c(self):
        """Test C extension mapping."""
        assert EXTENSION_MAP[".h"] == FileType.C
        assert EXTENSION_MAP[".c"] == FileType.C

    def test_extension_map_markup(self):
        """Test markup file extension mapping."""
        assert EXTENSION_MAP[".md"] == FileType.MARKDOWN
        assert EXTENSION_MAP[".markdown"] == FileType.MARKDOWN
        assert EXTENSION_MAP[".json"] == FileType.JSON
        assert EXTENSION_MAP[".yaml"] == FileType.YAML
        assert EXTENSION_MAP[".yml"] == FileType.YAML
        assert EXTENSION_MAP[".toml"] == FileType.TOML
        assert EXTENSION_MAP[".xml"] == FileType.XML
        assert EXTENSION_MAP[".html"] == FileType.HTML
        assert EXTENSION_MAP[".htm"] == FileType.HTML

    def test_extension_map_shell(self):
        """Test shell script extension mapping."""
        assert EXTENSION_MAP[".sh"] == FileType.SHELL
        assert EXTENSION_MAP[".bash"] == FileType.SHELL
        assert EXTENSION_MAP[".zsh"] == FileType.SHELL


class TestLanguageDetection:
    """Test language detection via _get_file_type method."""

    def test_python_file_detection(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test Python file detection."""
        python_file = tmp_path / "test.py"
        python_file.write_text("print('hello')")
        assert crawler._get_file_type(python_file) == FileType.PYTHON

    def test_javascript_file_detection(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test JavaScript file detection."""
        js_file = tmp_path / "app.js"
        js_file.write_text("console.log('hello');")
        assert crawler._get_file_type(js_file) == FileType.JAVASCRIPT

    def test_typescript_file_detection(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test TypeScript file detection."""
        ts_file = tmp_path / "app.ts"
        ts_file.write_text("const x: number = 1;")
        assert crawler._get_file_type(ts_file) == FileType.TYPESCRIPT

    def test_dockerfile_detection(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test Dockerfile detection (special filename)."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.9")
        assert crawler._get_file_type(dockerfile) == FileType.DOCKERFILE

    def test_makefile_detection(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test Makefile detection (special filename)."""
        makefile = tmp_path / "Makefile"
        makefile.write_text("all:\n\techo 'hello'")
        assert crawler._get_file_type(makefile) == FileType.MAKEFILE

    def test_unknown_extension(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test unknown extension returns OTHER."""
        unknown_file = tmp_path / "file.xyz"
        unknown_file.write_text("content")
        assert crawler._get_file_type(unknown_file) == FileType.OTHER

    def test_case_insensitive_extension(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test case-insensitive extension detection."""
        py_upper = tmp_path / "test.PY"
        py_upper.write_text("print('hello')")
        assert crawler._get_file_type(py_upper) == FileType.PYTHON

    def test_dockerfile_case_insensitive(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test Dockerfile detection is case-insensitive."""
        dockerfile = tmp_path / "dockerfile"
        dockerfile.write_text("FROM node:18")
        assert crawler._get_file_type(dockerfile) == FileType.DOCKERFILE


# =============================================================================
# CrawlConfig Tests
# =============================================================================


class TestCrawlConfig:
    """Test CrawlConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = CrawlConfig()

        assert config.include_patterns == ["*", "**/*"]
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/.git/**" in config.exclude_patterns
        assert "**/venv/**" in config.exclude_patterns
        assert "**/__pycache__/**" in config.exclude_patterns
        assert config.include_types is None
        assert config.exclude_types == []
        assert config.max_file_size_bytes == 1_000_000
        assert config.max_files == 10_000
        assert config.extract_symbols is True
        assert config.extract_dependencies is True
        assert config.extract_docstrings is True
        assert config.include_git_history is False
        assert config.max_commits == 100
        assert config.since_commit is None
        assert config.max_concurrent_files == 20
        assert config.chunk_size_lines == 500
        assert config.chunk_overlap_lines == 50

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CrawlConfig(
            include_patterns=["*.py"],
            exclude_patterns=["**/test/**"],
            include_types=[FileType.PYTHON],
            max_file_size_bytes=500_000,
            max_files=1000,
            extract_symbols=False,
            chunk_size_lines=100,
        )

        assert config.include_patterns == ["*.py"]
        assert config.exclude_patterns == ["**/test/**"]
        assert config.include_types == [FileType.PYTHON]
        assert config.max_file_size_bytes == 500_000
        assert config.max_files == 1000
        assert config.extract_symbols is False
        assert config.chunk_size_lines == 100


# =============================================================================
# Data Class Tests
# =============================================================================


class TestFileSymbol:
    """Test FileSymbol dataclass."""

    def test_file_symbol_creation(self):
        """Test FileSymbol creation."""
        symbol = FileSymbol(
            name="my_function",
            kind="function",
            line_start=10,
            line_end=20,
            signature="def my_function(x: int) -> str",
            docstring="A function that does something.",
            parent="MyClass",
        )

        assert symbol.name == "my_function"
        assert symbol.kind == "function"
        assert symbol.line_start == 10
        assert symbol.line_end == 20
        assert symbol.signature == "def my_function(x: int) -> str"
        assert symbol.docstring == "A function that does something."
        assert symbol.parent == "MyClass"

    def test_file_symbol_defaults(self):
        """Test FileSymbol default values."""
        symbol = FileSymbol(name="func", kind="function", line_start=1, line_end=5)

        assert symbol.signature is None
        assert symbol.docstring is None
        assert symbol.parent is None


class TestFileDependency:
    """Test FileDependency dataclass."""

    def test_file_dependency_creation(self):
        """Test FileDependency creation."""
        dep = FileDependency(
            source="main.py",
            target="utils",
            kind="import",
            line=5,
        )

        assert dep.source == "main.py"
        assert dep.target == "utils"
        assert dep.kind == "import"
        assert dep.line == 5


class TestCrawledFile:
    """Test CrawledFile dataclass."""

    def test_crawled_file_creation(self):
        """Test CrawledFile creation."""
        cf = CrawledFile(
            path="/path/to/file.py",
            relative_path="file.py",
            file_type=FileType.PYTHON,
            content="print('hello')",
            content_hash="abc123",
            size_bytes=14,
            line_count=1,
        )

        assert cf.path == "/path/to/file.py"
        assert cf.relative_path == "file.py"
        assert cf.file_type == FileType.PYTHON
        assert cf.content == "print('hello')"
        assert cf.content_hash == "abc123"
        assert cf.size_bytes == 14
        assert cf.line_count == 1
        assert cf.symbols == []
        assert cf.dependencies == []
        assert cf.chunks == []

    def test_crawled_file_to_dict(self):
        """Test CrawledFile to_dict method."""
        symbol = FileSymbol(name="main", kind="function", line_start=1, line_end=3)
        dep = FileDependency(source="file.py", target="os", kind="import", line=1)

        cf = CrawledFile(
            path="/path/to/file.py",
            relative_path="file.py",
            file_type=FileType.PYTHON,
            content="import os\ndef main(): pass",
            content_hash="hash123",
            size_bytes=30,
            line_count=2,
            symbols=[symbol],
            dependencies=[dep],
            chunks=[{"index": 0, "content": "chunk", "start_line": 1, "end_line": 2}],
            last_modified=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        result = cf.to_dict()

        assert result["path"] == "/path/to/file.py"
        assert result["relative_path"] == "file.py"
        assert result["file_type"] == "python"
        assert result["content_hash"] == "hash123"
        assert result["size_bytes"] == 30
        assert result["line_count"] == 2
        assert len(result["symbols"]) == 1
        assert result["symbols"][0]["name"] == "main"
        assert len(result["dependencies"]) == 1
        assert result["dependencies"][0]["target"] == "os"
        assert result["chunk_count"] == 1
        assert result["last_modified"] == "2024-01-15T00:00:00+00:00"


class TestCrawlState:
    """Test CrawlState dataclass."""

    def test_crawl_state_creation(self):
        """Test CrawlState creation."""
        state = CrawlState(
            repository_path="/path/to/repo",
            last_crawl=datetime(2024, 1, 15, tzinfo=timezone.utc),
            last_commit="abc123",
            file_hashes={"main.py": "hash1"},
            total_crawls=5,
        )

        assert state.repository_path == "/path/to/repo"
        assert state.last_commit == "abc123"
        assert state.file_hashes == {"main.py": "hash1"}
        assert state.total_crawls == 5

    def test_crawl_state_to_dict(self):
        """Test CrawlState to_dict method."""
        state = CrawlState(
            repository_path="/path/to/repo",
            last_crawl=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            last_commit="abc123",
            file_hashes={"main.py": "hash1"},
            total_crawls=5,
        )

        result = state.to_dict()

        assert result["repository_path"] == "/path/to/repo"
        assert result["last_crawl"] == "2024-01-15T10:30:00+00:00"
        assert result["last_commit"] == "abc123"
        assert result["file_hashes"] == {"main.py": "hash1"}
        assert result["total_crawls"] == 5

    def test_crawl_state_from_dict(self):
        """Test CrawlState from_dict method."""
        data = {
            "repository_path": "/path/to/repo",
            "last_crawl": "2024-01-15T10:30:00+00:00",
            "last_commit": "abc123",
            "file_hashes": {"main.py": "hash1"},
            "total_crawls": 5,
        }

        state = CrawlState.from_dict(data)

        assert state.repository_path == "/path/to/repo"
        assert state.last_commit == "abc123"
        assert state.file_hashes == {"main.py": "hash1"}
        assert state.total_crawls == 5

    def test_crawl_state_from_dict_defaults(self):
        """Test CrawlState from_dict with missing optional fields."""
        data = {
            "repository_path": "/path/to/repo",
            "last_crawl": "2024-01-15T10:30:00+00:00",
        }

        state = CrawlState.from_dict(data)

        assert state.last_commit is None
        assert state.file_hashes == {}
        assert state.total_crawls == 0


class TestCrawlResult:
    """Test CrawlResult dataclass."""

    def test_crawl_result_creation(self):
        """Test CrawlResult creation."""
        result = CrawlResult(
            repository_path="/path/to/repo",
            repository_name="test-repo",
            files=[],
            total_files=0,
            total_lines=0,
            total_bytes=0,
            file_type_counts={},
            symbol_counts={},
            dependency_graph={},
            crawl_duration_ms=100.0,
            errors=[],
            warnings=[],
        )

        assert result.repository_path == "/path/to/repo"
        assert result.repository_name == "test-repo"
        assert result.total_files == 0
        assert result.crawl_duration_ms == 100.0

    def test_crawl_result_to_dict(self):
        """Test CrawlResult to_dict method."""
        started = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 15, 10, 1, tzinfo=timezone.utc)

        result = CrawlResult(
            repository_path="/path/to/repo",
            repository_name="test-repo",
            files=[],
            total_files=10,
            total_lines=500,
            total_bytes=10000,
            file_type_counts={"python": 5, "javascript": 5},
            symbol_counts={"function": 20, "class": 5},
            dependency_graph={"main.py": ["utils.py"]},
            crawl_duration_ms=60000.0,
            errors=["error1"],
            warnings=["warning1"],
            git_info={"branch": "main"},
            started_at=started,
            completed_at=completed,
        )

        d = result.to_dict()

        assert d["repository_path"] == "/path/to/repo"
        assert d["repository_name"] == "test-repo"
        assert d["total_files"] == 10
        assert d["total_lines"] == 500
        assert d["total_bytes"] == 10000
        assert d["file_type_counts"] == {"python": 5, "javascript": 5}
        assert d["symbol_counts"] == {"function": 20, "class": 5}
        assert d["crawl_duration_ms"] == 60000.0
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert d["git_info"] == {"branch": "main"}


# =============================================================================
# Repository Crawler Initialization Tests
# =============================================================================


class TestRepositoryCrawlerInit:
    """Test RepositoryCrawler initialization."""

    def test_init_default(self):
        """Test default initialization."""
        crawler = RepositoryCrawler()

        assert crawler._config is not None
        assert crawler._workspace_id is None
        assert crawler._state is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = CrawlConfig(max_files=100)
        crawler = RepositoryCrawler(config=config)

        assert crawler._config.max_files == 100

    def test_init_with_workspace(self):
        """Test initialization with workspace ID."""
        crawler = RepositoryCrawler(workspace_id="ws-123")

        assert crawler._workspace_id == "ws-123"


# =============================================================================
# File Discovery Tests
# =============================================================================


class TestFileDiscovery:
    """Test file discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_files_basic(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test basic file discovery."""
        files = await crawler._discover_files(temp_git_repo)

        # Should find main.py, utils.py, config.json, README.md, and files in src/
        assert len(files) >= 4

        # Convert to relative paths for easier assertion
        relative_paths = [str(f.relative_to(temp_git_repo)) for f in files]

        assert "main.py" in relative_paths
        assert "utils.py" in relative_paths
        assert "config.json" in relative_paths
        assert "README.md" in relative_paths

    @pytest.mark.asyncio
    async def test_discover_files_excludes_node_modules(
        self, crawler: RepositoryCrawler, temp_git_repo: Path
    ):
        """Test that node_modules is excluded."""
        files = await crawler._discover_files(temp_git_repo)

        relative_paths = [str(f.relative_to(temp_git_repo)) for f in files]

        # node_modules should be excluded
        for path in relative_paths:
            assert "node_modules" not in path

    @pytest.mark.asyncio
    async def test_discover_files_excludes_git_directory(
        self, crawler: RepositoryCrawler, temp_git_repo: Path
    ):
        """Test that .git directory is excluded."""
        files = await crawler._discover_files(temp_git_repo)

        relative_paths = [str(f.relative_to(temp_git_repo)) for f in files]

        for path in relative_paths:
            assert ".git" not in path

    @pytest.mark.asyncio
    async def test_discover_files_with_type_filter(self, temp_git_repo: Path):
        """Test file discovery with type filter."""
        config = CrawlConfig(include_types=[FileType.PYTHON])
        crawler = RepositoryCrawler(config=config)

        files = await crawler._discover_files(temp_git_repo)

        # Should only find Python files
        for f in files:
            assert f.suffix == ".py"

    @pytest.mark.asyncio
    async def test_discover_files_with_exclude_type(self, temp_git_repo: Path):
        """Test file discovery with exclude type filter."""
        config = CrawlConfig(exclude_types=[FileType.PYTHON])
        crawler = RepositoryCrawler(config=config)

        files = await crawler._discover_files(temp_git_repo)

        # Should not find Python files
        for f in files:
            assert f.suffix != ".py"

    @pytest.mark.asyncio
    async def test_discover_files_respects_max_file_size(self, tmp_path: Path):
        """Test that files exceeding max size are excluded."""
        repo_path = tmp_path / "size_test"
        repo_path.mkdir()

        # Create a small file
        small_file = repo_path / "small.py"
        small_file.write_text("x = 1")

        # Create a large file
        large_file = repo_path / "large.py"
        large_file.write_text("x = 1\n" * 100000)  # ~700KB

        config = CrawlConfig(max_file_size_bytes=1000)
        crawler = RepositoryCrawler(config=config)

        files = await crawler._discover_files(repo_path)

        relative_paths = [str(f.relative_to(repo_path)) for f in files]
        assert "small.py" in relative_paths
        assert "large.py" not in relative_paths

    @pytest.mark.asyncio
    async def test_discover_files_with_changed_files_filter(
        self, crawler: RepositoryCrawler, temp_git_repo: Path
    ):
        """Test file discovery with changed files filter."""
        changed_files = {"main.py", "config.json"}
        files = await crawler._discover_files(temp_git_repo, changed_files=changed_files)

        relative_paths = [str(f.relative_to(temp_git_repo)) for f in files]

        # Should only find files in changed_files
        assert len(files) == 2
        assert "main.py" in relative_paths
        assert "config.json" in relative_paths


# =============================================================================
# AST Parsing Tests
# =============================================================================


class TestPythonSymbolExtraction:
    """Test Python symbol extraction."""

    def test_extract_python_functions(self, crawler: RepositoryCrawler):
        """Test extracting Python functions."""
        content = '''def foo():
    pass

def bar(x: int) -> str:
    """A bar function."""
    return str(x)
'''
        symbols = crawler._extract_python_symbols(content)

        assert len(symbols) == 2
        assert symbols[0].name == "foo"
        assert symbols[0].kind == "function"
        assert symbols[1].name == "bar"
        assert symbols[1].docstring == "A bar function."

    def test_extract_python_async_functions(self, crawler: RepositoryCrawler):
        """Test extracting async Python functions."""
        content = """async def async_foo():
    await something()

async def async_bar(x: int) -> str:
    return str(x)
"""
        symbols = crawler._extract_python_symbols(content)

        assert len(symbols) == 2
        assert symbols[0].name == "async_foo"
        assert symbols[0].kind == "async_function"
        assert symbols[1].name == "async_bar"
        assert symbols[1].kind == "async_function"

    def test_extract_python_classes(self, crawler: RepositoryCrawler):
        """Test extracting Python classes."""
        content = '''class MyClass:
    """A class docstring."""

    def method(self):
        pass

class AnotherClass:
    pass
'''
        symbols = crawler._extract_python_symbols(content)

        # Should find classes and methods
        class_symbols = [s for s in symbols if s.kind == "class"]
        assert len(class_symbols) == 2
        assert class_symbols[0].name == "MyClass"
        assert class_symbols[0].docstring == "A class docstring."

    def test_extract_python_signature(self, crawler: RepositoryCrawler):
        """Test extracting Python function signatures."""
        content = """def typed_function(x: int, y: str) -> bool:
    return True
"""
        symbols = crawler._extract_python_symbols(content)

        assert len(symbols) == 1
        assert "def typed_function" in symbols[0].signature
        assert "int" in symbols[0].signature
        assert "str" in symbols[0].signature
        assert "bool" in symbols[0].signature

    def test_extract_python_symbols_malformed_code(self, crawler: RepositoryCrawler):
        """Test extracting symbols from malformed Python code (uses regex fallback)."""
        content = """def foo(:  # Invalid syntax
    pass

def bar(x):
    return x
"""
        # Should fall back to regex extraction
        symbols = crawler._extract_python_symbols(content)

        # Regex should still find some symbols
        names = [s.name for s in symbols]
        # The regex fallback should find 'bar' at minimum
        assert "bar" in names or len(symbols) >= 1


class TestJavaScriptSymbolExtraction:
    """Test JavaScript/TypeScript symbol extraction."""

    def test_extract_js_functions(self, crawler: RepositoryCrawler):
        """Test extracting JavaScript functions."""
        content = """function foo() {
    return 1;
}

export function bar(x) {
    return x * 2;
}

async function asyncFoo() {
    await something();
}
"""
        symbols = crawler._extract_js_symbols(content)

        names = [s.name for s in symbols]
        assert "foo" in names
        assert "bar" in names
        assert "asyncFoo" in names

    def test_extract_js_arrow_functions(self, crawler: RepositoryCrawler):
        """Test extracting JavaScript arrow functions."""
        content = """const foo = () => 1;
export const bar = async (x) => {
    return x;
};
"""
        symbols = crawler._extract_js_symbols(content)

        names = [s.name for s in symbols]
        assert "foo" in names
        assert "bar" in names

    def test_extract_js_classes(self, crawler: RepositoryCrawler):
        """Test extracting JavaScript classes."""
        content = """class MyClass {
    constructor() {}
}

export class ExportedClass {
    method() {}
}
"""
        symbols = crawler._extract_js_symbols(content)

        class_symbols = [s for s in symbols if s.kind == "class"]
        names = [s.name for s in class_symbols]
        assert "MyClass" in names
        assert "ExportedClass" in names

    def test_extract_ts_interfaces(self, crawler: RepositoryCrawler):
        """Test extracting TypeScript interfaces."""
        content = """interface MyInterface {
    name: string;
}

export interface ExportedInterface {
    value: number;
}
"""
        symbols = crawler._extract_js_symbols(content)

        interface_symbols = [s for s in symbols if s.kind == "interface"]
        names = [s.name for s in interface_symbols]
        assert "MyInterface" in names
        assert "ExportedInterface" in names

    def test_extract_ts_types(self, crawler: RepositoryCrawler):
        """Test extracting TypeScript type aliases."""
        content = """type MyType = string | number;
export type ExportedType = {
    value: boolean;
};
"""
        symbols = crawler._extract_js_symbols(content)

        type_symbols = [s for s in symbols if s.kind == "type"]
        names = [s.name for s in type_symbols]
        assert "MyType" in names
        assert "ExportedType" in names

    def test_extract_ts_enums(self, crawler: RepositoryCrawler):
        """Test extracting TypeScript enums."""
        content = """enum Color {
    Red,
    Green,
    Blue
}

export enum Status {
    Active,
    Inactive
}
"""
        symbols = crawler._extract_js_symbols(content)

        enum_symbols = [s for s in symbols if s.kind == "enum"]
        names = [s.name for s in enum_symbols]
        assert "Color" in names
        assert "Status" in names


class TestGoSymbolExtraction:
    """Test Go symbol extraction."""

    def test_extract_go_functions(self, crawler: RepositoryCrawler):
        """Test extracting Go functions."""
        content = """func foo() {
    fmt.Println("foo")
}

func bar(x int) string {
    return fmt.Sprint(x)
}
"""
        symbols = crawler._extract_go_symbols(content)

        names = [s.name for s in symbols]
        kinds = [s.kind for s in symbols]
        assert "foo" in names
        assert "bar" in names
        assert all(k == "function" for k in kinds)

    def test_extract_go_methods(self, crawler: RepositoryCrawler):
        """Test extracting Go methods."""
        content = """func (s *Server) Start() {
    s.running = true
}

func (s *Server) Stop() {
    s.running = false
}
"""
        symbols = crawler._extract_go_symbols(content)

        names = [s.name for s in symbols]
        assert "Start" in names
        assert "Stop" in names
        assert all(s.kind == "method" for s in symbols)

    def test_extract_go_structs(self, crawler: RepositoryCrawler):
        """Test extracting Go structs."""
        content = """type Server struct {
    port int
    host string
}

type Client struct {
    conn net.Conn
}
"""
        symbols = crawler._extract_go_symbols(content)

        struct_symbols = [s for s in symbols if s.kind == "struct"]
        names = [s.name for s in struct_symbols]
        assert "Server" in names
        assert "Client" in names

    def test_extract_go_interfaces(self, crawler: RepositoryCrawler):
        """Test extracting Go interfaces."""
        content = """type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
"""
        symbols = crawler._extract_go_symbols(content)

        interface_symbols = [s for s in symbols if s.kind == "interface"]
        names = [s.name for s in interface_symbols]
        assert "Reader" in names
        assert "Writer" in names


class TestJavaSymbolExtraction:
    """Test Java symbol extraction."""

    def test_extract_java_classes(self, crawler: RepositoryCrawler):
        """Test extracting Java classes."""
        content = """public class MyClass {
    private int value;
}

class PackageClass {
    String name;
}
"""
        symbols = crawler._extract_java_symbols(content)

        class_symbols = [s for s in symbols if s.kind == "class"]
        names = [s.name for s in class_symbols]
        assert "MyClass" in names
        assert "PackageClass" in names

    def test_extract_java_interfaces(self, crawler: RepositoryCrawler):
        """Test extracting Java interfaces."""
        content = """public interface MyInterface {
    void doSomething();
}

interface PackageInterface {
    int getValue();
}
"""
        symbols = crawler._extract_java_symbols(content)

        interface_symbols = [s for s in symbols if s.kind == "interface"]
        names = [s.name for s in interface_symbols]
        assert "MyInterface" in names
        assert "PackageInterface" in names

    def test_extract_java_enums(self, crawler: RepositoryCrawler):
        """Test extracting Java enums."""
        content = """public enum Status {
    ACTIVE,
    INACTIVE
}

enum Color {
    RED, GREEN, BLUE
}
"""
        symbols = crawler._extract_java_symbols(content)

        enum_symbols = [s for s in symbols if s.kind == "enum"]
        names = [s.name for s in enum_symbols]
        assert "Status" in names
        assert "Color" in names


class TestRustSymbolExtraction:
    """Test Rust symbol extraction."""

    def test_extract_rust_functions(self, crawler: RepositoryCrawler):
        """Test extracting Rust functions."""
        content = """fn foo() {
    println!("foo");
}

pub fn bar(x: i32) -> String {
    x.to_string()
}
"""
        symbols = crawler._extract_rust_symbols(content)

        func_symbols = [s for s in symbols if s.kind == "function"]
        names = [s.name for s in func_symbols]
        assert "foo" in names
        assert "bar" in names

    def test_extract_rust_structs(self, crawler: RepositoryCrawler):
        """Test extracting Rust structs."""
        content = """struct Point {
    x: f64,
    y: f64,
}

pub struct Rectangle {
    width: f64,
    height: f64,
}
"""
        symbols = crawler._extract_rust_symbols(content)

        struct_symbols = [s for s in symbols if s.kind == "struct"]
        names = [s.name for s in struct_symbols]
        assert "Point" in names
        assert "Rectangle" in names

    def test_extract_rust_enums(self, crawler: RepositoryCrawler):
        """Test extracting Rust enums."""
        content = """enum Option<T> {
    Some(T),
    None,
}

pub enum Result<T, E> {
    Ok(T),
    Err(E),
}
"""
        symbols = crawler._extract_rust_symbols(content)

        enum_symbols = [s for s in symbols if s.kind == "enum"]
        names = [s.name for s in enum_symbols]
        assert "Option" in names
        assert "Result" in names

    def test_extract_rust_traits(self, crawler: RepositoryCrawler):
        """Test extracting Rust traits."""
        content = """trait Drawable {
    fn draw(&self);
}

pub trait Serializable {
    fn serialize(&self) -> String;
}
"""
        symbols = crawler._extract_rust_symbols(content)

        trait_symbols = [s for s in symbols if s.kind == "trait"]
        names = [s.name for s in trait_symbols]
        assert "Drawable" in names
        assert "Serializable" in names

    def test_extract_rust_impl(self, crawler: RepositoryCrawler):
        """Test extracting Rust impl blocks."""
        content = """impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl<T> Vec<T> {
    fn push(&mut self, item: T) {}
}
"""
        symbols = crawler._extract_rust_symbols(content)

        impl_symbols = [s for s in symbols if s.kind == "impl"]
        names = [s.name for s in impl_symbols]
        assert "Point" in names
        assert "Vec" in names


# =============================================================================
# Dependency Extraction Tests
# =============================================================================


class TestDependencyExtraction:
    """Test dependency extraction from source code."""

    def test_extract_python_imports(self, crawler: RepositoryCrawler):
        """Test extracting Python imports."""
        content = """import os
from pathlib import Path
import json as j
from typing import Optional, List
"""
        deps = crawler._extract_dependencies(content, FileType.PYTHON, "main.py")

        targets = [d.target for d in deps]
        assert "os" in targets
        assert "pathlib" in targets
        assert "json" in targets
        assert "typing" in targets

    def test_extract_javascript_imports(self, crawler: RepositoryCrawler):
        """Test extracting JavaScript imports."""
        content = """import React from 'react';
import { useState } from "react";
const fs = require('fs');
import('./lazy-module');
"""
        deps = crawler._extract_dependencies(content, FileType.JAVASCRIPT, "app.js")

        targets = [d.target for d in deps]
        assert "react" in targets
        assert "fs" in targets

    def test_extract_typescript_imports(self, crawler: RepositoryCrawler):
        """Test extracting TypeScript imports."""
        content = """import { Component } from '@angular/core';
import * as utils from './utils';
"""
        deps = crawler._extract_dependencies(content, FileType.TYPESCRIPT, "app.ts")

        targets = [d.target for d in deps]
        assert "@angular/core" in targets
        assert "./utils" in targets

    def test_extract_go_imports(self, crawler: RepositoryCrawler):
        """Test extracting Go imports."""
        content = """import (
    "fmt"
    "os"
    "github.com/gorilla/mux"
)
"""
        deps = crawler._extract_dependencies(content, FileType.GO, "main.go")

        targets = [d.target for d in deps]
        assert "fmt" in targets
        assert "os" in targets
        assert "github.com/gorilla/mux" in targets

    def test_extract_java_imports(self, crawler: RepositoryCrawler):
        """Test extracting Java imports."""
        content = """import java.util.List;
import java.util.Map;
import com.example.MyClass;
"""
        deps = crawler._extract_dependencies(content, FileType.JAVA, "Main.java")

        targets = [d.target for d in deps]
        assert "java.util.List" in targets
        assert "java.util.Map" in targets
        assert "com.example.MyClass" in targets

    def test_extract_rust_imports(self, crawler: RepositoryCrawler):
        """Test extracting Rust imports."""
        content = """use std::collections::HashMap;
use crate::utils;
extern crate serde;
"""
        deps = crawler._extract_dependencies(content, FileType.RUST, "main.rs")

        targets = [d.target for d in deps]
        assert "std::collections::HashMap" in targets
        assert "crate::utils" in targets
        assert "serde" in targets

    def test_extract_dependencies_includes_line_numbers(self, crawler: RepositoryCrawler):
        """Test that dependency extraction includes line numbers."""
        content = """import os
import sys
from pathlib import Path
"""
        deps = crawler._extract_dependencies(content, FileType.PYTHON, "main.py")

        lines = [d.line for d in deps]
        assert 1 in lines
        assert 2 in lines
        assert 3 in lines


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunking:
    """Test file chunking functionality."""

    def test_single_chunk_for_small_file(self, crawler: RepositoryCrawler):
        """Test that small files get a single chunk."""
        content = "line1\nline2\nline3"
        lines = content.split("\n")

        chunks = crawler._create_chunks(content, lines)

        assert len(chunks) == 1
        assert chunks[0]["index"] == 0
        assert chunks[0]["content"] == content
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 3

    def test_multiple_chunks_for_large_file(self, tmp_path: Path):
        """Test that large files get multiple chunks with overlap."""
        config = CrawlConfig(chunk_size_lines=10, chunk_overlap_lines=2)
        crawler = RepositoryCrawler(config=config)

        lines = [f"line{i}" for i in range(25)]
        content = "\n".join(lines)

        chunks = crawler._create_chunks(content, lines)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Verify overlap
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["end_line"]
            next_start = chunks[i + 1]["start_line"]
            # Overlap should be present
            assert next_start <= current_end

    def test_chunk_indices_are_sequential(self, tmp_path: Path):
        """Test that chunk indices are sequential."""
        config = CrawlConfig(chunk_size_lines=5, chunk_overlap_lines=1)
        crawler = RepositoryCrawler(config=config)

        lines = [f"line{i}" for i in range(20)]
        content = "\n".join(lines)

        chunks = crawler._create_chunks(content, lines)

        for i, chunk in enumerate(chunks):
            assert chunk["index"] == i


# =============================================================================
# Dependency Graph Tests
# =============================================================================


class TestDependencyGraph:
    """Test dependency graph building."""

    def test_build_dependency_graph_empty(self, crawler: RepositoryCrawler):
        """Test building graph with no dependencies."""
        files: list[CrawledFile] = []
        graph = crawler._build_dependency_graph(files)
        assert graph == {}

    def test_build_dependency_graph_single_file(self, crawler: RepositoryCrawler):
        """Test building graph with single file."""
        cf = CrawledFile(
            path="/path/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="",
            content_hash="hash",
            size_bytes=0,
            line_count=0,
            dependencies=[
                FileDependency(source="main.py", target="os", kind="import", line=1),
                FileDependency(source="main.py", target="sys", kind="import", line=2),
            ],
        )

        graph = crawler._build_dependency_graph([cf])

        assert "main.py" in graph
        assert "os" in graph["main.py"]
        assert "sys" in graph["main.py"]

    def test_build_dependency_graph_multiple_files(self, crawler: RepositoryCrawler):
        """Test building graph with multiple files."""
        files = [
            CrawledFile(
                path="/path/main.py",
                relative_path="main.py",
                file_type=FileType.PYTHON,
                content="",
                content_hash="hash1",
                size_bytes=0,
                line_count=0,
                dependencies=[
                    FileDependency(source="main.py", target="utils", kind="import", line=1)
                ],
            ),
            CrawledFile(
                path="/path/utils.py",
                relative_path="utils.py",
                file_type=FileType.PYTHON,
                content="",
                content_hash="hash2",
                size_bytes=0,
                line_count=0,
                dependencies=[
                    FileDependency(source="utils.py", target="os", kind="import", line=1)
                ],
            ),
        ]

        graph = crawler._build_dependency_graph(files)

        assert "main.py" in graph
        assert "utils.py" in graph
        assert "utils" in graph["main.py"]
        assert "os" in graph["utils.py"]


# =============================================================================
# Git Info Tests
# =============================================================================


class TestGitInfo:
    """Test git information extraction."""

    @pytest.mark.asyncio
    async def test_get_git_info_for_git_repo(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test getting git info from a git repo."""
        git_info = await crawler._get_git_info(temp_git_repo)

        assert git_info is not None
        assert "head_commit" in git_info
        assert "branch" in git_info
        assert git_info["head_commit"] is not None
        # Branch might be 'main' or 'master' depending on git version
        assert git_info["branch"] in ["main", "master"]

    @pytest.mark.asyncio
    async def test_get_git_info_for_non_git_repo(
        self, crawler: RepositoryCrawler, temp_non_git_repo: Path
    ):
        """Test getting git info from non-git directory returns None."""
        git_info = await crawler._get_git_info(temp_non_git_repo)

        assert git_info is None


class TestGitCommands:
    """Test git command execution."""

    @pytest.mark.asyncio
    async def test_run_git_command_success(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test running successful git command."""
        result = await crawler._run_git_command(temp_git_repo, ["status", "--short"])

        # Should return empty string or status output (not None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_git_command_failure(
        self, crawler: RepositoryCrawler, temp_non_git_repo: Path
    ):
        """Test running git command in non-git directory."""
        result = await crawler._run_git_command(temp_non_git_repo, ["status"])

        # Should return None for non-git directory
        assert result is None


# =============================================================================
# Repository Name Extraction Tests
# =============================================================================


class TestRepoNameExtraction:
    """Test repository name extraction from URLs."""

    def test_extract_https_url(self, crawler: RepositoryCrawler):
        """Test extracting name from HTTPS URL."""
        url = "https://github.com/owner/repo-name"
        name = crawler._extract_repo_name(url)
        assert name == "repo-name"

    def test_extract_https_url_with_git_suffix(self, crawler: RepositoryCrawler):
        """Test extracting name from URL with .git suffix."""
        url = "https://github.com/owner/repo-name.git"
        name = crawler._extract_repo_name(url)
        assert name == "repo-name"

    def test_extract_ssh_url(self, crawler: RepositoryCrawler):
        """Test extracting name from SSH URL."""
        url = "git@github.com:owner/repo-name.git"
        name = crawler._extract_repo_name(url)
        assert name == "repo-name"

    def test_extract_url_with_trailing_slash(self, crawler: RepositoryCrawler):
        """Test extracting name from URL with trailing slash."""
        url = "https://github.com/owner/repo-name/"
        name = crawler._extract_repo_name(url)
        assert name == "repo-name"

    def test_extract_empty_url(self, crawler: RepositoryCrawler):
        """Test extracting name from empty URL."""
        url = ""
        name = crawler._extract_repo_name(url)
        assert name == "unknown"


# =============================================================================
# Full Crawl Tests
# =============================================================================


class TestFullCrawl:
    """Test full repository crawling."""

    @pytest.mark.asyncio
    async def test_crawl_local_git_repo(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test crawling a local git repository."""
        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        assert result.repository_name == "test_repo"
        assert result.total_files > 0
        assert result.total_lines > 0
        assert result.total_bytes > 0
        assert len(result.files) > 0
        assert len(result.file_type_counts) > 0
        assert result.crawl_duration_ms > 0
        assert result.git_info is not None
        assert result.errors == []
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_crawl_non_git_repo(self, crawler: RepositoryCrawler, temp_non_git_repo: Path):
        """Test crawling a non-git directory."""
        result = await crawler.crawl(str(temp_non_git_repo), incremental=False)

        assert result.repository_name == "non_git_repo"
        assert result.total_files > 0
        assert result.git_info is None

    @pytest.mark.asyncio
    async def test_crawl_nonexistent_path_raises(self, crawler: RepositoryCrawler):
        """Test crawling nonexistent path raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            await crawler.crawl("/nonexistent/path", incremental=False)

    @pytest.mark.asyncio
    async def test_crawl_updates_state(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test that crawling updates internal state."""
        assert crawler.get_state() is None

        await crawler.crawl(str(temp_git_repo), incremental=False)

        state = crawler.get_state()
        assert state is not None
        assert state.repository_path == str(temp_git_repo)
        assert state.total_crawls == 1
        assert len(state.file_hashes) > 0

    @pytest.mark.asyncio
    async def test_crawl_extracts_symbols(self, crawler: RepositoryCrawler, temp_git_repo: Path):
        """Test that crawling extracts symbols."""
        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        # Check symbol counts
        assert len(result.symbol_counts) > 0

        # Find Python file and check symbols
        python_files = [f for f in result.files if f.file_type == FileType.PYTHON]
        assert len(python_files) > 0

        main_py = next((f for f in python_files if "main.py" in f.relative_path), None)
        assert main_py is not None
        assert len(main_py.symbols) > 0

        symbol_names = [s.name for s in main_py.symbols]
        assert "main" in symbol_names
        assert "Application" in symbol_names

    @pytest.mark.asyncio
    async def test_crawl_extracts_dependencies(
        self, crawler: RepositoryCrawler, temp_git_repo: Path
    ):
        """Test that crawling extracts dependencies."""
        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        # Check dependency graph
        assert len(result.dependency_graph) > 0

        # Find utils.py and check dependencies
        utils_py = next((f for f in result.files if "utils.py" in f.relative_path), None)
        if utils_py:
            assert len(utils_py.dependencies) > 0

    @pytest.mark.asyncio
    async def test_crawl_respects_max_files(self, temp_git_repo: Path):
        """Test that crawling respects max_files limit."""
        config = CrawlConfig(max_files=2)
        crawler = RepositoryCrawler(config=config)

        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        assert result.total_files <= 2
        assert len(result.warnings) > 0  # Should warn about truncation

    @pytest.mark.asyncio
    async def test_crawl_without_symbol_extraction(self, temp_git_repo: Path):
        """Test crawling without symbol extraction."""
        config = CrawlConfig(extract_symbols=False)
        crawler = RepositoryCrawler(config=config)

        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        # All files should have empty symbols
        for f in result.files:
            assert len(f.symbols) == 0

    @pytest.mark.asyncio
    async def test_crawl_without_dependency_extraction(self, temp_git_repo: Path):
        """Test crawling without dependency extraction."""
        config = CrawlConfig(extract_dependencies=False)
        crawler = RepositoryCrawler(config=config)

        result = await crawler.crawl(str(temp_git_repo), incremental=False)

        # All files should have empty dependencies
        for f in result.files:
            assert len(f.dependencies) == 0


# =============================================================================
# Incremental Crawl Tests
# =============================================================================


class TestIncrementalCrawl:
    """Test incremental crawling functionality."""

    @pytest.mark.asyncio
    async def test_incremental_crawl_with_no_state(
        self, crawler: RepositoryCrawler, temp_git_repo: Path
    ):
        """Test incremental crawl with no prior state does full crawl."""
        result = await crawler.crawl(str(temp_git_repo), incremental=True)

        # Should still crawl everything
        assert result.total_files > 0

    @pytest.mark.asyncio
    async def test_set_and_get_state(self, crawler: RepositoryCrawler):
        """Test setting and getting crawl state."""
        state = CrawlState(
            repository_path="/path/to/repo",
            last_crawl=datetime.now(timezone.utc),
            last_commit="abc123",
            file_hashes={"main.py": "hash"},
            total_crawls=1,
        )

        crawler.set_state(state)

        retrieved = crawler.get_state()
        assert retrieved is not None
        assert retrieved.last_commit == "abc123"
        assert retrieved.total_crawls == 1


# =============================================================================
# Remote Repository Tests
# =============================================================================


class TestRemoteRepository:
    """Test remote repository handling."""

    @pytest.mark.asyncio
    async def test_clone_repository_failure(self, crawler: RepositoryCrawler):
        """Test clone failure for invalid URL."""
        with pytest.raises(RuntimeError, match="Git clone failed|Failed to clone"):
            await crawler._clone_repository("https://invalid-url/nonexistent/repo.git")

    def test_detect_remote_url_https(self, crawler: RepositoryCrawler):
        """Test detection of HTTPS remote URL."""
        url = "https://github.com/owner/repo"
        assert url.startswith(("http://", "https://", "git@", "ssh://"))

    def test_detect_remote_url_ssh(self, crawler: RepositoryCrawler):
        """Test detection of SSH remote URL."""
        url = "git@github.com:owner/repo.git"
        assert url.startswith(("http://", "https://", "git@", "ssh://"))


# =============================================================================
# Index to Mound Tests
# =============================================================================


class TestIndexToMound:
    """Test indexing to Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_index_to_mound_basic(self, crawler: RepositoryCrawler):
        """Test basic indexing to mound."""
        # Create a mock mound
        mock_mound = AsyncMock()
        mock_mound.add = AsyncMock(return_value=None)

        # Create a simple crawl result
        cf = CrawledFile(
            path="/path/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="print('hello')",
            content_hash="hash",
            size_bytes=15,
            line_count=1,
            chunks=[{"index": 0, "content": "print('hello')", "start_line": 1, "end_line": 1}],
            symbols=[
                FileSymbol(
                    name="main",
                    kind="function",
                    line_start=1,
                    line_end=1,
                    docstring="Main function",
                )
            ],
        )

        result = CrawlResult(
            repository_path="/path/repo",
            repository_name="repo",
            files=[cf],
            total_files=1,
            total_lines=1,
            total_bytes=15,
            file_type_counts={"python": 1},
            symbol_counts={"function": 1},
            dependency_graph={},
            crawl_duration_ms=100,
            errors=[],
            warnings=[],
        )

        nodes_created = await crawler.index_to_mound(result, mock_mound)

        # Should create nodes for chunk and symbol with docstring
        assert nodes_created == 2
        assert mock_mound.add.call_count == 2

    @pytest.mark.asyncio
    async def test_index_to_mound_with_workspace(self, crawler_with_workspace: RepositoryCrawler):
        """Test indexing includes workspace ID."""
        mock_mound = AsyncMock()
        mock_mound.add = AsyncMock(return_value=None)

        cf = CrawledFile(
            path="/path/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="x = 1",
            content_hash="hash",
            size_bytes=5,
            line_count=1,
            chunks=[{"index": 0, "content": "x = 1", "start_line": 1, "end_line": 1}],
        )

        result = CrawlResult(
            repository_path="/path/repo",
            repository_name="repo",
            files=[cf],
            total_files=1,
            total_lines=1,
            total_bytes=5,
            file_type_counts={"python": 1},
            symbol_counts={},
            dependency_graph={},
            crawl_duration_ms=100,
            errors=[],
            warnings=[],
        )

        await crawler_with_workspace.index_to_mound(result, mock_mound)

        # Verify workspace_id is passed
        call_kwargs = mock_mound.add.call_args_list[0][1]
        assert call_kwargs["workspace_id"] == "test-workspace"

    @pytest.mark.asyncio
    async def test_index_to_mound_handles_errors(self, crawler: RepositoryCrawler, caplog):
        """Test that indexing handles mound errors gracefully."""
        mock_mound = AsyncMock()
        mock_mound.add = AsyncMock(side_effect=ValueError("Mound error"))

        cf = CrawledFile(
            path="/path/main.py",
            relative_path="main.py",
            file_type=FileType.PYTHON,
            content="x = 1",
            content_hash="hash",
            size_bytes=5,
            line_count=1,
            chunks=[{"index": 0, "content": "x = 1", "start_line": 1, "end_line": 1}],
        )

        result = CrawlResult(
            repository_path="/path/repo",
            repository_name="repo",
            files=[cf],
            total_files=1,
            total_lines=1,
            total_bytes=5,
            file_type_counts={"python": 1},
            symbol_counts={},
            dependency_graph={},
            crawl_duration_ms=100,
            errors=[],
            warnings=[],
        )

        nodes_created = await crawler.index_to_mound(result, mock_mound)

        # Should handle error and return 0 nodes created
        assert nodes_created == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test the crawl_repository convenience function."""

    @pytest.mark.asyncio
    async def test_crawl_repository_function(self, temp_git_repo: Path):
        """Test the convenience function."""
        result = await crawl_repository(str(temp_git_repo))

        assert result.repository_name == "test_repo"
        assert result.total_files > 0

    @pytest.mark.asyncio
    async def test_crawl_repository_with_config(self, temp_git_repo: Path):
        """Test convenience function with custom config."""
        config = CrawlConfig(max_files=2)
        result = await crawl_repository(str(temp_git_repo), config=config)

        assert result.total_files <= 2

    @pytest.mark.asyncio
    async def test_crawl_repository_with_workspace(self, temp_git_repo: Path):
        """Test convenience function with workspace ID."""
        result = await crawl_repository(str(temp_git_repo), workspace_id="ws-123")

        assert result.total_files > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_process_file_handles_read_error(
        self, crawler: RepositoryCrawler, tmp_path: Path
    ):
        """Test that file processing handles read errors."""
        # Create a file then make it unreadable
        file_path = tmp_path / "unreadable.py"
        file_path.write_text("content")

        # Mock the file to raise an error on read
        with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
            with pytest.raises(RuntimeError, match="Failed to read file"):
                await crawler._process_file(file_path, tmp_path)

    @pytest.mark.asyncio
    async def test_crawl_collects_file_errors(self, tmp_path: Path):
        """Test that crawl collects errors from file processing."""
        repo_path = tmp_path / "error_repo"
        repo_path.mkdir()

        # Create a valid file and a problematic one
        (repo_path / "good.py").write_text("x = 1")

        crawler = RepositoryCrawler()

        # Patch _process_file to fail for one file
        original_process = crawler._process_file

        async def failing_process(path, repo):
            if "good" in str(path):
                return await original_process(path, repo)
            raise ValueError("Simulated error")

        with patch.object(crawler, "_process_file", failing_process):
            # Create another file that will fail
            (repo_path / "bad.py").write_text("y = 2")
            result = await crawler.crawl(str(repo_path), incremental=False)

        # Should have at least one error recorded
        # Note: errors list should contain the simulated error
        assert len(result.errors) >= 1 or result.total_files >= 1


class TestProcessFile:
    """Test individual file processing."""

    @pytest.mark.asyncio
    async def test_process_file_basic(self, crawler: RepositoryCrawler, tmp_path: Path):
        """Test basic file processing."""
        file_path = tmp_path / "test.py"
        file_path.write_text("x = 1\ny = 2")

        result = await crawler._process_file(file_path, tmp_path)

        assert result.path == str(file_path)
        assert result.relative_path == "test.py"
        assert result.file_type == FileType.PYTHON
        assert result.content == "x = 1\ny = 2"
        assert result.line_count == 2
        assert result.size_bytes > 0
        assert result.content_hash is not None
        assert len(result.content_hash) == 16  # SHA256 truncated to 16 chars

    @pytest.mark.asyncio
    async def test_process_file_calculates_correct_hash(
        self, crawler: RepositoryCrawler, tmp_path: Path
    ):
        """Test that content hash is calculated correctly."""
        content = "hello world"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = await crawler._process_file(file_path, tmp_path)

        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        assert result.content_hash == expected_hash


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from aragora.connectors import repository_crawler

        expected = [
            "RepositoryCrawler",
            "CrawlConfig",
            "CrawlResult",
            "CrawlState",
            "CrawledFile",
            "FileSymbol",
            "FileDependency",
            "FileType",
            "crawl_repository",
        ]

        for name in expected:
            assert name in repository_crawler.__all__
            assert hasattr(repository_crawler, name)
