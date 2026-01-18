"""
Repository Crawler - Crawl git repositories for code and documentation.

Provides:
- Git repository cloning and fetching
- Incremental crawling based on commits
- File content extraction
- Basic symbol extraction (without tree-sitter)
- Integration with Knowledge Mound

For full code intelligence (AST parsing, dependency graphs), see code_intelligence.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional

from aragora.crawlers.base import (
    BaseCrawler,
    CrawlerConfig,
    CrawlResult,
    CrawlStats,
    CrawlStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class RepositoryInfo:
    """Information about a git repository."""

    path: Path
    remote_url: Optional[str] = None
    branch: str = "main"
    last_commit: Optional[str] = None
    commit_count: int = 0


@dataclass
class RepositoryCrawlerConfig(CrawlerConfig):
    """Configuration specific to repository crawling."""

    # Git options
    clone_depth: int = 1  # Shallow clone by default
    branch: str = "main"
    fetch_tags: bool = False

    # Symbol extraction (basic regex-based)
    extract_functions: bool = True
    extract_classes: bool = True
    extract_imports: bool = True

    # File handling
    binary_extensions: List[str] = field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".zip", ".tar", ".gz", ".rar",
        ".exe", ".dll", ".so", ".dylib",
        ".woff", ".woff2", ".ttf", ".eot",
        ".mp3", ".mp4", ".wav", ".avi",
    ])


class RepositoryCrawler(BaseCrawler):
    """
    Crawler for git repositories.

    Crawls local or remote git repositories, extracting:
    - File content (code, docs, config)
    - Basic symbols (functions, classes)
    - Import statements

    Usage:
        crawler = RepositoryCrawler()

        # Crawl local repository
        async for result in crawler.crawl("/path/to/repo"):
            print(f"Found: {result.path}")

        # Crawl remote repository
        async for result in crawler.crawl("https://github.com/user/repo"):
            print(f"Found: {result.path}")
    """

    def __init__(
        self,
        config: Optional[RepositoryCrawlerConfig] = None,
    ):
        super().__init__(config or RepositoryCrawlerConfig())
        self._repo_info: Optional[RepositoryInfo] = None

    @property
    def name(self) -> str:
        return "repository"

    @property
    def source_type(self) -> str:
        return "git"

    async def discover(self, source: str) -> List[str]:
        """
        Discover files in a repository.

        Args:
            source: Repository path (local) or URL (remote)

        Returns:
            List of file paths to crawl
        """
        repo_path = await self._ensure_repository(source)
        if repo_path is None:
            return []

        files: List[str] = []

        for root, dirs, filenames in os.walk(repo_path):
            # Filter directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)

                if self._should_include(rel_path):
                    files.append(rel_path)

                if len(files) >= self.config.max_files:
                    break

            if len(files) >= self.config.max_files:
                break

        logger.info(f"Discovered {len(files)} files in {source}")
        return files

    async def crawl(
        self,
        source: str,
    ) -> AsyncIterator[CrawlResult]:
        """
        Crawl a repository and yield results.

        Args:
            source: Repository path (local) or URL (remote)

        Yields:
            CrawlResult for each file
        """
        self._stats = CrawlStats(
            status=CrawlStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            repo_path = await self._ensure_repository(source)
            if repo_path is None:
                self._stats.status = CrawlStatus.FAILED
                self._stats.errors.append(f"Failed to access repository: {source}")
                return

            files = await self.discover(source)
            self._stats.total_files = len(files)

            for file_path in files:
                try:
                    result = await self._crawl_file(repo_path, file_path)
                    if result:
                        self._stats.processed_files += 1
                        self._stats.total_bytes += result.size_bytes
                        yield result
                    else:
                        self._stats.skipped_files += 1

                except Exception as e:
                    self._stats.failed_files += 1
                    self._stats.errors.append(f"{file_path}: {str(e)}")
                    logger.warning(f"Failed to crawl {file_path}: {e}")

            self._stats.status = CrawlStatus.COMPLETED

        except Exception as e:
            self._stats.status = CrawlStatus.FAILED
            self._stats.errors.append(str(e))
            logger.exception(f"Crawl failed: {e}")

        finally:
            self._stats.completed_at = datetime.utcnow()
            if self._stats.started_at:
                self._stats.duration_ms = (
                    (self._stats.completed_at - self._stats.started_at).total_seconds() * 1000
                )

    async def _ensure_repository(self, source: str) -> Optional[Path]:
        """
        Ensure repository is available locally.

        For local paths, validates the path exists.
        For remote URLs, clones the repository.
        """
        if source.startswith(("http://", "https://", "git@")):
            return await self._clone_repository(source)
        else:
            path = Path(source)
            if path.exists() and path.is_dir():
                self._repo_info = RepositoryInfo(path=path)
                return path
            else:
                logger.error(f"Repository path does not exist: {source}")
                return None

    async def _clone_repository(self, url: str) -> Optional[Path]:
        """Clone a remote repository."""
        import tempfile

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="aragora_crawl_"))
        config = self.config if isinstance(self.config, RepositoryCrawlerConfig) else RepositoryCrawlerConfig()

        try:
            # Clone with git
            cmd = [
                "git", "clone",
                "--depth", str(config.clone_depth),
                "--branch", config.branch,
                "--single-branch",
                url,
                str(temp_dir),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Git clone failed: {stderr.decode()}")
                return None

            self._repo_info = RepositoryInfo(
                path=temp_dir,
                remote_url=url,
                branch=config.branch,
            )
            return temp_dir

        except Exception as e:
            logger.exception(f"Failed to clone repository: {e}")
            return None

    async def _crawl_file(
        self,
        repo_path: Path,
        rel_path: str,
    ) -> Optional[CrawlResult]:
        """Crawl a single file and extract content."""
        config = self.config if isinstance(self.config, RepositoryCrawlerConfig) else RepositoryCrawlerConfig()
        full_path = repo_path / rel_path

        # Skip binary files
        if any(rel_path.endswith(ext) for ext in config.binary_extensions):
            return None

        # Check file size
        try:
            stat = full_path.stat()
            if stat.st_size > self.config.max_file_size_bytes:
                logger.debug(f"Skipping large file: {rel_path} ({stat.st_size} bytes)")
                return None
            if stat.st_size == 0:
                return None
        except OSError:
            return None

        # Read content
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {rel_path}: {e}")
            return None

        # Detect language and content type
        language = self._detect_language(rel_path)
        content_type = self._detect_content_type(rel_path, content)

        # Extract symbols
        symbols = []
        imports = []
        if language and config.extract_functions:
            symbols = self._extract_symbols(content, language)
        if language and config.extract_imports:
            imports = self._extract_imports(content, language)

        # Generate ID
        file_id = hashlib.sha256(
            f"{self._repo_info.remote_url or repo_path}:{rel_path}".encode()
        ).hexdigest()[:16]

        return CrawlResult(
            id=f"repo_{file_id}",
            path=rel_path,
            content=content,
            content_type=content_type,
            size_bytes=len(content.encode("utf-8")),
            created_at=datetime.fromtimestamp(stat.st_ctime) if stat else None,
            modified_at=datetime.fromtimestamp(stat.st_mtime) if stat else None,
            metadata={
                "repo": str(self._repo_info.remote_url or repo_path) if self._repo_info else str(repo_path),
                "branch": self._repo_info.branch if self._repo_info else "unknown",
            },
            language=language,
            symbols=symbols,
            imports=imports,
        )

    def _extract_symbols(self, content: str, language: str) -> List[str]:
        """Extract function and class names using regex patterns."""
        symbols = []

        if language == "python":
            # Python function definitions
            for match in re.finditer(r"^def\s+(\w+)\s*\(", content, re.MULTILINE):
                symbols.append(f"function:{match.group(1)}")
            # Python class definitions
            for match in re.finditer(r"^class\s+(\w+)\s*[:\(]", content, re.MULTILINE):
                symbols.append(f"class:{match.group(1)}")

        elif language in ("javascript", "typescript"):
            # Function declarations
            for match in re.finditer(r"(?:function|async function)\s+(\w+)\s*\(", content):
                symbols.append(f"function:{match.group(1)}")
            # Arrow functions assigned to const/let
            for match in re.finditer(r"(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\(", content):
                symbols.append(f"function:{match.group(1)}")
            # Class declarations
            for match in re.finditer(r"class\s+(\w+)", content):
                symbols.append(f"class:{match.group(1)}")

        elif language == "go":
            # Go functions
            for match in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", content):
                symbols.append(f"function:{match.group(1)}")
            # Go types
            for match in re.finditer(r"type\s+(\w+)\s+(?:struct|interface)", content):
                symbols.append(f"type:{match.group(1)}")

        elif language == "rust":
            # Rust functions
            for match in re.finditer(r"(?:pub\s+)?fn\s+(\w+)", content):
                symbols.append(f"function:{match.group(1)}")
            # Rust structs
            for match in re.finditer(r"(?:pub\s+)?struct\s+(\w+)", content):
                symbols.append(f"struct:{match.group(1)}")
            # Rust enums
            for match in re.finditer(r"(?:pub\s+)?enum\s+(\w+)", content):
                symbols.append(f"enum:{match.group(1)}")

        return symbols[:100]  # Limit to avoid huge lists

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements."""
        imports = []

        if language == "python":
            # from X import Y
            for match in re.finditer(r"^from\s+([\w.]+)\s+import", content, re.MULTILINE):
                imports.append(match.group(1))
            # import X
            for match in re.finditer(r"^import\s+([\w.]+)", content, re.MULTILINE):
                imports.append(match.group(1))

        elif language in ("javascript", "typescript"):
            # import from
            for match in re.finditer(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]", content):
                imports.append(match.group(1))
            # require
            for match in re.finditer(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", content):
                imports.append(match.group(1))

        elif language == "go":
            # Go imports
            for match in re.finditer(r'import\s+(?:\(\s*)?"([^"]+)"', content):
                imports.append(match.group(1))

        elif language == "rust":
            # Rust use statements
            for match in re.finditer(r"use\s+([\w:]+)", content):
                imports.append(match.group(1))

        return list(set(imports))[:50]  # Deduplicate and limit
