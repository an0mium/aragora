"""
Base Crawler - Abstract interface for data source crawlers.

Crawlers discover and index content from various sources:
- Git repositories (code, documentation)
- Web sites (documentation, APIs)
- Enterprise systems (SharePoint, Confluence)

Crawlers integrate with the Knowledge Mound for storage and retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class CrawlStatus(Enum):
    """Status of a crawl operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(Enum):
    """Types of crawled content."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    DATA = "data"
    BINARY = "binary"
    UNKNOWN = "unknown"


@dataclass
class CrawlResult:
    """
    Result from crawling a single item.

    Contains the content and metadata needed for indexing.
    """

    id: str
    path: str  # File path or URL
    content: str
    content_type: ContentType
    size_bytes: int
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Code-specific fields
    language: Optional[str] = None
    symbols: List[str] = field(default_factory=list)  # Functions, classes, etc.
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "path": self.path,
            "content": self.content,
            "content_type": self.content_type.value,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "metadata": self.metadata,
            "language": self.language,
            "symbols": self.symbols,
            "imports": self.imports,
            "dependencies": self.dependencies,
        }


@dataclass
class CrawlerConfig:
    """Configuration for crawler operations."""

    # Paths to include/exclude
    include_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/venv/**",
            "**/.venv/**",
        ]
    )

    # Content limits
    max_file_size_bytes: int = 1_000_000  # 1MB
    max_files: int = 10_000
    max_depth: int = 50

    # Processing options
    extract_symbols: bool = True
    extract_imports: bool = True
    compute_embeddings: bool = True

    # Incremental crawling
    incremental: bool = True
    last_crawl_timestamp: Optional[datetime] = None


@dataclass
class IndexResult:
    """Result from indexing crawled content."""

    total_items: int
    indexed_items: int
    failed_items: int
    skipped_items: int
    duration_ms: float
    errors: List[str] = field(default_factory=list)


@dataclass
class CrawlStats:
    """Statistics from a crawl operation."""

    status: CrawlStatus
    total_files: int = 0
    processed_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100


class BaseCrawler(ABC):
    """
    Abstract base for all crawlers.

    Crawlers implement:
    - discover(): Find items to crawl
    - crawl(): Process and yield crawl results
    - index(): Store results in Knowledge Mound
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self._stats = CrawlStats(status=CrawlStatus.PENDING)

    @property
    @abstractmethod
    def name(self) -> str:
        """Crawler name for identification."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type of source this crawler handles."""
        ...

    @abstractmethod
    async def discover(self, source: str) -> List[str]:
        """
        Discover items to crawl from the source.

        Args:
            source: Source identifier (path, URL, etc.)

        Returns:
            List of item paths/identifiers to crawl
        """
        ...

    @abstractmethod
    async def crawl(
        self,
        source: str,
    ) -> AsyncIterator[CrawlResult]:
        """
        Crawl the source and yield results.

        Args:
            source: Source identifier (path, URL, etc.)

        Yields:
            CrawlResult for each discovered item
        """
        ...

    async def index(
        self,
        results: List[CrawlResult],
        knowledge_mound: Any,  # KnowledgeMound
    ) -> IndexResult:
        """
        Index crawl results into the Knowledge Mound.

        Args:
            results: List of crawl results to index
            knowledge_mound: KnowledgeMound instance for storage

        Returns:
            IndexResult with indexing statistics
        """
        import time
        from aragora.knowledge.mound import KnowledgeSource

        start_time = time.time()
        indexed = 0
        failed = 0
        skipped = 0
        errors: List[str] = []

        for result in results:
            try:
                # Store in Knowledge Mound
                await knowledge_mound.store(
                    content=result.content,
                    source_type=KnowledgeSource.DOCUMENT,
                    metadata={
                        "path": result.path,
                        "language": result.language,
                        "symbols": result.symbols,
                        "content_type": result.content_type.value,
                        "crawler": self.name,
                    },
                )
                indexed += 1

            except Exception as e:
                errors.append(f"{result.path}: {str(e)}")
                failed += 1

        duration_ms = (time.time() - start_time) * 1000

        return IndexResult(
            total_items=len(results),
            indexed_items=indexed,
            failed_items=failed,
            skipped_items=skipped,
            duration_ms=duration_ms,
            errors=errors,
        )

    @property
    def stats(self) -> CrawlStats:
        """Get current crawl statistics."""
        return self._stats

    def _should_include(self, path: str) -> bool:
        """Check if path matches include patterns and not exclude patterns."""
        import fnmatch

        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return False

        # Check include patterns
        for pattern in self.config.include_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True

        return False

    def _detect_language(self, path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".cs": "csharp",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".sql": "sql",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
        }
        path_obj = Path(path)
        return ext_map.get(path_obj.suffix.lower())

    def _detect_content_type(self, path: str, content: Optional[str] = None) -> ContentType:
        """Detect content type from path and content."""
        language = self._detect_language(path)

        if language in ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp"):
            return ContentType.CODE
        elif language in ("markdown", "html"):
            return ContentType.DOCUMENTATION
        elif language in ("json", "yaml", "toml", "xml"):
            return ContentType.CONFIG
        elif language in ("sql",):
            return ContentType.DATA
        else:
            return ContentType.UNKNOWN
