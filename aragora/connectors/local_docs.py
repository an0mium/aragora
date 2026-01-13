"""
Local Documentation Connector - Search and fetch from local files.

Searches:
- Markdown files (.md)
- Code files (.py, .js, .ts, etc.)
- Text files (.txt, .rst)
- Config files (.yaml, .json, .toml)

Uses simple text search with optional regex support.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional
import hashlib

from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)
from aragora.connectors.base import BaseConnector, Evidence


class LocalDocsConnector(BaseConnector):
    """
    Connector for local documentation and code files.

    Searches through a directory tree for files matching
    queries, with support for file type filtering.
    """

    # File extensions to search by category
    EXTENSIONS = {
        "docs": [".md", ".rst", ".txt", ".adoc"],
        "code": [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"],
        "config": [".yaml", ".yml", ".json", ".toml", ".ini", ".env"],
        "all": None,  # Search all text files
    }

    def __init__(
        self,
        root_path: str = ".",
        provenance=None,
        file_types: str = "all",
        max_file_size: int = 1_000_000,  # 1MB
    ):
        super().__init__(provenance=provenance, default_confidence=0.7)
        self.root_path = Path(root_path).resolve()
        self.file_types = file_types
        self.max_file_size = max_file_size

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Local Documentation"

    def _get_extensions(self) -> Optional[list[str]]:
        """Get file extensions to search."""
        return self.EXTENSIONS.get(self.file_types)

    def _should_search_file(self, path: Path) -> bool:
        """Check if file should be searched."""
        # SECURITY: Check for symlinks that could escape root_path
        if path.is_symlink():
            return False

        # SECURITY: Verify resolved path stays within root_path
        try:
            path.resolve().relative_to(self.root_path.resolve())
        except ValueError:
            return False

        # Skip hidden files and directories
        if any(part.startswith(".") for part in path.parts):
            return False

        # Skip common non-text directories
        skip_dirs = {"node_modules", "__pycache__", ".git", "venv", "dist", "build"}
        if any(part in skip_dirs for part in path.parts):
            return False

        # Check extension
        extensions = self._get_extensions()
        if extensions and path.suffix.lower() not in extensions:
            return False

        # Check file size
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False

        return True

    def _search_in_file(
        self,
        path: Path,
        pattern: re.Pattern,
        context_lines: int = 2,
    ) -> list[dict]:
        """Search for pattern in file, return matches with context."""
        matches = []

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Get context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(lines[start:end])

                    matches.append(
                        {
                            "line_num": i + 1,
                            "line": line.strip(),
                            "context": context,
                        }
                    )

        except (OSError, UnicodeDecodeError, PermissionError) as e:
            logger.debug(f"Failed to search in {path}: {e}")

        return matches

    async def search(
        self,
        query: str,
        limit: int = 10,
        regex: bool = False,
        context_lines: int = 2,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search local files for query.

        Args:
            query: Search string or regex pattern
            limit: Max results
            regex: Treat query as regex
            context_lines: Lines of context around matches

        Returns:
            List of Evidence objects
        """
        # Compile pattern with ReDoS protection
        if regex:
            # Reject patterns that could cause catastrophic backtracking (ReDoS)
            # These patterns have polynomial or exponential time complexity
            dangerous_patterns = [
                # Nested quantifiers in groups
                r"\([^)]*[+*][^)]*\)[+*]",  # (x+)+ or (x*)*
                r"\([^)]*[+*][^)]*\)\{",  # (x+){n,m}
                # Alternation with quantifiers
                r"\([^)]*\|[^)]*\)[+*]",  # (a|b)+
                r"\([^)]*\|[^)]*\)\{",  # (a|b){n,m}
                # Overlapping patterns with quantifiers
                r"\.\*[^)]*\.\*",  # .*x.*
                r"\.\+[^)]*\.\+",  # .+x.+
                # Quantifier after quantifier (rare but dangerous)
                r"[+*]\s*[+*]",  # x++, x**
                r"[+*]\s*\{",  # x+{n}
                r"\}\s*[+*]",  # {n}+
                # Character class with quantifier followed by similar
                r"\[[^\]]+\][+*][^)]*\[[^\]]+\][+*]",  # [abc]+[abc]+
                # Greedy quantifiers with overlapping possibilities
                r"\\w[+*][^)]*\\w[+*]",  # \w+\w+
                r"\\d[+*][^)]*\\d[+*]",  # \d+\d+
                r"\\s[+*][^)]*\\s[+*]",  # \s+\s+
            ]
            for danger in dangerous_patterns:
                if re.search(danger, query, re.IGNORECASE):
                    logger.warning(f"Rejecting potentially dangerous regex pattern: {query}")
                    # Fall back to literal search
                    pattern = re.compile(re.escape(query), re.IGNORECASE)
                    break
            else:
                try:
                    pattern = re.compile(query, re.IGNORECASE)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{query}': {e}")
                    pattern = re.compile(re.escape(query), re.IGNORECASE)
        else:
            # Escape special chars for literal search
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        results: list[Evidence] = []

        # Walk directory tree
        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                if len(results) >= limit:
                    break

                path = Path(root) / filename
                if not self._should_search_file(path):
                    continue

                matches = self._search_in_file(path, pattern, context_lines)

                if matches:
                    # Get file stats
                    stat = path.stat()
                    relative_path = path.relative_to(self.root_path)

                    # Combine matches into single evidence
                    content = "\n---\n".join(
                        f"Line {m['line_num']}:\n{m['context']}"
                        for m in matches[:3]  # Top 3 matches per file
                    )

                    evidence = Evidence(
                        id=f"local:{hashlib.sha256(str(path).encode()).hexdigest()[:12]}",
                        source_type=self.source_type,
                        source_id=str(relative_path),
                        content=content,
                        title=str(relative_path),
                        url=f"file://{path}",
                        confidence=0.8,  # Local files are fairly reliable
                        freshness=(
                            self.calculate_freshness(str(stat.st_mtime))
                            if hasattr(stat, "st_mtime")
                            else 0.7
                        ),
                        authority=0.6,  # Code/docs have moderate authority
                        metadata={
                            "file_path": str(path),
                            "match_count": len(matches),
                            "file_size": stat.st_size,
                        },
                    )

                    results.append(evidence)

        return results[:limit]

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Fetch file content by path."""
        # Extract path from evidence_id
        if evidence_id.startswith("local:"):
            # Need to find file by hash - check cache
            return self._cache_get(evidence_id)

        # Treat as file path
        path = self.root_path / evidence_id

        # SECURITY: Validate path stays within root_path (prevent directory traversal)
        try:
            path.resolve().relative_to(self.root_path.resolve())
        except ValueError:
            logger.warning(f"[local_docs] Path traversal attempt blocked: {evidence_id}")
            return None

        # SECURITY: Reject symlinks that could escape root_path
        if path.is_symlink():
            logger.warning(f"[local_docs] Symlink access blocked: {evidence_id}")
            return None

        if not path.exists():
            return None

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            stat = path.stat()

            evidence = Evidence(
                id=f"local:{hashlib.sha256(str(path).encode()).hexdigest()[:12]}",
                source_type=self.source_type,
                source_id=evidence_id,
                content=content[:10000],  # Limit content size
                title=evidence_id,
                url=f"file://{path}",
                confidence=0.8,
                freshness=0.7,
                authority=0.6,
                metadata={
                    "file_path": str(path),
                    "file_size": stat.st_size,
                    "full_content": len(content) <= 10000,
                },
            )

            self._cache_put(evidence.id, evidence)
            return evidence

        except (OSError, UnicodeDecodeError, PermissionError) as e:
            logger.debug(f"[local_docs] Failed to read {path}: {e}")
            return None

    async def list_files(
        self,
        pattern: str = "*",
        limit: int = 100,
    ) -> list[str]:
        """List files matching glob pattern."""
        files = []
        for path in self.root_path.glob(f"**/{pattern}"):
            if self._should_search_file(path):
                files.append(str(path.relative_to(self.root_path)))
                if len(files) >= limit:
                    break
        return files
