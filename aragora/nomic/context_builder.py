"""
Context Builder: TRUE RLM-powered codebase context for Nomic loop.

Uses TRUE RLM (REPL-based recursive language models) to build queryable
codebase context that supports up to 10M tokens. Instead of stuffing
the entire codebase into a prompt, agents can programmatically query
and explore the context through REPL commands.

This replaces the shallow summary-based context gathering with deep,
searchable codebase comprehension.

Key features:
- Indexes the full codebase as a hierarchical RLM context
- Supports REPL-based queries (grep, partition_map, peek)
- Integrates with Knowledge Mound for semantic search
- Scales to 10M+ token codebases

Usage:
    from aragora.nomic.context_builder import NomicContextBuilder

    builder = NomicContextBuilder(aragora_path=Path("."))
    context = await builder.build_context()

    # Agents query the context via RLM
    result = await builder.query("What modules handle authentication?")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# File extensions to index
SOURCE_EXTENSIONS = {
    ".py",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".rst",
    ".txt",
}

# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}

# Maximum file size to index (1MB per file)
MAX_FILE_SIZE = 1_000_000


@dataclass
class CodebaseIndex:
    """Index of the codebase for RLM context building."""

    root_path: Path
    files: list[IndexedFile] = field(default_factory=list)
    total_bytes: int = 0
    total_files: int = 0
    total_lines: int = 0
    build_time_seconds: float = 0.0

    @property
    def total_tokens_estimate(self) -> int:
        """Estimate total tokens (rough: 1 token per 4 bytes)."""
        return self.total_bytes // 4

    def get_file(self, path: str) -> Optional[IndexedFile]:
        """Get an indexed file by relative path."""
        for f in self.files:
            if f.relative_path == path:
                return f
        return None

    def search_files(self, pattern: str) -> list[IndexedFile]:
        """Search files by path pattern (simple substring match)."""
        pattern_lower = pattern.lower()
        return [f for f in self.files if pattern_lower in f.relative_path.lower()]


@dataclass
class IndexedFile:
    """A file indexed for RLM context."""

    relative_path: str
    size_bytes: int
    line_count: int
    extension: str
    module_path: str = ""  # Python module path (e.g., aragora.debate.protocol)

    @property
    def token_estimate(self) -> int:
        """Estimate token count."""
        return self.size_bytes // 4


class NomicContextBuilder:
    """
    Builds TRUE RLM-powered codebase context for Nomic loop debates.

    Instead of feeding agents a shallow summary, this builder creates
    a searchable, hierarchical representation of the entire codebase
    that agents can query via REPL commands.

    Supports context windows up to 10M tokens by using the RLM's
    recursive decomposition strategy - agents never see the full
    context at once, but can drill into any part of it.
    """

    def __init__(
        self,
        aragora_path: Path,
        max_context_bytes: int = 0,
        include_tests: Optional[bool] = None,
        knowledge_mound: Optional[Any] = None,
    ) -> None:
        self._aragora_path = aragora_path
        self._max_context_bytes = max_context_bytes or int(
            os.environ.get("NOMIC_MAX_CONTEXT_BYTES", str(100_000_000))
        )
        if include_tests is None:
            self._include_tests = os.environ.get("NOMIC_INCLUDE_TESTS", "1") == "1"
        else:
            self._include_tests = include_tests
        self._knowledge_mound = knowledge_mound
        self._index: Optional[CodebaseIndex] = None
        self._rlm_context: Optional[Any] = None
        self._context_dir = self._aragora_path / ".nomic" / "context"
        self._context_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index(self) -> Optional[CodebaseIndex]:
        """Get the current codebase index."""
        return self._index

    async def build_index(self) -> CodebaseIndex:
        """
        Scan and index the codebase.

        Returns a CodebaseIndex with file metadata (not content) for
        efficient querying. Content is loaded on-demand via RLM.
        """
        start = time.monotonic()
        files: list[IndexedFile] = []
        total_bytes = 0

        for path in sorted(self._aragora_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in SOURCE_EXTENSIONS:
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            if not self._include_tests and "test" in path.parts:
                continue
            if path.stat().st_size > MAX_FILE_SIZE:
                continue

            rel_path = str(path.relative_to(self._aragora_path))
            size = path.stat().st_size

            # Count lines efficiently
            try:
                line_count = sum(1 for _ in open(path, "rb"))
            except (OSError, UnicodeDecodeError):
                continue

            # Derive Python module path
            module_path = ""
            if path.suffix == ".py":
                module_path = (
                    rel_path.replace("/", ".")
                    .replace("\\", ".")
                    .removesuffix(".py")
                    .removesuffix(".__init__")
                )

            files.append(
                IndexedFile(
                    relative_path=rel_path,
                    size_bytes=size,
                    line_count=line_count,
                    extension=path.suffix,
                    module_path=module_path,
                )
            )
            total_bytes += size

        elapsed = time.monotonic() - start
        self._index = CodebaseIndex(
            root_path=self._aragora_path,
            files=files,
            total_bytes=total_bytes,
            total_files=len(files),
            total_lines=sum(f.line_count for f in files),
            build_time_seconds=elapsed,
        )

        logger.info(
            "Codebase indexed: %d files, %d lines, ~%dK tokens in %.1fs",
            self._index.total_files,
            self._index.total_lines,
            self._index.total_tokens_estimate // 1000,
            elapsed,
        )
        return self._index

    async def build_rlm_context(self) -> Any:
        """
        Build a TRUE RLM context from the codebase index.

        Uses the official RLM library's REPL-based approach when available,
        falling back to hierarchical compression otherwise.

        Returns an RLMContext that agents can query programmatically.
        """
        if self._index is None:
            await self.build_index()

        try:
            from aragora.rlm.bridge import AragoraRLM
            from aragora.rlm.types import RLMConfig, RLMMode

            config = RLMConfig(
                mode=RLMMode.AUTO,
                prefer_true_rlm=True,
                max_content_bytes=self._max_context_bytes,
                target_tokens=8000,  # Larger target for codebase context
                max_depth=3,  # Deeper recursion for codebase navigation
                max_sub_calls=20,  # More sub-calls for thorough exploration
                parallel_sub_calls=True,
                cache_compressions=True,
            )

            rlm = AragoraRLM(config=config)

            # Build content from index (structure map + key files)
            content = self._build_structured_content()

            self._rlm_context = await rlm.build_context(content)
            logger.info(
                "RLM context built: %s mode",
                "TRUE RLM"
                if hasattr(rlm, "_official_rlm") and rlm._official_rlm
                else "compression",
            )
            return self._rlm_context

        except ImportError:
            logger.warning("RLM not available, using index-only context")
            return None

    async def query(self, question: str) -> str:
        """
        Query the codebase context using TRUE RLM.

        The agent's question is processed through the RLM's recursive
        decomposition - the model writes code to grep, peek, and
        partition the codebase to find the answer.
        """
        if self._rlm_context is not None:
            try:
                from aragora.rlm.bridge import AragoraRLM
                from aragora.rlm.types import RLMConfig, RLMMode

                config = RLMConfig(
                    mode=RLMMode.AUTO,
                    prefer_true_rlm=True,
                    max_content_bytes=self._max_context_bytes,
                )
                rlm = AragoraRLM(config=config)
                result = await rlm.query(question, self._rlm_context)
                return result.answer
            except Exception as exc:
                logger.warning("RLM query failed, falling back to index search: %s", exc)

        # Fallback: search index
        return self._index_search(question)

    async def build_debate_context(self) -> str:
        """
        Build a structured context string for debate agents.

        This provides a navigable map of the codebase that agents can
        reference during structured debate rounds. For TRUE RLM agents,
        this also registers the context for REPL queries.
        """
        if self._index is None:
            await self.build_index()
        assert self._index is not None

        sections = []
        sections.append(
            f"# Aragora Codebase Context ({self._index.total_files} files, "
            f"~{self._index.total_tokens_estimate // 1000}K tokens)"
        )
        sections.append("")

        # Group files by top-level directory
        dirs: dict[str, list[IndexedFile]] = {}
        for f in self._index.files:
            parts = f.relative_path.split("/")
            top_dir = parts[0] if len(parts) > 1 else "root"
            dirs.setdefault(top_dir, []).append(f)

        for dir_name in sorted(dirs.keys()):
            files = dirs[dir_name]
            total_lines = sum(f.line_count for f in files)
            sections.append(f"## {dir_name}/ ({len(files)} files, {total_lines} lines)")
            # Show largest files
            for f in sorted(files, key=lambda x: x.size_bytes, reverse=True)[:10]:
                sections.append(f"  - {f.relative_path} ({f.line_count} lines)")
            if len(files) > 10:
                sections.append(f"  ... and {len(files) - 10} more files")
            sections.append("")

        # Add Knowledge Mound context if available
        if self._knowledge_mound is not None:
            try:
                km_context = await self._query_knowledge_mound()
                if km_context:
                    sections.append("## Knowledge Mound Context")
                    sections.append(km_context)
                    sections.append("")
            except Exception as exc:
                logger.warning("Knowledge Mound query failed: %s", exc)

        return "\n".join(sections)

    def _build_structured_content(self) -> str:
        """Build structured content string from the codebase index."""
        if self._index is None:
            return ""

        parts = []
        parts.append(
            f"CODEBASE: aragora ({self._index.total_files} files, {self._index.total_lines} lines)"
        )
        parts.append("")

        # File tree
        parts.append("FILE TREE:")
        for f in sorted(self._index.files, key=lambda x: x.relative_path):
            parts.append(f"  {f.relative_path} [{f.line_count}L]")
        parts.append("")

        # Read key files inline (README, __init__, protocol, etc.)
        key_patterns = ["CLAUDE.md", "__init__.py", "protocol.py", "settings.py"]
        parts.append("KEY FILES CONTENT:")
        for f in self._index.files:
            if any(p in f.relative_path for p in key_patterns):
                try:
                    content = (self._index.root_path / f.relative_path).read_text(errors="replace")
                    parts.append(f"\n--- {f.relative_path} ---")
                    # Truncate large files
                    if len(content) > 50000:
                        content = content[:50000] + "\n... (truncated)"
                    parts.append(content)
                except OSError:
                    pass

        return "\n".join(parts)

    def _index_search(self, question: str) -> str:
        """Simple keyword-based search over the index."""
        if self._index is None:
            return "No index available."

        keywords = question.lower().split()
        scored: list[tuple[int, IndexedFile]] = []

        for f in self._index.files:
            path_lower = f.relative_path.lower()
            module_lower = f.module_path.lower()
            score = sum(1 for kw in keywords if kw in path_lower or kw in module_lower)
            if score > 0:
                scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "No matching files found."

        lines = [f"Found {len(scored)} relevant files:"]
        for score, f in scored[:20]:
            lines.append(f"  [{score}] {f.relative_path} ({f.line_count} lines)")
        return "\n".join(lines)

    async def _query_knowledge_mound(self) -> str:
        """Query Knowledge Mound for relevant context."""
        if self._knowledge_mound is None:
            return ""

        try:
            # Use RLM-powered query if available
            if hasattr(self._knowledge_mound, "query_with_true_rlm"):
                result = await self._knowledge_mound.query_with_true_rlm(
                    query="What are the most important recent insights about the codebase?",
                    limit=20,
                )
                if result:
                    return str(result.answer) if hasattr(result, "answer") else str(result)

            # Fallback to semantic query
            if hasattr(self._knowledge_mound, "query_semantic"):
                items = await self._knowledge_mound.query_semantic(
                    "codebase architecture improvements recent changes",
                    limit=20,
                )
                if items:
                    lines = []
                    for item in items[:10]:
                        title = getattr(item, "title", str(item))
                        lines.append(f"- {title}")
                    return "\n".join(lines)
        except Exception as exc:
            logger.warning("Knowledge Mound query error: %s", exc)

        return ""
