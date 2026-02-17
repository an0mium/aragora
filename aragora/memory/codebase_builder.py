"""
CodebaseKnowledgeBuilder — Builds structured knowledge about a codebase.

Ingests file structure, import graphs, test results, and architectural
patterns into the MemoryFabric for cross-session codebase understanding.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aragora.memory.fabric import FabricResult, MemoryFabric, RememberResult
from aragora.memory.surprise import ContentSurpriseScorer

logger = logging.getLogger(__name__)


# File extensions to index
_CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".yml", ".yaml", ".toml", ".json"}

# Directories to skip
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs"}


@dataclass
class IngestionStats:
    """Statistics from a codebase ingestion operation."""
    items_ingested: int = 0
    items_skipped: int = 0
    errors: int = 0


@dataclass
class ImportRelation:
    """Represents an import relationship between modules."""
    source_module: str
    imported_module: str
    import_type: str  # "import" or "from_import"


class CodebaseKnowledgeBuilder:
    """Builds structured knowledge about a codebase.

    Ingests file structure, imports, test results, and patterns
    into MemoryFabric for cross-session codebase understanding.
    """

    def __init__(
        self,
        fabric: MemoryFabric,
        repo_path: Path,
        surprise_scorer: ContentSurpriseScorer | None = None,
    ):
        self._fabric = fabric
        self._repo_path = Path(repo_path)
        self._scorer = surprise_scorer or ContentSurpriseScorer()

    async def ingest_structure(self, max_files: int = 500) -> IngestionStats:
        """Walk repo, create entries for directories and key files.

        Args:
            max_files: Maximum number of files to index

        Returns:
            IngestionStats with counts of ingested/skipped/errored items
        """
        stats = IngestionStats()

        if not self._repo_path.is_dir():
            logger.warning("Repo path does not exist: %s", self._repo_path)
            return stats

        file_count = 0
        for path in sorted(self._repo_path.rglob("*")):
            if file_count >= max_files:
                break

            # Skip unwanted directories
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue

            if path.is_file() and path.suffix in _CODE_EXTENSIONS:
                try:
                    relative = path.relative_to(self._repo_path)
                    stat = path.stat()
                    content = (
                        f"File: {relative}\n"
                        f"Language: {path.suffix}\n"
                        f"Size: {stat.st_size} bytes\n"
                        f"Directory: {relative.parent}"
                    )
                    result = await self._fabric.remember(
                        content=content,
                        source="codebase_structure",
                        metadata={
                            "file_path": str(relative),
                            "language": path.suffix,
                            "size": stat.st_size,
                            "directory": str(relative.parent),
                        },
                    )
                    if result.stored:
                        stats.items_ingested += 1
                    else:
                        stats.items_skipped += 1
                    file_count += 1
                except (OSError, ValueError) as exc:
                    logger.warning("Failed to ingest %s: %s", path, exc)
                    stats.errors += 1

        return stats

    async def ingest_imports(self, max_files: int = 200) -> IngestionStats:
        """Parse Python imports via AST, create import relationship entries.

        Returns:
            IngestionStats with counts
        """
        stats = IngestionStats()

        if not self._repo_path.is_dir():
            return stats

        file_count = 0
        for path in sorted(self._repo_path.rglob("*.py")):
            if file_count >= max_files:
                break

            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue

            try:
                source_code = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source_code)
                relative = str(path.relative_to(self._repo_path))

                imports = self._extract_imports(tree, relative)
                if imports:
                    import_summary = "\n".join(
                        f"  {rel.source_module} -> {rel.imported_module} ({rel.import_type})"
                        for rel in imports[:20]  # Limit per file
                    )
                    content = f"Import graph for {relative}:\n{import_summary}"
                    result = await self._fabric.remember(
                        content=content,
                        source="codebase_imports",
                        metadata={
                            "file_path": relative,
                            "import_count": len(imports),
                        },
                    )
                    if result.stored:
                        stats.items_ingested += 1
                    else:
                        stats.items_skipped += 1
                file_count += 1
            except (SyntaxError, OSError, ValueError) as exc:
                logger.debug("Failed to parse imports in %s: %s", path, exc)
                stats.errors += 1

        return stats

    def _extract_imports(self, tree: ast.AST, source_file: str) -> list[ImportRelation]:
        """Extract import relationships from an AST."""
        relations = []
        module_name = source_file.replace("/", ".").replace(".py", "")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    relations.append(ImportRelation(
                        source_module=module_name,
                        imported_module=alias.name,
                        import_type="import",
                    ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    relations.append(ImportRelation(
                        source_module=module_name,
                        imported_module=node.module,
                        import_type="from_import",
                    ))
        return relations

    async def ingest_test_results(self, results: dict[str, Any]) -> IngestionStats:
        """Ingest test results with surprise scoring.

        Args:
            results: Dict with keys like "passed", "failed", "errors", "test_details"
                     test_details is a list of {"name": str, "status": str, "duration": float}

        Returns:
            IngestionStats
        """
        stats = IngestionStats()

        # Summary entry
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        errors = results.get("errors", 0)
        total = passed + failed + errors

        summary = (
            f"Test results: {passed}/{total} passed, "
            f"{failed} failed, {errors} errors"
        )

        result = await self._fabric.remember(
            content=summary,
            source="codebase_tests",
            metadata={
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": total,
            },
        )
        if result.stored:
            stats.items_ingested += 1

        # Individual test details (focus on failures — they're more surprising)
        for detail in results.get("test_details", []):
            name = detail.get("name", "")
            status = detail.get("status", "")

            if status in ("failed", "error"):
                content = f"Test failure: {name} — status={status}"
                score = self._scorer.score(content, "test_result")

                result = await self._fabric.remember(
                    content=content,
                    source="codebase_test_failure",
                    metadata={
                        "test_name": name,
                        "status": status,
                        "surprise": score.combined,
                    },
                )
                if result.stored:
                    stats.items_ingested += 1
                else:
                    stats.items_skipped += 1

        return stats

    async def ingest_patterns(self, patterns: list[str]) -> IngestionStats:
        """Store architectural patterns as knowledge entries.

        Args:
            patterns: List of pattern descriptions (e.g., "handler pattern", "adapter factory")

        Returns:
            IngestionStats
        """
        stats = IngestionStats()

        for pattern in patterns:
            try:
                content = f"Architectural pattern: {pattern}"
                result = await self._fabric.remember(
                    content=content,
                    source="codebase_pattern",
                    metadata={"pattern": pattern},
                )
                if result.stored:
                    stats.items_ingested += 1
                else:
                    stats.items_skipped += 1
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Failed to ingest pattern '%s': %s", pattern, exc)
                stats.errors += 1

        return stats

    async def query_about(self, question: str, limit: int = 10) -> list[FabricResult]:
        """Natural language query about the codebase.

        Args:
            question: Natural language question
            limit: Max results

        Returns:
            List of FabricResult from the MemoryFabric
        """
        return await self._fabric.query(question, limit=limit)


__all__ = ["CodebaseKnowledgeBuilder", "IngestionStats", "ImportRelation"]
