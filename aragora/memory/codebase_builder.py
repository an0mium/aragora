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

from aragora.memory.fabric import FabricResult, MemoryFabric
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

    async def ingest_module_summaries(
        self,
        summarizer: Any | None = None,
        max_modules: int = 50,
    ) -> IngestionStats:
        """Ingest semantic summaries for top-level Python packages.

        For each top-level package under repo_path, extracts:
        - __init__.py docstring
        - Class and function names from *.py files
        - __all__ exports

        Optionally passes raw data to an LLM summarizer for richer descriptions.

        Args:
            summarizer: Optional callable(module_name, raw_info) -> str
            max_modules: Maximum number of modules to summarize

        Returns:
            IngestionStats
        """
        stats = IngestionStats()

        if not self._repo_path.is_dir():
            return stats

        module_count = 0
        for path in sorted(self._repo_path.iterdir()):
            if module_count >= max_modules:
                break

            if not path.is_dir() or path.name.startswith((".", "_")):
                continue

            # Only consider directories that look like Python packages
            init_path = path / "__init__.py"
            if not init_path.exists():
                continue

            try:
                summary = self._summarize_module(path)
                if not summary:
                    continue

                # Optionally enrich with LLM summarizer
                content = summary
                if summarizer is not None:
                    try:
                        enriched = summarizer(path.name, summary)
                        if enriched:
                            content = enriched
                    except (RuntimeError, ValueError, TypeError) as exc:
                        logger.debug("Summarizer failed for %s: %s", path.name, exc)

                result = await self._fabric.remember(
                    content=content,
                    source="codebase_module_summary",
                    metadata={
                        "module": path.name,
                        "type": "module_summary",
                    },
                )
                if result.stored:
                    stats.items_ingested += 1
                else:
                    stats.items_skipped += 1
                module_count += 1
            except (OSError, ValueError, SyntaxError) as exc:
                logger.warning("Failed to summarize module %s: %s", path.name, exc)
                stats.errors += 1

        return stats

    def _summarize_module(self, module_path: Path) -> str:
        """Build a structured summary of a Python module.

        Reads __init__.py docstring, scans *.py for class/def signatures,
        and collects __all__ exports.

        Args:
            module_path: Path to the module directory

        Returns:
            Formatted summary string, or empty string if nothing to summarize
        """
        parts: list[str] = [f"Module: {module_path.name}"]

        # Extract __init__.py docstring
        init_path = module_path / "__init__.py"
        if init_path.exists():
            try:
                source = init_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
                docstring = ast.get_docstring(tree)
                if docstring:
                    parts.append(f"Description: {docstring[:200]}")

                # Extract __all__
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "__all__"
                        and isinstance(node.value, ast.List)
                    ):
                        exports = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.append(elt.value)
                        if exports:
                            parts.append(f"Exports: {', '.join(exports[:20])}")
                        break
            except (SyntaxError, OSError):
                pass

        # Scan *.py files for class/function names (top-level only, skip __init__)
        names: list[str] = []
        py_files = sorted(module_path.glob("*.py"))
        for py_file in py_files[:30]:
            if py_file.name == "__init__.py":
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        names.append(f"class {node.name}")
                    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        if not node.name.startswith("_"):
                            names.append(f"def {node.name}")
            except (SyntaxError, OSError):
                continue

        if names:
            parts.append(f"Definitions: {', '.join(names[:30])}")

        if len(parts) <= 1:
            return ""

        return "\n".join(parts)

    async def ingest_dependency_graph(self, max_files: int = 200) -> IngestionStats:
        """Build and ingest a module-level dependency graph.

        Aggregates per-file imports into module-level dependency edges
        and stores them as knowledge items for codebase understanding.

        Args:
            max_files: Maximum number of Python files to analyze

        Returns:
            IngestionStats
        """
        stats = IngestionStats()

        if not self._repo_path.is_dir():
            return stats

        # Collect all import edges, aggregated to module level
        edges: dict[str, set[str]] = {}
        file_count = 0

        for path in sorted(self._repo_path.rglob("*.py")):
            if file_count >= max_files:
                break

            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue

            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
                relative = str(path.relative_to(self._repo_path))

                # Get source module (top-level package)
                parts = relative.replace("/", ".").replace(".py", "").split(".")
                source_module = parts[0] if parts else relative

                for node in ast.walk(tree):
                    target_module = None
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            target_module = alias.name.split(".")[0]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        target_module = node.module.split(".")[0]

                    if target_module and target_module != source_module:
                        if source_module not in edges:
                            edges[source_module] = set()
                        edges[source_module].add(target_module)

                file_count += 1
            except (SyntaxError, OSError, ValueError):
                stats.errors += 1

        # Store the dependency graph
        if edges:
            graph_lines = ["Module dependency graph:"]
            for source, targets in sorted(edges.items()):
                graph_lines.append(f"  {source} -> {', '.join(sorted(targets))}")

            content = "\n".join(graph_lines)
            result = await self._fabric.remember(
                content=content,
                source="codebase_dependency_graph",
                metadata={
                    "type": "dependency_graph",
                    "module_count": len(edges),
                    "edge_count": sum(len(t) for t in edges.values()),
                },
            )
            if result.stored:
                stats.items_ingested += 1
            else:
                stats.items_skipped += 1

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
