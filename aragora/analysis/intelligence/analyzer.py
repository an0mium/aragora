"""
Code Intelligence Analyzer.

Main analyzer class that provides semantic code analysis with accurate
symbol extraction, type information, and structural understanding
across multiple programming languages.

Example:
    >>> from aragora.analysis.intelligence import CodeIntelligence
    >>> intel = CodeIntelligence()
    >>> analysis = intel.analyze_file("src/main.py")
    >>> print(f"Found {len(analysis.classes)} classes, {len(analysis.functions)} functions")
    >>> for cls in analysis.classes:
    ...     print(f"  - {cls.name} with {len(cls.methods)} methods")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .ast_analysis import TreeSitterParser, analyze_with_tree_sitter
from .call_graph import CallGraph, build_call_graph
from .complexity import get_file_complexity_summary
from .dead_code import DeadCodeReport, detect_dead_code
from .symbols import (
    analyze_with_regex,
    find_symbol_usages as _find_symbol_usages,
    get_symbol_at_location as _get_symbol_at_location,
)
from .types import (
    ClassInfo,
    FileAnalysis,
    FunctionInfo,
    Language,
    SourceLocation,
)

logger = logging.getLogger(__name__)


class CodeIntelligence:
    """
    Semantic code analysis engine.

    Uses tree-sitter for accurate AST parsing when available,
    with regex fallback for basic symbol extraction.

    Example:
        intel = CodeIntelligence()

        # Analyze a single file
        analysis = intel.analyze_file("src/main.py")
        print(f"Classes: {[c.name for c in analysis.classes]}")

        # Analyze a directory
        results = intel.analyze_directory("src/")
        for path, analysis in results.items():
            print(f"{path}: {len(analysis.functions)} functions")

        # Find all usages of a symbol
        usages = intel.find_symbol_usages("src/", "MyClass")
    """

    def __init__(self) -> None:
        self._parser = TreeSitterParser()

    @property
    def tree_sitter_available(self) -> bool:
        """Check if tree-sitter is available for enhanced parsing."""
        return self._parser.available

    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a single source file.

        Args:
            file_path: Path to the source file

        Returns:
            FileAnalysis with extracted information
        """
        path = Path(file_path)
        language = Language.from_extension(path.suffix)

        analysis = FileAnalysis(
            file_path=str(path),
            language=language,
        )

        if not path.exists():
            analysis.errors.append(f"File not found: {file_path}")
            return analysis

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            analysis.errors.append(f"Failed to read file: {e}")
            return analysis

        # Count lines
        lines = content.split("\n")
        analysis.lines_of_code = len(lines)
        analysis.blank_lines = sum(1 for line in lines if not line.strip())
        analysis.comment_lines = self._count_comment_lines(content, language)

        # Try tree-sitter parsing first
        if self._parser.supports(language):
            tree = self._parser.parse(content.encode("utf-8"), language)
            if tree:
                analyze_with_tree_sitter(tree, content, analysis, language)
                return analysis

        # Fallback to regex-based parsing
        analyze_with_regex(content, analysis, language)
        return analysis

    def analyze_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, FileAnalysis]:
        """
        Analyze all source files in a directory.

        Args:
            directory: Root directory to analyze
            extensions: File extensions to include (default: all supported)
            exclude_patterns: Glob patterns to exclude

        Returns:
            Dictionary mapping file paths to their analysis
        """
        results = {}
        root = Path(directory)
        exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/vendor/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
        ]

        # Default extensions
        if extensions is None:
            extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"]

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix not in extensions:
                continue

            # Check exclusions
            path_str = str(path)
            excluded = False
            for pattern in exclude_patterns:
                if path.match(pattern):
                    excluded = True
                    break
            if excluded:
                continue

            try:
                results[path_str] = self.analyze_file(path_str)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                logger.warning("Failed to analyze %s: %s", path, e)

        return results

    def find_symbol_usages(
        self,
        directory: str,
        symbol_name: str,
        language: Language | None = None,
    ) -> list[SourceLocation]:
        """
        Find all usages of a symbol across the codebase.

        Args:
            directory: Directory to search
            symbol_name: Name of the symbol to find
            language: Optional language filter

        Returns:
            List of source locations where the symbol is used
        """
        return _find_symbol_usages(directory, symbol_name, language)

    def get_symbol_at_location(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> FunctionInfo | ClassInfo | None:
        """
        Get the symbol at a specific location.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            The symbol at that location, or None
        """
        analysis = self.analyze_file(file_path)
        return _get_symbol_at_location(analysis, line, column)

    def build_call_graph(
        self,
        directory: str | None = None,
        analyses: dict[str, FileAnalysis] | None = None,
    ) -> CallGraph:
        """
        Build a call graph for the codebase.

        Args:
            directory: Directory to analyze (if analyses not provided)
            analyses: Pre-computed file analyses

        Returns:
            CallGraph representing function dependencies
        """
        if analyses is None:
            if directory is None:
                raise ValueError("Either directory or analyses must be provided")
            analyses = self.analyze_directory(directory)

        return build_call_graph(analyses)

    def detect_dead_code(
        self,
        directory: str | None = None,
        analyses: dict[str, FileAnalysis] | None = None,
        entry_points: list[str] | None = None,
    ) -> DeadCodeReport:
        """
        Detect dead code in the codebase.

        Args:
            directory: Directory to analyze (if analyses not provided)
            analyses: Pre-computed file analyses
            entry_points: Known entry point functions

        Returns:
            DeadCodeReport with findings
        """
        if analyses is None:
            if directory is None:
                raise ValueError("Either directory or analyses must be provided")
            analyses = self.analyze_directory(directory)

        return detect_dead_code(analyses, entry_points=entry_points)

    def get_complexity_report(
        self,
        directory: str | None = None,
        analyses: dict[str, FileAnalysis] | None = None,
    ) -> dict[str, Any]:
        """
        Get complexity metrics for the codebase.

        Args:
            directory: Directory to analyze (if analyses not provided)
            analyses: Pre-computed file analyses

        Returns:
            Dictionary with complexity metrics per file
        """
        if analyses is None:
            if directory is None:
                raise ValueError("Either directory or analyses must be provided")
            analyses = self.analyze_directory(directory)

        report = {}
        for file_path, analysis in analyses.items():
            report[file_path] = get_file_complexity_summary(analysis)

        return report

    def _count_comment_lines(self, source: str, language: Language) -> int:
        """Count comment lines in source code."""
        count = 0
        lines = source.split("\n")

        in_block_comment = False
        block_start = "/*"
        block_end = "*/"
        line_comment = "//"

        if language == Language.PYTHON:
            block_start = '"""'
            block_end = '"""'
            line_comment = "#"

        for line in lines:
            stripped = line.strip()

            if in_block_comment:
                count += 1
                if block_end in stripped:
                    in_block_comment = False
            elif stripped.startswith(block_start):
                count += 1
                if block_end not in stripped[len(block_start) :]:
                    in_block_comment = True
            elif stripped.startswith(line_comment):
                count += 1

        return count


__all__ = [
    "CodeIntelligence",
]
