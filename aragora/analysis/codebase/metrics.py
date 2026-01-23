"""
Code Quality Metrics Module.

Provides tools for analyzing code complexity, maintainability, and quality metrics.
Includes cyclomatic complexity, cognitive complexity, and code duplication detection.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import CodeMetric, HotspotFinding, MetricType


@dataclass
class FunctionMetrics:
    """Metrics for a single function or method."""

    name: str
    file_path: str
    start_line: int
    end_line: int
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    parameter_count: int = 0
    return_count: int = 0
    nested_depth: int = 0
    class_name: Optional[str] = None


@dataclass
class FileMetrics:
    """Metrics for a single file."""

    file_path: str
    language: str
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    functions: List[FunctionMetrics] = field(default_factory=list)
    classes: int = 0
    imports: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 100.0


@dataclass
class DuplicateBlock:
    """A duplicated code block."""

    hash: str
    lines: int
    occurrences: List[Tuple[str, int, int]]  # [(file_path, start_line, end_line), ...]


@dataclass
class MetricsReport:
    """Aggregated metrics report for a codebase."""

    repository: str
    scan_id: str
    scanned_at: datetime = field(default_factory=datetime.now)
    total_files: int = 0
    total_lines: int = 0
    total_code_lines: int = 0
    total_comment_lines: int = 0
    total_blank_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    maintainability_index: float = 100.0
    files: List[FileMetrics] = field(default_factory=list)
    hotspots: List[HotspotFinding] = field(default_factory=list)
    duplicates: List[DuplicateBlock] = field(default_factory=list)
    metrics: List[CodeMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "repository": self.repository,
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "summary": {
                "total_files": self.total_files,
                "total_lines": self.total_lines,
                "total_code_lines": self.total_code_lines,
                "total_comment_lines": self.total_comment_lines,
                "total_blank_lines": self.total_blank_lines,
                "total_functions": self.total_functions,
                "total_classes": self.total_classes,
                "avg_complexity": round(self.avg_complexity, 2),
                "max_complexity": self.max_complexity,
                "maintainability_index": round(self.maintainability_index, 2),
            },
            "hotspots": [h.to_dict() for h in self.hotspots],
            "duplicates": [
                {
                    "hash": d.hash[:8],
                    "lines": d.lines,
                    "occurrences": [
                        {"file": occ[0], "start": occ[1], "end": occ[2]} for occ in d.occurrences
                    ],
                }
                for d in self.duplicates
            ],
            "metrics": [m.to_dict() for m in self.metrics],
        }


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for calculating complexity metrics."""

    def __init__(self) -> None:
        self.cyclomatic = 1  # Start at 1 for the function itself
        self.cognitive = 0
        self.nesting_level = 0
        self.return_count = 0
        self.max_nesting = 0

    def visit_If(self, node: ast.If) -> None:
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_For(self, node: ast.For) -> None:
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_While(self, node: ast.While) -> None:
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each additional and/or adds to cyclomatic complexity
        self.cyclomatic += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.cyclomatic += 1
        self.cognitive += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        # Python 3.10+ match statement
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)
        self.generic_visit(node)
        self.nesting_level -= 1

    def visit_match_case(self, node: ast.match_case) -> None:
        self.cyclomatic += 1
        self.cognitive += 1 + self.nesting_level
        self.generic_visit(node)


class PythonAnalyzer:
    """Analyzer for Python code metrics."""

    def __init__(self) -> None:
        pass

    def analyze_file(self, file_path: str, content: Optional[str] = None) -> FileMetrics:
        """Analyze a Python file for complexity metrics."""
        if content is None:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        lines = content.split("\n")
        metrics = FileMetrics(
            file_path=file_path,
            language="python",
        )

        # Count lines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("#"):
                metrics.lines_of_comments += 1
            else:
                metrics.lines_of_code += 1

        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return metrics

        # Analyze structure
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                metrics.imports += 1
            elif isinstance(node, ast.ClassDef):
                metrics.classes += 1

        # Analyze functions and methods
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_metrics = self._analyze_function(node, file_path)
                metrics.functions.append(func_metrics)

        # Calculate aggregates
        if metrics.functions:
            complexities = [f.cyclomatic_complexity for f in metrics.functions]
            metrics.avg_complexity = sum(complexities) / len(complexities)
            metrics.max_complexity = max(complexities)

        # Calculate maintainability index
        # Based on Halstead Volume, Cyclomatic Complexity, and Lines of Code
        # Simplified version using available metrics
        if metrics.lines_of_code > 0 and metrics.avg_complexity > 0:
            import math

            loc = metrics.lines_of_code
            cc = metrics.avg_complexity
            # Simplified maintainability index formula
            mi = max(
                0,
                (171 - 5.2 * math.log(loc + 1) - 0.23 * cc - 16.2 * math.log(loc + 1)) * 100 / 171,
            )
            metrics.maintainability_index = min(100, mi)

        return metrics

    def _analyze_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str
    ) -> FunctionMetrics:
        """Analyze a single function for complexity."""
        visitor = ComplexityVisitor()
        visitor.visit(node)

        # Get class name if this is a method
        class_name = None
        if hasattr(node, "_class_name"):
            class_name = node._class_name

        # Count lines
        if hasattr(node, "end_lineno"):
            lines = node.end_lineno - node.lineno + 1
        else:
            lines = len(ast.unparse(node).split("\n"))

        return FunctionMetrics(
            name=node.name,
            file_path=file_path,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno + lines - 1),
            lines_of_code=lines,
            cyclomatic_complexity=visitor.cyclomatic,
            cognitive_complexity=visitor.cognitive,
            parameter_count=len(node.args.args) + len(node.args.kwonlyargs),
            return_count=visitor.return_count,
            nested_depth=visitor.max_nesting,
            class_name=class_name,
        )


class TypeScriptAnalyzer:
    """Basic analyzer for TypeScript/JavaScript metrics."""

    # Patterns for complexity analysis
    DECISION_PATTERNS = [
        r"\bif\s*\(",
        r"\belse\s+if\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\bswitch\s*\(",
        r"\bcase\s+",
        r"\bcatch\s*\(",
        r"\?\s*[^:]+:",  # Ternary operator
        r"\?\.",  # Optional chaining
        r"\|\|",  # Logical OR
        r"&&",  # Logical AND
    ]

    def __init__(self) -> None:
        pass

    def analyze_file(self, file_path: str, content: Optional[str] = None) -> FileMetrics:
        """Analyze a TypeScript/JavaScript file for metrics."""
        if content is None:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        lines = content.split("\n")
        metrics = FileMetrics(
            file_path=file_path,
            language="typescript" if file_path.endswith(".ts") else "javascript",
        )

        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()

            # Handle multi-line comments
            if "/*" in stripped:
                in_multiline_comment = True
            if "*/" in stripped:
                in_multiline_comment = False
                metrics.lines_of_comments += 1
                continue

            if in_multiline_comment:
                metrics.lines_of_comments += 1
            elif not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("//"):
                metrics.lines_of_comments += 1
            else:
                metrics.lines_of_code += 1

        # Count imports
        metrics.imports = len(re.findall(r"^import\s+", content, re.MULTILINE))

        # Count classes
        metrics.classes = len(re.findall(r"\bclass\s+\w+", content))

        # Analyze functions
        function_pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?:=>|{)|(\w+)\s*\([^)]*\)\s*{)"
        for match in re.finditer(function_pattern, content):
            func_name = match.group(1) or match.group(2) or match.group(3) or "anonymous"
            start_line = content[: match.start()].count("\n") + 1

            # Simple complexity estimation based on patterns
            # Find the function body (simplified)
            func_start = match.start()
            brace_count = 0
            func_end = func_start
            for i, char in enumerate(content[func_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = func_start + i
                        break

            func_body = content[func_start:func_end]
            end_line = content[: func_end + 1].count("\n") + 1

            # Count complexity
            complexity = 1  # Base complexity
            for pattern in self.DECISION_PATTERNS:
                complexity += len(re.findall(pattern, func_body))

            func_metrics = FunctionMetrics(
                name=func_name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                lines_of_code=end_line - start_line + 1,
                cyclomatic_complexity=complexity,
                cognitive_complexity=complexity,  # Simplified
            )
            metrics.functions.append(func_metrics)

        if metrics.functions:
            complexities = [f.cyclomatic_complexity for f in metrics.functions]
            metrics.avg_complexity = sum(complexities) / len(complexities)
            metrics.max_complexity = max(complexities)

        return metrics


class DuplicateDetector:
    """Detects duplicate code blocks."""

    def __init__(self, min_lines: int = 6, min_tokens: int = 50) -> None:
        self.min_lines = min_lines
        self.min_tokens = min_tokens

    def detect_duplicates(self, files: List[Tuple[str, str]]) -> List[DuplicateBlock]:
        """Detect duplicate code blocks across files.

        Args:
            files: List of (file_path, content) tuples

        Returns:
            List of duplicate blocks found
        """
        # Hash blocks of code
        block_hashes: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)

        for file_path, content in files:
            lines = content.split("\n")

            for start in range(len(lines) - self.min_lines + 1):
                end = start + self.min_lines
                block_lines = lines[start:end]

                # Normalize the block (remove whitespace variations)
                normalized = "\n".join(line.strip() for line in block_lines if line.strip())

                if len(normalized) < self.min_tokens:
                    continue

                block_hash = hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()
                block_hashes[block_hash].append((file_path, start + 1, end, normalized))

        # Find duplicates
        duplicates = []
        for hash_val, occurrences in block_hashes.items():
            if len(occurrences) > 1:
                # Verify they're actually duplicates (not just hash collisions)
                unique_files = set(occ[0] for occ in occurrences)
                if len(unique_files) > 1 or len(occurrences) > 2:
                    duplicates.append(
                        DuplicateBlock(
                            hash=hash_val,
                            lines=self.min_lines,
                            occurrences=[(occ[0], occ[1], occ[2]) for occ in occurrences],
                        )
                    )

        return duplicates


class CodeMetricsAnalyzer:
    """Main analyzer for code metrics across a codebase."""

    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    }

    EXCLUDE_DIRS = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        "coverage",
        ".pytest_cache",
    }

    def __init__(
        self,
        complexity_warning: int = 10,
        complexity_error: int = 20,
        duplication_threshold: int = 6,
    ) -> None:
        self.complexity_warning = complexity_warning
        self.complexity_error = complexity_error
        self.python_analyzer = PythonAnalyzer()
        self.ts_analyzer = TypeScriptAnalyzer()
        self.duplicate_detector = DuplicateDetector(min_lines=duplication_threshold)

    def analyze_repository(
        self,
        repo_path: str,
        scan_id: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> MetricsReport:
        """Analyze an entire repository for code metrics.

        Args:
            repo_path: Path to the repository root
            scan_id: Optional scan identifier
            include_patterns: Glob patterns to include (default: all supported)
            exclude_patterns: Glob patterns to exclude

        Returns:
            MetricsReport with all metrics
        """
        scan_id = scan_id or f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        repo_name = os.path.basename(os.path.abspath(repo_path))

        report = MetricsReport(
            repository=repo_name,
            scan_id=scan_id,
        )

        files_to_analyze: List[Tuple[str, str]] = []

        # Walk directory and collect files
        for root, dirs, files in os.walk(repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]

            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, repo_path)

                # Check extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.LANGUAGE_EXTENSIONS:
                    continue

                # Check exclude patterns
                if exclude_patterns:
                    skip = False
                    for pattern in exclude_patterns:
                        if Path(rel_path).match(pattern):
                            skip = True
                            break
                    if skip:
                        continue

                # Check include patterns
                if include_patterns:
                    included = False
                    for pattern in include_patterns:
                        if Path(rel_path).match(pattern):
                            included = True
                            break
                    if not included:
                        continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    files_to_analyze.append((rel_path, content))
                except (OSError, IOError):
                    continue

        # Analyze each file
        for rel_path, content in files_to_analyze:
            ext = os.path.splitext(rel_path)[1].lower()
            language = self.LANGUAGE_EXTENSIONS.get(ext, "unknown")

            if language == "python":
                file_metrics = self.python_analyzer.analyze_file(rel_path, content)
            elif language in ("typescript", "javascript"):
                file_metrics = self.ts_analyzer.analyze_file(rel_path, content)
            else:
                # Basic line counting for other languages
                file_metrics = self._basic_analyze(rel_path, content, language)

            report.files.append(file_metrics)
            report.total_files += 1
            report.total_lines += (
                file_metrics.lines_of_code
                + file_metrics.lines_of_comments
                + file_metrics.blank_lines
            )
            report.total_code_lines += file_metrics.lines_of_code
            report.total_comment_lines += file_metrics.lines_of_comments
            report.total_blank_lines += file_metrics.blank_lines
            report.total_functions += len(file_metrics.functions)
            report.total_classes += file_metrics.classes

        # Calculate aggregate metrics
        all_complexities = []
        for file_metrics in report.files:
            for func in file_metrics.functions:
                all_complexities.append(func.cyclomatic_complexity)

        if all_complexities:
            report.avg_complexity = sum(all_complexities) / len(all_complexities)
            report.max_complexity = max(all_complexities)

        # Calculate overall maintainability
        if report.files:
            mi_scores = [
                f.maintainability_index for f in report.files if f.maintainability_index > 0
            ]
            if mi_scores:
                report.maintainability_index = sum(mi_scores) / len(mi_scores)

        # Detect duplicates
        report.duplicates = self.duplicate_detector.detect_duplicates(files_to_analyze)

        # Find hotspots
        report.hotspots = self._find_hotspots(report)

        # Generate CodeMetric objects
        report.metrics = self._generate_metrics(report)

        return report

    def _basic_analyze(self, file_path: str, content: str, language: str) -> FileMetrics:
        """Basic analysis for unsupported languages."""
        lines = content.split("\n")
        metrics = FileMetrics(
            file_path=file_path,
            language=language,
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith(("//", "#", "*", "/*", "*/")):
                metrics.lines_of_comments += 1
            else:
                metrics.lines_of_code += 1

        return metrics

    def _find_hotspots(self, report: MetricsReport, top_n: int = 10) -> List[HotspotFinding]:
        """Find complexity hotspots."""
        hotspots = []

        for file_metrics in report.files:
            for func in file_metrics.functions:
                if func.cyclomatic_complexity >= self.complexity_warning:
                    hotspot = HotspotFinding(
                        file_path=func.file_path,
                        function_name=func.name,
                        class_name=func.class_name,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        complexity=func.cyclomatic_complexity,
                        lines_of_code=func.lines_of_code,
                        cognitive_complexity=func.cognitive_complexity,
                    )
                    hotspots.append(hotspot)

        # Sort by risk score and return top N
        hotspots.sort(key=lambda h: h.risk_score, reverse=True)
        return hotspots[:top_n]

    def _generate_metrics(self, report: MetricsReport) -> List[CodeMetric]:
        """Generate CodeMetric objects from the report."""
        metrics = []

        # Overall complexity metric
        metrics.append(
            CodeMetric(
                type=MetricType.COMPLEXITY,
                value=report.avg_complexity,
                unit="cyclomatic",
                warning_threshold=self.complexity_warning,
                error_threshold=self.complexity_error,
                details={"max": report.max_complexity},
            )
        )

        # Maintainability index
        metrics.append(
            CodeMetric(
                type=MetricType.MAINTAINABILITY,
                value=report.maintainability_index,
                unit="index",
                warning_threshold=65,
                error_threshold=50,
            )
        )

        # Lines of code
        metrics.append(
            CodeMetric(
                type=MetricType.LINES_OF_CODE,
                value=report.total_code_lines,
                unit="lines",
                details={
                    "comments": report.total_comment_lines,
                    "blank": report.total_blank_lines,
                    "total": report.total_lines,
                },
            )
        )

        # Documentation ratio
        if report.total_code_lines > 0:
            doc_ratio = report.total_comment_lines / report.total_code_lines * 100
            metrics.append(
                CodeMetric(
                    type=MetricType.DOCUMENTATION,
                    value=doc_ratio,
                    unit="percent",
                    warning_threshold=5,
                    error_threshold=1,
                )
            )

        # Duplication
        duplicate_lines = sum(d.lines * (len(d.occurrences) - 1) for d in report.duplicates)
        if report.total_code_lines > 0:
            dup_ratio = duplicate_lines / report.total_code_lines * 100
            metrics.append(
                CodeMetric(
                    type=MetricType.DUPLICATION,
                    value=dup_ratio,
                    unit="percent",
                    warning_threshold=5,
                    error_threshold=10,
                    details={"duplicate_blocks": len(report.duplicates)},
                )
            )

        return metrics
