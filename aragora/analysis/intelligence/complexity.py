"""
Complexity Metrics Module.

Provides cyclomatic complexity calculation and other code complexity metrics
for analyzing code maintainability and quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import FileAnalysis, FunctionInfo, Language


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a code unit."""

    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    logical_lines: int = 0
    nesting_depth: int = 0


def calculate_cyclomatic_complexity(
    source: str,
    language: Language,
) -> int:
    """
    Calculate cyclomatic complexity for source code.

    Cyclomatic complexity measures the number of linearly independent paths
    through a program's source code.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        The cyclomatic complexity score (minimum 1)
    """
    complexity = 1

    if language == Language.PYTHON:
        # Count decision points
        complexity += source.count("if ")
        complexity += source.count("elif ")
        complexity += source.count("for ")
        complexity += source.count("while ")
        complexity += source.count("except ")
        complexity += source.count(" and ")
        complexity += source.count(" or ")
        complexity += source.count("case ")  # Match case (Python 3.10+)
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        complexity += len(re.findall(r"\bif\s*\(", source))
        complexity += len(re.findall(r"\belse\s+if\s*\(", source))
        complexity += len(re.findall(r"\bfor\s*\(", source))
        complexity += len(re.findall(r"\bwhile\s*\(", source))
        complexity += len(re.findall(r"\bcatch\s*\(", source))
        complexity += len(re.findall(r"\bcase\s+", source))
        complexity += len(re.findall(r"\?\s*[^:]+\s*:", source))  # Ternary
        complexity += source.count("&&")
        complexity += source.count("||")
    elif language == Language.GO:
        complexity += len(re.findall(r"\bif\s+", source))
        complexity += len(re.findall(r"\bfor\s+", source))
        complexity += len(re.findall(r"\bcase\s+", source))
        complexity += len(re.findall(r"\bselect\s+", source))
        complexity += source.count("&&")
        complexity += source.count("||")
    elif language == Language.RUST:
        complexity += len(re.findall(r"\bif\s+", source))
        complexity += len(re.findall(r"\belse\s+if\s+", source))
        complexity += len(re.findall(r"\bfor\s+", source))
        complexity += len(re.findall(r"\bwhile\s+", source))
        complexity += len(re.findall(r"\bloop\s*\{", source))
        complexity += len(re.findall(r"\bmatch\s+", source))
        complexity += source.count("&&")
        complexity += source.count("||")
    elif language == Language.JAVA:
        complexity += len(re.findall(r"\bif\s*\(", source))
        complexity += len(re.findall(r"\belse\s+if\s*\(", source))
        complexity += len(re.findall(r"\bfor\s*\(", source))
        complexity += len(re.findall(r"\bwhile\s*\(", source))
        complexity += len(re.findall(r"\bcatch\s*\(", source))
        complexity += len(re.findall(r"\bcase\s+", source))
        complexity += len(re.findall(r"\?\s*[^:]+\s*:", source))  # Ternary
        complexity += source.count("&&")
        complexity += source.count("||")

    return max(1, complexity)


def calculate_cognitive_complexity(
    source: str,
    language: Language,
) -> int:
    """
    Calculate cognitive complexity for source code.

    Cognitive complexity measures how difficult code is to understand,
    considering nesting depth and control flow breaks.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        The cognitive complexity score
    """
    complexity = 0
    nesting_level = 0
    lines = source.split("\n")

    # Language-specific patterns
    if language == Language.PYTHON:
        control_pattern = re.compile(r"^\s*(if|elif|for|while|with|try|except)\b")
        block_start = re.compile(r":\s*$")
        break_pattern = re.compile(r"\b(break|continue|return|raise)\b")
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.JAVA):
        control_pattern = re.compile(r"\b(if|else\s+if|for|while|switch|try|catch)\s*\(")
        block_start = re.compile(r"\{\s*$")
        break_pattern = re.compile(r"\b(break|continue|return|throw)\b")
    elif language == Language.GO:
        control_pattern = re.compile(r"\b(if|for|switch|select)\s+")
        block_start = re.compile(r"\{\s*$")
        break_pattern = re.compile(r"\b(break|continue|return|panic)\b")
    elif language == Language.RUST:
        control_pattern = re.compile(r"\b(if|for|while|loop|match)\s+")
        block_start = re.compile(r"\{\s*$")
        break_pattern = re.compile(r"\b(break|continue|return|panic!)\b")
    else:
        return 0

    for line in lines:
        stripped = line.strip()

        # Check for control flow
        if control_pattern.search(stripped):
            # Add 1 for the construct plus nesting level
            complexity += 1 + nesting_level

        # Check for logical operators (adds without nesting)
        if language == Language.PYTHON:
            complexity += stripped.count(" and ") + stripped.count(" or ")
        else:
            complexity += stripped.count("&&") + stripped.count("||")

        # Check for breaks in control flow
        if break_pattern.search(stripped) and nesting_level > 0:
            complexity += 1

        # Track nesting
        if block_start.search(stripped):
            nesting_level += 1
        if stripped == "}" or (language == Language.PYTHON and stripped.startswith("return")):
            nesting_level = max(0, nesting_level - 1)

    return complexity


def calculate_nesting_depth(source: str, language: Language) -> int:
    """
    Calculate the maximum nesting depth in source code.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        The maximum nesting depth
    """
    max_depth = 0
    current_depth = 0

    if language == Language.PYTHON:
        # Count by indentation
        lines = source.split("\n")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                # Assuming 4-space indent
                depth = indent // 4
                max_depth = max(max_depth, depth)
    else:
        # Count by braces
        for char in source:
            if char == "{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "}":
                current_depth = max(0, current_depth - 1)

    return max_depth


def get_complexity_metrics(
    source: str,
    language: Language,
) -> ComplexityMetrics:
    """
    Get all complexity metrics for source code.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        ComplexityMetrics with all calculated metrics
    """
    lines = source.split("\n")
    logical_lines = sum(
        1 for line in lines if line.strip() and not _is_comment_line(line, language)
    )

    return ComplexityMetrics(
        cyclomatic_complexity=calculate_cyclomatic_complexity(source, language),
        cognitive_complexity=calculate_cognitive_complexity(source, language),
        lines_of_code=len(lines),
        logical_lines=logical_lines,
        nesting_depth=calculate_nesting_depth(source, language),
    )


def _is_comment_line(line: str, language: Language) -> bool:
    """Check if a line is a comment."""
    stripped = line.strip()

    if language == Language.PYTHON:
        return stripped.startswith("#")
    elif language in (
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.GO,
        Language.RUST,
        Language.JAVA,
    ):
        return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")

    return False


def calculate_function_complexity(func: FunctionInfo, source: str | None = None) -> int:
    """
    Calculate or return complexity for a function.

    If source is provided, recalculates complexity. Otherwise returns
    the stored complexity value.

    Args:
        func: The function info
        source: Optional source code for recalculation

    Returns:
        The cyclomatic complexity score
    """
    if source:
        # Extract function body from source based on location
        lines = source.split("\n")
        start = func.location.start_line - 1
        end = func.location.end_line
        func_source = "\n".join(lines[start:end])
        return calculate_cyclomatic_complexity(func_source, Language.PYTHON)

    return func.complexity


def get_file_complexity_summary(analysis: FileAnalysis) -> dict[str, Any]:
    """
    Get a summary of complexity metrics for a file.

    Args:
        analysis: The file analysis result

    Returns:
        Dictionary with complexity summary
    """
    all_functions = list(analysis.functions)
    for cls in analysis.classes:
        all_functions.extend(cls.methods)

    if not all_functions:
        return {
            "total_complexity": 0,
            "average_complexity": 0.0,
            "max_complexity": 0,
            "high_complexity_functions": [],
            "function_count": 0,
        }

    complexities = [f.complexity for f in all_functions]
    high_threshold = 10  # Common threshold for high complexity

    return {
        "total_complexity": sum(complexities),
        "average_complexity": sum(complexities) / len(complexities),
        "max_complexity": max(complexities),
        "high_complexity_functions": [
            {"name": f.name, "complexity": f.complexity, "location": str(f.location)}
            for f in all_functions
            if f.complexity > high_threshold
        ],
        "function_count": len(all_functions),
    }


__all__ = [
    "ComplexityMetrics",
    "calculate_cyclomatic_complexity",
    "calculate_cognitive_complexity",
    "calculate_nesting_depth",
    "get_complexity_metrics",
    "calculate_function_complexity",
    "get_file_complexity_summary",
]
