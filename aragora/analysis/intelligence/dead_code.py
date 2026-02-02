"""
Dead Code Detection Module.

Provides analysis for detecting unused code, unreachable code,
and redundant code patterns in source files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .call_graph import CallGraph, build_call_graph
from .symbols import get_exported_symbols
from .types import ClassInfo, FileAnalysis, FunctionInfo, Language, SourceLocation


@dataclass
class DeadCodeFinding:
    """A finding of potentially dead code."""

    kind: str  # 'unused_function', 'unused_class', 'unreachable_code', etc.
    name: str
    location: SourceLocation
    reason: str
    confidence: float  # 0.0 to 1.0


@dataclass
class DeadCodeReport:
    """Report of dead code analysis."""

    findings: list[DeadCodeFinding] = field(default_factory=list)
    analyzed_files: int = 0
    total_functions: int = 0
    unused_functions: int = 0
    total_classes: int = 0
    unused_classes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "findings": [
                {
                    "kind": f.kind,
                    "name": f.name,
                    "location": str(f.location),
                    "reason": f.reason,
                    "confidence": f.confidence,
                }
                for f in self.findings
            ],
            "analyzed_files": self.analyzed_files,
            "total_functions": self.total_functions,
            "unused_functions": self.unused_functions,
            "total_classes": self.total_classes,
            "unused_classes": self.unused_classes,
        }


def detect_dead_code(
    analyses: dict[str, FileAnalysis],
    call_graph: CallGraph | None = None,
    entry_points: list[str] | None = None,
) -> DeadCodeReport:
    """
    Detect dead code across analyzed files.

    Args:
        analyses: Dictionary of file analyses
        call_graph: Optional pre-built call graph
        entry_points: Optional list of known entry point functions

    Returns:
        DeadCodeReport with findings
    """
    report = DeadCodeReport()
    report.analyzed_files = len(analyses)

    # Build call graph if not provided
    if call_graph is None:
        call_graph = build_call_graph(analyses)

    # Collect all functions and classes
    all_functions: dict[str, FunctionInfo] = {}
    all_classes: dict[str, ClassInfo] = {}
    exported_symbols: set[str] = set()

    for file_path, analysis in analyses.items():
        # Track exports
        exported_symbols.update(get_exported_symbols(analysis))

        for func in analysis.functions:
            all_functions[func.name] = func
            report.total_functions += 1

        for cls in analysis.classes:
            all_classes[cls.name] = cls
            report.total_classes += 1
            for method in cls.methods:
                all_functions[f"{cls.name}.{method.name}"] = method
                report.total_functions += 1

    # Default entry points include special Python functions
    default_entry_points = {
        "main",
        "__main__",
        "__init__",
        "setup",
        "teardown",
        "setUp",
        "tearDown",
        "test_*",
    }
    if entry_points:
        default_entry_points.update(entry_points)

    # Find unused functions
    for name, func in all_functions.items():
        callers = call_graph.get_callers(name)
        base_name = name.split(".")[-1]

        # Skip if it's an entry point
        is_entry_point = (
            name in default_entry_points
            or base_name in default_entry_points
            or any(
                base_name.startswith(ep.rstrip("*"))
                for ep in default_entry_points
                if ep.endswith("*")
            )
        )

        # Skip if exported
        is_exported = name in exported_symbols or base_name in exported_symbols

        # Skip special methods
        is_special = base_name.startswith("__") and base_name.endswith("__")

        # Skip if it has callers
        if not callers and not is_entry_point and not is_exported and not is_special:
            # Check if it might be used dynamically
            confidence = 0.8 if func.visibility.value == "private" else 0.6

            report.findings.append(
                DeadCodeFinding(
                    kind="unused_function",
                    name=name,
                    location=func.location,
                    reason=f"Function '{name}' appears to have no callers",
                    confidence=confidence,
                )
            )
            report.unused_functions += 1

    # Find unused classes
    for name, cls in all_classes.items():
        # Check if class is instantiated or inherited
        is_used = _is_class_used(name, analyses)

        # Skip if exported
        is_exported = name in exported_symbols

        if not is_used and not is_exported and not name.startswith("_"):
            report.findings.append(
                DeadCodeFinding(
                    kind="unused_class",
                    name=name,
                    location=cls.location,
                    reason=f"Class '{name}' appears to have no usages",
                    confidence=0.7,
                )
            )
            report.unused_classes += 1

    return report


def _is_class_used(class_name: str, analyses: dict[str, FileAnalysis]) -> bool:
    """Check if a class is used anywhere."""
    for file_path, analysis in analyses.items():
        # Check if used as base class
        for cls in analysis.classes:
            if class_name in cls.bases:
                return True

        # Check if imported
        for imp in analysis.imports:
            if class_name in imp.names:
                return True

    # Would need source code analysis for instantiation checks
    return False


def find_unreachable_code(
    source: str,
    language: Language,
) -> list[DeadCodeFinding]:
    """
    Find unreachable code in source.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        List of unreachable code findings
    """
    findings = []
    lines = source.split("\n")

    if language == Language.PYTHON:
        findings.extend(_find_unreachable_python(lines))
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        findings.extend(_find_unreachable_js(lines))

    return findings


def _find_unreachable_python(lines: list[str]) -> list[DeadCodeFinding]:
    """Find unreachable code in Python source."""
    findings = []
    in_function = False
    function_indent = 0
    after_return = False
    return_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # Track function boundaries
        if re.match(r"^\s*(?:async\s+)?def\s+", line):
            in_function = True
            function_indent = indent
            after_return = False
            continue

        if in_function:
            # Check if we're still in the function
            if stripped and indent <= function_indent:
                in_function = False
                after_return = False
                continue

            # Check for return statements
            if re.match(r"^\s*(return|raise|exit\(\))\b", line):
                after_return = True
                return_line = i
                continue

            # Check for code after return
            if after_return and stripped and not stripped.startswith("#"):
                # Check if it's at the same indentation level
                if indent > function_indent:
                    findings.append(
                        DeadCodeFinding(
                            kind="unreachable_code",
                            name="code_after_return",
                            location=SourceLocation(
                                file_path="",
                                start_line=i,
                                start_column=0,
                                end_line=i,
                                end_column=len(line),
                            ),
                            reason=f"Code after return/raise statement at line {return_line}",
                            confidence=0.9,
                        )
                    )
                after_return = False

    return findings


def _find_unreachable_js(lines: list[str]) -> list[DeadCodeFinding]:
    """Find unreachable code in JavaScript source."""
    findings = []
    after_return = False
    return_line = 0
    brace_count = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track braces to know scope
        brace_count += line.count("{") - line.count("}")

        # Check for return/throw statements
        if re.match(r"^\s*(return|throw)\b", line):
            after_return = True
            return_line = i
            continue

        # Check for code after return (before closing brace)
        if after_return and stripped and not stripped.startswith("//"):
            if stripped == "}":
                after_return = False
                continue

            findings.append(
                DeadCodeFinding(
                    kind="unreachable_code",
                    name="code_after_return",
                    location=SourceLocation(
                        file_path="",
                        start_line=i,
                        start_column=0,
                        end_line=i,
                        end_column=len(line),
                    ),
                    reason=f"Code after return/throw statement at line {return_line}",
                    confidence=0.9,
                )
            )
            after_return = False

    return findings


def find_unused_imports(analysis: FileAnalysis, source: str) -> list[DeadCodeFinding]:
    """
    Find unused imports in a file.

    Args:
        analysis: The file analysis
        source: The source code text

    Returns:
        List of unused import findings
    """
    findings = []

    for imp in analysis.imports:
        # Check each imported name
        names_to_check = imp.names if imp.names else [imp.module.split(".")[-1]]

        for name in names_to_check:
            # Count occurrences of the name (excluding the import line itself)
            pattern = rf"\b{re.escape(name)}\b"
            matches = list(re.finditer(pattern, source))

            # If only one match (the import itself), it's unused
            if len(matches) <= 1:
                findings.append(
                    DeadCodeFinding(
                        kind="unused_import",
                        name=name,
                        location=imp.location or SourceLocation(analysis.file_path, 0, 0, 0, 0),
                        reason=f"Import '{name}' appears to be unused",
                        confidence=0.85,
                    )
                )

    return findings


def find_unused_variables(source: str, language: Language) -> list[DeadCodeFinding]:
    """
    Find unused local variables in source.

    Args:
        source: The source code text
        language: The programming language

    Returns:
        List of unused variable findings
    """
    findings = []

    if language == Language.PYTHON:
        findings.extend(_find_unused_vars_python(source))
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        findings.extend(_find_unused_vars_js(source))

    return findings


def _find_unused_vars_python(source: str) -> list[DeadCodeFinding]:
    """Find unused variables in Python source."""
    findings = []
    lines = source.split("\n")

    # Simple assignment pattern
    assignment_pattern = re.compile(r"^\s*(\w+)\s*=\s*")

    for i, line in enumerate(lines, 1):
        match = assignment_pattern.match(line)
        if match:
            var_name = match.group(1)

            # Skip if it's a special variable or looks like a constant
            if var_name.startswith("_") or var_name.isupper():
                continue

            # Count usages in the rest of the file
            pattern = rf"\b{re.escape(var_name)}\b"
            usage_count = len(list(re.finditer(pattern, source)))

            # If only one occurrence (the assignment), it's unused
            if usage_count == 1:
                findings.append(
                    DeadCodeFinding(
                        kind="unused_variable",
                        name=var_name,
                        location=SourceLocation(
                            file_path="",
                            start_line=i,
                            start_column=0,
                            end_line=i,
                            end_column=len(line),
                        ),
                        reason=f"Variable '{var_name}' is assigned but never used",
                        confidence=0.7,
                    )
                )

    return findings


def _find_unused_vars_js(source: str) -> list[DeadCodeFinding]:
    """Find unused variables in JavaScript source."""
    findings = []
    lines = source.split("\n")

    # Variable declaration patterns
    decl_pattern = re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=")

    for i, line in enumerate(lines, 1):
        match = decl_pattern.match(line)
        if match:
            var_name = match.group(1)

            # Count usages
            pattern = rf"\b{re.escape(var_name)}\b"
            usage_count = len(list(re.finditer(pattern, source)))

            if usage_count == 1:
                findings.append(
                    DeadCodeFinding(
                        kind="unused_variable",
                        name=var_name,
                        location=SourceLocation(
                            file_path="",
                            start_line=i,
                            start_column=0,
                            end_line=i,
                            end_column=len(line),
                        ),
                        reason=f"Variable '{var_name}' is declared but never used",
                        confidence=0.75,
                    )
                )

    return findings


__all__ = [
    "DeadCodeFinding",
    "DeadCodeReport",
    "detect_dead_code",
    "find_unreachable_code",
    "find_unused_imports",
    "find_unused_variables",
]
