"""
Call Graph Analysis Module.

Provides call graph construction and analysis for understanding
function dependencies and call relationships in source code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import FileAnalysis, FunctionInfo, Language


@dataclass
class CallSite:
    """A function call site."""

    caller: str
    callee: str
    file_path: str
    line: int
    column: int


@dataclass
class CallGraph:
    """A call graph representing function dependencies."""

    nodes: set[str] = field(default_factory=set)
    edges: list[CallSite] = field(default_factory=list)
    callers: dict[str, list[str]] = field(default_factory=dict)
    callees: dict[str, list[str]] = field(default_factory=dict)

    def add_node(self, name: str) -> None:
        """Add a function node to the graph."""
        self.nodes.add(name)
        if name not in self.callers:
            self.callers[name] = []
        if name not in self.callees:
            self.callees[name] = []

    def add_edge(self, call_site: CallSite) -> None:
        """Add a call edge to the graph."""
        self.edges.append(call_site)

        # Add nodes if not present
        self.add_node(call_site.caller)
        self.add_node(call_site.callee)

        # Update caller/callee relationships
        if call_site.callee not in self.callees[call_site.caller]:
            self.callees[call_site.caller].append(call_site.callee)
        if call_site.caller not in self.callers[call_site.callee]:
            self.callers[call_site.callee].append(call_site.caller)

    def get_callers(self, name: str) -> list[str]:
        """Get all functions that call the given function."""
        return self.callers.get(name, [])

    def get_callees(self, name: str) -> list[str]:
        """Get all functions called by the given function."""
        return self.callees.get(name, [])

    def get_reachable_from(self, name: str) -> set[str]:
        """Get all functions reachable from the given function."""
        visited = set()
        stack = [name]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.callees.get(current, []))

        return visited

    def get_reaching(self, name: str) -> set[str]:
        """Get all functions that can reach the given function."""
        visited = set()
        stack = [name]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.callers.get(current, []))

        return visited

    def find_cycles(self) -> list[list[str]]:
        """Find all cycles in the call graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for callee in self.callees.get(node, []):
                if callee not in visited:
                    dfs(callee, path)
                elif callee in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(callee)
                    cycles.append(path[cycle_start:] + [callee])

            path.pop()
            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_entry_points(self) -> list[str]:
        """Get functions that are never called (potential entry points)."""
        return [name for name in self.nodes if not self.callers.get(name)]

    def get_leaf_functions(self) -> list[str]:
        """Get functions that call no other functions."""
        return [name for name in self.nodes if not self.callees.get(name)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "caller": e.caller,
                    "callee": e.callee,
                    "file": e.file_path,
                    "line": e.line,
                }
                for e in self.edges
            ],
            "callers": dict(self.callers),
            "callees": dict(self.callees),
        }


def build_call_graph(analyses: dict[str, FileAnalysis]) -> CallGraph:
    """
    Build a call graph from file analyses.

    Args:
        analyses: Dictionary mapping file paths to their analyses

    Returns:
        CallGraph representing function dependencies
    """
    graph = CallGraph()

    # First, collect all function definitions
    all_functions: dict[str, FunctionInfo] = {}
    for file_path, analysis in analyses.items():
        for func in analysis.functions:
            all_functions[func.name] = func
            graph.add_node(func.name)

        for cls in analysis.classes:
            for method in cls.methods:
                qualified_name = f"{cls.name}.{method.name}"
                all_functions[qualified_name] = method
                graph.add_node(qualified_name)

    # Now extract calls from function bodies
    for file_path, analysis in analyses.items():
        # Read file content for call extraction
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
        except (OSError, UnicodeDecodeError):
            continue

        for func in analysis.functions:
            _extract_calls_from_function(func, lines, all_functions, graph, file_path)

        for cls in analysis.classes:
            for method in cls.methods:
                _extract_calls_from_function(
                    method, lines, all_functions, graph, file_path, cls.name
                )

    return graph


def _extract_calls_from_function(
    func: FunctionInfo,
    lines: list[str],
    all_functions: dict[str, FunctionInfo],
    graph: CallGraph,
    file_path: str,
    class_name: str | None = None,
) -> None:
    """Extract function calls from a function body."""
    caller_name = f"{class_name}.{func.name}" if class_name else func.name

    # Get function body lines
    start = func.location.start_line - 1
    end = func.location.end_line
    body_lines = lines[start:end]

    # Pattern to match function calls
    call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

    for i, line in enumerate(body_lines, start=func.location.start_line):
        for match in call_pattern.finditer(line):
            callee_name = match.group(1)

            # Skip common keywords that aren't function calls
            if callee_name in (
                "if",
                "while",
                "for",
                "with",
                "except",
                "assert",
                "print",
                "return",
                "raise",
                "yield",
                "lambda",
            ):
                continue

            # Check if it's a known function
            if callee_name in all_functions:
                graph.add_edge(
                    CallSite(
                        caller=caller_name,
                        callee=callee_name,
                        file_path=file_path,
                        line=i,
                        column=match.start(),
                    )
                )

            # Check for method calls (self.method or Class.method)
            qualified_pattern = re.compile(
                rf"(?:self|{class_name if class_name else ''})\s*\.\s*{re.escape(callee_name)}\s*\("
            )
            if class_name and qualified_pattern.search(line):
                qualified_callee = f"{class_name}.{callee_name}"
                if qualified_callee in all_functions:
                    graph.add_edge(
                        CallSite(
                            caller=caller_name,
                            callee=qualified_callee,
                            file_path=file_path,
                            line=i,
                            column=match.start(),
                        )
                    )


def build_call_graph_from_source(
    source: str,
    file_path: str,
    language: Language,
) -> CallGraph:
    """
    Build a call graph from source code.

    Args:
        source: The source code text
        file_path: Path to the file
        language: The programming language

    Returns:
        CallGraph for the source code
    """
    graph = CallGraph()
    lines = source.split("\n")

    if language == Language.PYTHON:
        _build_python_call_graph(lines, file_path, graph)
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        _build_js_call_graph(lines, file_path, graph)
    # Add other languages as needed

    return graph


def _build_python_call_graph(
    lines: list[str],
    file_path: str,
    graph: CallGraph,
) -> None:
    """Build call graph for Python source."""
    func_pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
    call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

    current_function = None
    current_indent = 0

    for i, line in enumerate(lines, 1):
        # Check for function definition
        match = func_pattern.match(line)
        if match:
            current_function = match.group(1)
            current_indent = len(line) - len(line.lstrip())
            graph.add_node(current_function)
            continue

        # Check if we're still in the function
        if current_function:
            line_indent = len(line) - len(line.lstrip())
            if line.strip() and line_indent <= current_indent:
                current_function = None
                continue

            # Extract calls
            for call_match in call_pattern.finditer(line):
                callee = call_match.group(1)
                if callee not in (
                    "if",
                    "while",
                    "for",
                    "with",
                    "except",
                    "assert",
                    "print",
                    "return",
                    "raise",
                    "yield",
                    "lambda",
                ):
                    graph.add_edge(
                        CallSite(
                            caller=current_function,
                            callee=callee,
                            file_path=file_path,
                            line=i,
                            column=call_match.start(),
                        )
                    )


def _build_js_call_graph(
    lines: list[str],
    file_path: str,
    graph: CallGraph,
) -> None:
    """Build call graph for JavaScript/TypeScript source."""
    func_pattern = re.compile(
        r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\())"
    )
    call_pattern = re.compile(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")

    current_function = None
    brace_count = 0

    for i, line in enumerate(lines, 1):
        # Check for function definition
        match = func_pattern.search(line)
        if match:
            current_function = match.group(1) or match.group(2)
            graph.add_node(current_function)
            brace_count = line.count("{") - line.count("}")
            continue

        if current_function:
            brace_count += line.count("{") - line.count("}")
            if brace_count <= 0:
                current_function = None
                continue

            # Extract calls
            for call_match in call_pattern.finditer(line):
                callee = call_match.group(1)
                if callee not in ("if", "while", "for", "switch", "catch", "function"):
                    graph.add_edge(
                        CallSite(
                            caller=current_function,
                            callee=callee,
                            file_path=file_path,
                            line=i,
                            column=call_match.start(),
                        )
                    )


def get_call_chain(
    graph: CallGraph,
    start: str,
    end: str,
) -> list[str] | None:
    """
    Find a call chain from start function to end function.

    Args:
        graph: The call graph
        start: Starting function name
        end: Target function name

    Returns:
        List of function names in the call chain, or None if no path exists
    """
    if start not in graph.nodes or end not in graph.nodes:
        return None

    # BFS to find shortest path
    visited = {start}
    queue = [(start, [start])]

    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path

        for callee in graph.callees.get(current, []):
            if callee not in visited:
                visited.add(callee)
                queue.append((callee, path + [callee]))

    return None


__all__ = [
    "CallSite",
    "CallGraph",
    "build_call_graph",
    "build_call_graph_from_source",
    "get_call_chain",
]
