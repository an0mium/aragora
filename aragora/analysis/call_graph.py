"""
Call graph construction and analysis.

Builds inter-procedural call graphs from codebase analysis,
supports dead code detection, circular dependency detection,
and impact analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from aragora.analysis.code_intelligence import (
    CodeIntelligence,
    FileAnalysis,
    FunctionInfo,
    ImportInfo,
    SourceLocation,
    SymbolKind,
)


class EdgeType(str, Enum):
    """Type of relationship between nodes."""

    CALLS = "calls"  # Function calls another function
    IMPORTS = "imports"  # Module imports another
    INHERITS = "inherits"  # Class inherits from another
    USES = "uses"  # Symbol uses another (field access, etc.)
    CONTAINS = "contains"  # File contains symbol


@dataclass
class GraphNode:
    """A node in the call/dependency graph."""

    id: str  # Unique identifier (e.g., "module.Class.method")
    kind: SymbolKind
    name: str
    qualified_name: str
    location: Optional[SourceLocation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False


@dataclass
class GraphEdge:
    """An edge in the call/dependency graph."""

    source: str  # Source node ID
    target: str  # Target node ID
    edge_type: EdgeType
    location: Optional[SourceLocation] = None  # Where the reference occurs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallSite:
    """A location where a function is called."""

    caller: str  # Qualified name of caller
    callee: str  # Name being called (may be unresolved)
    location: SourceLocation
    arguments: List[str] = field(default_factory=list)
    is_method_call: bool = False
    receiver: Optional[str] = None  # For method calls, the receiver expression


@dataclass
class DeadCodeResult:
    """Result of dead code analysis."""

    unreachable_functions: List[GraphNode]
    unreachable_classes: List[GraphNode]
    unused_imports: List[ImportInfo]
    total_dead_lines: int = 0


@dataclass
class ImpactResult:
    """Result of impact analysis."""

    changed_node: str
    directly_affected: List[str]  # Nodes that directly depend on changed node
    transitively_affected: List[str]  # All nodes affected (transitive closure)
    affected_files: Set[str]


@dataclass
class CircularDependency:
    """A circular dependency in the codebase."""

    cycle: List[str]  # Node IDs forming the cycle
    edge_type: EdgeType
    locations: List[SourceLocation]


class CallGraph:
    """
    Inter-procedural call graph with dependency analysis.

    Builds a graph of function calls, imports, and class inheritance
    to support dead code detection, impact analysis, and dependency
    visualization.
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for call graph analysis. "
                "Install with: pip install 'aragora[code-intel]'"
            )
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._file_to_nodes: Dict[str, Set[str]] = {}
        self._entry_points: Set[str] = set()

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
        self._graph.add_node(
            node.id,
            kind=node.kind.value,
            name=node.name,
            qualified_name=node.qualified_name,
        )
        if node.location:
            file_path = node.location.file_path
            if file_path not in self._file_to_nodes:
                self._file_to_nodes[file_path] = set()
            self._file_to_nodes[file_path].add(node.id)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self._edges.append(edge)
        self._graph.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type.value,
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_callers(self, node_id: str) -> List[GraphNode]:
        """Get all functions that call this node."""
        if node_id not in self._graph:
            return []
        callers = []
        for pred in self._graph.predecessors(node_id):
            edge_data = self._graph.get_edge_data(pred, node_id)
            if edge_data and edge_data.get("edge_type") == EdgeType.CALLS.value:
                node = self._nodes.get(pred)
                if node:
                    callers.append(node)
        return callers

    def get_callees(self, node_id: str) -> List[GraphNode]:
        """Get all functions called by this node."""
        if node_id not in self._graph:
            return []
        callees = []
        for succ in self._graph.successors(node_id):
            edge_data = self._graph.get_edge_data(node_id, succ)
            if edge_data and edge_data.get("edge_type") == EdgeType.CALLS.value:
                node = self._nodes.get(succ)
                if node:
                    callees.append(node)
        return callees

    def get_dependents(self, node_id: str, transitive: bool = False) -> List[str]:
        """Get nodes that depend on this node."""
        if node_id not in self._graph:
            return []
        if transitive:
            # Get all ancestors (nodes that can reach this node)
            return list(nx.ancestors(self._graph, node_id))
        else:
            return list(self._graph.predecessors(node_id))

    def get_dependencies(self, node_id: str, transitive: bool = False) -> List[str]:
        """Get nodes that this node depends on."""
        if node_id not in self._graph:
            return []
        if transitive:
            # Get all descendants (nodes reachable from this node)
            return list(nx.descendants(self._graph, node_id))
        else:
            return list(self._graph.successors(node_id))

    def mark_entry_point(self, node_id: str) -> None:
        """Mark a node as an entry point (e.g., main, test, API endpoint)."""
        self._entry_points.add(node_id)

    def find_dead_code(self) -> DeadCodeResult:
        """
        Find unreachable code starting from entry points.

        Entry points include:
        - main() functions
        - Test functions (test_*)
        - Public API endpoints
        - Explicitly marked entry points
        """
        # Auto-detect entry points if none marked
        if not self._entry_points:
            self._auto_detect_entry_points()

        # Find all reachable nodes from entry points
        reachable: Set[str] = set()
        for entry in self._entry_points:
            if entry in self._graph:
                reachable.add(entry)
                reachable.update(nx.descendants(self._graph, entry))

        # Find unreachable nodes
        unreachable_functions: List[GraphNode] = []
        unreachable_classes: List[GraphNode] = []

        for node_id, node in self._nodes.items():
            if node_id not in reachable:
                if node.kind == SymbolKind.FUNCTION:
                    unreachable_functions.append(node)
                elif node.kind == SymbolKind.CLASS:
                    unreachable_classes.append(node)

        # Calculate dead lines
        total_dead_lines = 0
        for node in unreachable_functions + unreachable_classes:
            if node.location:
                lines = (
                    node.metadata.get("end_line", node.location.start_line)
                    - node.location.start_line
                    + 1
                )
                total_dead_lines += lines

        return DeadCodeResult(
            unreachable_functions=unreachable_functions,
            unreachable_classes=unreachable_classes,
            unused_imports=[],  # Would need import analysis
            total_dead_lines=total_dead_lines,
        )

    def _auto_detect_entry_points(self) -> None:
        """Auto-detect entry points based on naming conventions."""
        for node_id, node in self._nodes.items():
            name = node.name.lower()
            # Main functions
            if name in ("main", "__main__", "cli", "run"):
                self._entry_points.add(node_id)
            # Test functions
            elif name.startswith("test_") or name.startswith("test"):
                self._entry_points.add(node_id)
            # API endpoints
            elif node.metadata.get("is_endpoint"):
                self._entry_points.add(node_id)
            # Module-level code markers
            elif node.kind == SymbolKind.MODULE:
                self._entry_points.add(node_id)

    def find_circular_dependencies(
        self, edge_type: Optional[EdgeType] = None
    ) -> List[CircularDependency]:
        """Find circular dependencies in the graph."""
        # Filter graph by edge type if specified
        if edge_type:
            filtered = nx.DiGraph()
            for u, v, data in self._graph.edges(data=True):
                if data.get("edge_type") == edge_type.value:
                    filtered.add_edge(u, v)
            graph = filtered
        else:
            graph = self._graph

        cycles = []
        try:
            for cycle in nx.simple_cycles(graph):
                if len(cycle) > 1:  # Ignore self-loops
                    cycles.append(
                        CircularDependency(
                            cycle=cycle,
                            edge_type=edge_type or EdgeType.CALLS,
                            locations=[],  # Would need edge location tracking
                        )
                    )
        except nx.NetworkXError:
            pass

        return cycles

    def analyze_impact(self, changed_node: str) -> ImpactResult:
        """
        Analyze the impact of changing a specific node.

        Returns all nodes that might be affected by the change.
        """
        if changed_node not in self._graph:
            return ImpactResult(
                changed_node=changed_node,
                directly_affected=[],
                transitively_affected=[],
                affected_files=set(),
            )

        directly_affected = list(self._graph.predecessors(changed_node))
        transitively_affected = list(nx.ancestors(self._graph, changed_node))

        # Collect affected files
        affected_files: Set[str] = set()
        for node_id in transitively_affected + [changed_node]:
            node = self._nodes.get(node_id)
            if node and node.location:
                affected_files.add(node.location.file_path)

        return ImpactResult(
            changed_node=changed_node,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            affected_files=affected_files,
        )

    def get_hotspots(self, top_n: int = 10) -> List[Tuple[GraphNode, int]]:
        """
        Find hotspots - nodes with high in-degree (many callers).

        These are central nodes that many parts of the codebase depend on.
        """
        in_degrees = self._graph.in_degree()
        sorted_nodes = sorted(in_degrees, key=lambda x: x[1], reverse=True)

        hotspots = []
        for node_id, degree in sorted_nodes[:top_n]:
            node = self._nodes.get(node_id)
            if node:
                hotspots.append((node, degree))

        return hotspots

    def get_complexity_metrics(self) -> Dict[str, Any]:
        """Get graph complexity metrics."""
        metrics: Dict[str, Any] = {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "density": 0.0,
            "avg_degree": 0.0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "strongly_connected_components": 0,
            "weakly_connected_components": 0,
        }

        if self.node_count > 0:
            metrics["density"] = nx.density(self._graph)
            metrics["avg_degree"] = sum(dict(self._graph.degree()).values()) / self.node_count

            in_degrees = dict(self._graph.in_degree())
            out_degrees = dict(self._graph.out_degree())
            metrics["max_in_degree"] = max(in_degrees.values()) if in_degrees else 0
            metrics["max_out_degree"] = max(out_degrees.values()) if out_degrees else 0

            metrics["strongly_connected_components"] = nx.number_strongly_connected_components(
                self._graph
            )
            metrics["weakly_connected_components"] = nx.number_weakly_connected_components(
                self._graph
            )

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind.value,
                    "name": n.name,
                    "qualified_name": n.qualified_name,
                    "location": (
                        {
                            "file": n.location.file_path,
                            "line": n.location.start_line,
                        }
                        if n.location
                        else None
                    ),
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type.value,
                }
                for e in self._edges
            ],
            "entry_points": list(self._entry_points),
            "metrics": self.get_complexity_metrics(),
        }


class CallGraphBuilder:
    """
    Builds call graphs from CodeIntelligence analysis results.

    Performs inter-procedural analysis to resolve function calls
    and build the complete call graph.
    """

    def __init__(self, code_intel: Optional[CodeIntelligence] = None):
        self.code_intel = code_intel or CodeIntelligence()
        self._symbol_table: Dict[str, GraphNode] = {}
        self._import_map: Dict[str, Dict[str, str]] = {}  # file -> {alias: qualified_name}

    def build_from_directory(
        self,
        directory: str,
        exclude_patterns: Optional[List[str]] = None,
    ) -> CallGraph:
        """
        Build call graph from a directory of source files.

        Args:
            directory: Root directory to analyze
            exclude_patterns: Glob patterns to exclude

        Returns:
            Complete call graph for the codebase
        """
        graph = CallGraph()

        # First pass: collect all symbols
        analyses_dict = self.code_intel.analyze_directory(
            directory, exclude_patterns=exclude_patterns
        )
        analyses = list(analyses_dict.values())

        for analysis in analyses:
            self._process_file_symbols(analysis, graph)

        # Second pass: resolve calls and build edges
        for analysis in analyses:
            self._process_file_calls(analysis, graph)

        return graph

    def build_from_files(self, file_paths: List[str]) -> CallGraph:
        """Build call graph from specific files."""
        graph = CallGraph()
        analyses = []

        for file_path in file_paths:
            analysis = self.code_intel.analyze_file(file_path)
            if analysis:
                analyses.append(analysis)

        # First pass: symbols
        for analysis in analyses:
            self._process_file_symbols(analysis, graph)

        # Second pass: calls
        for analysis in analyses:
            self._process_file_calls(analysis, graph)

        return graph

    def _process_file_symbols(self, analysis: FileAnalysis, graph: CallGraph) -> None:
        """Extract symbols from file analysis and add to graph."""
        file_path = analysis.file_path
        module_name = self._path_to_module(file_path)

        # Add module node
        module_node = GraphNode(
            id=module_name,
            kind=SymbolKind.MODULE,
            name=Path(file_path).stem,
            qualified_name=module_name,
            location=SourceLocation(
                file_path=file_path, start_line=1, start_column=0, end_line=1, end_column=0
            ),
        )
        graph.add_node(module_node)
        self._symbol_table[module_name] = module_node

        # Build import map for this file
        self._import_map[file_path] = {}
        for imp in analysis.imports:
            if imp.alias:
                self._import_map[file_path][imp.alias] = imp.module
            elif imp.names:
                for name in imp.names:
                    self._import_map[file_path][name] = f"{imp.module}.{name}"
            else:
                # import module
                parts = imp.module.split(".")
                self._import_map[file_path][parts[-1]] = imp.module

        # Add class nodes
        for cls in analysis.classes:
            cls_qualified = f"{module_name}.{cls.name}"
            cls_node = GraphNode(
                id=cls_qualified,
                kind=SymbolKind.CLASS,
                name=cls.name,
                qualified_name=cls_qualified,
                location=cls.location,
                metadata={
                    "bases": cls.bases,
                    "decorators": cls.decorators,
                    "end_line": (
                        cls.location.start_line + len(cls.methods) * 5 if cls.location else 0
                    ),
                },
            )
            graph.add_node(cls_node)
            self._symbol_table[cls_qualified] = cls_node

            # Add contains edge from module to class
            graph.add_edge(
                GraphEdge(
                    source=module_name,
                    target=cls_qualified,
                    edge_type=EdgeType.CONTAINS,
                )
            )

            # Add inheritance edges
            for base in cls.bases:
                resolved_base = self._resolve_symbol(base, file_path)
                if resolved_base:
                    graph.add_edge(
                        GraphEdge(
                            source=cls_qualified,
                            target=resolved_base,
                            edge_type=EdgeType.INHERITS,
                            location=cls.location,
                        )
                    )

            # Add method nodes
            for method in cls.methods:
                method_qualified = f"{cls_qualified}.{method.name}"
                method_node = GraphNode(
                    id=method_qualified,
                    kind=SymbolKind.METHOD,
                    name=method.name,
                    qualified_name=method_qualified,
                    location=method.location,
                    metadata={
                        "is_async": method.is_async,
                        "decorators": method.decorators,
                        "complexity": method.complexity,
                        "parameters": [p.name for p in method.parameters],
                    },
                )
                graph.add_node(method_node)
                self._symbol_table[method_qualified] = method_node

                # Add contains edge from class to method
                graph.add_edge(
                    GraphEdge(
                        source=cls_qualified,
                        target=method_qualified,
                        edge_type=EdgeType.CONTAINS,
                    )
                )

        # Add standalone function nodes
        for func in analysis.functions:
            func_qualified = f"{module_name}.{func.name}"
            func_node = GraphNode(
                id=func_qualified,
                kind=SymbolKind.FUNCTION,
                name=func.name,
                qualified_name=func_qualified,
                location=func.location,
                metadata={
                    "is_async": func.is_async,
                    "decorators": func.decorators,
                    "complexity": func.complexity,
                    "parameters": [p.name for p in func.parameters],
                },
            )
            graph.add_node(func_node)
            self._symbol_table[func_qualified] = func_node

            # Add contains edge from module to function
            graph.add_edge(
                GraphEdge(
                    source=module_name,
                    target=func_qualified,
                    edge_type=EdgeType.CONTAINS,
                )
            )

            # Auto-detect entry points
            if func.name in ("main", "__main__", "cli") or func.name.startswith("test_"):
                graph.mark_entry_point(func_qualified)

    def _process_file_calls(self, analysis: FileAnalysis, graph: CallGraph) -> None:
        """Extract function calls and add edges to graph."""
        file_path = analysis.file_path
        _module_name = self._path_to_module(file_path)  # Reserved for module-level analysis

        # We need to re-parse to find call sites
        # This is a simplified approach - a full implementation would use AST
        call_sites = self._extract_call_sites(analysis)

        for call_site in call_sites:
            resolved_callee = self._resolve_symbol(call_site.callee, file_path)
            if resolved_callee and resolved_callee in self._symbol_table:
                graph.add_edge(
                    GraphEdge(
                        source=call_site.caller,
                        target=resolved_callee,
                        edge_type=EdgeType.CALLS,
                        location=call_site.location,
                    )
                )

    def _extract_call_sites(self, analysis: FileAnalysis) -> List[CallSite]:
        """Extract call sites from file analysis."""
        call_sites: List[CallSite] = []
        file_path = analysis.file_path
        module_name = self._path_to_module(file_path)

        # Read file content for call extraction
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                lines = content.split("\n")
        except (OSError, IOError):
            return call_sites

        # For each function/method, find calls within it
        all_funcs: List[Tuple[str, FunctionInfo]] = []

        for func in analysis.functions:
            all_funcs.append((f"{module_name}.{func.name}", func))

        for cls in analysis.classes:
            for method in cls.methods:
                all_funcs.append((f"{module_name}.{cls.name}.{method.name}", method))

        # Simple regex-based call extraction
        # Pattern matches: name(...) or obj.name(...)
        call_pattern = re.compile(r"(?:(\w+)\.)?(\w+)\s*\(")

        for caller_qualified, func in all_funcs:
            if not func.location:
                continue

            start_line = func.location.start_line - 1  # 0-indexed
            # Estimate end line (simplified)
            end_line = min(start_line + 100, len(lines))

            for line_idx in range(start_line, end_line):
                line = lines[line_idx]
                for match in call_pattern.finditer(line):
                    receiver = match.group(1)
                    callee_name = match.group(2)

                    # Skip common keywords and builtins
                    if callee_name in (
                        "if",
                        "for",
                        "while",
                        "with",
                        "print",
                        "len",
                        "str",
                        "int",
                        "list",
                        "dict",
                        "set",
                        "tuple",
                        "range",
                        "enumerate",
                        "isinstance",
                        "hasattr",
                        "getattr",
                        "setattr",
                    ):
                        continue

                    call_sites.append(
                        CallSite(
                            caller=caller_qualified,
                            callee=callee_name if not receiver else f"{receiver}.{callee_name}",
                            location=SourceLocation(
                                file_path=file_path,
                                start_line=line_idx + 1,
                                start_column=match.start(),
                                end_line=line_idx + 1,
                                end_column=match.end(),
                            ),
                            is_method_call=receiver is not None,
                            receiver=receiver,
                        )
                    )

        return call_sites

    def _resolve_symbol(self, name: str, file_path: str) -> Optional[str]:
        """
        Resolve a symbol name to its qualified name.

        Uses import map and symbol table to resolve references.
        """
        # Check if it's already qualified
        if name in self._symbol_table:
            return name

        # Check import map for this file
        imports = self._import_map.get(file_path, {})

        # Handle dotted names (e.g., "module.function")
        parts = name.split(".")
        first_part = parts[0]

        if first_part in imports:
            base = imports[first_part]
            if len(parts) > 1:
                qualified = f"{base}.{'.'.join(parts[1:])}"
            else:
                qualified = base

            # Check if resolved name exists
            if qualified in self._symbol_table:
                return qualified

        # Try module-local resolution
        module_name = self._path_to_module(file_path)
        local_qualified = f"{module_name}.{name}"
        if local_qualified in self._symbol_table:
            return local_qualified

        return None

    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to Python module name."""
        path = Path(file_path)

        # Remove .py extension
        if path.suffix == ".py":
            path = path.with_suffix("")

        # Convert path separators to dots
        parts = path.parts

        # Try to find package root (directory with __init__.py)
        # For simplicity, just use the last few parts
        if len(parts) > 3:
            parts = parts[-3:]

        module = ".".join(parts)

        # Remove __init__ suffix
        if module.endswith(".__init__"):
            module = module[:-9]

        return module


class ImportGraph:
    """
    Specialized graph for module import relationships.

    Useful for detecting circular imports and understanding
    module dependencies.
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for import graph analysis")
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_import(
        self, importer: str, imported: str, location: Optional[SourceLocation] = None
    ) -> None:
        """Add an import relationship."""
        self._graph.add_edge(importer, imported, location=location)

    def find_circular_imports(self) -> List[List[str]]:
        """Find all circular import chains."""
        try:
            cycles = list(nx.simple_cycles(self._graph))
            # Filter to meaningful cycles (length > 1)
            return [c for c in cycles if len(c) > 1]
        except nx.NetworkXError:
            return []

    def get_import_order(self) -> List[str]:
        """Get topological order for imports (if acyclic)."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            return []  # Has cycles

    def get_module_dependencies(self, module: str, transitive: bool = False) -> List[str]:
        """Get modules that a module imports."""
        if module not in self._graph:
            return []
        if transitive:
            return list(nx.descendants(self._graph, module))
        return list(self._graph.successors(module))

    def get_module_dependents(self, module: str, transitive: bool = False) -> List[str]:
        """Get modules that import a module."""
        if module not in self._graph:
            return []
        if transitive:
            return list(nx.ancestors(self._graph, module))
        return list(self._graph.predecessors(module))

    @classmethod
    def from_analyses(cls, analyses: List[FileAnalysis]) -> "ImportGraph":
        """Build import graph from file analyses."""
        graph = cls()

        for analysis in analyses:
            module = Path(analysis.file_path).stem
            for imp in analysis.imports:
                loc = (
                    imp.location
                    if imp.location
                    else SourceLocation(
                        file_path=analysis.file_path,
                        start_line=1,
                        start_column=0,
                        end_line=1,
                        end_column=0,
                    )
                )
                graph.add_import(module, imp.module, loc)

        return graph


def analyze_codebase_dependencies(
    directory: str,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    High-level function to analyze codebase dependencies.

    Returns a comprehensive report including:
    - Call graph metrics
    - Circular dependencies
    - Dead code
    - Hotspots

    Args:
        directory: Root directory to analyze
        exclude_patterns: Glob patterns to exclude

    Returns:
        Dictionary with analysis results
    """
    builder = CallGraphBuilder()
    graph = builder.build_from_directory(directory, exclude_patterns)

    dead_code = graph.find_dead_code()
    circular = graph.find_circular_dependencies()
    hotspots = graph.get_hotspots(top_n=10)

    return {
        "metrics": graph.get_complexity_metrics(),
        "dead_code": {
            "unreachable_functions": len(dead_code.unreachable_functions),
            "unreachable_classes": len(dead_code.unreachable_classes),
            "total_dead_lines": dead_code.total_dead_lines,
            "details": [
                {
                    "name": n.qualified_name,
                    "kind": n.kind.value,
                    "location": (
                        f"{n.location.file_path}:{n.location.start_line}" if n.location else None
                    ),
                }
                for n in dead_code.unreachable_functions[:20]
            ],
        },
        "circular_dependencies": [
            {"cycle": c.cycle, "type": c.edge_type.value} for c in circular[:10]
        ],
        "hotspots": [
            {
                "name": node.qualified_name,
                "callers": degree,
                "location": (
                    f"{node.location.file_path}:{node.location.start_line}"
                    if node.location
                    else None
                ),
            }
            for node, degree in hotspots
        ],
    }
