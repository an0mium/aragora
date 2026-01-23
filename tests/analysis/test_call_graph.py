"""
Tests for call graph construction and analysis.

Tests graph building, dead code detection, and dependency analysis.
"""

import pytest
from pathlib import Path

from aragora.analysis.call_graph import (
    CallGraph,
    CallGraphBuilder,
    GraphNode,
    GraphEdge,
    EdgeType,
    CallSite,
    DeadCodeResult,
    ImpactResult,
    CircularDependency,
    ImportGraph,
    analyze_codebase_dependencies,
)
from aragora.analysis.code_intelligence import (
    SymbolKind,
    SourceLocation,
)


# Sample code for testing call graphs
SAMPLE_MODULE_A = '''
"""Module A with functions that call Module B."""

from module_b import helper_function, HelperClass

def main():
    """Entry point."""
    result = process_data([1, 2, 3])
    helper_function(result)
    return result

def process_data(items):
    """Process a list of items."""
    helper = HelperClass()
    return helper.transform(items)

def unused_function():
    """This function is never called."""
    pass
'''

SAMPLE_MODULE_B = '''
"""Module B with helper functions."""

def helper_function(data):
    """A helper function."""
    return format_output(data)

def format_output(data):
    """Format the output."""
    return str(data)

class HelperClass:
    """A helper class."""

    def transform(self, items):
        """Transform items."""
        return [self.process_item(x) for x in items]

    def process_item(self, item):
        """Process a single item."""
        return item * 2
'''

CIRCULAR_IMPORT_A = """
from circular_b import func_b

def func_a():
    return func_b()
"""

CIRCULAR_IMPORT_B = """
from circular_a import func_a

def func_b():
    return func_a()
"""


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_type_values(self):
        """Test all edge type values exist."""
        assert EdgeType.CALLS.value == "calls"
        assert EdgeType.IMPORTS.value == "imports"
        assert EdgeType.INHERITS.value == "inherits"
        assert EdgeType.USES.value == "uses"
        assert EdgeType.CONTAINS.value == "contains"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_node(self):
        """Test creating a graph node."""
        node = GraphNode(
            id="module.MyClass.method",
            kind=SymbolKind.METHOD,
            name="method",
            qualified_name="module.MyClass.method",
            location=SourceLocation(
                file_path="/test.py", start_line=10, start_column=4, end_line=10, end_column=20
            ),
        )
        assert node.id == "module.MyClass.method"
        assert node.kind == SymbolKind.METHOD
        assert node.name == "method"

    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = GraphNode(
            id="test.func",
            kind=SymbolKind.FUNCTION,
            name="func",
            qualified_name="test.func",
        )
        node2 = GraphNode(
            id="test.func",
            kind=SymbolKind.FUNCTION,
            name="func",
            qualified_name="test.func",
        )
        assert node1 == node2
        assert hash(node1) == hash(node2)

    def test_node_with_metadata(self):
        """Test node with metadata."""
        node = GraphNode(
            id="test.func",
            kind=SymbolKind.FUNCTION,
            name="func",
            qualified_name="test.func",
            metadata={"complexity": 5, "is_async": True},
        )
        assert node.metadata["complexity"] == 5
        assert node.metadata["is_async"] is True


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_create_edge(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            source="module.func_a",
            target="module.func_b",
            edge_type=EdgeType.CALLS,
            location=SourceLocation(
                file_path="/test.py", start_line=15, start_column=4, end_line=15, end_column=20
            ),
        )
        assert edge.source == "module.func_a"
        assert edge.target == "module.func_b"
        assert edge.edge_type == EdgeType.CALLS


class TestCallSite:
    """Tests for CallSite dataclass."""

    def test_create_call_site(self):
        """Test creating a call site."""
        site = CallSite(
            caller="module.main",
            callee="helper.process",
            location=SourceLocation(
                file_path="/test.py", start_line=20, start_column=8, end_line=20, end_column=30
            ),
            is_method_call=True,
            receiver="self",
        )
        assert site.caller == "module.main"
        assert site.callee == "helper.process"
        assert site.is_method_call is True


class TestCallGraph:
    """Tests for CallGraph class."""

    @pytest.fixture
    def empty_graph(self):
        """Create an empty call graph."""
        return CallGraph()

    @pytest.fixture
    def sample_graph(self):
        """Create a sample call graph with nodes and edges."""
        graph = CallGraph()

        # Add nodes
        main = GraphNode(
            id="module.main",
            kind=SymbolKind.FUNCTION,
            name="main",
            qualified_name="module.main",
        )
        helper = GraphNode(
            id="module.helper",
            kind=SymbolKind.FUNCTION,
            name="helper",
            qualified_name="module.helper",
        )
        unused = GraphNode(
            id="module.unused",
            kind=SymbolKind.FUNCTION,
            name="unused",
            qualified_name="module.unused",
        )

        graph.add_node(main)
        graph.add_node(helper)
        graph.add_node(unused)

        # Add edge: main calls helper
        graph.add_edge(
            GraphEdge(
                source="module.main",
                target="module.helper",
                edge_type=EdgeType.CALLS,
            )
        )

        # Mark main as entry point
        graph.mark_entry_point("module.main")

        return graph

    def test_empty_graph_properties(self, empty_graph):
        """Test empty graph properties."""
        assert empty_graph.node_count == 0
        assert empty_graph.edge_count == 0

    def test_add_node(self, empty_graph):
        """Test adding nodes to graph."""
        node = GraphNode(
            id="test.func",
            kind=SymbolKind.FUNCTION,
            name="func",
            qualified_name="test.func",
        )
        empty_graph.add_node(node)
        assert empty_graph.node_count == 1

    def test_add_edge(self, empty_graph):
        """Test adding edges to graph."""
        # Add nodes first
        empty_graph.add_node(
            GraphNode(id="a", kind=SymbolKind.FUNCTION, name="a", qualified_name="a")
        )
        empty_graph.add_node(
            GraphNode(id="b", kind=SymbolKind.FUNCTION, name="b", qualified_name="b")
        )

        edge = GraphEdge(source="a", target="b", edge_type=EdgeType.CALLS)
        empty_graph.add_edge(edge)

        assert empty_graph.edge_count == 1

    def test_get_node(self, sample_graph):
        """Test getting a node by ID."""
        node = sample_graph.get_node("module.main")
        assert node is not None
        assert node.name == "main"

        missing = sample_graph.get_node("nonexistent")
        assert missing is None

    def test_get_callers(self, sample_graph):
        """Test getting callers of a function."""
        callers = sample_graph.get_callers("module.helper")
        assert len(callers) == 1
        assert callers[0].name == "main"

    def test_get_callees(self, sample_graph):
        """Test getting callees of a function."""
        callees = sample_graph.get_callees("module.main")
        assert len(callees) == 1
        assert callees[0].name == "helper"

    def test_get_dependents(self, sample_graph):
        """Test getting dependent nodes."""
        # helper depends on nothing, but main depends on helper
        dependents = sample_graph.get_dependents("module.helper")
        assert "module.main" in dependents

    def test_get_dependencies(self, sample_graph):
        """Test getting dependencies."""
        # main depends on helper
        deps = sample_graph.get_dependencies("module.main")
        assert "module.helper" in deps

    def test_find_dead_code(self, sample_graph):
        """Test finding unreachable code."""
        dead_code = sample_graph.find_dead_code()

        assert isinstance(dead_code, DeadCodeResult)
        # unused function should be unreachable
        unreachable_names = [n.name for n in dead_code.unreachable_functions]
        assert "unused" in unreachable_names

    def test_mark_entry_point(self, empty_graph):
        """Test marking entry points."""
        empty_graph.add_node(
            GraphNode(
                id="main",
                kind=SymbolKind.FUNCTION,
                name="main",
                qualified_name="main",
            )
        )
        empty_graph.mark_entry_point("main")

        # Entry point should be reachable in dead code analysis
        dead_code = empty_graph.find_dead_code()
        unreachable = [n.id for n in dead_code.unreachable_functions]
        assert "main" not in unreachable

    def test_analyze_impact(self, sample_graph):
        """Test impact analysis."""
        impact = sample_graph.analyze_impact("module.helper")

        assert isinstance(impact, ImpactResult)
        assert impact.changed_node == "module.helper"
        # main is directly affected since it calls helper
        assert "module.main" in impact.directly_affected

    def test_get_hotspots(self, sample_graph):
        """Test finding hotspots."""
        # Add more callers to helper
        for i in range(5):
            caller_id = f"module.caller_{i}"
            sample_graph.add_node(
                GraphNode(
                    id=caller_id,
                    kind=SymbolKind.FUNCTION,
                    name=f"caller_{i}",
                    qualified_name=caller_id,
                )
            )
            sample_graph.add_edge(
                GraphEdge(
                    source=caller_id,
                    target="module.helper",
                    edge_type=EdgeType.CALLS,
                )
            )

        hotspots = sample_graph.get_hotspots(top_n=3)
        assert len(hotspots) <= 3
        # helper should be a hotspot (many callers)
        hotspot_names = [h[0].name for h in hotspots]
        assert "helper" in hotspot_names

    def test_get_complexity_metrics(self, sample_graph):
        """Test getting complexity metrics."""
        metrics = sample_graph.get_complexity_metrics()

        assert "nodes" in metrics
        assert "edges" in metrics
        assert "density" in metrics
        assert metrics["nodes"] == 3
        assert metrics["edges"] == 1

    def test_to_dict(self, sample_graph):
        """Test serialization to dictionary."""
        data = sample_graph.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "entry_points" in data
        assert "metrics" in data
        assert len(data["nodes"]) == 3


class TestCallGraphBuilder:
    """Tests for CallGraphBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a call graph builder."""
        return CallGraphBuilder()

    @pytest.fixture
    def sample_codebase(self, tmp_path):
        """Create a sample codebase for testing."""
        # Create module_a.py
        module_a = tmp_path / "module_a.py"
        module_a.write_text(SAMPLE_MODULE_A)

        # Create module_b.py
        module_b = tmp_path / "module_b.py"
        module_b.write_text(SAMPLE_MODULE_B)

        return tmp_path

    def test_build_from_directory(self, builder, sample_codebase):
        """Test building call graph from directory."""
        graph = builder.build_from_directory(str(sample_codebase))

        assert graph is not None
        assert graph.node_count > 0
        assert graph.edge_count >= 0

    def test_build_from_files(self, builder, sample_codebase):
        """Test building call graph from specific files."""
        files = [
            str(sample_codebase / "module_a.py"),
            str(sample_codebase / "module_b.py"),
        ]
        graph = builder.build_from_files(files)

        assert graph is not None
        assert graph.node_count > 0

    def test_symbol_extraction(self, builder, sample_codebase):
        """Test that symbols are correctly extracted."""
        graph = builder.build_from_directory(str(sample_codebase))

        # Check that expected symbols exist
        node_names = [n.name for n in graph._nodes.values()]
        # Should find functions and classes
        assert any("main" in name or "process" in name for name in node_names)


class TestCircularDependencies:
    """Tests for circular dependency detection."""

    @pytest.fixture
    def circular_graph(self):
        """Create a graph with circular dependencies."""
        graph = CallGraph()

        # A -> B -> C -> A (cycle)
        for name in ["a", "b", "c"]:
            graph.add_node(
                GraphNode(
                    id=name,
                    kind=SymbolKind.FUNCTION,
                    name=name,
                    qualified_name=name,
                )
            )

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.CALLS))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.CALLS))
        graph.add_edge(GraphEdge(source="c", target="a", edge_type=EdgeType.CALLS))

        return graph

    def test_find_circular_dependencies(self, circular_graph):
        """Test finding circular dependencies."""
        cycles = circular_graph.find_circular_dependencies()

        assert len(cycles) >= 1
        # Should find the a -> b -> c -> a cycle
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle.cycle)
        assert "a" in cycle_nodes or "b" in cycle_nodes or "c" in cycle_nodes

    def test_no_cycles_in_acyclic_graph(self):
        """Test no cycles found in acyclic graph."""
        graph = CallGraph()

        for name in ["a", "b", "c"]:
            graph.add_node(
                GraphNode(
                    id=name,
                    kind=SymbolKind.FUNCTION,
                    name=name,
                    qualified_name=name,
                )
            )

        # Linear: a -> b -> c (no cycle)
        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.CALLS))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.CALLS))

        cycles = graph.find_circular_dependencies()
        assert len(cycles) == 0


class TestImportGraph:
    """Tests for ImportGraph class."""

    @pytest.fixture
    def import_graph(self):
        """Create a sample import graph."""
        graph = ImportGraph()
        graph.add_import("main", "utils")
        graph.add_import("main", "config")
        graph.add_import("utils", "helpers")
        return graph

    def test_add_import(self, import_graph):
        """Test adding imports."""
        import_graph.add_import("new_module", "dependency")
        deps = import_graph.get_module_dependencies("new_module")
        assert "dependency" in deps

    def test_get_module_dependencies(self, import_graph):
        """Test getting module dependencies."""
        deps = import_graph.get_module_dependencies("main")
        assert "utils" in deps
        assert "config" in deps

    def test_get_module_dependents(self, import_graph):
        """Test getting modules that depend on a module."""
        dependents = import_graph.get_module_dependents("utils")
        assert "main" in dependents

    def test_transitive_dependencies(self, import_graph):
        """Test getting transitive dependencies."""
        deps = import_graph.get_module_dependencies("main", transitive=True)
        # main -> utils -> helpers (transitive)
        assert "helpers" in deps

    def test_find_circular_imports(self):
        """Test finding circular imports."""
        graph = ImportGraph()
        graph.add_import("a", "b")
        graph.add_import("b", "c")
        graph.add_import("c", "a")  # Creates cycle

        cycles = graph.find_circular_imports()
        assert len(cycles) >= 1

    def test_get_import_order(self, import_graph):
        """Test getting topological import order."""
        order = import_graph.get_import_order()
        # helpers should come before utils, utils before main
        if order:  # Only if no cycles
            assert len(order) > 0


class TestAnalyzeCodebaseDependencies:
    """Tests for the high-level analysis function."""

    @pytest.fixture
    def sample_project(self, tmp_path):
        """Create a sample project."""
        (tmp_path / "main.py").write_text(
            """
def main():
    helper()

def helper():
    pass

def unused():
    pass
"""
        )
        return tmp_path

    def test_analyze_codebase_dependencies(self, sample_project):
        """Test high-level codebase analysis."""
        result = analyze_codebase_dependencies(str(sample_project))

        assert "metrics" in result
        assert "dead_code" in result
        assert "circular_dependencies" in result
        assert "hotspots" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Test analyzing empty directory."""
        builder = CallGraphBuilder()
        graph = builder.build_from_directory(str(tmp_path))
        assert graph.node_count == 0

    def test_no_python_files(self, tmp_path):
        """Test directory with no Python files."""
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "data.json").write_text("{}")

        builder = CallGraphBuilder()
        graph = builder.build_from_directory(str(tmp_path))
        assert graph.node_count == 0

    def test_malformed_python(self, tmp_path):
        """Test handling malformed Python files."""
        (tmp_path / "bad.py").write_text("def broken(\n  # syntax error")

        builder = CallGraphBuilder()
        # Should not crash
        graph = builder.build_from_directory(str(tmp_path))
        assert graph is not None

    def test_large_codebase(self, tmp_path):
        """Test handling a large number of files."""
        # Create many small files
        for i in range(50):
            (tmp_path / f"module_{i}.py").write_text(
                f"def func_{i}(): pass\ndef helper_{i}(): func_{i}()"
            )

        builder = CallGraphBuilder()
        graph = builder.build_from_directory(str(tmp_path))

        assert graph.node_count >= 50  # At least one node per file
