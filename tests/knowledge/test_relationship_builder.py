"""
Tests for RelationshipBuilder - cross-file dependency graph construction.

Tests cover:
- RelationshipKind enum
- CodeEntity dataclass and factory methods
- Relationship dataclass
- RelationshipGraph class
- RelationshipBuilder class with all public methods

Run with:
    pytest tests/knowledge/test_relationship_builder.py -v --asyncio-mode=auto
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.connectors.repository_crawler import (
    CrawledFile,
    CrawlResult,
    FileDependency,
    FileSymbol,
    FileType,
)
from aragora.knowledge.relationship_builder import (
    CodeEntity,
    Relationship,
    RelationshipBuilder,
    RelationshipGraph,
    RelationshipKind,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_file_symbol() -> FileSymbol:
    """Create a sample FileSymbol for testing."""
    return FileSymbol(
        name="my_function",
        kind="function",
        line_start=10,
        line_end=25,
        signature="def my_function(x: int, y: str) -> bool",
        docstring="A sample function for testing.",
        parent=None,
    )


@pytest.fixture
def sample_method_symbol() -> FileSymbol:
    """Create a sample method FileSymbol with a parent."""
    return FileSymbol(
        name="process",
        kind="method",
        line_start=20,
        line_end=30,
        signature="def process(self, data: dict) -> None",
        docstring="Process some data.",
        parent="MyClass",
    )


@pytest.fixture
def sample_class_symbol() -> FileSymbol:
    """Create a sample class FileSymbol."""
    return FileSymbol(
        name="MyClass",
        kind="class",
        line_start=5,
        line_end=50,
        signature="class MyClass",
        docstring="A sample class.",
        parent=None,
    )


@pytest.fixture
def sample_crawled_file(
    sample_file_symbol, sample_class_symbol, sample_method_symbol
) -> CrawledFile:
    """Create a sample CrawledFile for testing."""
    return CrawledFile(
        path="/repo/src/module.py",
        relative_path="src/module.py",
        file_type=FileType.PYTHON,
        content="# Sample content",
        content_hash="abc123",
        size_bytes=1024,
        line_count=100,
        symbols=[sample_class_symbol, sample_file_symbol, sample_method_symbol],
        dependencies=[
            FileDependency(
                source="src/module.py",
                target="utils.helpers",
                kind="import",
                line=1,
            ),
            FileDependency(
                source="src/module.py",
                target="os",
                kind="import",
                line=2,
            ),
        ],
    )


@pytest.fixture
def sample_crawled_file_utils() -> CrawledFile:
    """Create a utils file for dependency testing."""
    return CrawledFile(
        path="/repo/utils/helpers.py",
        relative_path="utils/helpers.py",
        file_type=FileType.PYTHON,
        content="# Utils content",
        content_hash="def456",
        size_bytes=512,
        line_count=50,
        symbols=[
            FileSymbol(
                name="helper_function",
                kind="function",
                line_start=5,
                line_end=15,
            ),
        ],
        dependencies=[],
    )


@pytest.fixture
def sample_crawl_result(sample_crawled_file, sample_crawled_file_utils) -> CrawlResult:
    """Create a sample CrawlResult for testing."""
    return CrawlResult(
        repository_path="/repo",
        repository_name="test-repo",
        files=[sample_crawled_file, sample_crawled_file_utils],
        total_files=2,
        total_lines=150,
        total_bytes=1536,
        file_type_counts={"python": 2},
        symbol_counts={"function": 2, "class": 1, "method": 1},
        dependency_graph={
            "src/module.py": ["utils/helpers.py"],
        },
        crawl_duration_ms=100.0,
        errors=[],
        warnings=[],
    )


@pytest.fixture
def relationship_builder() -> RelationshipBuilder:
    """Create a RelationshipBuilder instance."""
    return RelationshipBuilder(repository_name="test-repo")


# =============================================================================
# RelationshipKind Tests
# =============================================================================


class TestRelationshipKind:
    """Tests for RelationshipKind enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert RelationshipKind.IMPORTS.value == "imports"
        assert RelationshipKind.CALLS.value == "calls"
        assert RelationshipKind.INHERITS.value == "inherits"
        assert RelationshipKind.IMPLEMENTS.value == "implements"
        assert RelationshipKind.REFERENCES.value == "references"
        assert RelationshipKind.CONTAINS.value == "contains"
        assert RelationshipKind.MEMBER_OF.value == "member_of"
        assert RelationshipKind.USES.value == "uses"

    def test_enum_is_string(self):
        """Test enum inherits from str for easy comparison."""
        assert isinstance(RelationshipKind.IMPORTS, str)
        assert RelationshipKind.IMPORTS == "imports"


# =============================================================================
# CodeEntity Tests
# =============================================================================


class TestCodeEntity:
    """Tests for CodeEntity dataclass."""

    def test_basic_creation(self):
        """Test creating a CodeEntity with basic fields."""
        entity = CodeEntity(
            id="repo:file.py:function",
            name="function",
            kind="function",
            file_path="file.py",
            line_start=1,
            line_end=10,
        )
        assert entity.id == "repo:file.py:function"
        assert entity.name == "function"
        assert entity.kind == "function"
        assert entity.signature is None
        assert entity.docstring is None
        assert entity.parent_id is None
        assert entity.metadata == {}

    def test_full_creation(self):
        """Test creating a CodeEntity with all fields."""
        entity = CodeEntity(
            id="repo:file.py:MyClass.method",
            name="method",
            kind="method",
            file_path="file.py",
            line_start=20,
            line_end=30,
            signature="def method(self) -> None",
            docstring="A method.",
            parent_id="repo:file.py:MyClass",
            metadata={"visibility": "public"},
        )
        assert entity.signature == "def method(self) -> None"
        assert entity.docstring == "A method."
        assert entity.parent_id == "repo:file.py:MyClass"
        assert entity.metadata["visibility"] == "public"

    def test_from_file(self, sample_crawled_file):
        """Test creating CodeEntity from CrawledFile."""
        entity = CodeEntity.from_file(sample_crawled_file, "test-repo")

        assert entity.id == "test-repo:src/module.py"
        assert entity.name == "src/module.py"
        assert entity.kind == "file"
        assert entity.file_path == "src/module.py"
        assert entity.line_start == 1
        assert entity.line_end == 100
        assert entity.metadata["file_type"] == "python"
        assert entity.metadata["content_hash"] == "abc123"
        assert entity.metadata["size_bytes"] == 1024

    def test_from_symbol_no_parent(self, sample_file_symbol):
        """Test creating CodeEntity from FileSymbol without parent."""
        entity = CodeEntity.from_symbol(sample_file_symbol, "src/module.py", "test-repo")

        assert entity.id == "test-repo:src/module.py:my_function"
        assert entity.name == "my_function"
        assert entity.kind == "function"
        assert entity.file_path == "src/module.py"
        assert entity.line_start == 10
        assert entity.line_end == 25
        assert entity.signature == "def my_function(x: int, y: str) -> bool"
        assert entity.docstring == "A sample function for testing."
        assert entity.parent_id is None

    def test_from_symbol_with_parent(self, sample_method_symbol):
        """Test creating CodeEntity from FileSymbol with parent."""
        entity = CodeEntity.from_symbol(sample_method_symbol, "src/module.py", "test-repo")

        assert entity.id == "test-repo:src/module.py:MyClass.process"
        assert entity.name == "process"
        assert entity.kind == "method"
        assert entity.parent_id == "test-repo:src/module.py:MyClass"


# =============================================================================
# Relationship Tests
# =============================================================================


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_basic_creation(self):
        """Test creating a Relationship with basic fields."""
        rel = Relationship(
            source_id="repo:a.py",
            target_id="repo:b.py",
            kind=RelationshipKind.IMPORTS,
        )
        assert rel.source_id == "repo:a.py"
        assert rel.target_id == "repo:b.py"
        assert rel.kind == RelationshipKind.IMPORTS
        assert rel.weight == 1.0
        assert rel.line is None
        assert rel.metadata == {}

    def test_full_creation(self):
        """Test creating a Relationship with all fields."""
        rel = Relationship(
            source_id="repo:a.py:ClassA",
            target_id="repo:b.py:ClassB",
            kind=RelationshipKind.INHERITS,
            weight=0.9,
            line=15,
            metadata={"via": "extends"},
        )
        assert rel.weight == 0.9
        assert rel.line == 15
        assert rel.metadata["via"] == "extends"

    def test_to_dict(self):
        """Test converting Relationship to dictionary."""
        rel = Relationship(
            source_id="repo:a.py",
            target_id="repo:b.py",
            kind=RelationshipKind.CALLS,
            weight=0.8,
            line=25,
            metadata={"call_type": "direct"},
        )
        result = rel.to_dict()

        assert result["source_id"] == "repo:a.py"
        assert result["target_id"] == "repo:b.py"
        assert result["kind"] == "calls"
        assert result["weight"] == 0.8
        assert result["line"] == 25
        assert result["metadata"]["call_type"] == "direct"


# =============================================================================
# RelationshipGraph Tests
# =============================================================================


class TestRelationshipGraph:
    """Tests for RelationshipGraph class."""

    @pytest.fixture
    def graph(self) -> RelationshipGraph:
        """Create a fresh RelationshipGraph."""
        return RelationshipGraph()

    def test_init(self, graph):
        """Test graph initialization."""
        assert graph.entities == {}
        assert graph.relationships == []
        assert len(graph.outgoing) == 0
        assert len(graph.incoming) == 0

    def test_add_entity(self, graph):
        """Test adding an entity to the graph."""
        entity = CodeEntity(
            id="repo:file.py",
            name="file.py",
            kind="file",
            file_path="file.py",
            line_start=1,
            line_end=100,
        )
        graph.add_entity(entity)

        assert "repo:file.py" in graph.entities
        assert graph.entities["repo:file.py"] is entity

    def test_add_relationship(self, graph):
        """Test adding a relationship to the graph."""
        rel = Relationship(
            source_id="repo:a.py",
            target_id="repo:b.py",
            kind=RelationshipKind.IMPORTS,
        )
        graph.add_relationship(rel)

        assert len(graph.relationships) == 1
        assert rel in graph.relationships
        assert rel in graph.outgoing["repo:a.py"]
        assert rel in graph.incoming["repo:b.py"]

    def test_get_dependencies(self, graph):
        """Test getting dependencies of an entity."""
        # Add entities
        entity_a = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        entity_b = CodeEntity(
            id="repo:b.py", name="b.py", kind="file", file_path="b.py", line_start=1, line_end=10
        )
        entity_c = CodeEntity(
            id="repo:c.py", name="c.py", kind="file", file_path="c.py", line_start=1, line_end=10
        )
        graph.add_entity(entity_a)
        graph.add_entity(entity_b)
        graph.add_entity(entity_c)

        # Add relationships: a -> b, a -> c
        graph.add_relationship(Relationship("repo:a.py", "repo:b.py", RelationshipKind.IMPORTS))
        graph.add_relationship(Relationship("repo:a.py", "repo:c.py", RelationshipKind.CALLS))

        deps = graph.get_dependencies("repo:a.py")
        assert len(deps) == 2
        assert entity_b in deps
        assert entity_c in deps

    def test_get_dependencies_filtered_by_kind(self, graph):
        """Test getting dependencies filtered by relationship kind."""
        entity_a = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        entity_b = CodeEntity(
            id="repo:b.py", name="b.py", kind="file", file_path="b.py", line_start=1, line_end=10
        )
        entity_c = CodeEntity(
            id="repo:c.py", name="c.py", kind="file", file_path="c.py", line_start=1, line_end=10
        )
        graph.add_entity(entity_a)
        graph.add_entity(entity_b)
        graph.add_entity(entity_c)

        graph.add_relationship(Relationship("repo:a.py", "repo:b.py", RelationshipKind.IMPORTS))
        graph.add_relationship(Relationship("repo:a.py", "repo:c.py", RelationshipKind.CALLS))

        deps = graph.get_dependencies("repo:a.py", {RelationshipKind.IMPORTS})
        assert len(deps) == 1
        assert entity_b in deps

    def test_get_dependents(self, graph):
        """Test getting entities that depend on a given entity."""
        entity_a = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        entity_b = CodeEntity(
            id="repo:b.py", name="b.py", kind="file", file_path="b.py", line_start=1, line_end=10
        )
        entity_c = CodeEntity(
            id="repo:c.py", name="c.py", kind="file", file_path="c.py", line_start=1, line_end=10
        )
        graph.add_entity(entity_a)
        graph.add_entity(entity_b)
        graph.add_entity(entity_c)

        # a -> c, b -> c (both depend on c)
        graph.add_relationship(Relationship("repo:a.py", "repo:c.py", RelationshipKind.IMPORTS))
        graph.add_relationship(Relationship("repo:b.py", "repo:c.py", RelationshipKind.IMPORTS))

        dependents = graph.get_dependents("repo:c.py")
        assert len(dependents) == 2
        assert entity_a in dependents
        assert entity_b in dependents

    def test_get_dependents_filtered_by_kind(self, graph):
        """Test getting dependents filtered by relationship kind."""
        entity_a = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        entity_b = CodeEntity(
            id="repo:b.py", name="b.py", kind="file", file_path="b.py", line_start=1, line_end=10
        )
        entity_c = CodeEntity(
            id="repo:c.py", name="c.py", kind="file", file_path="c.py", line_start=1, line_end=10
        )
        graph.add_entity(entity_a)
        graph.add_entity(entity_b)
        graph.add_entity(entity_c)

        graph.add_relationship(Relationship("repo:a.py", "repo:c.py", RelationshipKind.IMPORTS))
        graph.add_relationship(Relationship("repo:b.py", "repo:c.py", RelationshipKind.CALLS))

        dependents = graph.get_dependents("repo:c.py", {RelationshipKind.CALLS})
        assert len(dependents) == 1
        assert entity_b in dependents

    def test_get_dependencies_entity_not_in_graph(self, graph):
        """Test getting dependencies returns empty if target entity not in graph."""
        entity_a = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        graph.add_entity(entity_a)

        # Add relationship to entity not in graph
        graph.add_relationship(
            Relationship("repo:a.py", "repo:missing.py", RelationshipKind.IMPORTS)
        )

        deps = graph.get_dependencies("repo:a.py")
        assert len(deps) == 0

    def test_to_dict(self, graph):
        """Test converting graph to dictionary."""
        entity = CodeEntity(
            id="repo:a.py", name="a.py", kind="file", file_path="a.py", line_start=1, line_end=10
        )
        graph.add_entity(entity)
        graph.add_relationship(Relationship("repo:a.py", "repo:b.py", RelationshipKind.IMPORTS))

        result = graph.to_dict()

        assert result["entity_count"] == 1
        assert result["relationship_count"] == 1
        assert "repo:a.py" in result["entities"]
        assert result["entities"]["repo:a.py"]["name"] == "a.py"
        assert len(result["relationships"]) == 1


# =============================================================================
# RelationshipBuilder Tests
# =============================================================================


class TestRelationshipBuilder:
    """Tests for RelationshipBuilder class."""

    def test_init(self, relationship_builder):
        """Test builder initialization."""
        assert relationship_builder.repository_name == "test-repo"
        assert isinstance(relationship_builder._graph, RelationshipGraph)
        assert relationship_builder._symbol_index == {}

    @pytest.mark.asyncio
    async def test_build_graph(self, relationship_builder, sample_crawl_result):
        """Test building a complete relationship graph."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        assert isinstance(graph, RelationshipGraph)
        # Should have file entities + symbol entities
        assert len(graph.entities) >= 2  # At least 2 files

    @pytest.mark.asyncio
    async def test_build_graph_indexes_files(self, relationship_builder, sample_crawl_result):
        """Test that build_graph indexes file entities."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        # Check file entities exist
        assert "test-repo:src/module.py" in graph.entities
        assert "test-repo:utils/helpers.py" in graph.entities

        file_entity = graph.entities["test-repo:src/module.py"]
        assert file_entity.kind == "file"

    @pytest.mark.asyncio
    async def test_build_graph_indexes_symbols(self, relationship_builder, sample_crawl_result):
        """Test that build_graph indexes symbol entities."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        # Check symbol entities
        assert "test-repo:src/module.py:MyClass" in graph.entities
        assert "test-repo:src/module.py:my_function" in graph.entities

    @pytest.mark.asyncio
    async def test_build_graph_indexes_methods_with_parent(
        self, relationship_builder, sample_crawl_result
    ):
        """Test that build_graph indexes methods with qualified names."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        # Method should have qualified name with parent
        assert "test-repo:src/module.py:MyClass.process" in graph.entities
        method = graph.entities["test-repo:src/module.py:MyClass.process"]
        assert method.parent_id == "test-repo:src/module.py:MyClass"

    @pytest.mark.asyncio
    async def test_build_graph_creates_contains_relationships(
        self, relationship_builder, sample_crawl_result
    ):
        """Test that build_graph creates CONTAINS relationships."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        # File should contain symbols
        contains_rels = [r for r in graph.relationships if r.kind == RelationshipKind.CONTAINS]
        assert len(contains_rels) > 0

    @pytest.mark.asyncio
    async def test_build_graph_creates_member_of_relationships(
        self, relationship_builder, sample_crawl_result
    ):
        """Test that build_graph creates MEMBER_OF relationships for methods."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        member_rels = [r for r in graph.relationships if r.kind == RelationshipKind.MEMBER_OF]
        assert len(member_rels) > 0

        # Method should be member of class
        method_rel = next((r for r in member_rels if "MyClass.process" in r.source_id), None)
        assert method_rel is not None
        assert "MyClass" in method_rel.target_id

    @pytest.mark.asyncio
    async def test_build_graph_creates_import_relationships(
        self, relationship_builder, sample_crawl_result
    ):
        """Test that build_graph creates IMPORTS relationships from dependency graph."""
        graph = await relationship_builder.build_graph(sample_crawl_result)

        import_rels = [r for r in graph.relationships if r.kind == RelationshipKind.IMPORTS]
        # Should have import from src/module.py to utils/helpers.py
        assert len(import_rels) >= 1

    @pytest.mark.asyncio
    async def test_find_dependencies_depth_1(self, relationship_builder, sample_crawl_result):
        """Test finding dependencies at depth 1."""
        await relationship_builder.build_graph(sample_crawl_result)

        deps = await relationship_builder.find_dependencies(
            "test-repo:src/module.py",
            depth=1,
        )
        # Should find utils/helpers.py
        dep_ids = [d.id for d in deps]
        assert "test-repo:utils/helpers.py" in dep_ids

    @pytest.mark.asyncio
    async def test_find_dependencies_filtered(self, relationship_builder, sample_crawl_result):
        """Test finding dependencies filtered by kind."""
        await relationship_builder.build_graph(sample_crawl_result)

        # Only look for CALLS (none in sample)
        deps = await relationship_builder.find_dependencies(
            "test-repo:src/module.py",
            depth=2,
            relationship_kinds={RelationshipKind.CALLS},
        )
        # Should be empty since no CALLS relationships
        assert len(deps) == 0

    @pytest.mark.asyncio
    async def test_find_dependents(self, relationship_builder, sample_crawl_result):
        """Test finding entities that depend on a given entity."""
        await relationship_builder.build_graph(sample_crawl_result)

        dependents = await relationship_builder.find_dependents(
            "test-repo:utils/helpers.py",
            depth=1,
        )
        # src/module.py depends on utils/helpers.py
        dep_ids = [d.id for d in dependents]
        assert "test-repo:src/module.py" in dep_ids

    @pytest.mark.asyncio
    async def test_find_dependents_empty_for_no_incoming(
        self, relationship_builder, sample_crawl_result
    ):
        """Test finding dependents returns empty when no incoming edges."""
        await relationship_builder.build_graph(sample_crawl_result)

        # src/module.py has no incoming import edges (nothing imports it)
        dependents = await relationship_builder.find_dependents(
            "test-repo:src/module.py",
            depth=1,
            relationship_kinds={RelationshipKind.IMPORTS},
        )
        assert len(dependents) == 0

    @pytest.mark.asyncio
    async def test_get_call_graph(self, relationship_builder, sample_crawl_result):
        """Test getting call graph for a function."""
        await relationship_builder.build_graph(sample_crawl_result)

        call_graph = await relationship_builder.get_call_graph(
            "test-repo:src/module.py:my_function",
            depth=3,
        )

        assert call_graph["root"] == "test-repo:src/module.py:my_function"
        assert call_graph["depth"] == 3
        assert "calls" in call_graph
        assert isinstance(call_graph["calls"], list)

    @pytest.mark.asyncio
    async def test_get_inheritance_tree_both_directions(
        self, relationship_builder, sample_crawl_result
    ):
        """Test getting inheritance tree in both directions."""
        await relationship_builder.build_graph(sample_crawl_result)

        tree = await relationship_builder.get_inheritance_tree(
            "test-repo:src/module.py:MyClass",
            direction="both",
        )

        assert tree["class_id"] == "test-repo:src/module.py:MyClass"
        assert "ancestors" in tree
        assert "descendants" in tree

    @pytest.mark.asyncio
    async def test_get_inheritance_tree_up_only(self, relationship_builder, sample_crawl_result):
        """Test getting inheritance tree upward only."""
        await relationship_builder.build_graph(sample_crawl_result)

        tree = await relationship_builder.get_inheritance_tree(
            "test-repo:src/module.py:MyClass",
            direction="up",
        )

        assert tree["class_id"] == "test-repo:src/module.py:MyClass"
        assert len(tree["ancestors"]) == 0  # No parents in sample
        assert len(tree["descendants"]) == 0  # Not searched

    @pytest.mark.asyncio
    async def test_get_inheritance_tree_down_only(self, relationship_builder, sample_crawl_result):
        """Test getting inheritance tree downward only."""
        await relationship_builder.build_graph(sample_crawl_result)

        tree = await relationship_builder.get_inheritance_tree(
            "test-repo:src/module.py:MyClass",
            direction="down",
        )

        assert tree["class_id"] == "test-repo:src/module.py:MyClass"
        assert len(tree["ancestors"]) == 0  # Not searched
        assert len(tree["descendants"]) == 0  # No children in sample

    def test_get_entity(self, relationship_builder):
        """Test getting an entity by ID."""
        # Add entity manually
        entity = CodeEntity(
            id="test-repo:file.py",
            name="file.py",
            kind="file",
            file_path="file.py",
            line_start=1,
            line_end=10,
        )
        relationship_builder._graph.add_entity(entity)

        result = relationship_builder.get_entity("test-repo:file.py")
        assert result is entity

    def test_get_entity_not_found(self, relationship_builder):
        """Test getting an entity that doesn't exist."""
        result = relationship_builder.get_entity("nonexistent")
        assert result is None

    def test_get_graph(self, relationship_builder):
        """Test getting the current graph."""
        graph = relationship_builder.get_graph()
        assert graph is relationship_builder._graph

    @pytest.mark.asyncio
    async def test_get_statistics(self, relationship_builder, sample_crawl_result):
        """Test getting graph statistics."""
        await relationship_builder.build_graph(sample_crawl_result)

        stats = relationship_builder.get_statistics()

        assert "total_entities" in stats
        assert "total_relationships" in stats
        assert "entity_kinds" in stats
        assert "relationship_kinds" in stats
        assert stats["total_entities"] > 0
        assert stats["total_relationships"] > 0

    def test_dependency_kind_to_relationship_import(self, relationship_builder):
        """Test converting 'import' kind to IMPORTS."""
        result = relationship_builder._dependency_kind_to_relationship("import")
        assert result == RelationshipKind.IMPORTS

    def test_dependency_kind_to_relationship_from_import(self, relationship_builder):
        """Test converting 'from_import' kind to IMPORTS."""
        result = relationship_builder._dependency_kind_to_relationship("from_import")
        assert result == RelationshipKind.IMPORTS

    def test_dependency_kind_to_relationship_extends(self, relationship_builder):
        """Test converting 'extends' kind to INHERITS."""
        result = relationship_builder._dependency_kind_to_relationship("extends")
        assert result == RelationshipKind.INHERITS

    def test_dependency_kind_to_relationship_implements(self, relationship_builder):
        """Test converting 'implements' kind to IMPLEMENTS."""
        result = relationship_builder._dependency_kind_to_relationship("implements")
        assert result == RelationshipKind.IMPLEMENTS

    def test_dependency_kind_to_relationship_call(self, relationship_builder):
        """Test converting 'call' kind to CALLS."""
        result = relationship_builder._dependency_kind_to_relationship("call")
        assert result == RelationshipKind.CALLS

    def test_dependency_kind_to_relationship_unknown(self, relationship_builder):
        """Test converting unknown kind to USES."""
        result = relationship_builder._dependency_kind_to_relationship("unknown")
        assert result == RelationshipKind.USES


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestRelationshipBuilderEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_build_graph_empty_crawl_result(self, relationship_builder):
        """Test building graph from empty crawl result."""
        empty_result = CrawlResult(
            repository_path="/empty",
            repository_name="empty-repo",
            files=[],
            total_files=0,
            total_lines=0,
            total_bytes=0,
            file_type_counts={},
            symbol_counts={},
            dependency_graph={},
            crawl_duration_ms=0.0,
            errors=[],
            warnings=[],
        )

        graph = await relationship_builder.build_graph(empty_result)

        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0

    @pytest.mark.asyncio
    async def test_find_dependencies_nonexistent_entity(
        self, relationship_builder, sample_crawl_result
    ):
        """Test finding dependencies for nonexistent entity."""
        await relationship_builder.build_graph(sample_crawl_result)

        deps = await relationship_builder.find_dependencies("nonexistent", depth=2)
        assert deps == []

    @pytest.mark.asyncio
    async def test_find_dependencies_depth_zero(self, relationship_builder, sample_crawl_result):
        """Test finding dependencies with depth 0."""
        await relationship_builder.build_graph(sample_crawl_result)

        deps = await relationship_builder.find_dependencies(
            "test-repo:src/module.py",
            depth=0,
        )
        assert deps == []

    @pytest.mark.asyncio
    async def test_resolve_dependency_with_extensions(self, relationship_builder):
        """Test resolving dependency targets with common extensions."""
        # Add entity with .py extension
        entity = CodeEntity(
            id="test-repo:utils/helpers.py",
            name="helpers.py",
            kind="file",
            file_path="utils/helpers.py",
            line_start=1,
            line_end=10,
        )
        relationship_builder._graph.add_entity(entity)

        # Create dependency without extension
        dep = FileDependency(
            source="src/main.py",
            target="utils/helpers",  # No extension
            kind="import",
            line=1,
        )

        result = relationship_builder._resolve_dependency_target(dep)
        assert result == "test-repo:utils/helpers.py"

    @pytest.mark.asyncio
    async def test_resolve_dependency_module_path(self, relationship_builder):
        """Test resolving dependency with module path (dots to slashes)."""
        # Add entity with proper file path
        entity = CodeEntity(
            id="test-repo:aragora/utils/helpers.py",
            name="helpers.py",
            kind="file",
            file_path="aragora/utils/helpers.py",
            line_start=1,
            line_end=10,
        )
        relationship_builder._graph.add_entity(entity)

        # Create dependency with module path
        dep = FileDependency(
            source="src/main.py",
            target="aragora.utils.helpers",  # Module path with dots
            kind="import",
            line=1,
        )

        result = relationship_builder._resolve_dependency_target(dep)
        assert result == "test-repo:aragora/utils/helpers.py"

    @pytest.mark.asyncio
    async def test_multiple_build_graph_resets_state(
        self, relationship_builder, sample_crawl_result
    ):
        """Test that calling build_graph multiple times resets state."""
        graph1 = await relationship_builder.build_graph(sample_crawl_result)
        entity_count_1 = len(graph1.entities)

        graph2 = await relationship_builder.build_graph(sample_crawl_result)
        entity_count_2 = len(graph2.entities)

        # Should have same count, not doubled
        assert entity_count_1 == entity_count_2

    @pytest.mark.asyncio
    async def test_build_relationships_skip_missing_parent(self, relationship_builder):
        """Test building relationships skips MEMBER_OF when parent doesn't exist."""
        # Create file with method but no class
        orphan_method = FileSymbol(
            name="orphan_method",
            kind="method",
            line_start=10,
            line_end=20,
            parent="MissingClass",  # Parent not in symbols list
        )

        crawled_file = CrawledFile(
            path="/repo/orphan.py",
            relative_path="orphan.py",
            file_type=FileType.PYTHON,
            content="# Content",
            content_hash="hash123",
            size_bytes=100,
            line_count=30,
            symbols=[orphan_method],
            dependencies=[],
        )

        crawl_result = CrawlResult(
            repository_path="/repo",
            repository_name="test-repo",
            files=[crawled_file],
            total_files=1,
            total_lines=30,
            total_bytes=100,
            file_type_counts={"python": 1},
            symbol_counts={"method": 1},
            dependency_graph={},
            crawl_duration_ms=10.0,
            errors=[],
            warnings=[],
        )

        graph = await relationship_builder.build_graph(crawl_result)

        # Should still build without error
        assert len(graph.entities) == 2  # File + orphan method

        # No MEMBER_OF relationship because parent doesn't exist
        member_rels = [r for r in graph.relationships if r.kind == RelationshipKind.MEMBER_OF]
        assert len(member_rels) == 0
