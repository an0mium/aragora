"""Tests for the symbol extraction module."""

import pytest
from pathlib import Path

from aragora.analysis.intelligence.types import (
    ClassInfo,
    FileAnalysis,
    FunctionInfo,
    ImportInfo,
    Language,
    SourceLocation,
    SymbolKind,
    Visibility,
)
from aragora.analysis.intelligence.symbols import (
    SymbolReference,
    SymbolTable,
    analyze_with_regex,
    extract_symbols_from_analysis,
    find_symbol_definition,
    find_symbol_usages,
    get_exported_symbols,
    get_symbol_at_location,
)


# ---------------------------------------------------------------------------
# SymbolReference dataclass
# ---------------------------------------------------------------------------


class TestSymbolReference:
    def test_basic_creation(self):
        loc = SourceLocation("test.py", 1, 0, 1, 10)
        ref = SymbolReference(name="foo", location=loc)
        assert ref.name == "foo"
        assert ref.kind is None
        assert ref.is_definition is False
        assert ref.is_import is False

    def test_definition_flag(self):
        loc = SourceLocation("test.py", 1, 0, 1, 10)
        ref = SymbolReference(
            name="MyClass", location=loc, kind=SymbolKind.CLASS, is_definition=True
        )
        assert ref.is_definition is True
        assert ref.kind == SymbolKind.CLASS

    def test_import_flag(self):
        loc = SourceLocation("test.py", 1, 0, 1, 10)
        ref = SymbolReference(name="os", location=loc, kind=SymbolKind.IMPORT, is_import=True)
        assert ref.is_import is True


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------


class TestSymbolTable:
    def _make_ref(self, name: str, *, is_definition: bool = False) -> SymbolReference:
        loc = SourceLocation("test.py", 1, 0, 1, 10)
        return SymbolReference(name=name, location=loc, is_definition=is_definition)

    def test_empty_table(self):
        table = SymbolTable()
        assert table.symbols == {}
        assert table.imports == {}

    def test_add_symbol(self):
        table = SymbolTable()
        ref = self._make_ref("foo")
        table.add_symbol(ref)
        assert "foo" in table.symbols
        assert len(table.symbols["foo"]) == 1

    def test_add_multiple_refs_same_name(self):
        table = SymbolTable()
        table.add_symbol(self._make_ref("foo"))
        table.add_symbol(self._make_ref("foo", is_definition=True))
        assert len(table.symbols["foo"]) == 2

    def test_get_definitions(self):
        table = SymbolTable()
        table.add_symbol(self._make_ref("foo"))
        table.add_symbol(self._make_ref("foo", is_definition=True))
        defs = table.get_definitions("foo")
        assert len(defs) == 1
        assert defs[0].is_definition is True

    def test_get_definitions_missing(self):
        table = SymbolTable()
        assert table.get_definitions("missing") == []

    def test_get_usages(self):
        table = SymbolTable()
        table.add_symbol(self._make_ref("bar"))
        table.add_symbol(self._make_ref("bar", is_definition=True))
        usages = table.get_usages("bar")
        assert len(usages) == 1
        assert usages[0].is_definition is False


# ---------------------------------------------------------------------------
# find_symbol_usages
# ---------------------------------------------------------------------------


class TestFindSymbolUsages:
    def test_find_in_directory(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = foo()\ny = bar()\nz = foo + 1")
        results = find_symbol_usages(str(tmp_path), "foo")
        assert len(results) == 2  # lines 1 and 3

    def test_find_with_language_filter(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("foo = 1")
        (tmp_path / "b.js").write_text("foo = 2")
        results = find_symbol_usages(str(tmp_path), "foo", language=Language.PYTHON)
        assert len(results) == 1
        assert "a.py" in results[0].file_path

    def test_no_match(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1")
        results = find_symbol_usages(str(tmp_path), "missing_symbol")
        assert results == []

    def test_word_boundary(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("foobar = 1\nfoo = 2")
        results = find_symbol_usages(str(tmp_path), "foo")
        # "foobar" should NOT match, only "foo"
        assert len(results) == 1

    def test_empty_directory(self, tmp_path: Path):
        results = find_symbol_usages(str(tmp_path), "foo")
        assert results == []


# ---------------------------------------------------------------------------
# get_symbol_at_location
# ---------------------------------------------------------------------------


class TestGetSymbolAtLocation:
    def _make_analysis(self) -> FileAnalysis:
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        analysis.functions.append(
            FunctionInfo(
                name="my_func",
                kind=SymbolKind.FUNCTION,
                location=SourceLocation("test.py", 5, 0, 10, 0),
            )
        )
        cls = ClassInfo(
            name="MyClass",
            kind=SymbolKind.CLASS,
            location=SourceLocation("test.py", 15, 0, 30, 0),
        )
        cls.methods.append(
            FunctionInfo(
                name="method",
                kind=SymbolKind.METHOD,
                location=SourceLocation("test.py", 18, 4, 22, 0),
            )
        )
        analysis.classes.append(cls)
        return analysis

    def test_find_function(self):
        analysis = self._make_analysis()
        result = get_symbol_at_location(analysis, 7, 5)
        assert result is not None
        assert result.name == "my_func"

    def test_find_class(self):
        analysis = self._make_analysis()
        result = get_symbol_at_location(analysis, 25, 0)
        assert result is not None
        assert result.name == "MyClass"

    def test_find_method_in_class(self):
        analysis = self._make_analysis()
        result = get_symbol_at_location(analysis, 20, 8)
        assert result is not None
        assert result.name == "method"

    def test_no_symbol(self):
        analysis = self._make_analysis()
        result = get_symbol_at_location(analysis, 1, 0)
        assert result is None


# ---------------------------------------------------------------------------
# extract_symbols_from_analysis
# ---------------------------------------------------------------------------


class TestExtractSymbolsFromAnalysis:
    def test_empty_analysis(self):
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        table = extract_symbols_from_analysis(analysis)
        assert table.symbols == {}

    def test_functions_extracted(self):
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        analysis.functions.append(
            FunctionInfo(
                name="helper",
                kind=SymbolKind.FUNCTION,
                location=SourceLocation("test.py", 1, 0, 5, 0),
            )
        )
        table = extract_symbols_from_analysis(analysis)
        assert "helper" in table.symbols
        assert table.symbols["helper"][0].is_definition is True

    def test_classes_and_methods_extracted(self):
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        cls = ClassInfo(
            name="Foo",
            kind=SymbolKind.CLASS,
            location=SourceLocation("test.py", 1, 0, 20, 0),
        )
        cls.methods.append(
            FunctionInfo(
                name="bar",
                kind=SymbolKind.METHOD,
                location=SourceLocation("test.py", 3, 4, 8, 0),
            )
        )
        analysis.classes.append(cls)
        table = extract_symbols_from_analysis(analysis)
        assert "Foo" in table.symbols
        assert "Foo.bar" in table.symbols

    def test_imports_extracted(self):
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        analysis.imports.append(
            ImportInfo(
                module="os",
                names=["path", "getcwd"],
                location=SourceLocation("test.py", 1, 0, 1, 30),
            )
        )
        table = extract_symbols_from_analysis(analysis)
        assert "os" in table.imports
        assert "path" in table.symbols
        assert "getcwd" in table.symbols

    def test_exports_extracted(self):
        analysis = FileAnalysis(file_path="test.py", language=Language.PYTHON)
        analysis.exports = ["public_func"]
        table = extract_symbols_from_analysis(analysis)
        assert "public_func" in table.symbols


# ---------------------------------------------------------------------------
# get_exported_symbols
# ---------------------------------------------------------------------------


class TestGetExportedSymbols:
    def test_uses_all_when_present(self):
        analysis = FileAnalysis(file_path="test.py")
        analysis.exports = ["A", "B"]
        analysis.functions.append(
            FunctionInfo(
                name="C",
                kind=SymbolKind.FUNCTION,
                location=SourceLocation("test.py", 1, 0, 1, 0),
            )
        )
        result = get_exported_symbols(analysis)
        assert result == ["A", "B"]

    def test_collects_public_when_no_all(self):
        analysis = FileAnalysis(file_path="test.py")
        analysis.functions.append(
            FunctionInfo(
                name="public_func",
                kind=SymbolKind.FUNCTION,
                location=SourceLocation("test.py", 1, 0, 1, 0),
            )
        )
        analysis.functions.append(
            FunctionInfo(
                name="_private_func",
                kind=SymbolKind.FUNCTION,
                location=SourceLocation("test.py", 5, 0, 5, 0),
            )
        )
        analysis.classes.append(
            ClassInfo(
                name="PublicClass",
                kind=SymbolKind.CLASS,
                location=SourceLocation("test.py", 10, 0, 20, 0),
            )
        )
        result = get_exported_symbols(analysis)
        assert "public_func" in result
        assert "PublicClass" in result
        assert "_private_func" not in result


# ---------------------------------------------------------------------------
# find_symbol_definition
# ---------------------------------------------------------------------------


class TestFindSymbolDefinition:
    def test_find_function(self):
        loc = SourceLocation("a.py", 10, 0, 15, 0)
        analysis = FileAnalysis(file_path="a.py")
        analysis.functions.append(
            FunctionInfo(name="target", kind=SymbolKind.FUNCTION, location=loc)
        )
        result = find_symbol_definition("target", {"a.py": analysis})
        assert result == loc

    def test_find_class(self):
        loc = SourceLocation("b.py", 20, 0, 40, 0)
        analysis = FileAnalysis(file_path="b.py")
        analysis.classes.append(ClassInfo(name="Target", kind=SymbolKind.CLASS, location=loc))
        result = find_symbol_definition("Target", {"b.py": analysis})
        assert result == loc

    def test_find_method(self):
        method_loc = SourceLocation("c.py", 25, 4, 30, 0)
        cls = ClassInfo(
            name="MyClass",
            kind=SymbolKind.CLASS,
            location=SourceLocation("c.py", 20, 0, 50, 0),
        )
        cls.methods.append(
            FunctionInfo(name="target_method", kind=SymbolKind.METHOD, location=method_loc)
        )
        analysis = FileAnalysis(file_path="c.py")
        analysis.classes.append(cls)
        result = find_symbol_definition("target_method", {"c.py": analysis})
        assert result == method_loc

    def test_not_found(self):
        analysis = FileAnalysis(file_path="a.py")
        result = find_symbol_definition("missing", {"a.py": analysis})
        assert result is None


# ---------------------------------------------------------------------------
# analyze_with_regex (Python)
# ---------------------------------------------------------------------------


class TestAnalyzeWithRegexPython:
    def test_detects_functions(self):
        source = "def foo():\n    pass\n\ndef bar(x, y):\n    return x + y"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        names = [f.name for f in analysis.functions]
        assert "foo" in names
        assert "bar" in names

    def test_detects_async_functions(self):
        source = "async def fetch():\n    pass"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        assert len(analysis.functions) == 1
        assert analysis.functions[0].name == "fetch"
        assert analysis.functions[0].is_async is True

    def test_detects_classes(self):
        source = "class Foo:\n    pass\n\nclass Bar(Base):\n    pass"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        names = [c.name for c in analysis.classes]
        assert "Foo" in names
        assert "Bar" in names

    def test_detects_imports(self):
        source = "import os\nfrom pathlib import Path\nfrom typing import Any, Optional"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        modules = [i.module for i in analysis.imports]
        assert "os" in modules
        assert "pathlib" in modules
        assert "typing" in modules

    def test_ignores_indented_functions(self):
        source = "class Foo:\n    def method(self):\n        pass\n\ndef top_level():\n    pass"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        # Only top-level functions should be detected
        func_names = [f.name for f in analysis.functions]
        assert "top_level" in func_names
        assert "method" not in func_names

    def test_empty_source(self):
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex("", analysis, Language.PYTHON)
        assert analysis.functions == []
        assert analysis.classes == []
        assert analysis.imports == []

    def test_class_bases_extracted(self):
        source = "class Child(Parent, Mixin):\n    pass"
        analysis = FileAnalysis(file_path="test.py")
        analyze_with_regex(source, analysis, Language.PYTHON)
        assert len(analysis.classes) == 1
        assert "Parent" in analysis.classes[0].bases
        assert "Mixin" in analysis.classes[0].bases
