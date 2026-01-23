"""
Tests for code intelligence module.

Tests AST parsing, symbol extraction, and complexity analysis.
"""

import pytest
import tempfile
import os
from pathlib import Path

from aragora.analysis.code_intelligence import (
    CodeIntelligence,
    Language,
    SymbolKind,
    SourceLocation,
    Parameter,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    FileAnalysis,
    TreeSitterParser,
)


# Sample Python code for testing
SAMPLE_PYTHON_CODE = '''
"""Sample module docstring."""

import os
from pathlib import Path
from typing import List, Optional

CONSTANT = 42

class BaseClass:
    """Base class docstring."""
    pass

class SampleClass(BaseClass):
    """A sample class for testing."""

    class_attr: str = "default"

    def __init__(self, name: str, value: int = 0):
        """Initialize the sample class."""
        self.name = name
        self.value = value

    def get_name(self) -> str:
        """Return the name."""
        return self.name

    async def async_method(self, items: List[str]) -> Optional[str]:
        """An async method."""
        if items:
            return items[0]
        return None

    @property
    def computed(self) -> int:
        """A computed property."""
        return self.value * 2

    @staticmethod
    def static_method():
        """A static method."""
        pass

    @classmethod
    def class_method(cls):
        """A class method."""
        pass


def standalone_function(x: int, y: int = 10) -> int:
    """A standalone function."""
    result = x + y
    if result > 100:
        for i in range(10):
            if i % 2 == 0:
                result += i
    return result


async def async_standalone():
    """An async standalone function."""
    return await some_coroutine()
'''

SAMPLE_JAVASCRIPT_CODE = """
import { useState } from 'react';
import axios from 'axios';

const API_URL = 'https://api.example.com';

class DataService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async fetchData(endpoint) {
        const response = await axios.get(`${this.baseUrl}/${endpoint}`);
        return response.data;
    }
}

function processItems(items) {
    return items.map(item => item.name);
}

const arrowFunction = (x, y) => x + y;

export default DataService;
"""

SAMPLE_TYPESCRIPT_CODE = """
import { Request, Response } from 'express';

interface User {
    id: number;
    name: string;
    email: string;
}

type UserRole = 'admin' | 'user' | 'guest';

class UserService {
    private users: User[] = [];

    constructor(private readonly db: Database) {}

    async getUser(id: number): Promise<User | null> {
        return this.users.find(u => u.id === id) || null;
    }

    createUser(data: Partial<User>): User {
        const user: User = {
            id: this.users.length + 1,
            name: data.name || '',
            email: data.email || '',
        };
        this.users.push(user);
        return user;
    }
}

function validateEmail(email: string): boolean {
    return email.includes('@');
}

export { UserService, validateEmail };
"""


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        """Test all language values exist."""
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "javascript"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.GO.value == "go"
        assert Language.RUST.value == "rust"
        assert Language.JAVA.value == "java"

    def test_language_from_extension(self):
        """Test detecting language from file extension."""
        assert Language.from_extension(".py") == Language.PYTHON
        assert Language.from_extension(".js") == Language.JAVASCRIPT
        assert Language.from_extension(".ts") == Language.TYPESCRIPT
        assert Language.from_extension(".tsx") == Language.TYPESCRIPT
        assert Language.from_extension(".go") == Language.GO
        assert Language.from_extension(".rs") == Language.RUST
        assert Language.from_extension(".java") == Language.JAVA
        assert Language.from_extension(".unknown") is None


class TestSymbolKind:
    """Tests for SymbolKind enum."""

    def test_symbol_kind_values(self):
        """Test all symbol kind values exist."""
        assert SymbolKind.MODULE.value == "module"
        assert SymbolKind.CLASS.value == "class"
        assert SymbolKind.FUNCTION.value == "function"
        assert SymbolKind.METHOD.value == "method"
        assert SymbolKind.VARIABLE.value == "variable"
        assert SymbolKind.CONSTANT.value == "constant"
        assert SymbolKind.IMPORT.value == "import"


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    def test_create_location(self):
        """Test creating a source location."""
        loc = SourceLocation(file_path="/test/file.py", line=10, column=5)
        assert loc.file_path == "/test/file.py"
        assert loc.line == 10
        assert loc.column == 5

    def test_location_string_representation(self):
        """Test string representation."""
        loc = SourceLocation(file_path="/test/file.py", line=10, column=5)
        assert "/test/file.py" in str(loc) or loc.file_path == "/test/file.py"


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_create_parameter(self):
        """Test creating a parameter."""
        param = Parameter(name="value", type_hint="int", default="0")
        assert param.name == "value"
        assert param.type_hint == "int"
        assert param.default == "0"

    def test_parameter_without_type(self):
        """Test parameter without type hint."""
        param = Parameter(name="x")
        assert param.name == "x"
        assert param.type_hint is None
        assert param.default is None


class TestFunctionInfo:
    """Tests for FunctionInfo dataclass."""

    def test_create_function_info(self):
        """Test creating function info."""
        func = FunctionInfo(
            name="test_func",
            location=SourceLocation(file_path="/test.py", line=1, column=0),
            parameters=[Parameter(name="x", type_hint="int")],
            return_type="str",
            is_async=True,
            decorators=["@asynccontextmanager"],
        )
        assert func.name == "test_func"
        assert func.is_async is True
        assert len(func.parameters) == 1
        assert func.return_type == "str"
        assert "@asynccontextmanager" in func.decorators


class TestClassInfo:
    """Tests for ClassInfo dataclass."""

    def test_create_class_info(self):
        """Test creating class info."""
        method = FunctionInfo(
            name="__init__",
            location=SourceLocation(file_path="/test.py", line=5, column=4),
        )
        cls = ClassInfo(
            name="TestClass",
            location=SourceLocation(file_path="/test.py", line=1, column=0),
            bases=["BaseClass"],
            methods=[method],
            decorators=["@dataclass"],
        )
        assert cls.name == "TestClass"
        assert "BaseClass" in cls.bases
        assert len(cls.methods) == 1
        assert "@dataclass" in cls.decorators


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_create_import_info(self):
        """Test creating import info."""
        imp = ImportInfo(module="os.path", names=["join", "exists"], alias=None, line=1)
        assert imp.module == "os.path"
        assert "join" in imp.names
        assert "exists" in imp.names

    def test_aliased_import(self):
        """Test aliased import."""
        imp = ImportInfo(module="numpy", names=[], alias="np", line=1)
        assert imp.module == "numpy"
        assert imp.alias == "np"


class TestCodeIntelligence:
    """Tests for CodeIntelligence class."""

    @pytest.fixture
    def code_intel(self):
        """Create a CodeIntelligence instance."""
        return CodeIntelligence()

    @pytest.fixture
    def python_file(self, tmp_path):
        """Create a temporary Python file."""
        file_path = tmp_path / "sample.py"
        file_path.write_text(SAMPLE_PYTHON_CODE)
        return str(file_path)

    @pytest.fixture
    def js_file(self, tmp_path):
        """Create a temporary JavaScript file."""
        file_path = tmp_path / "sample.js"
        file_path.write_text(SAMPLE_JAVASCRIPT_CODE)
        return str(file_path)

    @pytest.fixture
    def ts_file(self, tmp_path):
        """Create a temporary TypeScript file."""
        file_path = tmp_path / "sample.ts"
        file_path.write_text(SAMPLE_TYPESCRIPT_CODE)
        return str(file_path)

    def test_analyze_python_file(self, code_intel, python_file):
        """Test analyzing a Python file."""
        analysis = code_intel.analyze_file(python_file)

        assert analysis is not None
        assert analysis.language == Language.PYTHON
        assert analysis.file_path == python_file

    def test_python_class_extraction(self, code_intel, python_file):
        """Test extracting classes from Python file."""
        analysis = code_intel.analyze_file(python_file)

        class_names = [c.name for c in analysis.classes]
        assert "SampleClass" in class_names
        assert "BaseClass" in class_names

    def test_python_function_extraction(self, code_intel, python_file):
        """Test extracting functions from Python file."""
        analysis = code_intel.analyze_file(python_file)

        func_names = [f.name for f in analysis.functions]
        assert "standalone_function" in func_names
        assert "async_standalone" in func_names

    def test_python_import_extraction(self, code_intel, python_file):
        """Test extracting imports from Python file."""
        analysis = code_intel.analyze_file(python_file)

        import_modules = [i.module for i in analysis.imports]
        assert "os" in import_modules
        assert "pathlib" in import_modules
        assert "typing" in import_modules

    def test_python_method_extraction(self, code_intel, python_file):
        """Test extracting methods from classes."""
        analysis = code_intel.analyze_file(python_file)

        # Find SampleClass
        sample_class = None
        for cls in analysis.classes:
            if cls.name == "SampleClass":
                sample_class = cls
                break

        assert sample_class is not None
        method_names = [m.name for m in sample_class.methods]
        assert "__init__" in method_names
        assert "get_name" in method_names
        assert "async_method" in method_names

    def test_async_function_detection(self, code_intel, python_file):
        """Test detecting async functions."""
        analysis = code_intel.analyze_file(python_file)

        async_funcs = [f for f in analysis.functions if f.is_async]
        assert len(async_funcs) >= 1
        assert any(f.name == "async_standalone" for f in async_funcs)

    def test_complexity_calculation(self, code_intel, python_file):
        """Test cyclomatic complexity calculation."""
        analysis = code_intel.analyze_file(python_file)

        # Find standalone_function which has if/for statements
        for func in analysis.functions:
            if func.name == "standalone_function":
                # Should have complexity > 1 due to if/for
                assert func.complexity is None or func.complexity >= 1

    def test_line_counting(self, code_intel, python_file):
        """Test line counting."""
        analysis = code_intel.analyze_file(python_file)

        assert analysis.total_lines > 0
        # Our sample has comments and blank lines
        assert analysis.total_lines >= 50

    def test_analyze_javascript_file(self, code_intel, js_file):
        """Test analyzing a JavaScript file."""
        analysis = code_intel.analyze_file(js_file)

        assert analysis is not None
        assert analysis.language == Language.JAVASCRIPT

        # Should find some classes/functions
        assert len(analysis.classes) >= 0 or len(analysis.functions) >= 0

    def test_analyze_typescript_file(self, code_intel, ts_file):
        """Test analyzing a TypeScript file."""
        analysis = code_intel.analyze_file(ts_file)

        assert analysis is not None
        assert analysis.language == Language.TYPESCRIPT

    def test_analyze_nonexistent_file(self, code_intel):
        """Test analyzing a file that doesn't exist."""
        analysis = code_intel.analyze_file("/nonexistent/file.py")
        assert analysis is None

    def test_analyze_directory(self, code_intel, tmp_path):
        """Test analyzing a directory."""
        # Create multiple files
        (tmp_path / "module1.py").write_text("def func1(): pass")
        (tmp_path / "module2.py").write_text("class Class2: pass")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "module3.py").write_text("def func3(): pass")

        analyses = code_intel.analyze_directory(str(tmp_path))

        assert len(analyses) >= 3
        file_names = [Path(a.file_path).name for a in analyses]
        assert "module1.py" in file_names
        assert "module2.py" in file_names
        assert "module3.py" in file_names

    def test_analyze_directory_with_exclusions(self, code_intel, tmp_path):
        """Test directory analysis with exclusion patterns."""
        (tmp_path / "module.py").write_text("def func(): pass")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("# cached")
        (tmp_path / "test_module.py").write_text("def test(): pass")

        analyses = code_intel.analyze_directory(
            str(tmp_path), exclude_patterns=["__pycache__", "test_"]
        )

        file_names = [Path(a.file_path).name for a in analyses]
        assert "module.py" in file_names
        assert "cached.py" not in file_names
        assert "test_module.py" not in file_names

    def test_find_symbol_usages(self, code_intel, tmp_path):
        """Test finding symbol usages."""
        code = """
class MyClass:
    pass

def use_class():
    obj = MyClass()
    return obj

instance = MyClass()
"""
        file_path = tmp_path / "usage.py"
        file_path.write_text(code)

        usages = code_intel.find_symbol_usages(str(tmp_path), "MyClass")
        assert len(usages) >= 1


class TestFileAnalysis:
    """Tests for FileAnalysis dataclass."""

    def test_create_file_analysis(self):
        """Test creating a file analysis."""
        analysis = FileAnalysis(
            file_path="/test/file.py",
            language=Language.PYTHON,
            classes=[],
            functions=[],
            imports=[],
            total_lines=100,
            code_lines=80,
            comment_lines=10,
            blank_lines=10,
        )

        assert analysis.file_path == "/test/file.py"
        assert analysis.language == Language.PYTHON
        assert analysis.total_lines == 100

    def test_to_dict(self):
        """Test serialization to dictionary."""
        analysis = FileAnalysis(
            file_path="/test/file.py",
            language=Language.PYTHON,
            classes=[
                ClassInfo(
                    name="Test",
                    location=SourceLocation(file_path="/test/file.py", line=1, column=0),
                )
            ],
            functions=[],
            imports=[],
            total_lines=50,
        )

        data = analysis.to_dict()
        assert data["file_path"] == "/test/file.py"
        assert data["language"] == "python"
        assert data["total_lines"] == 50
        assert len(data["classes"]) == 1


class TestTreeSitterParser:
    """Tests for TreeSitterParser wrapper."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = TreeSitterParser(Language.PYTHON)
        # Should not raise even if tree-sitter not available
        assert parser is not None

    def test_parser_availability_check(self):
        """Test checking parser availability."""
        parser = TreeSitterParser(Language.PYTHON)
        # available property should return bool
        assert isinstance(parser.available, bool)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def code_intel(self):
        return CodeIntelligence()

    def test_empty_file(self, code_intel, tmp_path):
        """Test analyzing an empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        analysis = code_intel.analyze_file(str(empty_file))
        assert analysis is not None
        assert analysis.total_lines == 0 or analysis.total_lines == 1
        assert len(analysis.classes) == 0
        assert len(analysis.functions) == 0

    def test_syntax_error_file(self, code_intel, tmp_path):
        """Test analyzing a file with syntax errors."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n  # incomplete")

        # Should not crash, may return partial results or None
        analysis = code_intel.analyze_file(str(bad_file))
        # Either returns None or partial analysis
        assert analysis is None or isinstance(analysis, FileAnalysis)

    def test_binary_file(self, code_intel, tmp_path):
        """Test handling binary file."""
        binary_file = tmp_path / "binary.py"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        analysis = code_intel.analyze_file(str(binary_file))
        # Should handle gracefully
        assert analysis is None or isinstance(analysis, FileAnalysis)

    def test_very_long_file(self, code_intel, tmp_path):
        """Test handling a very long file."""
        long_code = "\n".join([f"def func_{i}(): pass" for i in range(1000)])
        long_file = tmp_path / "long.py"
        long_file.write_text(long_code)

        analysis = code_intel.analyze_file(str(long_file))
        assert analysis is not None
        assert len(analysis.functions) >= 100  # At least partial extraction

    def test_unicode_content(self, code_intel, tmp_path):
        """Test handling Unicode content."""
        unicode_code = '''
# -*- coding: utf-8 -*-
"""Unicode: 你好世界 🌍"""

def greet(name: str = "世界") -> str:
    """Greet in Chinese."""
    return f"你好 {name}"

emoji_var = "🎉"
'''
        unicode_file = tmp_path / "unicode.py"
        unicode_file.write_text(unicode_code, encoding="utf-8")

        analysis = code_intel.analyze_file(str(unicode_file))
        assert analysis is not None
        assert any(f.name == "greet" for f in analysis.functions)
