"""
Shared type definitions for code intelligence.

This module contains all the data classes and enums used throughout
the code intelligence package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "Language":
        """Detect language from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "py": cls.PYTHON,
            "pyi": cls.PYTHON,
            "js": cls.JAVASCRIPT,
            "jsx": cls.JAVASCRIPT,
            "mjs": cls.JAVASCRIPT,
            "ts": cls.TYPESCRIPT,
            "tsx": cls.TYPESCRIPT,
            "go": cls.GO,
            "rs": cls.RUST,
            "java": cls.JAVA,
        }
        return mapping.get(ext, cls.UNKNOWN)


class SymbolKind(str, Enum):
    """Kind of code symbol."""

    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    TYPE_ALIAS = "type_alias"


class Visibility(str, Enum):
    """Symbol visibility/access modifier."""

    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


@dataclass
class SourceLocation:
    """Location in source code."""

    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}:{self.start_column}"


@dataclass
class Parameter:
    """Function/method parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False  # *args or **kwargs
    is_keyword_only: bool = False


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    kind: SymbolKind  # FUNCTION or METHOD
    location: SourceLocation
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC
    is_async: bool = False
    is_generator: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    complexity: int = 1  # Cyclomatic complexity
    lines_of_code: int = 0
    calls: list[str] = field(default_factory=list)  # Functions this calls
    parent_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "location": str(self.location),
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_annotation,
                    "default": p.default_value,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "visibility": self.visibility.value,
            "is_async": self.is_async,
            "is_static": self.is_static,
            "complexity": self.complexity,
            "lines_of_code": self.lines_of_code,
            "calls": self.calls,
            "parent_class": self.parent_class,
        }


@dataclass
class ClassInfo:
    """Information about a class, struct, or interface."""

    name: str
    kind: SymbolKind  # CLASS, STRUCT, INTERFACE, ENUM
    location: SourceLocation
    bases: list[str] = field(default_factory=list)  # Parent classes/interfaces
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC
    methods: list[FunctionInfo] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    class_variables: list[str] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False
    generic_params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "location": str(self.location),
            "bases": self.bases,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "visibility": self.visibility.value,
            "methods": [m.to_dict() for m in self.methods],
            "properties": self.properties,
            "class_variables": self.class_variables,
            "is_abstract": self.is_abstract,
            "is_dataclass": self.is_dataclass,
        }


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: list[str] = field(default_factory=list)  # Specific imports
    alias: str | None = None
    is_relative: bool = False
    location: SourceLocation | None = None


@dataclass
class FileAnalysis:
    """Complete analysis of a source file."""

    file_path: str
    language: Language
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    module_docstring: str | None = None
    global_variables: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)  # For __all__, export statements
    lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_functions(self) -> int:
        """Total functions including methods."""
        return len(self.functions) + sum(len(c.methods) for c in self.classes)

    @property
    def total_complexity(self) -> int:
        """Sum of all function complexities."""
        total = sum(f.complexity for f in self.functions)
        for cls in self.classes:
            total += sum(m.complexity for m in cls.methods)
        return total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language.value,
            "imports": [
                {
                    "module": i.module,
                    "names": i.names,
                    "alias": i.alias,
                    "is_relative": i.is_relative,
                }
                for i in self.imports
            ],
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "module_docstring": self.module_docstring,
            "global_variables": self.global_variables,
            "exports": self.exports,
            "lines_of_code": self.lines_of_code,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "total_functions": self.total_functions,
            "total_complexity": self.total_complexity,
            "errors": self.errors,
        }


__all__ = [
    "Language",
    "SymbolKind",
    "Visibility",
    "SourceLocation",
    "Parameter",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    "FileAnalysis",
]
