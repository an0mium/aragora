"""
Code Intelligence Module using Tree-sitter AST parsing.

Provides semantic code analysis with accurate symbol extraction, type information,
and structural understanding across multiple programming languages.

Supported Languages:
- Python
- JavaScript/TypeScript
- Go
- Rust
- Java

Example:
    >>> from aragora.analysis.code_intelligence import CodeIntelligence
    >>> intel = CodeIntelligence()
    >>> analysis = intel.analyze_file("src/main.py")
    >>> print(f"Found {len(analysis.classes)} classes, {len(analysis.functions)} functions")
    >>> for cls in analysis.classes:
    ...     print(f"  - {cls.name} with {len(cls.methods)} methods")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


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


# =============================================================================
# Tree-sitter Parser Wrapper
# =============================================================================


class TreeSitterParser:
    """Wrapper for tree-sitter parsing with graceful fallback."""

    def __init__(self) -> None:
        self._parsers: dict[Language, Any] = {}
        self._available = False
        self._init_parsers()

    def _init_parsers(self) -> None:
        """Initialize tree-sitter parsers for each language."""
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            import tree_sitter_go
            import tree_sitter_rust
            import tree_sitter_java
            from tree_sitter import Parser, Language as TSLanguage

            self._available = True

            # Initialize parsers
            for lang, module in [
                (Language.PYTHON, tree_sitter_python),
                (Language.JAVASCRIPT, tree_sitter_javascript),
                (Language.GO, tree_sitter_go),
                (Language.RUST, tree_sitter_rust),
                (Language.JAVA, tree_sitter_java),
            ]:
                try:
                    parser = Parser()
                    ts_lang = TSLanguage(module.language())
                    parser.language = ts_lang
                    self._parsers[lang] = parser
                except Exception as e:
                    logger.warning(f"Failed to init tree-sitter for {lang.value}: {e}")

            # TypeScript shares JavaScript parser with different queries
            if Language.JAVASCRIPT in self._parsers:
                try:
                    import tree_sitter_typescript

                    parser = Parser()
                    ts_lang = TSLanguage(tree_sitter_typescript.language_typescript())
                    parser.language = ts_lang
                    self._parsers[Language.TYPESCRIPT] = parser
                except Exception as e:
                    logger.warning(f"Failed to init tree-sitter for TypeScript: {e}")

            logger.info(f"Tree-sitter initialized for: {list(self._parsers.keys())}")

        except ImportError as e:
            logger.info(f"Tree-sitter not available, using regex fallback: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        """Check if tree-sitter is available."""
        return self._available

    def parse(self, source: bytes, language: Language) -> Any | None:
        """Parse source code and return AST."""
        if not self._available or language not in self._parsers:
            return None

        parser = self._parsers[language]
        try:
            return parser.parse(source)
        except Exception as e:
            logger.error(f"Tree-sitter parse error: {e}")
            return None

    def supports(self, language: Language) -> bool:
        """Check if language is supported."""
        return language in self._parsers


# =============================================================================
# Code Intelligence Engine
# =============================================================================


class CodeIntelligence:
    """
    Semantic code analysis engine.

    Uses tree-sitter for accurate AST parsing when available,
    with regex fallback for basic symbol extraction.

    Example:
        intel = CodeIntelligence()

        # Analyze a single file
        analysis = intel.analyze_file("src/main.py")
        print(f"Classes: {[c.name for c in analysis.classes]}")

        # Analyze a directory
        results = intel.analyze_directory("src/")
        for path, analysis in results.items():
            print(f"{path}: {len(analysis.functions)} functions")

        # Find all usages of a symbol
        usages = intel.find_symbol_usages("src/", "MyClass")
    """

    def __init__(self) -> None:
        self._parser = TreeSitterParser()

    @property
    def tree_sitter_available(self) -> bool:
        """Check if tree-sitter is available for enhanced parsing."""
        return self._parser.available

    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a single source file.

        Args:
            file_path: Path to the source file

        Returns:
            FileAnalysis with extracted information
        """
        path = Path(file_path)
        language = Language.from_extension(path.suffix)

        analysis = FileAnalysis(
            file_path=str(path),
            language=language,
        )

        if not path.exists():
            analysis.errors.append(f"File not found: {file_path}")
            return analysis

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            analysis.errors.append(f"Failed to read file: {e}")
            return analysis

        # Count lines
        lines = content.split("\n")
        analysis.lines_of_code = len(lines)
        analysis.blank_lines = sum(1 for line in lines if not line.strip())
        analysis.comment_lines = self._count_comment_lines(content, language)

        # Try tree-sitter parsing first
        if self._parser.supports(language):
            tree = self._parser.parse(content.encode("utf-8"), language)
            if tree:
                self._analyze_with_tree_sitter(tree, content, analysis, language)
                return analysis

        # Fallback to regex-based parsing
        self._analyze_with_regex(content, analysis, language)
        return analysis

    def analyze_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, FileAnalysis]:
        """
        Analyze all source files in a directory.

        Args:
            directory: Root directory to analyze
            extensions: File extensions to include (default: all supported)
            exclude_patterns: Glob patterns to exclude

        Returns:
            Dictionary mapping file paths to their analysis
        """
        results = {}
        root = Path(directory)
        exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/vendor/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
        ]

        # Default extensions
        if extensions is None:
            extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"]

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix not in extensions:
                continue

            # Check exclusions
            path_str = str(path)
            excluded = False
            for pattern in exclude_patterns:
                if path.match(pattern):
                    excluded = True
                    break
            if excluded:
                continue

            try:
                results[path_str] = self.analyze_file(path_str)
            except Exception as e:
                logger.warning(f"Failed to analyze {path}: {e}")

        return results

    def find_symbol_usages(
        self,
        directory: str,
        symbol_name: str,
        language: Language | None = None,
    ) -> list[SourceLocation]:
        """
        Find all usages of a symbol across the codebase.

        Args:
            directory: Directory to search
            symbol_name: Name of the symbol to find
            language: Optional language filter

        Returns:
            List of source locations where the symbol is used
        """
        usages = []
        root = Path(directory)

        # Determine extensions to search
        extensions = None
        if language:
            ext_map = {
                Language.PYTHON: [".py"],
                Language.JAVASCRIPT: [".js", ".jsx", ".mjs"],
                Language.TYPESCRIPT: [".ts", ".tsx"],
                Language.GO: [".go"],
                Language.RUST: [".rs"],
                Language.JAVA: [".java"],
            }
            extensions = ext_map.get(language, [])

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if extensions and path.suffix not in extensions:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                # Simple pattern matching for symbol usages
                pattern = rf"\b{re.escape(symbol_name)}\b"
                for i, line in enumerate(content.split("\n"), 1):
                    for match in re.finditer(pattern, line):
                        usages.append(
                            SourceLocation(
                                file_path=str(path),
                                start_line=i,
                                start_column=match.start(),
                                end_line=i,
                                end_column=match.end(),
                            )
                        )
            except Exception as e:
                logger.debug(f"Error searching {path}: {e}")

        return usages

    def get_symbol_at_location(
        self,
        file_path: str,
        line: int,
        column: int,
    ) -> FunctionInfo | ClassInfo | None:
        """
        Get the symbol at a specific location.

        Args:
            file_path: Path to the file
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            The symbol at that location, or None
        """
        analysis = self.analyze_file(file_path)

        # Check functions
        for func in analysis.functions:
            loc = func.location
            if (
                loc.start_line <= line <= loc.end_line
                and (line != loc.start_line or column >= loc.start_column)
                and (line != loc.end_line or column <= loc.end_column)
            ):
                return func

        # Check classes and their methods
        for cls in analysis.classes:
            loc = cls.location
            if (
                loc.start_line <= line <= loc.end_line
                and (line != loc.start_line or column >= loc.start_column)
                and (line != loc.end_line or column <= loc.end_column)
            ):
                # Check if within a method
                for method in cls.methods:
                    mloc = method.location
                    if (
                        mloc.start_line <= line <= mloc.end_line
                        and (line != mloc.start_line or column >= mloc.start_column)
                        and (line != mloc.end_line or column <= mloc.end_column)
                    ):
                        return method
                return cls

        return None

    # =========================================================================
    # Tree-sitter Analysis
    # =========================================================================

    def _analyze_with_tree_sitter(
        self,
        tree: Any,
        source: str,
        analysis: FileAnalysis,
        language: Language,
    ) -> None:
        """Analyze using tree-sitter AST."""
        root = tree.root_node
        source_bytes = source.encode("utf-8")

        if language == Language.PYTHON:
            self._analyze_python_ast(root, source_bytes, analysis)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            self._analyze_js_ts_ast(root, source_bytes, analysis, language)
        elif language == Language.GO:
            self._analyze_go_ast(root, source_bytes, analysis)
        elif language == Language.RUST:
            self._analyze_rust_ast(root, source_bytes, analysis)
        elif language == Language.JAVA:
            self._analyze_java_ast(root, source_bytes, analysis)

    def _analyze_python_ast(
        self,
        root: Any,
        source: bytes,
        analysis: FileAnalysis,
    ) -> None:
        """Analyze Python AST."""
        file_path = analysis.file_path

        def get_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode("utf-8")

        def get_docstring(body_node: Any) -> str | None:
            if body_node.child_count > 0:
                first = body_node.children[0]
                if first.type == "expression_statement":
                    expr = first.children[0] if first.child_count > 0 else None
                    if expr and expr.type == "string":
                        text = get_text(expr)
                        # Remove quotes
                        if text.startswith('"""') or text.startswith("'''"):
                            return text[3:-3].strip()
                        elif text.startswith('"') or text.startswith("'"):
                            return text[1:-1].strip()
            return None

        def make_location(node: Any) -> SourceLocation:
            return SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                start_column=node.start_point[1],
                end_line=node.end_point[0] + 1,
                end_column=node.end_point[1],
            )

        def analyze_function(node: Any, parent_class: str | None = None) -> FunctionInfo | None:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            name = get_text(name_node)
            params_node = node.child_by_field_name("parameters")
            body_node = node.child_by_field_name("body")

            # Parse parameters
            parameters = []
            if params_node:
                for param in params_node.children:
                    if param.type in ("identifier", "typed_parameter", "default_parameter"):
                        if param.type == "identifier":
                            parameters.append(Parameter(name=get_text(param)))
                        elif param.type == "typed_parameter":
                            pname = param.child_by_field_name("name")
                            ptype = param.child_by_field_name("type")
                            parameters.append(
                                Parameter(
                                    name=get_text(pname) if pname else "",
                                    type_annotation=get_text(ptype) if ptype else None,
                                )
                            )
                        elif param.type == "default_parameter":
                            pname = param.child_by_field_name("name")
                            pvalue = param.child_by_field_name("value")
                            parameters.append(
                                Parameter(
                                    name=get_text(pname) if pname else "",
                                    default_value=get_text(pvalue) if pvalue else None,
                                )
                            )
                    elif param.type == "list_splat_pattern":
                        inner = param.children[0] if param.child_count > 0 else None
                        parameters.append(
                            Parameter(
                                name=get_text(inner) if inner else "*args",
                                is_variadic=True,
                            )
                        )
                    elif param.type == "dictionary_splat_pattern":
                        inner = param.children[0] if param.child_count > 0 else None
                        parameters.append(
                            Parameter(
                                name=get_text(inner) if inner else "**kwargs",
                                is_variadic=True,
                            )
                        )

            # Get return type
            return_type = None
            return_node = node.child_by_field_name("return_type")
            if return_node:
                return_type = get_text(return_node)

            # Get docstring
            docstring = get_docstring(body_node) if body_node else None

            # Check decorators
            decorators = []
            is_async = False
            is_static = False
            is_classmethod = False

            prev = node.prev_sibling
            while prev and prev.type == "decorator":
                dec_text = get_text(prev)
                decorators.append(dec_text)
                if "@staticmethod" in dec_text:
                    is_static = True
                if "@classmethod" in dec_text:
                    is_classmethod = True
                prev = prev.prev_sibling

            # Check if async
            for child in node.children:
                if get_text(child) == "async":
                    is_async = True
                    break

            # Calculate complexity (simple: count control flow statements)
            complexity = 1
            if body_node:
                body_text = get_text(body_node)
                complexity += body_text.count("if ")
                complexity += body_text.count("elif ")
                complexity += body_text.count("for ")
                complexity += body_text.count("while ")
                complexity += body_text.count("except ")
                complexity += body_text.count(" and ")
                complexity += body_text.count(" or ")

            # Count lines
            loc = make_location(node)
            lines_of_code = loc.end_line - loc.start_line + 1

            return FunctionInfo(
                name=name,
                kind=SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION,
                location=loc,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                decorators=decorators,
                is_async=is_async,
                is_static=is_static,
                is_classmethod=is_classmethod,
                complexity=complexity,
                lines_of_code=lines_of_code,
                parent_class=parent_class,
            )

        def analyze_class(node: Any) -> ClassInfo | None:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            name = get_text(name_node)
            body_node = node.child_by_field_name("body")

            # Get base classes
            bases = []
            superclass_node = node.child_by_field_name("superclasses")
            if superclass_node:
                for child in superclass_node.children:
                    if child.type in ("identifier", "attribute"):
                        bases.append(get_text(child))

            # Get docstring
            docstring = get_docstring(body_node) if body_node else None

            # Get decorators
            decorators = []
            is_dataclass = False
            prev = node.prev_sibling
            while prev and prev.type == "decorator":
                dec_text = get_text(prev)
                decorators.append(dec_text)
                if "@dataclass" in dec_text:
                    is_dataclass = True
                prev = prev.prev_sibling

            # Analyze methods and properties
            methods = []
            properties = []
            class_variables = []

            if body_node:
                for child in body_node.children:
                    if child.type == "function_definition":
                        method = analyze_function(child, parent_class=name)
                        if method:
                            methods.append(method)
                    elif child.type == "expression_statement":
                        # Class variable or property
                        expr = child.children[0] if child.child_count > 0 else None
                        if expr and expr.type == "assignment":
                            left = expr.child_by_field_name("left")
                            if left:
                                class_variables.append(get_text(left))

            return ClassInfo(
                name=name,
                kind=SymbolKind.CLASS,
                location=make_location(node),
                bases=bases,
                docstring=docstring,
                decorators=decorators,
                methods=methods,
                properties=properties,
                class_variables=class_variables,
                is_dataclass=is_dataclass,
            )

        # Walk the AST
        for child in root.children:
            if child.type == "import_statement":
                # import x, y, z
                for name_child in child.children:
                    if name_child.type == "dotted_name":
                        analysis.imports.append(
                            ImportInfo(
                                module=get_text(name_child),
                                location=make_location(child),
                            )
                        )
            elif child.type == "import_from_statement":
                # from x import y
                module_node = child.child_by_field_name("module_name")
                module = get_text(module_node) if module_node else ""

                names = []
                for name_child in child.children:
                    if name_child.type == "dotted_name":
                        if name_child != module_node:
                            names.append(get_text(name_child))
                    elif name_child.type == "aliased_import":
                        name_part = name_child.child_by_field_name("name")
                        if name_part:
                            names.append(get_text(name_part))

                analysis.imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        is_relative=module.startswith("."),
                        location=make_location(child),
                    )
                )
            elif child.type == "function_definition":
                func = analyze_function(child)
                if func:
                    analysis.functions.append(func)
            elif child.type == "class_definition":
                cls = analyze_class(child)
                if cls:
                    analysis.classes.append(cls)
            elif child.type == "expression_statement":
                # Check for module docstring (first string in module)
                if not analysis.module_docstring:
                    expr = child.children[0] if child.child_count > 0 else None
                    if expr and expr.type == "string":
                        text = get_text(expr)
                        if text.startswith('"""') or text.startswith("'''"):
                            analysis.module_docstring = text[3:-3].strip()

                # Check for __all__
                expr = child.children[0] if child.child_count > 0 else None
                if expr and expr.type == "assignment":
                    left = expr.child_by_field_name("left")
                    if left and get_text(left) == "__all__":
                        right = expr.child_by_field_name("right")
                        if right and right.type == "list":
                            for item in right.children:
                                if item.type == "string":
                                    analysis.exports.append(get_text(item).strip("\"'"))

    def _analyze_js_ts_ast(
        self,
        root: Any,
        source: bytes,
        analysis: FileAnalysis,
        language: Language,
    ) -> None:
        """Analyze JavaScript/TypeScript AST."""
        file_path = analysis.file_path

        def get_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode("utf-8")

        def make_location(node: Any) -> SourceLocation:
            return SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                start_column=node.start_point[1],
                end_line=node.end_point[0] + 1,
                end_column=node.end_point[1],
            )

        def walk(node: Any) -> None:
            node_type = node.type

            if node_type == "import_statement":
                # import x from 'y'
                source_node = node.child_by_field_name("source")
                module = get_text(source_node).strip("\"'") if source_node else ""
                analysis.imports.append(ImportInfo(module=module, location=make_location(node)))

            elif node_type in ("function_declaration", "arrow_function", "method_definition"):
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else "<anonymous>"

                # Check if async
                is_async = False
                for child in node.children:
                    if get_text(child) == "async":
                        is_async = True
                        break

                loc = make_location(node)
                func = FunctionInfo(
                    name=name,
                    kind=SymbolKind.FUNCTION,
                    location=loc,
                    is_async=is_async,
                    lines_of_code=loc.end_line - loc.start_line + 1,
                )
                analysis.functions.append(func)

            elif node_type == "class_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else "<anonymous>"

                # Get extends
                bases = []
                heritage = node.child_by_field_name("extends")
                if heritage:
                    bases.append(get_text(heritage))

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=make_location(node),
                        bases=bases,
                    )
                )

            elif node_type == "export_statement":
                # Track exports
                for child in node.children:
                    if child.type == "identifier":
                        analysis.exports.append(get_text(child))

            # Recurse
            for child in node.children:
                walk(child)

        walk(root)

    def _analyze_go_ast(
        self,
        root: Any,
        source: bytes,
        analysis: FileAnalysis,
    ) -> None:
        """Analyze Go AST."""
        file_path = analysis.file_path

        def get_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode("utf-8")

        def make_location(node: Any) -> SourceLocation:
            return SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                start_column=node.start_point[1],
                end_line=node.end_point[0] + 1,
                end_column=node.end_point[1],
            )

        def walk(node: Any) -> None:
            node_type = node.type

            if node_type == "import_declaration":
                for child in node.children:
                    if child.type == "import_spec" or child.type == "interpreted_string_literal":
                        module = get_text(child).strip('"')
                        analysis.imports.append(
                            ImportInfo(module=module, location=make_location(node))
                        )
                    elif child.type == "import_spec_list":
                        for spec in child.children:
                            if spec.type == "import_spec":
                                path = spec.child_by_field_name("path")
                                if path:
                                    analysis.imports.append(
                                        ImportInfo(
                                            module=get_text(path).strip('"'),
                                            location=make_location(node),
                                        )
                                    )

            elif node_type == "function_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                loc = make_location(node)
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        lines_of_code=loc.end_line - loc.start_line + 1,
                    )
                )

            elif node_type == "method_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                # Get receiver (the struct this method belongs to)
                receiver = node.child_by_field_name("receiver")
                parent_class = None
                if receiver:
                    for child in receiver.children:
                        if child.type == "type_identifier":
                            parent_class = get_text(child)
                            break

                loc = make_location(node)
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.METHOD,
                        location=loc,
                        lines_of_code=loc.end_line - loc.start_line + 1,
                        parent_class=parent_class,
                    )
                )

            elif node_type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        name = get_text(name_node) if name_node else ""

                        kind = SymbolKind.CLASS
                        if type_node and type_node.type == "struct_type":
                            kind = SymbolKind.STRUCT
                        elif type_node and type_node.type == "interface_type":
                            kind = SymbolKind.INTERFACE

                        analysis.classes.append(
                            ClassInfo(
                                name=name,
                                kind=kind,
                                location=make_location(child),
                            )
                        )

            for child in node.children:
                walk(child)

        walk(root)

    def _analyze_rust_ast(
        self,
        root: Any,
        source: bytes,
        analysis: FileAnalysis,
    ) -> None:
        """Analyze Rust AST."""
        file_path = analysis.file_path

        def get_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode("utf-8")

        def make_location(node: Any) -> SourceLocation:
            return SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                start_column=node.start_point[1],
                end_line=node.end_point[0] + 1,
                end_column=node.end_point[1],
            )

        def walk(node: Any) -> None:
            node_type = node.type

            if node_type == "use_declaration":
                # use x::y::z
                path_text = ""
                for child in node.children:
                    if child.type == "scoped_identifier" or child.type == "identifier":
                        path_text = get_text(child)
                        break
                if path_text:
                    analysis.imports.append(
                        ImportInfo(module=path_text, location=make_location(node))
                    )

            elif node_type == "function_item":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                # Check visibility
                visibility = Visibility.PRIVATE
                for child in node.children:
                    if child.type == "visibility_modifier":
                        if "pub" in get_text(child):
                            visibility = Visibility.PUBLIC
                        break

                # Check if async
                is_async = False
                for child in node.children:
                    if get_text(child) == "async":
                        is_async = True
                        break

                loc = make_location(node)
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        visibility=visibility,
                        is_async=is_async,
                        lines_of_code=loc.end_line - loc.start_line + 1,
                    )
                )

            elif node_type == "struct_item":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.STRUCT,
                        location=make_location(node),
                    )
                )

            elif node_type == "enum_item":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.ENUM,
                        location=make_location(node),
                    )
                )

            elif node_type == "trait_item":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=make_location(node),
                    )
                )

            for child in node.children:
                walk(child)

        walk(root)

    def _analyze_java_ast(
        self,
        root: Any,
        source: bytes,
        analysis: FileAnalysis,
    ) -> None:
        """Analyze Java AST."""
        file_path = analysis.file_path

        def get_text(node: Any) -> str:
            return source[node.start_byte : node.end_byte].decode("utf-8")

        def make_location(node: Any) -> SourceLocation:
            return SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                start_column=node.start_point[1],
                end_line=node.end_point[0] + 1,
                end_column=node.end_point[1],
            )

        def walk(node: Any) -> None:
            node_type = node.type

            if node_type == "import_declaration":
                for child in node.children:
                    if child.type == "scoped_identifier":
                        analysis.imports.append(
                            ImportInfo(module=get_text(child), location=make_location(node))
                        )

            elif node_type == "method_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                # Get visibility
                visibility = Visibility.PUBLIC  # Default in Java
                for child in node.children:
                    if child.type == "modifiers":
                        mod_text = get_text(child)
                        if "private" in mod_text:
                            visibility = Visibility.PRIVATE
                        elif "protected" in mod_text:
                            visibility = Visibility.PROTECTED

                loc = make_location(node)
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.METHOD,
                        location=loc,
                        visibility=visibility,
                        lines_of_code=loc.end_line - loc.start_line + 1,
                    )
                )

            elif node_type == "class_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                # Get extends and implements
                bases = []
                superclass = node.child_by_field_name("superclass")
                if superclass:
                    bases.append(get_text(superclass))

                interfaces = node.child_by_field_name("interfaces")
                if interfaces:
                    for child in interfaces.children:
                        if child.type == "type_identifier":
                            bases.append(get_text(child))

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=make_location(node),
                        bases=bases,
                    )
                )

            elif node_type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=make_location(node),
                    )
                )

            elif node_type == "enum_declaration":
                name_node = node.child_by_field_name("name")
                name = get_text(name_node) if name_node else ""

                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.ENUM,
                        location=make_location(node),
                    )
                )

            for child in node.children:
                walk(child)

        walk(root)

    # =========================================================================
    # Regex Fallback Analysis
    # =========================================================================

    def _analyze_with_regex(
        self,
        source: str,
        analysis: FileAnalysis,
        language: Language,
    ) -> None:
        """Fallback analysis using regex patterns."""
        file_path = analysis.file_path
        lines = source.split("\n")

        if language == Language.PYTHON:
            self._regex_analyze_python(lines, file_path, analysis)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            self._regex_analyze_js_ts(lines, file_path, analysis)
        elif language == Language.GO:
            self._regex_analyze_go(lines, file_path, analysis)
        elif language == Language.RUST:
            self._regex_analyze_rust(lines, file_path, analysis)
        elif language == Language.JAVA:
            self._regex_analyze_java(lines, file_path, analysis)

    def _regex_analyze_python(
        self,
        lines: list[str],
        file_path: str,
        analysis: FileAnalysis,
    ) -> None:
        """Regex-based Python analysis."""
        # Pattern for imports
        import_pattern = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+?)(?:\s+as\s+\S+)?$")
        # Pattern for functions
        func_pattern = re.compile(
            r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?:"
        )
        # Pattern for classes
        class_pattern = re.compile(r"^(\s*)class\s+(\w+)(?:\(([^)]*)\))?:")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check imports
            match = import_pattern.match(stripped)
            if match:
                from_module = match.group(1) or ""
                imports = match.group(2)
                if from_module:
                    names = [n.strip().split(" as ")[0] for n in imports.split(",")]
                    analysis.imports.append(ImportInfo(module=from_module, names=names))
                else:
                    modules = [n.strip().split(" as ")[0] for n in imports.split(",")]
                    for mod in modules:
                        analysis.imports.append(ImportInfo(module=mod))

            # Check functions (only top-level)
            match = func_pattern.match(line)
            if match:
                indent = len(match.group(1))
                if indent == 0:  # Top-level function
                    name = match.group(2)
                    is_async = "async def" in line
                    loc = SourceLocation(
                        file_path=file_path,
                        start_line=i,
                        start_column=0,
                        end_line=i,
                        end_column=len(line),
                    )
                    analysis.functions.append(
                        FunctionInfo(
                            name=name,
                            kind=SymbolKind.FUNCTION,
                            location=loc,
                            is_async=is_async,
                        )
                    )

            # Check classes
            match = class_pattern.match(line)
            if match:
                indent = len(match.group(1))
                if indent == 0:  # Top-level class
                    name = match.group(2)
                    bases_str = match.group(3) or ""
                    bases = [b.strip() for b in bases_str.split(",") if b.strip()]
                    loc = SourceLocation(
                        file_path=file_path,
                        start_line=i,
                        start_column=0,
                        end_line=i,
                        end_column=len(line),
                    )
                    analysis.classes.append(
                        ClassInfo(
                            name=name,
                            kind=SymbolKind.CLASS,
                            location=loc,
                            bases=bases,
                        )
                    )

    def _regex_analyze_js_ts(
        self,
        lines: list[str],
        file_path: str,
        analysis: FileAnalysis,
    ) -> None:
        """Regex-based JavaScript/TypeScript analysis."""
        import_pattern = re.compile(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]")
        func_pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|"
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"
        )
        class_pattern = re.compile(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?")

        for i, line in enumerate(lines, 1):
            # Imports
            match = import_pattern.search(line)
            if match:
                analysis.imports.append(ImportInfo(module=match.group(1)))

            # Functions
            match = func_pattern.search(line)
            if match:
                name = match.group(1) or match.group(2)
                if name:
                    loc = SourceLocation(
                        file_path=file_path,
                        start_line=i,
                        start_column=0,
                        end_line=i,
                        end_column=len(line),
                    )
                    analysis.functions.append(
                        FunctionInfo(
                            name=name,
                            kind=SymbolKind.FUNCTION,
                            location=loc,
                            is_async="async" in line,
                        )
                    )

            # Classes
            match = class_pattern.search(line)
            if match:
                name = match.group(1)
                bases = [match.group(2)] if match.group(2) else []
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        bases=bases,
                    )
                )

    def _regex_analyze_go(
        self,
        lines: list[str],
        file_path: str,
        analysis: FileAnalysis,
    ) -> None:
        """Regex-based Go analysis."""
        import_pattern = re.compile(r'^\s*"([^"]+)"')
        func_pattern = re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(")
        type_pattern = re.compile(r"^type\s+(\w+)\s+(struct|interface)")

        in_import = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("import ("):
                in_import = True
                continue
            elif stripped == ")":
                in_import = False
                continue

            if in_import or stripped.startswith("import "):
                match = import_pattern.search(line)
                if match:
                    analysis.imports.append(ImportInfo(module=match.group(1)))

            match = func_pattern.match(stripped)
            if match:
                name = match.group(1)
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.functions.append(
                    FunctionInfo(name=name, kind=SymbolKind.FUNCTION, location=loc)
                )

            match = type_pattern.match(stripped)
            if match:
                name = match.group(1)
                kind = SymbolKind.STRUCT if match.group(2) == "struct" else SymbolKind.INTERFACE
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(ClassInfo(name=name, kind=kind, location=loc))

    def _regex_analyze_rust(
        self,
        lines: list[str],
        file_path: str,
        analysis: FileAnalysis,
    ) -> None:
        """Regex-based Rust analysis."""
        use_pattern = re.compile(r"^use\s+(.+);")
        fn_pattern = re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")
        struct_pattern = re.compile(r"^(?:pub\s+)?struct\s+(\w+)")
        enum_pattern = re.compile(r"^(?:pub\s+)?enum\s+(\w+)")
        trait_pattern = re.compile(r"^(?:pub\s+)?trait\s+(\w+)")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            match = use_pattern.match(stripped)
            if match:
                analysis.imports.append(ImportInfo(module=match.group(1)))

            match = fn_pattern.match(stripped)
            if match:
                name = match.group(1)
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        visibility=Visibility.PUBLIC if "pub " in line else Visibility.PRIVATE,
                        is_async="async fn" in line,
                    )
                )

            match = struct_pattern.match(stripped)
            if match:
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(name=match.group(1), kind=SymbolKind.STRUCT, location=loc)
                )

            match = enum_pattern.match(stripped)
            if match:
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(name=match.group(1), kind=SymbolKind.ENUM, location=loc)
                )

            match = trait_pattern.match(stripped)
            if match:
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(name=match.group(1), kind=SymbolKind.INTERFACE, location=loc)
                )

    def _regex_analyze_java(
        self,
        lines: list[str],
        file_path: str,
        analysis: FileAnalysis,
    ) -> None:
        """Regex-based Java analysis."""
        import_pattern = re.compile(r"^import\s+(.+);")
        class_pattern = re.compile(r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?")
        interface_pattern = re.compile(r"(?:public\s+)?interface\s+(\w+)")
        method_pattern = re.compile(
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+)?\s*\{"
        )

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            match = import_pattern.match(stripped)
            if match:
                analysis.imports.append(ImportInfo(module=match.group(1)))

            match = class_pattern.search(stripped)
            if match:
                name = match.group(1)
                bases = [match.group(2)] if match.group(2) else []
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(name=name, kind=SymbolKind.CLASS, location=loc, bases=bases)
                )

            match = interface_pattern.search(stripped)
            if match:
                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.classes.append(
                    ClassInfo(name=match.group(1), kind=SymbolKind.INTERFACE, location=loc)
                )

            match = method_pattern.search(stripped)
            if match and match.group(1) not in ("if", "while", "for", "switch", "catch"):
                name = match.group(1)
                visibility = Visibility.PUBLIC
                if "private" in line:
                    visibility = Visibility.PRIVATE
                elif "protected" in line:
                    visibility = Visibility.PROTECTED

                loc = SourceLocation(
                    file_path=file_path,
                    start_line=i,
                    start_column=0,
                    end_line=i,
                    end_column=len(line),
                )
                analysis.functions.append(
                    FunctionInfo(
                        name=name,
                        kind=SymbolKind.METHOD,
                        location=loc,
                        visibility=visibility,
                        is_static="static " in line,
                    )
                )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _count_comment_lines(self, source: str, language: Language) -> int:
        """Count comment lines in source code."""
        count = 0
        lines = source.split("\n")

        in_block_comment = False
        block_start = "/*"
        block_end = "*/"
        line_comment = "//"

        if language == Language.PYTHON:
            block_start = '"""'
            block_end = '"""'
            line_comment = "#"

        for line in lines:
            stripped = line.strip()

            if in_block_comment:
                count += 1
                if block_end in stripped:
                    in_block_comment = False
            elif stripped.startswith(block_start):
                count += 1
                if block_end not in stripped[len(block_start) :]:
                    in_block_comment = True
            elif stripped.startswith(line_comment):
                count += 1

        return count


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "Language",
    "SymbolKind",
    "Visibility",
    # Data Classes
    "SourceLocation",
    "Parameter",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    "FileAnalysis",
    # Classes
    "CodeIntelligence",
    "TreeSitterParser",
]
