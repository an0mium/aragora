"""
AST Analysis Module using Tree-sitter parsing.

Provides AST-based code analysis with accurate symbol extraction
across multiple programming languages.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    ClassInfo,
    FileAnalysis,
    FunctionInfo,
    ImportInfo,
    Language,
    Parameter,
    SourceLocation,
    SymbolKind,
    Visibility,
)

logger = logging.getLogger(__name__)


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
                except (ImportError, RuntimeError, OSError) as e:
                    logger.warning(f"Failed to init tree-sitter for {lang.value}: {e}")

            # TypeScript shares JavaScript parser with different queries
            if Language.JAVASCRIPT in self._parsers:
                try:
                    import tree_sitter_typescript

                    parser = Parser()
                    ts_lang = TSLanguage(tree_sitter_typescript.language_typescript())
                    parser.language = ts_lang
                    self._parsers[Language.TYPESCRIPT] = parser
                except (ImportError, RuntimeError, OSError) as e:
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
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Tree-sitter parse error: {e}")
            return None

    def supports(self, language: Language) -> bool:
        """Check if language is supported."""
        return language in self._parsers


def analyze_with_tree_sitter(
    tree: Any,
    source: str,
    analysis: FileAnalysis,
    language: Language,
) -> None:
    """Analyze using tree-sitter AST."""
    root = tree.root_node
    source_bytes = source.encode("utf-8")

    if language == Language.PYTHON:
        _analyze_python_ast(root, source_bytes, analysis)
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        _analyze_js_ts_ast(root, source_bytes, analysis, language)
    elif language == Language.GO:
        _analyze_go_ast(root, source_bytes, analysis)
    elif language == Language.RUST:
        _analyze_rust_ast(root, source_bytes, analysis)
    elif language == Language.JAVA:
        _analyze_java_ast(root, source_bytes, analysis)


def _analyze_python_ast(
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
        methods: list[FunctionInfo] = []
        properties: list[str] = []
        class_variables: list[str] = []

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
                    analysis.imports.append(ImportInfo(module=module, location=make_location(node)))
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
                analysis.imports.append(ImportInfo(module=path_text, location=make_location(node)))

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


__all__ = [
    "TreeSitterParser",
    "analyze_with_tree_sitter",
]
