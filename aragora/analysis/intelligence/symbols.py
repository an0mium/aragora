"""
Symbol Extraction Module.

Provides utilities for extracting, finding, and analyzing symbols
(functions, classes, variables) in source code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .types import (
    ClassInfo,
    FileAnalysis,
    FunctionInfo,
    ImportInfo,
    Language,
    SourceLocation,
    SymbolKind,
    Visibility,
)


@dataclass
class SymbolReference:
    """A reference to a symbol in source code."""

    name: str
    location: SourceLocation
    kind: SymbolKind | None = None
    is_definition: bool = False
    is_import: bool = False


@dataclass
class SymbolTable:
    """Table of symbols in a file or scope."""

    symbols: dict[str, list[SymbolReference]] = field(default_factory=dict)
    imports: dict[str, ImportInfo] = field(default_factory=dict)

    def add_symbol(self, ref: SymbolReference) -> None:
        """Add a symbol reference to the table."""
        if ref.name not in self.symbols:
            self.symbols[ref.name] = []
        self.symbols[ref.name].append(ref)

    def get_definitions(self, name: str) -> list[SymbolReference]:
        """Get all definitions of a symbol."""
        refs = self.symbols.get(name, [])
        return [r for r in refs if r.is_definition]

    def get_usages(self, name: str) -> list[SymbolReference]:
        """Get all usages (non-definition references) of a symbol."""
        refs = self.symbols.get(name, [])
        return [r for r in refs if not r.is_definition]


def find_symbol_usages(
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
    import logging

    logger = logging.getLogger(__name__)
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
        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Error searching %s: %s", path, e)

    return usages


def get_symbol_at_location(
    analysis: FileAnalysis,
    line: int,
    column: int,
) -> FunctionInfo | ClassInfo | None:
    """
    Get the symbol at a specific location in analyzed file.

    Args:
        analysis: The file analysis result
        line: Line number (1-indexed)
        column: Column number (0-indexed)

    Returns:
        The symbol at that location, or None
    """
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


def extract_symbols_from_analysis(analysis: FileAnalysis) -> SymbolTable:
    """
    Extract all symbols from a file analysis into a symbol table.

    Args:
        analysis: The file analysis result

    Returns:
        SymbolTable with all symbols
    """
    table = SymbolTable()

    # Add imports
    for imp in analysis.imports:
        table.imports[imp.module] = imp
        if imp.names:
            for name in imp.names:
                ref = SymbolReference(
                    name=name,
                    location=imp.location or SourceLocation(analysis.file_path, 0, 0, 0, 0),
                    kind=SymbolKind.IMPORT,
                    is_import=True,
                )
                table.add_symbol(ref)

    # Add functions
    for func in analysis.functions:
        ref = SymbolReference(
            name=func.name,
            location=func.location,
            kind=func.kind,
            is_definition=True,
        )
        table.add_symbol(ref)

    # Add classes and their members
    for cls in analysis.classes:
        ref = SymbolReference(
            name=cls.name,
            location=cls.location,
            kind=cls.kind,
            is_definition=True,
        )
        table.add_symbol(ref)

        for method in cls.methods:
            ref = SymbolReference(
                name=f"{cls.name}.{method.name}",
                location=method.location,
                kind=method.kind,
                is_definition=True,
            )
            table.add_symbol(ref)

    # Add exports
    for export in analysis.exports:
        ref = SymbolReference(
            name=export,
            location=SourceLocation(analysis.file_path, 0, 0, 0, 0),
            kind=SymbolKind.VARIABLE,
            is_definition=False,
        )
        table.add_symbol(ref)

    return table


def get_exported_symbols(analysis: FileAnalysis) -> list[str]:
    """
    Get all exported/public symbols from a file.

    Args:
        analysis: The file analysis result

    Returns:
        List of exported symbol names
    """
    # If __all__ is defined, use it
    if analysis.exports:
        return analysis.exports

    # Otherwise, collect public symbols
    exported = []

    for func in analysis.functions:
        if not func.name.startswith("_"):
            exported.append(func.name)

    for cls in analysis.classes:
        if not cls.name.startswith("_"):
            exported.append(cls.name)

    return exported


def find_symbol_definition(
    symbol_name: str,
    analyses: dict[str, FileAnalysis],
) -> SourceLocation | None:
    """
    Find the definition of a symbol across multiple files.

    Args:
        symbol_name: Name of the symbol to find
        analyses: Dictionary of file path to analysis

    Returns:
        Location of the definition, or None if not found
    """
    for file_path, analysis in analyses.items():
        # Check functions
        for func in analysis.functions:
            if func.name == symbol_name:
                return func.location

        # Check classes
        for cls in analysis.classes:
            if cls.name == symbol_name:
                return cls.location

            # Check methods
            for method in cls.methods:
                if method.name == symbol_name:
                    return method.location

    return None


def analyze_with_regex(
    source: str,
    analysis: FileAnalysis,
    language: Language,
) -> None:
    """
    Fallback analysis using regex patterns.

    Args:
        source: The source code text
        analysis: The FileAnalysis to populate
        language: The programming language
    """
    file_path = analysis.file_path
    lines = source.split("\n")

    if language == Language.PYTHON:
        _regex_analyze_python(lines, file_path, analysis)
    elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        _regex_analyze_js_ts(lines, file_path, analysis)
    elif language == Language.GO:
        _regex_analyze_go(lines, file_path, analysis)
    elif language == Language.RUST:
        _regex_analyze_rust(lines, file_path, analysis)
    elif language == Language.JAVA:
        _regex_analyze_java(lines, file_path, analysis)


def _regex_analyze_python(
    lines: list[str],
    file_path: str,
    analysis: FileAnalysis,
) -> None:
    """Regex-based Python analysis."""
    # Pattern for imports
    import_pattern = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+?)(?:\s+as\s+\S+)?$")
    # Pattern for functions
    func_pattern = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?:")
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


__all__ = [
    "SymbolReference",
    "SymbolTable",
    "find_symbol_usages",
    "get_symbol_at_location",
    "extract_symbols_from_analysis",
    "get_exported_symbols",
    "find_symbol_definition",
    "analyze_with_regex",
]
