"""
Code Intelligence Module using Tree-sitter AST parsing.

Provides semantic code analysis with accurate symbol extraction, type information,
and structural understanding across multiple programming languages.

This module re-exports from the `aragora.analysis.intelligence` package for
backwards compatibility. For new code, prefer importing directly from the package:

    from aragora.analysis.intelligence import CodeIntelligence

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

# Re-export everything from the intelligence package for backwards compatibility
from aragora.analysis.intelligence import (
    # Main analyzer
    CodeIntelligence,
    # Type definitions
    Language,
    SymbolKind,
    Visibility,
    SourceLocation,
    Parameter,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    FileAnalysis,
    # AST analysis
    TreeSitterParser,
    analyze_with_tree_sitter,
    # Symbol utilities
    SymbolReference,
    SymbolTable,
    find_symbol_usages,
    get_symbol_at_location,
    extract_symbols_from_analysis,
    get_exported_symbols,
    find_symbol_definition,
    analyze_with_regex,
    # Complexity metrics
    ComplexityMetrics,
    calculate_cyclomatic_complexity,
    calculate_cognitive_complexity,
    calculate_nesting_depth,
    get_complexity_metrics,
    calculate_function_complexity,
    get_file_complexity_summary,
    # Call graph
    CallSite,
    CallGraph,
    build_call_graph,
    build_call_graph_from_source,
    get_call_chain,
    # Dead code detection
    DeadCodeFinding,
    DeadCodeReport,
    detect_dead_code,
    find_unreachable_code,
    find_unused_imports,
    find_unused_variables,
)


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
    # Symbol utilities
    "SymbolReference",
    "SymbolTable",
    "find_symbol_usages",
    "get_symbol_at_location",
    "extract_symbols_from_analysis",
    "get_exported_symbols",
    "find_symbol_definition",
    "analyze_with_regex",
    "analyze_with_tree_sitter",
    # Complexity metrics
    "ComplexityMetrics",
    "calculate_cyclomatic_complexity",
    "calculate_cognitive_complexity",
    "calculate_nesting_depth",
    "get_complexity_metrics",
    "calculate_function_complexity",
    "get_file_complexity_summary",
    # Call graph
    "CallSite",
    "CallGraph",
    "build_call_graph",
    "build_call_graph_from_source",
    "get_call_chain",
    # Dead code detection
    "DeadCodeFinding",
    "DeadCodeReport",
    "detect_dead_code",
    "find_unreachable_code",
    "find_unused_imports",
    "find_unused_variables",
]
