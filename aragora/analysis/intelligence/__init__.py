"""
Code Intelligence Package.

Provides semantic code analysis with accurate symbol extraction, type information,
and structural understanding across multiple programming languages.

Supported Languages:
- Python
- JavaScript/TypeScript
- Go
- Rust
- Java

Example:
    >>> from aragora.analysis.intelligence import CodeIntelligence
    >>> intel = CodeIntelligence()
    >>> analysis = intel.analyze_file("src/main.py")
    >>> print(f"Found {len(analysis.classes)} classes, {len(analysis.functions)} functions")
    >>> for cls in analysis.classes:
    ...     print(f"  - {cls.name} with {len(cls.methods)} methods")

Modules:
    - analyzer: Main CodeIntelligence class
    - types: Data classes and enums (Language, SymbolKind, etc.)
    - ast_analysis: Tree-sitter based AST parsing
    - symbols: Symbol extraction and usage finding
    - complexity: Cyclomatic and cognitive complexity metrics
    - call_graph: Call graph construction and analysis
    - dead_code: Dead code detection
"""

# Main analyzer class
from .analyzer import CodeIntelligence

# Type definitions
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

# AST analysis
from .ast_analysis import TreeSitterParser, analyze_with_tree_sitter

# Symbol utilities
from .symbols import (
    SymbolReference,
    SymbolTable,
    analyze_with_regex,
    extract_symbols_from_analysis,
    find_symbol_definition,
    find_symbol_usages,
    get_exported_symbols,
    get_symbol_at_location,
)

# Complexity metrics
from .complexity import (
    ComplexityMetrics,
    calculate_cognitive_complexity,
    calculate_cyclomatic_complexity,
    calculate_function_complexity,
    calculate_nesting_depth,
    get_complexity_metrics,
    get_file_complexity_summary,
)

# Call graph
from .call_graph import (
    CallGraph,
    CallSite,
    build_call_graph,
    build_call_graph_from_source,
    get_call_chain,
)

# Dead code detection
from .dead_code import (
    DeadCodeFinding,
    DeadCodeReport,
    detect_dead_code,
    find_unreachable_code,
    find_unused_imports,
    find_unused_variables,
)


__all__ = [
    # Main analyzer
    "CodeIntelligence",
    # Type definitions
    "Language",
    "SymbolKind",
    "Visibility",
    "SourceLocation",
    "Parameter",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    "FileAnalysis",
    # AST analysis
    "TreeSitterParser",
    "analyze_with_tree_sitter",
    # Symbol utilities
    "SymbolReference",
    "SymbolTable",
    "find_symbol_usages",
    "get_symbol_at_location",
    "extract_symbols_from_analysis",
    "get_exported_symbols",
    "find_symbol_definition",
    "analyze_with_regex",
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
