"""
MCP Tools for codebase exploration and code-grounded debates.

Provides tools for searching and navigating codebases:
- search_codebase: Search for code patterns, symbols, and content
- get_symbol: Look up a specific symbol definition
- get_dependencies: Get import/dependency graph for a file
- get_codebase_structure: Get directory structure overview
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _validate_codebase_path(codebase_path: str) -> str | None:
    """Return an error message if the codebase path does not exist, else None."""
    if not os.path.isdir(codebase_path):
        return f"Codebase path does not exist: {codebase_path}"
    return None


async def search_codebase_tool(
    query: str,
    codebase_path: str = ".",
    file_types: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search a codebase for symbols, patterns, or content matching a query.

    Args:
        query: Search query (symbol name, pattern, or text)
        codebase_path: Path to the repository root
        file_types: Comma-separated file extensions to filter (e.g., "py,ts")
        limit: Maximum results to return

    Returns:
        Dict with matching files, symbols, and snippets
    """
    path_err = _validate_codebase_path(codebase_path)
    if path_err:
        return {"error": path_err, "query": query}

    results: list[dict[str, Any]] = []

    try:
        from aragora.connectors.repository_crawler import RepositoryCrawler

        crawler = RepositoryCrawler(codebase_path)
        crawl_result = await crawler.crawl()

        type_filter = set()
        if file_types:
            type_filter = {f".{t.strip().lstrip('.')}" for t in file_types.split(",")}

        query_lower = query.lower()
        for crawled_file in crawl_result.files:
            file_path = getattr(crawled_file, "relative_path", "") or getattr(
                crawled_file, "path", ""
            )

            if type_filter:
                ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
                if ext not in type_filter:
                    continue

            # Check symbols
            for symbol in getattr(crawled_file, "symbols", []):
                name = getattr(symbol, "name", "")
                if query_lower in name.lower():
                    results.append(
                        {
                            "file": file_path,
                            "symbol": name,
                            "kind": getattr(symbol, "kind", "unknown"),
                            "line": getattr(symbol, "line", 0),
                            "match_type": "symbol",
                        }
                    )

            # Check file path
            if query_lower in file_path.lower():
                results.append(
                    {
                        "file": file_path,
                        "match_type": "path",
                        "lines": getattr(crawled_file, "line_count", 0),
                    }
                )

            if len(results) >= limit:
                break

        return {
            "results": results[:limit],
            "count": len(results[:limit]),
            "query": query,
            "codebase_path": codebase_path,
            "total_files_scanned": crawl_result.total_files,
        }

    except ImportError:
        logger.warning("RepositoryCrawler not available")
        return {"error": "Repository crawler module not available", "query": query}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Codebase search failed: %s", e)
        return {"error": "Codebase search failed", "query": query}


async def get_symbol_tool(
    symbol_name: str,
    codebase_path: str = ".",
    include_source: bool = False,
) -> dict[str, Any]:
    """
    Look up a specific symbol (function, class, variable) in the codebase.

    Args:
        symbol_name: Name of the symbol to find
        codebase_path: Path to the repository root
        include_source: Whether to include source code snippet

    Returns:
        Dict with symbol definitions and locations
    """
    matches: list[dict[str, Any]] = []

    try:
        from aragora.connectors.repository_crawler import RepositoryCrawler

        crawler = RepositoryCrawler(codebase_path)
        crawl_result = await crawler.crawl()

        name_lower = symbol_name.lower()
        for crawled_file in crawl_result.files:
            file_path = getattr(crawled_file, "relative_path", "") or getattr(
                crawled_file, "path", ""
            )
            for symbol in getattr(crawled_file, "symbols", []):
                name = getattr(symbol, "name", "")
                if name_lower == name.lower():
                    match: dict[str, Any] = {
                        "name": name,
                        "file": file_path,
                        "kind": getattr(symbol, "kind", "unknown"),
                        "line": getattr(symbol, "line", 0),
                        "docstring": (getattr(symbol, "docstring", "") or "")[:500],
                    }
                    if include_source:
                        source = getattr(symbol, "source", "") or ""
                        match["source"] = source[:2000]
                    matches.append(match)

        return {
            "symbol": symbol_name,
            "matches": matches,
            "count": len(matches),
            "codebase_path": codebase_path,
        }

    except ImportError:
        logger.warning("RepositoryCrawler not available")
        return {"error": "Repository crawler module not available", "symbol": symbol_name}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Symbol lookup failed: %s", e)
        return {"error": "Symbol lookup failed", "symbol": symbol_name}


async def get_dependencies_tool(
    file_path: str,
    codebase_path: str = ".",
    direction: str = "outgoing",
) -> dict[str, Any]:
    """
    Get the dependency graph for a file.

    Args:
        file_path: Relative path to the file
        codebase_path: Path to the repository root
        direction: "outgoing" (imports from this file) or "incoming" (files importing this)

    Returns:
        Dict with dependency information
    """
    try:
        from aragora.connectors.repository_crawler import RepositoryCrawler

        crawler = RepositoryCrawler(codebase_path)
        crawl_result = await crawler.crawl()

        dep_graph = getattr(crawl_result, "dependency_graph", {})

        if direction == "outgoing":
            deps = dep_graph.get(file_path, [])
            return {
                "file": file_path,
                "direction": "outgoing",
                "dependencies": deps,
                "count": len(deps),
                "codebase_path": codebase_path,
            }
        else:
            # Find files that import this file
            incoming: list[str] = []
            for source, targets in dep_graph.items():
                if file_path in targets:
                    incoming.append(source)
            return {
                "file": file_path,
                "direction": "incoming",
                "dependents": incoming,
                "count": len(incoming),
                "codebase_path": codebase_path,
            }

    except ImportError:
        logger.warning("RepositoryCrawler not available")
        return {"error": "Repository crawler module not available", "file": file_path}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Dependency lookup failed: %s", e)
        return {"error": "Dependency lookup failed", "file": file_path}


async def get_codebase_structure_tool(
    codebase_path: str = ".",
    max_depth: int = 3,
) -> dict[str, Any]:
    """
    Get a directory structure overview of the codebase.

    Args:
        codebase_path: Path to the repository root
        max_depth: Maximum directory depth to include

    Returns:
        Dict with directory tree and file statistics
    """
    try:
        from aragora.connectors.repository_crawler import RepositoryCrawler

        crawler = RepositoryCrawler(codebase_path)
        crawl_result = await crawler.crawl()

        # Build directory tree
        tree: dict[str, Any] = {}
        for crawled_file in crawl_result.files:
            file_path = getattr(crawled_file, "relative_path", "") or getattr(
                crawled_file, "path", ""
            )
            parts = file_path.split("/")
            if len(parts) > max_depth + 1:
                # Truncate deep paths
                parts = parts[: max_depth + 1]

            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Leaf: store file info
            current[parts[-1]] = {
                "lines": getattr(crawled_file, "line_count", 0),
                "symbols": len(getattr(crawled_file, "symbols", [])),
            }

        return {
            "structure": tree,
            "total_files": crawl_result.total_files,
            "total_lines": crawl_result.total_lines,
            "file_types": crawl_result.file_type_counts,
            "symbol_counts": crawl_result.symbol_counts,
            "codebase_path": codebase_path,
        }

    except ImportError:
        logger.warning("RepositoryCrawler not available")
        return {"error": "Repository crawler module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Structure lookup failed: %s", e)
        return {"error": "Codebase structure lookup failed"}


__all__ = [
    "search_codebase_tool",
    "get_symbol_tool",
    "get_dependencies_tool",
    "get_codebase_structure_tool",
]
