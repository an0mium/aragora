"""
Full-Text Search (FTS) utilities for SQLite FTS5.

Provides sanitization and query processing for safe FTS5 queries
across all Aragora storage modules.
"""

from typing import Set

# FTS query limits
MAX_FTS_QUERY_LENGTH = 500
MAX_FTS_TERMS = 20
FTS_SPECIAL_CHARS: Set[str] = set('"*(){}[]^:?-+~')


def sanitize_fts_query(
    query: str,
    max_length: int = MAX_FTS_QUERY_LENGTH,
    max_terms: int = MAX_FTS_TERMS,
) -> str:
    """Sanitize and limit FTS query complexity.

    Prevents FTS injection and ensures query complexity stays bounded.
    Used by FactStore, EvidenceStore, and other FTS-enabled stores.

    Args:
        query: Raw search query
        max_length: Maximum query length (default 500)
        max_terms: Maximum number of terms (default 20)

    Returns:
        Sanitized query safe for FTS5

    Example:
        >>> sanitize_fts_query("hello world")
        'hello world'
        >>> sanitize_fts_query("user:admin OR password:*")
        'useradmin OR password*'
    """
    if not query or not query.strip():
        return ""

    # Truncate to max length
    query = query[:max_length]

    # Remove dangerous FTS special characters (keep * for prefix search)
    sanitized = []
    for char in query:
        if char in FTS_SPECIAL_CHARS:
            if char == "*":
                sanitized.append(char)
            # Skip other special chars
        else:
            sanitized.append(char)
    query = "".join(sanitized)

    # Limit number of terms
    terms = query.split()
    if len(terms) > max_terms:
        terms = terms[:max_terms]

    return " ".join(terms)


def build_fts_match_query(
    terms: str,
    prefix_match: bool = False,
) -> str:
    """Build an FTS5 MATCH query from terms.

    Args:
        terms: Space-separated search terms
        prefix_match: If True, add * suffix for prefix matching

    Returns:
        FTS5 MATCH expression

    Example:
        >>> build_fts_match_query("hello world")
        'hello world'
        >>> build_fts_match_query("hello world", prefix_match=True)
        'hello* world*'
    """
    sanitized = sanitize_fts_query(terms)
    if not sanitized:
        return ""

    if prefix_match:
        words = sanitized.split()
        return " ".join(f"{word}*" for word in words)

    return sanitized


def escape_fts_string(value: str) -> str:
    """Escape a string for use in FTS5 queries.

    Args:
        value: String to escape

    Returns:
        Escaped string safe for FTS5
    """
    # Double any quotes
    return value.replace('"', '""')
