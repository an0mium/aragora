"""Tests for SQL helper utilities."""

from collections.abc import Iterator
import sqlite3

import pytest

from aragora.utils.sql_helpers import _escape_like_pattern, escape_like_pattern


@pytest.fixture
def sqlite_items() -> Iterator[sqlite3.Connection]:
    """In-memory table for validating real LIKE ESCAPE behavior."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (value TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO items (value) VALUES (?)",
        [
            ("100% match",),
            ("1000 match",),
            ("user_name",),
            ("userXname",),
            (r"folder\name",),
            ("folder/name",),
        ],
    )
    try:
        yield conn
    finally:
        conn.close()


def _literal_like_matches(conn: sqlite3.Connection, term: str) -> list[str]:
    escaped = escape_like_pattern(term)
    rows = conn.execute(
        "SELECT value FROM items WHERE value LIKE ? ESCAPE '\\' ORDER BY value",
        (f"%{escaped}%",),
    ).fetchall()
    return [str(row[0]) for row in rows]


def test_escape_like_pattern_leaves_plain_text_unchanged():
    """Returns plain text unchanged when it has no LIKE metacharacters."""
    assert escape_like_pattern("plain text") == "plain text"


def test_escape_like_pattern_escapes_percent():
    """Escapes percent characters used as LIKE wildcards."""
    assert escape_like_pattern("100% match") == "100\\% match"


def test_escape_like_pattern_escapes_underscore():
    """Escapes underscore characters used as LIKE wildcards."""
    assert escape_like_pattern("test_value") == "test\\_value"


def test_escape_like_pattern_escapes_backslash():
    """Escapes backslashes before LIKE metacharacters are processed."""
    assert escape_like_pattern(r"folder\name") == r"folder\\name"


def test_escape_like_pattern_escapes_mixed_metacharacters():
    """Escapes backslashes, percent, and underscore in one pass."""
    assert escape_like_pattern(r"100%_match\path") == r"100\%\_match\\path"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("%%", "\\%\\%"),
        ("__init__", "\\_\\_init\\_\\_"),
        ("%_\\", "\\%\\_\\\\"),
        ("\\%", "\\\\\\%"),
    ],
)
def test_escape_like_pattern_escapes_repeated_metacharacters(value: str, expected: str):
    """Escapes repeated and adjacent LIKE metacharacters consistently."""
    assert escape_like_pattern(value) == expected


def test_escape_like_pattern_returns_empty_string_for_empty_input():
    """Preserves empty string input."""
    assert escape_like_pattern("") == ""


def test_escape_like_pattern_preserves_unicode_while_escaping_metacharacters():
    """Leaves unicode untouched while escaping LIKE metacharacters."""
    assert escape_like_pattern("caf\u00e9_100%") == "caf\u00e9\\_100\\%"


def test_escape_like_pattern_raises_type_error_for_none():
    """Rejects None input."""
    with pytest.raises(TypeError, match="value must be a string"):
        escape_like_pattern(None)


def test_escape_like_pattern_raises_type_error_for_non_string():
    """Rejects non-string input types."""
    with pytest.raises(TypeError, match="value must be a string"):
        escape_like_pattern(123)


def test__escape_like_pattern_matches_public_function():
    """Backward-compatible wrapper delegates to the public helper."""
    value = r"report_%\2026"
    assert _escape_like_pattern(value) == escape_like_pattern(value)


def test_escape_like_pattern_prevents_percent_wildcard_expansion_in_sqlite(
    sqlite_items: sqlite3.Connection,
):
    """Escaped percent signs match literal percent signs in SQLite LIKE."""
    assert _literal_like_matches(sqlite_items, "100%") == ["100% match"]


def test_escape_like_pattern_prevents_underscore_wildcard_expansion_in_sqlite(
    sqlite_items: sqlite3.Connection,
):
    """Escaped underscores match literal underscores in SQLite LIKE."""
    assert _literal_like_matches(sqlite_items, "user_name") == ["user_name"]


def test_escape_like_pattern_matches_literal_backslashes_in_sqlite(
    sqlite_items: sqlite3.Connection,
):
    """Escaped backslashes remain searchable as literal characters."""
    assert _literal_like_matches(sqlite_items, r"folder\name") == [r"folder\name"]
