"""Tests for SQL helper utilities."""

import pytest

from aragora.utils.sql_helpers import _escape_like_pattern, escape_like_pattern


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
