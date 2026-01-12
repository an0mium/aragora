"""Tests for SQL helper utilities."""

import pytest

from aragora.utils.sql_helpers import escape_like_pattern, _escape_like_pattern


class TestEscapeLikePattern:
    """Tests for escape_like_pattern function."""

    def test_escape_percent(self):
        """Percent wildcard is escaped."""
        assert escape_like_pattern("100%") == "100\\%"
        assert escape_like_pattern("%admin%") == "\\%admin\\%"

    def test_escape_underscore(self):
        """Underscore wildcard is escaped."""
        assert escape_like_pattern("user_name") == "user\\_name"
        assert escape_like_pattern("_prefix") == "\\_prefix"

    def test_escape_backslash(self):
        """Backslash is escaped first to avoid double-escaping."""
        assert escape_like_pattern("path\\file") == "path\\\\file"
        assert escape_like_pattern("\\\\server") == "\\\\\\\\server"

    def test_escape_combined(self):
        """All special characters are escaped correctly together."""
        # Order matters: backslash first, then % and _
        assert escape_like_pattern("100%_test\\path") == "100\\%\\_test\\\\path"
        assert escape_like_pattern("a\\b%c_d") == "a\\\\b\\%c\\_d"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert escape_like_pattern("") == ""

    def test_no_special_chars(self):
        """String without special chars is unchanged."""
        assert escape_like_pattern("hello world") == "hello world"
        assert escape_like_pattern("simple123") == "simple123"

    def test_unicode(self):
        """Unicode characters are preserved."""
        assert escape_like_pattern("hello\u4e16\u754c") == "hello\u4e16\u754c"
        assert escape_like_pattern("\u00e9clair%") == "\u00e9clair\\%"
        # Unicode with special chars
        assert escape_like_pattern("caf\u00e9_100%") == "caf\u00e9\\_100\\%"

    def test_multiple_consecutive_specials(self):
        """Multiple consecutive special characters are all escaped."""
        assert escape_like_pattern("%%%") == "\\%\\%\\%"
        assert escape_like_pattern("___") == "\\_\\_\\_"
        assert escape_like_pattern("\\\\\\") == "\\\\\\\\\\\\"

    def test_mixed_consecutive(self):
        """Mixed consecutive special characters are handled."""
        assert escape_like_pattern("%_\\") == "\\%\\_\\\\"
        assert escape_like_pattern("\\_%") == "\\\\\\_\\%"


class TestBackwardsCompatibilityAlias:
    """Tests for _escape_like_pattern alias."""

    def test_alias_exists(self):
        """The underscore-prefixed alias exists for backwards compatibility."""
        assert _escape_like_pattern is not None
        assert callable(_escape_like_pattern)

    def test_alias_same_behavior(self):
        """Alias behaves identically to main function."""
        test_cases = [
            "simple",
            "100%",
            "user_name",
            "path\\file",
            "a\\b%c_d",
            "",
            "hello\u4e16\u754c",
        ]
        for case in test_cases:
            assert _escape_like_pattern(case) == escape_like_pattern(case)


class TestSqlInjectionPrevention:
    """Tests verifying SQL injection prevention."""

    def test_prevents_wildcard_injection(self):
        """User input with wildcards cannot match unintended rows."""
        # User tries to search for all users by injecting %
        malicious_input = "%"
        escaped = escape_like_pattern(malicious_input)
        # Should match literal % not all rows
        assert escaped == "\\%"
        # When used in LIKE with ESCAPE, this matches only '%' character

    def test_prevents_single_char_wildcard(self):
        """User input with _ cannot match single characters."""
        malicious_input = "admin_"  # Tries to match admin1, admin2, etc.
        escaped = escape_like_pattern(malicious_input)
        assert escaped == "admin\\_"

    def test_real_world_attack_pattern(self):
        """Tests against realistic attack patterns."""
        # Someone trying to search all debates
        attack = "%' OR '1'='1"
        escaped = escape_like_pattern(attack)
        assert escaped == "\\%' OR '1'='1"

        # SQL with LIKE won't be fooled
        # The ' characters are handled by parameterized queries,
        # this function only handles LIKE wildcards
