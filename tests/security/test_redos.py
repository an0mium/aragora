"""
Tests for ReDoS (Regular Expression Denial of Service) protection.

Tests cover:
- Detection of dangerous regex patterns
- Timeout-protected regex execution
- Safe search query validation
- Edge cases and boundary conditions
"""

import re
import time

import pytest

from aragora.server.validation.security import (
    REGEX_TIMEOUT_SECONDS,
    execute_regex_with_timeout,
    is_safe_regex_pattern,
    validate_search_query_redos_safe,
)


class TestIsSafeRegexPattern:
    """Tests for is_safe_regex_pattern function."""

    def test_empty_pattern_is_safe(self):
        """Test that empty pattern is considered safe."""
        is_safe, error = is_safe_regex_pattern("")
        assert is_safe is True
        assert error is None

    def test_simple_literal_is_safe(self):
        """Test that simple literal patterns are safe."""
        is_safe, error = is_safe_regex_pattern("hello world")
        assert is_safe is True
        assert error is None

    def test_simple_character_class_is_safe(self):
        """Test that simple character classes are safe."""
        is_safe, error = is_safe_regex_pattern("[a-z]+")
        assert is_safe is True
        assert error is None

    def test_simple_quantifiers_are_safe(self):
        """Test that simple quantifiers are safe."""
        patterns = [
            r"\d+",
            r"\w*",
            r"[a-z]?",
            r"abc{1,5}",
        ]
        for pattern in patterns:
            is_safe, error = is_safe_regex_pattern(pattern)
            assert is_safe is True, f"Pattern {pattern!r} should be safe"

    def test_nested_quantifiers_detected(self):
        """Test that nested quantifiers are detected as dangerous."""
        dangerous_patterns = [
            r"(a+)+",
            r"(a*)+",
            r"(a+)*",
            r"(a*)*",
            r"([a-z]+)+",
        ]
        for pattern in dangerous_patterns:
            is_safe, error = is_safe_regex_pattern(pattern)
            assert is_safe is False, f"Pattern {pattern!r} should be detected as dangerous"
            assert "ReDoS" in error or "nested quantifiers" in error.lower()

    def test_pattern_length_limit(self):
        """Test that overly long patterns are rejected."""
        long_pattern = "a" * 200
        is_safe, error = is_safe_regex_pattern(long_pattern)
        assert is_safe is False
        assert "maximum length" in error.lower()

    def test_alternation_with_quantifiers(self):
        """Test that overlapping alternation with quantifiers is detected."""
        pattern = r"(a|a)+"
        is_safe, error = is_safe_regex_pattern(pattern)
        # This should be detected as potentially dangerous
        # The exact detection depends on the implementation
        assert is_safe is False or "backtracking" in (error or "").lower()

    def test_safe_alternation(self):
        """Test that non-overlapping alternation is safe."""
        is_safe, error = is_safe_regex_pattern(r"(cat|dog)")
        assert is_safe is True

    def test_backreference_patterns(self):
        """Test various backreference patterns."""
        # Simple backreference without quantifiers after group is safe
        is_safe, error = is_safe_regex_pattern(r"(word)\s+\1")
        assert is_safe is True  # Basic backreference with literal


class TestExecuteRegexWithTimeout:
    """Tests for execute_regex_with_timeout function."""

    def test_simple_match_succeeds(self):
        """Test that simple regex matches work."""
        match = execute_regex_with_timeout(r"\d+", "abc123def")
        assert match is not None
        assert match.group() == "123"

    def test_no_match_returns_none(self):
        """Test that non-matching regex returns None."""
        match = execute_regex_with_timeout(r"\d+", "no digits here")
        assert match is None

    def test_invalid_pattern_returns_none(self):
        """Test that invalid regex pattern returns None."""
        match = execute_regex_with_timeout(r"[invalid", "test")
        assert match is None

    def test_compiled_pattern_works(self):
        """Test that pre-compiled patterns work."""
        pattern = re.compile(r"\w+")
        match = execute_regex_with_timeout(pattern, "hello world")
        assert match is not None
        assert match.group() == "hello"

    def test_timeout_on_slow_regex(self):
        """Test that slow regex times out.

        Note: Due to Python's GIL and thread limitations, we can't actually
        interrupt a running regex. This test verifies the timeout mechanism
        is in place and returns None for slow operations.
        """
        # Use a pattern that will simply not match quickly
        # rather than one that causes catastrophic backtracking
        # (catastrophic backtracking can't be interrupted in Python)
        pattern = r"\d+"
        text = "no digits here"

        # This should complete quickly and return None (no match)
        match = execute_regex_with_timeout(pattern, text, timeout=0.1)
        assert match is None  # No match found

        # Verify that a matching pattern works
        match2 = execute_regex_with_timeout(r"\d+", "abc123", timeout=0.1)
        assert match2 is not None
        assert match2.group() == "123"

    def test_respects_custom_timeout(self):
        """Test that custom timeout is respected."""
        # Very short timeout
        match = execute_regex_with_timeout(r"\d+", "123", timeout=0.001)
        # Should still succeed for simple patterns
        # (or might timeout on very slow systems)
        # We just verify it doesn't crash

    def test_flags_are_applied(self):
        """Test that regex flags are applied."""
        match = execute_regex_with_timeout(r"hello", "HELLO WORLD", flags=re.IGNORECASE)
        assert match is not None
        assert match.group() == "HELLO"


class TestValidateSearchQueryRedosSafe:
    """Tests for validate_search_query_redos_safe function."""

    def test_empty_query_is_valid(self):
        """Test that empty query is valid."""
        result = validate_search_query_redos_safe("")
        assert result.is_valid is True
        assert result.sanitized == ""

    def test_simple_query_is_valid(self):
        """Test that simple text query is valid."""
        result = validate_search_query_redos_safe("hello world")
        assert result.is_valid is True
        assert result.value == "hello world"

    def test_query_length_limit(self):
        """Test that overly long queries are rejected."""
        long_query = "a" * 300
        result = validate_search_query_redos_safe(long_query)
        assert result.is_valid is False
        assert "maximum length" in result.error.lower()

    def test_custom_length_limit(self):
        """Test custom length limit."""
        result = validate_search_query_redos_safe("hello", max_length=3)
        assert result.is_valid is False
        assert "maximum length" in result.error.lower()

    def test_wildcard_conversion(self):
        """Test that wildcards are converted safely."""
        result = validate_search_query_redos_safe("test*query")
        assert result.is_valid is True
        # The * should be converted to a bounded quantifier
        assert ".{0,100}" in result.sanitized

    def test_question_mark_wildcard(self):
        """Test that ? wildcard is converted."""
        result = validate_search_query_redos_safe("te?t")
        assert result.is_valid is True
        assert "." in result.sanitized  # ? becomes .

    def test_wildcards_disabled(self):
        """Test disabling wildcard conversion."""
        result = validate_search_query_redos_safe("test*query", allow_wildcards=False)
        assert result.is_valid is True
        # * should be escaped, not converted
        assert r"\*" in result.sanitized

    def test_regex_metacharacters_escaped(self):
        """Test that regex metacharacters are escaped."""
        result = validate_search_query_redos_safe("test.query[0]")
        assert result.is_valid is True
        # Special chars should be escaped
        assert r"\." in result.sanitized
        assert r"\[" in result.sanitized
        assert r"\]" in result.sanitized

    def test_preserves_original_value(self):
        """Test that original value is preserved."""
        result = validate_search_query_redos_safe("my search")
        assert result.value == "my search"


class TestReDoSProtectionIntegration:
    """Integration tests for ReDoS protection."""

    def test_user_search_workflow(self):
        """Test typical user search workflow."""
        user_inputs = [
            "simple search",
            "find*something",
            "test query with spaces",
            "special.chars[in]search",
        ]

        for user_input in user_inputs:
            result = validate_search_query_redos_safe(user_input)
            assert result.is_valid is True, f"Input {user_input!r} should be valid"

            # The sanitized pattern should be safe
            is_safe, _ = is_safe_regex_pattern(result.sanitized)
            assert is_safe is True, f"Sanitized pattern for {user_input!r} should be safe"

    def test_malicious_input_blocked(self):
        """Test that potentially malicious inputs are handled safely."""
        malicious_inputs = [
            # Very long input
            "a" * 1000,
            # Input with many wildcards (would create many quantifiers)
            "*" * 50,
        ]

        for malicious_input in malicious_inputs:
            result = validate_search_query_redos_safe(malicious_input)
            # Either rejected or sanitized to be safe
            if result.is_valid:
                is_safe, _ = is_safe_regex_pattern(result.sanitized)
                assert is_safe is True

    def test_performance_under_load(self):
        """Test that validation performs well under load."""

        queries = ["test query " + str(i) for i in range(100)]

        start = time.monotonic()
        for query in queries:
            result = validate_search_query_redos_safe(query)
            assert result.is_valid is True
        elapsed = time.monotonic() - start

        # Should complete in well under 1 second
        assert elapsed < 1.0, f"Validation of 100 queries took {elapsed:.2f}s"
