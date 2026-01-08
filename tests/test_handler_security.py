"""Security tests for handler input validation - path traversal, injection attacks."""

import pytest

from aragora.server.validation import (
    SAFE_ID_PATTERN,
    SAFE_AGENT_PATTERN,
    SAFE_SLUG_PATTERN,
    SAFE_ID_PATTERN_WITH_DOTS,
    validate_path_segment,
    validate_agent_name,
    validate_debate_id,
    validate_plugin_name,
    validate_genome_id,
    validate_json_body,
    validate_string_field,
    sanitize_string,
    sanitize_id,
)


# =============================================================================
# Path Traversal Attack Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Tests to ensure path traversal attacks are blocked."""

    @pytest.mark.parametrize("malicious_path", [
        "../etc/passwd",
        "..\\windows\\system32",
        "....//....//etc/passwd",
        "%2e%2e%2f",  # URL encoded ../
        "%2e%2e/",
        "..%2f",
        "%2e%2e%5c",  # URL encoded ..\
        "..%5c",
        "..;/",
        "..%00/",  # Null byte
        "..%0d/",  # Carriage return
        "..%0a/",  # Newline
        "....//",
        "..../",
        "/..",
        "/./../../",
        "file://etc/passwd",
        "file:///etc/passwd",
    ])
    def test_path_traversal_blocked_in_id(self, malicious_path):
        """Path traversal attempts are rejected by ID validation."""
        is_valid, _ = validate_path_segment(malicious_path, "id", SAFE_ID_PATTERN)
        assert not is_valid

    @pytest.mark.parametrize("malicious_path", [
        "../secret",
        "..\\secret",
        "parent/../sibling",
        "./current",
        "~/home",
        "$HOME/secret",
        "${HOME}/secret",
    ])
    def test_path_traversal_blocked_in_agent_name(self, malicious_path):
        """Path traversal blocked in agent names."""
        is_valid, _ = validate_agent_name(malicious_path)
        assert not is_valid

    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "debate-id/../../../secret",
        "valid..invalid",  # Double dots in middle
    ])
    def test_path_traversal_blocked_in_debate_id(self, malicious_path):
        """Path traversal blocked in debate IDs."""
        is_valid, _ = validate_debate_id(malicious_path)
        assert not is_valid


# =============================================================================
# SQL Injection Attack Tests
# =============================================================================


class TestSQLInjectionPrevention:
    """Tests to ensure SQL injection patterns are blocked."""

    @pytest.mark.parametrize("sql_injection", [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "1; SELECT * FROM users",
        "admin'--",
        "' OR '1'='1",
        "1 UNION SELECT * FROM passwords",
        "'; EXEC xp_cmdshell('dir'); --",
        "1; INSERT INTO users VALUES('hacker')",
        "' OR ''='",
        "1; DELETE FROM users; --",
    ])
    def test_sql_injection_blocked_in_id(self, sql_injection):
        """SQL injection patterns are rejected by ID validation."""
        is_valid, _ = validate_path_segment(sql_injection, "id", SAFE_ID_PATTERN)
        assert not is_valid

    @pytest.mark.parametrize("sql_injection", [
        "agent'; DROP TABLE agents; --",
        "claude OR 1=1",
        "claude' AND '1'='1",
    ])
    def test_sql_injection_blocked_in_agent_name(self, sql_injection):
        """SQL injection patterns blocked in agent names."""
        is_valid, _ = validate_agent_name(sql_injection)
        assert not is_valid


# =============================================================================
# Command Injection Attack Tests
# =============================================================================


class TestCommandInjectionPrevention:
    """Tests to ensure command injection patterns are blocked."""

    @pytest.mark.parametrize("cmd_injection", [
        "; ls -la",
        "| cat /etc/passwd",
        "& whoami",
        "`id`",
        "$(whoami)",
        "|| rm -rf /",
        "&& cat /etc/shadow",
        "\n/bin/sh",
        "\r\ncat /etc/passwd",
        "> /tmp/pwned",
        "< /etc/passwd",
        "id > /tmp/out",
        "$(cat /etc/passwd)",
        "${PATH}",
        "$PATH",
        "!ls",
        "^cat",
    ])
    def test_command_injection_blocked_in_id(self, cmd_injection):
        """Command injection patterns are rejected by ID validation."""
        is_valid, _ = validate_path_segment(cmd_injection, "id", SAFE_ID_PATTERN)
        assert not is_valid

    @pytest.mark.parametrize("cmd_injection", [
        "agent;id",
        "agent|whoami",
        "agent`id`",
        "agent$(id)",
    ])
    def test_command_injection_blocked_in_agent_name(self, cmd_injection):
        """Command injection patterns blocked in agent names."""
        is_valid, _ = validate_agent_name(cmd_injection)
        assert not is_valid


# =============================================================================
# XSS Attack Tests
# =============================================================================


class TestXSSPrevention:
    """Tests to ensure XSS patterns are blocked in IDs."""

    @pytest.mark.parametrize("xss_payload", [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        "javascript:alert(1)",
        "<iframe src='javascript:alert(1)'>",
        "<body onload=alert(1)>",
        "<div onclick=alert(1)>",
        "';alert(1)//",
        "\"><script>alert(1)</script>",
        "<script>document.location='http://evil.com'</script>",
    ])
    def test_xss_blocked_in_id(self, xss_payload):
        """XSS payloads are rejected by ID validation."""
        is_valid, _ = validate_path_segment(xss_payload, "id", SAFE_ID_PATTERN)
        assert not is_valid


# =============================================================================
# Unicode/Encoding Attack Tests
# =============================================================================


class TestUnicodeAttackPrevention:
    """Tests to ensure Unicode-based attacks are blocked."""

    @pytest.mark.parametrize("unicode_attack", [
        "admin\u0000",  # Null byte
        "test\u200b",  # Zero-width space
        "test\u2028",  # Line separator
        "test\u2029",  # Paragraph separator
        "test\ufeff",  # BOM
        "тест",  # Cyrillic lookalike
        "aɗmin",  # Latin Extended-B lookalike
        "../\u0000",  # Null byte in path traversal
        "test\x00test",  # Null byte
        "test\x0d\x0a",  # CRLF injection
    ])
    def test_unicode_attacks_blocked_in_id(self, unicode_attack):
        """Unicode-based attacks are rejected by ID validation."""
        is_valid, _ = validate_path_segment(unicode_attack, "id", SAFE_ID_PATTERN)
        assert not is_valid


# =============================================================================
# Length/DoS Attack Tests
# =============================================================================


class TestLengthLimitEnforcement:
    """Tests for length limits to prevent DoS."""

    def test_id_max_length_enforced(self):
        """IDs exceeding max length (64 chars) are rejected."""
        long_id = "a" * 65
        is_valid, _ = validate_path_segment(long_id, "id", SAFE_ID_PATTERN)
        assert not is_valid

    def test_id_at_max_length_accepted(self):
        """IDs at exactly max length are accepted."""
        valid_id = "a" * 64
        is_valid, _ = validate_path_segment(valid_id, "id", SAFE_ID_PATTERN)
        assert is_valid

    def test_agent_name_max_length_enforced(self):
        """Agent names exceeding 32 chars are rejected."""
        long_name = "a" * 33
        is_valid, _ = validate_agent_name(long_name)
        assert not is_valid

    def test_slug_max_length_enforced(self):
        """Slugs exceeding 128 chars are rejected."""
        long_slug = "a" * 129
        is_valid, _ = validate_debate_id(long_slug)
        assert not is_valid

    def test_empty_id_rejected(self):
        """Empty IDs are rejected."""
        is_valid, _ = validate_path_segment("", "id", SAFE_ID_PATTERN)
        assert not is_valid

    def test_whitespace_only_rejected(self):
        """Whitespace-only IDs are rejected."""
        is_valid, _ = validate_path_segment("   ", "id", SAFE_ID_PATTERN)
        assert not is_valid


# =============================================================================
# JSON Body Validation Security Tests
# =============================================================================


class TestJSONBodySecurity:
    """Tests for JSON body validation security."""

    def test_oversized_body_rejected(self):
        """Bodies exceeding max size are rejected."""
        large_body = b'{"data": "' + b'x' * (2 * 1024 * 1024) + b'"}'
        result = validate_json_body(large_body, max_size=1024 * 1024)
        assert not result.is_valid
        assert "too large" in result.error.lower()

    def test_empty_body_rejected(self):
        """Empty bodies are rejected."""
        result = validate_json_body(b'')
        assert not result.is_valid
        assert "empty" in result.error.lower()

    def test_invalid_json_rejected(self):
        """Malformed JSON is rejected."""
        result = validate_json_body(b'{invalid json}')
        assert not result.is_valid
        assert "invalid json" in result.error.lower()

    def test_invalid_utf8_rejected(self):
        """Invalid UTF-8 is rejected."""
        result = validate_json_body(b'\xff\xfe{"test": "value"}')
        assert not result.is_valid
        assert "utf-8" in result.error.lower()

    def test_deeply_nested_json_handled(self):
        """Deeply nested JSON doesn't crash."""
        nested = '{"a":' * 100 + '"value"' + '}' * 100
        result = validate_json_body(nested.encode())
        # Should either succeed or fail gracefully
        assert isinstance(result.is_valid, bool)


# =============================================================================
# String Field Validation Security Tests
# =============================================================================


class TestStringFieldSecurity:
    """Tests for string field validation security."""

    def test_string_field_length_enforced(self):
        """String fields exceeding max length are rejected."""
        data = {"field": "x" * 2000}
        result = validate_string_field(data, "field", max_length=1000)
        assert not result.is_valid
        assert "at most" in result.error

    def test_string_field_pattern_enforced(self):
        """String fields not matching pattern are rejected."""
        data = {"field": "invalid!@#$"}
        result = validate_string_field(data, "field", pattern=SAFE_ID_PATTERN)
        assert not result.is_valid
        assert "format" in result.error

    def test_non_string_type_rejected(self):
        """Non-string values are rejected for string fields."""
        data = {"field": 12345}
        result = validate_string_field(data, "field")
        assert not result.is_valid
        assert "must be a string" in result.error


# =============================================================================
# Sanitization Function Tests
# =============================================================================


class TestSanitizationFunctions:
    """Tests for sanitization functions."""

    def test_sanitize_string_strips_whitespace(self):
        """Sanitize strips leading/trailing whitespace."""
        assert sanitize_string("  hello  ") == "hello"

    def test_sanitize_string_truncates(self):
        """Sanitize truncates to max length."""
        result = sanitize_string("x" * 2000, max_length=100)
        assert len(result) == 100

    def test_sanitize_string_non_string_returns_empty(self):
        """Sanitize returns empty for non-string input."""
        assert sanitize_string(12345) == ""
        assert sanitize_string(None) == ""
        assert sanitize_string(["list"]) == ""

    def test_sanitize_id_valid(self):
        """Sanitize ID passes valid IDs."""
        assert sanitize_id("valid-id_123") == "valid-id_123"

    def test_sanitize_id_strips_whitespace(self):
        """Sanitize ID strips whitespace."""
        assert sanitize_id("  valid-id  ") == "valid-id"

    def test_sanitize_id_rejects_invalid(self):
        """Sanitize ID returns None for invalid IDs."""
        assert sanitize_id("invalid!@#") is None
        assert sanitize_id("../etc/passwd") is None
        assert sanitize_id("a" * 100) is None


# =============================================================================
# Genome ID with Dots Tests
# =============================================================================


class TestGenomeIDValidation:
    """Tests for genome ID validation (supports dots)."""

    def test_genome_id_with_version(self):
        """Genome IDs with version dots are accepted."""
        is_valid, _ = validate_genome_id("claude-3.5-sonnet")
        assert is_valid

    def test_genome_id_simple(self):
        """Simple genome IDs are accepted."""
        is_valid, _ = validate_genome_id("genome-v1")
        assert is_valid

    def test_genome_id_path_traversal_blocked(self):
        """Path traversal in genome IDs is blocked."""
        is_valid, _ = validate_genome_id("../genome")
        assert not is_valid

    def test_genome_id_cannot_start_with_dot(self):
        """Genome IDs cannot start with a dot."""
        is_valid, _ = validate_genome_id(".hidden")
        assert not is_valid


# =============================================================================
# Plugin Name Validation Tests
# =============================================================================


class TestPluginNameValidation:
    """Tests for plugin name validation."""

    def test_valid_plugin_name(self):
        """Valid plugin names are accepted."""
        is_valid, _ = validate_plugin_name("my-plugin")
        assert is_valid
        is_valid, _ = validate_plugin_name("plugin_v2")
        assert is_valid

    def test_plugin_name_command_injection(self):
        """Command injection in plugin names is blocked."""
        is_valid, _ = validate_plugin_name("plugin;rm -rf /")
        assert not is_valid

    def test_plugin_name_path_traversal(self):
        """Path traversal in plugin names is blocked."""
        is_valid, _ = validate_plugin_name("../malicious")
        assert not is_valid


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for validation."""

    def test_single_char_id_valid(self):
        """Single character IDs are valid."""
        is_valid, _ = validate_path_segment("a", "id", SAFE_ID_PATTERN)
        assert is_valid

    def test_numeric_id_valid(self):
        """Numeric IDs are valid."""
        is_valid, _ = validate_path_segment("12345", "id", SAFE_ID_PATTERN)
        assert is_valid

    def test_hyphen_underscore_allowed(self):
        """Hyphens and underscores are allowed."""
        is_valid, _ = validate_path_segment("my-id_123", "id", SAFE_ID_PATTERN)
        assert is_valid

    def test_uppercase_allowed(self):
        """Uppercase letters are allowed."""
        is_valid, _ = validate_path_segment("MyID123", "id", SAFE_ID_PATTERN)
        assert is_valid

    def test_spaces_not_allowed(self):
        """Spaces are not allowed in IDs."""
        is_valid, _ = validate_path_segment("my id", "id", SAFE_ID_PATTERN)
        assert not is_valid

    def test_newlines_not_allowed(self):
        """Newlines are not allowed in IDs."""
        is_valid, _ = validate_path_segment("my\nid", "id", SAFE_ID_PATTERN)
        assert not is_valid

    def test_tabs_not_allowed(self):
        """Tabs are not allowed in IDs."""
        is_valid, _ = validate_path_segment("my\tid", "id", SAFE_ID_PATTERN)
        assert not is_valid


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidationIntegration:
    """Integration tests for validation pipeline."""

    def test_full_validation_pipeline(self):
        """Full validation pipeline for a debate request."""
        body = b'{"task": "Discuss AI safety", "agents": ["claude", "gpt4"], "rounds": 3}'

        # Step 1: Validate JSON body
        result = validate_json_body(body)
        assert result.is_valid
        data = result.data

        # Step 2: Validate required fields
        assert "task" in data
        assert "agents" in data

        # Step 3: Validate agent names
        for agent in data["agents"]:
            is_valid, err = validate_agent_name(agent)
            assert is_valid, f"Agent validation failed: {err}"

    def test_malicious_request_fully_blocked(self):
        """Malicious request is blocked at every layer."""
        # Try to inject via agent name
        body = b'{"task": "test", "agents": ["claude; rm -rf /"]}'
        result = validate_json_body(body)
        assert result.is_valid  # JSON is valid

        # But agent validation should fail
        is_valid, _ = validate_agent_name(result.data["agents"][0])
        assert not is_valid
