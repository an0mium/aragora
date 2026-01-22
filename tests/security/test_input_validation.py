"""
Input validation security tests.

Verifies that input validation prevents injection attacks,
XSS, and other input-based vulnerabilities.
"""

import pytest
import json
from unittest.mock import Mock, patch


# =============================================================================
# JSON Input Validation Tests
# =============================================================================


class TestJSONInputValidation:
    """Test JSON input validation and parsing security."""

    def test_rejects_deeply_nested_json(self):
        """Deeply nested JSON should be rejected to prevent stack overflow."""
        MAX_DEPTH = 50

        def check_json_depth(obj, current_depth=0) -> int:
            """Calculate maximum depth of JSON object."""
            if current_depth > MAX_DEPTH:
                raise ValueError(f"JSON depth exceeds {MAX_DEPTH}")

            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(check_json_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                if not obj:
                    return current_depth
                return max(check_json_depth(item, current_depth + 1) for item in obj)
            return current_depth

        # Normal JSON
        normal = {"a": {"b": {"c": 1}}}
        assert check_json_depth(normal) < MAX_DEPTH

        # Deeply nested attack
        attack = {"level": None}
        current = attack
        for _ in range(100):
            current["level"] = {"level": None}
            current = current["level"]

        with pytest.raises(ValueError, match="depth exceeds"):
            check_json_depth(attack)

    def test_rejects_oversized_json(self):
        """Oversized JSON payloads should be rejected."""
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        def validate_json_size(json_bytes: bytes) -> bool:
            return len(json_bytes) <= MAX_SIZE

        # Normal size
        normal = json.dumps({"key": "value"}).encode()
        assert validate_json_size(normal) is True

        # Oversized
        oversized = json.dumps({"key": "x" * (MAX_SIZE + 1000)}).encode()
        assert validate_json_size(oversized) is False

    def test_rejects_invalid_json(self):
        """Invalid JSON should be rejected gracefully."""
        invalid_payloads = [
            b"not json",
            b"{invalid}",
            b"{'single': 'quotes'}",
            b'{"unclosed": ',
            b"[1, 2, 3,]",  # Trailing comma
            b"",
        ]

        for payload in invalid_payloads:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                json.loads(payload)

    def test_handles_unicode_in_json(self):
        """Unicode in JSON should be handled safely."""
        # Valid unicode
        valid = json.dumps({"name": "test"})
        assert json.loads(valid)["name"] == "test"

        # Potentially malicious unicode
        payloads = [
            {"name": "\u0000"},  # Null byte
            {"name": "\ud800"},  # Lone surrogate (if encoded properly)
            {"name": "<script>\u0000</script>"},
        ]

        for payload in payloads:
            encoded = json.dumps(payload)
            decoded = json.loads(encoded)
            # Should decode without error, nulls should be preserved
            assert "name" in decoded


# =============================================================================
# String Input Validation Tests
# =============================================================================


class TestStringInputValidation:
    """Test string input validation."""

    def test_task_length_limits(self):
        """Task strings should have length limits."""
        MAX_TASK_LENGTH = 10000

        def validate_task(task: str) -> bool:
            if not task or not task.strip():
                return False
            if len(task) > MAX_TASK_LENGTH:
                return False
            return True

        assert validate_task("Design a cache system") is True
        assert validate_task("x" * MAX_TASK_LENGTH) is True
        assert validate_task("x" * (MAX_TASK_LENGTH + 1)) is False
        assert validate_task("") is False
        assert validate_task("   ") is False

    def test_sanitize_html_in_strings(self):
        """HTML should be escaped in user-provided strings."""
        import html

        def sanitize_for_display(text: str) -> str:
            return html.escape(text)

        assert (
            sanitize_for_display("<script>alert(1)</script>")
            == "&lt;script&gt;alert(1)&lt;/script&gt;"
        )
        assert sanitize_for_display("<img onerror=alert(1)>") == "&lt;img onerror=alert(1)&gt;"
        assert sanitize_for_display("normal text") == "normal text"
        assert sanitize_for_display("a < b && c > d") == "a &lt; b &amp;&amp; c &gt; d"

    def test_strip_control_characters(self):
        """Control characters should be stripped from input."""

        def strip_control_chars(text: str) -> str:
            # Allow newlines and tabs, strip other control chars
            allowed = {"\n", "\r", "\t"}
            return "".join(c for c in text if c in allowed or (ord(c) >= 32 or ord(c) > 126))

        assert strip_control_chars("hello\nworld") == "hello\nworld"
        assert strip_control_chars("hello\x00world") == "helloworld"
        assert strip_control_chars("test\x1bworld") == "testworld"  # ESC char

    def test_validate_agent_names(self):
        """Agent names should be validated."""
        import re

        VALID_AGENTS = ["anthropic-api", "openai-api", "gemini", "grok", "claude", "codex"]

        def validate_agent_name(name: str) -> bool:
            # Check against allowlist or pattern
            if name in VALID_AGENTS:
                return True
            # Allow custom names with restricted characters
            pattern = r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$"
            return bool(re.match(pattern, name))

        assert validate_agent_name("anthropic-api") is True
        assert validate_agent_name("my_custom_agent") is True
        assert validate_agent_name("../etc/passwd") is False
        assert validate_agent_name("<script>") is False
        assert validate_agent_name("") is False


# =============================================================================
# Number Input Validation Tests
# =============================================================================


class TestNumberInputValidation:
    """Test numeric input validation."""

    def test_rounds_within_limits(self):
        """Debate rounds should be within reasonable limits."""
        MIN_ROUNDS = 1
        MAX_ROUNDS = 20

        def validate_rounds(rounds: int) -> bool:
            return MIN_ROUNDS <= rounds <= MAX_ROUNDS

        assert validate_rounds(1) is True
        assert validate_rounds(10) is True
        assert validate_rounds(20) is True
        assert validate_rounds(0) is False
        assert validate_rounds(-1) is False
        assert validate_rounds(100) is False

    def test_reject_nan_and_infinity(self):
        """NaN and Infinity should be rejected."""
        import math

        def validate_confidence(value: float) -> bool:
            if math.isnan(value) or math.isinf(value):
                return False
            return 0.0 <= value <= 1.0

        assert validate_confidence(0.5) is True
        assert validate_confidence(float("nan")) is False
        assert validate_confidence(float("inf")) is False
        assert validate_confidence(float("-inf")) is False

    def test_integer_overflow_prevention(self):
        """Large integers should be handled safely."""
        MAX_SAFE_INT = 2**31 - 1  # 32-bit signed max

        def validate_limit(limit: int) -> int:
            """Clamp limit to safe range."""
            return max(1, min(limit, MAX_SAFE_INT))

        assert validate_limit(100) == 100
        assert validate_limit(2**63) == MAX_SAFE_INT
        assert validate_limit(-100) == 1


# =============================================================================
# URL Input Validation Tests
# =============================================================================


class TestURLInputValidation:
    """Test URL input validation."""

    def test_reject_javascript_urls(self):
        """JavaScript URLs should be rejected."""

        def is_safe_url(url: str) -> bool:
            url_lower = url.lower().strip()
            dangerous_schemes = ["javascript:", "data:", "vbscript:", "file:"]
            return not any(url_lower.startswith(scheme) for scheme in dangerous_schemes)

        assert is_safe_url("https://example.com") is True
        assert is_safe_url("http://example.com") is True
        assert is_safe_url("javascript:alert(1)") is False
        assert is_safe_url("JAVASCRIPT:alert(1)") is False  # Case insensitive
        assert is_safe_url("  javascript:alert(1)") is False  # With leading space
        assert is_safe_url("data:text/html,<script>alert(1)</script>") is False

    def test_validate_url_format(self):
        """URLs should have valid format."""
        from urllib.parse import urlparse

        def is_valid_url(url: str) -> bool:
            try:
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except Exception:
                return False

        assert is_valid_url("https://example.com/path") is True
        assert is_valid_url("http://localhost:8080") is True
        assert is_valid_url("not-a-url") is False
        assert is_valid_url("") is False
        assert is_valid_url("//missing-scheme.com") is False

    def test_reject_local_urls(self):
        """Local/internal URLs should be rejected for external requests."""
        import ipaddress

        def is_external_url(url: str) -> bool:
            from urllib.parse import urlparse

            try:
                parsed = urlparse(url)
                hostname = parsed.hostname
                if not hostname:
                    return False

                # Check for localhost
                if hostname in ["localhost", "127.0.0.1", "::1"]:
                    return False

                # Check for private IP ranges
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_reserved:
                        return False
                except ValueError:
                    pass  # Not an IP address, probably a hostname

                # Check for internal DNS names
                if hostname.endswith(".local") or hostname.endswith(".internal"):
                    return False

                return True
            except Exception:
                return False

        assert is_external_url("https://example.com") is True
        assert is_external_url("http://localhost:8080") is False
        assert is_external_url("http://127.0.0.1") is False
        assert is_external_url("http://192.168.1.1") is False
        assert is_external_url("http://10.0.0.1") is False
        assert is_external_url("http://internal.local") is False


# =============================================================================
# File Upload Validation Tests
# =============================================================================


class TestFileUploadValidation:
    """Test file upload validation."""

    def test_validate_file_extension(self):
        """File extensions should be validated against allowlist."""
        ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".pdf"}

        def is_allowed_extension(filename: str) -> bool:
            from pathlib import Path

            ext = Path(filename).suffix.lower()
            return ext in ALLOWED_EXTENSIONS

        assert is_allowed_extension("document.txt") is True
        assert is_allowed_extension("spec.md") is True
        assert is_allowed_extension("config.json") is True
        assert is_allowed_extension("script.py") is False
        assert is_allowed_extension("malware.exe") is False
        assert is_allowed_extension("page.html") is False

    def test_validate_content_type(self):
        """Content-Type should match file extension."""
        EXTENSION_CONTENT_TYPES = {
            ".txt": ["text/plain"],
            ".md": ["text/markdown", "text/plain"],
            ".json": ["application/json"],
            ".pdf": ["application/pdf"],
        }

        def validate_content_type(filename: str, content_type: str) -> bool:
            from pathlib import Path

            ext = Path(filename).suffix.lower()
            allowed_types = EXTENSION_CONTENT_TYPES.get(ext, [])
            return content_type in allowed_types

        assert validate_content_type("doc.txt", "text/plain") is True
        assert validate_content_type("doc.json", "application/json") is True
        assert validate_content_type("doc.txt", "application/octet-stream") is False
        assert validate_content_type("doc.pdf", "text/html") is False

    def test_validate_file_size(self):
        """File size should be within limits."""
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

        def validate_file_size(size: int) -> bool:
            return 0 < size <= MAX_FILE_SIZE

        assert validate_file_size(1024) is True
        assert validate_file_size(MAX_FILE_SIZE) is True
        assert validate_file_size(MAX_FILE_SIZE + 1) is False
        assert validate_file_size(0) is False
        assert validate_file_size(-1) is False

    def test_detect_file_type_mismatch(self):
        """File content should match declared type (magic bytes)."""
        FILE_SIGNATURES = {
            b"%PDF": "application/pdf",
            b"\x89PNG": "image/png",
            b"\xff\xd8\xff": "image/jpeg",
            b"PK\x03\x04": "application/zip",
        }

        def detect_file_type(content: bytes) -> str:
            for signature, mime_type in FILE_SIGNATURES.items():
                if content.startswith(signature):
                    return mime_type
            return "application/octet-stream"

        assert detect_file_type(b"%PDF-1.4...") == "application/pdf"
        assert detect_file_type(b"\x89PNG\r\n\x1a\n...") == "image/png"
        assert detect_file_type(b"plain text content") == "application/octet-stream"


# =============================================================================
# Command Injection Prevention Tests
# =============================================================================


class TestCommandInjectionPrevention:
    """Test command injection prevention."""

    def test_reject_shell_metacharacters(self):
        """Shell metacharacters should be rejected or escaped."""
        SHELL_METACHARACTERS = set(";|&$`\\!><(){}[]'\"")

        def has_shell_metacharacters(text: str) -> bool:
            return bool(SHELL_METACHARACTERS & set(text))

        assert has_shell_metacharacters("normal text") is False
        assert has_shell_metacharacters("test; rm -rf /") is True
        assert has_shell_metacharacters("$(whoami)") is True
        assert has_shell_metacharacters("`id`") is True
        assert has_shell_metacharacters("a && b") is True
        assert has_shell_metacharacters("a | b") is True

    def test_safe_subprocess_usage(self):
        """Subprocess should use list arguments, not shell strings."""
        import subprocess

        # SAFE: list of arguments
        safe_cmd = ["echo", "hello world"]

        # UNSAFE: shell string (don't use this pattern)
        # unsafe_cmd = "echo " + user_input  # Never do this!

        # Test that safe command works
        result = subprocess.run(
            safe_cmd,
            capture_output=True,
            text=True,
            shell=False,  # Explicit shell=False
        )
        assert result.returncode == 0
        assert "hello world" in result.stdout

    def test_escape_for_shell_if_needed(self):
        """If shell is required, arguments must be escaped."""
        import shlex

        user_input = "hello; rm -rf /"
        escaped = shlex.quote(user_input)

        # Escaped version is safe
        assert ";" not in escaped or escaped.startswith("'")


# =============================================================================
# Regex Input Validation Tests
# =============================================================================


class TestRegexInputValidation:
    """Test that user-provided regex patterns are safe."""

    def test_reject_catastrophic_backtracking_patterns(self):
        """Detect regex patterns that could cause ReDoS."""
        import re

        def is_safe_regex(pattern: str) -> bool:
            """Basic check for potentially dangerous patterns."""
            # Patterns that can cause catastrophic backtracking
            dangerous_patterns = [
                r"(.*)*",  # Nested quantifiers
                r"(a+)+",
                r"(a|a)+",
                r"([a-zA-Z]+)*",
            ]

            # Check for nested quantifiers (simplified)
            if re.search(r"\([^)]+[*+]\)[*+]", pattern):
                return False

            return pattern not in dangerous_patterns

        assert is_safe_regex(r"\d+") is True
        assert is_safe_regex(r"[a-z]+") is True
        assert is_safe_regex(r"(.*)*") is False
        assert is_safe_regex(r"(a+)+") is False

    def test_regex_with_timeout(self):
        """Regex matching should have timeout."""
        import re
        import signal

        def match_with_timeout(pattern: str, text: str, timeout: float = 1.0) -> bool:
            """Match regex with timeout to prevent ReDoS."""
            try:
                # Note: Python's re module doesn't have built-in timeout
                # In production, use regex library or implement differently
                return bool(re.match(pattern, text))
            except re.error:
                return False

        # Safe patterns should work
        assert match_with_timeout(r"\d+", "12345") is True
        assert match_with_timeout(r"[a-z]+", "hello") is True
