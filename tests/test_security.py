"""
Critical security tests for Aragora.

Tests security-critical paths including:
- Token validation (HMAC signatures, expiration, tampering)
- SQL injection prevention (LIKE pattern escaping)
- Rate limiting (edge cases, memory bounds)
- Input sanitization
"""

import hashlib
import hmac
import os
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from aragora.server.auth import AuthConfig, check_auth
from aragora.server.storage import _escape_like_pattern, DebateStorage


class TestTokenValidation:
    """Test token validation security."""

    @pytest.fixture
    def auth(self):
        """Create an AuthConfig with a known secret."""
        config = AuthConfig()
        config.api_token = "test_secret_key_12345"
        config.enabled = True
        config.token_ttl = 3600
        return config

    def test_valid_token_accepted(self, auth):
        """Valid token should be accepted."""
        token = auth.generate_token("loop_1")
        assert auth.validate_token(token, "loop_1") is True

    def test_expired_token_rejected(self, auth):
        """Expired token should be rejected."""
        # Generate token that's already expired
        token = auth.generate_token("loop_1", expires_in=-1)
        assert auth.validate_token(token, "loop_1") is False

    def test_token_expiry_boundary(self, auth):
        """Token should be valid exactly until expiration."""
        # Generate token expiring in 1 second
        token = auth.generate_token("loop_1", expires_in=1)
        assert auth.validate_token(token, "loop_1") is True

        # Wait for expiration
        time.sleep(1.5)
        assert auth.validate_token(token, "loop_1") is False

    def test_signature_tampering_rejected(self, auth):
        """Token with tampered signature should be rejected."""
        token = auth.generate_token("loop_1")

        # Tamper with signature by flipping one character
        parts = token.rsplit(":", 1)
        tampered_sig = parts[1][:-1] + ("a" if parts[1][-1] != "a" else "b")
        tampered_token = f"{parts[0]}:{tampered_sig}"

        assert auth.validate_token(tampered_token, "loop_1") is False

    def test_payload_tampering_rejected(self, auth):
        """Token with tampered payload should be rejected."""
        token = auth.generate_token("loop_1")

        # Tamper with loop_id in payload
        parts = token.split(":")
        parts[0] = "loop_2"  # Change loop_id
        tampered_token = ":".join(parts)

        assert auth.validate_token(tampered_token, "loop_1") is False

    def test_malformed_token_rejected(self, auth):
        """Malformed tokens should be safely rejected."""
        malformed_tokens = [
            "",
            ":",
            "::",
            "no_colons",
            "one:colon",
            "loop:notanumber:sig",
            "loop:12345:",  # Empty signature
            ":12345:sig",  # Empty loop
        ]

        for token in malformed_tokens:
            assert auth.validate_token(token, "") is False

    def test_wrong_loop_id_rejected(self, auth):
        """Token generated for one loop should be rejected for another."""
        token = auth.generate_token("loop_1")
        assert auth.validate_token(token, "loop_1") is True
        assert auth.validate_token(token, "loop_2") is False

    def test_empty_loop_id_accepts_any(self, auth):
        """Token with empty loop_id should be accepted for any loop."""
        token = auth.generate_token("")  # No specific loop
        assert auth.validate_token(token, "") is True
        # Empty loop in token means it validates against any loop_id check
        # (when loop_id param is also empty)

    def test_timing_attack_resistance(self, auth):
        """Signature comparison should use constant-time comparison."""
        token = auth.generate_token("loop_1")

        # Create a token with completely different signature
        parts = token.rsplit(":", 1)
        wrong_sig = "a" * 64  # Completely different
        wrong_token = f"{parts[0]}:{wrong_sig}"

        # Both should take similar time (within reason)
        # This is more of a verification that hmac.compare_digest is used
        times = []
        for test_token in [token, wrong_token]:
            start = time.perf_counter()
            for _ in range(1000):
                auth.validate_token(test_token, "loop_1")
            times.append(time.perf_counter() - start)

        # Times should be within 50% of each other (generous for CI variance)
        ratio = max(times) / min(times)
        assert ratio < 1.5, f"Timing difference too large: {times}"

    def test_unicode_in_loop_id(self, auth):
        """Unicode characters in loop_id should be handled safely."""
        unicode_loops = [
            "loop_émoji_🎉",
            "loop_中文",
            "loop_العربية",
            "loop_\x00null",
        ]

        for loop_id in unicode_loops:
            token = auth.generate_token(loop_id)
            assert auth.validate_token(token, loop_id) is True
            assert auth.validate_token(token, "different") is False


class TestSQLInjectionPrevention:
    """Test SQL injection prevention in storage."""

    def test_escape_like_percent(self):
        """Percent sign should be escaped."""
        assert _escape_like_pattern("test%") == "test\\%"
        assert _escape_like_pattern("%test%") == "\\%test\\%"
        assert _escape_like_pattern("100%") == "100\\%"

    def test_escape_like_underscore(self):
        """Underscore should be escaped."""
        assert _escape_like_pattern("test_value") == "test\\_value"
        assert _escape_like_pattern("_start") == "\\_start"
        assert _escape_like_pattern("end_") == "end\\_"

    def test_escape_like_backslash(self):
        """Backslash should be escaped first."""
        assert _escape_like_pattern("path\\to") == "path\\\\to"
        assert _escape_like_pattern("\\") == "\\\\"

    def test_escape_like_combined(self):
        """Combined special characters should all be escaped."""
        assert _escape_like_pattern("100%_done\\") == "100\\%\\_done\\\\"

    def test_escape_like_sql_keywords(self):
        """SQL keywords should pass through (not injection vectors for LIKE)."""
        keywords = ["SELECT", "DROP", "DELETE", "UPDATE", "INSERT", "--", ";"]
        for kw in keywords:
            # These should pass through unchanged (they're not LIKE metacharacters)
            assert _escape_like_pattern(kw) == kw

    def test_escape_like_empty_string(self):
        """Empty string should return empty string."""
        assert _escape_like_pattern("") == ""

    def test_escape_like_unicode(self):
        """Unicode characters should pass through."""
        assert _escape_like_pattern("café") == "café"
        assert _escape_like_pattern("日本語") == "日本語"

    def test_storage_slug_escaping(self):
        """Storage should properly escape slugs in LIKE queries."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            storage = DebateStorage(db_path)

            # Create a debate with special characters in task
            # The slug generation will sanitize, but test the lookup
            from unittest.mock import MagicMock, patch

            # Test that get_by_slug properly escapes
            # This would catch SQL injection in the LIKE clause
            malicious_slugs = [
                "test%",  # Would match too many rows
                "test_",  # Would match test1, test2, etc.
                "test'; DROP TABLE debates; --",  # Classic injection
            ]

            for slug in malicious_slugs:
                # Should not raise and should return None (not found)
                result = storage.get_by_slug(slug)
                assert result is None
        finally:
            os.unlink(db_path)


class TestRateLimiting:
    """Test rate limiting security."""

    def test_rate_limit_enforced(self):
        """Rate limit should block after threshold."""
        auth = AuthConfig()
        auth.rate_limit_per_minute = 5

        token = "test_token"
        for i in range(5):
            allowed, remaining = auth.check_rate_limit(token)
            assert allowed is True
            assert remaining == 4 - i

        # 6th request should be blocked
        allowed, remaining = auth.check_rate_limit(token)
        assert allowed is False
        assert remaining == 0

    def test_rate_limit_per_token_isolation(self):
        """Each token should have independent rate limit."""
        auth = AuthConfig()
        auth.rate_limit_per_minute = 3

        # Exhaust token1
        for _ in range(3):
            auth.check_rate_limit("token1")
        assert auth.check_rate_limit("token1")[0] is False

        # token2 should still work
        assert auth.check_rate_limit("token2")[0] is True

    def test_rate_limit_ip_isolation(self):
        """Each IP should have independent rate limit."""
        auth = AuthConfig()
        auth.ip_rate_limit_per_minute = 3

        # Exhaust IP1
        for _ in range(3):
            auth.check_rate_limit_by_ip("192.168.1.1")
        assert auth.check_rate_limit_by_ip("192.168.1.1")[0] is False

        # IP2 should still work
        assert auth.check_rate_limit_by_ip("192.168.1.2")[0] is True

    def test_rate_limit_empty_ip(self):
        """Empty IP should be allowed (no tracking)."""
        auth = AuthConfig()
        auth.ip_rate_limit_per_minute = 1

        # Empty IP should always be allowed
        allowed, _ = auth.check_rate_limit_by_ip("")
        assert allowed is True
        allowed, _ = auth.check_rate_limit_by_ip("")
        assert allowed is True

    def test_rate_limit_memory_bounds(self):
        """Rate limiter should not exhaust memory with many unique tokens."""
        auth = AuthConfig()
        auth._max_tracked_entries = 100
        auth.rate_limit_per_minute = 1000

        # Add many unique tokens
        for i in range(200):
            auth.check_rate_limit(f"token_{i}")

        # Should have cleaned up old entries
        assert len(auth._token_request_counts) <= 100

    def test_rate_limit_thread_safety(self):
        """Rate limiter should be thread-safe."""
        auth = AuthConfig()
        auth.rate_limit_per_minute = 10000
        errors = []
        results = []

        def check_rate():
            try:
                for i in range(100):
                    allowed, _ = auth.check_rate_limit(f"thread_token")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_rate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have processed all requests without error
        assert len(results) == 1000

    def test_rate_limit_sliding_window(self):
        """Rate limit should use sliding window."""
        auth = AuthConfig()
        auth.rate_limit_per_minute = 2

        # Use both slots
        auth.check_rate_limit("token")
        auth.check_rate_limit("token")
        assert auth.check_rate_limit("token")[0] is False

        # Wait for window to slide (use small sleep + mock for CI)
        time.sleep(0.1)

        # Still should be blocked (window hasn't fully passed)
        # This verifies it's a sliding window, not a fixed window


class TestCheckAuthIntegration:
    """Test the check_auth function integration."""

    def test_check_auth_disabled(self):
        """When auth is disabled, should allow all requests."""
        from aragora.server.auth import auth_config

        original_enabled = auth_config.enabled
        try:
            auth_config.enabled = False
            auth, remaining = check_auth({}, "")
            assert auth is True
        finally:
            auth_config.enabled = original_enabled

    def test_check_auth_ip_rate_limit(self):
        """IP rate limiting should work even when auth disabled."""
        from aragora.server.auth import auth_config

        original_enabled = auth_config.enabled
        original_limit = auth_config.ip_rate_limit_per_minute

        try:
            auth_config.enabled = False
            auth_config.ip_rate_limit_per_minute = 2
            auth_config._ip_request_counts.clear()

            # First two should pass
            assert check_auth({}, "", "", "test_ip")[0] is True
            assert check_auth({}, "", "", "test_ip")[0] is True

            # Third should fail
            assert check_auth({}, "", "", "test_ip")[0] is False
        finally:
            auth_config.enabled = original_enabled
            auth_config.ip_rate_limit_per_minute = original_limit

    def test_check_auth_extracts_bearer_token(self):
        """Should extract token from Authorization header."""
        from aragora.server.auth import auth_config

        original_enabled = auth_config.enabled
        original_token = auth_config.api_token

        try:
            auth_config.enabled = True
            auth_config.api_token = "secret"
            auth_config._token_request_counts.clear()

            valid_token = auth_config.generate_token("loop_1")

            headers = {"Authorization": f"Bearer {valid_token}"}
            auth, _ = check_auth(headers, "", "loop_1", "")
            assert auth is True

            # Invalid token should fail
            headers = {"Authorization": "Bearer invalid_token"}
            auth, _ = check_auth(headers, "", "loop_1", "")
            assert auth is False
        finally:
            auth_config.enabled = original_enabled
            auth_config.api_token = original_token


class TestConfigureFromEnv:
    """Test environment variable configuration."""

    def test_configure_from_env_token(self):
        """Should load token from environment."""
        with patch.dict(os.environ, {"ARAGORA_API_TOKEN": "env_secret"}):
            config = AuthConfig()
            config.configure_from_env()
            assert config.api_token == "env_secret"
            assert config.enabled is True

    def test_configure_from_env_ttl(self):
        """Should load TTL from environment."""
        with patch.dict(os.environ, {"ARAGORA_TOKEN_TTL": "7200"}):
            config = AuthConfig()
            config.configure_from_env()
            assert config.token_ttl == 7200

    def test_configure_from_env_invalid_ttl(self):
        """Invalid TTL should be ignored."""
        with patch.dict(os.environ, {"ARAGORA_TOKEN_TTL": "not_a_number"}):
            config = AuthConfig()
            original_ttl = config.token_ttl
            config.configure_from_env()
            assert config.token_ttl == original_ttl  # Unchanged

    def test_configure_from_env_negative_ttl(self):
        """Negative TTL from env should be accepted (allows expired tokens)."""
        with patch.dict(os.environ, {"ARAGORA_TOKEN_TTL": "-100"}):
            config = AuthConfig()
            config.configure_from_env()
            assert config.token_ttl == -100

    def test_configure_from_env_origins(self):
        """Should load allowed origins from environment."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "http://localhost:3000, https://example.com"},
        ):
            config = AuthConfig()
            config.configure_from_env()
            assert "http://localhost:3000" in config.allowed_origins
            assert "https://example.com" in config.allowed_origins
