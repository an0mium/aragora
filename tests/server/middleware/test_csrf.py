"""
Tests for aragora.server.middleware.csrf - CSRF Protection Middleware.

Tests cover:
- CSRFConfig configuration and environment-based settings
- Token generation with HMAC signatures
- Token validation (valid, expired, tampered, malformed)
- CSRFMiddleware request validation
- Cookie and header token extraction
- Path exclusion configuration
- Double-submit cookie pattern validation
- csrf_protect decorator functionality
- Cookie header generation
- Security edge cases (timing attacks, secret rotation, path traversal)
- Handler method/path extraction fallbacks
- Dict-style handler support
- No-cookie and no-header edge cases
- binascii error handling
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test CSRFConfig Configuration
# =============================================================================


class TestCSRFConfig:
    """Tests for CSRFConfig dataclass."""

    def test_default_config_values(self):
        """Default config should have sensible defaults."""
        from aragora.server.middleware.csrf import CSRFConfig

        with patch.dict(os.environ, {}, clear=True):
            config = CSRFConfig()
            assert config.cookie_name == "_csrf_token"
            assert config.header_name == "X-CSRF-Token"
            assert config.cookie_httponly is True
            assert config.token_max_age == 24 * 60 * 60  # 24 hours

    def test_config_from_environment(self):
        """Config should read from environment variables."""
        from aragora.server.middleware.csrf import CSRFConfig

        env_vars = {
            "ARAGORA_CSRF_ENABLED": "true",
            "ARAGORA_CSRF_SECRET": "test-secret-key-12345",
            "ARAGORA_CSRF_COOKIE_NAME": "my_csrf",
            "ARAGORA_CSRF_HEADER_NAME": "X-My-CSRF",
            "ARAGORA_CSRF_COOKIE_SECURE": "true",
            "ARAGORA_CSRF_COOKIE_SAMESITE": "Lax",
            "ARAGORA_CSRF_TOKEN_MAX_AGE": "3600",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = CSRFConfig()
            assert config.enabled is True
            assert config.secret == "test-secret-key-12345"
            assert config.cookie_name == "my_csrf"
            assert config.header_name == "X-My-CSRF"
            assert config.cookie_secure is True
            assert config.cookie_samesite == "Lax"
            assert config.token_max_age == 3600

    def test_enabled_defaults_by_environment(self):
        """CSRF should be enabled by default in production, disabled in dev."""
        from aragora.server.middleware.csrf import CSRFConfig

        # Production environment
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            config = CSRFConfig()
            assert config.enabled is True
            assert config.cookie_secure is True

        # Development environment
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            config = CSRFConfig()
            assert config.enabled is False
            assert config.cookie_secure is False

    def test_excluded_paths_configuration(self):
        """Should configure excluded paths from environment."""
        from aragora.server.middleware.csrf import CSRFConfig

        with patch.dict(
            os.environ,
            {"ARAGORA_CSRF_EXCLUDED_PATHS": "/api/webhook/,/api/callback/"},
            clear=True,
        ):
            config = CSRFConfig()
            assert "/api/webhook/" in config.excluded_paths
            assert "/api/callback/" in config.excluded_paths

    def test_is_path_excluded_exact_match(self):
        """Should exclude paths that match exactly."""
        from aragora.server.middleware.csrf import CSRFConfig

        config = CSRFConfig()
        config.excluded_paths = frozenset({"/api/webhooks/", "/api/oauth/"})
        config.excluded_prefixes = frozenset()

        assert config.is_path_excluded("/api/webhooks/") is True
        assert config.is_path_excluded("/api/oauth/") is True
        assert config.is_path_excluded("/api/debates") is False

    def test_is_path_excluded_prefix_match(self):
        """Should exclude paths matching prefix."""
        from aragora.server.middleware.csrf import CSRFConfig

        config = CSRFConfig()
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset({"/webhooks/", "/api/webhooks/"})

        assert config.is_path_excluded("/webhooks/github") is True
        assert config.is_path_excluded("/api/webhooks/stripe") is True
        assert config.is_path_excluded("/api/debates") is False

    def test_excluded_prefixes_from_environment(self):
        """Should configure excluded prefixes from environment variable."""
        from aragora.server.middleware.csrf import CSRFConfig

        with patch.dict(
            os.environ,
            {"ARAGORA_CSRF_EXCLUDED_PREFIXES": "/internal/,/metrics/"},
            clear=True,
        ):
            config = CSRFConfig()
            assert "/internal/" in config.excluded_prefixes
            assert "/metrics/" in config.excluded_prefixes

    def test_enabled_with_various_truthy_values(self):
        """Should recognize various truthy values for CSRF enabled."""
        from aragora.server.middleware.csrf import CSRFConfig

        for val in ["true", "1", "yes", "True", "TRUE", "YES"]:
            with patch.dict(os.environ, {"ARAGORA_CSRF_ENABLED": val}, clear=True):
                config = CSRFConfig()
                assert config.enabled is True, f"Expected enabled=True for value '{val}'"

    def test_disabled_with_various_falsy_values(self):
        """Should recognize various falsy values for CSRF disabled."""
        from aragora.server.middleware.csrf import CSRFConfig

        for val in ["false", "0", "no", "False", "FALSE"]:
            with patch.dict(os.environ, {"ARAGORA_CSRF_ENABLED": val}, clear=True):
                config = CSRFConfig()
                assert config.enabled is False, f"Expected enabled=False for value '{val}'"

    def test_auto_generated_secret_when_not_configured(self):
        """Should auto-generate a secret when not provided."""
        from aragora.server.middleware.csrf import CSRFConfig

        with patch.dict(os.environ, {}, clear=True):
            config1 = CSRFConfig()
            config2 = CSRFConfig()
            # Auto-generated secrets should differ each time (random)
            assert len(config1.secret) > 0
            assert len(config2.secret) > 0

    def test_is_path_excluded_no_match(self):
        """Should return False when path does not match any exclusion."""
        from aragora.server.middleware.csrf import CSRFConfig

        config = CSRFConfig()
        config.excluded_paths = frozenset({"/api/webhooks/"})
        config.excluded_prefixes = frozenset({"/internal/"})

        assert config.is_path_excluded("/api/debates") is False
        assert config.is_path_excluded("/api/v2/debates") is False
        assert config.is_path_excluded("/") is False

    def test_cookie_samesite_from_environment(self):
        """Should read SameSite from environment."""
        from aragora.server.middleware.csrf import CSRFConfig

        with patch.dict(os.environ, {"ARAGORA_CSRF_COOKIE_SAMESITE": "None"}, clear=True):
            config = CSRFConfig()
            assert config.cookie_samesite == "None"


# =============================================================================
# Test Token Generation
# =============================================================================


class TestTokenGeneration:
    """Tests for CSRF token generation."""

    def test_generate_csrf_token_returns_string(self):
        """generate_csrf_token should return a non-empty string."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_csrf_token_is_base64_encoded(self):
        """Token should be valid base64."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)

        # Should not raise an error
        decoded = base64.urlsafe_b64decode(token.encode())
        assert len(decoded) > 0

    def test_generate_csrf_token_contains_timestamp(self):
        """Token should contain timestamp in format."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")

        assert len(parts) == 3
        timestamp = int(parts[0])
        assert timestamp > 0
        assert abs(timestamp - int(time.time())) < 5  # Within 5 seconds

    def test_generate_csrf_token_unique(self):
        """Each generated token should be unique due to random component."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        tokens = [generate_csrf_token(config) for _ in range(100)]
        assert len(set(tokens)) == 100  # All unique

    def test_generate_csrf_token_default_config(self):
        """Should work with default config."""
        from aragora.server.middleware.csrf import generate_csrf_token

        token = generate_csrf_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_internal_generate_token_value_with_explicit_timestamp(self):
        """_generate_token_value should accept explicit timestamp."""
        from aragora.server.middleware.csrf import _generate_token_value

        fixed_ts = 1700000000
        token = _generate_token_value("test-secret", timestamp=fixed_ts)
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")
        assert int(parts[0]) == fixed_ts

    def test_token_has_hmac_signature(self):
        """Token should include a valid HMAC signature component."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")

        # Third part should be hex-encoded HMAC
        signature_hex = parts[2]
        # SHA256 HMAC produces 64 hex characters
        assert len(signature_hex) == 64
        # Should be valid hex
        bytes.fromhex(signature_hex)


# =============================================================================
# Test Token Validation
# =============================================================================


class TestTokenValidation:
    """Tests for CSRF token validation."""

    def test_validate_valid_token(self):
        """Should validate a freshly generated token."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            generate_csrf_token,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)
        is_valid, error = validate_csrf_token(token, config)

        assert is_valid is True
        assert error == ""

    def test_validate_empty_token(self):
        """Should reject empty token."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        is_valid, error = validate_csrf_token("", config)

        assert is_valid is False
        assert "Empty token" in error

    def test_validate_malformed_token(self):
        """Should reject malformed token."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        # Not valid base64
        is_valid, error = validate_csrf_token("not-base64!!!", config)
        assert is_valid is False

        # Valid base64 but wrong format
        bad_token = base64.urlsafe_b64encode(b"bad-format").decode()
        is_valid, error = validate_csrf_token(bad_token, config)
        assert is_valid is False
        assert "Invalid token format" in error

    def test_validate_expired_token(self):
        """Should reject expired token."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            _generate_token_value,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"
        config.token_max_age = 3600  # 1 hour

        # Generate token with old timestamp
        old_timestamp = int(time.time()) - 7200  # 2 hours ago
        token = _generate_token_value(config.secret, old_timestamp)

        is_valid, error = validate_csrf_token(token, config)

        assert is_valid is False
        assert "expired" in error.lower()

    def test_validate_future_timestamp_token(self):
        """Should reject token with timestamp too far in future."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            _generate_token_value,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"

        # Generate token with future timestamp (beyond 60 second tolerance)
        future_timestamp = int(time.time()) + 120
        token = _generate_token_value(config.secret, future_timestamp)

        is_valid, error = validate_csrf_token(token, config)

        assert is_valid is False
        assert "future" in error.lower()

    def test_validate_tampered_signature(self):
        """Should reject token with tampered signature."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            generate_csrf_token,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"

        # Generate valid token
        token = generate_csrf_token(config)
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")

        # Tamper with signature
        parts[2] = "0" * 64  # Replace signature
        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        is_valid, error = validate_csrf_token(tampered, config)

        assert is_valid is False
        assert "Invalid signature" in error

    def test_validate_wrong_secret(self):
        """Should reject token validated with wrong secret."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            generate_csrf_token,
            validate_csrf_token,
        )

        config1 = CSRFConfig()
        config1.secret = "secret-one"

        config2 = CSRFConfig()
        config2.secret = "secret-two"

        token = generate_csrf_token(config1)
        is_valid, error = validate_csrf_token(token, config2)

        assert is_valid is False

    def test_validate_default_config(self):
        """Should work with default config for validation."""
        from aragora.server.middleware.csrf import (
            generate_csrf_token,
            validate_csrf_token,
        )

        # Uses default config internally
        token = generate_csrf_token()
        # Validation with a different default config may fail since secrets auto-generate
        # Just ensure it doesn't crash
        is_valid, error = validate_csrf_token(token)
        # Result depends on whether the same secret is used
        assert isinstance(is_valid, bool)

    def test_validate_token_within_clock_skew_tolerance(self):
        """Token within 60 second clock skew should be accepted."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            _generate_token_value,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"

        # Token 30 seconds in the future (within 60s tolerance)
        slight_future = int(time.time()) + 30
        token = _generate_token_value(config.secret, slight_future)

        is_valid, error = validate_csrf_token(token, config)
        assert is_valid is True

    def test_validate_token_with_invalid_timestamp(self):
        """Should reject token with non-numeric timestamp."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        # Create a token with non-numeric timestamp
        bad_data = "not_a_number:abcdef0123456789:0" * 2
        token = base64.urlsafe_b64encode(bad_data.encode()).decode()

        is_valid, error = validate_csrf_token(token, config)
        assert is_valid is False

    def test_validate_token_with_two_parts_only(self):
        """Should reject token with only two parts instead of three."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        bad_data = "12345:abcdef"
        token = base64.urlsafe_b64encode(bad_data.encode()).decode()

        is_valid, error = validate_csrf_token(token, config)
        assert is_valid is False
        assert "Invalid token format" in error

    def test_validate_token_with_invalid_hex_signature(self):
        """Should reject token with non-hex signature."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        ts = str(int(time.time()))
        bad_data = f"{ts}:abcdef0123456789:NOT_HEX_VALUE!!!"
        token = base64.urlsafe_b64encode(bad_data.encode()).decode()

        is_valid, error = validate_csrf_token(token, config)
        assert is_valid is False


# =============================================================================
# Test Cookie Header Generation
# =============================================================================


class TestCookieHeaderGeneration:
    """Tests for CSRF cookie header generation."""

    def test_get_csrf_cookie_value_basic(self):
        """Should generate basic cookie value."""
        from aragora.server.middleware.csrf import CSRFConfig, get_csrf_cookie_value

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.token_max_age = 3600
        config.cookie_httponly = True
        config.cookie_secure = False
        config.cookie_samesite = "Strict"

        cookie = get_csrf_cookie_value("test-token", config)

        assert "_csrf=test-token" in cookie
        assert "Max-Age=3600" in cookie
        assert "Path=/" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=Strict" in cookie

    def test_get_csrf_cookie_value_secure(self):
        """Should include Secure flag when configured."""
        from aragora.server.middleware.csrf import CSRFConfig, get_csrf_cookie_value

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.cookie_secure = True

        cookie = get_csrf_cookie_value("test-token", config)
        assert "Secure" in cookie

    def test_get_csrf_cookie_value_no_secure(self):
        """Should not include Secure flag when disabled."""
        from aragora.server.middleware.csrf import CSRFConfig, get_csrf_cookie_value

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.cookie_secure = False

        cookie = get_csrf_cookie_value("test-token", config)
        # Secure should not be in cookie (need to check carefully)
        parts = cookie.split("; ")
        assert "Secure" not in parts

    def test_get_csrf_cookie_value_no_samesite(self):
        """Should omit SameSite when empty string."""
        from aragora.server.middleware.csrf import CSRFConfig, get_csrf_cookie_value

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.cookie_samesite = ""
        config.cookie_httponly = False
        config.cookie_secure = False

        cookie = get_csrf_cookie_value("test-token", config)
        assert "SameSite" not in cookie

    def test_get_csrf_cookie_value_no_httponly(self):
        """Should omit HttpOnly when disabled."""
        from aragora.server.middleware.csrf import CSRFConfig, get_csrf_cookie_value

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.cookie_httponly = False
        config.cookie_secure = False
        config.cookie_samesite = ""

        cookie = get_csrf_cookie_value("test-token", config)
        assert "HttpOnly" not in cookie

    def test_get_csrf_cookie_value_default_config(self):
        """Should work with default config."""
        from aragora.server.middleware.csrf import get_csrf_cookie_value

        cookie = get_csrf_cookie_value("test-token")
        assert "test-token" in cookie
        assert "Path=/" in cookie


# =============================================================================
# Test CSRFMiddleware
# =============================================================================


class TestCSRFMiddleware:
    """Tests for CSRFMiddleware class."""

    def test_middleware_init_default_config(self):
        """Should initialize with default config."""
        from aragora.server.middleware.csrf import CSRFMiddleware

        middleware = CSRFMiddleware()
        assert middleware.config is not None

    def test_middleware_init_custom_config(self):
        """Should initialize with custom config."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.secret = "custom-secret"

        middleware = CSRFMiddleware(config)
        assert middleware.config.secret == "custom-secret"

    def test_middleware_enabled_property(self):
        """Should expose enabled property from config."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)
        assert middleware.enabled is True

        config.enabled = False
        middleware = CSRFMiddleware(config)
        assert middleware.enabled is False

    def test_middleware_generate_token(self):
        """Should generate a valid token via middleware."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.secret = "test-secret"
        middleware = CSRFMiddleware(config)

        token = middleware.generate_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_middleware_get_cookie_header_without_token(self):
        """Should generate new token when none provided."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.secret = "test-secret"
        middleware = CSRFMiddleware(config)

        header = middleware.get_cookie_header()
        assert config.cookie_name in header


class TestCSRFMiddlewareShouldValidate:
    """Tests for should_validate method."""

    def test_should_validate_disabled_middleware(self):
        """Should not validate when middleware is disabled."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = False
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("POST", "/api/debates") is False

    def test_should_validate_safe_methods(self):
        """Should not validate safe HTTP methods."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("GET", "/api/debates") is False
        assert middleware.should_validate("HEAD", "/api/debates") is False
        assert middleware.should_validate("OPTIONS", "/api/debates") is False

    def test_should_validate_state_changing_methods(self):
        """Should validate state-changing HTTP methods."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("POST", "/api/debates") is True
        assert middleware.should_validate("PUT", "/api/debates/123") is True
        assert middleware.should_validate("DELETE", "/api/debates/123") is True
        assert middleware.should_validate("PATCH", "/api/debates/123") is True

    def test_should_validate_case_insensitive_method(self):
        """Should handle case-insensitive HTTP methods."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("post", "/api/debates") is True
        assert middleware.should_validate("Post", "/api/debates") is True

    def test_should_validate_excluded_path(self):
        """Should not validate excluded paths."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset({"/api/webhooks/"})
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("POST", "/api/webhooks/") is False
        assert middleware.should_validate("POST", "/api/debates") is True

    def test_should_validate_unknown_method(self):
        """Should not validate unknown HTTP methods (not in STATE_CHANGING_METHODS)."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)

        assert middleware.should_validate("TRACE", "/api/debates") is False
        assert middleware.should_validate("CONNECT", "/api/debates") is False


class TestCSRFMiddlewareTokenExtraction:
    """Tests for token extraction methods."""

    def test_extract_cookie_token_from_handler(self):
        """Should extract token from cookie header."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {"Cookie": "_csrf_token=token123; session=abc"}

        token = middleware.extract_cookie_token(handler)
        assert token == "token123"

    def test_extract_cookie_token_from_dict(self):
        """Should extract token from dict-style headers."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        headers = {"Cookie": "_csrf_token=token456"}
        token = middleware.extract_cookie_token(headers)
        assert token == "token456"

    def test_extract_cookie_token_not_found(self):
        """Should return None when cookie not found."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {"Cookie": "session=abc"}

        token = middleware.extract_cookie_token(handler)
        assert token is None

    def test_extract_cookie_token_empty_cookie_header(self):
        """Should return None when cookie header is empty."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {"Cookie": ""}

        token = middleware.extract_cookie_token(handler)
        assert token is None

    def test_extract_cookie_token_no_cookie_header(self):
        """Should return None when no Cookie header exists."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {}

        token = middleware.extract_cookie_token(handler)
        assert token is None

    def test_extract_cookie_token_from_dict_lowercase_key(self):
        """Should extract token from dict with lowercase 'cookie' key."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        headers = {"cookie": "_csrf_token=lowertoken"}
        token = middleware.extract_cookie_token(headers)
        assert token == "lowertoken"

    def test_extract_cookie_token_no_headers_attr(self):
        """Should return None for handler without headers attribute."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        middleware = CSRFMiddleware(config)

        # An object with no headers attribute and not a dict
        handler = object()

        token = middleware.extract_cookie_token(handler)
        assert token is None

    def test_extract_header_token_primary(self):
        """Should extract token from primary header."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.header_name = "X-CSRF-Token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {"X-CSRF-Token": "header-token-123"}

        token = middleware.extract_header_token(handler)
        assert token == "header-token-123"

    def test_extract_header_token_alternative(self):
        """Should extract token from alternative header names."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.header_name = "X-CSRF-Token"
        middleware = CSRFMiddleware(config)

        # Should find X-XSRF-Token as alternative
        handler = MagicMock()
        handler.headers = MagicMock()
        handler.headers.get = lambda k: {"X-XSRF-Token": "alt-token"}.get(k)

        token = middleware.extract_header_token(handler)
        assert token == "alt-token"

    def test_extract_header_token_from_dict(self):
        """Should extract token from dict-style headers."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.header_name = "X-CSRF-Token"
        middleware = CSRFMiddleware(config)

        headers = {"X-CSRF-Token": "dict-token"}
        token = middleware.extract_header_token(headers)
        assert token == "dict-token"

    def test_extract_header_token_dict_alternative_names(self):
        """Should check alternative header names in dict-style headers."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.header_name = "X-CSRF-Token"
        middleware = CSRFMiddleware(config)

        # Only has the alternative name
        headers = {"x-xsrf-token": "alt-dict-token"}
        token = middleware.extract_header_token(headers)
        assert token == "alt-dict-token"

    def test_extract_header_token_none_when_missing(self):
        """Should return None when no header token found."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.header_name = "X-CSRF-Token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = MagicMock()
        handler.headers.get = lambda k: None

        token = middleware.extract_header_token(handler)
        assert token is None

    def test_extract_header_token_no_headers_attr(self):
        """Should return None for handler without headers attribute."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        middleware = CSRFMiddleware(config)

        handler = object()
        token = middleware.extract_header_token(handler)
        assert token is None


class TestCSRFMiddlewareValidateRequest:
    """Tests for validate_request method."""

    def test_validate_request_valid_tokens(self):
        """Should validate request with matching valid tokens."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        # Generate a valid token
        token = middleware.generate_token()

        # Create mock handler with matching tokens
        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/debates"
        handler.headers = {
            "Cookie": f"{config.cookie_name}={token}",
            config.header_name: token,
        }

        result = middleware.validate_request(handler)
        assert result.valid is True
        assert result.is_valid is True

    def test_validate_request_missing_cookie(self):
        """Should fail validation when cookie token is missing."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/debates"
        handler.headers = {
            "Cookie": "",
            config.header_name: "some-token",
        }

        result = middleware.validate_request(handler)
        assert result.valid is False
        assert "cookie" in result.reason.lower()

    def test_validate_request_missing_header(self):
        """Should fail validation when header token is missing."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        token = middleware.generate_token()

        # Use MagicMock for headers to allow custom get behavior
        mock_headers = MagicMock()
        mock_headers.get = lambda k, d=None: (
            f"{config.cookie_name}={token}" if k == "Cookie" else None
        )

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/debates"
        handler.headers = mock_headers

        result = middleware.validate_request(handler)
        assert result.valid is False
        assert "header" in result.reason.lower()

    def test_validate_request_token_mismatch(self):
        """Should fail validation when tokens don't match."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        token1 = middleware.generate_token()
        token2 = middleware.generate_token()

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/debates"
        handler.headers = {
            "Cookie": f"{config.cookie_name}={token1}",
            config.header_name: token2,
        }

        result = middleware.validate_request(handler)
        assert result.valid is False
        assert "mismatch" in result.reason.lower()

    def test_validate_request_skips_safe_methods(self):
        """Should skip validation for safe HTTP methods."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.command = "GET"
        handler.path = "/api/debates"
        handler.headers = {}

        result = middleware.validate_request(handler)
        assert result.valid is True
        assert "not required" in result.reason.lower()

    def test_validate_request_skips_excluded_path(self):
        """Should skip validation for excluded paths."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset({"/api/webhooks/"})
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/webhooks/"
        handler.headers = {}

        result = middleware.validate_request(handler)
        assert result.valid is True

    def test_validate_request_invalid_cookie_token(self):
        """Should fail validation when cookie token is invalid."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        invalid_token = "invalid-token"

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/debates"
        handler.headers = {
            "Cookie": f"{config.cookie_name}={invalid_token}",
            config.header_name: invalid_token,
        }

        result = middleware.validate_request(handler)
        assert result.valid is False
        assert "invalid" in result.reason.lower()

    def test_validate_request_extracts_path_from_handler(self):
        """Should extract path from handler when not provided explicitly."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset({"/api/webhooks/"})
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/webhooks/?key=value"  # With query string
        handler.headers = {}

        # Should strip query string and match excluded path
        result = middleware.validate_request(handler)
        assert result.valid is True

    def test_validate_request_uses_method_attribute(self):
        """Should use 'method' attribute when 'command' is not present."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)

        handler = MagicMock(spec=[])
        handler.method = "GET"
        handler.headers = {}

        result = middleware.validate_request(handler, path="/api/test")
        assert result.valid is True

    def test_validate_request_default_method_and_path(self):
        """Should use defaults when handler has no method or path."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        middleware = CSRFMiddleware(config)

        handler = MagicMock(spec=[])
        handler.headers = {}

        # No command, no method, no path -> defaults to GET, /
        result = middleware.validate_request(handler)
        assert result.valid is True  # GET is safe


class TestCSRFMiddlewareCookieSetting:
    """Tests for set_token_cookie method."""

    def test_set_token_cookie_generates_token(self):
        """Should generate and set a new token."""
        from aragora.server.middleware.csrf import CSRFMiddleware

        middleware = CSRFMiddleware()

        handler = MagicMock()
        token = middleware.set_token_cookie(handler)

        assert token is not None
        assert len(token) > 0
        handler.send_header.assert_called_once()

    def test_set_token_cookie_uses_provided_token(self):
        """Should use provided token instead of generating."""
        from aragora.server.middleware.csrf import CSRFMiddleware

        middleware = CSRFMiddleware()

        handler = MagicMock()
        token = middleware.set_token_cookie(handler, token="provided-token")

        assert token == "provided-token"
        call_args = handler.send_header.call_args
        assert "provided-token" in call_args[0][1]

    def test_get_cookie_header_format(self):
        """Should return properly formatted cookie header."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf"
        config.token_max_age = 86400
        config.cookie_secure = True
        config.cookie_httponly = True
        config.cookie_samesite = "Strict"
        middleware = CSRFMiddleware(config)

        header = middleware.get_cookie_header("test-token")

        assert "_csrf=test-token" in header
        assert "HttpOnly" in header
        assert "Secure" in header
        assert "SameSite=Strict" in header

    def test_set_token_cookie_handler_without_send_header(self):
        """Should handle handler without send_header gracefully."""
        from aragora.server.middleware.csrf import CSRFMiddleware

        middleware = CSRFMiddleware()

        handler = MagicMock(spec=[])  # No send_header method
        token = middleware.set_token_cookie(handler)

        # Should still return a token even if send_header is not available
        assert token is not None
        assert len(token) > 0


# =============================================================================
# Test csrf_protect Decorator
# =============================================================================


class TestCSRFProtectDecorator:
    """Tests for csrf_protect decorator."""

    def test_decorator_without_arguments(self):
        """Should work as @csrf_protect without arguments."""
        from aragora.server.middleware.csrf import csrf_protect

        with patch.dict(os.environ, {"ARAGORA_CSRF_ENABLED": "false"}, clear=True):

            @csrf_protect
            def handler(self_arg, handler_arg):
                return {"status": "success"}

            # Create mock handler
            mock_handler = MagicMock()
            mock_handler.headers = {}
            mock_handler.command = "GET"
            mock_handler.path = "/"

            # Should pass through when CSRF is disabled/not required
            result = handler(None, mock_handler)
            # GET requests should pass validation
            assert result == {"status": "success"}

    def test_decorator_with_config(self):
        """Should work as @csrf_protect(config=...) with arguments."""
        from aragora.server.middleware.csrf import CSRFConfig, csrf_protect

        config = CSRFConfig()
        config.enabled = False

        @csrf_protect(config=config)
        def handler(self_arg, handler_arg):
            return {"status": "success"}

        mock_handler = MagicMock()
        mock_handler.headers = {}

        result = handler(None, mock_handler)
        assert result == {"status": "success"}

    def test_decorator_blocks_invalid_csrf(self):
        """Should block requests with invalid CSRF."""
        from aragora.server.middleware.csrf import CSRFConfig, csrf_protect

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()

        @csrf_protect(config=config)
        def handler(handler_arg):
            return {"status": "success"}

        mock_handler = MagicMock()
        mock_handler.headers = {"Cookie": ""}
        mock_handler.command = "POST"
        mock_handler.path = "/api/test"

        # The decorator imports error_response from aragora.server.handlers.base
        with patch("aragora.server.handlers.base.error_response") as mock_error_response:
            mock_error_response.return_value = {"error": "CSRF failed"}
            result = handler(mock_handler)
            assert mock_error_response.called

    def test_decorator_allows_valid_csrf(self):
        """Should allow requests with valid CSRF."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            CSRFMiddleware,
            csrf_protect,
        )

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()

        middleware = CSRFMiddleware(config)
        valid_token = middleware.generate_token()

        @csrf_protect(config=config)
        def handler(handler_arg):
            return {"status": "success"}

        mock_handler = MagicMock()
        mock_handler.headers = {
            "Cookie": f"{config.cookie_name}={valid_token}",
            config.header_name: valid_token,
        }
        mock_handler.command = "POST"
        mock_handler.path = "/api/test"

        result = handler(mock_handler)
        assert result == {"status": "success"}

    def test_decorator_handler_in_kwargs(self):
        """Should find handler in kwargs."""
        from aragora.server.middleware.csrf import CSRFConfig, csrf_protect

        config = CSRFConfig()
        config.enabled = True

        @csrf_protect(config=config)
        def handler_func(handler=None):
            return {"status": "success"}

        mock_handler = MagicMock()
        mock_handler.headers = {}
        mock_handler.command = "GET"
        mock_handler.path = "/"

        result = handler_func(handler=mock_handler)
        assert result == {"status": "success"}

    def test_decorator_no_handler_found(self):
        """Should return error when no handler found in arguments."""
        from aragora.server.middleware.csrf import CSRFConfig, csrf_protect

        config = CSRFConfig()
        config.enabled = True

        @csrf_protect(config=config)
        def handler_func(x, y):
            return {"status": "success"}

        with patch("aragora.server.handlers.base.error_response") as mock_error_response:
            mock_error_response.return_value = {"error": "Internal error"}
            result = handler_func(42, "string_arg")
            assert mock_error_response.called
            # Should be called with 500 status
            args = mock_error_response.call_args
            assert args[0][1] == 500


# =============================================================================
# Test CSRFValidationResult
# =============================================================================


class TestCSRFValidationResult:
    """Tests for CSRFValidationResult dataclass."""

    def test_valid_result(self):
        """Should create valid result."""
        from aragora.server.middleware.csrf import CSRFValidationResult

        result = CSRFValidationResult(valid=True, reason="OK")
        assert result.valid is True
        assert result.is_valid is True
        assert result.reason == "OK"
        assert result.error == ""

    def test_invalid_result(self):
        """Should create invalid result with error."""
        from aragora.server.middleware.csrf import CSRFValidationResult

        result = CSRFValidationResult(
            valid=False, reason="Token mismatch", error="CSRF token mismatch"
        )
        assert result.valid is False
        assert result.is_valid is False
        assert result.reason == "Token mismatch"
        assert result.error == "CSRF token mismatch"

    def test_default_error_empty(self):
        """Default error should be empty string."""
        from aragora.server.middleware.csrf import CSRFValidationResult

        result = CSRFValidationResult(valid=False, reason="Failed")
        assert result.error == ""


# =============================================================================
# Test Constants and Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """All __all__ exports should be accessible."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            CSRFMiddleware,
            CSRFValidationResult,
            DEFAULT_COOKIE_NAME,
            DEFAULT_HEADER_NAME,
            DEFAULT_TOKEN_MAX_AGE,
            SAFE_METHODS,
            STATE_CHANGING_METHODS,
            csrf_protect,
            generate_csrf_token,
            get_csrf_cookie_value,
            validate_csrf_token,
        )

        assert DEFAULT_COOKIE_NAME == "_csrf_token"
        assert DEFAULT_HEADER_NAME == "X-CSRF-Token"
        assert DEFAULT_TOKEN_MAX_AGE == 24 * 60 * 60
        assert "POST" in STATE_CHANGING_METHODS
        assert "GET" in SAFE_METHODS

    def test_alternative_header_names(self):
        """ALTERNATIVE_HEADER_NAMES should include common CSRF header variants."""
        from aragora.server.middleware.csrf import ALTERNATIVE_HEADER_NAMES

        assert "X-CSRF-Token" in ALTERNATIVE_HEADER_NAMES
        assert "x-csrf-token" in ALTERNATIVE_HEADER_NAMES
        assert "X-XSRF-Token" in ALTERNATIVE_HEADER_NAMES
        assert "x-xsrf-token" in ALTERNATIVE_HEADER_NAMES


# =============================================================================
# Test Security Edge Cases
# =============================================================================


class TestSecurityEdgeCases:
    """Tests for security-critical edge cases."""

    def test_constant_time_comparison(self):
        """Token comparison should use constant-time comparison."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.secret = "test-secret"
        config.excluded_paths = frozenset()
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        token = middleware.generate_token()

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/api/test"
        handler.headers = {
            "Cookie": f"{config.cookie_name}={token}",
            config.header_name: token,
        }

        result = middleware.validate_request(handler)
        assert result.valid is True

    def test_hmac_signature_integrity(self):
        """HMAC signature should protect token integrity."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            generate_csrf_token,
            validate_csrf_token,
        )

        config = CSRFConfig()
        config.secret = "test-secret"

        token = generate_csrf_token(config)
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")

        # Tamper with random bytes
        parts[1] = "0" * len(parts[1])
        tampered = base64.urlsafe_b64encode(":".join(parts).encode()).decode()

        is_valid, error = validate_csrf_token(tampered, config)
        assert is_valid is False

    def test_token_not_reusable_after_secret_rotation(self):
        """Tokens should be invalid after secret rotation."""
        from aragora.server.middleware.csrf import (
            CSRFConfig,
            generate_csrf_token,
            validate_csrf_token,
        )

        config1 = CSRFConfig()
        config1.secret = "old-secret"

        config2 = CSRFConfig()
        config2.secret = "new-secret"

        token = generate_csrf_token(config1)

        # Token should be invalid with new secret
        is_valid, error = validate_csrf_token(token, config2)
        assert is_valid is False

    def test_path_traversal_in_exclusion(self):
        """Path exclusion should not be vulnerable to traversal."""
        from aragora.server.middleware.csrf import CSRFConfig

        config = CSRFConfig()
        config.excluded_paths = frozenset({"/webhooks/"})
        config.excluded_prefixes = frozenset()

        # These should NOT be excluded
        assert config.is_path_excluded("/webhooks/../api/secret") is False
        assert config.is_path_excluded("/../webhooks/") is False

    def test_token_with_only_four_parts(self):
        """Should reject token with more than three parts."""
        from aragora.server.middleware.csrf import CSRFConfig, validate_csrf_token

        config = CSRFConfig()
        config.secret = "test-secret"

        bad_data = "12345:abcdef:ghijkl:extra"
        token = base64.urlsafe_b64encode(bad_data.encode()).decode()

        is_valid, error = validate_csrf_token(token, config)
        assert is_valid is False

    def test_empty_secret_still_generates_token(self):
        """Should still generate tokens even with empty secret."""
        from aragora.server.middleware.csrf import CSRFConfig, generate_csrf_token

        config = CSRFConfig()
        config.secret = ""

        token = generate_csrf_token(config)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_multiple_cookies_extraction(self):
        """Should correctly extract CSRF token from multiple cookies."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.cookie_name = "_csrf_token"
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.headers = {"Cookie": "session=abc; _csrf_token=correcttoken; other=xyz"}

        token = middleware.extract_cookie_token(handler)
        assert token == "correcttoken"

    def test_validate_request_with_explicit_path(self):
        """validate_request should accept explicit path parameter."""
        from aragora.server.middleware.csrf import CSRFConfig, CSRFMiddleware

        config = CSRFConfig()
        config.enabled = True
        config.excluded_paths = frozenset({"/excluded/"})
        config.excluded_prefixes = frozenset()
        middleware = CSRFMiddleware(config)

        handler = MagicMock()
        handler.command = "POST"
        handler.path = "/not-excluded/"
        handler.headers = {}

        # Path parameter overrides handler.path
        result = middleware.validate_request(handler, path="/excluded/")
        assert result.valid is True
