"""
Tests for token extraction utilities.

Tests cover:
- extract_bearer_token
- extract_api_key_token
- extract_auth_token
- extract_token_from_headers
- extract_x_api_key
- extract_token_from_request
- extract_token_from_websocket
"""

import pytest
from unittest.mock import MagicMock

from aragora.utils.token_helpers import (
    extract_bearer_token,
    extract_api_key_token,
    extract_auth_token,
    extract_token_from_headers,
    extract_x_api_key,
    extract_token_from_request,
    extract_token_from_websocket,
)


class TestExtractBearerToken:
    """Tests for extract_bearer_token function."""

    def test_valid_bearer_token(self):
        """Extracts token from valid Bearer header."""
        assert extract_bearer_token("Bearer abc123") == "abc123"

    def test_bearer_with_long_token(self):
        """Handles long JWT-style tokens."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert extract_bearer_token(f"Bearer {jwt}") == jwt

    def test_none_input(self):
        """Returns None for None input."""
        assert extract_bearer_token(None) is None

    def test_empty_string(self):
        """Returns None for empty string."""
        assert extract_bearer_token("") is None

    def test_basic_auth(self):
        """Returns None for Basic auth."""
        assert extract_bearer_token("Basic dXNlcjpwYXNz") is None

    def test_api_key_auth(self):
        """Returns None for ApiKey auth."""
        assert extract_bearer_token("ApiKey xyz789") is None

    def test_bearer_lowercase(self):
        """Returns None for lowercase 'bearer'."""
        assert extract_bearer_token("bearer abc123") is None

    def test_bearer_no_space(self):
        """Returns None for 'Bearer' without space."""
        assert extract_bearer_token("Bearerabc123") is None

    def test_bearer_empty_token(self):
        """Returns empty string for 'Bearer ' with empty token."""
        assert extract_bearer_token("Bearer ") == ""


class TestExtractApiKeyToken:
    """Tests for extract_api_key_token function."""

    def test_valid_api_key(self):
        """Extracts token from valid ApiKey header."""
        assert extract_api_key_token("ApiKey xyz789") == "xyz789"

    def test_none_input(self):
        """Returns None for None input."""
        assert extract_api_key_token(None) is None

    def test_empty_string(self):
        """Returns None for empty string."""
        assert extract_api_key_token("") is None

    def test_bearer_auth(self):
        """Returns None for Bearer auth."""
        assert extract_api_key_token("Bearer abc123") is None

    def test_api_key_lowercase(self):
        """Returns None for lowercase 'apikey'."""
        assert extract_api_key_token("apikey xyz789") is None


class TestExtractAuthToken:
    """Tests for extract_auth_token function."""

    def test_bearer_token(self):
        """Extracts Bearer token."""
        assert extract_auth_token("Bearer abc123") == "abc123"

    def test_api_key_token(self):
        """Extracts ApiKey token."""
        assert extract_auth_token("ApiKey xyz789") == "xyz789"

    def test_prefers_bearer(self):
        """Bearer is tried before ApiKey (by design)."""
        # Note: single header can only be one type, this tests priority
        assert extract_auth_token("Bearer abc") == "abc"
        assert extract_auth_token("ApiKey xyz") == "xyz"

    def test_none_input(self):
        """Returns None for None input."""
        assert extract_auth_token(None) is None

    def test_unsupported_auth(self):
        """Returns None for unsupported auth types."""
        assert extract_auth_token("Basic dXNlcjpwYXNz") is None
        assert extract_auth_token("Digest xyz") is None


class TestExtractTokenFromHeaders:
    """Tests for extract_token_from_headers function."""

    def test_standard_header(self):
        """Extracts token from standard Authorization header."""
        headers = {"Authorization": "Bearer abc123"}
        assert extract_token_from_headers(headers) == "abc123"

    def test_lowercase_header(self):
        """Handles lowercase authorization header."""
        headers = {"authorization": "Bearer abc123"}
        assert extract_token_from_headers(headers) == "abc123"

    def test_mixed_case_header(self):
        """Handles mixed case header."""
        headers = {"AUTHORIZATION": "Bearer abc123"}
        assert extract_token_from_headers(headers) == "abc123"

    def test_api_key_auth(self):
        """Extracts ApiKey token from headers."""
        headers = {"Authorization": "ApiKey xyz789"}
        assert extract_token_from_headers(headers) == "xyz789"

    def test_missing_header(self):
        """Returns None when header is missing."""
        headers = {"Content-Type": "application/json"}
        assert extract_token_from_headers(headers) is None

    def test_empty_headers(self):
        """Returns None for empty headers dict."""
        assert extract_token_from_headers({}) is None

    def test_custom_header_name(self):
        """Supports custom header name."""
        headers = {"X-Custom-Auth": "Bearer custom-token"}
        assert extract_token_from_headers(headers, header_name="X-Custom-Auth") == "custom-token"

    def test_exact_match_preferred(self):
        """Exact case match is tried before case-insensitive."""
        headers = {
            "Authorization": "Bearer exact",
            "authorization": "Bearer lower",
        }
        # Exact match should be found first
        assert extract_token_from_headers(headers) == "exact"


class TestExtractXApiKey:
    """Tests for extract_x_api_key function."""

    def test_standard_header(self):
        """Extracts from X-API-Key header."""
        headers = {"X-API-Key": "my-secret-key"}
        assert extract_x_api_key(headers) == "my-secret-key"

    def test_mixed_case_variants(self):
        """Handles various case variants."""
        assert extract_x_api_key({"X-Api-Key": "key1"}) == "key1"
        assert extract_x_api_key({"x-api-key": "key2"}) == "key2"

    def test_missing_header(self):
        """Returns None when header is missing."""
        headers = {"Authorization": "Bearer abc"}
        assert extract_x_api_key(headers) is None

    def test_empty_headers(self):
        """Returns None for empty headers dict."""
        assert extract_x_api_key({}) is None


class TestExtractTokenFromRequest:
    """Tests for extract_token_from_request function."""

    def test_bearer_from_request(self):
        """Extracts Bearer token from request object."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer req-token"}
        assert extract_token_from_request(request) == "req-token"

    def test_x_api_key_from_request(self):
        """Extracts X-API-Key when no Authorization."""
        request = MagicMock()
        request.headers = {"X-API-Key": "api-key-123"}
        assert extract_token_from_request(request) == "api-key-123"

    def test_authorization_preferred_over_x_api_key(self):
        """Authorization header is tried before X-API-Key."""
        request = MagicMock()
        request.headers = {
            "Authorization": "Bearer auth-token",
            "X-API-Key": "api-key-123",
        }
        assert extract_token_from_request(request) == "auth-token"

    def test_no_headers_attribute(self):
        """Returns None when request has no headers attribute."""
        request = MagicMock(spec=[])  # No headers attribute
        assert extract_token_from_request(request) is None

    def test_empty_headers(self):
        """Returns None for empty headers."""
        request = MagicMock()
        request.headers = {}
        assert extract_token_from_request(request) is None

    def test_dict_like_headers(self):
        """Works with dict-like header objects."""
        class DictLikeHeaders:
            def __iter__(self):
                return iter(["Authorization"])
            def items(self):
                return [("Authorization", "Bearer dict-token")]
            def get(self, key, default=None):
                if key == "Authorization":
                    return "Bearer dict-token"
                return default

        request = MagicMock()
        request.headers = DictLikeHeaders()
        # After dict() conversion, should work
        assert extract_token_from_request(request) == "dict-token"


class TestExtractTokenFromWebsocket:
    """Tests for extract_token_from_websocket function."""

    def test_request_headers_attribute(self):
        """Extracts from request_headers attribute."""
        ws = MagicMock()
        ws.request_headers = {"Authorization": "Bearer ws-token"}
        assert extract_token_from_websocket(ws) == "ws-token"

    def test_headers_attribute(self):
        """Extracts from headers attribute."""
        ws = MagicMock(spec=["headers"])
        ws.headers = {"Authorization": "Bearer ws-token2"}
        assert extract_token_from_websocket(ws) == "ws-token2"

    def test_request_nested_headers(self):
        """Extracts from request.headers attribute."""
        ws = MagicMock(spec=["request"])
        ws.request.headers = {"Authorization": "Bearer nested-token"}
        assert extract_token_from_websocket(ws) == "nested-token"

    def test_prefers_request_headers(self):
        """request_headers is checked first."""
        ws = MagicMock()
        ws.request_headers = {"Authorization": "Bearer first"}
        ws.headers = {"Authorization": "Bearer second"}
        assert extract_token_from_websocket(ws) == "first"

    def test_no_headers_available(self):
        """Returns None when no headers found."""
        ws = MagicMock(spec=[])
        assert extract_token_from_websocket(ws) is None

    def test_empty_headers(self):
        """Returns None for empty headers."""
        ws = MagicMock()
        ws.request_headers = {}
        assert extract_token_from_websocket(ws) is None


class TestEdgeCases:
    """Edge case tests."""

    def test_token_with_special_characters(self):
        """Handles tokens with special characters."""
        assert extract_bearer_token("Bearer abc+/=123") == "abc+/=123"

    def test_token_with_spaces(self):
        """Token after 'Bearer ' can contain spaces."""
        # This might be invalid but the function just extracts
        assert extract_bearer_token("Bearer token with spaces") == "token with spaces"

    def test_unicode_token(self):
        """Handles unicode in tokens."""
        assert extract_bearer_token("Bearer token-café-123") == "token-café-123"

    def test_very_long_token(self):
        """Handles very long tokens."""
        long_token = "x" * 10000
        assert extract_bearer_token(f"Bearer {long_token}") == long_token

    def test_whitespace_only_header(self):
        """Returns None for whitespace-only header."""
        assert extract_bearer_token("   ") is None
        assert extract_api_key_token("   ") is None
