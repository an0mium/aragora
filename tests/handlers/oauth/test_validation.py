"""Tests for OAuth URL validation handler (aragora/server/handlers/oauth/validation.py).

Covers all public functions, edge cases, and error handling:
- _validate_redirect_url: scheme checks, host allowlist, subdomain matching,
  case normalization, exception handling, logging
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET = "aragora.server.handlers.oauth.validation"
CONFIG_TARGET = f"{TARGET}._get_allowed_redirect_hosts"


def _validate(url: str, allowed: frozenset[str] | None = None) -> bool:
    """Call _validate_redirect_url with a patched allowlist."""
    from aragora.server.handlers.oauth.validation import _validate_redirect_url

    if allowed is None:
        allowed = frozenset({"localhost", "127.0.0.1", "example.com"})
    with patch(CONFIG_TARGET, return_value=allowed):
        return _validate_redirect_url(url)


# ===========================================================================
# Scheme validation
# ===========================================================================


class TestSchemeValidation:
    """Tests for URL scheme checks in _validate_redirect_url."""

    def test_https_scheme_allowed(self):
        """HTTPS URLs with allowed hosts pass validation."""
        assert _validate("https://example.com/callback") is True

    def test_http_scheme_allowed(self):
        """HTTP URLs with allowed hosts pass validation."""
        assert _validate("http://localhost:3000/callback") is True

    def test_javascript_scheme_blocked(self):
        """javascript: scheme is blocked to prevent XSS."""
        assert _validate("javascript:alert(1)") is False

    def test_data_scheme_blocked(self):
        """data: scheme is blocked to prevent exfiltration."""
        assert _validate("data:text/html,<h1>evil</h1>") is False

    def test_ftp_scheme_blocked(self):
        """ftp: scheme is not in the allowed set."""
        assert _validate("ftp://example.com/file") is False

    def test_empty_scheme_blocked(self):
        """URLs without a scheme are rejected."""
        assert _validate("://example.com/path") is False

    def test_file_scheme_blocked(self):
        """file: scheme is blocked."""
        assert _validate("file:///etc/passwd") is False

    def test_custom_scheme_blocked(self):
        """Arbitrary custom schemes are blocked."""
        assert _validate("myapp://callback") is False

    def test_scheme_logs_warning(self, caplog):
        """Blocked schemes produce a warning log."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"example.com"})):
            with caplog.at_level(logging.WARNING, logger=TARGET):
                _validate_redirect_url("javascript:void(0)")
        assert "oauth_redirect_blocked" in caplog.text
        assert "scheme=javascript" in caplog.text


# ===========================================================================
# Host allowlist checks
# ===========================================================================


class TestHostAllowlist:
    """Tests for host allowlist matching."""

    def test_exact_host_match(self):
        """Exact host match returns True."""
        assert _validate("https://example.com/path") is True

    def test_host_not_in_allowlist(self):
        """Hosts not in the allowlist are rejected."""
        assert _validate("https://evil.com/steal") is False

    def test_localhost_allowed(self):
        """localhost is in the default test allowlist."""
        assert _validate("http://localhost:8080/cb") is True

    def test_loopback_ip_allowed(self):
        """127.0.0.1 is in the default test allowlist."""
        assert _validate("http://127.0.0.1:3000/auth") is True

    def test_unknown_host_logs_warning(self, caplog):
        """Unknown hosts produce a warning log with the blocked host."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"safe.com"})):
            with caplog.at_level(logging.WARNING, logger=TARGET):
                _validate_redirect_url("https://attacker.com/phish")
        assert "oauth_redirect_blocked" in caplog.text
        assert "attacker.com" in caplog.text

    def test_empty_allowlist_rejects_all(self):
        """An empty allowlist rejects every URL."""
        assert _validate("https://example.com/cb", allowed=frozenset()) is False

    def test_host_with_port_still_matches(self):
        """urlparse.hostname strips ports; host should still match."""
        assert _validate("https://example.com:8443/cb") is True


# ===========================================================================
# Subdomain matching
# ===========================================================================


class TestSubdomainMatching:
    """Tests for subdomain matching against allowed hosts."""

    def test_subdomain_of_allowed_host(self):
        """Subdomains of allowed hosts are accepted."""
        assert _validate("https://auth.example.com/cb") is True

    def test_deep_subdomain(self):
        """Deeply nested subdomains are accepted."""
        assert _validate("https://a.b.c.example.com/cb") is True

    def test_partial_match_not_accepted(self):
        """A host that merely ends with the allowed host string (without dot) is rejected."""
        # notexample.com ends with "example.com" but is NOT a subdomain
        assert _validate("https://notexample.com/cb") is False

    def test_subdomain_of_localhost(self):
        """Subdomains of localhost are accepted."""
        assert _validate("http://app.localhost:3000/cb") is True

    def test_subdomain_not_matching_any_allowed(self):
        """Subdomains of non-allowed hosts are rejected."""
        assert _validate("https://sub.evil.com/cb") is False


# ===========================================================================
# Host normalization (case insensitivity)
# ===========================================================================


class TestHostNormalization:
    """Tests for host case normalization."""

    def test_uppercase_host_normalized(self):
        """Uppercase hosts are normalized to lowercase for comparison."""
        assert _validate("https://EXAMPLE.COM/cb") is True

    def test_mixed_case_host(self):
        """Mixed-case hosts are normalized."""
        assert _validate("https://Example.Com/cb") is True

    def test_uppercase_subdomain(self):
        """Uppercase subdomains are normalized."""
        assert _validate("https://AUTH.Example.COM/cb") is True


# ===========================================================================
# Missing / empty host
# ===========================================================================


class TestMissingHost:
    """Tests for URLs with no extractable host."""

    def test_no_host_returns_false(self):
        """A URL with no hostname is rejected."""
        assert _validate("https:///path/only") is False

    def test_bare_path_returns_false(self):
        """A bare path (no scheme) is rejected (scheme check fails first)."""
        assert _validate("/just/a/path") is False

    def test_empty_string_returns_false(self):
        """Empty string input is rejected."""
        assert _validate("") is False


# ===========================================================================
# Exception handling
# ===========================================================================


class TestExceptionHandling:
    """Tests for exception handling in _validate_redirect_url."""

    def test_none_input_returns_false(self):
        """None input yields bytes scheme from urlparse, which is rejected."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"example.com"})):
            assert _validate_redirect_url(None) is False

    def test_integer_input_returns_false(self):
        """Integer input triggers AttributeError, caught and returns False."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"example.com"})):
            assert _validate_redirect_url(12345) is False

    def test_integer_input_logs_validation_error(self, caplog):
        """Integer input logs oauth_redirect_validation_error."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"example.com"})):
            with caplog.at_level(logging.WARNING, logger=TARGET):
                _validate_redirect_url(12345)
        assert "oauth_redirect_validation_error" in caplog.text

    def test_urlparse_value_error_returns_false(self):
        """If urlparse raises ValueError, function returns False."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(f"{TARGET}.urlparse", side_effect=ValueError("bad url")):
            with patch(CONFIG_TARGET, return_value=frozenset()):
                assert _validate_redirect_url("something") is False

    def test_urlparse_value_error_logs_warning(self, caplog):
        """ValueError from urlparse logs oauth_redirect_validation_error."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(f"{TARGET}.urlparse", side_effect=ValueError("bad url")):
            with patch(CONFIG_TARGET, return_value=frozenset()):
                with caplog.at_level(logging.WARNING, logger=TARGET):
                    _validate_redirect_url("something")
        assert "oauth_redirect_validation_error" in caplog.text

    def test_list_input_returns_false(self):
        """List input triggers AttributeError in urlparse, returns False."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"example.com"})):
            assert _validate_redirect_url(["http://example.com"]) is False

    def test_type_error_returns_false(self):
        """TypeError from urlparse is caught and returns False."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(f"{TARGET}.urlparse", side_effect=TypeError("bad type")):
            with patch(CONFIG_TARGET, return_value=frozenset()):
                assert _validate_redirect_url("something") is False


# ===========================================================================
# _get_allowed_redirect_hosts integration
# ===========================================================================


class TestAllowedHostsIntegration:
    """Tests verifying _validate_redirect_url calls _get_allowed_redirect_hosts."""

    def test_calls_get_allowed_redirect_hosts(self):
        """The function calls _get_allowed_redirect_hosts at runtime."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"myapp.io"})) as mock_hosts:
            _validate_redirect_url("https://myapp.io/cb")
            mock_hosts.assert_called_once()

    def test_dynamic_allowlist_respected(self):
        """Changing the allowlist between calls changes the result."""
        from aragora.server.handlers.oauth.validation import _validate_redirect_url

        with patch(CONFIG_TARGET, return_value=frozenset({"first.com"})):
            assert _validate_redirect_url("https://first.com/cb") is True
            assert _validate_redirect_url("https://second.com/cb") is False

        with patch(CONFIG_TARGET, return_value=frozenset({"second.com"})):
            assert _validate_redirect_url("https://first.com/cb") is False
            assert _validate_redirect_url("https://second.com/cb") is True

    def test_multiple_allowed_hosts(self):
        """Multiple hosts in the allowlist all pass validation."""
        allowed = frozenset({"a.com", "b.com", "c.com"})
        assert _validate("https://a.com/cb", allowed=allowed) is True
        assert _validate("https://b.com/cb", allowed=allowed) is True
        assert _validate("https://c.com/cb", allowed=allowed) is True
        assert _validate("https://d.com/cb", allowed=allowed) is False


# ===========================================================================
# URL edge cases
# ===========================================================================


class TestURLEdgeCases:
    """Additional edge cases for URL parsing and validation."""

    def test_url_with_query_params(self):
        """URLs with query parameters still validate the host correctly."""
        assert _validate("https://example.com/cb?code=abc&state=xyz") is True

    def test_url_with_fragment(self):
        """URLs with fragments still validate the host correctly."""
        assert _validate("https://example.com/cb#token=abc") is True

    def test_url_with_userinfo(self):
        """URLs with userinfo (user:pass@host) still match the host."""
        assert _validate("https://user:pass@example.com/cb") is True

    def test_url_with_path_traversal(self):
        """Path traversal does not affect host validation."""
        assert _validate("https://example.com/../../../etc/passwd") is True

    def test_url_with_only_scheme_and_host(self):
        """Minimal URL with just scheme and host is valid."""
        assert _validate("https://example.com") is True

    def test_ip_address_host(self):
        """IP addresses in the allowlist work correctly."""
        assert _validate("http://127.0.0.1/cb") is True

    def test_ipv6_host_not_in_allowlist(self):
        """IPv6 addresses not in the allowlist are rejected."""
        assert _validate("http://[::1]/cb") is False

    def test_ipv6_host_in_allowlist(self):
        """IPv6 addresses in the allowlist are accepted."""
        assert _validate("http://[::1]/cb", allowed=frozenset({"::1"})) is True
