"""Tests for url_security module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

from unittest.mock import patch, MagicMock
import socket

import pytest

from aragora.server.handlers.utils.url_security import (
    validate_webhook_url,
    _validate_ip_address,
    DNS_RESOLUTION_TIMEOUT,
    BLOCKED_METADATA_IPS,
    BLOCKED_METADATA_HOSTNAMES,
    BLOCKED_HOSTNAME_SUFFIXES,
)


# =============================================================================
# Test validate_webhook_url - Basic Validation
# =============================================================================


class TestValidateWebhookUrlBasic:
    """Tests for basic URL validation."""

    def test_allows_https_url(self):
        """Should allow HTTPS URLs."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))
            ]
            valid, error = validate_webhook_url("https://example.com/webhook")
            assert valid is True
            assert error == ""

    def test_allows_http_url(self):
        """Should allow HTTP URLs."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80))
            ]
            valid, error = validate_webhook_url("http://example.com/webhook")
            assert valid is True
            assert error == ""

    def test_rejects_non_http_schemes(self):
        """Should reject non-HTTP/HTTPS schemes."""
        schemes = ["ftp://example.com", "file:///etc/passwd", "javascript:alert(1)"]
        for url in schemes:
            valid, error = validate_webhook_url(url)
            assert valid is False
            assert "Only HTTP/HTTPS allowed" in error

    def test_rejects_missing_hostname(self):
        """Should reject URLs without hostname."""
        valid, error = validate_webhook_url("https:///path")
        assert valid is False
        assert "hostname" in error.lower()

    def test_rejects_invalid_url_format(self):
        """Should reject malformed URLs."""
        valid, error = validate_webhook_url("not-a-valid-url")
        assert valid is False


# =============================================================================
# Test validate_webhook_url - SSRF Protection
# =============================================================================


class TestValidateWebhookUrlSSRF:
    """Tests for SSRF (Server-Side Request Forgery) protection."""

    def test_blocks_private_ip_10_range(self):
        """Should block 10.x.x.x private IP range."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80))]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            assert "Private IP" in error

    def test_blocks_private_ip_172_range(self):
        """Should block 172.16-31.x.x private IP range."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.16.0.1", 80))
            ]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            assert "Private IP" in error

    def test_blocks_private_ip_192_168_range(self):
        """Should block 192.168.x.x private IP range."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 80))
            ]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            assert "Private IP" in error

    def test_blocks_loopback_ipv4(self):
        """Should block IPv4 loopback addresses."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 80))]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            # Implementation checks private first, loopback IPs may be flagged as private
            assert "127.0.0.1" in error

    def test_blocks_loopback_ipv6(self):
        """Should block IPv6 loopback addresses."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 80, 0, 0))
            ]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            # Implementation checks private first for some addresses
            assert "::1" in error

    def test_blocks_link_local_ipv4(self):
        """Should block IPv4 link-local addresses."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.1", 80))
            ]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            # Implementation may flag as private or link-local
            assert "169.254.1.1" in error


# =============================================================================
# Test validate_webhook_url - Cloud Metadata
# =============================================================================


class TestValidateWebhookUrlMetadata:
    """Tests for cloud metadata endpoint blocking."""

    def test_blocks_aws_metadata_ip(self):
        """Should block AWS metadata endpoint IP."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 80))
            ]
            valid, error = validate_webhook_url("https://internal.example.com")
            assert valid is False
            # Implementation may flag this as private, link-local, or metadata
            assert "169.254.169.254" in error

    def test_blocks_gcp_metadata_hostname(self):
        """Should block GCP metadata hostname."""
        valid, error = validate_webhook_url("https://metadata.google.internal/computeMetadata")
        assert valid is False
        assert "metadata" in error.lower()

    def test_blocks_metadata_hostname(self):
        """Should block known metadata hostnames."""
        for hostname in BLOCKED_METADATA_HOSTNAMES:
            if not hostname.startswith("169"):  # Skip IPs
                valid, error = validate_webhook_url(f"https://{hostname}/")
                assert valid is False, f"Should block {hostname}"


# =============================================================================
# Test validate_webhook_url - Internal Hostnames
# =============================================================================


class TestValidateWebhookUrlHostnames:
    """Tests for internal hostname blocking."""

    def test_blocks_internal_suffix(self):
        """Should block .internal hostname suffix."""
        valid, error = validate_webhook_url("https://api.internal/webhook")
        assert valid is False
        assert "Internal hostname" in error

    def test_blocks_local_suffix(self):
        """Should block .local hostname suffix."""
        valid, error = validate_webhook_url("https://server.local/webhook")
        assert valid is False
        assert "Internal hostname" in error

    def test_blocks_localhost_suffix(self):
        """Should block .localhost hostname suffix."""
        valid, error = validate_webhook_url("https://api.localhost/webhook")
        assert valid is False
        assert "Internal hostname" in error

    def test_blocks_lan_suffix(self):
        """Should block .lan hostname suffix."""
        valid, error = validate_webhook_url("https://router.lan/api")
        assert valid is False
        assert "Internal hostname" in error

    def test_blocks_corp_suffix(self):
        """Should block .corp hostname suffix."""
        valid, error = validate_webhook_url("https://intranet.corp/api")
        assert valid is False
        assert "Internal hostname" in error


# =============================================================================
# Test validate_webhook_url - Localhost Allowance
# =============================================================================


class TestValidateWebhookUrlLocalhost:
    """Tests for localhost allowance for testing."""

    def test_blocks_localhost_by_default(self):
        """Should block localhost by default."""
        valid, error = validate_webhook_url("https://localhost/webhook")
        assert valid is False

    def test_allows_localhost_when_flag_set(self):
        """Should allow localhost when allow_localhost=True."""
        valid, error = validate_webhook_url("https://localhost/webhook", allow_localhost=True)
        assert valid is True
        assert error == ""

    def test_allows_127_0_0_1_when_flag_set(self):
        """Should allow 127.0.0.1 when allow_localhost=True."""
        valid, error = validate_webhook_url("https://127.0.0.1/webhook", allow_localhost=True)
        assert valid is True
        assert error == ""


# =============================================================================
# Test validate_webhook_url - DNS Resolution
# =============================================================================


class TestValidateWebhookUrlDNS:
    """Tests for DNS resolution handling."""

    def test_handles_dns_resolution_failure(self):
        """Should handle DNS resolution failures gracefully."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("DNS resolution failed")
            valid, error = validate_webhook_url("https://nonexistent.example.com")
            # Should pass because DNS failure is okay (request will fail naturally)
            assert valid is True

    def test_rejects_on_dns_timeout(self):
        """Should reject when DNS resolution times out."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = TimeoutError("DNS timeout")
            valid, error = validate_webhook_url("https://slow.example.com")
            assert valid is False
            # Error message is "DNS resolution timed out"
            assert "timed out" in error.lower()


# =============================================================================
# Test _validate_ip_address
# =============================================================================


class TestValidateIPAddress:
    """Tests for _validate_ip_address function."""

    def test_allows_public_ipv4(self):
        """Should allow public IPv4 addresses."""
        valid, error = _validate_ip_address("93.184.216.34")
        assert valid is True
        assert error == ""

    def test_allows_public_ipv6(self):
        """Should allow public IPv6 addresses."""
        valid, error = _validate_ip_address("2001:4860:4860::8888")
        assert valid is True
        assert error == ""

    def test_blocks_private_ipv4(self):
        """Should block private IPv4 addresses."""
        private_ips = ["10.0.0.1", "172.16.0.1", "192.168.1.1"]
        for ip in private_ips:
            valid, error = _validate_ip_address(ip)
            assert valid is False
            assert "Private" in error

    def test_blocks_loopback(self):
        """Should block loopback addresses."""
        valid, error = _validate_ip_address("127.0.0.1")
        assert valid is False
        # Implementation may flag as private or loopback
        assert "127.0.0.1" in error

    def test_blocks_multicast(self):
        """Should block multicast addresses."""
        valid, error = _validate_ip_address("224.0.0.1")
        assert valid is False
        # May be flagged as multicast or reserved
        assert "224.0.0.1" in error or "Multicast" in error

    def test_blocks_reserved(self):
        """Should block reserved addresses."""
        valid, error = _validate_ip_address("0.0.0.0")
        assert valid is False
        # May be flagged as private, unspecified, or reserved
        assert "0.0.0.0" in error

    def test_blocks_ipv6_mapped_private_ipv4(self):
        """Should block IPv6-mapped private IPv4 addresses."""
        valid, error = _validate_ip_address("::ffff:192.168.1.1")
        assert valid is False
        assert "Private" in error

    def test_passes_invalid_ip_string(self):
        """Should pass through non-IP strings."""
        valid, error = _validate_ip_address("not-an-ip")
        # Invalid IPs are passed through (request will handle it)
        assert valid is True


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_dns_timeout_is_reasonable(self):
        """Should have a reasonable DNS timeout."""
        assert 1 <= DNS_RESOLUTION_TIMEOUT <= 30

    def test_blocked_metadata_ips_includes_aws(self):
        """Should include AWS metadata endpoint."""
        assert "169.254.169.254" in BLOCKED_METADATA_IPS

    def test_blocked_hostname_suffixes_is_tuple(self):
        """Should be a tuple for efficient endswith()."""
        assert isinstance(BLOCKED_HOSTNAME_SUFFIXES, tuple)

    def test_blocked_hostname_suffixes_includes_common(self):
        """Should include common internal suffixes."""
        assert ".internal" in BLOCKED_HOSTNAME_SUFFIXES
        assert ".local" in BLOCKED_HOSTNAME_SUFFIXES
        assert ".localhost" in BLOCKED_HOSTNAME_SUFFIXES
