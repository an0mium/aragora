"""
SSRF (Server-Side Request Forgery) Protection Utilities.

Provides centralized URL validation and protection against SSRF attacks.
All external URL fetching should use these utilities to prevent:
- Requests to internal networks (10.x.x.x, 192.168.x.x, 127.x.x.x)
- Requests to cloud metadata endpoints (169.254.169.254)
- Requests to localhost and loopback addresses
- Protocol smuggling (file://, gopher://, etc.)
- DNS rebinding attacks (optional resolution check)

Usage:
    from aragora.security.ssrf_protection import (
        validate_url,
        is_url_safe,
        SSRFValidationError,
    )

    # Simple check
    if not is_url_safe(user_provided_url):
        raise ValueError("Invalid URL")

    # Detailed validation with error message
    result = validate_url(user_provided_url)
    if not result.is_safe:
        logger.warning(f"SSRF attempt blocked: {result.error}")
        raise SSRFValidationError(result.error)

    # Domain whitelist validation
    result = validate_url(url, allowed_domains={"api.example.com", "cdn.example.com"})

Security Notes:
    - Always validate URLs before making HTTP requests
    - Use allowed_domains when possible for defense in depth
    - Enable DNS resolution checks (resolve_dns=True) for high-security contexts
    - Log blocked attempts for security monitoring
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SSRFValidationError(Exception):
    """Raised when URL validation fails due to SSRF risk."""

    def __init__(self, message: str, url: str = ""):
        self.url = url
        super().__init__(message)


class SecurityConfigurationError(Exception):
    """Raised when security configuration is invalid or dangerous.

    This exception is raised when security settings would create vulnerabilities,
    such as allowing localhost in production environments.
    """

    def __init__(self, message: str, setting: str = "", environment: str = ""):
        self.setting = setting
        self.environment = environment
        super().__init__(message)


# Allowed protocols - only HTTP(S) for external requests
ALLOWED_PROTOCOLS: frozenset[str] = frozenset({"http", "https"})

# Dangerous protocols that should never be allowed
BLOCKED_PROTOCOLS: frozenset[str] = frozenset(
    {
        "file",
        "ftp",
        "gopher",
        "data",
        "javascript",
        "vbscript",
        "ldap",
        "ldaps",
        "dict",
        "sftp",
        "tftp",
        "jar",
        "netdoc",
    }
)

# Private/internal IP ranges (RFC 1918, RFC 5737, RFC 6598, loopback, link-local)
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (includes cloud metadata)
    ipaddress.ip_network("100.64.0.0/10"),  # Carrier-grade NAT (RFC 6598)
    ipaddress.ip_network("192.0.0.0/24"),  # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1 (RFC 5737)
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2 (RFC 5737)
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3 (RFC 5737)
    ipaddress.ip_network("0.0.0.0/8"),  # Current network
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved for future use
    ipaddress.ip_network("255.255.255.255/32"),  # Broadcast
]

# IPv6 private/internal ranges
PRIVATE_IPV6_RANGES = [
    ipaddress.ip_network("::1/128"),  # Loopback
    ipaddress.ip_network("fc00::/7"),  # Unique local address
    ipaddress.ip_network("fe80::/10"),  # Link-local
    ipaddress.ip_network("ff00::/8"),  # Multicast
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped (check underlying IPv4)
    ipaddress.ip_network("::/128"),  # Unspecified
]

# Cloud metadata service IPs that should always be blocked
CLOUD_METADATA_IPS: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS, GCP, Azure, DigitalOcean, etc.
        "fd00:ec2::254",  # AWS IPv6 metadata
        "metadata.google.internal",
        "metadata.gcp.internal",
    }
)

# Common localhost hostnames
LOCALHOST_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "local",
        "127.0.0.1",
        "::1",
        "[::1]",
        "0.0.0.0",
        "0",
    }
)

# DNS rebinding protection - suspicious hostname patterns
SUSPICIOUS_HOSTNAME_PATTERNS = [
    re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$"),  # Raw IPv4
    re.compile(r"^\[.*\]$"),  # IPv6 in brackets
    re.compile(r"^0x[0-9a-f]+$", re.IGNORECASE),  # Hex encoded
    re.compile(r"^[0-9]+$"),  # Decimal encoded IP
    re.compile(r"\.local$", re.IGNORECASE),  # mDNS
    re.compile(r"\.internal$", re.IGNORECASE),  # Internal domains
    re.compile(r"\.localhost$", re.IGNORECASE),  # Localhost subdomains
]


@dataclass
class SSRFValidationResult:
    """Result of SSRF URL validation."""

    is_safe: bool
    url: str
    error: str = ""
    resolved_ip: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def safe(cls, url: str, resolved_ip: str | None = None) -> SSRFValidationResult:
        """Create a safe validation result."""
        return cls(is_safe=True, url=url, resolved_ip=resolved_ip)

    @classmethod
    def unsafe(cls, url: str, error: str) -> SSRFValidationResult:
        """Create an unsafe validation result."""
        return cls(is_safe=False, url=url, error=error)


def _is_ip_private(ip_str: str) -> bool:
    """Check if an IP address is in a private/internal range.

    Args:
        ip_str: IP address string to check

    Returns:
        True if IP is private/internal, False if public
    """
    try:
        ip = ipaddress.ip_address(ip_str)

        # Check IPv4 ranges
        if isinstance(ip, ipaddress.IPv4Address):
            for network in PRIVATE_IP_RANGES:
                if ip in network:
                    return True

        # Check IPv6 ranges
        elif isinstance(ip, ipaddress.IPv6Address):
            for network in PRIVATE_IPV6_RANGES:
                if ip in network:
                    return True

            # Check for IPv4-mapped IPv6 addresses (::ffff:x.x.x.x)
            if ip.ipv4_mapped:
                return _is_ip_private(str(ip.ipv4_mapped))

        return False

    except ValueError:
        # Invalid IP address - treat as potentially unsafe
        return True


def _is_hostname_suspicious(hostname: str) -> tuple[bool, str]:
    """Check if hostname matches suspicious patterns.

    Args:
        hostname: Hostname to check

    Returns:
        Tuple of (is_suspicious, reason)
    """
    hostname_lower = hostname.lower()

    # Check for localhost aliases (unless explicitly allowed for testing)
    if hostname_lower in LOCALHOST_HOSTNAMES:
        allow_localhost = os.environ.get("ARAGORA_SSRF_ALLOW_LOCALHOST", "").lower() == "true"
        is_production = os.environ.get("ARAGORA_ENV", "").lower() == "production"

        # SECURITY: Never allow localhost in production, even if override is set
        # This provides defense-in-depth against accidental misconfiguration
        if is_production:
            logger.warning(
                "Localhost access blocked in production environment. "
                "ARAGORA_SSRF_ALLOW_LOCALHOST has no effect when ARAGORA_ENV=production."
            )
            return True, "Localhost hostname detected (blocked in production)"

        if allow_localhost:
            # Non-production environment with explicit override - allow localhost for tests
            return False, ""
        return True, "Localhost hostname detected"

    # Check for cloud metadata hostnames
    if hostname_lower in CLOUD_METADATA_IPS:
        return True, "Cloud metadata endpoint detected"

    # Check for suspicious patterns
    for pattern in SUSPICIOUS_HOSTNAME_PATTERNS:
        if pattern.search(hostname_lower):
            return True, f"Suspicious hostname pattern: {hostname}"

    return False, ""


def _resolve_hostname(hostname: str, timeout: float = 2.0) -> list[str]:
    """Resolve hostname to IP addresses.

    Args:
        hostname: Hostname to resolve
        timeout: DNS resolution timeout in seconds

    Returns:
        List of resolved IP addresses
    """
    try:
        # Set socket timeout for DNS resolution
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            # Get all IP addresses for the hostname
            results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
            ips: list[str] = list({str(result[4][0]) for result in results})
            return ips
        finally:
            socket.setdefaulttimeout(old_timeout)
    except (TimeoutError, socket.gaierror, OSError) as e:
        logger.debug(f"DNS resolution failed for {hostname}: {e}")
        return []


def validate_url(
    url: str,
    allowed_protocols: set[str] | frozenset[str] | None = None,
    allowed_domains: set[str] | frozenset[str] | None = None,
    blocked_domains: set[str] | frozenset[str] | None = None,
    resolve_dns: bool = False,
    allow_private_ips: bool = False,
    dns_timeout: float = 2.0,
) -> SSRFValidationResult:
    """Validate a URL for SSRF attacks.

    Args:
        url: URL to validate
        allowed_protocols: Set of allowed protocols (default: http, https)
        allowed_domains: If provided, only these domains are allowed
        blocked_domains: Domains to explicitly block
        resolve_dns: If True, resolve hostname and check resolved IP
        allow_private_ips: If True, allow private IP ranges (NOT recommended)
        dns_timeout: Timeout for DNS resolution in seconds

    Returns:
        SSRFValidationResult indicating if URL is safe

    Security:
        - Set allowed_domains for defense in depth when possible
        - Enable resolve_dns=True for high-security contexts (prevents DNS rebinding)
        - Never set allow_private_ips=True in production
    """
    if not url:
        return SSRFValidationResult.unsafe(url, "Empty URL")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        return SSRFValidationResult.unsafe(url, f"URL parsing failed: {e}")

    # Validate protocol/scheme
    scheme = (parsed.scheme or "").lower()
    protocols = allowed_protocols or ALLOWED_PROTOCOLS

    if not scheme:
        return SSRFValidationResult.unsafe(url, "Missing protocol/scheme")

    if scheme in BLOCKED_PROTOCOLS:
        return SSRFValidationResult.unsafe(url, f"Blocked protocol: {scheme}")

    if scheme not in protocols:
        return SSRFValidationResult.unsafe(url, f"Protocol not allowed: {scheme}")

    # Extract and validate hostname
    hostname = parsed.hostname or parsed.netloc
    if not hostname:
        return SSRFValidationResult.unsafe(url, "Missing hostname")

    hostname_lower = hostname.lower()

    # Check if localhost is allowed for testing
    localhost_allowed = os.environ.get("ARAGORA_SSRF_ALLOW_LOCALHOST", "").lower() == "true"
    is_localhost = hostname_lower in LOCALHOST_HOSTNAMES

    # Check for suspicious hostname patterns
    is_suspicious, reason = _is_hostname_suspicious(hostname)
    if is_suspicious:
        return SSRFValidationResult.unsafe(url, reason)

    # Allow private IPs for localhost in test mode
    if localhost_allowed and is_localhost:
        allow_private_ips = True

    # Check domain whitelist
    if allowed_domains:
        if hostname_lower not in {d.lower() for d in allowed_domains}:
            return SSRFValidationResult.unsafe(url, f"Domain not in whitelist: {hostname}")

    # Check domain blocklist
    if blocked_domains:
        if hostname_lower in {d.lower() for d in blocked_domains}:
            return SSRFValidationResult.unsafe(url, f"Domain is blocked: {hostname}")

    # Check if hostname is an IP address
    try:
        ip = ipaddress.ip_address(hostname)
        if not allow_private_ips and _is_ip_private(str(ip)):
            return SSRFValidationResult.unsafe(url, f"Private/internal IP address: {ip}")
        return SSRFValidationResult.safe(url, resolved_ip=str(ip))
    except ValueError:
        # Not an IP address, it's a hostname - continue validation
        pass

    # DNS resolution check (prevents DNS rebinding attacks)
    resolved_ip = None
    if resolve_dns:
        resolved_ips = _resolve_hostname(hostname, timeout=dns_timeout)
        if not resolved_ips:
            # DNS resolution failed - this could be intentional DNS rebinding
            # In strict mode, we might want to block this
            logger.warning(f"DNS resolution failed for {hostname}")
        else:
            for resolved_addr in resolved_ips:
                if not allow_private_ips and _is_ip_private(resolved_addr):
                    return SSRFValidationResult.unsafe(
                        url, f"Hostname resolves to private IP: {resolved_addr}"
                    )
            resolved_ip = resolved_ips[0]

    return SSRFValidationResult.safe(url, resolved_ip=resolved_ip)


def is_url_safe(
    url: str,
    allowed_domains: set[str] | None = None,
    resolve_dns: bool = False,
) -> bool:
    """Simple check if URL is safe from SSRF attacks.

    Args:
        url: URL to validate
        allowed_domains: If provided, only these domains are allowed
        resolve_dns: If True, resolve hostname and check resolved IP

    Returns:
        True if URL is safe, False otherwise
    """
    result = validate_url(url, allowed_domains=allowed_domains, resolve_dns=resolve_dns)
    return result.is_safe


def validate_webhook_url(
    url: str,
    allowed_domains: set[str] | frozenset[str] | None = None,
    service_name: str = "webhook",
) -> SSRFValidationResult:
    """Validate a webhook URL with strict settings.

    Webhooks require stricter validation since they involve outbound HTTP calls
    initiated by external input.

    Args:
        url: Webhook URL to validate
        allowed_domains: Required set of allowed domains for this webhook service
        service_name: Name of the service for logging

    Returns:
        SSRFValidationResult indicating if URL is safe
    """
    result = validate_url(
        url,
        allowed_protocols={"https"},  # Webhooks should use HTTPS only
        allowed_domains=allowed_domains,
        resolve_dns=True,  # Strict DNS checking for webhooks
        allow_private_ips=False,
    )

    if not result.is_safe:
        logger.warning(f"SSRF blocked for {service_name} webhook: url={url}, error={result.error}")

    return result


# Pre-configured validators for common services
SLACK_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "hooks.slack.com",
        "api.slack.com",
    }
)

DISCORD_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "discord.com",
        "discordapp.com",
    }
)

GITHUB_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "api.github.com",
        "github.com",
        "raw.githubusercontent.com",
    }
)

MICROSOFT_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        "graph.microsoft.com",
        "outlook.office.com",
        "login.microsoftonline.com",
    }
)


def validate_slack_url(url: str) -> SSRFValidationResult:
    """Validate a Slack webhook/API URL."""
    return validate_webhook_url(url, SLACK_ALLOWED_DOMAINS, "Slack")


def validate_discord_url(url: str) -> SSRFValidationResult:
    """Validate a Discord webhook/API URL."""
    return validate_webhook_url(url, DISCORD_ALLOWED_DOMAINS, "Discord")


def validate_github_url(url: str) -> SSRFValidationResult:
    """Validate a GitHub API URL."""
    return validate_webhook_url(url, GITHUB_ALLOWED_DOMAINS, "GitHub")


def validate_microsoft_url(url: str) -> SSRFValidationResult:
    """Validate a Microsoft Graph/Office URL."""
    return validate_webhook_url(url, MICROSOFT_ALLOWED_DOMAINS, "Microsoft")


# Environment-based configuration
def _is_production_environment() -> bool:
    """Check if we're running in a production environment.

    Returns:
        True if ARAGORA_ENV is set to 'production', False otherwise.
    """
    env = os.environ.get("ARAGORA_ENV", "").lower()
    return env == "production"


def get_ssrf_config() -> dict:
    """Get SSRF protection configuration from environment.

    Returns:
        Dict with SSRF configuration settings
    """
    return {
        "strict_mode": os.environ.get("ARAGORA_SSRF_STRICT", "true").lower() == "true",
        "resolve_dns": os.environ.get("ARAGORA_SSRF_RESOLVE_DNS", "false").lower() == "true",
        "dns_timeout": float(os.environ.get("ARAGORA_SSRF_DNS_TIMEOUT", "2.0")),
        "log_blocked": os.environ.get("ARAGORA_SSRF_LOG_BLOCKED", "true").lower() == "true",
        "allow_localhost": os.environ.get("ARAGORA_SSRF_ALLOW_LOCALHOST", "false").lower()
        == "true",
    }


def validate_ssrf_security_settings() -> None:
    """Validate SSRF security settings to prevent dangerous configurations.

    This function checks for security misconfigurations that could lead to
    vulnerabilities in production environments.

    Raises:
        SecurityConfigurationError: If dangerous configuration is detected in production.

    Security:
        - In production (ARAGORA_ENV=production), ARAGORA_SSRF_ALLOW_LOCALHOST must not be set
        - In non-production environments, a warning is logged if localhost is allowed
    """
    allow_localhost = os.environ.get("ARAGORA_SSRF_ALLOW_LOCALHOST", "").lower() == "true"
    is_production = _is_production_environment()

    if is_production and allow_localhost:
        raise SecurityConfigurationError(
            "ARAGORA_SSRF_ALLOW_LOCALHOST cannot be enabled in production environment. "
            "This setting disables critical SSRF protections and could allow attackers "
            "to access internal services. Remove this environment variable or set "
            "ARAGORA_ENV to a non-production value.",
            setting="ARAGORA_SSRF_ALLOW_LOCALHOST",
            environment="production",
        )

    if allow_localhost and not is_production:
        logger.warning(
            "SSRF localhost protection is disabled (ARAGORA_SSRF_ALLOW_LOCALHOST=true). "
            "This is acceptable for testing/development but must never be used in production."
        )


# Module-level initialization: validate security settings on import
# This catches dangerous configurations early, preventing production deployments
# with insecure settings. Uses try/except for graceful degradation in edge cases.
try:
    validate_ssrf_security_settings()
except SecurityConfigurationError:
    # Re-raise security errors - these must not be silently ignored
    raise
except Exception as e:
    # Log other unexpected errors but don't crash the application
    # This provides graceful degradation for edge cases like missing env access
    logger.error(f"SSRF security validation failed unexpectedly: {e}")


__all__ = [
    # Core functions
    "validate_url",
    "is_url_safe",
    "validate_webhook_url",
    "validate_ssrf_security_settings",
    # Service-specific validators
    "validate_slack_url",
    "validate_discord_url",
    "validate_github_url",
    "validate_microsoft_url",
    # Result and error types
    "SSRFValidationResult",
    "SSRFValidationError",
    "SecurityConfigurationError",
    # Domain constants
    "SLACK_ALLOWED_DOMAINS",
    "DISCORD_ALLOWED_DOMAINS",
    "GITHUB_ALLOWED_DOMAINS",
    "MICROSOFT_ALLOWED_DOMAINS",
    # Configuration
    "get_ssrf_config",
    "ALLOWED_PROTOCOLS",
    "BLOCKED_PROTOCOLS",
]
