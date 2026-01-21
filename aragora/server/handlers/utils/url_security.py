"""
URL security utilities for webhook and external request validation.

Provides SSRF (Server-Side Request Forgery) protection by validating URLs
before allowing outbound requests.

Security measures:
- Blocks private IP ranges (10.x, 172.16-31.x, 192.168.x)
- Blocks loopback addresses (127.x, ::1)
- Blocks link-local addresses (169.254.x, fe80::)
- Blocks cloud metadata endpoints (AWS, GCP, Azure)
- Blocks internal hostnames (.internal, .local, .localhost)
- Blocks IPv6-mapped IPv4 addresses (::ffff:127.0.0.1)
- Adds timeout on DNS resolution to prevent slow lookups
"""

import ipaddress
import socket
from typing import Tuple
from urllib.parse import urlparse


# DNS resolution timeout (seconds)
DNS_RESOLUTION_TIMEOUT = 5.0

# Cloud metadata endpoints that should never be accessible
BLOCKED_METADATA_IPS = frozenset([
    "169.254.169.254",  # AWS, Azure, DigitalOcean
    "fd00:ec2::254",    # AWS IPv6
])

BLOCKED_METADATA_HOSTNAMES = frozenset([
    "metadata.google.internal",
    "metadata.goog",
    "169.254.169.254",
    "instance-data",
])

# Internal hostname suffixes to block
BLOCKED_HOSTNAME_SUFFIXES = (
    ".internal",
    ".local",
    ".localhost",
    ".lan",
    ".corp",
    ".home",
    ".private",
)


def validate_webhook_url(url: str, allow_localhost: bool = False) -> Tuple[bool, str]:
    """
    Validate webhook URL is not pointing to internal services (SSRF protection).

    Blocks:
    - Non HTTP/HTTPS schemes
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Loopback addresses (127.x, ::1)
    - Link-local addresses (169.254.x, fe80::)
    - Reserved addresses
    - Cloud metadata endpoints (AWS, GCP, Azure, etc.)
    - Internal hostnames (.internal, .local, .localhost, .lan, etc.)

    Args:
        url: The URL to validate
        allow_localhost: If True, allows localhost for testing purposes

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "Invalid URL format"

    # Only allow HTTP/HTTPS
    if parsed.scheme not in ("http", "https"):
        return False, f"Only HTTP/HTTPS allowed, got: {parsed.scheme}"

    if not parsed.hostname:
        return False, "URL must have a hostname"

    hostname_lower = parsed.hostname.lower()

    # Block known metadata hostnames
    if hostname_lower in BLOCKED_METADATA_HOSTNAMES:
        return False, f"Blocked metadata hostname: {parsed.hostname}"

    # Block hostnames ending with internal suffixes
    if any(hostname_lower.endswith(suffix) for suffix in BLOCKED_HOSTNAME_SUFFIXES):
        return False, f"Internal hostname not allowed: {parsed.hostname}"

    # Skip IP validation only for actual localhost (for testing)
    if allow_localhost and hostname_lower in ("localhost", "127.0.0.1", "::1"):
        return True, ""

    # Try to resolve hostname and check all returned IPs
    try:
        # Set DNS resolution timeout to prevent slow lookups
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(DNS_RESOLUTION_TIMEOUT)
        try:
            addr_info = socket.getaddrinfo(
                parsed.hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        finally:
            socket.setdefaulttimeout(original_timeout)

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            valid, error = _validate_ip_address(ip_str)
            if not valid:
                return False, error

    except socket.gaierror:
        # DNS resolution failed - this is actually okay, the request will fail naturally
        pass
    except socket.timeout:
        return False, "DNS resolution timed out"
    except OSError:
        # Other socket errors - allow and let request handle it
        pass

    return True, ""


def _validate_ip_address(ip_str: str) -> Tuple[bool, str]:
    """
    Validate a single IP address for SSRF vulnerabilities.

    Args:
        ip_str: IP address string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return True, ""  # Not a valid IP, let request handle it

    # Check for IPv6-mapped IPv4 addresses (::ffff:127.0.0.1)
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        mapped_ipv4 = ip_obj.ipv4_mapped
        valid, error = _validate_ip_address(str(mapped_ipv4))
        if not valid:
            return False, f"IPv6-mapped {error}"

    if ip_obj.is_private:
        return False, f"Private IP not allowed: {ip_str}"
    if ip_obj.is_loopback:
        return False, f"Loopback IP not allowed: {ip_str}"
    if ip_obj.is_link_local:
        return False, f"Link-local IP not allowed: {ip_str}"
    if ip_obj.is_reserved:
        return False, f"Reserved IP not allowed: {ip_str}"
    if ip_obj.is_multicast:
        return False, f"Multicast IP not allowed: {ip_str}"
    if ip_obj.is_unspecified:
        return False, f"Unspecified IP not allowed: {ip_str}"

    # Block cloud metadata endpoints
    if ip_str in BLOCKED_METADATA_IPS:
        return False, f"Cloud metadata endpoint not allowed: {ip_str}"

    return True, ""


__all__ = ["validate_webhook_url", "_validate_ip_address", "DNS_RESOLUTION_TIMEOUT"]
