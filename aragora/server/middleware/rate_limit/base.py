"""
Rate limiting base types and configuration.

Contains shared types, configuration, and helper functions used by
all rate limiting components.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import posixpath
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_RATE_LIMIT = int(os.environ.get("ARAGORA_RATE_LIMIT", "60"))
IP_RATE_LIMIT = int(os.environ.get("ARAGORA_IP_RATE_LIMIT", "120"))
BURST_MULTIPLIER = float(os.environ.get("ARAGORA_BURST_MULTIPLIER", "2.0"))

# Trusted proxies for X-Forwarded-For header (comma-separated IPs/CIDRs)
TRUSTED_PROXIES_RAW = os.environ.get(
    "ARAGORA_TRUSTED_PROXIES",
    "127.0.0.1,::1,localhost",
).strip()
TRUSTED_PROXIES: frozenset[str] = frozenset(
    p.strip() for p in TRUSTED_PROXIES_RAW.split(",") if p.strip()
)
_TRUSTED_PROXY_IPS: set[str] = set()
_TRUSTED_PROXY_NETS: list[ipaddress._BaseNetwork] = []

for entry in TRUSTED_PROXIES:
    if entry == "localhost":
        _TRUSTED_PROXY_IPS.update({"127.0.0.1", "::1"})
        continue
    try:
        if "/" in entry:
            _TRUSTED_PROXY_NETS.append(ipaddress.ip_network(entry, strict=False))
        else:
            _TRUSTED_PROXY_IPS.add(str(ipaddress.ip_address(entry)))
    except ValueError:
        logger.debug("Ignoring invalid trusted proxy entry: %s", entry)


def _normalize_ip(ip_value: str) -> str:
    """Normalize IP addresses for consistent rate limiting.

    - IPv4: Used as-is after validation
    - IPv6: Normalized to standard form, with /64 prefix grouping for fairness
    - Invalid: Returns original string (will be treated as unique key)
    """
    if not ip_value:
        return ""

    try:
        addr = ipaddress.ip_address(ip_value.strip())
        if isinstance(addr, ipaddress.IPv6Address):
            # Group by /64 prefix for IPv6 (standard allocation size)
            network = ipaddress.ip_network(f"{addr}/64", strict=False)
            return str(network.network_address)
        return str(addr)
    except ValueError:
        return ip_value.strip()


def _is_trusted_proxy(ip: str) -> bool:
    """Check if IP is a trusted proxy for XFF header processing.

    Uses pre-parsed IP sets and networks for efficient lookup.
    Supports both individual IPs and CIDR ranges.
    """
    if not ip:
        return False

    normalized = _normalize_ip(ip)
    if normalized in _TRUSTED_PROXY_IPS:
        return True

    if _TRUSTED_PROXY_NETS:
        try:
            addr = ipaddress.ip_address(normalized)
            for net in _TRUSTED_PROXY_NETS:
                if addr in net:
                    return True
        except ValueError:
            pass

    return False


def _extract_client_ip(
    headers: dict,
    remote_addr: str,
    trust_xff_from_proxies: bool = True,
) -> str:
    """Extract real client IP from request headers.

    Priority:
    1. X-Real-IP (if from trusted proxy)
    2. X-Forwarded-For leftmost non-trusted IP (if from trusted proxy)
    3. Direct connection IP (remote_addr)
    """
    remote_ip = _normalize_ip(remote_addr)

    if not trust_xff_from_proxies:
        return remote_ip

    if not _is_trusted_proxy(remote_ip):
        return remote_ip

    # Check X-Real-IP first (simpler, less prone to spoofing)
    x_real_ip = headers.get("X-Real-IP", "").strip()
    if x_real_ip:
        return _normalize_ip(x_real_ip)

    # Parse X-Forwarded-For: client, proxy1, proxy2, ...
    xff = headers.get("X-Forwarded-For", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        # Find leftmost non-trusted IP (the real client)
        for ip in parts:
            normalized = _normalize_ip(ip)
            if not _is_trusted_proxy(normalized):
                return normalized

    return remote_ip


def sanitize_rate_limit_key_component(value: str) -> str:
    """Sanitize a component used in rate limit keys.

    Prevents injection attacks via malformed IPs or tokens.
    Replaces colons with underscores to prevent key injection.
    """
    if value is None:
        return ""
    value = str(value)
    # Replace colon to prevent Redis key injection, remove newlines
    return value.replace(":", "_").replace("\n", "").replace("\r", "").strip()


def normalize_rate_limit_path(path: str) -> str:
    """Normalize URL path for rate limit endpoint matching.

    Handles:
    - URL decoding
    - Path traversal prevention
    - Trailing slash normalization
    - Multiple slash collapse
    """
    if not path:
        return "/"

    # Decode URL-encoded characters
    decoded = unquote(path)

    # Normalize path (resolves .., collapses //)
    normalized = posixpath.normpath(decoded)

    # Ensure leading slash
    if not normalized.startswith("/"):
        normalized = "/" + normalized

    # Remove trailing slash (except for root)
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    # Lowercase for consistent matching
    normalized = normalized.lower()

    return normalized


__all__ = [
    "DEFAULT_RATE_LIMIT",
    "IP_RATE_LIMIT",
    "BURST_MULTIPLIER",
    "TRUSTED_PROXIES",
    "_normalize_ip",
    "_is_trusted_proxy",
    "_extract_client_ip",
    "sanitize_rate_limit_key_component",
    "normalize_rate_limit_path",
    "logger",
]
