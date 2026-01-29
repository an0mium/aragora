"""
OAuth URL validation utilities.

Provides security checks for redirect URLs to prevent open redirect attacks.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from .config import _get_allowed_redirect_hosts

logger = logging.getLogger(__name__)


def _validate_redirect_url(redirect_url: str) -> bool:
    """
    Validate that redirect URL is in the allowed hosts list and uses safe scheme.

    This prevents open redirect vulnerabilities where an attacker could
    craft an OAuth URL that redirects tokens to a malicious domain or uses
    dangerous URL schemes (javascript:, data:, etc.).

    Args:
        redirect_url: The URL to validate

    Returns:
        True if URL is allowed, False otherwise
    """
    try:
        parsed = urlparse(redirect_url)

        # Security: Only allow http/https schemes to prevent javascript:/data:/etc attacks
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"oauth_redirect_blocked: scheme={parsed.scheme} not allowed")
            return False

        host = parsed.hostname
        if not host:
            return False

        # Normalize host for comparison
        host = host.lower()

        # Get allowed hosts at runtime from Secrets Manager
        allowed_hosts = _get_allowed_redirect_hosts()

        # Check against allowlist
        if host in allowed_hosts:
            return True

        # Check if it's a subdomain of allowed hosts
        for allowed in allowed_hosts:
            if host.endswith(f".{allowed}"):
                return True

        logger.warning(f"oauth_redirect_blocked: host={host} not in allowlist")
        return False
    except Exception as e:
        logger.warning(f"oauth_redirect_validation_error: {e}")
        return False
