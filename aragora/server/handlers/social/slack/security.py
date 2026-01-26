"""
Slack security utilities.

Provides SSRF protection and request signature verification for Slack webhooks.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Allowed domains for Slack response URLs (SSRF protection)
SLACK_ALLOWED_DOMAINS = frozenset({"hooks.slack.com", "api.slack.com"})


def validate_slack_url(url: str) -> bool:
    """Validate that a URL is a legitimate Slack endpoint.

    This prevents SSRF attacks by ensuring we only POST to Slack's servers.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is a valid Slack endpoint, False otherwise
    """
    try:
        parsed = urlparse(url)
        # Must be HTTPS
        if parsed.scheme != "https":
            return False
        # Must be a Slack domain
        if parsed.netloc not in SLACK_ALLOWED_DOMAINS:
            return False
        return True
    except Exception as e:
        logger.debug(f"URL validation failed for slack: {e}")
        return False


class SignatureVerifierMixin:
    """Mixin providing Slack signature verification.

    Should be mixed into a handler class that has access to HTTP headers.
    """

    def verify_signature(self, handler: Any, body: str, signing_secret: str) -> bool:
        """Verify Slack request signature.

        Uses centralized webhook verification for consistent security handling.
        See: https://api.slack.com/authentication/verifying-requests-from-slack

        Args:
            handler: HTTP request handler
            body: Pre-read request body
            signing_secret: Signing secret to use (workspace-specific or global)

        Returns:
            True if signature is valid, False otherwise
        """
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        try:
            result = verify_slack_signature(
                timestamp=handler.headers.get("X-Slack-Request-Timestamp", ""),
                body=body,
                signature=handler.headers.get("X-Slack-Signature", ""),
                signing_secret=signing_secret or "",
            )
            if not result.verified and result.error:
                logger.warning(f"Slack signature verification failed: {result.error}")
            return result.verified
        except Exception as e:
            logger.exception(f"Unexpected signature verification error: {e}")
            return False
