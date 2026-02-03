"""
Slack Signature Verification.

This module handles HMAC-SHA256 signature verification for Slack webhook requests.
"""

import hashlib
import hmac
import logging
import time

logger = logging.getLogger(__name__)


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
) -> bool:
    """Verify Slack request signature.

    Args:
        body: Raw request body bytes
        timestamp: X-Slack-Request-Timestamp header value
        signature: X-Slack-Signature header value
        signing_secret: Slack signing secret from app configuration

    Returns:
        True if signature is valid, False otherwise
    """
    # Check timestamp to prevent replay attacks
    current_time = int(time.time())
    try:
        request_time = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Invalid Slack signature timestamp format")
        return False

    if abs(current_time - request_time) > 60 * 5:
        logger.warning("Slack signature timestamp too old")
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_signature = (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(my_signature, signature)


__all__ = [
    "verify_slack_signature",
]
