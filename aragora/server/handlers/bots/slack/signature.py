"""
Slack Signature Verification.

This module handles HMAC-SHA256 signature verification for Slack webhook requests.
"""

import hashlib
import hmac
import logging
import time

logger = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SECONDS = 60 * 5


def compute_slack_signature(body: bytes, timestamp: str, signing_secret: str) -> str:
    """Compute the expected HMAC-SHA256 signature for a Slack request.

    Args:
        body: Raw request body bytes
        timestamp: Unix timestamp string
        signing_secret: Slack signing secret

    Returns:
        The computed ``v0=...`` signature string.

    Raises:
        ValueError: If any required parameter is empty or has an invalid type.
    """
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    if not timestamp:
        raise ValueError("timestamp must not be empty")
    if not signing_secret:
        raise ValueError("signing_secret must not be empty")
    body_text = body.decode("utf-8")
    sig_basestring = f"v0:{timestamp}:{body_text}"
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
    *,
    now: float | None = None,
) -> bool:
    """Verify Slack request signature.

    Args:
        body: Raw request body bytes
        timestamp: X-Slack-Request-Timestamp header value
        signature: X-Slack-Signature header value
        signing_secret: Slack signing secret from app configuration
        now: Optional current unix timestamp (defaults to ``time.time()``).

    Returns:
        True if signature is valid, False otherwise
    """
    if not signature or not signing_secret:
        logger.warning("Missing Slack signature verification inputs")
        return False

    if now is not None and now < 0:
        logger.warning("Negative timestamp provided for Slack signature verification")
        return False

    current_time = int(now if now is not None else time.time())
    try:
        request_time = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Invalid Slack signature timestamp format")
        return False

    if abs(current_time - request_time) > TIMESTAMP_TOLERANCE_SECONDS:
        logger.warning("Slack signature timestamp too old")
        return False

    try:
        expected = compute_slack_signature(body, timestamp, signing_secret)
    except (AttributeError, UnicodeDecodeError):
        logger.warning("Invalid Slack request body encoding")
        return False

    return hmac.compare_digest(expected, signature)


__all__ = [
    "TIMESTAMP_TOLERANCE_SECONDS",
    "compute_slack_signature",
    "verify_slack_signature",
]
