"""
Webhook Security Utilities.

Centralizes webhook verification logic with production-safe defaults.

SECURITY:
- ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS is ignored in production/staging environments
- All webhook sources (Slack, Teams, Discord, etc.) use this module
- Fails closed by default: unverified webhooks are rejected

Usage:
    from aragora.connectors.chat.webhook_security import (
        is_webhook_verification_required,
        should_allow_unverified,
        WebhookVerificationError,
    )

    if not self.signing_secret:
        if should_allow_unverified("slack"):
            logger.warning("Webhook verification skipped (dev mode)")
            return True
        raise WebhookVerificationError("slack", "signing_secret not configured")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class WebhookVerificationError(Exception):
    """
    Raised when webhook verification fails and cannot be bypassed.

    This exception indicates a security configuration issue that must be
    resolved before the webhook can be processed.
    """

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(
            f"Webhook verification failed for '{source}': {reason}. "
            f"Configure the signing secret or set ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS=true "
            f"(development environments only)."
        )


def get_environment() -> str:
    """Get the current environment."""
    return os.environ.get("ARAGORA_ENV", "development").lower()


def is_production_environment() -> bool:
    """Check if running in production or staging environment."""
    env = get_environment()
    return env in ("production", "prod", "staging", "stage")


def is_webhook_verification_required() -> bool:
    """
    Check if webhook verification is required.

    Always required in production/staging environments.
    Can be disabled in development with ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS.

    Returns:
        True if verification is required, False if it can be skipped
    """
    # SECURITY: Always require verification in production
    if is_production_environment():
        return True

    # Check override flag (development only)
    allow_unverified = os.environ.get("ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS", "").lower() in (
        "1",
        "true",
        "yes",
    )

    return not allow_unverified


def should_allow_unverified(source: str) -> bool:
    """
    Check if unverified webhooks should be allowed.

    SECURITY:
    - Always returns False in production/staging
    - Returns True only in development with explicit env var

    Args:
        source: Webhook source name (for logging)

    Returns:
        True if unverified webhooks are allowed, False otherwise
    """
    # SECURITY: Never allow unverified webhooks in production
    if is_production_environment():
        env = get_environment()
        logger.warning(
            f"Webhook verification for '{source}' cannot be bypassed in {env} environment. "
            f"ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS is ignored."
        )
        return False

    # Check if bypass is explicitly allowed
    allow_unverified = os.environ.get("ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if allow_unverified:
        logger.warning(
            f"Webhook verification for '{source}' bypassed in {get_environment()} environment. "
            f"This is a security risk - do not use in production."
        )

    return allow_unverified


@dataclass
class WebhookVerificationResult:
    """Result of webhook verification."""

    verified: bool
    source: str
    method: str
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.verified


def log_verification_attempt(
    source: str,
    success: bool,
    method: str,
    error: Optional[str] = None,
) -> WebhookVerificationResult:
    """
    Log a webhook verification attempt for audit purposes.

    Args:
        source: Webhook source (slack, teams, discord, etc.)
        success: Whether verification succeeded
        method: Verification method used
        error: Error message if verification failed

    Returns:
        WebhookVerificationResult
    """
    result = WebhookVerificationResult(
        verified=success,
        source=source,
        method=method,
        error=error,
    )

    if success:
        logger.debug(f"Webhook verified: source={source}, method={method}")
    else:
        logger.warning(
            f"Webhook verification failed: source={source}, method={method}, error={error}"
        )

    return result


def verify_slack_signature(
    timestamp: str,
    body: str | bytes,
    signature: str,
    signing_secret: str,
) -> WebhookVerificationResult:
    """
    Verify Slack webhook signature using HMAC-SHA256.

    This is the canonical implementation used by both SlackConnector
    and SlackHandler to ensure consistent verification.

    Args:
        timestamp: X-Slack-Request-Timestamp header value
        body: Request body (str or bytes)
        signature: X-Slack-Signature header value
        signing_secret: Slack signing secret

    Returns:
        WebhookVerificationResult with verification status

    Example:
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        result = verify_slack_signature(
            timestamp=headers.get("X-Slack-Request-Timestamp", ""),
            body=body,
            signature=headers.get("X-Slack-Signature", ""),
            signing_secret=SLACK_SIGNING_SECRET,
        )
        if not result.verified:
            return error_response(401, result.error)
    """
    import hashlib
    import hmac
    import time

    # Check required inputs
    if not timestamp or not signature:
        return log_verification_attempt(
            source="slack",
            success=False,
            method="hmac-sha256",
            error="Missing timestamp or signature header",
        )

    if not signing_secret:
        # Check if unverified webhooks are allowed (dev mode only)
        if should_allow_unverified("slack"):
            return log_verification_attempt(
                source="slack",
                success=True,
                method="bypassed",
                error="No signing secret - verification skipped (dev mode)",
            )
        return log_verification_attempt(
            source="slack",
            success=False,
            method="hmac-sha256",
            error="signing_secret not configured",
        )

    # Check timestamp to prevent replay attacks (5 minute window)
    try:
        request_time = int(timestamp)
        if abs(time.time() - request_time) > 300:
            return log_verification_attempt(
                source="slack",
                success=False,
                method="hmac-sha256",
                error="Request timestamp too old (>5 minutes)",
            )
    except ValueError:
        return log_verification_attempt(
            source="slack",
            success=False,
            method="hmac-sha256",
            error="Invalid timestamp format",
        )

    # Normalize body to string
    body_str = body.decode("utf-8") if isinstance(body, bytes) else body

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{body_str}"
    expected_sig = (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )

    # Timing-safe comparison
    if hmac.compare_digest(expected_sig, signature):
        return log_verification_attempt(
            source="slack",
            success=True,
            method="hmac-sha256",
        )
    else:
        return log_verification_attempt(
            source="slack",
            success=False,
            method="hmac-sha256",
            error="Signature mismatch",
        )


__all__ = [
    "WebhookVerificationError",
    "WebhookVerificationResult",
    "is_webhook_verification_required",
    "should_allow_unverified",
    "is_production_environment",
    "log_verification_attempt",
    "verify_slack_signature",
]
