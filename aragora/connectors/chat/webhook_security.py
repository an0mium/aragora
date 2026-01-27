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
    # Unified verifier classes
    "WebhookVerifier",
    "HMACVerifier",
    "Ed25519Verifier",
]


# =============================================================================
# Unified Webhook Verifier Protocol
# =============================================================================


from abc import ABC, abstractmethod


class WebhookVerifier(ABC):
    """
    Abstract base class for webhook signature verification.

    Provides a unified interface for verifying webhooks from different platforms.
    Each platform (Slack, Discord, WhatsApp, etc.) can use a specialized verifier.

    Usage:
        verifier = HMACVerifier(secret=signing_secret, source="slack")
        result = verifier.verify(headers, body)
        if not result.verified:
            raise WebhookVerificationError(result.source, result.error)
    """

    def __init__(self, source: str):
        """
        Initialize the verifier.

        Args:
            source: Name of the webhook source (slack, discord, whatsapp, etc.)
        """
        self.source = source

    @abstractmethod
    def verify(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookVerificationResult:
        """
        Verify webhook signature.

        Args:
            headers: Request headers (case-insensitive access recommended)
            body: Raw request body as bytes

        Returns:
            WebhookVerificationResult with verification status
        """
        pass

    def _get_header(self, headers: dict[str, str], key: str) -> str:
        """Get header value with case-insensitive lookup."""
        # Try exact match first
        if key in headers:
            return headers[key]
        # Try lowercase
        key_lower = key.lower()
        for k, v in headers.items():
            if k.lower() == key_lower:
                return v
        return ""


class HMACVerifier(WebhookVerifier):
    """
    HMAC-based webhook signature verifier.

    Supports:
    - Slack: HMAC-SHA256 with v0:timestamp:body format
    - WhatsApp/Meta: HMAC-SHA256 with sha256= prefix
    - Generic: Configurable format

    Example:
        # Slack-style verification
        verifier = HMACVerifier(
            secret=signing_secret,
            source="slack",
            algorithm="sha256",
            signature_header="X-Slack-Signature",
            timestamp_header="X-Slack-Request-Timestamp",
            signature_prefix="v0=",
            body_template="v0:{timestamp}:{body}",
        )

        # WhatsApp-style verification
        verifier = HMACVerifier(
            secret=app_secret,
            source="whatsapp",
            algorithm="sha256",
            signature_header="X-Hub-Signature-256",
            signature_prefix="sha256=",
        )
    """

    def __init__(
        self,
        secret: str,
        source: str,
        algorithm: str = "sha256",
        signature_header: str = "X-Signature",
        timestamp_header: Optional[str] = None,
        signature_prefix: str = "",
        body_template: str = "{body}",
        max_timestamp_age: int = 300,  # 5 minutes
    ):
        """
        Initialize HMAC verifier.

        Args:
            secret: Signing secret for HMAC
            source: Webhook source name
            algorithm: Hash algorithm (sha256, sha1, etc.)
            signature_header: Header containing the signature
            timestamp_header: Header containing timestamp (for replay protection)
            signature_prefix: Prefix before hex signature (e.g., "sha256=", "v0=")
            body_template: Template for HMAC message (use {timestamp}, {body})
            max_timestamp_age: Maximum age of timestamp in seconds
        """
        super().__init__(source)
        self.secret = secret
        self.algorithm = algorithm
        self.signature_header = signature_header
        self.timestamp_header = timestamp_header
        self.signature_prefix = signature_prefix
        self.body_template = body_template
        self.max_timestamp_age = max_timestamp_age

    def verify(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookVerificationResult:
        """Verify HMAC signature."""
        import hashlib
        import hmac
        import time

        # Check if secret is configured
        if not self.secret:
            if should_allow_unverified(self.source):
                return log_verification_attempt(
                    source=self.source,
                    success=True,
                    method="bypassed",
                    error="No signing secret - verification skipped (dev mode)",
                )
            return log_verification_attempt(
                source=self.source,
                success=False,
                method=f"hmac-{self.algorithm}",
                error="signing_secret not configured",
            )

        # Get signature from headers
        signature = self._get_header(headers, self.signature_header)
        if not signature:
            return log_verification_attempt(
                source=self.source,
                success=False,
                method=f"hmac-{self.algorithm}",
                error=f"Missing {self.signature_header} header",
            )

        # Get and validate timestamp if configured
        timestamp = ""
        if self.timestamp_header:
            timestamp = self._get_header(headers, self.timestamp_header)
            if not timestamp:
                return log_verification_attempt(
                    source=self.source,
                    success=False,
                    method=f"hmac-{self.algorithm}",
                    error=f"Missing {self.timestamp_header} header",
                )
            # Validate timestamp age
            try:
                request_time = int(timestamp)
                if abs(time.time() - request_time) > self.max_timestamp_age:
                    return log_verification_attempt(
                        source=self.source,
                        success=False,
                        method=f"hmac-{self.algorithm}",
                        error=f"Request timestamp too old (>{self.max_timestamp_age}s)",
                    )
            except ValueError:
                return log_verification_attempt(
                    source=self.source,
                    success=False,
                    method=f"hmac-{self.algorithm}",
                    error="Invalid timestamp format",
                )

        # Build message for HMAC
        body_str = body.decode("utf-8") if isinstance(body, bytes) else body
        message = self.body_template.format(timestamp=timestamp, body=body_str)

        # Compute expected signature
        hash_func = getattr(hashlib, self.algorithm)
        expected_sig = hmac.new(
            self.secret.encode(),
            message.encode(),
            hash_func,
        ).hexdigest()

        # Add prefix back for comparison if needed
        if self.signature_prefix:
            expected_sig = self.signature_prefix + expected_sig

        # Timing-safe comparison
        if hmac.compare_digest(expected_sig, signature):
            return log_verification_attempt(
                source=self.source,
                success=True,
                method=f"hmac-{self.algorithm}",
            )
        else:
            return log_verification_attempt(
                source=self.source,
                success=False,
                method=f"hmac-{self.algorithm}",
                error="Signature mismatch",
            )


class Ed25519Verifier(WebhookVerifier):
    """
    Ed25519-based webhook signature verifier.

    Used by Discord for interaction endpoint verification.

    Requires PyNaCl library: pip install pynacl

    Example:
        verifier = Ed25519Verifier(
            public_key=discord_public_key,
            source="discord",
        )
        result = verifier.verify(headers, body)
    """

    def __init__(
        self,
        public_key: str,
        source: str = "discord",
        signature_header: str = "X-Signature-Ed25519",
        timestamp_header: str = "X-Signature-Timestamp",
    ):
        """
        Initialize Ed25519 verifier.

        Args:
            public_key: Hex-encoded Ed25519 public key
            source: Webhook source name
            signature_header: Header containing the signature
            timestamp_header: Header containing timestamp
        """
        super().__init__(source)
        self.public_key = public_key
        self.signature_header = signature_header
        self.timestamp_header = timestamp_header

    def verify(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookVerificationResult:
        """Verify Ed25519 signature."""
        # Check if PyNaCl is available
        try:
            from nacl.signing import VerifyKey
            from nacl.exceptions import BadSignatureError
        except ImportError:
            if should_allow_unverified(self.source):
                return log_verification_attempt(
                    source=self.source,
                    success=True,
                    method="bypassed",
                    error="PyNaCl not available - verification skipped (dev mode)",
                )
            return log_verification_attempt(
                source=self.source,
                success=False,
                method="ed25519",
                error="PyNaCl not installed - pip install pynacl",
            )

        # Check if public key is configured
        if not self.public_key:
            if should_allow_unverified(self.source):
                return log_verification_attempt(
                    source=self.source,
                    success=True,
                    method="bypassed",
                    error="No public key - verification skipped (dev mode)",
                )
            return log_verification_attempt(
                source=self.source,
                success=False,
                method="ed25519",
                error="public_key not configured",
            )

        # Get signature and timestamp from headers
        signature = self._get_header(headers, self.signature_header)
        timestamp = self._get_header(headers, self.timestamp_header)

        if not signature or not timestamp:
            return log_verification_attempt(
                source=self.source,
                success=False,
                method="ed25519",
                error=f"Missing {self.signature_header} or {self.timestamp_header} header",
            )

        # Verify signature
        try:
            verify_key = VerifyKey(bytes.fromhex(self.public_key))
            body_str = body.decode("utf-8") if isinstance(body, bytes) else body
            message = f"{timestamp}{body_str}".encode()
            verify_key.verify(message, bytes.fromhex(signature))
            return log_verification_attempt(
                source=self.source,
                success=True,
                method="ed25519",
            )
        except BadSignatureError:
            return log_verification_attempt(
                source=self.source,
                success=False,
                method="ed25519",
                error="Signature mismatch",
            )
        except Exception as e:
            return log_verification_attempt(
                source=self.source,
                success=False,
                method="ed25519",
                error=f"Verification error: {e}",
            )
