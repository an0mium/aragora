"""
Webhook Handling Mixin for Chat Platform Connectors.

Contains methods for webhook verification and event parsing.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import WebhookEvent

logger = logging.getLogger(__name__)


class WebhookMixin:
    """
    Mixin providing webhook handling for chat connectors.

    Includes:
    - Webhook signature verification
    - Webhook event parsing
    """

    # These attributes are expected from the base class
    signing_secret: str | None

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier."""
        raise NotImplementedError

    # ==========================================================================
    # Webhook Handling
    # ==========================================================================

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """
        Verify webhook signature for security.

        Default implementation performs HMAC-SHA256 verification using
        ``signing_secret`` if configured.  Looks for the signature in
        common header locations (``X-Signature-256``, ``X-Hub-Signature-256``).
        In development mode with no ``signing_secret``, returns True with
        a warning.  In production, fails closed (returns False).

        Subclasses should override for platform-specific verification
        (e.g., Discord Ed25519, Telegram secret token header).

        Args:
            headers: HTTP headers from the webhook request
            body: Raw request body

        Returns:
            True if signature is valid
        """
        import hashlib
        import hmac as _hmac
        import os as _os

        if not self.signing_secret:
            env = _os.environ.get("ARAGORA_ENV", "production").lower()
            is_production = env not in ("development", "dev", "local", "test")
            if is_production:
                logger.error(
                    f"SECURITY: {self.platform_name} signing_secret not configured "
                    f"in production. Rejecting webhook to prevent signature bypass."
                )
                return False
            logger.warning(
                f"{self.platform_name} signing_secret not set - skipping verification. "
                f"This is only acceptable in development!"
            )
            return True

        # Look for signature in common header locations (case-insensitive)
        signature = ""
        for header_name in (
            "X-Signature-256",
            "x-signature-256",
            "X-Hub-Signature-256",
            "x-hub-signature-256",
        ):
            if header_name in headers:
                signature = headers[header_name]
                break

        if not signature:
            logger.warning(f"{self.platform_name} webhook missing signature header")
            return False

        # Strip sha256= prefix if present
        sig_value = signature
        if sig_value.startswith("sha256="):
            sig_value = sig_value[7:]

        computed = _hmac.new(
            self.signing_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        if _hmac.compare_digest(computed, sig_value):
            return True

        logger.warning(f"{self.platform_name} webhook signature mismatch")
        return False

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> "WebhookEvent":
        """
        Parse a webhook payload into a WebhookEvent.

        Default implementation parses the body as JSON and wraps it in a
        generic ``WebhookEvent``.  Subclasses should override for
        platform-specific event parsing (message types, interactions, etc.).

        Args:
            headers: HTTP headers from the request
            body: Raw request body

        Returns:
            Parsed WebhookEvent
        """
        import json

        from ..models import WebhookEvent

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(f"{self.platform_name} webhook body is not valid JSON")
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
                metadata={"error": "invalid_json"},
            )

        # Try to infer event type from common payload structures
        event_type = (
            payload.get("type") or payload.get("event_type") or payload.get("event", {}).get("type")
            if isinstance(payload, dict)
            else None
        ) or "unknown"

        return WebhookEvent(
            platform=self.platform_name,
            event_type=str(event_type),
            raw_payload=payload if isinstance(payload, dict) else {"data": payload},
        )


__all__ = ["WebhookMixin"]
