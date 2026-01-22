"""
Receipt webhook notifications.

Provides webhook notifications for gauntlet receipt lifecycle events:
- receipt_generated: New receipt created after gauntlet completion
- receipt_verified: Receipt integrity successfully verified
- receipt_exported: Receipt exported to external format
- receipt_shared: Shareable link created for receipt
- receipt_integrity_failed: Receipt integrity verification failed
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aragora.integrations.webhooks import (
    WebhookDispatcher,
    get_dispatcher as get_webhook_dispatcher,
)

logger = logging.getLogger(__name__)


@dataclass
class ReceiptWebhookPayload:
    """Payload structure for receipt webhook notifications."""

    event_type: str
    receipt_id: str
    debate_id: str
    timestamp: float = field(default_factory=time.time)
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    hash: Optional[str] = None
    export_format: Optional[str] = None
    share_url: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for webhook payload."""
        result = {
            "event_type": self.event_type,
            "receipt_id": self.receipt_id,
            "debate_id": self.debate_id,
            "timestamp": self.timestamp,
        }
        if self.verdict is not None:
            result["verdict"] = self.verdict
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.hash is not None:
            result["hash"] = self.hash
        if self.export_format is not None:
            result["export_format"] = self.export_format
        if self.share_url is not None:
            result["share_url"] = self.share_url
        if self.error_message is not None:
            result["error_message"] = self.error_message
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class ReceiptWebhookNotifier:
    """
    Webhook notifier for receipt lifecycle events.

    Integrates with the WebhookDispatcher to send notifications
    when receipt events occur.

    Usage:
        notifier = ReceiptWebhookNotifier()

        # When a receipt is generated
        notifier.notify_receipt_generated(
            receipt_id="receipt-123",
            debate_id="debate-456",
            verdict="pass",
            confidence=0.92,
            hash="sha256:abc123..."
        )

        # When a receipt is exported
        notifier.notify_receipt_exported(
            receipt_id="receipt-123",
            debate_id="debate-456",
            export_format="sarif"
        )
    """

    def __init__(self, dispatcher: Optional[WebhookDispatcher] = None):
        """Initialize with optional custom dispatcher."""
        self._dispatcher = dispatcher

    @property
    def dispatcher(self) -> WebhookDispatcher:
        """Get the webhook dispatcher (lazy initialization)."""
        if self._dispatcher is None:
            self._dispatcher = get_webhook_dispatcher()
        return self._dispatcher

    def _emit(self, payload: ReceiptWebhookPayload) -> None:
        """Emit a webhook event via the dispatcher."""
        try:
            self.dispatcher.emit(payload.event_type, payload.to_dict())  # type: ignore[attr-defined]
            logger.debug(
                f"Emitted receipt webhook: {payload.event_type} "
                f"for receipt {payload.receipt_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to emit receipt webhook: {e}")

    def notify_receipt_generated(
        self,
        receipt_id: str,
        debate_id: str,
        verdict: str,
        confidence: float,
        hash: str,
        agents: Optional[List[str]] = None,
        rounds: Optional[int] = None,
        findings_count: Optional[int] = None,
    ) -> None:
        """Notify that a new receipt has been generated.

        Args:
            receipt_id: Unique receipt identifier
            debate_id: Associated debate ID
            verdict: Gauntlet verdict (pass/fail/warn)
            confidence: Confidence score (0-1)
            hash: SHA-256 hash of receipt content
            agents: List of participating agents
            rounds: Number of debate rounds
            findings_count: Number of findings in the receipt
        """
        metadata = {}
        if agents:
            metadata["agents"] = agents
        if rounds is not None:
            metadata["rounds"] = rounds  # type: ignore[assignment]
        if findings_count is not None:
            metadata["findings_count"] = findings_count  # type: ignore[assignment]

        payload = ReceiptWebhookPayload(
            event_type="receipt_generated",
            receipt_id=receipt_id,
            debate_id=debate_id,
            verdict=verdict,
            confidence=confidence,
            hash=hash,
            metadata=metadata,
        )
        self._emit(payload)

    def notify_receipt_verified(
        self,
        receipt_id: str,
        debate_id: str,
        hash: str,
        computed_hash: str,
        valid: bool,
    ) -> None:
        """Notify that a receipt has been verified.

        Args:
            receipt_id: Receipt identifier
            debate_id: Associated debate ID
            hash: Original hash from receipt
            computed_hash: Hash computed during verification
            valid: Whether verification succeeded
        """
        payload = ReceiptWebhookPayload(
            event_type="receipt_verified",
            receipt_id=receipt_id,
            debate_id=debate_id,
            hash=hash,
            metadata={
                "computed_hash": computed_hash,
                "valid": valid,
            },
        )
        self._emit(payload)

    def notify_receipt_exported(
        self,
        receipt_id: str,
        debate_id: str,
        export_format: str,
        file_size: Optional[int] = None,
    ) -> None:
        """Notify that a receipt has been exported.

        Args:
            receipt_id: Receipt identifier
            debate_id: Associated debate ID
            export_format: Export format (json/html/markdown/sarif)
            file_size: Size of exported file in bytes
        """
        metadata = {}
        if file_size is not None:
            metadata["file_size"] = file_size

        payload = ReceiptWebhookPayload(
            event_type="receipt_exported",
            receipt_id=receipt_id,
            debate_id=debate_id,
            export_format=export_format,
            metadata=metadata,
        )
        self._emit(payload)

    def notify_receipt_shared(
        self,
        receipt_id: str,
        debate_id: str,
        share_url: str,
        expires_at: Optional[str] = None,
        allow_download: bool = True,
    ) -> None:
        """Notify that a receipt share link has been created.

        Args:
            receipt_id: Receipt identifier
            debate_id: Associated debate ID
            share_url: The shareable URL
            expires_at: Expiration timestamp (ISO 8601)
            allow_download: Whether download is allowed
        """
        metadata = {"allow_download": allow_download}
        if expires_at:
            metadata["expires_at"] = expires_at  # type: ignore[assignment]

        payload = ReceiptWebhookPayload(
            event_type="receipt_shared",
            receipt_id=receipt_id,
            debate_id=debate_id,
            share_url=share_url,
            metadata=metadata,
        )
        self._emit(payload)

    def notify_receipt_integrity_failed(
        self,
        receipt_id: str,
        debate_id: str,
        expected_hash: str,
        computed_hash: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Notify that a receipt integrity check has failed.

        Args:
            receipt_id: Receipt identifier
            debate_id: Associated debate ID
            expected_hash: Expected hash from receipt
            computed_hash: Hash computed during verification
            error_message: Optional error details
        """
        payload = ReceiptWebhookPayload(
            event_type="receipt_integrity_failed",
            receipt_id=receipt_id,
            debate_id=debate_id,
            hash=expected_hash,
            error_message=error_message,
            metadata={
                "expected_hash": expected_hash,
                "computed_hash": computed_hash,
            },
        )
        self._emit(payload)


# Singleton instance
_receipt_notifier: Optional[ReceiptWebhookNotifier] = None


def get_receipt_notifier() -> ReceiptWebhookNotifier:
    """Get the singleton receipt webhook notifier."""
    global _receipt_notifier
    if _receipt_notifier is None:
        _receipt_notifier = ReceiptWebhookNotifier()
    return _receipt_notifier


__all__ = [
    "ReceiptWebhookPayload",
    "ReceiptWebhookNotifier",
    "get_receipt_notifier",
]
