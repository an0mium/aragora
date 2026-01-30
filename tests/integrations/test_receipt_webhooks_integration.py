"""Tests for Receipt Webhook notifications."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.integrations.receipt_webhooks import (
    ReceiptWebhookNotifier,
    ReceiptWebhookPayload,
    get_receipt_notifier,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_dispatcher():
    dispatcher = MagicMock()
    dispatcher.emit = MagicMock()
    return dispatcher


@pytest.fixture
def notifier(mock_dispatcher):
    return ReceiptWebhookNotifier(dispatcher=mock_dispatcher)


# =============================================================================
# ReceiptWebhookPayload Tests
# =============================================================================


class TestReceiptWebhookPayload:
    def test_basic_to_dict(self):
        payload = ReceiptWebhookPayload(
            event_type="receipt_generated",
            receipt_id="r-123",
            debate_id="d-456",
            timestamp=1700000000.0,
        )
        d = payload.to_dict()
        assert d["event_type"] == "receipt_generated"
        assert d["receipt_id"] == "r-123"
        assert d["debate_id"] == "d-456"
        assert d["timestamp"] == 1700000000.0
        # Optional fields should not appear
        assert "verdict" not in d
        assert "confidence" not in d
        assert "hash" not in d

    def test_full_to_dict(self):
        payload = ReceiptWebhookPayload(
            event_type="receipt_generated",
            receipt_id="r-123",
            debate_id="d-456",
            verdict="pass",
            confidence=0.95,
            hash="sha256:abc123",
            export_format="sarif",
            share_url="https://aragora.ai/receipt/r-123",
            error_message="some error",
            metadata={"agents": ["claude", "gpt4"]},
        )
        d = payload.to_dict()
        assert d["verdict"] == "pass"
        assert d["confidence"] == 0.95
        assert d["hash"] == "sha256:abc123"
        assert d["export_format"] == "sarif"
        assert d["share_url"] == "https://aragora.ai/receipt/r-123"
        assert d["error_message"] == "some error"
        assert d["metadata"]["agents"] == ["claude", "gpt4"]

    def test_partial_fields(self):
        payload = ReceiptWebhookPayload(
            event_type="receipt_verified",
            receipt_id="r-1",
            debate_id="d-1",
            hash="sha256:def",
        )
        d = payload.to_dict()
        assert "hash" in d
        assert "verdict" not in d
        assert "confidence" not in d

    def test_empty_metadata_excluded(self):
        payload = ReceiptWebhookPayload(
            event_type="receipt_verified",
            receipt_id="r-1",
            debate_id="d-1",
        )
        d = payload.to_dict()
        assert "metadata" not in d


# =============================================================================
# ReceiptWebhookNotifier Tests
# =============================================================================


class TestReceiptWebhookNotifier:
    def test_initialization_with_dispatcher(self, mock_dispatcher):
        notifier = ReceiptWebhookNotifier(dispatcher=mock_dispatcher)
        assert notifier.dispatcher is mock_dispatcher

    def test_lazy_dispatcher(self):
        notifier = ReceiptWebhookNotifier()
        with patch("aragora.integrations.receipt_webhooks.get_webhook_dispatcher") as mock_get:
            mock_get.return_value = MagicMock()
            _ = notifier.dispatcher
            mock_get.assert_called_once()

    def test_notify_receipt_generated(self, notifier, mock_dispatcher):
        notifier.notify_receipt_generated(
            receipt_id="r-123",
            debate_id="d-456",
            verdict="pass",
            confidence=0.92,
            hash="sha256:abc",
            agents=["claude", "gpt4"],
            rounds=3,
            findings_count=5,
        )
        mock_dispatcher.emit.assert_called_once()
        call_args = mock_dispatcher.emit.call_args
        assert call_args[0][0] == "receipt_generated"
        payload = call_args[0][1]
        assert payload["receipt_id"] == "r-123"
        assert payload["verdict"] == "pass"
        assert payload["confidence"] == 0.92
        assert payload["metadata"]["agents"] == ["claude", "gpt4"]
        assert payload["metadata"]["rounds"] == 3
        assert payload["metadata"]["findings_count"] == 5

    def test_notify_receipt_generated_minimal(self, notifier, mock_dispatcher):
        notifier.notify_receipt_generated(
            receipt_id="r-1",
            debate_id="d-1",
            verdict="fail",
            confidence=0.3,
            hash="sha256:xyz",
        )
        mock_dispatcher.emit.assert_called_once()
        payload = mock_dispatcher.emit.call_args[0][1]
        # When no agents/rounds/findings provided, metadata is empty and excluded from to_dict
        assert "metadata" not in payload or payload["metadata"] == {}

    def test_notify_receipt_verified(self, notifier, mock_dispatcher):
        notifier.notify_receipt_verified(
            receipt_id="r-123",
            debate_id="d-456",
            hash="sha256:original",
            computed_hash="sha256:computed",
            valid=True,
        )
        mock_dispatcher.emit.assert_called_once()
        call_args = mock_dispatcher.emit.call_args
        assert call_args[0][0] == "receipt_verified"
        payload = call_args[0][1]
        assert payload["hash"] == "sha256:original"
        assert payload["metadata"]["computed_hash"] == "sha256:computed"
        assert payload["metadata"]["valid"] is True

    def test_notify_receipt_exported(self, notifier, mock_dispatcher):
        notifier.notify_receipt_exported(
            receipt_id="r-123",
            debate_id="d-456",
            export_format="sarif",
            file_size=4096,
        )
        mock_dispatcher.emit.assert_called_once()
        payload = mock_dispatcher.emit.call_args[0][1]
        assert payload["export_format"] == "sarif"
        assert payload["metadata"]["file_size"] == 4096

    def test_notify_receipt_exported_no_size(self, notifier, mock_dispatcher):
        notifier.notify_receipt_exported(
            receipt_id="r-1",
            debate_id="d-1",
            export_format="json",
        )
        payload = mock_dispatcher.emit.call_args[0][1]
        assert "file_size" not in payload.get("metadata", {})

    def test_notify_receipt_shared(self, notifier, mock_dispatcher):
        notifier.notify_receipt_shared(
            receipt_id="r-123",
            debate_id="d-456",
            share_url="https://aragora.ai/receipt/r-123",
            expires_at="2024-12-31T23:59:59Z",
            allow_download=True,
        )
        mock_dispatcher.emit.assert_called_once()
        payload = mock_dispatcher.emit.call_args[0][1]
        assert payload["share_url"] == "https://aragora.ai/receipt/r-123"
        assert payload["metadata"]["expires_at"] == "2024-12-31T23:59:59Z"
        assert payload["metadata"]["allow_download"] is True

    def test_notify_receipt_shared_no_expiry(self, notifier, mock_dispatcher):
        notifier.notify_receipt_shared(
            receipt_id="r-1",
            debate_id="d-1",
            share_url="url",
            allow_download=False,
        )
        payload = mock_dispatcher.emit.call_args[0][1]
        assert "expires_at" not in payload["metadata"]
        assert payload["metadata"]["allow_download"] is False

    def test_notify_receipt_integrity_failed(self, notifier, mock_dispatcher):
        notifier.notify_receipt_integrity_failed(
            receipt_id="r-123",
            debate_id="d-456",
            expected_hash="sha256:expected",
            computed_hash="sha256:computed",
            error_message="Hash mismatch detected",
        )
        mock_dispatcher.emit.assert_called_once()
        call_args = mock_dispatcher.emit.call_args
        assert call_args[0][0] == "receipt_integrity_failed"
        payload = call_args[0][1]
        assert payload["error_message"] == "Hash mismatch detected"
        assert payload["metadata"]["expected_hash"] == "sha256:expected"
        assert payload["metadata"]["computed_hash"] == "sha256:computed"

    def test_notify_receipt_integrity_failed_no_error_msg(self, notifier, mock_dispatcher):
        notifier.notify_receipt_integrity_failed(
            receipt_id="r-1",
            debate_id="d-1",
            expected_hash="h1",
            computed_hash="h2",
        )
        payload = mock_dispatcher.emit.call_args[0][1]
        assert "error_message" not in payload

    def test_emit_handles_exception(self, mock_dispatcher):
        mock_dispatcher.emit.side_effect = Exception("dispatch failed")
        notifier = ReceiptWebhookNotifier(dispatcher=mock_dispatcher)
        # Should not raise
        notifier.notify_receipt_generated(
            receipt_id="r-1",
            debate_id="d-1",
            verdict="pass",
            confidence=0.5,
            hash="h1",
        )


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetReceiptNotifier:
    def test_singleton(self):
        import aragora.integrations.receipt_webhooks as mod

        mod._receipt_notifier = None
        n1 = get_receipt_notifier()
        n2 = get_receipt_notifier()
        assert n1 is n2
        mod._receipt_notifier = None
