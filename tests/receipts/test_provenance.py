"""Tests for best-effort operational provenance receipts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestEmitOperationalReceipt:
    def test_persists_signed_receipt_when_dependencies_available(self) -> None:
        from aragora.receipts.provenance import emit_operational_receipt

        facade = MagicMock()
        signer = MagicMock()
        signed = MagicMock()
        signed.signature = "sig-123"
        signed.signature_metadata.key_id = "key-1"
        signed.signature_metadata.timestamp = "2026-03-19T00:00:00+00:00"
        signed.signature_metadata.algorithm = "hmac-sha256"
        signer.sign.return_value = signed

        with (
            patch("aragora.receipts.provenance._get_facade", return_value=facade),
            patch("aragora.receipts.provenance._get_signer", return_value=signer),
        ):
            receipt_id = emit_operational_receipt(
                source="boss_loop",
                action="run_completed",
                actor="boss-loop",
                inputs={"run_id": "boss-123"},
                outputs={"stop_reason": "no_suitable_issue"},
                verdict="blocked",
                confidence=0.5,
            )

        assert receipt_id is not None
        facade.persist_and_save.assert_called_once()
        persisted_receipt_id, payload = facade.persist_and_save.call_args.args[:2]
        assert persisted_receipt_id == receipt_id
        assert payload["receipt_type"] == "operational"
        assert payload["source"] == "boss_loop"
        assert payload["action"] == "run_completed"
        assert payload["verdict"] == "blocked"
        kwargs = facade.persist_and_save.call_args.kwargs
        assert kwargs["signature"] == "sig-123"
        assert kwargs["signature_key_id"] == "key-1"
        assert kwargs["state"] == "CREATED"

    def test_returns_none_when_facade_is_unavailable(self) -> None:
        from aragora.receipts.provenance import emit_operational_receipt

        with patch("aragora.receipts.provenance._get_facade", return_value=None):
            receipt_id = emit_operational_receipt(
                source="ralph",
                action="campaign_completed",
                actor="ralph-123",
                inputs={},
                outputs={},
                verdict="pass",
            )

        assert receipt_id is None

    def test_signing_failure_does_not_block_persistence(self) -> None:
        from aragora.receipts.provenance import emit_operational_receipt

        facade = MagicMock()
        signer = MagicMock()
        signer.sign.side_effect = RuntimeError("no signer")

        with (
            patch("aragora.receipts.provenance._get_facade", return_value=facade),
            patch("aragora.receipts.provenance._get_signer", return_value=signer),
        ):
            receipt_id = emit_operational_receipt(
                source="ralph",
                action="escalated",
                actor="ralph-123",
                inputs={"campaign_id": "camp-1"},
                outputs={"reason": "budget"},
                verdict="escalated",
            )

        assert receipt_id is not None
        assert facade.persist_and_save.call_args.kwargs["signature"] is None

    def test_content_hash_changes_when_payload_changes(self) -> None:
        from aragora.receipts.provenance import _content_hash

        first = _content_hash(
            receipt_id="r-1",
            source="ralph",
            action="campaign_completed",
            actor="ralph-123",
            inputs={"campaign_id": "camp-1"},
            outputs={"status": "completed"},
            verdict="pass",
            confidence=1.0,
        )
        second = _content_hash(
            receipt_id="r-1",
            source="ralph",
            action="campaign_completed",
            actor="ralph-123",
            inputs={"campaign_id": "camp-1"},
            outputs={"status": "escalated"},
            verdict="escalated",
            confidence=0.0,
        )

        assert first != second
