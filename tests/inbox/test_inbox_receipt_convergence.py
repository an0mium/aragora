"""Tests for inbox receipt convergence — Phase 3 Decision Integrity Kernel.

Validates:
- PersistedReceipt has canonical_receipt_id field
- ReceiptState.normalize works case-insensitively
- Receipt creation in InboxTrustWedgeStore persists to canonical facade
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.gauntlet.receipt_store import ReceiptState as GauntletReceiptState
from aragora.inbox.trust_wedge import (
    PersistedReceipt,
    ReceiptState,
)


class TestPersistedReceiptCanonicalId:
    """PersistedReceipt has the new canonical_receipt_id field."""

    def test_field_defaults_to_none(self):
        receipt = PersistedReceipt(
            receipt_id="r-100",
            intent_hash="hash",
            signature="sig",
            signing_key_id="key-1",
            state=ReceiptState.CREATED,
            created_at=datetime.now(timezone.utc),
        )
        assert receipt.canonical_receipt_id is None

    def test_field_can_be_set(self):
        receipt = PersistedReceipt(
            receipt_id="r-101",
            intent_hash="hash",
            signature="sig",
            signing_key_id="key-1",
            state=ReceiptState.CREATED,
            created_at=datetime.now(timezone.utc),
            canonical_receipt_id="canonical-101",
        )
        assert receipt.canonical_receipt_id == "canonical-101"

    def test_to_dict_includes_canonical_id_when_set(self):
        receipt = PersistedReceipt(
            receipt_id="r-102",
            intent_hash="hash",
            signature="sig",
            signing_key_id="key-1",
            state=ReceiptState.CREATED,
            created_at=datetime.now(timezone.utc),
            canonical_receipt_id="canonical-102",
        )
        d = receipt.to_dict()
        assert d["canonical_receipt_id"] == "canonical-102"

    def test_to_dict_omits_canonical_id_when_none(self):
        receipt = PersistedReceipt(
            receipt_id="r-103",
            intent_hash="hash",
            signature="sig",
            signing_key_id="key-1",
            state=ReceiptState.CREATED,
            created_at=datetime.now(timezone.utc),
        )
        d = receipt.to_dict()
        assert "canonical_receipt_id" not in d


class TestReceiptStateNormalize:
    """GauntletReceiptState.normalize() handles case-insensitive input."""

    def test_uppercase(self):
        assert GauntletReceiptState.normalize("CREATED") == GauntletReceiptState.CREATED

    def test_lowercase(self):
        assert GauntletReceiptState.normalize("approved") == GauntletReceiptState.APPROVED

    def test_mixed_case(self):
        assert GauntletReceiptState.normalize("Executed") == GauntletReceiptState.EXECUTED

    def test_expired(self):
        assert GauntletReceiptState.normalize("expired") == GauntletReceiptState.EXPIRED

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            GauntletReceiptState.normalize("invalid_state")


class TestReceiptCreationPersistsToFacade:
    """InboxTrustWedgeStore.create_receipt best-effort persists to facade."""

    def test_facade_called_on_create(self):
        """Verify the canonical facade is called during receipt creation."""
        mock_facade = MagicMock()

        with (
            patch(
                "aragora.pipeline.receipt_store_facade.get_receipt_store_facade",
                return_value=mock_facade,
            ),
            patch(
                "aragora.inbox.trust_wedge.get_inbox_trust_wedge_signer",
            ) as mock_signer_fn,
        ):
            # Set up signer mock
            mock_signer = MagicMock()
            mock_signed = MagicMock()
            mock_signed.signature = "test-sig"
            mock_signed.signature_metadata.key_id = "key-1"
            mock_signed.signature_metadata.timestamp = "2026-01-01T00:00:00"
            mock_signed.signature_metadata.algorithm = "hmac-sha256"
            mock_signed.to_dict.return_value = {"signature": "test-sig"}
            mock_signer.sign.return_value = mock_signed
            mock_signer_fn.return_value = mock_signer

            # Import and create store (will try to init SQLite)
            from aragora.inbox.trust_wedge import (
                ActionIntent,
                InboxTrustWedgeStore,
                TriageDecision,
            )

            try:
                store = InboxTrustWedgeStore(db_path=":memory:")
            except (OSError, RuntimeError):
                pytest.skip("Could not create in-memory trust wedge store")

            intent = ActionIntent.create(
                provider="gmail",
                message_id="msg-001",
                action="archive",
                content_hash="hash123",
                synthesized_rationale="test rationale",
                confidence=0.9,
                provider_route="direct",
            )
            decision = TriageDecision.create(
                final_action="archive",
                confidence=0.9,
                dissent_summary="no dissent",
            )

            envelope = store.create_receipt(intent, decision)

            # Verify facade was called
            mock_facade.persist_and_save.assert_called_once()
            call_args = mock_facade.persist_and_save.call_args
            assert call_args[0][0] == envelope.receipt.receipt_id
            assert call_args[0][1]["verdict"] == "CONDITIONAL"
            assert call_args[0][1]["confidence"] == pytest.approx(0.9)
            assert call_args[1]["signature"] == "test-sig"
            assert call_args[1]["state"] == "CREATED"

            # Verify canonical_receipt_id was set
            assert envelope.receipt.canonical_receipt_id == envelope.receipt.receipt_id

    def test_facade_failure_does_not_break_create(self):
        """If facade import fails, receipt creation still succeeds."""
        with (
            patch(
                "aragora.inbox.trust_wedge.get_inbox_trust_wedge_signer",
            ) as mock_signer_fn,
            patch.dict(
                "sys.modules",
                {"aragora.pipeline.receipt_store_facade": None},
            ),
        ):
            mock_signer = MagicMock()
            mock_signed = MagicMock()
            mock_signed.signature = "test-sig"
            mock_signed.signature_metadata.key_id = "key-1"
            mock_signed.signature_metadata.timestamp = "2026-01-01T00:00:00"
            mock_signed.signature_metadata.algorithm = "hmac-sha256"
            mock_signed.to_dict.return_value = {"signature": "test-sig"}
            mock_signer.sign.return_value = mock_signed
            mock_signer_fn.return_value = mock_signer

            from aragora.inbox.trust_wedge import (
                ActionIntent,
                InboxTrustWedgeStore,
                TriageDecision,
            )

            try:
                store = InboxTrustWedgeStore(db_path=":memory:")
            except (OSError, RuntimeError):
                pytest.skip("Could not create in-memory trust wedge store")

            intent = ActionIntent.create(
                provider="gmail",
                message_id="msg-002",
                action="star",
                content_hash="hash456",
                synthesized_rationale="test",
                confidence=0.8,
                provider_route="direct",
            )
            decision = TriageDecision.create(
                final_action="star",
                confidence=0.8,
                dissent_summary="minor dissent",
            )

            # Should succeed even though facade is unavailable
            envelope = store.create_receipt(intent, decision)
            assert envelope.receipt.receipt_id is not None
            assert envelope.receipt.canonical_receipt_id is None
