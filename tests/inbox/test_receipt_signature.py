from __future__ import annotations

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner, SignedReceipt
from aragora.inbox.trust_wedge import ActionIntent, InboxTrustWedgeStore, TriageDecision


def _create_signed_receipt(tmp_path):
    signer = ReceiptSigner(DurableFileSigner(key_path=str(tmp_path / "signing.key")))
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "wedge.db"))
    envelope = store.create_receipt(
        ActionIntent.create(
            provider="gmail",
            user_id="user-1",
            message_id="msg-1",
            action="archive",
            content_hash=ActionIntent.compute_content_hash("subject", "body"),
            synthesized_rationale="Debated rationale",
            confidence=0.91,
            provider_route="direct",
            debate_id="debate-123",
        ),
        TriageDecision.create(final_action="archive", confidence=0.91, dissent_summary=""),
        signer=signer,
    )
    return store, envelope.signed_receipt


def test_durable_file_signer_verifies_signed_receipt(tmp_path):
    store, signed = _create_signed_receipt(tmp_path)
    try:
        verifier = ReceiptSigner(DurableFileSigner(key_path=str(tmp_path / "signing.key")))
        assert verifier.verify(signed)
    finally:
        store.close()


def test_durable_file_signer_detects_tampered_receipt_data(tmp_path):
    store, signed = _create_signed_receipt(tmp_path)
    try:
        verifier = ReceiptSigner(DurableFileSigner(key_path=str(tmp_path / "signing.key")))
        tampered = SignedReceipt(
            receipt_data={**signed.receipt_data, "action": "star"},
            signature=signed.signature,
            signature_metadata=signed.signature_metadata,
        )
        assert not verifier.verify(tampered)
    finally:
        store.close()
