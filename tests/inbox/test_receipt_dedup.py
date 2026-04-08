from aragora.gauntlet.signing import HMACSigner, ReceiptSigner
from aragora.inbox.trust_wedge import ActionIntent, InboxTrustWedgeStore, TriageDecision


def test_create_receipt_reuses_existing_receipt_for_same_gmail_message_id(tmp_path):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "wedge.db"))
    signer = ReceiptSigner(HMACSigner(secret_key=b"\x01" * 32, key_id="test-inbox-key"))
    intent = ActionIntent.create(
        provider="gmail",
        user_id="user-1",
        message_id="18f4b6c2a1d9e5f7",
        action="archive",
        content_hash=ActionIntent.compute_content_hash("subject", "body"),
        synthesized_rationale="Debated rationale",
        confidence=0.91,
        provider_route="openrouter-fallback",
    )
    decision = TriageDecision.create(
        final_action="archive",
        confidence=0.91,
        dissent_summary="none",
    )
    try:
        first = store.create_receipt(intent, decision, signer=signer)
        second = store.create_receipt(intent, decision, signer=signer)
        stored = store.get_receipt_by_message_id(
            intent.message_id, provider="gmail", user_id="user-1"
        )
        assert stored is not None
        assert second.receipt.receipt_id == first.receipt.receipt_id
        assert stored.receipt.receipt_id == first.receipt.receipt_id
        assert len(store.list_receipts()) == 1
    finally:
        store.close()
