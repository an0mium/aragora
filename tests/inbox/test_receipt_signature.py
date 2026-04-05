"""Tests for durable inbox receipt signature verification."""

from __future__ import annotations

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner
from aragora.inbox.trust_wedge import ActionIntent, InboxTrustWedgeStore, TriageDecision


def _build_intent() -> ActionIntent:
    return ActionIntent.create(
        provider="gmail",
        user_id="user-1",
        message_id="provider-msg-1",
        action="archive",
        content_hash=ActionIntent.compute_content_hash("Subject", "Body"),
        synthesized_rationale="Archive low-priority newsletter",
        confidence=0.93,
        provider_route="direct",
        debate_id="debate-123",
    )


def _build_decision() -> TriageDecision:
    return TriageDecision.create(
        final_action="archive",
        confidence=0.93,
        dissent_summary="No dissent",
    )


def test_durable_file_signer_verifies_receipt_and_detects_tampering(tmp_path) -> None:
    key_path = tmp_path / "inbox_wedge_signing.key"
    signer = ReceiptSigner(DurableFileSigner(key_path=str(key_path)))
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "wedge.db"))

    try:
        created = store.create_receipt(_build_intent(), _build_decision(), signer=signer)
        stored = store.get_receipt(created.receipt.receipt_id)

        assert stored is not None
        assert key_path.is_file()
        assert signer.verify(created.signed_receipt)

        reloaded_signer = ReceiptSigner(DurableFileSigner(key_path=str(key_path)))
        assert reloaded_signer.verify(stored.signed_receipt)

        tampered = stored.signed_receipt.to_dict()
        tampered["receipt"]["triage_decision"]["confidence"] = 0.01

        assert reloaded_signer.verify_dict(tampered) is False
    finally:
        store.close()
