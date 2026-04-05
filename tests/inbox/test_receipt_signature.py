from __future__ import annotations

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner


def test_durable_file_signer_rejects_tampered_receipt(tmp_path):
    key_path = tmp_path / "receipt-signing.key"
    signer = ReceiptSigner(DurableFileSigner(key_path=str(key_path)))
    verifier = ReceiptSigner(DurableFileSigner(key_path=str(key_path)))
    receipt = {
        "receipt_id": "rcpt-123",
        "provider": "gmail",
        "message_id": "msg-123",
        "action": "archive",
        "confidence": 0.91,
        "timestamp": "2026-04-05T12:00:00Z",
    }

    signed = signer.sign(receipt)

    assert verifier.verify(signed) is True

    tampered = signed.to_dict()
    tampered["receipt"]["action"] = "trash"

    assert verifier.verify_dict(tampered) is False
