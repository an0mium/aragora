"""Inbox receipt signature lifecycle coverage."""

from __future__ import annotations

import copy

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner, SignedReceipt


def test_receipt_signature_lifecycle_detects_tampering(tmp_path) -> None:
    key_path = tmp_path / "receipt-signing.key"
    signer = ReceiptSigner(backend=DurableFileSigner(key_path=str(key_path)))
    receipt = {
        "receipt_id": "rcpt-001",
        "decision": "approve",
        "topic": "Rate limiter design",
        "consensus_score": 0.87,
        "agents": ["claude", "gpt-4", "gemini"],
    }

    signed = signer.sign(receipt)
    assert key_path.exists()
    assert signed.signature
    assert signed.signature_metadata.algorithm == "HMAC-SHA256"
    assert signed.signature_metadata.key_id.startswith("durable-")

    verifier = ReceiptSigner(backend=DurableFileSigner(key_path=str(key_path)))
    round_tripped = SignedReceipt.from_json(signed.to_json())
    assert verifier.verify(round_tripped) is True

    tampered_payload = copy.deepcopy(round_tripped.to_dict())
    tampered_payload["receipt"]["decision"] = "reject"
    tampered = SignedReceipt.from_dict(tampered_payload)

    assert tampered.signature == round_tripped.signature
    assert verifier.verify(tampered) is False
    assert verifier.verify(round_tripped) is True
