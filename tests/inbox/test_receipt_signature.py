"""Tests for DurableFileSigner signature verification and tampering detection.

Covers the complete signature lifecycle:
1. Create a receipt and sign it with DurableFileSigner
2. Verify the signature is valid
3. Tamper with receipt data and confirm detection
4. Reload key from disk and verify signature persists
"""

from __future__ import annotations

import pytest

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner, SignedReceipt


@pytest.fixture()
def signer(tmp_path):
    """Create a DurableFileSigner backed by a temp key file."""
    key_path = str(tmp_path / "test_signing.key")
    backend = DurableFileSigner(key_path=key_path)
    return ReceiptSigner(backend=backend), key_path


SAMPLE_RECEIPT = {
    "receipt_id": "r-001",
    "debate_id": "d-100",
    "decision": "approved",
    "confidence": 0.92,
    "participants": ["agent-a", "agent-b"],
}


class TestDurableFileSignerLifecycle:
    """End-to-end sign → verify → tamper → detect cycle."""

    def test_sign_and_verify(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        assert rs.verify(signed)

    def test_tamper_decision_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        signed.receipt_data = {**signed.receipt_data, "decision": "rejected"}
        assert not rs.verify(signed)

    def test_tamper_confidence_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        signed.receipt_data = {**signed.receipt_data, "confidence": 0.01}
        assert not rs.verify(signed)

    def test_tamper_add_field_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        signed.receipt_data = {**signed.receipt_data, "extra": "injected"}
        assert not rs.verify(signed)

    def test_tamper_remove_field_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        trimmed = {k: v for k, v in signed.receipt_data.items() if k != "participants"}
        signed.receipt_data = trimmed
        assert not rs.verify(signed)

    def test_tamper_signature_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        # Flip a character in the base64 signature
        sig = list(signed.signature)
        sig[0] = "A" if sig[0] != "A" else "B"
        signed.signature = "".join(sig)
        assert not rs.verify(signed)

    def test_key_persistence_across_instances(self, signer):
        rs, key_path = signer
        signed = rs.sign(SAMPLE_RECEIPT)

        # Create a second signer from the same key file
        backend2 = DurableFileSigner(key_path=key_path)
        rs2 = ReceiptSigner(backend=backend2)
        assert rs2.verify(signed)

    def test_different_key_rejects(self, tmp_path):
        key1 = str(tmp_path / "key1.key")
        key2 = str(tmp_path / "key2.key")
        rs1 = ReceiptSigner(backend=DurableFileSigner(key_path=key1))
        rs2 = ReceiptSigner(backend=DurableFileSigner(key_path=key2))

        signed = rs1.sign(SAMPLE_RECEIPT)
        assert not rs2.verify(signed)

    def test_serialization_roundtrip(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        json_str = signed.to_json()

        restored = SignedReceipt.from_json(json_str)
        assert rs.verify(restored)

    def test_tampered_json_roundtrip_detected(self, signer):
        rs, _ = signer
        signed = rs.sign(SAMPLE_RECEIPT)
        json_str = signed.to_json().replace('"approved"', '"denied"')

        restored = SignedReceipt.from_json(json_str)
        assert not rs.verify(restored)
