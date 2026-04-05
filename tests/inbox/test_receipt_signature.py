"""Tests for receipt signature lifecycle using DurableFileSigner.

Covers: creation, signing, verification, tamper detection, and
cross-session persistence via DurableFileSigner's on-disk key.
"""

from __future__ import annotations

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    ReceiptSigner,
    SignedReceipt,
)


@pytest.fixture()
def signer(tmp_path):
    """ReceiptSigner backed by a DurableFileSigner in a temp directory."""
    key_path = str(tmp_path / "signing.key")
    backend = DurableFileSigner(key_path=key_path, key_id="test-durable")
    return ReceiptSigner(backend), key_path


def _sample_receipt() -> dict:
    return {
        "receipt_id": "r-001",
        "verdict": "PASS",
        "confidence": 0.95,
        "input_summary": "Should we adopt micro-services?",
        "input_hash": "abc123",
        "robustness_score": 0.88,
    }


class TestReceiptSignatureLifecycle:
    """Full lifecycle: create -> sign -> verify -> tamper -> detect."""

    def test_sign_and_verify(self, signer):
        receipt_signer, _ = signer
        receipt = _sample_receipt()

        signed = receipt_signer.sign(receipt)

        assert isinstance(signed, SignedReceipt)
        assert signed.signature
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signed.signature_metadata.key_id == "test-durable"
        assert receipt_signer.verify(signed) is True

    def test_tamper_receipt_data_detected(self, signer):
        receipt_signer, _ = signer
        signed = receipt_signer.sign(_sample_receipt())

        # Mutate the receipt payload after signing
        signed.receipt_data["verdict"] = "FAIL"

        assert receipt_signer.verify(signed) is False

    def test_tamper_signature_detected(self, signer):
        receipt_signer, _ = signer
        signed = receipt_signer.sign(_sample_receipt())

        # Corrupt the base64 signature
        signed.signature = "AAAA" + signed.signature[4:]

        assert receipt_signer.verify(signed) is False

    def test_cross_session_verify(self, tmp_path):
        """A receipt signed in one session verifies after re-loading the key."""
        key_path = str(tmp_path / "signing.key")

        signer1 = ReceiptSigner(DurableFileSigner(key_path=key_path))
        signed = signer1.sign(_sample_receipt())

        # Simulate a new process by creating a fresh signer from the same path
        signer2 = ReceiptSigner(DurableFileSigner(key_path=key_path))
        assert signer2.verify(signed) is True

    def test_different_key_rejects(self, tmp_path):
        """A receipt signed with one key fails verification with another."""
        signer_a = ReceiptSigner(DurableFileSigner(key_path=str(tmp_path / "a.key")))
        signer_b = ReceiptSigner(DurableFileSigner(key_path=str(tmp_path / "b.key")))

        signed = signer_a.sign(_sample_receipt())
        assert signer_b.verify(signed) is False

    def test_roundtrip_dict_serialization(self, signer):
        receipt_signer, _ = signer
        signed = receipt_signer.sign(_sample_receipt())

        restored = SignedReceipt.from_dict(signed.to_dict())
        assert receipt_signer.verify(restored) is True

    def test_roundtrip_json_serialization(self, signer):
        receipt_signer, _ = signer
        signed = receipt_signer.sign(_sample_receipt())

        restored = SignedReceipt.from_json(signed.to_json())
        assert receipt_signer.verify(restored) is True
