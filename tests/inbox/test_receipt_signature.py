"""Tests for DurableFileSigner signature verification lifecycle.

Covers the complete signature lifecycle:
1. Create a receipt and sign it with DurableFileSigner
2. Verify the signature is valid
3. Tamper with the signed data
4. Verify that tampering is properly detected
"""

from __future__ import annotations

import os
import tempfile

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    ReceiptSigner,
    SignedReceipt,
)


@pytest.fixture()
def tmp_signer():
    """Create a DurableFileSigner with a temp key file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_path = os.path.join(tmpdir, "test.key")
        signer = DurableFileSigner(key_path=key_path)
        yield ReceiptSigner(backend=signer)


SAMPLE_RECEIPT = {
    "receipt_id": "r-sig-001",
    "debate_id": "debate-sig-001",
    "task": "Approve production deploy",
    "verdict": "approved",
    "confidence": 0.92,
    "participants": ["claude", "gpt"],
}


class TestDurableFileSignerLifecycle:
    """Full sign → verify → tamper → detect lifecycle."""

    def test_sign_and_verify_valid(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        assert isinstance(signed, SignedReceipt)
        assert signed.signature
        assert tmp_signer.verify(signed) is True

    def test_tampered_receipt_detected(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        assert tmp_signer.verify(signed) is True

        # Tamper with the receipt data
        signed.receipt_data["verdict"] = "TAMPERED"
        assert tmp_signer.verify(signed) is False

    def test_tampered_confidence_detected(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        signed.receipt_data["confidence"] = 0.01
        assert tmp_signer.verify(signed) is False

    def test_tampered_signature_detected(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        # Corrupt the base64 signature
        import base64

        bad_sig = base64.b64encode(b"\x00" * 32).decode("ascii")
        signed.signature = bad_sig
        assert tmp_signer.verify(signed) is False

    def test_different_key_rejects(self):
        """A receipt signed with one key cannot be verified with another."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signer1 = ReceiptSigner(
                backend=DurableFileSigner(key_path=os.path.join(tmpdir, "k1.key"))
            )
            signer2 = ReceiptSigner(
                backend=DurableFileSigner(key_path=os.path.join(tmpdir, "k2.key"))
            )
            signed = signer1.sign(SAMPLE_RECEIPT)
            assert signer1.verify(signed) is True
            assert signer2.verify(signed) is False

    def test_key_persistence_across_instances(self):
        """Key loaded from disk produces identical signatures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "persist.key")
            s1 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
            signed = s1.sign(SAMPLE_RECEIPT)

            s2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
            assert s2.verify(signed) is True

    def test_roundtrip_json_serialization(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)
        assert tmp_signer.verify(restored) is True

    def test_tampered_after_roundtrip_detected(self, tmp_signer: ReceiptSigner):
        signed = tmp_signer.sign(SAMPLE_RECEIPT)
        restored = SignedReceipt.from_json(signed.to_json())
        restored.receipt_data["task"] = "HACKED"
        assert tmp_signer.verify(restored) is False
