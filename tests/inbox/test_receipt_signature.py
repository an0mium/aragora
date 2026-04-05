"""Tests for DurableFileSigner signature verification.

Verifies that:
1. A receipt signed with DurableFileSigner produces a valid signature
2. The same signer (reloaded from disk) can verify the signature
3. Tampering with signed data is detected
4. A different key rejects the signature
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
def tmp_key_path(tmp_path):
    """Return a temporary key file path (file does not yet exist)."""
    return str(tmp_path / "test_signing.key")


def _sample_receipt() -> dict:
    return {
        "receipt_id": "r-sig-001",
        "debate_id": "debate-sig-001",
        "task": "Evaluate deployment risk",
        "verdict": "APPROVE",
        "confidence": 0.92,
        "participants": ["claude", "gpt"],
    }


class TestDurableFileSignerSignatureVerification:
    """End-to-end sign → verify → tamper → detect cycle."""

    def test_sign_and_verify_receipt(self, tmp_key_path):
        """A freshly signed receipt verifies successfully."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)

        receipt_data = _sample_receipt()
        signed = signer.sign(receipt_data)

        assert isinstance(signed, SignedReceipt)
        assert signer.verify(signed) is True

    def test_reloaded_signer_verifies(self, tmp_key_path):
        """A second DurableFileSigner loading the same key file verifies."""
        backend1 = DurableFileSigner(key_path=tmp_key_path)
        signer1 = ReceiptSigner(backend=backend1)
        signed = signer1.sign(_sample_receipt())

        # Reload from the same key file
        backend2 = DurableFileSigner(key_path=tmp_key_path)
        signer2 = ReceiptSigner(backend=backend2)

        assert signer2.verify(signed) is True

    def test_tampered_verdict_detected(self, tmp_key_path):
        """Modifying the verdict after signing invalidates the signature."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)
        signed = signer.sign(_sample_receipt())

        # Tamper with the receipt data
        signed.receipt_data["verdict"] = "REJECT"

        assert signer.verify(signed) is False

    def test_tampered_confidence_detected(self, tmp_key_path):
        """Modifying a numeric field after signing invalidates the signature."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)
        signed = signer.sign(_sample_receipt())

        signed.receipt_data["confidence"] = 0.01

        assert signer.verify(signed) is False

    def test_added_field_detected(self, tmp_key_path):
        """Adding a new field after signing invalidates the signature."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)
        signed = signer.sign(_sample_receipt())

        signed.receipt_data["injected"] = "malicious"

        assert signer.verify(signed) is False

    def test_different_key_rejects(self, tmp_path):
        """A receipt signed with one key is rejected by a different key."""
        key_a = str(tmp_path / "key_a.key")
        key_b = str(tmp_path / "key_b.key")

        signer_a = ReceiptSigner(backend=DurableFileSigner(key_path=key_a))
        signer_b = ReceiptSigner(backend=DurableFileSigner(key_path=key_b))

        signed = signer_a.sign(_sample_receipt())

        # Verify with a different key must fail
        assert signer_b.verify(signed) is False

    def test_roundtrip_via_json(self, tmp_key_path):
        """Serialize to JSON and back; verification still passes."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)
        signed = signer.sign(_sample_receipt())

        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)

        assert signer.verify(restored) is True

    def test_roundtrip_tampered_json_detected(self, tmp_key_path):
        """Tamper with JSON payload after serialization; verification fails."""
        backend = DurableFileSigner(key_path=tmp_key_path)
        signer = ReceiptSigner(backend=backend)
        signed = signer.sign(_sample_receipt())

        json_str = signed.to_json().replace('"APPROVE"', '"DENY"')
        restored = SignedReceipt.from_json(json_str)

        assert signer.verify(restored) is False
