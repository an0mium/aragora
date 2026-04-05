"""Tests for receipt signature lifecycle with DurableFileSigner.

Covers the full signature lifecycle:
1. Create a receipt and sign it with DurableFileSigner
2. Verify the valid signature
3. Tamper with receipt data
4. Verify tampering detection
"""

from __future__ import annotations

import copy
import os
import tempfile

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    ReceiptSigner,
    SignatoryInfo,
    SignedReceipt,
)


@pytest.fixture()
def tmp_key_path(tmp_path):
    """Provide a temporary key file path."""
    return str(tmp_path / "test_signing.key")


@pytest.fixture()
def signer(tmp_key_path):
    """Create a ReceiptSigner backed by a DurableFileSigner."""
    backend = DurableFileSigner(key_path=tmp_key_path)
    return ReceiptSigner(backend=backend)


@pytest.fixture()
def sample_receipt():
    """A representative receipt data dict."""
    return {
        "receipt_id": "r-sig-001",
        "debate_id": "debate-sig-001",
        "task": "Should we deploy v2.0?",
        "verdict": "Yes, with staged rollout",
        "confidence": 0.91,
        "consensus_reached": True,
        "participants": ["claude", "gpt"],
        "rounds_used": 3,
    }


class TestReceiptSignatureLifecycle:
    """Full sign-verify-tamper-detect lifecycle."""

    def test_sign_and_verify_valid(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)

        assert isinstance(signed, SignedReceipt)
        assert signed.receipt_data == sample_receipt
        assert signed.signature  # non-empty
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signer.verify(signed) is True

    def test_tampered_verdict_detected(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)
        assert signer.verify(signed) is True

        # Tamper with the verdict
        signed.receipt_data["verdict"] = "TAMPERED"
        assert signer.verify(signed) is False

    def test_tampered_confidence_detected(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)

        signed.receipt_data["confidence"] = 0.01
        assert signer.verify(signed) is False

    def test_added_field_detected(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)

        signed.receipt_data["injected"] = "malicious"
        assert signer.verify(signed) is False

    def test_removed_field_detected(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)

        del signed.receipt_data["task"]
        assert signer.verify(signed) is False

    def test_tampered_signature_detected(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)

        # Corrupt the base64 signature
        signed.signature = "AAAA" + signed.signature[4:]
        assert signer.verify(signed) is False


class TestDurableFileSignerPersistence:
    """DurableFileSigner key survives across instances."""

    def test_second_instance_verifies_first(self, tmp_key_path, sample_receipt):
        signer1 = ReceiptSigner(backend=DurableFileSigner(key_path=tmp_key_path))
        signed = signer1.sign(sample_receipt)

        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=tmp_key_path))
        assert signer2.verify(signed) is True

    def test_different_key_rejects(self, tmp_path, sample_receipt):
        path_a = str(tmp_path / "key_a.key")
        path_b = str(tmp_path / "key_b.key")

        signer_a = ReceiptSigner(backend=DurableFileSigner(key_path=path_a))
        signed = signer_a.sign(sample_receipt)

        signer_b = ReceiptSigner(backend=DurableFileSigner(key_path=path_b))
        assert signer_b.verify(signed) is False


class TestSignatoryMetadata:
    """Signatory info is preserved in signed receipts."""

    def test_signatory_round_trip(self, signer, sample_receipt):
        info = SignatoryInfo(
            name="Alice",
            email="alice@example.com",
            role="Approver",
        )
        signed = signer.sign(sample_receipt, signatory=info)

        assert signed.signature_metadata.signatory is not None
        assert signed.signature_metadata.signatory.name == "Alice"
        assert signed.signature_metadata.signatory.role == "Approver"
        assert signer.verify(signed) is True

    def test_json_round_trip(self, signer, sample_receipt):
        signed = signer.sign(sample_receipt)
        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)

        assert restored.receipt_data == sample_receipt
        assert restored.signature == signed.signature
        assert signer.verify(restored) is True
