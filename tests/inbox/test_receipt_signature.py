"""Tests for the full receipt signature lifecycle.

Covers creation, signing with DurableFileSigner, verification of valid
signatures, and detection of tampered receipt data.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    HMACSigner,
    ReceiptSigner,
    SignedReceipt,
)


@pytest.fixture()
def tmp_key_path(tmp_path):
    """Return a path for a temporary signing key file."""
    return str(tmp_path / "test_signing.key")


@pytest.fixture()
def signer(tmp_key_path):
    """ReceiptSigner backed by a DurableFileSigner in a temp directory."""
    backend = DurableFileSigner(key_path=tmp_key_path)
    return ReceiptSigner(backend=backend)


def _sample_receipt() -> dict:
    return {
        "receipt_id": "r-sig-001",
        "debate_id": "debate-sig-001",
        "task": "Should we deploy v2?",
        "verdict": "Yes, with staged rollout",
        "confidence": 0.92,
        "participants": ["claude", "gpt"],
    }


class TestReceiptSignatureLifecycle:
    """Full lifecycle: create -> sign -> verify -> tamper -> detect."""

    def test_sign_and_verify_valid(self, signer):
        """A freshly signed receipt verifies successfully."""
        receipt_data = _sample_receipt()
        signed = signer.sign(receipt_data)

        assert isinstance(signed, SignedReceipt)
        assert signed.signature  # non-empty
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signer.verify(signed) is True

    def test_tampered_verdict_detected(self, signer):
        """Modifying the verdict after signing invalidates the signature."""
        signed = signer.sign(_sample_receipt())
        assert signer.verify(signed) is True

        signed.receipt_data["verdict"] = "No, abort deployment"
        assert signer.verify(signed) is False

    def test_tampered_confidence_detected(self, signer):
        """Modifying the confidence score after signing is detected."""
        signed = signer.sign(_sample_receipt())
        signed.receipt_data["confidence"] = 0.01
        assert signer.verify(signed) is False

    def test_tampered_participants_detected(self, signer):
        """Adding or removing participants after signing is detected."""
        signed = signer.sign(_sample_receipt())
        signed.receipt_data["participants"].append("adversary")
        assert signer.verify(signed) is False

    def test_extra_field_injection_detected(self, signer):
        """Injecting a new field after signing is detected."""
        signed = signer.sign(_sample_receipt())
        signed.receipt_data["admin_override"] = True
        assert signer.verify(signed) is False

    def test_durable_key_persists_across_signers(self, tmp_key_path):
        """A receipt signed by one DurableFileSigner instance can be
        verified by a second instance loaded from the same key file."""
        signer1 = ReceiptSigner(backend=DurableFileSigner(key_path=tmp_key_path))
        signed = signer1.sign(_sample_receipt())

        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=tmp_key_path))
        assert signer2.verify(signed) is True

    def test_different_key_rejects_signature(self, tmp_path):
        """A receipt signed with one key fails verification with another."""
        path_a = str(tmp_path / "key_a.key")
        path_b = str(tmp_path / "key_b.key")

        signer_a = ReceiptSigner(backend=DurableFileSigner(key_path=path_a))
        signed = signer_a.sign(_sample_receipt())

        signer_b = ReceiptSigner(backend=DurableFileSigner(key_path=path_b))
        assert signer_b.verify(signed) is False

    def test_roundtrip_via_dict(self, signer):
        """SignedReceipt survives dict serialization round-trip."""
        signed = signer.sign(_sample_receipt())
        restored = SignedReceipt.from_dict(signed.to_dict())
        assert signer.verify(restored) is True

    def test_roundtrip_via_json(self, signer):
        """SignedReceipt survives JSON serialization round-trip."""
        signed = signer.sign(_sample_receipt())
        restored = SignedReceipt.from_json(signed.to_json())
        assert signer.verify(restored) is True
