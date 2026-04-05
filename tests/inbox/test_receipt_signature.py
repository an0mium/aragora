"""Tests for DurableFileSigner signature verification lifecycle."""

from __future__ import annotations

import copy
import os
import tempfile

import pytest

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner, SignedReceipt


@pytest.fixture()
def durable_signer(tmp_path):
    """Create a DurableFileSigner backed by a temporary key file."""
    key_path = str(tmp_path / "test_signing.key")
    return DurableFileSigner(key_path=key_path)


@pytest.fixture()
def receipt_data():
    """Sample receipt data mimicking an inbox trust wedge receipt."""
    return {
        "receipt_id": "rcpt-001",
        "intent_hash": "abc123def456",
        "action": "archive",
        "provider": "gmail",
        "user_id": "user-1",
        "message_id": "msg-42",
        "confidence": 0.91,
        "debate_id": "debate-99",
    }


class TestDurableFileSignerLifecycle:
    """Full sign -> verify -> tamper -> detect lifecycle."""

    def test_sign_and_verify_round_trip(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        assert isinstance(signed, SignedReceipt)
        assert signed.receipt_data == receipt_data
        assert signed.signature  # non-empty
        assert signer.verify(signed) is True

    def test_tampered_receipt_data_detected(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        # Tamper with the receipt data
        signed.receipt_data["confidence"] = 0.01
        assert signer.verify(signed) is False

    def test_tampered_action_detected(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        signed.receipt_data["action"] = "delete"
        assert signer.verify(signed) is False

    def test_tampered_signature_detected(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        # Replace signature with garbage
        signed.signature = "dGFtcGVyZWQ="  # base64("tampered")
        assert signer.verify(signed) is False

    def test_added_field_detected(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        signed.receipt_data["injected"] = "malicious"
        assert signer.verify(signed) is False

    def test_removed_field_detected(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        del signed.receipt_data["debate_id"]
        assert signer.verify(signed) is False

    def test_key_persistence_across_instances(self, tmp_path, receipt_data):
        key_path = str(tmp_path / "persist.key")
        signer1 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        signed = signer1.sign(receipt_data)

        # New instance loading the same key file must verify
        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        assert signer2.verify(signed) is True

    def test_different_key_rejects(self, tmp_path, receipt_data):
        signer_a = ReceiptSigner(backend=DurableFileSigner(key_path=str(tmp_path / "key_a.key")))
        signer_b = ReceiptSigner(backend=DurableFileSigner(key_path=str(tmp_path / "key_b.key")))
        signed = signer_a.sign(receipt_data)
        assert signer_b.verify(signed) is False

    def test_serialization_round_trip(self, durable_signer, receipt_data):
        signer = ReceiptSigner(backend=durable_signer)
        signed = signer.sign(receipt_data)

        rebuilt = SignedReceipt.from_dict(signed.to_dict())
        assert signer.verify(rebuilt) is True
