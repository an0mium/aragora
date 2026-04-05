"""Tests for DurableFileSigner signature verification and tampering detection."""

from __future__ import annotations

import copy
import pytest

from aragora.gauntlet.signing import DurableFileSigner, ReceiptSigner, SignedReceipt


@pytest.fixture()
def signer(tmp_path):
    """ReceiptSigner backed by a DurableFileSigner in a temp directory."""
    key_path = str(tmp_path / "signing.key")
    return ReceiptSigner(backend=DurableFileSigner(key_path=key_path))


@pytest.fixture()
def receipt_data():
    return {"decision": "approve", "score": 0.95, "agent": "claude"}


def test_sign_and_verify(signer, receipt_data):
    """A freshly signed receipt must verify successfully."""
    signed = signer.sign(receipt_data)
    assert isinstance(signed, SignedReceipt)
    assert signer.verify(signed)


def test_tamper_receipt_data(signer, receipt_data):
    """Modifying receipt_data after signing must fail verification."""
    signed = signer.sign(receipt_data)
    signed.receipt_data["decision"] = "reject"
    assert not signer.verify(signed)


def test_tamper_signature(signer, receipt_data):
    """Corrupting the signature itself must fail verification."""
    signed = signer.sign(receipt_data)
    signed.signature = "AAAA" + signed.signature[4:]
    assert not signer.verify(signed)


def test_roundtrip_json(signer, receipt_data):
    """Serialize to JSON and back; verification still passes."""
    signed = signer.sign(receipt_data)
    restored = SignedReceipt.from_json(signed.to_json())
    assert signer.verify(restored)


def test_different_key_rejects(tmp_path, receipt_data):
    """A receipt signed by one key must not verify with a different key."""
    s1 = ReceiptSigner(backend=DurableFileSigner(key_path=str(tmp_path / "k1.key")))
    s2 = ReceiptSigner(backend=DurableFileSigner(key_path=str(tmp_path / "k2.key")))
    signed = s1.sign(receipt_data)
    assert not s2.verify(signed)


def test_key_persistence(tmp_path, receipt_data):
    """Re-loading the same key file must verify old receipts."""
    kp = str(tmp_path / "persist.key")
    signed = ReceiptSigner(backend=DurableFileSigner(key_path=kp)).sign(receipt_data)
    assert ReceiptSigner(backend=DurableFileSigner(key_path=kp)).verify(signed)
