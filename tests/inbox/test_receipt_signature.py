"""Tests for receipt signature lifecycle: create, sign, verify, tamper-detect.

Covers the full DurableFileSigner + ReceiptSigner flow including
tampering detection to ensure data integrity violations are caught.
"""

from __future__ import annotations

import copy
import tempfile
import os

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    ReceiptSigner,
    SignedReceipt,
    SignatoryInfo,
)


@pytest.fixture()
def signer(tmp_path: object) -> ReceiptSigner:
    """ReceiptSigner backed by a DurableFileSigner in a temp directory."""
    key_path = os.path.join(str(tmp_path), "test_signing.key")
    backend = DurableFileSigner(key_path=key_path)
    return ReceiptSigner(backend=backend)


@pytest.fixture()
def sample_receipt() -> dict:
    return {
        "receipt_id": "rcpt-001",
        "decision": "approve",
        "topic": "Rate limiter design",
        "consensus_score": 0.87,
        "agents": ["claude", "gpt-4", "gemini"],
    }


class TestReceiptSignatureLifecycle:
    """Full lifecycle: create -> sign -> verify -> tamper -> detect."""

    def test_sign_and_verify(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)

        assert isinstance(signed, SignedReceipt)
        assert signed.receipt_data == sample_receipt
        assert signed.signature  # non-empty
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signer.verify(signed)

    def test_tamper_receipt_data_detected(
        self, signer: ReceiptSigner, sample_receipt: dict
    ) -> None:
        signed = signer.sign(sample_receipt)
        assert signer.verify(signed)

        # Tamper with the receipt data
        signed.receipt_data["decision"] = "reject"
        assert not signer.verify(signed)

    def test_tamper_consensus_score_detected(
        self, signer: ReceiptSigner, sample_receipt: dict
    ) -> None:
        signed = signer.sign(sample_receipt)

        signed.receipt_data["consensus_score"] = 0.01
        assert not signer.verify(signed)

    def test_tamper_add_field_detected(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)

        signed.receipt_data["injected"] = True
        assert not signer.verify(signed)

    def test_tamper_remove_field_detected(
        self, signer: ReceiptSigner, sample_receipt: dict
    ) -> None:
        signed = signer.sign(sample_receipt)

        del signed.receipt_data["agents"]
        assert not signer.verify(signed)

    def test_tamper_signature_detected(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)

        # Corrupt the base64 signature
        sig_bytes = list(signed.signature)
        sig_bytes[0] = "A" if sig_bytes[0] != "A" else "B"
        signed.signature = "".join(sig_bytes)
        assert not signer.verify(signed)

    def test_roundtrip_json(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)
        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)

        assert signer.verify(restored)
        assert restored.receipt_data == sample_receipt

    def test_signatory_info_preserved(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signatory = SignatoryInfo(name="Alice", email="alice@example.com", role="Approver")
        signed = signer.sign(sample_receipt, signatory=signatory)

        assert signed.signature_metadata.signatory is not None
        assert signed.signature_metadata.signatory.name == "Alice"
        assert signer.verify(signed)

    def test_durable_key_persists(self, tmp_path: object, sample_receipt: dict) -> None:
        key_path = os.path.join(str(tmp_path), "persist.key")

        # Sign with first instance
        s1 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        signed = s1.sign(sample_receipt)

        # Verify with second instance (same key file)
        s2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        assert s2.verify(signed)

    def test_different_key_rejects(self, tmp_path: object, sample_receipt: dict) -> None:
        s1 = ReceiptSigner(
            backend=DurableFileSigner(key_path=os.path.join(str(tmp_path), "key_a.key"))
        )
        s2 = ReceiptSigner(
            backend=DurableFileSigner(key_path=os.path.join(str(tmp_path), "key_b.key"))
        )

        signed = s1.sign(sample_receipt)
        assert not s2.verify(signed)


class TestDurableFileSignerSecurity:
    """Security-focused tests for DurableFileSigner."""

    def test_key_file_permissions(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "secure.key")
        DurableFileSigner(key_path=key_path)
        mode = os.stat(key_path).st_mode & 0o777
        assert mode == 0o600, f"Key file should be 0600, got {oct(mode)}"

    def test_key_id_is_deterministic(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "det.key")
        s1 = DurableFileSigner(key_path=key_path)
        s2 = DurableFileSigner(key_path=key_path)
        assert s1.key_id == s2.key_id

    def test_verify_dict_convenience(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "conv.key")
        signer = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        receipt = {"id": "r1", "action": "deploy"}
        signed = signer.sign(receipt)
        assert signer.verify_dict(signed.to_dict())

    def test_empty_receipt_signs_and_verifies(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "empty.key")
        signer = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        signed = signer.sign({})
        assert signer.verify(signed)

    def test_signature_metadata_has_timestamp(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "ts.key")
        signer = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        signed = signer.sign({"x": 1})
        assert signed.signature_metadata.timestamp
        assert "T" in signed.signature_metadata.timestamp  # ISO format

    def test_signing_does_not_mutate_input(self, tmp_path: object) -> None:
        key_path = os.path.join(str(tmp_path), "mut.key")
        signer = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        receipt = {"id": "r1", "items": [1, 2, 3]}
        original = copy.deepcopy(receipt)
        signer.sign(receipt)
        assert receipt == original
