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

    def test_full_lifecycle_end_to_end(self, tmp_path: object) -> None:
        """Complete lifecycle: create receipt, sign, verify, tamper, detect."""
        key_path = os.path.join(str(tmp_path), "lifecycle.key")
        backend = DurableFileSigner(key_path=key_path)
        signer = ReceiptSigner(backend=backend)

        # Step 1: Create receipt data
        receipt = {
            "receipt_id": "rcpt-lifecycle-001",
            "decision": "approve",
            "topic": "Security audit findings",
            "consensus_score": 0.92,
            "agents": ["claude", "gpt-4", "gemini"],
            "timestamp": "2026-04-05T12:00:00Z",
        }

        # Step 2: Sign with DurableFileSigner
        signatory = SignatoryInfo(name="TestBot", email="bot@test.com", role="Auditor")
        signed = signer.sign(receipt, signatory=signatory)
        assert isinstance(signed, SignedReceipt)
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signed.signature_metadata.signatory.name == "TestBot"

        # Step 3: Verify valid signature
        assert signer.verify(signed)

        # Step 4: Roundtrip through JSON serialization and re-verify
        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)
        assert signer.verify(restored)

        # Step 5: Verify with a fresh signer instance (same key file)
        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        assert signer2.verify(restored)

        # Step 6: Tamper with data and confirm detection
        tampered = copy.deepcopy(restored)
        tampered.receipt_data["decision"] = "reject"
        assert not signer.verify(tampered)

        # Step 7: Ensure original is still valid after tampering a copy
        assert signer.verify(restored)

    def test_verify_dict_roundtrip(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        """verify_dict accepts the dict form of a signed receipt."""
        signed = signer.sign(sample_receipt)
        assert signer.verify_dict(signed.to_dict())

        # Tamper via dict path
        d = signed.to_dict()
        d["receipt"]["decision"] = "reject"
        assert not signer.verify_dict(d)

    def test_metadata_timestamp_present(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)
        ts = signed.signature_metadata.timestamp
        assert ts  # non-empty
        assert "T" in ts  # ISO format contains T separator
