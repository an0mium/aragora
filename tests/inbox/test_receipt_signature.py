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

    def test_full_lifecycle(self, tmp_path: object) -> None:
        """End-to-end: create receipt, sign, verify, serialize, tamper, detect."""
        key_path = os.path.join(str(tmp_path), "lifecycle.key")
        backend = DurableFileSigner(key_path=key_path)
        signer = ReceiptSigner(backend=backend)

        receipt = {
            "receipt_id": "rcpt-lifecycle",
            "decision": "approve",
            "score": 0.95,
            "participants": ["a1", "a2"],
        }

        # Step 1: sign
        signatory = SignatoryInfo(name="Bot", email="bot@test.com", role="System")
        signed = signer.sign(receipt, signatory=signatory)

        # Step 2: verify original
        assert signer.verify(signed)
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signed.signature_metadata.signatory.name == "Bot"

        # Step 3: roundtrip via JSON
        restored = SignedReceipt.from_json(signed.to_json())
        assert signer.verify(restored)

        # Step 4: verify with fresh instance (same key file)
        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        assert signer2.verify(restored)

        # Step 5: tamper and detect
        restored.receipt_data["decision"] = "reject"
        assert not signer2.verify(restored)

    def test_verify_dict_roundtrip(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)
        assert signer.verify_dict(signed.to_dict())

        # Tampered dict should fail
        d = signed.to_dict()
        d["receipt"]["decision"] = "tampered"
        assert not signer.verify_dict(d)

    def test_deep_copy_independence(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        """Signing a deep copy produces a valid, independent receipt."""
        original_signed = signer.sign(sample_receipt)
        cloned = copy.deepcopy(original_signed)

        assert signer.verify(cloned)

        # Mutating clone doesn't affect original verification
        cloned.receipt_data["decision"] = "tampered"
        assert not signer.verify(cloned)
        assert signer.verify(original_signed)

    def test_empty_receipt(self, signer: ReceiptSigner) -> None:
        signed = signer.sign({})
        assert signer.verify(signed)
        signed.receipt_data["new_key"] = "value"
        assert not signer.verify(signed)

    def test_timestamp_present(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        signed = signer.sign(sample_receipt)
        ts = signed.signature_metadata.timestamp
        assert ts  # non-empty
        # Should be a valid ISO timestamp
        from datetime import datetime

        datetime.fromisoformat(ts.replace("Z", "+00:00"))
