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

    def test_verify_dict_roundtrip(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        """verify_dict should accept the dict form of a signed receipt."""
        signed = signer.sign(sample_receipt)
        assert signer.verify_dict(signed.to_dict())

    def test_verify_dict_tamper_detected(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        d = signer.sign(sample_receipt).to_dict()
        d["receipt"]["decision"] = "reject"
        assert not signer.verify_dict(d)

    def test_nested_receipt_data(self, signer: ReceiptSigner) -> None:
        """Deeply nested data should still be signed deterministically."""
        nested = {
            "id": "rcpt-nested",
            "meta": {"tags": ["a", "b"], "scores": {"quality": 0.9}},
        }
        signed = signer.sign(nested)
        assert signer.verify(signed)
        signed.receipt_data["meta"]["scores"]["quality"] = 0.1
        assert not signer.verify(signed)

    def test_empty_receipt(self, signer: ReceiptSigner) -> None:
        signed = signer.sign({})
        assert signer.verify(signed)

    def test_signature_metadata_timestamp(
        self, signer: ReceiptSigner, sample_receipt: dict
    ) -> None:
        signed = signer.sign(sample_receipt)
        # Metadata should carry a valid ISO timestamp
        ts = signed.signature_metadata.timestamp
        from datetime import datetime

        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_sign_shares_reference(self, signer: ReceiptSigner, sample_receipt: dict) -> None:
        """sign() stores a reference; callers must not mutate after signing."""
        data = copy.deepcopy(sample_receipt)
        signed = signer.sign(data)
        assert signer.verify(signed)
        # Mutating the dict invalidates the signature (shared reference)
        data["decision"] = "reject"
        assert not signer.verify(signed)

    def test_full_lifecycle_end_to_end(self, tmp_path: object) -> None:
        """Complete lifecycle: create key, sign, serialise, reload key, verify, tamper, detect."""
        key_path = os.path.join(str(tmp_path), "lifecycle.key")

        # Phase 1: sign and serialise
        signer1 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        receipt = {"id": "e2e", "action": "deploy", "risk": 0.3}
        signed = signer1.sign(
            receipt, signatory=SignatoryInfo(name="Bot", email="bot@a.io", role="CI")
        )
        json_blob = signed.to_json()

        # Phase 2: new process loads same key, verifies
        signer2 = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
        restored = SignedReceipt.from_json(json_blob)
        assert signer2.verify(restored)

        # Phase 3: tamper with serialised form
        restored.receipt_data["risk"] = 0.0
        assert not signer2.verify(restored)
