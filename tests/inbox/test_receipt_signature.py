"""Tests for inbox receipt signing lifecycle and tamper detection."""

from __future__ import annotations

import copy
import os
from datetime import datetime

import pytest

from aragora.gauntlet.signing import (
    DurableFileSigner,
    ReceiptSigner,
    SignedReceipt,
    SignatoryInfo,
)


@pytest.fixture()
def signer(tmp_path):
    key_path = str(tmp_path / "receipt-signing.key")
    return ReceiptSigner(backend=DurableFileSigner(key_path=key_path))


@pytest.fixture()
def receipt_data():
    return {
        "receipt_id": "rcpt-001",
        "decision": "approve",
        "topic": "Rate limiter design",
        "consensus_score": 0.87,
        "agents": ["claude", "gpt-4", "gemini"],
    }


def test_receipt_signature_lifecycle_detects_tampering(tmp_path) -> None:
    key_path = tmp_path / "receipt-signing.key"
    signer = ReceiptSigner(backend=DurableFileSigner(key_path=str(key_path)))
    receipt = {
        "receipt_id": "rcpt-001",
        "decision": "approve",
        "topic": "Rate limiter design",
        "consensus_score": 0.87,
        "agents": ["claude", "gpt-4", "gemini"],
    }

    signed = signer.sign(receipt)
    assert key_path.exists()
    assert signed.signature
    assert signed.signature_metadata.algorithm == "HMAC-SHA256"
    assert signed.signature_metadata.key_id.startswith("durable-")

    verifier = ReceiptSigner(backend=DurableFileSigner(key_path=str(key_path)))
    round_tripped = SignedReceipt.from_json(signed.to_json())
    assert verifier.verify(round_tripped) is True

    tampered_payload = copy.deepcopy(round_tripped.to_dict())
    tampered_payload["receipt"]["decision"] = "reject"
    tampered = SignedReceipt.from_dict(tampered_payload)

    assert tampered.signature == round_tripped.signature
    assert verifier.verify(tampered) is False
    assert verifier.verify(round_tripped) is True


def test_tampered_json_roundtrip_detected(signer, receipt_data) -> None:
    signed = signer.sign(receipt_data)
    json_str = signed.to_json().replace('"approve"', '"reject"')
    restored = SignedReceipt.from_json(json_str)

    assert not signer.verify(restored)


def test_verify_dict_roundtrip_and_tamper_detection(signer, receipt_data) -> None:
    signed = signer.sign(receipt_data)
    raw = signed.to_dict()

    assert signer.verify_dict(raw)

    raw["receipt"]["decision"] = "reject"
    assert not signer.verify_dict(raw)


def test_full_lifecycle_create_sign_verify_tamper_detect(tmp_path) -> None:
    key_path = str(tmp_path / "lifecycle.key")
    signer = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
    receipt = {
        "receipt_id": "rcpt-lifecycle",
        "decision": "approve",
        "topic": "E2E lifecycle validation",
        "consensus_score": 0.92,
        "agents": ["claude", "gpt-4"],
    }

    signed = signer.sign(
        receipt,
        signatory=SignatoryInfo(name="Bot", email="bot@test.com", role="Auditor"),
    )

    assert signer.verify(signed)
    assert signer.verify_dict(signed.to_dict())

    restored = SignedReceipt.from_json(signed.to_json())
    assert signer.verify(restored)

    signer_reloaded = ReceiptSigner(backend=DurableFileSigner(key_path=key_path))
    assert signer_reloaded.verify(restored)

    tampered = SignedReceipt.from_json(signed.to_json())
    tampered.receipt_data["decision"] = "reject"
    assert not signer_reloaded.verify(tampered)


def test_nested_data_tamper_detected(signer) -> None:
    receipt = {"id": "rcpt-nested", "meta": {"priority": "high", "tags": ["a", "b"]}}
    signed = signer.sign(receipt)

    assert signer.verify(signed)

    signed.receipt_data["meta"]["priority"] = "low"
    assert not signer.verify(signed)


def test_empty_receipt_signs_and_detects_mutation(signer) -> None:
    signed = signer.sign({})
    assert signer.verify(signed)

    signed.receipt_data["injected"] = True
    assert not signer.verify(signed)


def test_signature_metadata_timestamp_is_isoformat(signer, receipt_data) -> None:
    signed = signer.sign(receipt_data)
    timestamp = signed.signature_metadata.timestamp

    assert timestamp
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_durable_key_id_is_stable_for_same_key_file(tmp_path) -> None:
    key_path = str(tmp_path / "stable.key")
    first = DurableFileSigner(key_path=key_path)
    second = DurableFileSigner(key_path=key_path)

    assert first.key_id == second.key_id


def test_key_file_permissions_are_private(tmp_path) -> None:
    key_path = str(tmp_path / "secure.key")
    DurableFileSigner(key_path=key_path)
    mode = os.stat(key_path).st_mode & 0o777

    assert mode == 0o600


def test_signatory_roundtrip_survives_json(signer, receipt_data) -> None:
    signatory = SignatoryInfo(
        name="Bob",
        email="bob@example.com",
        role="Auditor",
        title="Senior Engineer",
        organization="Acme",
        department="Security",
    )
    signed = signer.sign(receipt_data, signatory=signatory)
    restored = SignedReceipt.from_json(signed.to_json())

    assert restored.signature_metadata.signatory is not None
    assert restored.signature_metadata.signatory.name == "Bob"
    assert restored.signature_metadata.signatory.department == "Security"
    assert signer.verify(restored)


def test_signing_does_not_mutate_input(signer) -> None:
    receipt = {"id": "r1", "items": [1, 2, 3]}
    original = copy.deepcopy(receipt)

    signer.sign(receipt)

    assert receipt == original
