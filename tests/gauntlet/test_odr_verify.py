"""Tests for the in-package ODR verification engine (aragora.gauntlet.odr_verify).

Mirrors the standalone ``aragora-verify`` package's behavior; both must agree on
the same content profile and signature construction so a receipt verifies
identically here and for an external auditor.
"""

from __future__ import annotations

import base64
import copy
from typing import Any

import pytest

from aragora.gauntlet.odr_export import odr_content_digest
from aragora.gauntlet.odr_verify import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    ODRVerificationError,
    compute_key_id,
    load_public_key,
    verify_odr_document,
)

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402


def _valid_odr() -> dict[str, Any]:
    return {
        "odr_version": "0.1",
        "profile": "https://aragora.ai/specs/open-decision-receipt/v0.1",
        "receipt_id": "rcpt-0001",
        "issued_at": "2026-06-14T00:00:00Z",
        "subject": {
            "identifier": "5f1b14e4b5e113dc978d60d1f6bd21b5a478c744",
            "digest": {"status": "present", "alg": "sha-256", "value": "deadbeef"},
            "summary": "PR #8360",
        },
        "claim": {"verdict": "PASS", "statement": "merge PR #8360"},
        "reasoning": {"status": "present", "summary": "all checks green; quorum reached"},
        "quorum": {
            "status": "present",
            "method": "majority",
            "reached": True,
            "supporting_agents": ["claude", "grok"],
            "participants": [
                {"agent": "claude", "model_family": "anthropic", "model_id": "claude-opus-4-8"},
                {"agent": "grok", "model_family": "xai", "model_id": "grok-4"},
            ],
            "independence": {
                "disclosed": True,
                "distinct_model_families": 2,
                "model_families": ["anthropic", "xai"],
            },
            "dissent": {"present": False, "dissenting_agents": [], "views": []},
        },
        "confidence": {
            "status": "present",
            "value": 0.9,
            "scale": "unit_interval",
            "calibration": {"status": "absent", "reason": "no calibration record"},
        },
        "cruxes": {"status": "absent", "reason": "no crux set supplied"},
        "attestation": {"disposition": "autonomous"},
        "routing": {"status": "reserved"},
        "signatures": [],
    }


def _sign(doc: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    signed = copy.deepcopy(doc)
    message = bytes.fromhex(odr_content_digest(signed))
    signature = private_key.sign(message)
    signed["signatures"] = [
        {
            "alg": "Ed25519",
            "key_id": compute_key_id(private_key.public_key()),
            "signature": base64.b64encode(signature).decode("ascii"),
            "signed_at": "2026-06-14T00:00:01Z",
        }
    ]
    return signed


def _pem(public_key: Any) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def _check(result: Any, name: str) -> Any:
    return next(c for c in result.checks if c.name == name)


def test_valid_unsigned_receipt_passes_structurally() -> None:
    result = verify_odr_document(_valid_odr())
    assert result.ok is True
    assert _check(result, "schema_conformance").status == PASS
    assert _check(result, "signature").status == WARN


def test_digest_matches_emitter() -> None:
    doc = _valid_odr()
    assert verify_odr_document(doc).odr_digest == odr_content_digest(doc)


def test_missing_member_fails_schema() -> None:
    doc = _valid_odr()
    del doc["claim"]
    result = verify_odr_document(doc)
    assert result.ok is False
    assert _check(result, "schema_conformance").status == FAIL


def test_routing_must_be_reserved() -> None:
    doc = _valid_odr()
    doc["routing"] = {"status": "active"}
    assert verify_odr_document(doc).ok is False


def test_signed_verifies_with_correct_key() -> None:
    priv = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    result = verify_odr_document(signed, public_key=load_public_key(_pem(priv.public_key())))
    assert result.ok is True
    assert _check(result, "signature").status == PASS


def test_mutated_byte_fails_signature() -> None:
    priv = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    signed["claim"]["verdict"] = "FAIL"
    result = verify_odr_document(signed, public_key=load_public_key(_pem(priv.public_key())))
    assert result.ok is False
    assert _check(result, "signature").status == FAIL


def test_wrong_key_fails() -> None:
    priv = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    result = verify_odr_document(signed, public_key=load_public_key(_pem(other.public_key())))
    assert result.ok is False


def test_signed_without_key_is_skipped() -> None:
    priv = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    result = verify_odr_document(signed)
    assert result.ok is True
    assert _check(result, "signature").status == SKIP


def test_raw_key_loads() -> None:
    priv = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    assert verify_odr_document(signed, public_key=load_public_key(raw)).ok is True


def test_quorum_inconsistency_fails() -> None:
    doc = _valid_odr()
    doc["quorum"]["supporting_agents"].append("ghost")
    result = verify_odr_document(doc)
    assert result.ok is False
    assert _check(result, "quorum_consistency").status == FAIL
    assert "ghost" in _check(result, "quorum_consistency").detail


@pytest.mark.parametrize("field", ["participants", "supporting_agents"])
def test_quorum_present_but_null_list_subfield_fails_not_crash(field: str) -> None:
    # A present-but-null list subfield (e.g. ``participants: null``) is a
    # malformed/tamper signal: ``dict.get(key, [])`` returns None on a present
    # null, so the engine must turn it into a FAIL verdict, not raise TypeError
    # downstream. Regression for the #8389 review finding.
    doc = _valid_odr()
    doc["quorum"][field] = None
    result = verify_odr_document(doc)  # must not raise
    assert result.ok is False


def test_quorum_null_dissenting_agents_fails_not_crash() -> None:
    doc = _valid_odr()
    doc["quorum"]["dissent"] = {"status": "present", "dissenting_agents": None}
    result = verify_odr_document(doc)  # must not raise
    assert result.ok is False


def test_chain_anchored_passes_and_broken_fails() -> None:
    doc = _valid_odr()
    digest = odr_content_digest(doc)
    good = [{"hash": "h0"}, {"hash": "h1", "prev_hash": "h0", "odr_digest": digest}]
    assert _check(verify_odr_document(doc, chain=good), "chain_link").status == PASS
    bad = [{"hash": "h0"}, {"hash": "h1", "prev_hash": "WRONG", "odr_digest": digest}]
    assert verify_odr_document(doc, chain=bad).ok is False


def test_chain_unanchored_fails() -> None:
    chain = [{"hash": "h0"}, {"hash": "h1", "prev_hash": "h0"}]
    assert verify_odr_document(_valid_odr(), chain=chain).ok is False


def test_weakening_signals_do_not_fail() -> None:
    result = verify_odr_document(_valid_odr())
    joined = " ".join(result.warnings)
    assert "autonomous" in joined
    assert "uncalibrated" in joined
    assert result.ok is True


def test_to_dict_json_serializable() -> None:
    import json

    priv = Ed25519PrivateKey.generate()
    signed = _sign(_valid_odr(), priv)
    result = verify_odr_document(signed, public_key=load_public_key(_pem(priv.public_key())))
    json.dumps(result.to_dict())


def test_load_public_key_rejects_garbage() -> None:
    with pytest.raises(ODRVerificationError):
        load_public_key(b"not a key")
