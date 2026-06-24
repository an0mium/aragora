"""Post-quantum (ML-DSA / FIPS 204) hybrid ODR signing round-trips (#8602).

Gated on ``pqc_available()`` so the suite skips cleanly where the installed
``cryptography`` build lacks the ``mldsa`` module, and runs the real ML-DSA
round-trips where it is present (cryptography>=49 / OpenSSL 3.5+).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aragora.gauntlet import odr_signing as s

pytestmark = pytest.mark.skipif(
    not s.pqc_available(),
    reason="cryptography build lacks ML-DSA (needs cryptography>=49 / OpenSSL 3.5+)",
)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "aragora" / "gauntlet" / "odr_schema.json"


def _odr() -> dict:
    """A schema-conformant unsigned ODR v0.1 doc (mirrors the aragora-verify fixture),
    so the hybrid tests exercise the *real* ODR shape + schema, not a stub."""
    return {
        "odr_version": "0.1",
        "profile": "https://aragora.ai/specs/open-decision-receipt/v0.1",
        "receipt_id": "rcpt-pqc-0001",
        "issued_at": "2026-06-24T00:00:00Z",
        "subject": {
            "identifier": "5f1b14e4b5e113dc978d60d1f6bd21b5a478c744",
            "digest": {"status": "present", "alg": "sha-256", "value": "deadbeef"},
            "summary": "PR #8608",
        },
        "claim": {"verdict": "PASS", "statement": "merge PR #8608"},
        "reasoning": {"status": "present", "summary": "all required checks green"},
        "quorum": {
            "status": "present",
            "method": "majority",
            "reached": True,
            "supporting_agents": ["claude", "openai"],
            "participants": [
                {"agent": "claude", "model_family": "anthropic", "model_id": "claude-opus-4-8"},
                {"agent": "openai", "model_family": "openai", "model_id": "gpt-5.5"},
            ],
            "independence": {
                "disclosed": True,
                "distinct_model_families": 2,
                "model_families": ["anthropic", "openai"],
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


def _ed25519_key():
    return s.generate_signing_key()


# --- ML-DSA only -------------------------------------------------------------
def test_mldsa_sign_verify_roundtrip():
    key = s.generate_mldsa_signing_key()
    signed = s.sign_odr_receipt_mldsa(_odr(), key)
    entry = signed["signatures"][-1]
    assert entry["alg"] == s.ODR_PQC_SIGNATURE_ALG == "ML-DSA-65"
    assert entry["key_id"] == s.compute_mldsa_key_id(key.public_key())
    assert s.verify_odr_signature_mldsa(signed, public_key=key.public_key()) is True


def test_mldsa_tamper_rejected():
    key = s.generate_mldsa_signing_key()
    signed = s.sign_odr_receipt_mldsa(_odr(), key)
    tampered = copy.deepcopy(signed)
    tampered["decision"] = "ship something else"  # changes the content digest
    assert s.verify_odr_signature_mldsa(tampered, public_key=key.public_key()) is False


def test_mldsa_wrong_key_rejected():
    signed = s.sign_odr_receipt_mldsa(_odr(), s.generate_mldsa_signing_key())
    assert (
        s.verify_odr_signature_mldsa(signed, public_key=s.generate_mldsa_signing_key().public_key())
        is False
    )


def test_mldsa_seed_roundtrip_is_deterministic():
    key = s.generate_mldsa_signing_key()
    seed = key.private_bytes_raw()
    assert len(seed) == 32
    reloaded = s.load_mldsa_key_from_seed(seed)
    # same seed -> same public key -> same key_id
    assert s.compute_mldsa_key_id(reloaded.public_key()) == s.compute_mldsa_key_id(key.public_key())
    signed = s.sign_odr_receipt_mldsa(_odr(), reloaded)
    assert s.verify_odr_signature_mldsa(signed, public_key=key.public_key()) is True


def test_load_mldsa_key_from_bad_seed_raises():
    with pytest.raises(s.OdrSigningError):
        s.load_mldsa_key_from_seed(b"too short")


# --- Hybrid (Ed25519 + ML-DSA) ----------------------------------------------
def test_hybrid_attaches_both_and_both_verify():
    ed = _ed25519_key()
    pq = s.generate_mldsa_signing_key()
    signed = s.sign_odr_hybrid(_odr(), ed25519_key=ed, mldsa_key=pq)
    algs = [e["alg"] for e in signed["signatures"]]
    assert algs == ["Ed25519", "ML-DSA-65"]
    # both halves verify independently over the same content digest
    assert s.verify_odr_signature_ed25519(signed, public_key=ed.public_key()) is True
    assert s.verify_odr_signature_mldsa(signed, public_key=pq.public_key()) is True


def test_hybrid_tamper_breaks_both():
    ed = _ed25519_key()
    pq = s.generate_mldsa_signing_key()
    signed = s.sign_odr_hybrid(_odr(), ed25519_key=ed, mldsa_key=pq)
    signed["claim"]["statement"] = "merge something else"  # mutate content digest
    assert s.verify_odr_signature_ed25519(signed, public_key=ed.public_key()) is False
    assert s.verify_odr_signature_mldsa(signed, public_key=pq.public_key()) is False


def test_adding_mldsa_does_not_invalidate_existing_ed25519():
    # Sign Ed25519 first (as existing producers do), then add ML-DSA later.
    ed = _ed25519_key()
    ed_signed = s.sign_odr_receipt(_odr(), ed)
    both = s.sign_odr_receipt_mldsa(ed_signed, s.generate_mldsa_signing_key())
    # the digest excludes signatures, so the prior Ed25519 signature still verifies
    assert s.verify_odr_signature_ed25519(both, public_key=ed.public_key()) is True
    assert len(both["signatures"]) == 2


def test_ed25519_only_signing_unchanged_by_pqc_additions():
    # The classical path must remain byte-identical in shape.
    ed = _ed25519_key()
    signed = s.sign_odr_receipt(_odr(), ed)
    assert [e["alg"] for e in signed["signatures"]] == ["Ed25519"]
    assert s.verify_odr_signature_ed25519(signed, public_key=ed.public_key()) is True


# --- Schema conformance (resolves #8608 P1: hybrid receipts were hard-rejected) ---
def _validate_against_schema(doc: dict) -> None:
    import jsonschema

    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(doc, schema)  # raises jsonschema.ValidationError on reject


def test_hybrid_receipt_validates_against_odr_schema():
    # The core backward-compat proof: a receipt carrying BOTH an Ed25519 and an
    # ML-DSA-65 signature entry must satisfy the canonical ODR schema (the enum
    # now allows ML-DSA-65). Before the fix this raised ValidationError.
    signed = s.sign_odr_hybrid(
        _odr(), ed25519_key=_ed25519_key(), mldsa_key=s.generate_mldsa_signing_key()
    )
    assert [e["alg"] for e in signed["signatures"]] == ["Ed25519", "ML-DSA-65"]
    _validate_against_schema(signed)  # must not raise


def test_ed25519_only_receipt_still_schema_valid():
    signed = s.sign_odr_receipt(_odr(), _ed25519_key())
    _validate_against_schema(signed)


def test_unsigned_odr_is_schema_valid():
    _validate_against_schema(_odr())
