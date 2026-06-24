"""Post-quantum (ML-DSA / FIPS 204) hybrid ODR signing round-trips (#8602).

Gated on ``pqc_available()`` so the suite skips cleanly where the installed
``cryptography`` build lacks the ``mldsa`` module, and runs the real ML-DSA
round-trips where it is present (cryptography>=49 / OpenSSL 3.5+).
"""

from __future__ import annotations

import copy

import pytest

from aragora.gauntlet import odr_signing as s

pytestmark = pytest.mark.skipif(
    not s.pqc_available(),
    reason="cryptography build lacks ML-DSA (needs cryptography>=49 / OpenSSL 3.5+)",
)


def _odr() -> dict:
    return {
        "schema": "open-decision-receipt/v1",
        "receipt_id": "odr-test-0001",
        "decision": "ship the rate limiter",
        "evidence": ["a", "b"],
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
    signed["evidence"].append("forged")  # mutate content
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
