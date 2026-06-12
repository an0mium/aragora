"""Tests for Ed25519 detached signing of Open Decision Receipts (ODR-2, #8225).

Acceptance criteria from the issue:
- Receipt carries ``signatures[]`` with alg=Ed25519, key id, detached sig
  over the canonical bytes (SHA-256 of the JCS payload sans ``signatures``).
- Verification with ONLY the public key + receipt JSON passes; any tampered
  byte fails.
- Private key comes from the secrets layer, never the repo.
"""

from __future__ import annotations

import base64
import copy

import pytest

from aragora.gauntlet.odr_export import (
    decision_receipt_to_odr,
    load_odr_schema,
    odr_content_digest,
)
from aragora.gauntlet.odr_signing import (
    ODR_SIGNING_KEY_SECRET,
    derive_key_id,
    generate_odr_keypair,
    load_odr_signer,
    public_key_b64,
    sign_odr,
    signer_from_seed,
    verify_odr,
)
from aragora.gauntlet.receipt_models import DecisionReceipt


@pytest.fixture()
def receipt() -> DecisionReceipt:
    return DecisionReceipt(
        receipt_id="r-odr2-test",
        gauntlet_id="g-odr2-test",
        timestamp="2026-06-12T00:00:00+00:00",
        input_summary="Ship the Ed25519 signing change",
        input_hash="b" * 64,
        risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
        attacks_attempted=0,
        attacks_successful=0,
        probes_run=0,
        vulnerabilities_found=0,
        verdict="APPROVED",
        confidence=0.84,
        robustness_score=1.0,
        verdict_reasoning="Quorum approved with no dissent",
    )


@pytest.fixture()
def odr(receipt: DecisionReceipt) -> dict:
    return decision_receipt_to_odr(receipt)


@pytest.fixture()
def signer():
    return generate_odr_keypair()


class TestSignOdr:
    def test_appends_schema_conformant_signature_entry(self, odr, signer):
        signed = sign_odr(odr, signer=signer)
        assert len(signed["signatures"]) == 1
        entry = signed["signatures"][0]
        assert entry["alg"] == "Ed25519"
        assert entry["key_id"] == signer.key_id
        assert entry["signature"]
        base64.b64decode(entry["signature"])  # decodes cleanly
        assert "signed_at" in entry

        # Entry must validate against the bundled ODR JSON Schema.
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(signed, load_odr_schema())

    def test_does_not_mutate_input(self, odr, signer):
        before = copy.deepcopy(odr)
        sign_odr(odr, signer=signer)
        assert odr == before

    def test_signing_does_not_change_content_digest(self, odr, signer):
        digest_before = odr_content_digest(odr)
        signed = sign_odr(odr, signer=signer)
        assert odr_content_digest(signed) == digest_before

    def test_multiple_signatures_accumulate(self, odr, signer):
        other = generate_odr_keypair()
        signed = sign_odr(sign_odr(odr, signer=signer), signer=other)
        assert len(signed["signatures"]) == 2
        key_ids = {entry["key_id"] for entry in signed["signatures"]}
        assert key_ids == {signer.key_id, other.key_id}


class TestVerifyOdr:
    def test_verifies_with_only_public_key_and_json(self, odr, signer):
        signed = sign_odr(odr, signer=signer)
        # The verifier side gets nothing but the receipt JSON and the
        # base64 public key — no signer object, no shared secret.
        pub = public_key_b64(signer)
        result = verify_odr(signed, public_key=pub)
        assert result.valid
        assert result.verified_key_ids == [signer.key_id]

    def test_tampered_payload_fails(self, odr, signer):
        signed = sign_odr(odr, signer=signer)
        tampered = copy.deepcopy(signed)
        tampered["claim"]["verdict"] = "REJECTED"
        result = verify_odr(tampered, public_key=public_key_b64(signer))
        assert not result.valid

    def test_tampered_signature_fails(self, odr, signer):
        signed = sign_odr(odr, signer=signer)
        raw = bytearray(base64.b64decode(signed["signatures"][0]["signature"]))
        raw[0] ^= 0x01
        signed["signatures"][0]["signature"] = base64.b64encode(bytes(raw)).decode()
        result = verify_odr(signed, public_key=public_key_b64(signer))
        assert not result.valid

    def test_wrong_public_key_fails(self, odr, signer):
        signed = sign_odr(odr, signer=signer)
        other = generate_odr_keypair()
        result = verify_odr(signed, public_key=public_key_b64(other))
        assert not result.valid

    def test_key_order_independence(self, receipt, signer):
        """JCS canonicalization makes signatures stable across key order."""
        odr_a = decision_receipt_to_odr(receipt)
        # Same content, different insertion order.
        odr_b = dict(reversed(list(odr_a.items())))
        signed = sign_odr(odr_a, signer=signer)
        sig = signed["signatures"][0]
        odr_b_signed = dict(odr_b)
        odr_b_signed["signatures"] = [sig]
        result = verify_odr(odr_b_signed, public_key=public_key_b64(signer))
        assert result.valid

    def test_unsigned_receipt_reports_no_signatures(self, odr, signer):
        result = verify_odr(odr, public_key=public_key_b64(signer))
        assert not result.valid
        assert "no signatures" in result.reason.lower()


class TestKeyHandling:
    def test_seed_round_trip(self):
        seed = b"\x01" * 32
        a = signer_from_seed(seed)
        b = signer_from_seed(seed)
        assert a.key_id == b.key_id
        assert public_key_b64(a) == public_key_b64(b)

    def test_key_id_derived_from_public_key(self, signer):
        pub_raw = base64.b64decode(public_key_b64(signer))
        assert signer.key_id == derive_key_id(pub_raw)
        assert signer.key_id.startswith("ed25519-")

    def test_rejects_bad_seed_length(self):
        with pytest.raises(ValueError, match="32"):
            signer_from_seed(b"short")

    def test_load_signer_from_secrets(self, monkeypatch):
        seed = b"\x07" * 32
        encoded = base64.b64encode(seed).decode()
        monkeypatch.setattr(
            "aragora.gauntlet.odr_signing.get_secret",
            lambda name, default=None, strict=None: (
                encoded if name == ODR_SIGNING_KEY_SECRET else default
            ),
        )
        signer = load_odr_signer()
        assert signer is not None
        assert signer.key_id == signer_from_seed(seed).key_id

    def test_load_signer_accepts_hex_seed(self, monkeypatch):
        seed = b"\x09" * 32
        monkeypatch.setattr(
            "aragora.gauntlet.odr_signing.get_secret",
            lambda name, default=None, strict=None: (
                seed.hex() if name == ODR_SIGNING_KEY_SECRET else default
            ),
        )
        signer = load_odr_signer()
        assert signer is not None
        assert signer.key_id == signer_from_seed(seed).key_id

    def test_load_signer_returns_none_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(
            "aragora.gauntlet.odr_signing.get_secret",
            lambda name, default=None, strict=None: default,
        )
        assert load_odr_signer() is None

    def test_load_signer_rejects_garbage(self, monkeypatch):
        monkeypatch.setattr(
            "aragora.gauntlet.odr_signing.get_secret",
            lambda name, default=None, strict=None: "not-a-key!!",
        )
        with pytest.raises(ValueError):
            load_odr_signer()


class TestEndToEnd:
    def test_full_flow_matches_acceptance_criteria(self, receipt):
        """Sign with secrets-loaded key; verify with only pubkey + JSON."""
        signer = generate_odr_keypair()
        odr = decision_receipt_to_odr(receipt)
        signed = sign_odr(odr, signer=signer)

        # A third party holding only these two artifacts:
        receipt_json = signed
        pub = public_key_b64(signer)

        assert verify_odr(receipt_json, public_key=pub).valid

        # Flip one byte anywhere in the payload -> fail.
        tampered = copy.deepcopy(receipt_json)
        tampered["confidence"]["value"] = 0.85
        assert not verify_odr(tampered, public_key=pub).valid
