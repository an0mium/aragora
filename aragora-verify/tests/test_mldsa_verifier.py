"""Canonical verifier support for ML-DSA-65 ODR signatures (#8612)."""

# ruff: noqa: E402

from __future__ import annotations

import base64
import copy
import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "aragora-verify" / "src", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aragora_verify.cli import main
from aragora_verify.verifier import FAIL, PASS, load_mldsa_public_key, pqc_available, verify

from _fixtures import make_keypair, sign_odr, valid_odr

signing = pytest.importorskip(
    "aragora.gauntlet.odr_signing",
    reason="hybrid producer package is unavailable in standalone verifier installs",
)

pytestmark = pytest.mark.skipif(
    not pqc_available(),
    reason="cryptography build lacks ML-DSA (needs cryptography>=49 / OpenSSL 3.5+)",
)


def _signature_check(result):
    return next(c for c in result.checks if c.name == "signature")


def _ed25519_public_bytes(public_key) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _write_json(tmp_path: Path, name: str, doc: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_mldsa_key_id_matches_producer_and_public_key_loader_accepts_encodings() -> None:
    key = signing.generate_mldsa_signing_key()
    public_key = key.public_key()
    raw = public_key.public_bytes_raw()
    pem = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    from aragora_verify.verifier import compute_mldsa_key_id

    assert compute_mldsa_key_id(public_key) == signing.compute_mldsa_key_id(public_key)
    for encoded in (
        raw,
        pem,
        der,
        base64.b64encode(raw),
        raw.hex().encode("ascii"),
    ):
        assert compute_mldsa_key_id(load_mldsa_public_key(encoded)) == compute_mldsa_key_id(
            public_key
        )


def test_hybrid_receipt_verifies_both_signature_algorithms() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )

    result = verify(
        signed,
        public_key=ed_public,
        mldsa_public_key=mldsa_private.public_key(),
    )

    assert result.ok is True
    signature = _signature_check(result)
    assert signature.status == PASS
    assert "sig[0] Ed25519" in signature.detail
    assert "sig[1] ML-DSA-65" in signature.detail
    assert "verified" in signature.detail


def test_hybrid_receipt_tamper_fails_signature_check() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )
    tampered = copy.deepcopy(signed)
    tampered["claim"]["statement"] = "merge a different PR"

    result = verify(
        tampered,
        public_key=ed_public,
        mldsa_public_key=mldsa_private.public_key(),
    )

    assert result.ok is False
    assert _signature_check(result).status == FAIL


def test_hybrid_receipt_wrong_mldsa_key_fails_even_when_ed25519_key_is_valid() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    wrong_mldsa_public = signing.generate_mldsa_signing_key().public_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )

    result = verify(
        signed,
        public_key=ed_public,
        mldsa_public_key=wrong_mldsa_public,
    )

    assert result.ok is False
    signature = _signature_check(result)
    assert signature.status == FAIL
    assert "ML-DSA-65" in signature.detail


def test_hybrid_receipt_mldsa_key_id_mismatch_fails() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )
    signed["signatures"][1]["key_id"] = "ml-dsa-65-not-the-key"

    result = verify(
        signed,
        public_key=ed_public,
        mldsa_public_key=mldsa_private.public_key(),
    )

    assert result.ok is False
    assert _signature_check(result).status == FAIL
    assert "key_id mismatch" in _signature_check(result).detail


def test_hybrid_receipt_ed25519_key_id_mismatch_fails() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )
    signed["signatures"][0]["key_id"] = "ed25519-not-the-key"

    result = verify(
        signed,
        public_key=ed_public,
        mldsa_public_key=mldsa_private.public_key(),
    )

    assert result.ok is False
    assert _signature_check(result).status == FAIL
    assert "key_id mismatch" in _signature_check(result).detail


def test_hybrid_receipt_malformed_mldsa_signature_fails_cleanly() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )
    signed["signatures"][1]["signature"] = base64.b64encode(b"not-an-mldsa-signature").decode(
        "ascii"
    )

    result = verify(
        signed,
        public_key=ed_public,
        mldsa_public_key=mldsa_private.public_key(),
    )

    assert result.ok is False
    assert _signature_check(result).status == FAIL


def test_hybrid_receipt_without_mldsa_key_skips_mldsa_entry_but_ed25519_passes() -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )

    result = verify(signed, public_key=ed_public)

    assert result.ok is True
    signature = _signature_check(result)
    assert signature.status == PASS
    assert "sig[0] Ed25519" in signature.detail
    assert "sig[1] ML-DSA-65" in signature.detail
    assert "skipped" in signature.detail


def test_ed25519_only_receipt_still_verifies_with_ed25519_key() -> None:
    private_key, public_key = make_keypair()
    signed = sign_odr(valid_odr(), private_key)

    result = verify(signed, public_key=public_key)

    assert result.ok is True
    assert _signature_check(result).status == PASS


def test_ed25519_only_receipt_fails_when_mldsa_key_was_required() -> None:
    private_key, public_key = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    mldsa_public = signing.generate_mldsa_signing_key().public_key()

    result = verify(signed, public_key=public_key, mldsa_public_key=mldsa_public)

    assert result.ok is False
    signature = _signature_check(result)
    assert signature.status == FAIL
    assert "no ML-DSA-65 signature present" in signature.detail


def test_cli_accepts_mldsa_pubkey_for_hybrid_receipt(tmp_path: Path) -> None:
    ed_private, ed_public = make_keypair()
    mldsa_private = signing.generate_mldsa_signing_key()
    signed = signing.sign_odr_hybrid(
        valid_odr(),
        ed25519_key=ed_private,
        mldsa_key=mldsa_private,
    )
    receipt_path = _write_json(tmp_path, "hybrid.json", signed)
    ed_path = _write(tmp_path, "ed25519.pem", _ed25519_public_bytes(ed_public))
    mldsa_path = _write(tmp_path, "mldsa.raw", mldsa_private.public_key().public_bytes_raw())

    assert main([receipt_path, "--pubkey", ed_path, "--mldsa-pubkey", mldsa_path]) == 0
