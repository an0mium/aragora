"""Regression tests for the two [P2] integrity-messaging findings (PR #8388 review).

[P2a] A *signed* receipt verified WITHOUT a public key must NOT read "VERIFIED"
      / exit 0 -- present signatures that were never checked are not authenticity.
[P2b] Hash-chain anchoring must key on the content digest, not the mutable
      non-cryptographic receipt_id.
"""

from __future__ import annotations

import json

from aragora_verify import odr_content_digest, verify
from aragora_verify.cli import main
from aragora_verify.verifier import FAIL, PASS, load_public_key

from _fixtures import make_keypair, sign_odr, valid_odr


def _pem(public_key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


# --- [P2a] signed-but-unverified must not read VERIFIED --------------------


def test_signed_receipt_without_pubkey_is_authenticity_unverified():
    private_key, _ = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    result = verify(signed, public_key=None)
    # present signatures, none checked -> NOT a clean "verified"
    assert result.authenticity_unverified is True


def test_signed_receipt_with_pubkey_is_verified_and_not_flagged():
    private_key, public_key = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    result = verify(signed, public_key=load_public_key(_pem(public_key)))
    assert result.ok is True
    assert result.authenticity_unverified is False


def test_unsigned_receipt_is_not_flagged_authenticity_unverified():
    # An unsigned v0.1 receipt is the norm (WARN), not the "signed-but-unchecked" trap.
    result = verify(valid_odr(), public_key=None)
    assert result.authenticity_unverified is False


def test_cli_signed_receipt_without_pubkey_does_not_exit_zero(tmp_path):
    private_key, _ = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(signed))
    rc = main([str(path)])
    assert rc != 0  # must not silently report a green VERIFIED


# --- [P2b] chain anchoring keys on digest, not receipt_id ------------------


def _chain_check(result):
    return next(c for c in result.checks if c.name == "chain_link")


def test_chain_not_anchored_when_only_receipt_id_matches():
    doc = valid_odr()
    # entry references the mutable receipt_id but NOT the content digest
    chain = [{"hash": "a" * 64, "receipt_id": doc["receipt_id"]}]
    result = verify(doc, chain=chain)
    assert _chain_check(result).status == FAIL


def test_chain_anchored_when_digest_matches():
    doc = valid_odr()
    digest = odr_content_digest(doc)
    chain = [{"hash": digest}]
    result = verify(doc, chain=chain)
    assert _chain_check(result).status == PASS
