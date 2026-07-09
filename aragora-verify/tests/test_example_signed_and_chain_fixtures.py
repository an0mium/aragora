"""Independent verification of the SIGNED golden example fixture and the
``--chain`` JSONL fixtures (m2-odr-signed-and-chain-fixtures): a signed ODR
receipt + its Ed25519 public key, and three hash-chain anchoring outcomes
(anchored -> WARN, not-anchored -> FAIL, broken-links -> FAIL). Mirrors
test_example_merge_quorum_receipt.py / test_example_unsigned_state_fixtures.py
(dict-level ``verify()`` plus the packaged ``aragora-verify`` CLI via
``aragora_verify.cli.main()``). The committed example files are the only
artifacts crossing the package boundary.

Signing recipe (documented for reproducibility): a fixed unsigned BASE
document was held constant; its content digest was pinned via
``aragora_verify.odr_content_digest(BASE)`` (== the in-tree
``aragora.gauntlet.odr_export.odr_content_digest`` -- signatures are excluded
from the digest, so signing never changes it). The committed
example-signed.odr.json is BASE signed via
``aragora.gauntlet.odr_signing.generate_signing_key()`` ->
``sign_odr_receipt(BASE, key)``; the matching public key was written via
``public_key_pem(key)`` to example-signed.pubkey.pem. Because signing is
detached, re-running ``odr_content_digest`` against the committed (signed)
file reproduces the exact digest pinned below and in
example-signed.digest.json.

Signing stays opt-in throughout this fixture set: no shipping default is
flipped to signed, and the unsigned fixtures elsewhere in this directory are
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization

from aragora_verify import odr_content_digest, verify
from aragora_verify.cli import main
from aragora_verify.verifier import FAIL, PASS

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "specs" / "examples"
SIGNED = EXAMPLES_DIR / "example-signed.odr.json"
PUBKEY = EXAMPLES_DIR / "example-signed.pubkey.pem"
DIGEST_SIDECAR = EXAMPLES_DIR / "example-signed.digest.json"
CHAIN_ANCHORED = EXAMPLES_DIR / "example-chain-anchored.jsonl"
CHAIN_NOT_ANCHORED = EXAMPLES_DIR / "example-chain-not-anchored.jsonl"
CHAIN_BROKEN = EXAMPLES_DIR / "example-chain-broken-links.jsonl"

# Pinned reproducible content digest for example-signed.odr.json (VAL-RECON-011).
# Recomputing odr_content_digest on the committed file must always yield this
# value; a mismatch would signal a JCS/serialization regression that silently
# invalidates every signature riding on this digest.
PINNED_DIGEST = "6d7f70d080876e0f9d58b2016725a70285bdfdb4244b9341436afa4308d40405"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chain(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _check(result: Any, name: str) -> Any:
    check = next((c for c in result.checks if c.name == name), None)
    assert check is not None, f"check {name!r} not found in {[c.name for c in result.checks]}"
    return check


# --- signed golden: structural conformance + digest pin --------------------


def test_signed_golden_verifies_independently_without_pubkey() -> None:
    doc = _load_json(SIGNED)
    result = verify(doc)  # no pubkey: signature check is SKIP, not FAIL
    failed = [c for c in result.checks if c.status == FAIL]
    assert not failed, failed
    assert _check(result, "schema_conformance").status == PASS
    assert _check(result, "canonical_digest").status == PASS
    # signed, but nothing checked it yet -> not a clean "verified"
    assert result.authenticity_unverified is True


def test_signed_golden_carries_at_least_one_signature() -> None:
    doc = _load_json(SIGNED)
    assert isinstance(doc.get("signatures"), list)
    assert len(doc["signatures"]) >= 1
    assert doc["signatures"][0]["alg"] == "Ed25519"


def test_signed_golden_digest_matches_pinned_sidecar() -> None:
    doc = _load_json(SIGNED)
    sidecar = _load_json(DIGEST_SIDECAR)
    digest = odr_content_digest(doc)
    assert digest == sidecar["odr_digest"]
    assert digest == PINNED_DIGEST


def test_signed_golden_digest_reproducible_across_runs() -> None:
    doc = _load_json(SIGNED)
    first = odr_content_digest(doc)
    # Recompute against a freshly round-tripped copy -- same bytes in, same
    # digest out, independent of dict identity/ordering in memory.
    second = odr_content_digest(json.loads(json.dumps(doc)))
    assert first == second == PINNED_DIGEST


def test_signed_golden_digest_matches_in_tree_emitter() -> None:
    # The standalone JCS port must stay byte-identical to the canonical
    # in-tree emitter (guards against the two implementations drifting).
    canonical = pytest.importorskip("aragora.gauntlet.odr_export")
    doc = _load_json(SIGNED)
    assert odr_content_digest(doc) == canonical.odr_content_digest(doc)


# --- signed golden: CLI exit-code contract ----------------------------------


def test_cli_signed_golden_with_correct_pubkey_exits_zero() -> None:
    rc = main([str(SIGNED), "--pubkey", str(PUBKEY)])
    assert rc == 0


def test_cli_signed_golden_without_pubkey_exits_three() -> None:
    rc = main([str(SIGNED)])
    assert rc == 3


def test_cli_signed_golden_json_reports_pinned_digest(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(SIGNED), "--pubkey", str(PUBKEY), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["odr_digest"] == PINNED_DIGEST
    assert payload["ok"] is True
    assert payload["authenticity_unverified"] is False


def test_cli_signed_golden_single_byte_tamper_exits_one(tmp_path: Path) -> None:
    raw_text = SIGNED.read_text(encoding="utf-8")
    marker = '"odr_version": "0.1"'
    assert marker in raw_text
    tampered_text = raw_text.replace(marker, '"odr_version": "0.2"', 1)
    path = tmp_path / "example-signed-tampered.odr.json"
    path.write_text(tampered_text, encoding="utf-8")
    rc = main([str(path), "--pubkey", str(PUBKEY)])
    assert rc == 1


def test_cli_signed_golden_content_tamper_caught_by_signature_not_schema() -> None:
    # A tamper that keeps the document schema-conformant (a different, but
    # still valid, verdict string) must still be caught -- by the Ed25519
    # signature, since the digest it covers has changed underneath it.
    doc = _load_json(SIGNED)
    doc["claim"]["verdict"] = "CHANGES_REQUESTED"
    result = verify(doc, public_key=_load_public_key(PUBKEY))
    assert result.ok is False
    assert _check(result, "schema_conformance").status == PASS
    assert _check(result, "signature").status == FAIL


def test_cli_signed_golden_with_wrong_ed25519_pubkey_exits_one(tmp_path: Path) -> None:
    # A structurally valid Ed25519 key that did NOT sign this receipt is a
    # signature failure, not a usage error: exit 1, distinct from the
    # "key rejected before any verification" exit 2 below (VAL-VERIFY-012a).
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    wrong_public_key = Ed25519PrivateKey.generate().public_key()
    wrong_pem_path = tmp_path / "wrong-ed25519.pem"
    wrong_pem_path.write_bytes(
        wrong_public_key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    rc = main([str(SIGNED), "--pubkey", str(wrong_pem_path)])
    assert rc == 1


def test_cli_signed_golden_with_non_ed25519_pubkey_exits_two(tmp_path: Path) -> None:
    # An RSA key is well-formed PEM but the wrong algorithm; load_public_key()
    # rejects it BEFORE any signature check runs -- a usage/input error
    # (exit 2), distinct from the signature mismatch (exit 1) above
    # (VAL-VERIFY-012b).
    from cryptography.hazmat.primitives.asymmetric import rsa

    rsa_public_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    rsa_pem_path = tmp_path / "wrong-rsa.pem"
    rsa_pem_path.write_bytes(
        rsa_public_key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    rc = main([str(SIGNED), "--pubkey", str(rsa_pem_path)])
    assert rc == 2


def _load_public_key(path: Path):  # noqa: ANN202 - thin test helper
    from aragora_verify.verifier import load_public_key

    return load_public_key(path.read_bytes())


# --- chain fixtures: anchored (WARN) / not-anchored (FAIL) / broken-links (FAIL) ---


def test_chain_anchored_variant_contains_the_pinned_digest() -> None:
    chain = _load_chain(CHAIN_ANCHORED)
    values = {str(v) for entry in chain for v in entry.values() if isinstance(v, (str, int))}
    assert PINNED_DIGEST in values


def test_chain_not_anchored_variant_omits_the_digest() -> None:
    chain = _load_chain(CHAIN_NOT_ANCHORED)
    values = {str(v) for entry in chain for v in entry.values() if isinstance(v, (str, int))}
    assert PINNED_DIGEST not in values


def test_chain_anchored_variant_exits_zero_with_warn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([str(SIGNED), "--pubkey", str(PUBKEY), "--chain", str(CHAIN_ANCHORED), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks_by_name = {c["name"]: c for c in payload["checks"]}
    assert checks_by_name["chain_link"]["status"] == "warn"
    assert "not an integrity proof" in checks_by_name["chain_link"]["detail"]


def test_chain_not_anchored_variant_exits_one_with_not_anchored_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([str(SIGNED), "--pubkey", str(PUBKEY), "--chain", str(CHAIN_NOT_ANCHORED), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    checks_by_name = {c["name"]: c for c in payload["checks"]}
    assert checks_by_name["chain_link"]["status"] == "fail"
    assert "not anchored" in checks_by_name["chain_link"]["detail"]


def test_chain_broken_links_variant_exits_one_with_broken_linkage_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([str(SIGNED), "--pubkey", str(PUBKEY), "--chain", str(CHAIN_BROKEN), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    checks_by_name = {c["name"]: c for c in payload["checks"]}
    assert checks_by_name["chain_link"]["status"] == "fail"
    assert "broken" in checks_by_name["chain_link"]["detail"]
    assert "linkage" in checks_by_name["chain_link"]["detail"]


def test_not_anchored_and_broken_links_report_distinct_chain_link_details() -> None:
    doc = _load_json(SIGNED)
    not_anchored_result = verify(doc, chain=_load_chain(CHAIN_NOT_ANCHORED))
    broken_result = verify(doc, chain=_load_chain(CHAIN_BROKEN))
    not_anchored_detail = _check(not_anchored_result, "chain_link").detail
    broken_detail = _check(broken_result, "chain_link").detail
    assert not_anchored_detail != broken_detail
    assert "not anchored" in not_anchored_detail
    assert "broken" in broken_detail


def test_broken_links_variant_is_anchored_but_still_fails_on_linkage() -> None:
    # The broken-links variant DOES contain the digest (it is "anchored" in
    # the data sense) -- it fails purely because its declared prev_hash chain
    # is inconsistent, proving the two failure modes are independently caught.
    chain = _load_chain(CHAIN_BROKEN)
    values = {str(v) for entry in chain for v in entry.values() if isinstance(v, (str, int))}
    assert PINNED_DIGEST in values
    doc = _load_json(SIGNED)
    result = verify(doc, chain=chain)
    assert _check(result, "chain_link").status == FAIL
