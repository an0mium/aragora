"""Regression guard for the compliance-walkthrough fixture.

``docs/compliance/ODR_VERIFICATION_WALKTHROUGH.md`` promises an outsider that
the checked-in sample receipt (a) verifies against the checked-in public key,
(b) fails verification when tampered with, and (c) is exactly what the
generator script and the reference emitter produce. These tests keep those
promises true on every commit.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from aragora.gauntlet.odr_export import (
    decision_receipt_to_odr,
    load_odr_schema,
    odr_content_digest,
)
from aragora.gauntlet.odr_verify import load_public_key, verify_odr_document
from aragora.gauntlet.receipt_models import DecisionReceipt

FIXTURES = Path("docs/compliance/fixtures")
ODR_PATH = FIXTURES / "sample_decision_receipt.odr.json"
NATIVE_PATH = FIXTURES / "sample_decision_receipt.json"
PUBKEY_PATH = FIXTURES / "odr_sample_signing_public_key.pem"
GENERATOR = Path("scripts/generate_odr_fixture.py")


@pytest.fixture(scope="module")
def odr_doc() -> dict:
    return json.loads(ODR_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def public_key():
    return load_public_key(PUBKEY_PATH.read_bytes())


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_odr_fixture", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_verifies_with_checked_in_public_key(odr_doc, public_key):
    result = verify_odr_document(odr_doc, public_key=public_key)
    by_name = {check.name: check.status for check in result.checks}
    assert result.ok, [f"{c.name}: {c.detail}" for c in result.checks if c.status == "fail"]
    assert by_name["schema_conformance"] == "pass"
    assert by_name["canonical_digest"] == "pass"
    assert by_name["signature"] == "pass"
    assert by_name["quorum_consistency"] == "pass"


def test_fixture_is_schema_conformant(odr_doc):
    jsonschema.validate(odr_doc, load_odr_schema())


def test_tampered_verdict_fails_signature(odr_doc, public_key):
    tampered = copy.deepcopy(odr_doc)
    tampered["claim"]["verdict"] = "PASS"  # the walkthrough's attacker scenario
    result = verify_odr_document(tampered, public_key=public_key)
    by_name = {check.name: check.status for check in result.checks}
    assert not result.ok
    assert by_name["signature"] == "fail"


def test_fixture_matches_generator_receipt_content(odr_doc):
    """The signed fixture's content (minus signatures) is exactly what the
    generator script's receipt exports to today — regeneration guard."""
    module = _load_generator_module()
    expected = decision_receipt_to_odr(module.build_sample_receipt())
    actual = {k: v for k, v in odr_doc.items() if k != "signatures"}
    expected = {k: v for k, v in expected.items() if k != "signatures"}
    assert actual == expected, (
        "walkthrough fixture is stale; regenerate with "
        "`python scripts/generate_odr_fixture.py --output-dir docs/compliance/fixtures`"
    )


def test_native_fixture_exports_to_the_signed_odr_fixture(odr_doc):
    """The two fixture files describe the same decision: exporting the native
    receipt reproduces the ODR document byte-for-byte (minus signatures)."""
    native = json.loads(NATIVE_PATH.read_text(encoding="utf-8"))
    receipt = DecisionReceipt.from_dict(native)
    exported = decision_receipt_to_odr(receipt)
    assert {k: v for k, v in exported.items() if k != "signatures"} == {
        k: v for k, v in odr_doc.items() if k != "signatures"
    }
    # And the digest the signature covers is recomputable from either path.
    assert odr_content_digest(exported) == odr_content_digest(odr_doc)


def test_walkthrough_references_exist():
    """Files the walkthrough tells an auditor to use must exist."""
    for path in (
        ODR_PATH,
        NATIVE_PATH,
        PUBKEY_PATH,
        FIXTURES / "README.md",
        Path("docs/compliance/ODR_VERIFICATION_WALKTHROUGH.md"),
        Path("docs/specs/OPEN_DECISION_RECEIPT.md"),
        GENERATOR,
    ):
        assert path.exists(), f"missing walkthrough artifact: {path}"
