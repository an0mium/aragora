"""The committed example receipt is the emitter<->verifier contract. It must
be (a) exactly what odr_export emits today (regeneration guard) and (b)
schema-conformant with a recomputable JCS digest."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from aragora.gauntlet.odr_export import (
    decision_receipt_to_odr,
    load_odr_schema,
    odr_content_digest,
)

from tests.gauntlet.test_odr_export import _full_receipt

EXAMPLE = Path("docs/specs/examples/example-decision-receipt.odr.json")


def test_example_matches_current_emitter_output():
    expected = decision_receipt_to_odr(_full_receipt())
    actual = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert actual["odr_version"] == "0.1" and expected["odr_version"] == "0.2"
    jsonschema.validate(expected, load_odr_schema())
    expected.update(odr_version=actual["odr_version"], profile=actual["profile"])
    assert actual == expected, "example receipt is stale; regenerate it"


def test_example_is_schema_conformant_and_digestible():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    jsonschema.validate(doc, load_odr_schema())
    digest = odr_content_digest(doc)
    assert len(digest) == 64  # sha-256 hex
