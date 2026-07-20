"""Independent verification of an emitter-produced ODR receipt. This is M1's
external proof: aragora-verify (zero Aragora dependency) confirms a receipt
that aragora/gauntlet/odr_export.py produced. The example JSON is the only
artifact crossing the package boundary."""

from __future__ import annotations

import json
from pathlib import Path

from aragora_verify import verify
from aragora_verify.verifier import FAIL, PASS  # status constants (verified)

# repo root = three parents up from aragora-verify/tests/<file>
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "specs"
    / "examples"
    / "example-decision-receipt.odr.json"
)


def test_emitter_receipt_verifies_independently():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    result = verify(doc)  # unsigned: the "signature" check is WARN, not FAIL
    failed = [c for c in result.checks if c.status == FAIL]
    assert result.ok, failed
    assert not failed
    statuses = {c.name: c.status for c in result.checks}
    # schema + digest must affirmatively pass; quorum is PASS or SKIP, never FAIL.
    assert statuses["schema_conformance"] == PASS
    assert statuses["canonical_digest"] == PASS
    assert statuses.get("quorum_consistency") != FAIL
