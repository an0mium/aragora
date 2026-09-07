"""Independent verification of a PR-review (merge-quorum) receipt. This is M2's
external proof: a heterogeneous-model PR review, bridged to a DecisionReceipt
and exported to ODR, is verified by aragora-verify with zero Aragora dependency.
The committed example JSON is the only artifact crossing the package boundary."""

from __future__ import annotations

import json
from pathlib import Path

from aragora_verify import verify
from aragora_verify.verifier import FAIL, PASS  # status constants (verified)

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "specs"
    / "examples"
    / "example-merge-quorum-receipt.odr.json"
)


def test_merge_quorum_receipt_verifies_independently():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    result = verify(doc)  # unsigned: the "signature" check is WARN, not FAIL
    failed = [c for c in result.checks if c.status == FAIL]
    assert result.ok, failed
    assert not failed
    statuses = {c.name: c.status for c in result.checks}
    assert statuses["schema_conformance"] == PASS
    assert statuses["canonical_digest"] == PASS
    # the bridge keeps supporting/dissenting agents within participants
    assert statuses.get("quorum_consistency") != FAIL


def test_merge_quorum_receipt_discloses_multiple_model_families():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    independence = doc["quorum"]["independence"]
    assert independence["disclosed"] is True
    assert independence["distinct_model_families"] >= 2
