"""Both verifiers accept new optional content and unchanged v0.1 receipts."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "aragora-verify" / "src"))

from aragora_verify import schema, verify  # noqa: E402
from aragora.gauntlet.odr_export import decision_receipt_to_odr  # noqa: E402
from aragora.gauntlet.odr_verify import verify_odr_document  # noqa: E402
from aragora.gauntlet.receipt_models import DecisionReceipt  # noqa: E402
from aragora.gauntlet.odr_signing import sign_odr_receipt  # noqa: E402
from tests.gauntlet.odr_test_keys import odr_test_key  # noqa: E402


def legacy_document():
    return json.loads((ROOT / "docs/specs/examples/example-approved-clean.odr.json").read_text())


def test_emitter_v02_preserves_v01_absence():
    legacy = legacy_document()
    assert verify(legacy).ok and verify_odr_document(legacy).ok
    receipt = DecisionReceipt.from_dict(
        {"receipt_id": "test", "gauntlet_id": "demo", "verdict": "PASS"}
    )
    doc = decision_receipt_to_odr(receipt)
    assert doc["odr_version"] == "0.2"
    assert doc["profile"].endswith("/v0.2")
    assert doc["signatures"] == [] and "adjudication" not in doc
    assert not {"repository", "pr_number", "head_sha"} & doc["subject"].keys()
    assert verify(doc).ok and verify_odr_document(doc).ok


@pytest.mark.parametrize("version", ["0.1", "0.2"])
def test_three_member_signatures_verify_for_both_versions(version):
    doc = legacy_document()
    assert verify(doc).ok and verify_odr_document(doc).ok
    doc.update(
        odr_version=version, profile=f"https://aragora.ai/specs/open-decision-receipt/v{version}"
    )
    key = odr_test_key()
    signed = sign_odr_receipt(doc, key)
    assert set(signed["signatures"][0]) == {"alg", "key_id", "signature"}
    assert verify(signed, public_key=key.public_key()).ok
    assert verify_odr_document(signed, public_key=key.public_key()).ok


@pytest.mark.parametrize("version", ["0.1", "0.2"])
@pytest.mark.parametrize(
    "member", ["verdicts", "rule", "findings", "observations", "adjudication", "subject"]
)
def test_optional_content_types_and_unknowns(monkeypatch, version, member):
    monkeypatch.setattr(schema, "_jsonschema_errors", lambda doc: [])
    doc = legacy_document()
    assert verify(doc).ok and verify_odr_document(doc).ok
    doc.update(
        odr_version=version, profile=f"https://aragora.ai/specs/open-decision-receipt/v{version}"
    )
    blocks = {
        "verdicts": (doc["quorum"], [{"issuer": "reviewer", "counted": False}]),
        "rule": (doc["quorum"], {"required_signals": 2, "counted_families": ["claude"]}),
        "findings": (
            doc["quorum"]["dissent"],
            [{"issuer": "reviewer", "severity": "P1", "blocking": True, "text": "finding"}],
        ),
        "observations": (
            doc["reasoning"],
            [{"kind": "timeout", "family": "grok", "detail": "deadline"}],
        ),
        "adjudication": (
            doc,
            {
                "status": "present",
                "kind": "review_adjudication.v1",
                "verdict": "settle",
                "policy": {},
            },
        ),
        "subject": (
            doc,
            {**doc["subject"], "repository": "o/r", "pr_number": 1, "head_sha": "a" * 40},
        ),
    }
    parent, value = blocks[member]
    parent[member] = copy.deepcopy(value)
    assert verify(doc).ok and verify_odr_document(doc).ok
    target = parent[member][0] if isinstance(value, list) else parent[member]
    target["unexpected"] = True
    assert not verify(doc).ok and not verify_odr_document(doc).ok
    parent[member] = 42
    assert not verify(doc).ok and not verify_odr_document(doc).ok
