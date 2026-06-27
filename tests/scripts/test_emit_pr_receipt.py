"""The PR-receipt emitter is the offline glue the M2 Action calls: it turns a
merge-quorum CollectOutcome JSON into a verifiable ODR receipt file. Pure
transformation — no model calls, no network."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from aragora.gauntlet.odr_export import load_odr_schema
from aragora.swarm.quorum_evidence import CollectOutcome, EvidenceItem

from scripts.emit_pr_receipt import build_receipt, main


def _outcome_dict() -> dict:
    outcome = CollectOutcome(
        repo="synaptent/aragora",
        pr=8667,
        head_sha="a" * 40,
        head_committed_at="2026-06-27T08:00:00+00:00",
        tier=1,
        action="prepare",
        action_reason="supportive quorum reached",
        items=[
            EvidenceItem(family="claude", body="PASS", would_count=True, verdict="pass"),
            EvidenceItem(family="openai", body="PASS", would_count=True, verdict="pass"),
        ],
    )
    return outcome.to_dict()


def test_build_receipt_is_schema_conformant():
    odr = build_receipt(_outcome_dict())
    jsonschema.validate(odr, load_odr_schema())
    assert odr["source"]["system"] == "aragora"
    assert "8667" in odr["receipt_id"]


def test_main_writes_receipt_and_github_outputs(tmp_path: Path):
    outcome_path = tmp_path / "outcome.json"
    outcome_path.write_text(json.dumps(_outcome_dict()), encoding="utf-8")
    out_path = tmp_path / "receipt.odr.json"
    gh_out = tmp_path / "gh_output"

    rc = main(
        [
            "--outcome",
            str(outcome_path),
            "--out",
            str(out_path),
            "--verify",
            "--github-output",
            str(gh_out),
        ]
    )
    assert rc == 0

    # receipt written and re-verifiable
    odr = json.loads(out_path.read_text(encoding="utf-8"))
    jsonschema.validate(odr, load_odr_schema())

    # GitHub Actions key=value outputs emitted
    gh = gh_out.read_text(encoding="utf-8")
    assert "receipt_verdict=PASS" in gh
    assert "receipt_verified=true" in gh
    assert "receipt_digest=" in gh
    assert "receipt_path=" in gh
