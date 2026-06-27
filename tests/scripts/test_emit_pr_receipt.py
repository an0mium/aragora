"""The PR-receipt emitter is the offline glue the M2 Action calls: it turns a
merge-quorum CollectOutcome JSON into a verifiable ODR receipt file. Pure
transformation — no model calls, no network."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from aragora.gauntlet.odr_export import load_odr_schema
from aragora.swarm.quorum_evidence import CollectOutcome, EvidenceItem

from scripts.emit_pr_receipt import build_receipt, main, verify_receipt


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


def test_verify_degrades_without_jsonschema(monkeypatch):
    # Regression for the live-CI crash: a slim runtime without jsonschema must
    # degrade to digest-only, never raise ModuleNotFoundError.
    import builtins

    odr = build_receipt(_outcome_dict())
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ModuleNotFoundError("No module named 'jsonschema'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    digest, fully = verify_receipt(odr)
    assert len(digest) == 64
    assert fully is False


def test_main_writes_receipt_even_when_verify_degrades(tmp_path: Path, monkeypatch):
    import builtins

    outcome_path = tmp_path / "outcome.json"
    outcome_path.write_text(json.dumps(_outcome_dict()), encoding="utf-8")
    out_path = tmp_path / "receipt.odr.json"
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ModuleNotFoundError("No module named 'jsonschema'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = main(["--outcome", str(outcome_path), "--out", str(out_path), "--verify"])
    assert rc == 0
    assert out_path.is_file()  # receipt written despite no jsonschema


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
