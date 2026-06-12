"""Tests for the public crux-finder CLI surface (ODR-4 / #8227).

Covers:
- `aragora ask --crux` parser wiring (flag exists, defaults off)
- crux payload extraction helpers (`extract_crux_payload`, skip reason)
- crux report shaping/printing for the CLI
- `crux_set` persisted into the debate receipt for crux_finder runs
- `aragora receipt export --format odr` populating the ODR cruxes field
  from a stored `crux_set` (and the absent marker otherwise)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.cli.commands.debate import (
    _crux_report_from_result,
    _persist_debate_receipt,
    _print_crux_report,
)
from aragora.cli.commands.receipt import _export_odr
from aragora.cli.parser import build_parser

CRUX_ITEM = {
    "claim_id": "claim_1",
    "statement": "The migration is reversible",
    "author": "claude",
    "crux_score": 0.77,
    "influence_score": 0.6,
    "disagreement_score": 0.8,
    "uncertainty_score": 0.55,
    "centrality_score": 0.5,
    "affected_claims": ["claim_2"],
    "contesting_agents": ["claude", "codex"],
    "resolution_impact": 0.35,
}


def _crux_result(cruxes: list[dict] | None = None) -> SimpleNamespace:
    """DebateResult-shaped object as produced by a crux_finder run."""
    items = CRUX_ITEM.copy() if cruxes is None else None
    crux_list = [items] if cruxes is None else cruxes
    proof = SimpleNamespace(
        final_claim="__CRUX_MAP__: no verdict by design; see CruxReceipt.cruxes",
        metadata={
            "consensus_mode": "crux_finder",
            "cruxes": crux_list,
            "crux_count": len(crux_list),
            "convergence_barrier": 0.31,
            "counterfactuals": [{"claim_id": "claim_1"}],
            "recommended_focus": ["claim_1"],
        },
    )
    return SimpleNamespace(
        debate_id="debate-crux",
        task="Should we migrate?",
        consensus_reached=False,
        confidence=0.0,
        final_answer=proof.final_claim,
        rounds_used=2,
        dissenting_views=[],
        consensus_proof=proof,
        metadata={},
        messages=[
            SimpleNamespace(agent="claude", role="proposer", round=0, content="Proposal A"),
        ],
    )


class TestAskCruxFlag:
    def test_parser_accepts_crux_flag(self):
        parser = build_parser()
        ns = parser.parse_args(["ask", "should we migrate?", "--crux"])
        assert ns.crux is True

    def test_crux_flag_defaults_off(self):
        parser = build_parser()
        ns = parser.parse_args(["ask", "should we migrate?"])
        assert ns.crux is False


class TestCruxReportFromResult:
    def test_present_report_shape(self):
        report = _crux_report_from_result(_crux_result())
        assert report["debate_id"] == "debate-crux"
        assert report["cruxes"]["status"] == "present"
        assert report["cruxes"]["items"][0]["statement"] == CRUX_ITEM["statement"]
        assert report["crux_count"] == 1
        assert report["convergence_barrier"] == 0.31
        assert report["consensus_mode"] == "crux_finder"

    def test_absent_report_when_no_crux_data(self):
        result = SimpleNamespace(
            debate_id="debate-plain",
            consensus_proof=SimpleNamespace(metadata={}),
            metadata={},
        )
        report = _crux_report_from_result(result)
        assert report["cruxes"]["status"] == "absent"
        assert report["crux_count"] == 0
        assert "items" not in report["cruxes"]

    def test_absent_report_surfaces_skip_reason(self):
        result = SimpleNamespace(
            debate_id="debate-skipped",
            consensus_proof=None,
            metadata={"crux_finder_skipped_reason": "no_belief_network"},
        )
        report = _crux_report_from_result(result)
        assert report["cruxes"]["status"] == "absent"
        assert "no_belief_network" in report["cruxes"]["reason"]


class TestPrintCruxReport:
    def test_prints_present_cruxes(self, capsys: pytest.CaptureFixture[str]):
        _print_crux_report(_crux_report_from_result(_crux_result()))
        out = capsys.readouterr().out
        assert "CRUXES" in out
        assert CRUX_ITEM["statement"] in out
        assert "claude, codex" in out
        assert "Convergence barrier: 0.31" in out
        assert "claim_1" in out

    def test_prints_absent_reason(self, capsys: pytest.CaptureFixture[str]):
        _print_crux_report(
            {
                "debate_id": "d",
                "cruxes": {"status": "absent", "reason": "crux mode was not enabled"},
                "crux_count": 0,
            }
        )
        out = capsys.readouterr().out
        assert "No crux set recorded" in out
        assert "crux mode was not enabled" in out


class TestReceiptCruxSet:
    def test_crux_set_persisted_for_crux_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        receipt_path = _persist_debate_receipt(_crux_result())
        assert receipt_path is not None
        data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        assert data["crux_set"] == [CRUX_ITEM]
        assert data["crux_metadata"]["consensus_mode"] == "crux_finder"
        assert data["crux_metadata"]["convergence_barrier"] == 0.31
        assert data["consensus_proof"]["method"] == "crux_finder"

    def test_no_crux_set_for_plain_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = SimpleNamespace(
            debate_id="debate-plain",
            task="Plain debate",
            consensus_reached=True,
            confidence=0.9,
            final_answer="Use Redis",
            rounds_used=1,
            dissenting_views=[],
            metadata={},
            messages=[
                SimpleNamespace(agent="claude", role="proposer", round=0, content="Use Redis"),
            ],
        )
        receipt_path = _persist_debate_receipt(result)
        assert receipt_path is not None
        data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        # Honest absence: no crux keys are fabricated for non-crux runs.
        assert "crux_set" not in data
        assert "crux_metadata" not in data


class TestOdrExportCruxes:
    def test_odr_export_populates_cruxes_from_crux_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        receipt_path = _persist_debate_receipt(_crux_result())
        assert receipt_path is not None
        data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))

        odr = json.loads(_export_odr(data))
        assert odr["cruxes"]["status"] == "present"
        assert odr["cruxes"]["items"] == [CRUX_ITEM]

    def test_odr_export_absent_marker_without_crux_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = SimpleNamespace(
            debate_id="debate-plain",
            task="Plain debate",
            consensus_reached=True,
            confidence=0.9,
            final_answer="Use Redis",
            rounds_used=1,
            dissenting_views=[],
            metadata={},
            messages=[
                SimpleNamespace(agent="claude", role="proposer", round=0, content="Use Redis"),
            ],
        )
        receipt_path = _persist_debate_receipt(result)
        assert receipt_path is not None
        data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))

        odr = json.loads(_export_odr(data))
        assert odr["cruxes"]["status"] == "absent"
        assert odr["cruxes"]["reason"]

    def test_odr_export_with_cruxes_validates_against_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        jsonschema = pytest.importorskip("jsonschema")
        from aragora.gauntlet.odr_export import load_odr_schema

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        receipt_path = _persist_debate_receipt(_crux_result())
        assert receipt_path is not None
        data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))

        odr = json.loads(_export_odr(data))
        validator = jsonschema.Draft202012Validator(load_odr_schema())
        errors = list(validator.iter_errors(odr))
        assert not errors, [e.message for e in errors]
