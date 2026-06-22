"""Tests for crux_set plumbing into the ODR exporter (#8227).

The receipt ODR export must surface the crux-finder map when present
(reading consensus_proof.metadata["cruxes"], #8366-aware) while still
passing calibration_provenance — neither is dropped.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from aragora.cli.commands.receipt import _crux_set_for_receipt, _export_odr


class TestCruxSetForReceipt:
    def test_extracts_cruxes_from_proof_metadata(self) -> None:
        proof = SimpleNamespace(
            metadata={
                "consensus_mode": "crux_finder",
                "cruxes": [
                    {"claim_id": "c0", "statement": "X", "crux_score": 0.9},
                    {"claim_id": "c1", "statement": "Y", "crux_score": 0.8},
                ],
            }
        )
        receipt = SimpleNamespace(consensus_proof=proof)
        crux_set = _crux_set_for_receipt(receipt)
        assert crux_set is not None
        assert len(crux_set) == 2
        assert crux_set[0]["claim_id"] == "c0"

    def test_returns_none_when_no_proof(self) -> None:
        receipt = SimpleNamespace(consensus_proof=None)
        assert _crux_set_for_receipt(receipt) is None

    def test_returns_none_when_no_cruxes(self) -> None:
        proof = SimpleNamespace(metadata={"consensus_mode": "majority"})
        receipt = SimpleNamespace(consensus_proof=proof)
        assert _crux_set_for_receipt(receipt) is None

    def test_returns_none_when_proof_has_no_metadata_attr(self) -> None:
        # Gauntlet ConsensusProof has no `metadata` field — must not crash.
        proof = SimpleNamespace(reached=True, confidence=0.9)
        receipt = SimpleNamespace(consensus_proof=proof)
        assert _crux_set_for_receipt(receipt) is None


def _minimal_receipt_dict() -> dict:
    """A minimal receipt dict that DecisionReceipt.from_dict accepts."""
    return {
        "receipt_id": "r-crux-1",
        "gauntlet_id": "g-1",
        "timestamp": "2026-06-22T00:00:00+00:00",
        "input_summary": "Should we use rust or go?",
        "input_hash": "0" * 64,
        "risk_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "attacks_attempted": 0,
        "attacks_successful": 0,
        "probes_run": 0,
        "vulnerabilities_found": 0,
        "verdict": "PASS",
        "confidence": 0.0,
        "robustness_score": 0.0,
    }


class TestExportOdrCruxPlumbing:
    def test_export_odr_absent_when_no_cruxes(self) -> None:
        odr = json.loads(_export_odr(_minimal_receipt_dict()))
        # Honest absence: no cruxes were carried, exporter records absent.
        assert odr["cruxes"]["status"] == "absent"

    def test_export_odr_preserves_calibration_block(self) -> None:
        # The confidence/calibration block must still be present in the output
        # structure regardless of crux presence (calibration_provenance not dropped).
        odr = json.loads(_export_odr(_minimal_receipt_dict()))
        assert "confidence" in odr
        assert "calibration" in odr["confidence"]
