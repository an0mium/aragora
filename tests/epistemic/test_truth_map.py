"""Tests for DIC-18 Organizational Truth Map (aragora.epistemic.truth_map)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.epistemic.truth_map import build_truth_map, build_truth_map_from_manifests


def _cr(cid: str, status: ClaimStatus, detail: dict | None = None) -> ClaimResult:
    return ClaimResult(claim_id=cid, status=status, message="", detail=detail or {})


def _mock_cfr(
    debate_id: str, question: str, scores: list[float], barrier: float = 0.5
) -> MagicMock:
    cruxes = []
    for i, s in enumerate(scores):
        c = MagicMock()
        c.crux_score = s
        c.to_dict.return_value = {"claim_id": f"c{i}", "crux_score": s}
        cruxes.append(c)
    analysis = MagicMock()
    analysis.cruxes = cruxes
    cfr = MagicMock()
    cfr.debate_id, cfr.question, cfr.analysis = debate_id, question, analysis
    cfr.top_cruxes.return_value = cruxes
    cfr.convergence_barrier.return_value = barrier
    return cfr


class TestBuildTruthMap:
    def test_empty_inputs_zero_counts(self) -> None:
        r = build_truth_map(claim_results=[])
        assert r.total_claims == 0 and r.open_crux_count == 0

    def test_generated_at_is_utc_iso(self) -> None:
        assert build_truth_map(claim_results=[]).generated_at.endswith("Z")

    def test_mixed_status_counts(self) -> None:
        results = [
            _cr("a", ClaimStatus.PASS),
            _cr("b", ClaimStatus.FAIL),
            _cr("c", ClaimStatus.STALE),
            _cr("d", ClaimStatus.UNSUPPORTED),
            _cr("e", ClaimStatus.ERROR),
        ]
        r = build_truth_map(claim_results=results)
        assert r.total_claims == 5
        assert (r.passing_claims, r.failing_claims, r.stale_claims) == (1, 1, 1)

    def test_claim_row_status_is_string(self) -> None:
        r = build_truth_map(claim_results=[_cr("x", ClaimStatus.FAIL)])
        assert r.claims[0].status == "fail"

    def test_metadata_populates_fields(self) -> None:
        meta = {
            "m": {
                "statement": "OK",
                "owner": "team",
                "verification": {"kind": "command", "command": "pytest"},
            }
        }
        r = build_truth_map(claim_results=[_cr("m", ClaimStatus.PASS)], claim_metadata=meta)
        row = r.claims[0]
        assert row.statement == "OK" and row.owner == "team"
        assert row.verifier_kind == "command" and row.verifier_command == "pytest"

    def test_missing_metadata_falls_back_to_empty(self) -> None:
        r = build_truth_map(claim_results=[_cr("x", ClaimStatus.PASS)])
        assert r.claims[0].statement == "" and r.claims[0].owner == ""

    def test_detail_evidence_age_and_follow_up_link(self) -> None:
        r = build_truth_map(
            claim_results=[
                _cr(
                    "x",
                    ClaimStatus.STALE,
                    {"evidence_age_hours": 36.5, "follow_up_link": "https://gh/1"},
                )
            ]
        )
        assert r.claims[0].evidence_age_hours == pytest.approx(36.5)
        assert r.claims[0].follow_up_link == "https://gh/1"

    def test_crux_summary_open_count_and_barrier(self) -> None:
        cfr = _mock_cfr("d1", "B2?", [0.8, 0.5, 0.1], barrier=0.6)
        r = build_truth_map(claim_results=[], crux_results=[cfr], open_crux_score_threshold=0.3)
        assert r.open_crux_count == 2
        row = r.crux_summaries[0]
        assert row.debate_id == "d1" and row.open_cruxes == 2
        assert row.convergence_barrier == pytest.approx(0.6)

    def test_top_k_limits_cruxes_in_summary(self) -> None:
        cfr = _mock_cfr("d", "Q", [0.9, 0.8, 0.7, 0.6])
        r = build_truth_map(claim_results=[], crux_results=[cfr], top_k_cruxes=2)
        assert len(r.crux_summaries[0].top_cruxes) == 2

    def test_multiple_crux_results_aggregate(self) -> None:
        r = build_truth_map(
            claim_results=[],
            crux_results=[_mock_cfr("d1", "Q1", [0.9, 0.1]), _mock_cfr("d2", "Q2", [0.7, 0.6])],
            open_crux_score_threshold=0.3,
        )
        assert r.open_crux_count == 3

    def test_to_dict_structure(self) -> None:
        results = [_cr("x", ClaimStatus.PASS)]
        d = build_truth_map(claim_results=results).to_dict()
        assert {"generated_at", "claims", "crux_summaries", "summary"} <= d.keys()
        assert d["summary"].keys() == {
            "total_claims",
            "passing",
            "failing",
            "stale",
            "unsupported",
            "error",
            "open_crux_count",
        }


class TestBuildTruthMapFromManifests:
    def test_real_manifest_dry_run(self) -> None:
        manifest = Path("docs/status/claims/proof_first_claims.yaml")
        if not manifest.exists():
            pytest.skip("DIC-13 manifest fixture not present")
        r = build_truth_map_from_manifests(manifest_paths=[manifest])
        assert r.total_claims > 0
        assert all(row.statement for row in r.claims)
        assert "b0.benchmark_truth.complete_current_corpus" in {row.claim_id for row in r.claims}
