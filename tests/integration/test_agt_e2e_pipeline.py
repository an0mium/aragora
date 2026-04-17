"""End-to-end AGT pipeline integration test.

Runs the full synthetic pipeline implemented in
``scripts/agt_pipeline_dry_run.py`` and verifies every stage's output
against the canonical fixture at
``docs/status/generated/agt_e2e_trace.json``.

Why this test matters:
- Each of the 11 AGT modules has unit tests in isolation
- This test is the only place that exercises their shapes against
  each other — if a module tightens its input contract, this test
  catches the integration break before downstream consumers do
- The fixture is a committed artifact; drift is visible in a diff

When the fixture legitimately needs to change (e.g. a new field
added to a schema), regenerate it:
    python3 scripts/agt_pipeline_dry_run.py --pin-timestamp

and commit the updated JSON alongside the schema change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agt_pipeline_dry_run import (  # noqa: E402
    DEFAULT_OUTPUT,
    _pinned_now,
    run_pipeline,
)

from aragora.reasoning.cruxset import CruxSet  # noqa: E402
from aragora.reputation.types import ReputationDelta  # noqa: E402


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    """Run the pipeline once per test module using the pinned timestamp."""
    tmp = tmp_path_factory.mktemp("agt_e2e")
    return run_pipeline(now=_pinned_now(), tmp_dir=tmp)


class TestPipelineStructure:
    def test_trace_has_all_stages(self, trace) -> None:
        expected_keys = {
            "pipeline",
            "generated_at",
            "cruxset",
            "agent_receipt",
            "dic17_proposals_from_cruxset",
            "market",
            "position",
            "resolution",
            "claim",
            "resolved_claim",
            "reputation_delta",
            "anchor_receipt",
            "failure_branch",
            "viah_report",
        }
        assert expected_keys.issubset(trace.keys())

    def test_generated_at_is_pinned(self, trace) -> None:
        assert trace["generated_at"] == "2026-04-17T12:00:00Z"


class TestCruxSetStage:
    def test_cruxset_has_content_addressed_id(self, trace) -> None:
        cruxset_id = trace["cruxset"]["cruxset_id"]
        assert cruxset_id.startswith("crxset_")

    def test_cruxset_checksum_verifies(self, trace) -> None:
        cs = CruxSet.from_json(trace["cruxset"])
        assert cs.verify_checksum() is True

    def test_cruxset_cruxes_sorted_by_score_desc(self, trace) -> None:
        scores = [c["load_bearing_score"] for c in trace["cruxset"]["cruxes"]]
        assert scores == sorted(scores, reverse=True)


class TestAgentReceiptStage:
    def test_agent_receipt_wraps_cruxset(self, trace) -> None:
        receipt = trace["agent_receipt"]
        assert receipt["subject_kind"] == "debate_outcome"
        assert receipt["cruxset"]["cruxset_id"] == trace["cruxset"]["cruxset_id"]

    def test_agent_receipt_signature_present(self, trace) -> None:
        receipt = trace["agent_receipt"]
        # Post-PR#6095, receipts carry an Ed25519 signature string
        assert "signature" in receipt
        assert receipt["signature"]

    def test_agent_receipt_dissent_captured(self, trace) -> None:
        receipt = trace["agent_receipt"]
        assert len(receipt["dissent"]) == 2


class TestDic17FromCruxSet:
    def test_only_load_bearing_cruxes_produce_proposals(self, trace) -> None:
        proposals = trace["dic17_proposals_from_cruxset"]
        # Top-k=3 with threshold 0.6 → the 2 cruxes scoring >= 0.6 clear
        assert len(proposals) == 2
        # Titles reference the load-bearing cruxes
        titles = " ".join(p["title"] for p in proposals)
        assert "benchmark" in titles.lower() or "soak" in titles.lower()

    def test_proposals_never_carry_boss_ready(self, trace) -> None:
        for proposal in trace["dic17_proposals_from_cruxset"]:
            assert "boss-ready" not in proposal["labels"]

    def test_proposals_carry_epistemic_and_crux_labels(self, trace) -> None:
        for proposal in trace["dic17_proposals_from_cruxset"]:
            assert "epistemic" in proposal["labels"]
            assert "crux" in proposal["labels"]

    def test_proposal_source_keys_are_deterministic(self, trace) -> None:
        # Rerunning the pipeline with the same input must yield same source_keys
        from agt_pipeline_dry_run import run_pipeline as rerun

        rerun_trace = rerun(now=_pinned_now(), tmp_dir=Path("/tmp/agt_e2e_rerun_check"))
        first = [p["source_key"] for p in trace["dic17_proposals_from_cruxset"]]
        second = [p["source_key"] for p in rerun_trace["dic17_proposals_from_cruxset"]]
        assert first == second


class TestAgt04MarketStage:
    def test_market_has_content_addressed_id(self, trace) -> None:
        assert trace["market"]["market_id"].startswith("mkt_pr_merge_")

    def test_resolution_outcome_matches_expected(self, trace) -> None:
        assert trace["resolution"]["outcome"] == "yes"


class TestAgt05SettlementStage:
    def test_claim_bridged_from_market(self, trace) -> None:
        claim = trace["claim"]
        assert claim["domain"] == "prediction_market"
        assert claim["agent_id"] == "alice-predictor"
        assert claim["stake_units"] == 50

    def test_delta_is_positive_when_agent_wins(self, trace) -> None:
        delta = trace["reputation_delta"]
        # probability=0.92, outcome=yes → Brier = 0.0064 → payout = ~49.4
        assert delta["delta"] > 40.0
        assert delta["delta"] < 50.0
        assert delta["scoring_rule"] == "brier_proper"


class TestAgt05AnchorStage:
    def test_anchor_is_dry_run_by_default(self, trace) -> None:
        receipt = trace["anchor_receipt"]
        assert receipt["dry_run"] is True
        assert receipt["tx_hash"] is None
        assert receipt["error"] is None

    def test_anchor_encodes_delta_as_int128(self, trace) -> None:
        receipt = trace["anchor_receipt"]
        delta = trace["reputation_delta"]
        # With default value_decimals=6, value ≈ delta * 1e6
        expected_scaled = int(round(delta["delta"] * 10**6))
        assert abs(receipt["value"] - expected_scaled) <= 1

    def test_anchor_feedback_hash_is_64_hex_chars(self, trace) -> None:
        receipt = trace["anchor_receipt"]
        hex_hash = receipt["feedback_hash_hex"]
        assert len(hex_hash) == 64  # SHA-256 as hex
        int(hex_hash, 16)  # Must parse as hex


class TestFailureBranch:
    def test_losing_claim_produces_negative_delta(self, trace) -> None:
        losing = trace["failure_branch"]["losing_delta"]
        # probability=0.95 on YES, outcome=NO → Brier = 0.9025 → delta ≈ -40.25
        assert losing["delta"] < -10.0

    def test_large_loss_produces_followup_proposal(self, trace) -> None:
        branch = trace["failure_branch"]
        assert branch["dic17_proposal_for_failed_claim"] is not None
        proposal = branch["dic17_proposal_for_failed_claim"]
        assert "boss-ready" not in proposal["labels"]
        assert "failed-claim" in proposal["labels"]


class TestViahStage:
    def test_viah_computed_from_synthetic_ledger(self, trace) -> None:
        viah = trace["viah_report"]
        # 2 merged PRs, 0 rescues, 3 agent-hours → VIAH = 2/3
        assert viah["merged_autonomous_prs"] == 2
        assert viah["rescues_required"] == 0
        assert viah["viah"] is not None
        assert viah["viah"] == pytest.approx(2.0 / 3.0)


class TestFixtureExists:
    def test_committed_fixture_is_current(self, trace) -> None:
        """The committed fixture should match what run_pipeline produces.

        If this test fails after a schema change, regenerate with:
            python3 scripts/agt_pipeline_dry_run.py --pin-timestamp
        and commit the updated JSON.
        """
        fixture_path = REPO_ROOT / "docs" / "status" / "generated" / "agt_e2e_trace.json"
        assert fixture_path.exists(), (
            "committed fixture missing; run python3 scripts/agt_pipeline_dry_run.py --pin-timestamp"
        )
        committed = json.loads(fixture_path.read_text(encoding="utf-8"))
        # Structural equality; precise-float ok because of deterministic inputs
        assert committed == trace


class TestReputationDeltaSerialization:
    def test_reputation_delta_roundtrips(self, trace) -> None:
        delta = ReputationDelta.from_json(trace["reputation_delta"])
        roundtrip = delta.to_json()
        # Round-tripping shouldn't change anything
        for key in ("delta", "scoring_rule", "agent_id", "domain"):
            assert roundtrip[key] == trace["reputation_delta"][key]
