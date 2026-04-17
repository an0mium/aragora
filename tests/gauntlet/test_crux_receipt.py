"""Tests for the CruxReceipt export surface (Crux A2 / #6036).

The receipt is the signed, exportable artifact for crux-finder debates —
explicitly distinct from `DecisionReceipt` because the deliverable is
*not* a verdict. Mirrors the SHA-256 signing pattern used by
`ConsensusProof.checksum` in `aragora.debate.consensus`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aragora.debate.crux_mode import CruxFinderResult
from aragora.gauntlet.receipt import (
    CruxReceipt,
    build_crux_receipt,
    crux_receipt_to_markdown,
)
from aragora.reasoning.crux_detector import CruxAnalysisResult, CruxClaim


def _sample_analysis() -> CruxAnalysisResult:
    return CruxAnalysisResult(
        cruxes=[
            CruxClaim(
                claim_id="c1",
                statement="Are the assumptions sound?",
                author="agent-alpha",
                crux_score=0.82,
                influence_score=0.7,
                disagreement_score=0.6,
                uncertainty_score=0.5,
                centrality_score=0.8,
                affected_claims=["c2", "c3"],
                contesting_agents=["agent-alpha", "agent-beta"],
                resolution_impact=0.4,
            ),
            CruxClaim(
                claim_id="c4",
                statement="Does the cost estimate hold?",
                author="agent-beta",
                crux_score=0.64,
                influence_score=0.5,
                disagreement_score=0.5,
                uncertainty_score=0.6,
                centrality_score=0.4,
                affected_claims=[],
                contesting_agents=["agent-beta"],
                resolution_impact=0.25,
            ),
        ],
        total_claims=4,
        total_disagreements=2,
        average_uncertainty=0.55,
        convergence_barrier=0.62,
        recommended_focus=["c1", "c4"],
    )


def _sample_result() -> CruxFinderResult:
    return CruxFinderResult(
        debate_id="debate-abc",
        question="Should we ship feature X?",
        analysis=_sample_analysis(),
        counterfactuals=[
            {
                "claim_id": "c1",
                "condition": "Resolve 'Are the assumptions sound?' to high confidence",
                "outcome_change": "Reduces total network uncertainty by 0.400",
                "likelihood": 0.5,
                "affected_claims": ["c2", "c3"],
            },
            {
                "claim_id": "c4",
                "condition": "Resolve 'Does the cost estimate hold?' to high confidence",
                "outcome_change": "Reduces total network uncertainty by 0.250",
                "likelihood": 0.6,
                "affected_claims": [],
            },
        ],
        agents=["agent-alpha", "agent-beta"],
        rounds=3,
        raw_claims=[{"claim_id": "c1"}, {"claim_id": "c4"}],
        metadata={"mode": "crux_finder", "approach": "A"},
    )


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


def test_crux_receipt_import_surface() -> None:
    """The receipt and builders must be importable from both canonical paths."""
    from aragora.gauntlet import receipt as receipt_facade
    from aragora.gauntlet import receipt_models

    assert receipt_facade.CruxReceipt is receipt_models.CruxReceipt
    assert receipt_facade.build_crux_receipt is receipt_models.build_crux_receipt
    assert hasattr(receipt_facade, "crux_receipt_to_markdown")


def test_crux_receipt_to_dict_includes_checksum() -> None:
    receipt = build_crux_receipt(_sample_result())
    payload = receipt.to_dict()
    assert payload["receipt_id"].startswith("crux-")
    assert payload["question"] == "Should we ship feature X?"
    assert payload["convergence_barrier"] == pytest.approx(0.62)
    assert payload["checksum"] == receipt.checksum
    assert isinstance(payload["cruxes"], list) and len(payload["cruxes"]) == 2


# ---------------------------------------------------------------------------
# Checksum semantics
# ---------------------------------------------------------------------------


def test_crux_receipt_checksum_stable() -> None:
    """Same inputs → same checksum (two independent builds)."""
    r1 = build_crux_receipt(_sample_result())
    # Rebuild an *identical* receipt — same receipt_id, same timestamp — to
    # isolate the checksum from the random receipt_id / timestamp fields.
    r2 = CruxReceipt(
        receipt_id=r1.receipt_id,
        debate_id=r1.debate_id,
        question=r1.question,
        timestamp=r1.timestamp,
        agents=list(r1.agents),
        rounds=r1.rounds,
        cruxes=[dict(c) for c in r1.cruxes],
        convergence_barrier=r1.convergence_barrier,
        counterfactuals=[dict(c) for c in r1.counterfactuals],
        recommended_focus=list(r1.recommended_focus),
        resolution_strategies=list(r1.resolution_strategies),
        raw_claims_hash=r1.raw_claims_hash,
        metadata=dict(r1.metadata),
    )
    assert r1.checksum == r2.checksum
    assert isinstance(r1.checksum, str) and len(r1.checksum) == 16


def test_crux_receipt_checksum_changes_on_mutation() -> None:
    r1 = build_crux_receipt(_sample_result())
    # Mutate the top crux's score and confirm the checksum flips.
    r2 = CruxReceipt(
        receipt_id=r1.receipt_id,
        debate_id=r1.debate_id,
        question=r1.question,
        timestamp=r1.timestamp,
        agents=list(r1.agents),
        rounds=r1.rounds,
        cruxes=[{**r1.cruxes[0], "crux_score": 0.99}, *r1.cruxes[1:]],
        convergence_barrier=r1.convergence_barrier,
        counterfactuals=list(r1.counterfactuals),
        recommended_focus=list(r1.recommended_focus),
        resolution_strategies=list(r1.resolution_strategies),
        raw_claims_hash=r1.raw_claims_hash,
        metadata=dict(r1.metadata),
    )
    assert r1.checksum != r2.checksum


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_crux_receipt_from_result_carries_all_fields() -> None:
    result = _sample_result()
    receipt = build_crux_receipt(result)

    assert receipt.debate_id == "debate-abc"
    assert receipt.question == "Should we ship feature X?"
    assert receipt.agents == ["agent-alpha", "agent-beta"]
    assert receipt.rounds == 3
    assert receipt.convergence_barrier == pytest.approx(0.62)
    assert receipt.recommended_focus == ["c1", "c4"]
    assert len(receipt.cruxes) == 2
    assert len(receipt.counterfactuals) == 2

    # Raw claims provenance anchor — SHA-256 over canonical JSON.
    expected_hash = hashlib.sha256(
        json.dumps(result.raw_claims, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert receipt.raw_claims_hash == expected_hash


def test_build_crux_receipt_accepts_resolution_strategies() -> None:
    strategies = [
        {
            "claim_id": "c1",
            "strategy": "Commission targeted study",
            "likely_impact": "Reduces uncertainty by 0.3",
        }
    ]
    receipt = build_crux_receipt(
        _sample_result(),
        resolution_strategies=strategies,
    )
    assert receipt.resolution_strategies == strategies


# ---------------------------------------------------------------------------
# Markdown exporter
# ---------------------------------------------------------------------------


def test_crux_receipt_markdown_headline_is_not_a_decision() -> None:
    """The exporter must NOT render \"Decision\" — the deliverable is a crux
    map, not a verdict. This guards against accidental template reuse.
    """
    receipt = build_crux_receipt(_sample_result())
    rendered = crux_receipt_to_markdown(receipt)
    first_line = rendered.splitlines()[0]
    assert first_line.startswith("# Crux Map — ")
    assert "Decision" not in rendered.split("\n", 1)[0]


def test_crux_receipt_markdown_contains_all_cruxes() -> None:
    receipt = build_crux_receipt(_sample_result())
    rendered = crux_receipt_to_markdown(receipt)
    for crux in receipt.cruxes:
        assert crux["statement"] in rendered, (
            f"crux {crux['claim_id']!r} missing from rendered markdown"
        )
    # Checksum + receipt id must appear for audit traceability.
    assert receipt.checksum in rendered
    assert receipt.receipt_id in rendered
    # Convergence barrier is rendered (rounded to 3 decimals).
    assert f"{receipt.convergence_barrier:.3f}" in rendered
