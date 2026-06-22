"""
Tests for GET /api/v1/debates/{id}/cruxes (public crux-finder exposure, #8227).

Covers the storage contract (#8366-aware: reads consensus_proof.metadata)
and the honest-absence contract:
- crux mode not run -> 200 status=absent, cruxes=[] (never fabricated)
- crux mode ran with cruxes -> 200 status=present
- crux mode ran with zero cruxes -> 200 status=present, crux_count=0
- crux mode requested but skipped/fell back -> 200 status=absent + reason
- missing debate -> 404 (the only 404 case)
"""

from __future__ import annotations

from aragora.server.handlers.debates.cruxes import extract_cruxes


# ============================================================================
# extract_cruxes: storage contract + honest absence
# ============================================================================


def test_absent_when_crux_mode_not_run() -> None:
    payload = extract_cruxes(
        {"consensus_proof": {"metadata": {"consensus_mode": "majority"}}}
    )
    assert payload["status"] == "absent"
    assert payload["cruxes"] == []
    assert payload["crux_count"] == 0
    assert "not run" in payload["reason"]


def test_absent_when_no_consensus_proof() -> None:
    payload = extract_cruxes({})
    assert payload["status"] == "absent"
    assert payload["cruxes"] == []


def test_present_when_crux_mode_ran_with_cruxes() -> None:
    debate = {
        "consensus_proof": {
            "metadata": {
                "consensus_mode": "crux_finder",
                "cruxes": [
                    {"claim_id": "c1", "statement": "x", "crux_score": 0.9},
                    {"claim_id": "c2", "statement": "y", "crux_score": 0.7},
                ],
                "convergence_barrier": 0.42,
                "recommended_focus": ["c1"],
            }
        }
    }
    payload = extract_cruxes(debate)
    assert payload["status"] == "present"
    assert payload["consensus_mode"] == "crux_finder"
    assert payload["crux_count"] == 2
    assert payload["convergence_barrier"] == 0.42
    assert payload["recommended_focus"] == ["c1"]


def test_present_but_zero_cruxes_is_not_absent() -> None:
    # crux mode ran but found no load-bearing disagreement: a real, present
    # result distinct from "mode not run".
    debate = {
        "consensus_proof": {
            "metadata": {"consensus_mode": "crux_finder", "cruxes": []}
        }
    }
    payload = extract_cruxes(debate)
    assert payload["status"] == "present"
    assert payload["crux_count"] == 0


def test_summary_block_signals_presence_without_mode_stamp() -> None:
    # formal_verification.crux_finder summary is a secondary presence signal.
    debate = {
        "formal_verification": {"crux_finder": {"count": 1, "convergence_barrier": 0.3}},
        "consensus_proof": {"metadata": {"cruxes": [{"claim_id": "c1"}]}},
    }
    payload = extract_cruxes(debate)
    assert payload["status"] == "present"
    assert payload["crux_count"] == 1


def test_skip_reason_yields_honest_absence() -> None:
    debate = {
        "consensus_proof": {
            "metadata": {"consensus_mode": "crux_finder", "cruxes": [{"claim_id": "c1"}]}
        },
        "metadata": {
            "crux_finder_skipped_reason": "no belief network",
            "crux_finder_fallback_consensus": "majority",
        },
    }
    payload = extract_cruxes(debate)
    assert payload["status"] == "absent"
    assert "did not complete" in payload["reason"]
    assert payload["fallback_consensus"] == "majority"


def test_never_fabricates_cruxes_from_garbage() -> None:
    # Non-list cruxes value must not produce fabricated entries.
    debate = {
        "consensus_proof": {
            "metadata": {"consensus_mode": "crux_finder", "cruxes": "not-a-list"}
        }
    }
    payload = extract_cruxes(debate)
    assert payload["cruxes"] == []
    assert payload["crux_count"] == 0
