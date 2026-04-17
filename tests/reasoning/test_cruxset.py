"""Tests for the AGT-01 CruxSet contract."""

from __future__ import annotations

import pytest

from aragora.reasoning.cruxset import (
    CRUXSET_SCHEMA_VERSION,
    Crux,
    CruxPosition,
    CruxSet,
    build_cruxset_from_analysis,
)


def _make_crux(
    *,
    crux_id: str = "claim_1",
    statement: str = "X is true",
    score: float = 0.7,
) -> Crux:
    return Crux(
        crux_id=crux_id,
        statement=statement,
        positions=(
            CruxPosition(side="for", agents=("alice",)),
            CruxPosition(side="against", agents=("bob",)),
        ),
        load_bearing_score=score,
        evidence_gaps=("missing-data",),
        counterfactual="if X is false, decision flips",
        candidate_verifier="docs/spec.md",
    )


class TestCruxValidation:
    def test_crux_requires_score_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            Crux(
                crux_id="c1",
                statement="x",
                positions=(CruxPosition(side="for"),),
                load_bearing_score=1.1,
            )
        with pytest.raises(ValueError):
            Crux(
                crux_id="c1",
                statement="x",
                positions=(CruxPosition(side="for"),),
                load_bearing_score=-0.1,
            )

    def test_crux_requires_non_empty_id(self) -> None:
        with pytest.raises(ValueError):
            Crux(
                crux_id="",
                statement="x",
                positions=(),
                load_bearing_score=0.5,
            )

    def test_crux_requires_non_empty_statement(self) -> None:
        with pytest.raises(ValueError):
            Crux(
                crux_id="c1",
                statement=" ",
                positions=(),
                load_bearing_score=0.5,
            )


class TestCruxSetBuild:
    def test_build_rejects_empty_question(self) -> None:
        with pytest.raises(ValueError):
            CruxSet.build(question="   ", cruxes=[_make_crux()])

    def test_build_requires_at_least_one_crux(self) -> None:
        with pytest.raises(ValueError):
            CruxSet.build(question="should we ship?", cruxes=[])

    def test_build_orders_cruxes_by_load_bearing_score_desc(self) -> None:
        cruxes = [
            _make_crux(crux_id="c1", score=0.3),
            _make_crux(crux_id="c2", score=0.9),
            _make_crux(crux_id="c3", score=0.7),
        ]
        cs = CruxSet.build(question="q", cruxes=cruxes)
        assert [c.crux_id for c in cs.cruxes] == ["c2", "c3", "c1"]

    def test_build_breaks_score_ties_by_crux_id_asc(self) -> None:
        cruxes = [
            _make_crux(crux_id="c2", score=0.5),
            _make_crux(crux_id="c1", score=0.5),
        ]
        cs = CruxSet.build(question="q", cruxes=cruxes)
        assert [c.crux_id for c in cs.cruxes] == ["c1", "c2"]

    def test_cruxset_id_is_content_addressed(self) -> None:
        cruxes = [_make_crux(crux_id="c1", score=0.5)]
        a = CruxSet.build(question="should we ship?", cruxes=cruxes)
        b = CruxSet.build(
            question="should we ship?",
            cruxes=[_make_crux(crux_id="c1", score=0.5)],
        )
        assert a.cruxset_id == b.cruxset_id

    def test_distinct_questions_yield_distinct_ids(self) -> None:
        cruxes = [_make_crux(crux_id="c1", score=0.5)]
        a = CruxSet.build(question="ship now?", cruxes=cruxes)
        b = CruxSet.build(
            question="ship later?",
            cruxes=[_make_crux(crux_id="c1", score=0.5)],
        )
        assert a.cruxset_id != b.cruxset_id

    def test_schema_version_recorded(self) -> None:
        cs = CruxSet.build(question="q", cruxes=[_make_crux()])
        assert cs.schema_version == CRUXSET_SCHEMA_VERSION


class TestCruxSetChecksum:
    def test_checksum_is_stable_across_builds(self) -> None:
        cruxes = [_make_crux(crux_id="c1", score=0.5)]
        a = CruxSet.build(
            question="q",
            cruxes=cruxes,
            created_at="2026-04-17T00:00:00Z",
            provenance={"debate_id": "d1"},
        )
        b = CruxSet.build(
            question="q",
            cruxes=[_make_crux(crux_id="c1", score=0.5)],
            created_at="2026-04-17T00:00:00Z",
            provenance={"debate_id": "d1"},
        )
        assert a.checksum == b.checksum

    def test_checksum_changes_with_decision(self) -> None:
        cruxes = [_make_crux(crux_id="c1", score=0.5)]
        a = CruxSet.build(
            question="q",
            cruxes=cruxes,
            decision="ship",
            created_at="2026-04-17T00:00:00Z",
        )
        b = CruxSet.build(
            question="q",
            cruxes=[_make_crux(crux_id="c1", score=0.5)],
            decision="hold",
            created_at="2026-04-17T00:00:00Z",
        )
        assert a.checksum != b.checksum

    def test_verify_checksum_succeeds_on_valid_set(self) -> None:
        cs = CruxSet.build(question="q", cruxes=[_make_crux()])
        assert cs.verify_checksum() is True

    def test_verify_checksum_fails_when_question_mutated(self) -> None:
        cs = CruxSet.build(question="q", cruxes=[_make_crux()])
        tampered = CruxSet(
            cruxset_id=cs.cruxset_id,
            schema_version=cs.schema_version,
            question="DIFFERENT",
            decision=cs.decision,
            cruxes=cs.cruxes,
            evidence_gaps=cs.evidence_gaps,
            counterfactual_notes=cs.counterfactual_notes,
            verifier_candidates=cs.verifier_candidates,
            receipt_id=cs.receipt_id,
            provenance=cs.provenance,
            created_at=cs.created_at,
            checksum=cs.checksum,
        )
        assert tampered.verify_checksum() is False


class TestCruxSetJson:
    def test_to_json_roundtrip_preserves_checksum(self) -> None:
        cs = CruxSet.build(
            question="should we ship?",
            cruxes=[_make_crux(crux_id="c1"), _make_crux(crux_id="c2", score=0.4)],
            decision="ship",
            evidence_gaps=("benchmark gap",),
            counterfactual_notes=("note A",),
            verifier_candidates=("docs/spec.md",),
            receipt_id="rcpt_x",
            provenance={"debate_id": "d1"},
        )
        roundtrip = CruxSet.from_json(cs.to_json())
        assert roundtrip == cs
        assert roundtrip.verify_checksum()

    def test_position_to_from_json(self) -> None:
        position = CruxPosition(side="for", agents=("alice", "carol"), rationale="x")
        roundtrip = CruxPosition.from_json(position.to_json())
        assert roundtrip == position


class TestBuildFromAnalysis:
    def test_builds_from_crux_analysis_payload(self) -> None:
        payload = {
            "cruxes": [
                {
                    "claim_id": "claim_1",
                    "statement": "X is load-bearing",
                    "author": "alice",
                    "crux_score": 0.85,
                    "influence_score": 0.7,
                    "disagreement_score": 0.9,
                    "uncertainty_score": 0.6,
                    "centrality_score": 0.5,
                    "affected_claims": ["claim_2"],
                    "contesting_agents": ["bob"],
                    "resolution_impact": 0.4,
                },
                {
                    "claim_id": "claim_2",
                    "statement": "Y is contingent on X",
                    "author": "carol",
                    "crux_score": 0.55,
                    "influence_score": 0.4,
                    "disagreement_score": 0.5,
                    "uncertainty_score": 0.5,
                    "centrality_score": 0.3,
                    "affected_claims": [],
                    "contesting_agents": [],
                    "resolution_impact": 0.2,
                },
            ],
            "total_claims": 5,
            "total_disagreements": 2,
            "average_uncertainty": 0.65,
            "convergence_barrier": 0.42,
            "recommended_focus": ["claim_1", "claim_2"],
        }
        cs = build_cruxset_from_analysis(
            question="should X be assumed?",
            analysis_payload=payload,
            decision="hold",
            receipt_id="rcpt_y",
            provenance={"debate_id": "d2"},
        )
        assert len(cs.cruxes) == 2
        # Higher crux_score sorts first
        assert cs.cruxes[0].crux_id == "claim_1"
        assert cs.cruxes[0].load_bearing_score == pytest.approx(0.85)
        # Contesting agents become an "against" position
        sides = {p.side for p in cs.cruxes[0].positions}
        assert sides == {"for", "against"}
        # recommended_focus becomes verifier_candidates
        assert "claim_1" in cs.verifier_candidates
        # Aggregate counterfactual notes carry uncertainty + barrier
        joined = "|".join(cs.counterfactual_notes)
        assert "avg_uncertainty=0.65" in joined
        assert "convergence_barrier=0.42" in joined

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(ValueError):
            build_cruxset_from_analysis(question="q", analysis_payload={"cruxes": []})

    def test_max_cruxes_caps_output(self) -> None:
        payload = {
            "cruxes": [
                {
                    "claim_id": f"claim_{i}",
                    "statement": f"S{i}",
                    "author": "alice",
                    "crux_score": 1.0 - 0.01 * i,
                    "contesting_agents": [],
                    "resolution_impact": 0.1,
                }
                for i in range(10)
            ],
            "average_uncertainty": 0.5,
            "convergence_barrier": 0.3,
            "recommended_focus": [],
        }
        cs = build_cruxset_from_analysis(question="q", analysis_payload=payload, max_cruxes=3)
        assert len(cs.cruxes) == 3
