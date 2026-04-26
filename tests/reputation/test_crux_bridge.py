"""Tests for aragora.reputation.crux_bridge — AGT-05 CruxSet resolution bridge.

Verifies:
- CruxPositionRecord content-addressing and validation
- CruxResolutionEvent factory methods and inconclusive sentinel
- bridge_from_crux_position outcome derivation and cross-checks
- Full pipeline: bridge → settle_claim → ReputationDelta (binary scoring)

Note: ``Crux`` is accessed via duck-typing in the bridge (TYPE_CHECKING-only
import).  Tests use a ``SimpleNamespace`` stub carrying the three fields the
bridge reads (``crux_id``, ``statement``, ``load_bearing_score``) to avoid
pulling the full ``aragora.reasoning`` package initialisation chain, which
requires optional heavy dependencies not present in this CI environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aragora.reputation.crux_bridge import (
    CruxPositionRecord,
    CruxResolutionEvent,
    bridge_from_crux_position,
)
from aragora.reputation.settlement import settle_claim
from aragora.reputation.types import DOMAIN_CRUX_RESOLUTION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_crux(
    crux_id: str = "c1",
    statement: str = "X is the load-bearing constraint",
    score: float = 0.75,
) -> SimpleNamespace:
    # Minimal duck-typed stub: bridge only reads crux_id, statement,
    # and load_bearing_score — no need for the full Crux dataclass.
    return SimpleNamespace(
        crux_id=crux_id,
        statement=statement,
        load_bearing_score=score,
    )


def _make_position(
    *,
    agent_id: str = "alice",
    crux_id: str = "c1",
    cruxset_id: str = "cs_abc",
    side: str = "for",
    stake_units: int = 20,
    submitted_at: str = "2026-04-25T08:00:00Z",
) -> CruxPositionRecord:
    return CruxPositionRecord.create(
        agent_id=agent_id,
        crux_id=crux_id,
        cruxset_id=cruxset_id,
        side=side,
        stake_units=stake_units,
        submitted_at=submitted_at,
    )


def _make_resolution(
    winning_side: str,
    *,
    crux_id: str = "c1",
    cruxset_id: str = "cs_abc",
    resolution_source: str = "debate_consensus",
    resolved_at: str = "2026-04-25T12:00:00Z",
) -> CruxResolutionEvent:
    return CruxResolutionEvent.resolved(
        crux_id=crux_id,
        cruxset_id=cruxset_id,
        winning_side=winning_side,
        resolution_source=resolution_source,
        resolved_at=resolved_at,
        evidence={"rounds": 3, "consensus_mode": "crux_finder"},
    )


# ---------------------------------------------------------------------------
# CruxPositionRecord
# ---------------------------------------------------------------------------


class TestCruxPositionRecord:
    def test_create_produces_content_addressed_id(self) -> None:
        p1 = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=10
        )
        p2 = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=10
        )
        assert p1.position_id == p2.position_id
        assert p1.position_id.startswith("cp_")

    def test_different_sides_produce_different_ids(self) -> None:
        p_for = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=10
        )
        p_against = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="against", stake_units=10
        )
        assert p_for.position_id != p_against.position_id

    def test_different_agents_produce_different_ids(self) -> None:
        pa = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=10
        )
        pb = CruxPositionRecord.create(
            agent_id="bob", crux_id="c1", cruxset_id="cs1", side="for", stake_units=10
        )
        assert pa.position_id != pb.position_id

    def test_rejects_zero_stake(self) -> None:
        with pytest.raises(ValueError, match="stake_units must be >= 1"):
            CruxPositionRecord.create(
                agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=0
            )

    def test_rejects_negative_stake(self) -> None:
        with pytest.raises(ValueError, match="stake_units must be >= 1"):
            CruxPositionRecord.create(
                agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=-5
            )

    def test_rejects_empty_agent_id(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            CruxPositionRecord.create(
                agent_id="  ", crux_id="c1", cruxset_id="cs1", side="for", stake_units=5
            )

    def test_rejects_empty_side(self) -> None:
        with pytest.raises(ValueError, match="side"):
            CruxPositionRecord.create(
                agent_id="alice", crux_id="c1", cruxset_id="cs1", side="", stake_units=5
            )

    def test_rejects_empty_crux_id(self) -> None:
        with pytest.raises(ValueError, match="crux_id"):
            CruxPositionRecord.create(
                agent_id="alice", crux_id="", cruxset_id="cs1", side="for", stake_units=5
            )

    def test_provenance_defaults_to_empty_dict(self) -> None:
        p = CruxPositionRecord.create(
            agent_id="alice", crux_id="c1", cruxset_id="cs1", side="for", stake_units=1
        )
        assert p.provenance == {}

    def test_custom_provenance_is_copied(self) -> None:
        prov = {"source": "arena_round_2"}
        p = CruxPositionRecord.create(
            agent_id="alice",
            crux_id="c1",
            cruxset_id="cs1",
            side="for",
            stake_units=1,
            provenance=prov,
        )
        assert p.provenance == prov
        prov["source"] = "mutated"
        assert p.provenance["source"] == "arena_round_2"  # copy not reference


# ---------------------------------------------------------------------------
# CruxResolutionEvent
# ---------------------------------------------------------------------------


class TestCruxResolutionEvent:
    def test_resolved_factory_preserves_fields(self) -> None:
        ev = CruxResolutionEvent.resolved(
            crux_id="c1",
            cruxset_id="cs1",
            winning_side="for",
            resolution_source="debate",
            resolved_at="2026-04-25T10:00:00Z",
            evidence={"k": "v"},
        )
        assert ev.crux_id == "c1"
        assert ev.winning_side == "for"
        assert not ev.is_inconclusive
        assert ev.evidence == {"k": "v"}

    def test_resolved_requires_nonempty_winning_side(self) -> None:
        with pytest.raises(ValueError, match="winning_side must be non-empty"):
            CruxResolutionEvent.resolved(
                crux_id="c1",
                cruxset_id="cs1",
                winning_side="  ",
                resolution_source="debate",
            )

    def test_inconclusive_factory_sets_empty_winning_side(self) -> None:
        ev = CruxResolutionEvent.inconclusive(
            crux_id="c1",
            cruxset_id="cs1",
            resolution_source="oracle",
        )
        assert ev.is_inconclusive
        assert ev.winning_side == ""

    def test_resolved_is_not_inconclusive(self) -> None:
        ev = CruxResolutionEvent.resolved(
            crux_id="c1",
            cruxset_id="cs1",
            winning_side="for",
            resolution_source="debate",
        )
        assert not ev.is_inconclusive

    def test_rejects_empty_crux_id(self) -> None:
        with pytest.raises(ValueError, match="crux_id"):
            CruxResolutionEvent.resolved(
                crux_id="",
                cruxset_id="cs1",
                winning_side="for",
                resolution_source="debate",
            )

    def test_rejects_empty_resolution_source(self) -> None:
        with pytest.raises(ValueError, match="resolution_source"):
            CruxResolutionEvent.resolved(
                crux_id="c1",
                cruxset_id="cs1",
                winning_side="for",
                resolution_source="",
            )


# ---------------------------------------------------------------------------
# bridge_from_crux_position — outcome derivation
# ---------------------------------------------------------------------------


class TestBridgeOutcomeDerivation:
    def test_winner_gets_yes(self) -> None:
        position = _make_position(side="for")
        crux = _make_crux()
        resolution = _make_resolution("for")
        _, resolved = bridge_from_crux_position(position, crux, resolution)
        assert resolved.outcome == "yes"

    def test_loser_gets_no(self) -> None:
        position = _make_position(side="for")
        crux = _make_crux()
        resolution = _make_resolution("against")
        _, resolved = bridge_from_crux_position(position, crux, resolution)
        assert resolved.outcome == "no"

    def test_inconclusive_resolution_yields_inconclusive(self) -> None:
        position = _make_position()
        crux = _make_crux()
        ev = CruxResolutionEvent.inconclusive(
            crux_id="c1",
            cruxset_id="cs_abc",
            resolution_source="oracle",
            resolved_at="2026-04-25T12:00:00Z",
        )
        _, resolved = bridge_from_crux_position(position, crux, ev)
        assert resolved.outcome == "inconclusive"

    def test_against_side_wins(self) -> None:
        position = _make_position(side="against")
        crux = _make_crux()
        resolution = _make_resolution("against")
        _, resolved = bridge_from_crux_position(position, crux, resolution)
        assert resolved.outcome == "yes"

    def test_against_side_loses(self) -> None:
        position = _make_position(side="against")
        crux = _make_crux()
        resolution = _make_resolution("for")
        _, resolved = bridge_from_crux_position(position, crux, resolution)
        assert resolved.outcome == "no"


# ---------------------------------------------------------------------------
# bridge_from_crux_position — claim shape
# ---------------------------------------------------------------------------


class TestBridgeClaimShape:
    def test_domain_is_crux_resolution(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(), _make_crux(), _make_resolution("for")
        )
        assert claim.domain == DOMAIN_CRUX_RESOLUTION

    def test_predicted_probability_is_none(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(), _make_crux(), _make_resolution("for")
        )
        assert claim.predicted_probability is None

    def test_position_is_normalised_to_yes(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(side="for"), _make_crux(), _make_resolution("for")
        )
        assert claim.position == "yes"

    def test_statement_carried_from_crux(self) -> None:
        crux = _make_crux(statement="The deployment window is safe")
        claim, _ = bridge_from_crux_position(_make_position(), crux, _make_resolution("for"))
        assert claim.statement == "The deployment window is safe"

    def test_provenance_contains_agent_side(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(side="for"), _make_crux(), _make_resolution("for")
        )
        assert claim.provenance["agent_side"] == "for"

    def test_provenance_contains_winning_side(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(), _make_crux(), _make_resolution("against")
        )
        assert claim.provenance["winning_side"] == "against"

    def test_provenance_contains_load_bearing_score(self) -> None:
        crux = _make_crux(score=0.88)
        claim, _ = bridge_from_crux_position(_make_position(), crux, _make_resolution("for"))
        assert claim.provenance["load_bearing_score"] == pytest.approx(0.88)

    def test_resolution_id_encodes_cruxset_and_crux(self) -> None:
        claim, _ = bridge_from_crux_position(
            _make_position(cruxset_id="cs_abc", crux_id="c1"),
            _make_crux(crux_id="c1"),
            _make_resolution("for", cruxset_id="cs_abc", crux_id="c1"),
        )
        assert claim.resolution_id == "cs_abc:c1"

    def test_resolved_carries_evidence(self) -> None:
        _, resolved = bridge_from_crux_position(
            _make_position(), _make_crux(), _make_resolution("for")
        )
        assert resolved.evidence == {"rounds": 3, "consensus_mode": "crux_finder"}


# ---------------------------------------------------------------------------
# bridge_from_crux_position — cross-checks
# ---------------------------------------------------------------------------


class TestBridgeCrossChecks:
    def test_rejects_crux_id_mismatch_position_vs_crux(self) -> None:
        position = _make_position(crux_id="c1")
        crux = _make_crux(crux_id="c2")
        resolution = _make_resolution("for", crux_id="c1")
        with pytest.raises(ValueError, match="crux_id"):
            bridge_from_crux_position(position, crux, resolution)

    def test_rejects_crux_id_mismatch_position_vs_resolution(self) -> None:
        position = _make_position(crux_id="c1")
        crux = _make_crux(crux_id="c1")
        resolution = _make_resolution("for", crux_id="c99")
        with pytest.raises(ValueError, match="crux_id"):
            bridge_from_crux_position(position, crux, resolution)

    def test_rejects_cruxset_id_mismatch(self) -> None:
        position = _make_position(cruxset_id="cs_abc")
        crux = _make_crux()
        resolution = _make_resolution("for", cruxset_id="cs_xyz")
        with pytest.raises(ValueError, match="cruxset_id"):
            bridge_from_crux_position(position, crux, resolution)


# ---------------------------------------------------------------------------
# End-to-end settlement pipeline
# ---------------------------------------------------------------------------


class TestSettlementPipeline:
    def test_winner_gains_stake(self) -> None:
        position = _make_position(side="for", stake_units=30)
        crux = _make_crux()
        resolution = _make_resolution("for")
        claim, resolved = bridge_from_crux_position(position, crux, resolution)
        delta = settle_claim(claim, resolved, scoring_rule="binary")
        assert delta.delta == pytest.approx(30.0)
        assert delta.domain == DOMAIN_CRUX_RESOLUTION
        assert delta.agent_id == "alice"

    def test_loser_loses_stake(self) -> None:
        position = _make_position(side="for", stake_units=30)
        crux = _make_crux()
        resolution = _make_resolution("against")
        claim, resolved = bridge_from_crux_position(position, crux, resolution)
        delta = settle_claim(claim, resolved, scoring_rule="binary")
        assert delta.delta == pytest.approx(-30.0)

    def test_inconclusive_yields_zero_delta(self) -> None:
        position = _make_position(side="for", stake_units=50)
        crux = _make_crux()
        ev = CruxResolutionEvent.inconclusive(
            crux_id="c1",
            cruxset_id="cs_abc",
            resolution_source="oracle",
            resolved_at="2026-04-25T12:00:00Z",
        )
        claim, resolved = bridge_from_crux_position(position, crux, ev)
        delta = settle_claim(claim, resolved, scoring_rule="binary")
        assert delta.delta == pytest.approx(0.0)

    def test_two_agents_on_opposing_sides(self) -> None:
        crux = _make_crux()
        resolution = _make_resolution("for")

        pos_alice = _make_position(agent_id="alice", side="for", stake_units=40)
        pos_bob = _make_position(agent_id="bob", side="against", stake_units=40)

        claim_alice, resolved_alice = bridge_from_crux_position(pos_alice, crux, resolution)
        claim_bob, resolved_bob = bridge_from_crux_position(pos_bob, crux, resolution)

        delta_alice = settle_claim(claim_alice, resolved_alice, scoring_rule="binary")
        delta_bob = settle_claim(claim_bob, resolved_bob, scoring_rule="binary")

        assert delta_alice.delta == pytest.approx(40.0)
        assert delta_bob.delta == pytest.approx(-40.0)

    def test_claim_ids_differ_across_agents(self) -> None:
        crux = _make_crux()
        resolution = _make_resolution("for")
        pos_alice = _make_position(agent_id="alice", side="for")
        pos_bob = _make_position(agent_id="bob", side="against")
        claim_alice, _ = bridge_from_crux_position(pos_alice, crux, resolution)
        claim_bob, _ = bridge_from_crux_position(pos_bob, crux, resolution)
        assert claim_alice.claim_id != claim_bob.claim_id
