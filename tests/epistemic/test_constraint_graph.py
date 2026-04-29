"""Tests for ProofUnitConstraintGraph (DIC-19 / #6030).

Pure schema/logic — no network, no subprocess, no queue mutation.
"""

from __future__ import annotations

import json

import pytest

from aragora.epistemic.constraint_graph import ProofUnitConstraintGraph
from aragora.epistemic.proof_unit_model import (
    DecayPolicy,
    FallbackPolicy,
    ProofCarryingCodeUnit,
)


def _unit(
    uid: str,
    claims: list[str] | None = None,
    receipts: list[str] | None = None,
    crux_ids: list[str] | None = None,
) -> ProofCarryingCodeUnit:
    return ProofCarryingCodeUnit(
        code_unit_id=uid,
        symbol=f"aragora.epistemic.{uid}",
        source_path=f"aragora/{uid}.py",
        owner="test",
        decision_receipts=receipts or [],
        claims=claims or [],
        assumptions=[],
        verifiers=[],
        freshness_sla_hours=24,
        decay_policy=DecayPolicy(),
        fallback_policy=FallbackPolicy(),
        linked_crux_ids=crux_ids or [],
    )


class TestBuild:
    def test_empty_graph(self) -> None:
        g = ProofUnitConstraintGraph([])
        assert g.unit_count == 0
        assert g.claim_count == 0

    def test_single_unit_indexed_by_claim(self) -> None:
        u = _unit("alpha", claims=["claim.x"])
        g = ProofUnitConstraintGraph([u])
        assert g.unit_count == 1
        assert g.units_by_claim("claim.x") == [u]

    def test_duplicate_unit_id_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate code_unit_id"):
            ProofUnitConstraintGraph([_unit("alpha"), _unit("alpha")])

    def test_generator_input(self) -> None:
        g = ProofUnitConstraintGraph(_unit(f"u{i}") for i in range(3))
        assert g.unit_count == 3


class TestUnitsByClaim:
    def test_shared_claim_returns_all_owners(self) -> None:
        u1 = _unit("alpha", claims=["shared"])
        u2 = _unit("beta", claims=["shared"])
        g = ProofUnitConstraintGraph([u1, u2])
        assert {u.code_unit_id for u in g.units_by_claim("shared")} == {"alpha", "beta"}

    def test_unknown_claim_empty(self) -> None:
        g = ProofUnitConstraintGraph([_unit("alpha", claims=["x"])])
        assert g.units_by_claim("missing") == []

    def test_result_deterministically_sorted(self) -> None:
        units = [_unit(f"unit{x}", claims=["shared"]) for x in ["zzz", "aaa", "mmm"]]
        g = ProofUnitConstraintGraph(units)
        ids = [u.code_unit_id for u in g.units_by_claim("shared")]
        assert ids == sorted(ids)

    def test_unit_with_many_claims_indexed_under_each(self) -> None:
        u = _unit("alpha", claims=["a", "b", "c"])
        g = ProofUnitConstraintGraph([u])
        for c in ["a", "b", "c"]:
            assert g.units_by_claim(c) == [u]


class TestUnitsByReceiptAndCrux:
    def test_by_receipt(self) -> None:
        u = _unit("alpha", receipts=["rcpt.001"])
        g = ProofUnitConstraintGraph([u])
        assert (
            g.units_by_receipt("rcpt.001") == [u]
            and g.units_by_receipt("missing") == []
            and g.receipt_count == 1
        )

    def test_by_crux(self) -> None:
        u = _unit("alpha", crux_ids=["crux.42"])
        g = ProofUnitConstraintGraph([u])
        assert (
            g.units_by_crux("crux.42") == [u]
            and g.units_by_crux("missing") == []
            and g.crux_count == 1
        )

    def test_no_crux_ids_gives_zero_crux_count(self) -> None:
        assert ProofUnitConstraintGraph([_unit("alpha", claims=["c"])]).crux_count == 0


class TestImpactSet:
    def test_single_claim_impacts_direct_owners_only(self) -> None:
        u1 = _unit("alpha", claims=["a"])
        u2 = _unit("beta", claims=["b"])
        g = ProofUnitConstraintGraph([u1, u2])
        assert g.impact_set({"a"}) == {"alpha"}

    def test_multiple_claims_union(self) -> None:
        g = ProofUnitConstraintGraph(
            [
                _unit("alpha", claims=["a"]),
                _unit("beta", claims=["b"]),
                _unit("gamma", claims=["a", "b"]),
            ]
        )
        assert g.impact_set({"a", "b"}) == {"alpha", "beta", "gamma"}

    def test_empty_input_returns_empty(self) -> None:
        g = ProofUnitConstraintGraph([_unit("alpha", claims=["a"])])
        assert g.impact_set(set()) == set()
        assert g.impact_set([]) == set()

    def test_unknown_claims_are_ignored(self) -> None:
        g = ProofUnitConstraintGraph([])
        assert g.impact_set({"nonexistent"}) == set()


class TestToDict:
    def test_structure_and_json_round_trip(self) -> None:
        u = _unit("alpha", claims=["c.x"], receipts=["r.1"], crux_ids=["crux.9"])
        g = ProofUnitConstraintGraph([u])
        d = g.to_dict()
        assert d["unit_count"] == 1
        assert d["claim_count"] == 1
        assert d["receipt_count"] == 1
        assert d["crux_count"] == 1
        assert d["claim_index"]["c.x"] == ["alpha"]
        assert d["receipt_index"]["r.1"] == ["alpha"]
        assert d["crux_index"]["crux.9"] == ["alpha"]
        # Must be JSON-serializable
        reloaded = json.loads(json.dumps(d))
        assert reloaded["unit_count"] == 1

    def test_empty_graph_gives_empty_dicts(self) -> None:
        d = ProofUnitConstraintGraph([]).to_dict()
        assert d["units"] == {} and d["claim_index"] == {}

    def test_units_and_claim_index_are_sorted(self) -> None:
        units = [_unit(f"unit{x}", claims=["shared"]) for x in ["zzz", "aaa"]]
        g = ProofUnitConstraintGraph(units)
        d = g.to_dict()
        assert list(d["units"].keys()) == sorted(d["units"].keys())
        assert d["claim_index"]["shared"] == sorted(d["claim_index"]["shared"])
