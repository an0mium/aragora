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


# =============================================================================
# Phase F additions: explicit unit-to-unit dependency edges + multi-hop
# impact propagation (Round 2026-04-30b).
# =============================================================================


class TestDependencyEdgeConstruction:
    """Constructor validates and indexes ``dependency_edges``."""

    def test_default_no_edges(self) -> None:
        g = ProofUnitConstraintGraph([_unit("a"), _unit("b")])
        assert g.edge_count == 0
        assert g.direct_dependencies("a") == []
        assert g.direct_dependents("b") == []

    def test_single_edge_indexes_both_directions(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a"), _unit("b")],
            dependency_edges=[("a", "b")],
        )
        assert g.edge_count == 1
        assert g.direct_dependencies("a") == ["b"]
        assert g.direct_dependents("b") == ["a"]
        # Reverse direction empty.
        assert g.direct_dependencies("b") == []
        assert g.direct_dependents("a") == []

    def test_unknown_from_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="dependency edge from unknown unit"):
            ProofUnitConstraintGraph(
                [_unit("a")],
                dependency_edges=[("ghost", "a")],
            )

    def test_unknown_to_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="dependency edge to unknown unit"):
            ProofUnitConstraintGraph(
                [_unit("a")],
                dependency_edges=[("a", "ghost")],
            )

    def test_self_loop_raises(self) -> None:
        with pytest.raises(ValueError, match="self-loop"):
            ProofUnitConstraintGraph(
                [_unit("a")],
                dependency_edges=[("a", "a")],
            )

    def test_duplicate_edges_idempotent(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a"), _unit("b")],
            dependency_edges=[("a", "b"), ("a", "b")],  # duplicate
        )
        # Set semantics: duplicates collapse to one edge.
        assert g.edge_count == 1
        assert g.direct_dependencies("a") == ["b"]


class TestMultiHopImpact:
    """``multi_hop_impact_set`` walks reverse-dependency BFS."""

    def test_no_edges_equals_single_hop(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b", claims=["y"])],
        )
        assert g.multi_hop_impact_set({"x"}) == g.impact_set({"x"}) == {"a"}

    def test_one_hop_propagation(self) -> None:
        # B depends on A. If A's claim is invalidated, B is also impacted.
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b")],
            dependency_edges=[("b", "a")],
        )
        assert g.impact_set({"x"}) == {"a"}
        assert g.multi_hop_impact_set({"x"}) == {"a", "b"}

    def test_chain_propagates_transitively(self) -> None:
        # C → B → A. Invalidating A's claim cascades to B and C.
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["root"]), _unit("b"), _unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        assert g.multi_hop_impact_set({"root"}) == {"a", "b", "c"}

    def test_max_depth_zero_equals_seed(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b"), _unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        assert g.multi_hop_impact_set({"x"}, max_depth=0) == {"a"}

    def test_max_depth_one_limits_propagation(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b"), _unit("c")],
            dependency_edges=[("b", "a"), ("c", "b")],
        )
        # Depth 1: only first-degree dependent (b), not c.
        assert g.multi_hop_impact_set({"x"}, max_depth=1) == {"a", "b"}

    def test_diamond_dependencies_visited_once(self) -> None:
        # B and C both depend on A; D depends on both B and C.
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b"), _unit("c"), _unit("d")],
            dependency_edges=[("b", "a"), ("c", "a"), ("d", "b"), ("d", "c")],
        )
        result = g.multi_hop_impact_set({"x"})
        assert result == {"a", "b", "c", "d"}

    def test_cycle_in_dependency_graph_is_safe(self) -> None:
        # A is impacted by claim, B and C depend on A, and there's a cycle B↔C.
        # BFS must not loop forever.
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b"), _unit("c")],
            dependency_edges=[("b", "a"), ("c", "b"), ("b", "c")],
        )
        result = g.multi_hop_impact_set({"x"})
        assert result == {"a", "b", "c"}

    def test_unrelated_units_not_impacted(self) -> None:
        # D is in the graph but neither claims X nor depends on anything that does.
        g = ProofUnitConstraintGraph(
            [_unit("a", claims=["x"]), _unit("b"), _unit("c"), _unit("d")],
            dependency_edges=[("b", "a")],
        )
        result = g.multi_hop_impact_set({"x"})
        assert result == {"a", "b"}
        assert "c" not in result
        assert "d" not in result

    def test_empty_seed_returns_empty(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a"), _unit("b")],
            dependency_edges=[("b", "a")],
        )
        assert g.multi_hop_impact_set(set()) == set()
        assert g.multi_hop_impact_set({"unknown"}) == set()


class TestToDictWithEdges:
    """``to_dict`` serialises dependency edges deterministically."""

    def test_edges_sorted(self) -> None:
        g = ProofUnitConstraintGraph(
            [_unit("a"), _unit("b"), _unit("c")],
            dependency_edges=[("c", "a"), ("b", "a"), ("c", "b")],
        )
        d = g.to_dict()
        assert d["edge_count"] == 3
        # Sorted by from_uid then to_uid.
        assert d["dependency_edges"] == [["b", "a"], ["c", "a"], ["c", "b"]]

    def test_edges_empty_by_default(self) -> None:
        g = ProofUnitConstraintGraph([_unit("a")])
        d = g.to_dict()
        assert d["edge_count"] == 0
        assert d["dependency_edges"] == []
