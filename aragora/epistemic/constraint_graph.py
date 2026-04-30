"""Proof-Carrying Code Unit constraint graph (DIC-19 / #6030).

Builds a read-only dependency graph from a collection of
:class:`aragora.epistemic.proof_unit_model.ProofCarryingCodeUnit` instances.

The graph indexes each unit by its shared ``claims``, ``decision_receipts``,
and ``linked_crux_ids`` so callers can answer in O(1):

- "Which units depend on claim X?"
- "Which units are tied to receipt R?"
- "Which units reference crux C?"
- "What is the full impact set if these claims become invalid?"

Optional explicit unit-to-unit dependency edges (Round 2026-04-30b Phase F)
unlock multi-hop impact propagation: when unit A's verification depends on
unit B's proof being intact, an invalidation that hits B's claims also
surfaces A in the impact set.  Edges are explicit (no implicit inference
from claim/receipt/crux names) so callers retain full control over the
dependency model.

Construction is explicit (``ProofUnitConstraintGraph(units)``); nothing in this
module runs automatically.  The dataclass is always importable; the scanner that
populates the unit list is gated by ``ARAGORA_PROOF_UNIT_SCAN_ENABLED``.

Out of scope (deferred):
- Mutating units or triggering quarantine/repair — see DIC-21/DIC-22.
- Persisting the graph — callers serialize via :meth:`to_dict`.
- Inferring unit-to-unit edges from claim/receipt overlap.  Caller passes
  ``dependency_edges`` explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Collection, Iterable

from .proof_unit_model import ProofCarryingCodeUnit


class ProofUnitConstraintGraph:
    """Cross-reference index over a set of ProofCarryingCodeUnits.

    After construction, the graph is immutable: the underlying dicts are
    never modified. Call :meth:`to_dict` to obtain a JSON-serializable
    snapshot for operator dashboards or debugging.

    Parameters
    ----------
    units:
        Iterable of :class:`ProofCarryingCodeUnit`.  Unit IDs must be
        unique within the graph; duplicates raise ``ValueError``.
    dependency_edges:
        Optional iterable of ``(from_unit_id, to_unit_id)`` tuples.  An
        edge ``(A, B)`` means "unit A's verification depends on unit B's
        proof being intact" — i.e., if B's claims become invalid, A
        is transitively impacted.  Edges referencing unknown units, or
        self-loops, raise ``ValueError`` at construction time so the
        graph cannot be built into an inconsistent state.
    """

    def __init__(
        self,
        units: Iterable[ProofCarryingCodeUnit],
        *,
        dependency_edges: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._units: dict[str, ProofCarryingCodeUnit] = {}
        self._by_claim: dict[str, set[str]] = defaultdict(set)
        self._by_receipt: dict[str, set[str]] = defaultdict(set)
        self._by_crux: dict[str, set[str]] = defaultdict(set)
        self._depends_on: dict[str, set[str]] = defaultdict(set)
        # Reverse adjacency: who depends on me? (precomputed for BFS).
        self._depended_on_by: dict[str, set[str]] = defaultdict(set)

        for unit in units:
            uid = unit.code_unit_id
            if uid in self._units:
                raise ValueError(
                    f"duplicate code_unit_id {uid!r}; each unit must be unique within a graph"
                )
            self._units[uid] = unit
            for claim in unit.claims:
                self._by_claim[claim].add(uid)
            for receipt in unit.decision_receipts:
                self._by_receipt[receipt].add(uid)
            for crux_id in unit.linked_crux_ids:
                self._by_crux[crux_id].add(uid)

        for from_uid, to_uid in dependency_edges:
            if from_uid not in self._units:
                raise ValueError(
                    f"dependency edge from unknown unit {from_uid!r}; "
                    "every edge endpoint must reference a unit in the graph"
                )
            if to_uid not in self._units:
                raise ValueError(
                    f"dependency edge to unknown unit {to_uid!r}; "
                    "every edge endpoint must reference a unit in the graph"
                )
            if from_uid == to_uid:
                raise ValueError(
                    f"self-loop dependency edge on {from_uid!r}; a unit cannot depend on itself"
                )
            self._depends_on[from_uid].add(to_uid)
            self._depended_on_by[to_uid].add(from_uid)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def unit_count(self) -> int:
        """Number of proof units in the graph."""
        return len(self._units)

    @property
    def claim_count(self) -> int:
        """Number of distinct claim IDs referenced across all units."""
        return len(self._by_claim)

    @property
    def receipt_count(self) -> int:
        """Number of distinct decision receipt IDs referenced across all units."""
        return len(self._by_receipt)

    @property
    def crux_count(self) -> int:
        """Number of distinct crux IDs referenced across all units."""
        return len(self._by_crux)

    @property
    def edge_count(self) -> int:
        """Number of explicit unit-to-unit dependency edges in the graph."""
        return sum(len(targets) for targets in self._depends_on.values())

    # ------------------------------------------------------------------
    # Lookup helpers — all return deterministic sorted lists
    # ------------------------------------------------------------------

    def units_by_claim(self, claim_id: str) -> list[ProofCarryingCodeUnit]:
        """Return units whose ``claims`` list contains *claim_id*."""
        return [self._units[uid] for uid in sorted(self._by_claim.get(claim_id, set()))]

    def units_by_receipt(self, receipt_id: str) -> list[ProofCarryingCodeUnit]:
        """Return units whose ``decision_receipts`` list contains *receipt_id*."""
        return [self._units[uid] for uid in sorted(self._by_receipt.get(receipt_id, set()))]

    def units_by_crux(self, crux_id: str) -> list[ProofCarryingCodeUnit]:
        """Return units whose ``linked_crux_ids`` list contains *crux_id*."""
        return [self._units[uid] for uid in sorted(self._by_crux.get(crux_id, set()))]

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def impact_set(self, claim_ids: Collection[str]) -> set[str]:
        """Return ``code_unit_id``s of all units directly impacted by *claim_ids*.

        A unit is impacted if at least one of the given claim IDs appears in
        its ``claims`` list.  The result is a flat set of unit IDs — not unit
        objects — so callers can log, diff, or feed into repair pipelines
        without holding object references.

        This is a single-hop query — only direct claim-owners are returned.
        For transitive propagation through unit-to-unit dependency edges,
        use :meth:`multi_hop_impact_set`.
        """
        impacted: set[str] = set()
        for claim_id in claim_ids:
            impacted.update(self._by_claim.get(claim_id, set()))
        return impacted

    def direct_dependencies(self, unit_id: str) -> list[str]:
        """Return the sorted list of unit_ids that *unit_id* directly depends on.

        Returns an empty list if *unit_id* is unknown or has no edges.
        """
        return sorted(self._depends_on.get(unit_id, set()))

    def direct_dependents(self, unit_id: str) -> list[str]:
        """Return the sorted list of unit_ids that directly depend on *unit_id*.

        Returns an empty list if *unit_id* is unknown or no other unit
        depends on it.
        """
        return sorted(self._depended_on_by.get(unit_id, set()))

    def multi_hop_impact_set(
        self,
        claim_ids: Collection[str],
        *,
        max_depth: int | None = None,
    ) -> set[str]:
        """Return ``code_unit_id``s transitively impacted by *claim_ids*.

        First seeds the impact set with the direct claim-owners (same as
        :meth:`impact_set`).  Then walks the reverse-dependency graph via
        breadth-first search, gathering every unit that transitively
        depends on a seed unit.

        ``max_depth`` (``None`` = unbounded) limits how far the BFS will
        propagate.  ``max_depth=0`` returns just the seed set, equivalent
        to :meth:`impact_set`.  ``max_depth=1`` adds first-degree
        dependents, and so on.

        Cycles in the dependency graph are handled correctly: each unit
        is visited at most once, regardless of edge multiplicity or
        cyclic structure.  Without explicit dependency edges (the default
        construction), this method is identical to :meth:`impact_set`.
        """
        seed = self.impact_set(claim_ids)
        if not seed:
            return set()
        visited: set[str] = set(seed)
        frontier: list[str] = list(seed)
        depth = 0
        while frontier and (max_depth is None or depth < max_depth):
            next_frontier: list[str] = []
            for uid in frontier:
                for dependent in self._depended_on_by.get(uid, set()):
                    if dependent not in visited:
                        visited.add(dependent)
                        next_frontier.append(dependent)
            frontier = next_frontier
            depth += 1
        return visited

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary of the graph.

        Suitable for operator dashboards, debug logging, or snapshotting
        the state of the constraint graph at a point in time.
        """
        return {
            "unit_count": self.unit_count,
            "claim_count": self.claim_count,
            "receipt_count": self.receipt_count,
            "crux_count": self.crux_count,
            "edge_count": self.edge_count,
            "units": {
                uid: {
                    "symbol": u.symbol,
                    "source_path": u.source_path,
                    "owner": u.owner,
                    "claims": u.claims,
                    "decision_receipts": u.decision_receipts,
                    "linked_crux_ids": u.linked_crux_ids,
                    "freshness_sla_hours": u.freshness_sla_hours,
                }
                for uid, u in sorted(self._units.items())
            },
            "claim_index": {claim: sorted(uids) for claim, uids in sorted(self._by_claim.items())},
            "receipt_index": {
                receipt: sorted(uids) for receipt, uids in sorted(self._by_receipt.items())
            },
            "crux_index": {
                crux_id: sorted(uids) for crux_id, uids in sorted(self._by_crux.items())
            },
            "dependency_edges": [
                [from_uid, to_uid]
                for from_uid in sorted(self._depends_on)
                for to_uid in sorted(self._depends_on[from_uid])
            ],
        }
