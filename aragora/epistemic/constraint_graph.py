"""Proof-Carrying Code Unit constraint graph (DIC-19 / #6030).

Builds a read-only dependency graph from a collection of
:class:`aragora.epistemic.proof_unit_model.ProofCarryingCodeUnit` instances.

The graph indexes each unit by its shared ``claims``, ``decision_receipts``,
and ``linked_crux_ids`` so callers can answer in O(1):

- "Which units depend on claim X?"
- "Which units are tied to receipt R?"
- "Which units reference crux C?"
- "What is the full impact set if these claims become invalid?"

Construction is explicit (``ProofUnitConstraintGraph(units)``); nothing in this
module runs automatically.  The dataclass is always importable; the scanner that
populates the unit list is gated by ``ARAGORA_PROOF_UNIT_SCAN_ENABLED``.

Out of scope (deferred):
- Multi-hop unit-to-unit dependency propagation (requires explicit edges).
- Mutating units or triggering quarantine/repair — see DIC-21/DIC-22.
- Persisting the graph — callers serialize via :meth:`to_dict`.
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
    """

    def __init__(self, units: Iterable[ProofCarryingCodeUnit]) -> None:
        self._units: dict[str, ProofCarryingCodeUnit] = {}
        self._by_claim: dict[str, set[str]] = defaultdict(set)
        self._by_receipt: dict[str, set[str]] = defaultdict(set)
        self._by_crux: dict[str, set[str]] = defaultdict(set)

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

        This is a single-hop query.  Multi-hop propagation (unit A relies on
        unit B's proof) is deferred until explicit unit-to-unit dependency
        edges are tracked in a future DIC-19 follow-on.
        """
        impacted: set[str] = set()
        for claim_id in claim_ids:
            impacted.update(self._by_claim.get(claim_id, set()))
        return impacted

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
        }
