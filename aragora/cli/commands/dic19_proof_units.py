"""CLI command: ``aragora proof-units``.

DIC-19 operator surface for the proof-carrying code unit constraint graph
(issue #6030).

Reads proof-unit YAML manifests from a directory, builds the constraint
graph, and reports all units or the impact set for specified claim IDs.

Flag: ``ARAGORA_PROOF_UNIT_SCAN_ENABLED`` (default OFF).
Live queue effect: none — read-only operator surface.
Advances: issue #6030 (DIC-19).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_FLAG = "ARAGORA_PROOF_UNIT_SCAN_ENABLED"
_DEFAULT_DIR = "docs/status/proof_units"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def cmd_proof_units(args: argparse.Namespace) -> int:
    """Show proof-carrying code units and their constraint-graph relationships."""
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable proof-units commands",
            file=sys.stderr,
        )
        return 1

    proof_units_dir = Path(getattr(args, "proof_units_dir", None) or _DEFAULT_DIR).expanduser()
    impact_of: list[str] = list(getattr(args, "impact_of", None) or [])
    multi_hop: bool = bool(getattr(args, "multi_hop", False))
    as_json: bool = bool(getattr(args, "json", False))

    if not proof_units_dir.exists():
        print(
            f"error: --proof-units-dir {proof_units_dir} does not exist",
            file=sys.stderr,
        )
        return 1

    from aragora.epistemic.proof_unit import (
        ProofUnitConstraintGraph,
        load_proof_units_from_dir,
    )

    units = load_proof_units_from_dir(proof_units_dir)
    graph = ProofUnitConstraintGraph(units)
    generated_at = datetime.now(tz=timezone.utc).isoformat()

    if impact_of:
        if multi_hop:
            impacted = sorted(graph.multi_hop_impact_set(impact_of))
        else:
            impacted = sorted(graph.impact_set(impact_of))

        if as_json:
            print(
                json.dumps(
                    {
                        "generated_at": generated_at,
                        "query_claims": impact_of,
                        "multi_hop": multi_hop,
                        "impacted_units": impacted,
                        "total": len(impacted),
                    },
                    indent=2,
                )
            )
        else:
            hop_note = " (multi-hop)" if multi_hop else " (direct)"
            print(f"Impact set{hop_note} for {len(impact_of)} claim(s):")
            for cid in impact_of:
                print(f"  claim: {cid}")
            print()
            print(f"Impacted units: {len(impacted)}")
            for uid in impacted:
                print(f"  - {uid}")
    else:
        if as_json:
            snap = graph.to_dict()
            snap["generated_at"] = generated_at
            print(json.dumps(snap, indent=2))
        else:
            print(f"Proof-carrying code units ({graph.unit_count} total)")
            print(f"  distinct claims   : {graph.claim_count}")
            print(f"  distinct receipts : {graph.receipt_count}")
            print(f"  distinct cruxes   : {graph.crux_count}")
            print(f"  dependency edges  : {graph.edge_count}")
            if units:
                print()
                for u in sorted(units, key=lambda x: x.code_unit_id):
                    print(f"  [{u.owner}] {u.code_unit_id}")
                    if u.claims:
                        print(f"    claims : {', '.join(u.claims)}")
                    if u.linked_crux_ids:
                        print(f"    cruxes : {', '.join(u.linked_crux_ids)}")

    return 0
