"""CLI command: ``aragora cruxset show``.

Pretty-prints a CruxSet from a JSON file or stdin. Useful for operators
inspecting CruxSets emitted by the AGT-01 debate-phase wiring (PR
#6110) once the ARAGORA_CRUXSET_EMISSION_ENABLED flag is on.

See aragora.reasoning.cruxset (issue #6062).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aragora.reasoning.cruxset import CruxSet


def cmd_cruxset_show(args: argparse.Namespace) -> int:
    """Load a CruxSet JSON payload, verify checksum, print summary."""
    source = getattr(args, "source", None)
    if source and source != "-":
        path = Path(source).expanduser()
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        raw = path.read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: input is not JSON: {exc}", file=sys.stderr)
        return 2

    try:
        cruxset = CruxSet.from_json(payload)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: payload is not a CruxSet: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "json", False):
        print(json.dumps(cruxset.to_json(), sort_keys=True, indent=2))
        return 0

    verified = cruxset.verify_checksum()
    print(f"CruxSet {cruxset.cruxset_id} (schema {cruxset.schema_version})")
    print(f"  question:          {cruxset.question}")
    print(f"  decision:          {cruxset.decision or '(none)'}")
    print(f"  receipt_id:        {cruxset.receipt_id or '(none)'}")
    print(f"  cruxes:            {len(cruxset.cruxes)}")
    print(f"  evidence_gaps:     {len(cruxset.evidence_gaps)}")
    print(f"  checksum verified: {verified}")
    print()
    for idx, crux in enumerate(cruxset.cruxes, start=1):
        sides = ", ".join(p.side for p in crux.positions) or "(no positions)"
        print(f"  [{idx}] load_bearing={crux.load_bearing_score:.3f}  sides=[{sides}]")
        print(f"      {crux.statement}")
        if crux.counterfactual:
            print(f"      counterfactual: {crux.counterfactual}")
    return 0 if verified else 3


__all__ = ["cmd_cruxset_show"]
