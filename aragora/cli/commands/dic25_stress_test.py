"""CLI command: ``aragora stress-test`` (DIC-25 / #6219).

Flag: ``ARAGORA_STRESS_TEST_ENABLED`` (default OFF).
Live queue effect: none — read-only fragility report; no queue writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_FLAG = "ARAGORA_STRESS_TEST_ENABLED"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_catalog(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else [parsed]


def _load_units(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_stress_test(args: argparse.Namespace) -> int:
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable stress-test",
            file=sys.stderr,
        )
        return 1

    catalog_path = Path(args.catalog).expanduser()
    if not catalog_path.exists():
        print(f"error: catalog file not found: {catalog_path}", file=sys.stderr)
        return 1

    units_path = Path(args.units).expanduser()
    if not units_path.exists():
        print(f"error: units file not found: {units_path}", file=sys.stderr)
        return 1

    try:
        raw_perturbations = _load_catalog(catalog_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: invalid catalog JSON: {exc}", file=sys.stderr)
        return 1

    try:
        integrities: dict[str, float] = _load_units(units_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: invalid units JSON: {exc}", file=sys.stderr)
        return 1

    from aragora.epistemic.stress_test import StressPerturbation, run_stress_test

    perturbations: list[StressPerturbation] = []
    for row in raw_perturbations:
        try:
            perturbations.append(
                StressPerturbation(
                    perturbation_id=str(row["perturbation_id"]),
                    kind=row["kind"],
                    description=str(row["description"]),
                    simulated_impact=float(row.get("simulated_impact", 0.0)),
                    affected_claim_ids=list(row.get("affected_claim_ids", [])),
                    affected_proof_unit_ids=list(row.get("affected_proof_unit_ids", [])),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"warning: perturbation row skipped: {exc}", file=sys.stderr)

    result = run_stress_test(perturbations, integrities, enabled=True)

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"Stress-test: {result.perturbations_tested} perturbation(s) × "
            f"{result.proof_units_probed} unit(s)\n"
        )
        high = result.high_fragility_units
        if high:
            print(f"High-fragility reports ({len(high)}):")
            for r in high:
                print(
                    f"  {r.proof_unit_id} [{r.perturbation_id}]: "
                    f"delta={r.fragility_delta:.3f}  action={r.recommended_action}"
                )
        else:
            print("No high-fragility reports (delta ≤ 0.3 for all units).")
        if result.most_fragile_unit_id:
            print(
                f"\nMost fragile: {result.most_fragile_unit_id} "
                f"(max delta={result.max_fragility_delta:.3f})"
            )
    return 0
