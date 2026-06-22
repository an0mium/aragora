"""CLI verb: ``aragora crux-arbitrate`` (DIC-27 / #6221).

Operator surface for resolving persistent cruxes as reversible, signed
arbitration receipts.  Thin wrapper around
``aragora.epistemic.arbitration`` — no new logic, no queue mutation.

Usage (dry-run, always works regardless of flag):
    aragora crux-arbitrate --input cruxes.json --dry-run

Usage (live, requires ARAGORA_CRUX_ARBITRATION_ENABLED=1):
    aragora crux-arbitrate --input cruxes.json \\
        --crux-id crux_abc123 --side accept \\
        --rationale "Soaks prove the claim holds" \\
        --operator alice --expires-days 90

Input JSON: a list of PersistentCrux dicts (from ``PersistentCrux.to_dict()``)
or a single PersistentCrux dict.

Flag gate: ``ARAGORA_CRUX_ARBITRATION_ENABLED`` (default off).
``--dry-run`` always works.  Any action that produces a
:class:`~aragora.epistemic.arbitration.CruxArbitration` requires the flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aragora.epistemic.arbitration import (
    CruxArbitration,
    PersistentCrux,
    build_arbitration,
    crux_arbitration_enabled,
)

_SIDE_CHOICES = ("accept", "reject", "defer", "split")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cruxes(path: str) -> list[PersistentCrux]:
    """Load one or more PersistentCrux records from a JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    payload: Any = json.loads(raw)

    records: list[dict[str, Any]]
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Expected JSON object or array, got {type(payload).__name__}")

    cruxes: list[PersistentCrux] = []
    for rec in records:
        cruxes.append(
            PersistentCrux(
                crux_id=rec["crux_id"],
                statement=rec["statement"],
                question_family_id=rec["question_family_id"],
                consecutive_debate_count=int(rec["consecutive_debate_count"]),
                load_bearing_score=float(rec["load_bearing_score"]),
                cruxset_receipt_ids=tuple(rec.get("cruxset_receipt_ids", [])),
            )
        )
    return cruxes


def _render_dry_run(cruxes: list[PersistentCrux], *, json_output: bool) -> str:
    """Render qualifying/non-qualifying summary for dry-run mode."""
    qualifying = [c for c in cruxes if c.qualifies]
    not_qualifying = [c for c in cruxes if not c.qualifies]

    if json_output:
        return json.dumps(
            {
                "qualifying": [c.to_dict() for c in qualifying],
                "not_qualifying": [c.to_dict() for c in not_qualifying],
            },
            indent=2,
            sort_keys=True,
        )

    lines: list[str] = []
    lines.append(
        f"[dry-run] {len(cruxes)} crux(es) loaded — {len(qualifying)} qualify for arbitration"
    )
    if qualifying:
        lines.append("\nQualifying cruxes:")
        for c in qualifying:
            lines.append(
                f"  {c.crux_id}  lb={c.load_bearing_score:.3f}"
                f"  consecutive={c.consecutive_debate_count}"
            )
            lines.append(f"    {c.statement}")
    if not_qualifying:
        lines.append("\nNot qualifying (below thresholds):")
        for c in not_qualifying:
            lines.append(
                f"  {c.crux_id}  lb={c.load_bearing_score:.3f}"
                f"  consecutive={c.consecutive_debate_count}"
            )
    return "\n".join(lines)


def _render_arbitration(arb: CruxArbitration, *, json_output: bool) -> str:
    if json_output:
        return json.dumps(arb.to_dict(), indent=2, sort_keys=True)
    lines = [
        f"Arbitration created: {arb.arbitration_id}",
        f"  crux_id:   {arb.crux.crux_id}",
        f"  operator:  {arb.operator}",
        f"  side:      {arb.side}",
        f"  rationale: {arb.rationale}",
        f"  expires:   {arb.expires_at}",
        f"  checksum:  {arb.checksum}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def cmd_crux_arbitrate(args: argparse.Namespace) -> None:
    """Handle the ``aragora crux-arbitrate`` subcommand."""
    input_path = getattr(args, "input", None)
    if not input_path:
        print("crux-arbitrate: --input <json> is required", file=sys.stderr)
        sys.exit(1)

    try:
        cruxes = _load_cruxes(input_path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"crux-arbitrate: failed to load --input: {exc}", file=sys.stderr)
        sys.exit(1)

    json_output: bool = bool(getattr(args, "json", False))
    dry_run: bool = bool(getattr(args, "dry_run", False))

    if dry_run:
        print(_render_dry_run(cruxes, json_output=json_output))
        return

    # --- live arbitration path ---
    if not crux_arbitration_enabled():
        print(
            "crux-arbitrate: ARAGORA_CRUX_ARBITRATION_ENABLED is not set. "
            "Set it to 1 to create arbitrations, or pass --dry-run to inspect "
            "qualifying cruxes without creating a record.",
            file=sys.stderr,
        )
        sys.exit(1)

    crux_id: str | None = getattr(args, "crux_id", None)
    side: str | None = getattr(args, "side", None)
    rationale: str | None = getattr(args, "rationale", None)

    if not crux_id:
        print("crux-arbitrate: --crux-id is required in live mode", file=sys.stderr)
        sys.exit(1)
    if not side:
        print(
            f"crux-arbitrate: --side is required in live mode (choices: {_SIDE_CHOICES})",
            file=sys.stderr,
        )
        sys.exit(1)
    if not rationale:
        print("crux-arbitrate: --rationale is required in live mode", file=sys.stderr)
        sys.exit(1)

    matching = [c for c in cruxes if c.crux_id == crux_id]
    if not matching:
        print(
            f"crux-arbitrate: crux_id {crux_id!r} not found in --input file",
            file=sys.stderr,
        )
        sys.exit(1)

    crux = matching[0]
    if not crux.qualifies:
        print(
            f"crux-arbitrate: crux {crux_id!r} does not qualify for arbitration "
            f"(load_bearing={crux.load_bearing_score:.3f}, "
            f"consecutive={crux.consecutive_debate_count}). "
            "Use --dry-run to see thresholds.",
            file=sys.stderr,
        )
        sys.exit(1)

    operator: str = getattr(args, "operator", None) or "operator"
    expires_days: int = int(getattr(args, "expires_days", None) or 90)
    evidence: list[str] = list(getattr(args, "evidence", None) or [])

    try:
        arb = build_arbitration(
            crux,
            operator=operator,
            side=side,  # type: ignore[arg-type]
            rationale=rationale,
            evidence_citations=evidence or None,
            expiry_days=expires_days,
        )
    except (ValueError, TypeError) as exc:
        print(f"crux-arbitrate: failed to build arbitration: {exc}", file=sys.stderr)
        sys.exit(1)

    rendered = _render_arbitration(arb, json_output=json_output)
    print(rendered)

    output_path: str | None = getattr(args, "output", None)
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(arb.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"Arbitration written to: {target}", file=sys.stderr)
