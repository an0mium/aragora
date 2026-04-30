"""Render the H1 multi-gate readiness summary.

Loads each H1 sub-gate's contract status from its canonical surface
and feeds it through ``aragora.swarm.h1_readiness.aggregate_readiness``
to produce one Markdown summary covering all four gates.

Status detection (heuristic, contract-doc driven):

- **H1-01**: parses ``scripts/render_rev4_promotion_readiness.py``
  output JSON file (if present) and maps ``status`` →
  ``ready`` (``promotion_ready``) or ``advisory_in_progress``.
- **H1-02**: detects ``H1_02_SCORECARD_CONTRACT.md`` "Status: IN
  PLACE" → ``ready``.
- **H1-03**: detects ``H1_03_SANITATION_GATE_CONTRACT.md`` "Status:
  IN PLACE" → ``ready``.
- **H1-04**: detects ``H1_04_LEDGER_SELF_HEAL_CONTRACT.md`` "Status:
  IN PLACE" → ``ready``.

If a contract doc is missing or doesn't match either pattern, the
gate is reported as ``unknown``.

Usage::

    python3 scripts/render_h1_multi_gate_readiness.py
    python3 scripts/render_h1_multi_gate_readiness.py --json
    python3 scripts/render_h1_multi_gate_readiness.py \\
        --h1-01-readiness-json path/to/rev4_promotion_readiness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.h1_readiness import (  # noqa: E402
    GateInput,
    GateStatus,
    aggregate_readiness,
    render_markdown,
)

CONTRACT_DOCS = {
    "H1-01": REPO_ROOT / "docs/status/H1_01_REV4_PROMOTION_READINESS.md",
    "H1-02": REPO_ROOT / "docs/status/H1_02_SCORECARD_CONTRACT.md",
    "H1-03": REPO_ROOT / "docs/status/H1_03_SANITATION_GATE_CONTRACT.md",
    "H1-04": REPO_ROOT / "docs/status/H1_04_LEDGER_SELF_HEAL_CONTRACT.md",
}


def _detect_in_place(contract_path: Path) -> bool:
    """Heuristic: contract doc claims 'Status: IN PLACE'."""
    if not contract_path.exists():
        return False
    text = contract_path.read_text(encoding="utf-8")
    return "Status:**" in text and "IN PLACE" in text


def _detect_h1_01_from_doc(contract_path: Path) -> GateStatus:
    """H1-01's contract doc embeds a verdict line like
    ``- Status: \\`promotion_ready\\``` (or ``needs_more_dispatch_evidence``).
    Extract that without requiring a separate JSON artifact.
    """
    if not contract_path.exists():
        return "unknown"
    text = contract_path.read_text(encoding="utf-8")
    if "Status: `promotion_ready`" in text:
        return "ready"
    if "Status: `needs_more_dispatch_evidence`" in text or "Status: `advisory_only`" in text:
        return "advisory_in_progress"
    return "unknown"


def _h1_01_status(readiness_json: Path | None) -> GateStatus:
    """H1-01 status from the rev-4 promotion readiness JSON if provided,
    otherwise from the contract doc's embedded verdict line."""
    if readiness_json is not None and readiness_json.exists():
        try:
            payload = json.loads(readiness_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "unknown"
        status = str(payload.get("status") or "")
        if status == "promotion_ready":
            return "ready"
        if status in {"needs_more_dispatch_evidence", "advisory_only"}:
            return "advisory_in_progress"
        return "unknown"
    return _detect_h1_01_from_doc(CONTRACT_DOCS["H1-01"])


def _build_inputs(args: argparse.Namespace) -> list[GateInput]:
    inputs: list[GateInput] = []

    h1_01_status = _h1_01_status(args.h1_01_readiness_json)
    inputs.append(
        GateInput(
            gate_id="H1-01",
            status=h1_01_status,
            contract_doc=str(CONTRACT_DOCS["H1-01"].relative_to(REPO_ROOT)),
        )
    )

    for gid in ("H1-02", "H1-03", "H1-04"):
        status: GateStatus = "ready" if _detect_in_place(CONTRACT_DOCS[gid]) else "unknown"
        inputs.append(
            GateInput(
                gate_id=gid,
                status=status,
                contract_doc=str(CONTRACT_DOCS[gid].relative_to(REPO_ROOT)),
            )
        )

    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h1-01-readiness-json",
        type=Path,
        default=None,
        help="Optional rev-4 promotion readiness JSON; if omitted, "
        "the contract doc heuristic is used.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    args = parser.parse_args(argv)

    inputs = _build_inputs(args)
    readiness = aggregate_readiness(inputs)

    if args.json:
        sys.stdout.write(json.dumps(asdict(readiness), default=str, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(readiness))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
