"""Render the cross-round cadence summary.

Walks ``.aragora/evolve-round/<round-id>/dogfood/phase-*-receipt.json``
across all rounds, loads each receipt into a ``PhaseReceipt``, and
feeds them through ``aragora.swarm.round_cadence.aggregate_cadence``
to produce a Markdown or JSON summary.

Usage::

    python3 scripts/render_round_cadence.py
    python3 scripts/render_round_cadence.py --json
    python3 scripts/render_round_cadence.py \\
        --rounds-dir .aragora/evolve-round \\
        --output round-cadence.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.round_cadence import (  # noqa: E402
    PhaseReceipt,
    aggregate_cadence,
    render_markdown,
)

DEFAULT_ROUNDS_DIR = REPO_ROOT / ".aragora/evolve-round"
RECEIPT_GLOB = "phase-*-receipt.json"
PHASE_FROM_FILENAME = re.compile(r"phase-([a-zA-Z])-receipt\.json$")


def _phase_from_path(path: Path) -> str:
    m = PHASE_FROM_FILENAME.search(path.name)
    return m.group(1).upper() if m else "?"


def _load_receipt(round_id: str, path: Path) -> PhaseReceipt | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    phase = str(payload.get("phase") or _phase_from_path(path)).upper()[:1]
    status = str(payload.get("status") or "unknown")
    raw_pr = payload.get("pr_number")
    pr_number: int | None = raw_pr if isinstance(raw_pr, int) else None
    halt_tripped = bool(payload.get("halt_tripped", False))
    name = str(payload.get("name") or "")
    return PhaseReceipt(
        round_id=round_id,
        phase=phase,
        status=status,
        pr_number=pr_number,
        halt_tripped=halt_tripped,
        name=name,
    )


def _scan_rounds(rounds_dir: Path) -> list[PhaseReceipt]:
    if not rounds_dir.is_dir():
        return []
    receipts: list[PhaseReceipt] = []
    for round_dir in sorted(rounds_dir.iterdir()):
        if not round_dir.is_dir():
            continue
        round_id = round_dir.name
        dogfood = round_dir / "dogfood"
        if not dogfood.is_dir():
            continue
        for receipt_path in sorted(dogfood.glob(RECEIPT_GLOB)):
            r = _load_receipt(round_id, receipt_path)
            if r is not None:
                receipts.append(r)
    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds-dir",
        type=Path,
        default=DEFAULT_ROUNDS_DIR,
        help="Directory containing per-round subdirectories.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write to this file instead of stdout.",
    )
    args = parser.parse_args(argv)

    receipts = _scan_rounds(args.rounds_dir)
    summary = aggregate_cadence(receipts)
    if args.json:
        rendered = json.dumps(asdict(summary), default=str, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
