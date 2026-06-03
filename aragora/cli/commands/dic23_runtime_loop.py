"""CLI verb: ``aragora dialectical-loop`` (DIC-23 / #6217).

Reads a DecaySignal JSON file, runs one DIC-23 orchestration pass, and
emits the DialecticalEvent as a text report or JSON.

Flag gate: ``ARAGORA_DIALECTICAL_RUNTIME_ENABLED`` (default off).
Live queue effect: none — report-only trace; no issues created.

Signal JSON shape (from DecaySignal.to_dict()):
    {"code_unit_id": "...", "integrity_score": 0.45,
     "recommended_action": "repair_required",
     "reasons": [{"kind": "failed_claim", "detail": "..."}]}

Advances: issue #6217 (DIC-23).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aragora.epistemic.decay_monitor import DecayReason, DecaySignal
from aragora.epistemic.runtime_loop import (
    DialecticalRuntimeError,
    dialectical_runtime_enabled,
    run_dialectical_loop,
)

_FLAG = "ARAGORA_DIALECTICAL_RUNTIME_ENABLED"


def _load_signal(path: Path) -> DecaySignal:
    obj = json.loads(path.read_text(encoding="utf-8"))
    reasons = [
        DecayReason(
            kind=str(r["kind"]),
            detail=str(r.get("detail", "")),
            claim_id=str(r.get("claim_id", "")),
            crux_id=str(r.get("crux_id", "")),
        )
        for r in obj.get("reasons", [])
    ]
    return DecaySignal(
        code_unit_id=str(obj["code_unit_id"]),
        integrity_score=float(obj["integrity_score"]),
        reasons=reasons,
        recommended_action=str(obj.get("recommended_action", "report_only")),
    )


def cmd_dialectical_loop(args: argparse.Namespace) -> int:
    """Handle the ``aragora dialectical-loop`` subcommand."""
    if not dialectical_runtime_enabled():
        print(f"error: {_FLAG} is not set; set it to '1' to enable", file=sys.stderr)
        return 1

    signal_path = Path(str(getattr(args, "signal", ""))).expanduser()
    if not signal_path.exists():
        print(f"error: signal file not found: {signal_path}", file=sys.stderr)
        return 1

    try:
        signal = _load_signal(signal_path)
    except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
        print(f"error: failed to load signal: {exc}", file=sys.stderr)
        return 1

    try:
        event = run_dialectical_loop(
            signal,
            code_unit_class=str(getattr(args, "unit_class", "default")),
            enable_repair_proposal=bool(getattr(args, "repair", False)),
            require_enabled=True,
        )
    except DialecticalRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(event.to_dict(), indent=2))
        return 0

    print(f"Dialectical Runtime Loop  {event.event_id}")
    print(f"  code_unit_id    : {event.code_unit_id}")
    print(f"  integrity_score : {event.integrity_score:.4f}")
    print(f"  recommended     : {event.recommended_action}")
    print(f"  quarantine      : {event.quarantine_action}")
    print(f"  crux_probe      : {'skipped' if event.crux_probe_skipped else 'ran'}")
    if event.repair_spec is not None:
        print(f"  repair_spec     : {event.repair_spec.code_unit_id} (kind={event.repair_spec.repair_kind})")
    print(f"  created_at      : {event.created_at}")
    return 0


__all__ = ["cmd_dialectical_loop"]
