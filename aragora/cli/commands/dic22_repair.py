"""CLI command: ``aragora repair-spec``.

DIC-22 operator surface for the verified replacement pipeline (issue #6033).

Reads a :class:`~aragora.epistemic.decay_monitor.DecaySignal` JSON file
(produced by ``aragora decay-monitor --json``) and emits a bounded
:class:`~aragora.epistemic.repair.RepairSpec`.

Flag: ``ARAGORA_REPAIR_PIPELINE_ENABLED`` (default OFF).
Live queue effect: none — produces a spec artifact only; no queue writes.
``live_swap`` repair_kind is unconditionally blocked.
Advances: issue #6033 (DIC-22).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_FLAG = "ARAGORA_REPAIR_PIPELINE_ENABLED"
_ALLOWED_KINDS = ("report_only", "shadow_candidate", "pr_candidate")


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _parse_decay_signal(data: dict[str, Any]):  # type: ignore[return]
    """Reconstruct a DecaySignal from its to_dict() output."""
    from aragora.epistemic.decay_monitor import DecayReason, DecaySignal

    if not isinstance(data, dict):
        raise ValueError("decay signal JSON must be an object")
    code_unit_id = data.get("code_unit_id")
    if not code_unit_id:
        raise ValueError("decay signal missing required field: code_unit_id")
    raw_score = data.get("integrity_score", 1.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"integrity_score is not a number: {raw_score!r}") from exc

    reasons = []
    for r in data.get("reasons", []):
        if isinstance(r, dict):
            reasons.append(
                DecayReason(
                    kind=str(r.get("kind", "unknown")),
                    detail=str(r.get("detail", "")),
                    claim_id=str(r.get("claim_id", "")),
                    crux_id=str(r.get("crux_id", "")),
                )
            )

    return DecaySignal(
        code_unit_id=str(code_unit_id),
        integrity_score=max(0.0, min(1.0, score)),
        reasons=reasons,
        recommended_action=str(data.get("recommended_action", "report_only")),
    )


def cmd_repair_spec(args: argparse.Namespace) -> int:
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable repair-spec",
            file=sys.stderr,
        )
        return 1

    signal_file = Path(getattr(args, "signal_file", "")).expanduser()
    if not signal_file.exists():
        print(f"error: signal file not found: {signal_file}", file=sys.stderr)
        return 1

    try:
        raw = signal_file.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: signal file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        signal = _parse_decay_signal(data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    kind = getattr(args, "kind", "report_only")
    if kind == "live_swap":
        print("error: repair_kind 'live_swap' is unconditionally blocked", file=sys.stderr)
        return 1

    try:
        from aragora.epistemic.repair import propose_repair

        spec = propose_repair(signal, repair_kind=kind)  # type: ignore[arg-type]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(spec.to_dict(), indent=2))
    else:
        print(f"repair-spec: {spec.spec_id}")
        print(f"  code_unit_id : {spec.code_unit_id}")
        print(f"  repair_kind  : {spec.repair_kind}")
        print(f"  integrity    : {spec.decay_signal.integrity_score:.3f}")
        if spec.linked_claims:
            print(f"  linked_claims: {', '.join(spec.linked_claims)}")
        if spec.linked_crux_ids:
            print(f"  linked_cruxes: {', '.join(spec.linked_crux_ids)}")
        if spec.provenance_hash:
            print(f"  provenance   : {spec.provenance_hash}")

    return 0
