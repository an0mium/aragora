"""CLI command: ``aragora quarantine-eval``.

DIC-21 operator surface for the fail-closed quarantine policy (issue #6032).

Reads a DecaySignal JSON file, applies quarantine policy, and emits the
QuarantineDecision as text or JSON.

Flag: ``ARAGORA_QUARANTINE_POLICY_ENABLED`` (default OFF).
Live queue effect: none — read-only report; no queue writes.
Advances: issue #6032 (DIC-21).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_FLAG = "ARAGORA_QUARANTINE_POLICY_ENABLED"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_signal(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read signal file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"signal file must be a JSON object, got {type(data).__name__}")
    if "code_unit_id" not in data:
        raise ValueError("signal JSON missing required field 'code_unit_id'")
    return data


class _Reason:
    """Minimal duck-typed reason; quarantine_policy only reads ``r.kind``."""

    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


class _Signal:
    """Minimal duck-typed DecaySignal for quarantine_policy consumption.

    Avoids importing decay_monitor → claim_verifier → pyyaml at CLI load
    time.  apply_quarantine_policy uses duck-typed attribute access only.
    """

    __slots__ = ("code_unit_id", "integrity_score", "reasons", "recommended_action")

    def __init__(self, uid: str, score: float, reasons: list, action: str) -> None:
        self.code_unit_id = uid
        self.integrity_score = score
        self.reasons = reasons
        self.recommended_action = action


def _parse_signal(data: dict) -> _Signal:
    reasons = [
        _Reason(kind=str(r.get("kind", "")))
        for r in (data.get("reasons") or [])
        if isinstance(r, dict)
    ]
    return _Signal(
        uid=str(data["code_unit_id"]),
        score=float(data.get("integrity_score", 1.0)),
        reasons=reasons,
        action=str(data.get("recommended_action", "report_only")),
    )


def cmd_quarantine_eval(args: argparse.Namespace) -> int:
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable quarantine-eval", file=sys.stderr
        )
        return 1

    signal_path = Path(getattr(args, "signal", "")).expanduser()
    if not signal_path.exists():
        print(f"error: signal file not found: {signal_path}", file=sys.stderr)
        return 1

    try:
        data = _load_signal(signal_path)
        signal = _parse_signal(data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    code_unit_class: str = getattr(args, "code_unit_class", "default") or "default"
    request_live_swap: bool = bool(getattr(args, "request_live_swap", False))

    from aragora.epistemic.quarantine_policy import apply_quarantine_policy

    decision = apply_quarantine_policy(
        signal,  # type: ignore[arg-type]  # duck-typed _Signal; runtime OK
        code_unit_class=code_unit_class,
        request_live_swap=request_live_swap,
    )

    if getattr(args, "json", False):
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        fc = "YES" if decision.fail_closed else "no"
        ls = "BLOCKED" if decision.live_swap_blocked else "allowed"
        print(f"quarantine-eval: {decision.code_unit_id}")
        print(f"  action:      {decision.policy_action}")
        print(f"  integrity:   {decision.integrity_score:.3f}")
        print(f"  fail_closed: {fc}  live_swap: {ls}")
        print(f"  rationale:   {decision.rationale}")
        if decision.provenance_hash:
            print(f"  hash:        {decision.provenance_hash[:16]}…")
    return 0
