"""CLI command: ``aragora decay-monitor``.

DIC-20 operator surface for the epistemic decay monitor (issue #6031).

Loads proof-carrying code unit YAML manifests from ``--units-dir``, then
optionally reads pre-computed ClaimResult rows from ``--claim-results``
JSONL, and emits per-unit DecaySignal assessments.

Flag: ``ARAGORA_DECAY_MONITOR_ENABLED`` (default OFF).
Live queue effect: none — read-only report; no queue writes.
Advances: issue #6031 (DIC-20).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FLAG = "ARAGORA_DECAY_MONITOR_ENABLED"
_DEFAULT_UNITS_DIR = ".aragora_proof_units"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _load_manifests(units_dir: Path) -> list[dict[str, Any]]:
    import yaml  # type: ignore[import-untyped]  # ImportError propagates to cmd_decay_monitor

    out: list[dict[str, Any]] = []
    for p in sorted(units_dir.glob("*.yaml")):
        try:
            with p.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                out.append(data)
            else:
                logger.warning("manifest %s: not a dict, skipped", p.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("manifest %s skipped: %s", p.name, exc)
    return out


def _parse_claim_results(path: Path) -> dict[str, Any]:
    from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus

    raw = path.read_text(encoding="utf-8").strip()
    rows: list[dict[str, Any]] = []
    try:
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        for i, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line.strip()))
                except json.JSONDecodeError as exc:
                    logger.warning("claim-results line %d skipped: %s", i, exc)
    out: dict[str, ClaimResult] = {}
    for row in rows:
        try:
            cid = str(row["claim_id"])
            out[cid] = ClaimResult(
                claim_id=cid,
                status=ClaimStatus(row["status"]),
                message=str(row.get("message", "")),
                severity=str(row.get("severity", "info")),
                allowed_action=str(row.get("allowed_action", "report_only")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("claim-results row skipped: %s", exc)
    return out


def cmd_decay_monitor(args: argparse.Namespace) -> int:
    if not _flag_enabled():
        print(f"error: {_FLAG} is not set; set it to '1' to enable decay-monitor", file=sys.stderr)
        return 1

    units_dir = Path(getattr(args, "units_dir", _DEFAULT_UNITS_DIR)).expanduser()
    if not units_dir.is_dir():
        print(f"error: units-dir not found: {units_dir}", file=sys.stderr)
        return 1

    cr_str: str | None = getattr(args, "claim_results", None)
    claim_results: dict[str, Any] = {}
    if cr_str:
        cr_path = Path(cr_str).expanduser()
        if not cr_path.exists():
            print(f"error: claim-results file not found: {cr_path}", file=sys.stderr)
            return 1
        try:
            claim_results = _parse_claim_results(cr_path)
        except ImportError:
            # ``_parse_claim_results`` imports the epistemic package, which pulls
            # in pyyaml at module load. Fail closed with the same clear message
            # rather than a raw ModuleNotFoundError traceback.
            print(
                "error: pyyaml is required but not installed; install it to use decay-monitor",
                file=sys.stderr,
            )
            return 1

    try:
        manifests = _load_manifests(units_dir)
    except ImportError:
        print(
            "error: pyyaml is required but not installed; install it to use decay-monitor",
            file=sys.stderr,
        )
        return 1
    signals = []
    if manifests:
        from aragora.epistemic.proof_unit_model import load_proof_unit
        from aragora.epistemic.decay_monitor import evaluate_unit

        for data in manifests:
            try:
                signals.append(
                    evaluate_unit(load_proof_unit(data), claim_results=claim_results or None)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("unit %s skipped: %s", data.get("code_unit_id", "?"), exc)

    ts = datetime.now(timezone.utc).isoformat()
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "generated_at": ts,
                    "total": len(signals),
                    "signals": [s.to_dict() for s in signals],
                },
                indent=2,
            )
        )
    else:
        print(f"Decay monitor — {ts}\n{len(signals)} unit(s) evaluated\n")
        for s in signals:
            print(
                f"  {s.code_unit_id}: integrity={s.integrity_score:.3f}  action={s.recommended_action}"
            )
            for r in s.reasons:
                print(f"    [{r.kind}] {r.detail}")
        if not signals:
            print("  (no proof-unit manifests found)")
    return 0
