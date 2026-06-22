"""CLI command: aragora crux-garden.

DIC-28 operator surface for the proactive crux gardening pass (issue #6222).

Reads a JSONL or JSON-array file of CruxReceipt dicts, runs a full
gardening pass, and prints the GardeningReport as JSON or a text summary.
No debate is started; no issue is created; no queue is touched.

Flag: ARAGORA_CRUX_GARDENING_ENABLED (default OFF).
Live queue effect: none — read-only analysis.
Advances: issue #6222 (DIC-28).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt
from aragora.epistemic.gardening import GardeningConfig, GardeningReport, run_gardening_pass

_FLAG = "ARAGORA_CRUX_GARDENING_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag_enabled() -> bool:
    return str(os.environ.get(_FLAG) or "").strip().lower() in _TRUTHY


def _parse_receipt(d: dict[str, Any]) -> CruxReceipt:
    """Deserialise one CruxReceipt from a plain dict (JSON/JSONL)."""
    cruxes = [
        CruxEntry(
            crux_id=str(c.get("crux_id", "")),
            statement=str(c.get("statement", "")),
            load_bearing_score=float(c.get("load_bearing_score", 0.0)),
            uncertainty_score=float(c.get("uncertainty_score", 0.0)),
            contesting_agents=list(c.get("contesting_agents") or []),
            affected_claims=list(c.get("affected_claims") or []),
            resolution_impact=float(c.get("resolution_impact", 0.0)),
        )
        for c in (d.get("cruxes") or [])
    ]
    return CruxReceipt(
        receipt_id=str(d.get("receipt_id", "")),
        debate_id=str(d.get("debate_id", "")),
        question=str(d.get("question", "")),
        cruxes=cruxes,
        convergence_barrier=float(d.get("convergence_barrier", 0.0)),
        counterfactuals=list(d.get("counterfactuals") or []),
        agents=list(d.get("agents") or []),
        rounds=int(d.get("rounds", 0)),
        metadata=dict(d.get("metadata") or {}),
        checksum=str(d.get("checksum", "")),
    )


def _load_receipts(path: Path) -> list[CruxReceipt] | str:
    """Load CruxReceipts from a JSON array or JSONL file.

    Returns a list on success or an error string on failure.
    """
    if not path.exists():
        return f"input file not found: {path}"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            raw = json.loads(text)
            return [_parse_receipt(item) for item in (raw if isinstance(raw, list) else [])]
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return f"invalid JSON: {exc}"
    receipts: list[CruxReceipt] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            receipts.append(_parse_receipt(json.loads(line)))
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return f"line {lineno}: {exc}"
    return receipts


def _format_text(report: GardeningReport) -> str:
    lines = [
        f"Gardening report  {report.generated_at}",
        f"  resolved: {len(report.resolved_results)}  outstanding: {len(report.outstanding_results)}",
        "Summary: " + "  ".join(f"{k}={v}" for k, v in sorted(report.summary.items())),
    ]
    findings = [
        r for r in report.resolved_results + report.outstanding_results if r.status != "healthy"
    ]
    if findings:
        lines.append("Findings:")
        for r in findings:
            suffix = " [needs-followup]" if r.needs_followup else ""
            lines.append(f"  [{r.status}] {r.crux_id}: {r.detail}{suffix}")
    return "\n".join(lines)


def cmd_crux_garden(args: argparse.Namespace) -> int:
    if not _flag_enabled():
        print(f"error: {_FLAG} is not set; crux gardening is disabled", file=sys.stderr)
        return 1
    result = _load_receipts(Path(args.input))
    if isinstance(result, str):
        print(f"error: {result}", file=sys.stderr)
        return 1
    report = run_gardening_pass(result, [], config=GardeningConfig(enabled=True))
    print(report.to_json() if args.json else _format_text(report))
    return 0
