"""CLI command: ``aragora coherence-scan``.

DIC-26 operator surface for the belief coherence monitor (issue #6220).

Reads a JSON file where each element is a BeliefEntry dict:
    {"belief_id": "...", "subject": "...", "confidence": 0.8,
     "status": "pass", "evidence_paths": ["docs/status/foo.md"]}

Flag: ``ARAGORA_COHERENCE_MONITOR_ENABLED`` (default OFF).
Live queue effect: none — read-only operator report.
Advances: issue #6220 (DIC-26).

Report persistence (``--write``):
    Pass ``--write`` to write a timestamped JSON report under the output
    directory (default: ``docs/status/generated/coherence_reports/``
    relative to the current working directory). Override with
    ``--output-dir``. Writing is a separate, additive action — stdout
    output is still emitted regardless.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aragora.epistemic.coherence import (
    BeliefEntry,
    CoherenceReport,
    coherence_monitor_enabled,
    scan_coherence,
)

logger = logging.getLogger(__name__)

_FLAG = "ARAGORA_COHERENCE_MONITOR_ENABLED"
_DEFAULT_GAP: float = 0.5
_DEFAULT_MIN_CONFIDENCE: float = 0.3
_DEFAULT_OUTPUT_DIR = "docs/status/generated/coherence_reports"


def _load_entries(path: Path) -> list[BeliefEntry]:
    """Parse *path* as JSON into a list of :class:`BeliefEntry`.

    Accepts a JSON array or a single JSON object. Malformed entries are
    logged at WARNING and skipped so one bad row does not abort the scan.
    """
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    entries: list[BeliefEntry] = []
    for idx, obj in enumerate(raw, 1):
        try:
            entries.append(
                BeliefEntry(
                    belief_id=str(obj["belief_id"]),
                    subject=str(obj["subject"]),
                    confidence=float(obj["confidence"]),
                    status=str(obj.get("status", "unknown")),
                    evidence_paths=tuple(str(p) for p in obj.get("evidence_paths") or []),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("belief entry %d skipped: %s", idx, exc)
    return entries


def _write_report(report: CoherenceReport, output_dir: Path) -> Path:
    """Write *report* as timestamped JSON under *output_dir*.

    Creates the directory (and all parents) if absent. Returns the path
    of the written file. Never raises on directory creation — callers
    should handle ``OSError`` from ``write_text``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"coherence_report_{stamp}.json"
    dest = output_dir / filename
    dest.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return dest


def cmd_coherence_scan(args: argparse.Namespace) -> int:
    """Handle the ``aragora coherence-scan`` subcommand."""
    if not coherence_monitor_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable coherence-scan",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        entries = _load_entries(input_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: failed to load {input_path}: {exc}", file=sys.stderr)
        return 1

    report = scan_coherence(
        entries,
        contradiction_gap=float(getattr(args, "contradiction_gap", _DEFAULT_GAP)),
        min_confidence=float(getattr(args, "min_confidence", _DEFAULT_MIN_CONFIDENCE)),
        enabled=True,
    )

    as_json: bool = getattr(args, "json", False)
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Coherence scan: {input_path}")
        print(f"  scanned            : {report.scanned}")
        print(f"  coherent           : {report.coherent}")
        print(f"  contradictions     : {report.contradiction_count}")
        print(f"  evidence conflicts : {report.evidence_conflict_count}")
        print(f"  confidence rot     : {report.confidence_rot_count}")
        if report.issues:
            print()
            for issue in report.issues:
                ids = ", ".join(issue.belief_ids)
                print(f"  [{issue.severity}] {issue.kind.value}: {ids}")
                print(f"    {issue.detail}")

    do_write: bool = getattr(args, "write", False)
    if do_write:
        raw_dir: str = getattr(args, "output_dir", None) or _DEFAULT_OUTPUT_DIR
        output_dir = Path(raw_dir).expanduser()
        try:
            dest = _write_report(report, output_dir)
            print(f"report written: {dest}", file=sys.stderr)
        except OSError as exc:
            print(f"error: could not write report to {output_dir}: {exc}", file=sys.stderr)
            return 1

    return 0
