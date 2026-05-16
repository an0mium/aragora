"""CLI commands: ``aragora metrics viah`` and ``aragora metrics status``.

Reads the ShiftLedger and prints VIAH (verifiable improvements per
agent-hour) operator surfaces. See aragora.metrics.viah for the metric
definition (issue #6067, AGT-06).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aragora.metrics.viah import VIAH_TREND_FLAG, compute_viah, viah_trend_enabled
from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH, ShiftLedger


def cmd_metrics_viah(args: argparse.Namespace) -> int:
    """Compute and print the VIAH report over a window.

    Returns 0 on success; non-zero only on argparse-level usage error
    (which argparse handles itself before reaching here).
    """
    ledger_path: Path | None
    raw_path = getattr(args, "ledger_path", None)
    if raw_path:
        ledger_path = Path(raw_path).expanduser()
    else:
        ledger_path = Path(DEFAULT_LEDGER_PATH)
    ledger = ShiftLedger(path=ledger_path)

    window_hours = float(getattr(args, "window_hours", 168.0))
    cruxes = int(getattr(args, "cruxes_correctly_detected", 0))
    predictions = int(getattr(args, "predictions_above_brier_threshold", 0))
    failed_claims = int(getattr(args, "failed_claims_promoted_without_repair", 0))

    report = compute_viah(
        ledger=ledger,
        window_hours=window_hours,
        cruxes_correctly_detected=cruxes,
        predictions_above_brier_threshold=predictions,
        failed_claims_promoted_without_repair=failed_claims,
    )

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
        return 0

    viah_str = "n/a" if report.viah is None else f"{report.viah:.3f}"
    print(f"VIAH: {viah_str} (window={window_hours:.1f}h, ledger={ledger_path})")
    print(f"  agent_hours:                              {report.agent_hours:.2f}")
    print(f"  merged_autonomous_prs:                    {report.merged_autonomous_prs}")
    print(f"  cruxes_correctly_detected:                {report.cruxes_correctly_detected}")
    print(f"  predictions_above_brier_threshold:        {report.predictions_above_brier_threshold}")
    print(f"  rescues_required:                         {report.rescues_required}")
    print(
        f"  failed_claims_promoted_without_repair:    "
        f"{report.failed_claims_promoted_without_repair}"
    )
    return 0


def cmd_metrics_status(args: argparse.Namespace) -> int:
    """Print the VIAH operator-truth Markdown status report.

    Gated behind ARAGORA_VIAH_TREND_ENABLED (default off). When the flag
    is set, generates a report mirroring docs/status/B0_BENCHMARK_TRUTH_STATUS.md.
    Pass --output to write to disk instead of stdout.
    """
    from aragora.metrics.viah_status import generate_viah_status_report

    if not viah_trend_enabled():
        print(
            f"error: VIAH status surface is disabled; set {VIAH_TREND_FLAG}=1 to enable",
            file=sys.stderr,
        )
        return 1

    raw_path = getattr(args, "ledger_path", None)
    ledger_path = Path(raw_path).expanduser() if raw_path else Path(DEFAULT_LEDGER_PATH)
    ledger = ShiftLedger(path=ledger_path)
    weeks = int(getattr(args, "weeks", 4))

    try:
        report_md = generate_viah_status_report(ledger, weeks=weeks)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path_str = getattr(args, "output", None)
    if output_path_str:
        out_path = Path(output_path_str).expanduser()
        out_path.write_text(report_md, encoding="utf-8")
        print(f"Written to {out_path}")
    else:
        print(report_md)
    return 0


__all__ = ["cmd_metrics_status", "cmd_metrics_viah"]
