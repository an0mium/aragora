"""CLI command: ``aragora metrics viah``.

Reads the ShiftLedger and prints the latest VIAH (verifiable improvements
per agent-hour) report. See aragora.metrics.viah for the metric
definition (issue #6067).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aragora.metrics.viah import compute_viah
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


__all__ = ["cmd_metrics_viah"]
