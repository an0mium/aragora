"""CLI command: ``aragora truth-map``.

DIC-18 operator surface for the organizational truth map (issue #6028).

Reads DIC-13 claim manifests from the canonical claims directory,
verifies them via the DIC-14 ClaimVerifier (dry-run by default — no
subprocess execution), and emits a read-only report.

Flag: ``ARAGORA_TRUTH_MAP_ENABLED`` (default OFF).
Live queue effect: none — read-only operator surface.
Advances: issue #6028 (DIC-18).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_FLAG = "ARAGORA_TRUTH_MAP_ENABLED"
_DEFAULT_CLAIMS_DIR = "docs/status/claims"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").lower() in {"1", "true", "yes", "on"}


def cmd_truth_map(args: argparse.Namespace) -> int:
    """Run the organizational truth map report."""
    if not _flag_enabled():
        print(
            f"error: {_FLAG} is not set; set it to '1' to enable truth-map",
            file=sys.stderr,
        )
        return 1

    claims_dir = Path(getattr(args, "claims_dir", _DEFAULT_CLAIMS_DIR)).expanduser()
    if not claims_dir.is_dir():
        print(f"error: claims directory not found: {claims_dir}", file=sys.stderr)
        return 1

    manifest_paths = sorted(claims_dir.glob("*.yaml"))
    as_json: bool = getattr(args, "json", False)

    from aragora.epistemic.truth_map import build_truth_map_from_manifests

    report = build_truth_map_from_manifests(
        manifest_paths=manifest_paths,
        dry_run=True,
    )

    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_text_report(report)

    return 1 if (report.failing_claims + report.error_claims) > 0 else 0


def _print_text_report(report) -> None:
    """Emit a human-readable truth map to stdout."""
    print(f"Organizational Truth Map  (generated: {report.generated_at})")
    print(
        f"  Claims: {report.total_claims} total  "
        f"pass={report.passing_claims}  "
        f"fail={report.failing_claims}  "
        f"stale={report.stale_claims}  "
        f"unsupported={report.unsupported_claims}  "
        f"error={report.error_claims}"
    )
    if report.open_crux_count:
        print(f"  Open cruxes: {report.open_crux_count}")
    if report.claims:
        print()
        for row in report.claims:
            marker = "PASS" if row.status == "pass" else row.status.upper()
            print(f"  [{marker:11s}] {row.claim_id:40s}  owner={row.owner}")
    if report.crux_summaries:
        print()
        print("  Crux summaries:")
        for cs in report.crux_summaries:
            print(f"    debate={cs.debate_id}  cruxes={cs.crux_count}  open={cs.open_cruxes}")
