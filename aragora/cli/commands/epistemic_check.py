"""CLI command: ``aragora epistemic-check``.

Loads claim manifests from ``docs/status/claims/`` (or a path you supply)
and runs the DIC-14 ClaimVerifier against them, emitting a JSON or
human-readable status report.  No queue mutation, no issue creation.

Read-only by default: manifest-provided ``command``-kind verifications are
**not** executed unless the operator opts in with ``--execute``.  Without
``--execute`` the verifier runs in dry-run mode and reports command-kind
claims as UNSUPPORTED, so pointing the CLI at an untrusted manifest cannot
run arbitrary subprocesses.  ``--execute`` runs those commands with the
caller's shell privileges and should only be used for trusted manifests.

Flag-gated: set ``ARAGORA_EPISTEMIC_CLAIMS_ENABLED=1`` to enable command
execution.  When the flag is not set the command exits 0 and prints a
brief reminder — so CI jobs and operator scripts can call it unconditionally
without breaking.

Advances: issue #6024 (DIC-14 — claim verification runner CLI surface).
Live queue effect: none.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_FLAG = "ARAGORA_EPISTEMIC_CLAIMS_ENABLED"
_DEFAULT_CLAIMS_DIR = Path("docs/status/claims")


def _enabled() -> bool:
    return str(os.environ.get(_FLAG) or "").strip().lower() in {"1", "true", "yes", "on"}


def _render_text(results: list) -> str:
    """Return a compact human-readable status table."""
    from aragora.epistemic.claim_verifier import ClaimStatus

    if not results:
        return "No claims found."

    width_id = max(len(r.claim_id) for r in results)
    lines: list[str] = [f"{'CLAIM_ID':<{width_id}}  STATUS       SEVERITY"]
    lines.append("-" * (width_id + 28))
    for r in results:
        lines.append(f"{r.claim_id:<{width_id}}  {r.status.value:<12} {r.severity}")

    counts = {s.value: sum(1 for r in results if r.status == s) for s in ClaimStatus}
    lines.append("")
    parts = [f"{v} {k}" for k, v in counts.items() if v]
    lines.append("Summary: " + ", ".join(parts))
    return "\n".join(lines)


def cmd_epistemic_check(args: argparse.Namespace) -> int:
    """Entry point for ``aragora epistemic-check``."""
    if not _enabled():
        print(
            f"epistemic-check: skipped (set {_FLAG}=1 to enable)",
            file=sys.stderr,
        )
        return 0

    from aragora.epistemic.claim_verifier import ClaimStatus, ClaimVerifier

    target = Path(args.path).expanduser() if args.path else _DEFAULT_CLAIMS_DIR
    repo_root = Path(args.repo_root).expanduser() if args.repo_root else Path.cwd()

    # Read-only by default: only run manifest-provided commands when the
    # operator explicitly opts in with --execute. --dry-run is accepted for
    # explicitness/back-compat but is already the default. This keeps the
    # documented "read-only" invariant true even for untrusted manifests.
    execute = bool(getattr(args, "execute", False))
    dry_run = True if not execute else bool(getattr(args, "dry_run", False))

    verifier = ClaimVerifier(repo_root=repo_root, dry_run=dry_run)

    if target.is_file():
        manifest_files = [target]
    elif target.is_dir():
        manifest_files = sorted(target.glob("*.yaml"))
    else:
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 1

    if not manifest_files:
        print(f"No *.yaml manifests found under {target}", file=sys.stderr)
        return 0

    all_results = []
    for mf in manifest_files:
        try:
            all_results.extend(verifier.verify_manifest(mf))
        except Exception as exc:  # noqa: BLE001
            print(f"error loading {mf}: {exc}", file=sys.stderr)
            return 1

    if args.json:
        # Call on the instance so tests can inject a replacement class.
        print(verifier.report_json(all_results))
    else:
        print(_render_text(all_results))

    # Exit 1 only if a *blocking* claim failed or errored.
    blocking_failures = [
        r
        for r in all_results
        if r.status in (ClaimStatus.FAIL, ClaimStatus.ERROR) and r.severity == "blocking"
    ]
    return 1 if blocking_failures else 0
