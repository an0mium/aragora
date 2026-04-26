#!/usr/bin/env python3
"""Executable claim verification CLI (DIC-14 / #6024).

Scans *.yaml manifests in a claims directory, verifies each claim via
ClaimVerifier, and emits a JSON report.  Explicit invocation is always
permitted; automatic production-path loading via
``load_claims_from_dir`` is separately gated behind
``ARAGORA_EPISTEMIC_CLAIMS_ENABLED`` (default off).

Exit codes: 0 = ok/skipped, 1 = fail/stale/error (with --exit-code), 2 = fatal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UNHEALTHY = frozenset({"fail", "stale", "error"})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify DIC-14 executable claim manifests.")
    p.add_argument(
        "--claims-dir", type=Path, default=_REPO_ROOT / "docs" / "status" / "claims", metavar="DIR"
    )
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT, metavar="DIR")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip command execution; return UNSUPPORTED for command claims.",
    )
    p.add_argument("--output", type=Path, default=None, metavar="FILE")
    p.add_argument(
        "--exit-code", action="store_true", help="Exit 1 when any claim is fail, stale, or error."
    )
    return p.parse_args(argv)


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    if output is not None:
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from aragora.epistemic.claim_verifier import ClaimStatus, ClaimVerifier
    except ImportError as exc:
        print(f"error: cannot import aragora.epistemic.claim_verifier: {exc}", file=sys.stderr)
        return 2

    claims_dir: Path = args.claims_dir
    if not claims_dir.is_dir():
        print(f"error: claims directory not found: {claims_dir}", file=sys.stderr)
        return 2

    manifest_paths = sorted(claims_dir.glob("*.yaml"))
    empty_summary = {s.value: 0 for s in ClaimStatus}

    if not manifest_paths:
        print(f"warning: no *.yaml manifests found in {claims_dir}", file=sys.stderr)
        _emit(
            {"schema_version": 1, "manifests_scanned": 0, "results": [], "summary": empty_summary},
            args.output,
        )
        return 0

    verifier = ClaimVerifier(repo_root=args.repo_root, dry_run=args.dry_run)
    result_dicts: list[dict[str, Any]] = []
    summary = dict(empty_summary)

    for path in manifest_paths:
        try:
            results = verifier.verify_manifest(path)
        except Exception as exc:  # noqa: BLE001
            result_dicts.append(
                {
                    "claim_id": f"<manifest:{path.name}>",
                    "status": "error",
                    "message": f"failed to load manifest: {exc}",
                    "severity": "warning",
                    "allowed_action": "report_only",
                    "elapsed_ms": 0.0,
                    "detail": {},
                }
            )
            summary["error"] += 1
            continue
        for r in results:
            result_dicts.append(r.to_dict())
            summary[r.status.value] += 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifests_scanned": len(manifest_paths),
        "results": result_dicts,
        "summary": summary,
    }
    _emit(payload, args.output)

    if args.exit_code and any(summary.get(s, 0) > 0 for s in _UNHEALTHY):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
