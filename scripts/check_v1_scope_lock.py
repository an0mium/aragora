#!/usr/bin/env python3
"""Enforce Aragora V1 scope lock for pull requests.

Blocks changes in explicitly out-of-scope feature areas unless
`ARAGORA_ALLOW_SCOPE_EXPANSION=1` is set.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BLOCKED_PREFIXES = (
    "aragora/server/handlers/social/",
    "aragora/server/handlers/features/advertising.py",
    "aragora/server/handlers/features/analytics_platforms.py",
    "aragora/server/handlers/features/cross_platform_analytics.py",
    "aragora/server/handlers/features/crm.py",
    "aragora/server/handlers/features/ecommerce.py",
    "aragora/server/handlers/features/support.py",
    "aragora/server/handlers/features/pulse.py",
    "aragora/server/handlers/features/broadcast.py",
    "aragora/server/handlers/features/plugins.py",
    "aragora/blockchain/contracts/staking.py",
)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_base_ref() -> str:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        return f"origin/{base_ref}"
    return "origin/main"


def _detect_head_ref() -> str:
    return os.environ.get("GITHUB_SHA") or "HEAD"


def _changed_files(base_ref: str, head_ref: str) -> list[str]:
    cmd = [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        f"{base_ref}...{head_ref}",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to compute changed files ({proc.returncode}): {proc.stderr.strip()}"
        )
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return files


def _scope_violations(files: list[str]) -> list[str]:
    violations: list[str] = []
    for path in files:
        if path.startswith("docs/deprecated/"):
            continue
        if any(path.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            violations.append(path)
    return sorted(set(violations))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default=_detect_base_ref(), help="Base git ref (default: origin/main)"
    )
    parser.add_argument(
        "--head", default=_detect_head_ref(), help="Head git ref (default: HEAD/GITHUB_SHA)"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit changed files (bypasses git diff; used for tests)",
    )
    parser.add_argument(
        "--lock-file",
        default="docs/status/V1_EXECUTION_LOCK.md",
        help="Path to V1 scope lock doc",
    )
    args = parser.parse_args(argv)

    if _is_truthy(os.environ.get("ARAGORA_ALLOW_SCOPE_EXPANSION")):
        print("V1 scope lock gate bypassed via ARAGORA_ALLOW_SCOPE_EXPANSION")
        return 0

    lock_file = Path(args.lock_file)
    if not lock_file.exists():
        print(
            f"V1 scope lock file missing: {lock_file}. "
            "Create docs/status/V1_EXECUTION_LOCK.md before merging.",
            file=sys.stderr,
        )
        return 1

    files = (
        list(args.files)
        if args.files is not None and len(args.files) > 0
        else _changed_files(args.base, args.head)
    )
    if not files:
        print("V1 scope lock gate: no changed files detected")
        return 0

    violations = _scope_violations(files)
    if violations:
        print("V1 scope lock violation: out-of-scope files changed", file=sys.stderr)
        for path in violations:
            print(f"  - {path}", file=sys.stderr)
        print(
            "If scope expansion is intentional, set ARAGORA_ALLOW_SCOPE_EXPANSION=1 "
            "and document rationale in PR description.",
            file=sys.stderr,
        )
        return 1

    print(f"V1 scope lock gate passed ({len(files)} changed files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
