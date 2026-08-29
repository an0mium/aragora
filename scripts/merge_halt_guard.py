#!/usr/bin/env python3
"""Compatibility shim: the merge halt guard now lives in the package.

It moved to :mod:`aragora.governance.merge_halt` because merge-capable code
exists in ``aragora/`` too (``swarm/merge_arbiter.py``, ``missions/live_gate.py``),
and package modules cannot import ``scripts.*`` — the editable install maps only
``aragora``, so a ``from scripts...`` import inside the package dies at runtime.

Operator scripts keep importing ``scripts.merge_halt_guard`` and the CLI entry
point (``python3 scripts/merge_halt_guard.py``) keeps working.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if str(_Path(__file__).resolve().parent.parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from aragora.governance.merge_halt import (  # noqa: F401,E402
    DEFAULT_HALT_FILE,
    DEFAULT_WAIVER_FILE,
    MERGE_CAPABLE_SCRIPTS,
    NON_MERGE_MENTIONS,
    SHARED_REPO_ROOT,
    HaltDecision,
    MergeHalted,
    assert_merge_allowed,
    evaluate,
)


def main(argv: list[str] | None = None) -> int:
    """CLI lives here, not in the library: package code must not print (T201)."""
    parser = argparse.ArgumentParser(description="Check the merge halt for one PR head.")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--halt-file", type=Path, default=DEFAULT_HALT_FILE)
    parser.add_argument("--waiver-file", type=Path, default=DEFAULT_WAIVER_FILE)
    args = parser.parse_args(argv)

    decision = evaluate(
        args.pr, args.head_sha, halt_file=args.halt_file, waiver_file=args.waiver_file
    )
    print(("ALLOW: " if decision.allowed else "BLOCK: ") + decision.reason)
    return 0 if decision.allowed else 3


if __name__ == "__main__":
    raise SystemExit(main())
