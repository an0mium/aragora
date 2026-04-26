#!/usr/bin/env python3
"""Run mypy and filter its output through mypy-baseline.

Purpose
-------
Aragora has ~4,100 pre-existing mypy errors (see ``.mypy-baseline``). Failing
the pre-push hook on those known errors makes the gate useless -- every push
fails and automations resort to ``--no-verify``. This wrapper preserves the
hook's value by baselining existing debt and surfacing only *new* errors.

Usage
-----
Invoked from ``.pre-commit-config.yaml``. All arguments are forwarded to
mypy. Output of mypy is piped through ``mypy-baseline filter`` which removes
lines present in ``.mypy-baseline`` (the committed debt snapshot) and exits
non-zero when new errors are introduced.

Exit codes
----------
Exit code matches ``mypy-baseline filter``:
  * 0 -- no new errors, no unexpectedly fixed baseline entries
  * >0 -- new errors introduced (hook fails, pushing author must fix)

We pass ``--allow-unsynced`` so that *accidentally* fixing a baselined error
does not fail the push; the baseline is resynced explicitly via
``python scripts/ci/mypy_with_baseline.py --sync``.

Sync mode
---------
``python scripts/ci/mypy_with_baseline.py --sync`` regenerates
``.mypy-baseline`` from a fresh mypy run. Use after landing a PR that
intentionally clears a batch of existing errors.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / ".mypy-baseline"
DEFAULT_MYPY_ARGS: tuple[str, ...] = (
    "aragora/",
    "scripts/",
    "--config-file=pyproject.toml",
    "--ignore-missing-imports",
)


def _run_mypy(mypy_args: tuple[str, ...]) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, "-m", "mypy", *mypy_args]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )


def _filter(mypy_proc: subprocess.Popen[bytes]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "mypy_baseline",
        "filter",
        "--baseline-path",
        str(BASELINE_PATH),
        "--no-colors",
        "--allow-unsynced",
        # Notes are flaky: mypy emits overload/assignment hints whose wording
        # is not stable across runs (e.g. "__init__" vs "dict" in dict
        # overloads). We baseline them out here too so they do not register
        # as new violations.
        "--ignore-categories",
        "note",
    ]
    assert mypy_proc.stdout is not None
    filter_proc = subprocess.Popen(cmd, stdin=mypy_proc.stdout, cwd=str(REPO_ROOT))
    mypy_proc.stdout.close()
    filter_rc = filter_proc.wait()
    mypy_proc.wait()
    return filter_rc


def _sync(mypy_proc: subprocess.Popen[bytes]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "mypy_baseline",
        "sync",
        "--baseline-path",
        str(BASELINE_PATH),
        "--no-colors",
        "--sort-baseline",
        "--ignore-categories",
        "note",
    ]
    assert mypy_proc.stdout is not None
    sync_proc = subprocess.Popen(cmd, stdin=mypy_proc.stdout, cwd=str(REPO_ROOT))
    mypy_proc.stdout.close()
    sync_rc = sync_proc.wait()
    mypy_proc.wait()
    return sync_rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run mypy and filter through mypy-baseline.",
        add_help=True,
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Regenerate .mypy-baseline from a fresh mypy run and exit 0.",
    )
    parser.add_argument(
        "mypy_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to mypy. Defaults to 'aragora/ scripts/ "
        "--config-file=pyproject.toml --ignore-missing-imports'.",
    )
    args = parser.parse_args(argv)

    raw_args = tuple(a for a in args.mypy_args if a != "--")
    mypy_args = raw_args or DEFAULT_MYPY_ARGS

    mypy_proc = _run_mypy(mypy_args)
    if args.sync:
        return _sync(mypy_proc)
    return _filter(mypy_proc)


if __name__ == "__main__":
    sys.exit(main())
