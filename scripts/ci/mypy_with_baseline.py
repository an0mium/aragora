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
from importlib import metadata
import re
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
    # Baseline identity must not depend on a developer or runner's prior cache.
    "--no-incremental",
)
MYPY_DIAGNOSTIC_RE = re.compile(r"^[^:\n]+:\d+(?::\d+)?: error:", re.MULTILINE)
TOOL_FAILURE = 2
EXPECTED_TOOLCHAIN_VERSIONS = {
    "mypy": "2.3.0",
    "mypy-baseline": "0.7.4",
}


def _validate_toolchain_versions() -> str | None:
    for distribution, expected in EXPECTED_TOOLCHAIN_VERSIONS.items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return f"required distribution {distribution}=={expected} is not installed"
        if actual != expected:
            return f"required {distribution}=={expected}, found {actual}"
    return None


def _run_mypy(mypy_args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "mypy", *mypy_args]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def _filter(mypy_output: str) -> int:
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
    return subprocess.run(
        cmd,
        input=mypy_output,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    ).returncode


def _sync(mypy_output: str) -> int:
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
    return subprocess.run(
        cmd,
        input=mypy_output,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    ).returncode


def _report_tool_failure(message: str, output: str = "") -> int:
    if output:
        sys.stderr.write(output)
        if not output.endswith("\n"):
            sys.stderr.write("\n")
    print(f"typecheck tool failure - failing closed: {message}", file=sys.stderr)
    return TOOL_FAILURE


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

    toolchain_error = _validate_toolchain_versions()
    if toolchain_error is not None:
        return _report_tool_failure(toolchain_error)

    try:
        mypy_result = _run_mypy(mypy_args)
    except OSError as exc:
        return _report_tool_failure(f"could not start mypy: {exc}")

    output = mypy_result.stdout or ""
    if mypy_result.returncode not in {0, 1}:
        return _report_tool_failure(
            f"mypy exited with unexpected status {mypy_result.returncode}",
            output,
        )
    if mypy_result.returncode == 1 and MYPY_DIAGNOSTIC_RE.search(output) is None:
        return _report_tool_failure(
            "mypy exited 1 without recognized file:line[:column]: error diagnostics",
            output,
        )

    if args.sync:
        return _sync(output)
    return _filter(output)


if __name__ == "__main__":
    sys.exit(main())
