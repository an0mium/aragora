#!/usr/bin/env python3
"""Diagnostic surface for the codex automation publisher.

Joins three independent signals into a single readiness verdict:

1. launchd job loaded? (via ``launchctl print``)
2. status cache age (mtime of ``.aragora/automation-github-status/latest.json``)
3. outbox-vs-cache drift (does the cache's ``outbox_count`` agree with the
   real file count under ``.aragora/automation-outbox``?)

The previous-task contract requires all three to be healthy before opening
new product lanes. Until this script existed, an operator had to run three
different commands and join the results manually.

Examples
--------
$ python3 scripts/publisher_freshness_check.py
publisher: degraded (launchd: not loaded; cache: 48.5h stale; drift: outbox=18 cache=18)
$ python3 scripts/publisher_freshness_check.py --json
{"verdict": "degraded", ...}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAUNCHD_LABEL = "com.aragora.codex-automation-publisher"
DEFAULT_CACHE_PATH = Path(".aragora/automation-github-status/latest.json")
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")


@dataclass
class FreshnessReport:
    verdict: str
    summary: str
    launchd_loaded: bool
    launchd_detail: str
    launchd_last_exit_code: int | None
    cache_path: str
    cache_present: bool
    cache_age_seconds: float | None
    cache_age_human: str
    cache_stale: bool
    cache_stale_threshold_seconds: int
    outbox_dir: str
    outbox_real_count: int
    outbox_cache_count: int | None
    outbox_drift: bool
    drift_detail: str
    blockers: list[str] = field(default_factory=list)


# Map of common launchd exit codes to human-readable labels (sysexits.h subset).
# Reported alongside the raw integer so the operator surface explains *why*
# `last exit code = 78` is bad without requiring them to know sysexits.h.
_LAUNCHD_EXIT_CODE_NAMES: dict[int, str] = {
    64: "EX_USAGE",
    65: "EX_DATAERR",
    66: "EX_NOINPUT",
    67: "EX_NOUSER",
    68: "EX_NOHOST",
    69: "EX_UNAVAILABLE",
    70: "EX_SOFTWARE",
    71: "EX_OSERR",
    72: "EX_OSFILE",
    73: "EX_CANTCREAT",
    74: "EX_IOERR",
    75: "EX_TEMPFAIL",
    76: "EX_PROTOCOL",
    77: "EX_NOPERM",
    78: "EX_CONFIG",
    127: "command-not-found",
    137: "SIGKILL",
}


def _parse_last_exit_code(launchctl_print_stdout: str) -> int | None:
    """Extract ``last exit code = N`` from ``launchctl print`` output.

    Returns ``None`` when the field is absent (e.g., the job has never run, or
    the launchctl output format changes).  Robust to whitespace and surrounding
    fields.
    """
    for line in launchctl_print_stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("last exit code"):
            # Form: "last exit code = N: NAME" or "last exit code = N".
            _, _, rest = stripped.partition("=")
            rest = rest.strip()
            if not rest:
                return None
            num_str = rest.split(":", 1)[0].strip()
            try:
                return int(num_str)
            except ValueError:
                return None
    return None


def _launchd_loaded(label: str) -> tuple[bool, str, int | None]:
    """Return (loaded, detail, last_exit_code).

    A launchd job that is *loaded* may still be persistently failing — the
    plist on disk can hard-code a missing WorkingDirectory or a bad command.
    Surfacing the last-known exit code lets the caller distinguish "loaded
    and healthy" from "loaded but persistently exit-code-failing" — the
    latter is strictly more degraded.
    """
    try:
        domain = f"gui/{os.getuid()}"
    except AttributeError:
        return False, "non-posix-platform", None
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, "launchctl-not-found", None
    except subprocess.TimeoutExpired:
        return False, "launchctl-timeout", None
    last_exit = _parse_last_exit_code(proc.stdout or "")
    if proc.returncode == 0:
        return True, "loaded", last_exit
    stderr = (proc.stderr or "").strip().splitlines()
    detail = stderr[-1] if stderr else f"rc={proc.returncode}"
    return False, detail, last_exit


def _human_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _count_outbox_files(outbox_dir: Path) -> int:
    if not outbox_dir.is_dir():
        return 0
    return sum(1 for p in outbox_dir.iterdir() if p.is_file() and p.suffix == ".json")


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _same_git_origin(left: Path, right: Path) -> bool:
    left_proc = _run_git(["config", "--get", "remote.origin.url"], cwd=left)
    right_proc = _run_git(["config", "--get", "remote.origin.url"], cwd=right)
    if left_proc.returncode != 0 or right_proc.returncode != 0:
        return False
    return bool(left_proc.stdout.strip()) and left_proc.stdout.strip() == right_proc.stdout.strip()


def _automation_state_root(repo_root: Path) -> Path:
    """Return the checkout whose shared .aragora state should back publisher checks."""

    if (repo_root / ".aragora").is_dir():
        return repo_root

    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    candidates: list[tuple[Path, bool]] = []
    if configured:
        candidates.append((Path(configured).expanduser(), True))
    candidates.append((Path.home() / "Development" / "aragora", False))

    for candidate, explicit in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.name == ".aragora" and resolved.is_dir():
            return resolved
        if not (resolved / ".aragora").is_dir():
            continue
        if explicit or _same_git_origin(repo_root, resolved):
            return resolved
    return repo_root


def _automation_state_default_path(state_root: Path, default_relative: Path) -> Path:
    expanded = state_root.expanduser()
    if default_relative.parts[:1] == (".aragora",) and expanded.name == ".aragora":
        return expanded.joinpath(*default_relative.parts[1:])
    return expanded / default_relative


def _read_cache_outbox_count(cache_path: Path) -> int | None:
    try:
        with cache_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    local_queue = payload.get("local_queue")
    if not isinstance(local_queue, dict):
        return None
    raw = local_queue.get("outbox_count")
    return int(raw) if isinstance(raw, int) else None


def evaluate(
    repo_root: Path,
    *,
    cache_path: Path | None = None,
    outbox_dir: Path | None = None,
    state_root: Path | None = None,
    stale_threshold_seconds: int = 1800,
    now: float | None = None,
) -> FreshnessReport:
    default_state_root = (
        state_root.expanduser() if state_root is not None else _automation_state_root(repo_root)
    )
    if cache_path is None:
        cache_path = _automation_state_default_path(default_state_root, DEFAULT_CACHE_PATH)
    if outbox_dir is None:
        outbox_dir = _automation_state_default_path(default_state_root, DEFAULT_OUTBOX_DIR)
    now = now if now is not None else time.time()

    loaded, detail, last_exit_code = _launchd_loaded(LAUNCHD_LABEL)
    # A non-zero last exit code on a *loaded* job is a strictly worse state
    # than "loaded healthy": the plist is installed but the program is
    # consistently failing.  Caught in Round 2026-04-30c Phase B (launchd
    # plist pointing at a missing WorkingDirectory after a worktree rename
    # caused 375 consecutive EX_CONFIG=78 fires before being noticed).
    launchd_failing = loaded and last_exit_code is not None and last_exit_code != 0

    cache_present = cache_path.is_file()
    cache_age: float | None
    if cache_present:
        cache_age = max(0.0, now - cache_path.stat().st_mtime)
    else:
        cache_age = None
    cache_stale = (cache_age is None) or (cache_age > stale_threshold_seconds)

    outbox_real = _count_outbox_files(outbox_dir)
    outbox_cache = _read_cache_outbox_count(cache_path) if cache_present else None
    # A stale cache will *necessarily* disagree with the live outbox (the
    # outbox has changed since the snapshot was taken). Reporting drift in
    # that case is double-flagging: ``cache: Nh stale`` already explains the
    # disagreement. Only treat drift as a real signal when the cache is
    # fresh and the counts still don't match — that is the actual operator-
    # actionable case (e.g. publisher writing the wrong count).
    drift_meaningful = outbox_cache is not None and outbox_cache != outbox_real and not cache_stale
    drift = outbox_cache is not None and outbox_cache != outbox_real
    if outbox_cache is None:
        drift_detail = f"outbox={outbox_real} cache=missing"
    elif drift:
        drift_detail = f"outbox={outbox_real} cache={outbox_cache}"
    else:
        drift_detail = f"outbox={outbox_real} cache={outbox_cache}"

    blockers: list[str] = []
    if not loaded:
        blockers.append(f"launchd: {detail}")
    elif launchd_failing:
        # Loaded but persistently failing: surface the exit code with name.
        name = _LAUNCHD_EXIT_CODE_NAMES.get(last_exit_code or -1, "unknown")
        blockers.append(f"launchd: exit_code={last_exit_code} ({name})")
    if not cache_present:
        blockers.append("cache: missing")
    elif cache_stale:
        blockers.append(f"cache: {_human_age(cache_age)} stale")
    if drift_meaningful:
        blockers.append(f"drift: {drift_detail}")

    if not blockers:
        verdict = "ready"
    elif loaded and not launchd_failing and not drift_meaningful:
        # cache-stale alone (without meaningful drift, with healthy launchd)
        # is "warming": the next publisher run will refresh the cache and
        # the operator signal will resolve without intervention.
        verdict = "warming"
    else:
        verdict = "degraded"

    summary_parts = []
    if loaded and not launchd_failing:
        summary_parts.append("launchd: loaded")
    elif loaded and launchd_failing:
        name = _LAUNCHD_EXIT_CODE_NAMES.get(last_exit_code or -1, "unknown")
        summary_parts.append(f"launchd: loaded but failing exit_code={last_exit_code} ({name})")
    else:
        summary_parts.append(f"launchd: {detail}")
    summary_parts.append(f"cache: {_human_age(cache_age) if cache_present else 'missing'}")
    summary_parts.append(f"drift: {drift_detail}")
    summary = f"publisher: {verdict} ({'; '.join(summary_parts)})"

    return FreshnessReport(
        verdict=verdict,
        summary=summary,
        launchd_loaded=loaded,
        launchd_detail=detail,
        launchd_last_exit_code=last_exit_code,
        cache_path=str(cache_path),
        cache_present=cache_present,
        cache_age_seconds=cache_age,
        cache_age_human=_human_age(cache_age),
        cache_stale=cache_stale,
        cache_stale_threshold_seconds=stale_threshold_seconds,
        outbox_dir=str(outbox_dir),
        outbox_real_count=outbox_real,
        outbox_cache_count=outbox_cache,
        outbox_drift=drift,
        drift_detail=drift_detail,
        blockers=blockers,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default=os.getcwd(), help="Path inside the target repository")
    parser.add_argument(
        "--cache-path",
        "--status-cache",
        dest="cache_path",
        default=None,
        help="Path to the publisher status cache JSON",
    )
    parser.add_argument("--outbox-dir", default=None)
    parser.add_argument(
        "--state-root",
        default=None,
        help=(
            "Checkout or .aragora directory that owns shared automation state. "
            "Defaults to ARAGORA_AUTOMATION_STATE_ROOT, then ~/Development/aragora "
            "when it has the same git origin as --repo."
        ),
    )
    parser.add_argument(
        "--stale-threshold-seconds",
        type=int,
        default=1800,
        help="Cache mtime older than this is considered stale (default: 1800s = 30min)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON to stdout instead of the 1-line summary"
    )
    parser.add_argument(
        "--exit-nonzero-on-degraded",
        action="store_true",
        help="Exit 1 when verdict is degraded (useful in CI/launchd post-flight)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(args.repo).resolve()
    cache_path = Path(args.cache_path).resolve() if args.cache_path else None
    outbox_dir = Path(args.outbox_dir).resolve() if args.outbox_dir else None
    state_root = Path(args.state_root).expanduser().resolve() if args.state_root else None
    report = evaluate(
        repo_root,
        cache_path=cache_path,
        outbox_dir=outbox_dir,
        state_root=state_root,
        stale_threshold_seconds=args.stale_threshold_seconds,
    )
    if args.json:
        payload: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            **asdict(report),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(report.summary)
        if report.blockers and not args.json:
            for blocker in report.blockers:
                print(f"  - {blocker}", file=sys.stderr)
    if args.exit_nonzero_on_degraded and report.verdict == "degraded":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
