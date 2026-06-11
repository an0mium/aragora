#!/usr/bin/env python3
"""Fleet sentinel — Dead Man's Signals for the aragora agent fleet.

Steering Leverage Operating Plan v2, Pillar 6 / Phase 0.1
(docs/superpowers/plans/2026-06-10-steering-leverage-operating-plan-v2.md).

The 2026-06-10 steering audit found that every loss in the window —
publisher dead three weeks (auth_ok:false since 2026-05-18), boss metrics
silent ten days, an empty daemon plist, a 217-item outbox — shared one
shape: a cheap signal existed on disk, nothing was contracted to read it,
and the human was the only fallback reader.  This sentinel IS that
contracted reader.

Contract:
  * stdlib only; intended to run from cron/launchd every 10 minutes.
  * Each check returns ``{check, status: ok|breach|unknown, detail}``.
  * ``--json`` emits one JSON object
    ``{generated_at, checks, breaches, blind_checks}``.
  * Every run appends one line to the JSONL ledger.
  * Exit codes: 0 all ok; 1 any breach; 2 any errored/unknown check.
    Unknown takes precedence over breach — a sentinel that cannot see is
    worse than one that sees a fire.  Silence is never success.
  * A breach additionally invokes ``--notify-cmd`` (template; ``{summary}``
    placeholder is replaced with the one-line breach summary).
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

CheckResult = dict[str, Any]

ALL_CHECKS = (
    "publisher_status",
    "boss_metrics_heartbeat",
    "launchd_plists",
    "gh_auth",
    "checkout_invariant",
    "outbox_depth",
    "disk_free",
    "lane_liveness",
    "github_api_health",
)

# Branch namespaces owned by autonomous lanes; only these are eligible for the
# ledger-less orphan-branch sweep (failure class A, 2026-06-10/11: coordinator
# lanes died at setup leaving empty elves/run-* branches nobody noticed).
ORPHAN_BRANCH_PATTERNS = ("elves/*", "aragora/boss*")


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (``Z`` suffix accepted) to aware UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _result(check: str, status: str, detail: str) -> CheckResult:
    return {"check": check, "status": status, "detail": detail}


def _age_hours(path: Path, now: datetime) -> float:
    return (now.timestamp() - path.stat().st_mtime) / 3600.0


# ---------------------------------------------------------------------------
# Checks (each parameterized so tests inject fixtures; no live state touched)
# ---------------------------------------------------------------------------


def check_publisher_status(path: Path, *, max_age_hours: float, now: datetime) -> CheckResult:
    """Publisher status file must be fresh AND report healthy GitHub auth."""
    name = "publisher_status"
    if not path.exists():
        return _result(name, "breach", f"missing: {path} — publisher never reported")
    problems: list[str] = []
    age = _age_hours(path, now)
    if age > max_age_hours:
        problems.append(f"stale: mtime {age:.1f}h old (max {max_age_hours}h)")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"unreadable status JSON ({exc.__class__.__name__})")
        payload = {}
    health = payload.get("github_health") or {}
    if health.get("auth_ok") is not True:
        problems.append(f"github_health.auth_ok is {health.get('auth_ok')!r} (expected true)")
    generated_at = payload.get("generated_at")
    if generated_at:
        try:
            gen_age = (now - parse_iso(str(generated_at))).total_seconds() / 3600.0
            if gen_age > max_age_hours:
                problems.append(
                    f"generated_at {generated_at} is {gen_age:.0f}h old (max {max_age_hours}h)"
                )
        except ValueError:
            problems.append(f"unparseable generated_at: {generated_at!r}")
    if problems:
        return _result(name, "breach", "; ".join(problems))
    return _result(name, "ok", f"fresh ({age:.1f}h) and auth_ok")


def check_boss_metrics(path: Path, *, max_age_hours: float, now: datetime) -> CheckResult:
    """Boss metrics heartbeat: the JSONL must have been written recently."""
    name = "boss_metrics_heartbeat"
    if not path.exists():
        return _result(name, "breach", f"missing: {path} — heartbeat never written")
    age = _age_hours(path, now)
    if age > max_age_hours:
        return _result(
            name, "breach", f"heartbeat stale: mtime {age:.1f}h old (max {max_age_hours}h)"
        )
    return _result(name, "ok", f"heartbeat {age:.1f}h old")


def check_launchd_plists(launch_agents_dir: Path) -> CheckResult:
    """Every com.aragora.*.plist must be non-empty and plist-parseable.

    Motivating incident: the zero-byte boss-loop plist of 2026-06-10, which
    launchd silently refused to load while everything looked installed.
    """
    name = "launchd_plists"
    if not launch_agents_dir.is_dir():
        return _result(name, "ok", f"no LaunchAgents dir at {launch_agents_dir}")
    bad: list[str] = []
    plists = sorted(launch_agents_dir.glob("com.aragora.*.plist"))
    for plist in plists:
        if plist.stat().st_size == 0:
            bad.append(f"{plist.name}: zero-byte")
            continue
        try:
            with plist.open("rb") as fh:
                plistlib.load(fh)
        except Exception as exc:  # noqa: BLE001 - plistlib raises various types
            bad.append(f"{plist.name}: unparseable ({exc.__class__.__name__})")
    if bad:
        return _result(name, "breach", "; ".join(bad))
    return _result(name, "ok", f"{len(plists)} aragora plist(s) valid")


def _default_command_runner(cmd: list[str]) -> int:
    proc = subprocess.run(  # noqa: S603 - operator-configured command
        cmd, capture_output=True, text=True, timeout=60
    )
    return proc.returncode


def check_gh_auth(
    *, runner: Callable[[list[str]], int], cmd: list[str] | None = None
) -> CheckResult:
    """``gh auth status`` must exit zero — the publisher dies without it."""
    name = "gh_auth"
    command = cmd or ["gh", "auth", "status"]
    try:
        rc = runner(command)
    except Exception as exc:  # noqa: BLE001 - any runner failure means we are blind
        return _result(
            name, "unknown", f"could not run {command[0]}: {exc.__class__.__name__}: {exc}"
        )
    if rc != 0:
        return _result(name, "breach", f"{' '.join(command)} exited {rc}")
    return _result(name, "ok", "gh auth healthy")


def _default_branch_reader(repo: Path) -> str:
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return proc.stdout.strip()


def check_checkout_invariant(repo: Path, *, branch_reader: Callable[[Path], str]) -> CheckResult:
    """The root checkout must stay on main (worktrees carry branches)."""
    name = "checkout_invariant"
    try:
        branch = branch_reader(repo)
    except Exception as exc:  # noqa: BLE001 - git failure means we are blind
        return _result(name, "unknown", f"could not read branch: {exc.__class__.__name__}: {exc}")
    if branch != "main":
        return _result(name, "breach", f"root checkout {repo} is on {branch!r}, not main")
    return _result(name, "ok", "root checkout on main")


def check_outbox(
    outbox_dir: Path, *, max_items: int, max_age_days: float, now: datetime
) -> CheckResult:
    """Outbox depth and oldest-item age (archive/ excluded)."""
    name = "outbox_depth"
    if not outbox_dir.is_dir():
        return _result(name, "ok", f"no outbox dir at {outbox_dir}")
    items = sorted(p for p in outbox_dir.glob("*.json") if p.is_file())
    problems: list[str] = []
    if len(items) > max_items:
        problems.append(f"{len(items)} items queued (max {max_items})")
    if items:
        oldest = min(items, key=lambda p: p.stat().st_mtime)
        oldest_days = _age_hours(oldest, now) / 24.0
        if oldest_days > max_age_days:
            problems.append(
                f"oldest item {oldest.name} is {oldest_days:.1f}d old (max {max_age_days}d)"
            )
    if problems:
        return _result(name, "breach", "; ".join(problems))
    return _result(name, "ok", f"{len(items)} item(s) queued")


def check_disk_free(
    path: Path, *, min_free_gib: float, usage_fn: Callable[[Path], Any] = shutil.disk_usage
) -> CheckResult:
    name = "disk_free"
    try:
        usage = usage_fn(path)
    except OSError as exc:
        return _result(name, "unknown", f"disk_usage failed: {exc}")
    free_gib = usage.free / 2**30
    if free_gib < min_free_gib:
        return _result(name, "breach", f"{free_gib:.1f} GiB free (min {min_free_gib} GiB)")
    return _result(name, "ok", f"{free_gib:.1f} GiB free")


# ---------------------------------------------------------------------------
# lane_liveness (failure class A — silent lane death, 2026-06-10/11)
# ---------------------------------------------------------------------------


def _default_remote_heads(repo: Path) -> dict[str, str]:
    """``git ls-remote --heads origin`` → ``{branch: sha}``."""
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "ls-remote", "--heads", "origin"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    heads: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        prefix = "refs/heads/"
        if ref.startswith(prefix):
            heads[ref[len(prefix) :].strip()] = sha.strip()
    return heads


def _default_ahead_counter(repo: Path) -> Callable[[str], int | None]:
    """Commits ahead of origin/main for a sha; None when unresolvable.

    A zero-ahead branch tip is by definition an ancestor of origin/main and
    therefore present locally once origin/main is fetched.  An unresolvable
    sha thus means the branch carries unfetched unique commits — callers must
    treat None as "has commits" (never as empty).
    """

    def count(sha: str) -> int | None:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-list", "--count", f"origin/main..{sha}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return None
        try:
            return int(proc.stdout.strip())
        except ValueError:
            return None

    return count


def _default_commit_dater(repo: Path) -> Callable[[str], datetime | None]:
    def commit_date(sha: str) -> datetime | None:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "log", "-1", "--format=%cI", sha],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = proc.stdout.strip()
        if proc.returncode != 0 or not out:
            return None
        try:
            return parse_iso(out)
        except ValueError:
            return None

    return commit_date


def check_lane_liveness(
    lanes_glob: str,
    *,
    lane_max_age_hours: float,
    orphan_age_hours: float,
    now: datetime,
    remote_heads: Callable[[], dict[str, str]],
    ahead_counter: Callable[[str], int | None],
    commit_dater: Callable[[str], datetime | None],
) -> CheckResult:
    """Detect silently dead lanes and ledger-less orphan branches.

    Rule 1 (ledger): a lane-ledger entry (``.aragora/run-*/lanes/<lane>.json``)
    with ``status=in_progress``, ``launched_at`` older than
    ``lane_max_age_hours``, whose branch has zero commits ahead of origin/main
    (or is absent from origin entirely) is a dead lane — it burned its setup
    window and produced nothing durable.

    Rule 2 (orphan sweep): any origin branch matching
    ``ORPHAN_BRANCH_PATTERNS`` with zero commits ahead of origin/main whose
    tip commit is older than ``orphan_age_hours`` is an orphan — a lane died
    before its first commit and left no ledger to read.
    """
    name = "lane_liveness"
    try:
        heads = remote_heads()
    except Exception as exc:  # noqa: BLE001 - git failure means we are blind
        return _result(
            name, "unknown", f"could not list origin heads: {exc.__class__.__name__}: {exc}"
        )
    unreadable: list[str] = []
    problems: list[str] = []
    in_progress = 0
    ledger_files = sorted(glob_module.glob(str(Path(lanes_glob) / "*.json")))
    for ledger_file in ledger_files:
        try:
            entry = json.loads(Path(ledger_file).read_text())
            lane = str(entry["lane"])
            status = str(entry.get("status", ""))
            launched_at = parse_iso(str(entry["launched_at"]))
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            unreadable.append(f"{ledger_file} ({exc.__class__.__name__})")
            continue
        if status != "in_progress":
            continue
        in_progress += 1
        age_hours = (now - launched_at).total_seconds() / 3600.0
        if age_hours <= lane_max_age_hours:
            continue
        branch = str(entry.get("branch", ""))
        sha = heads.get(branch)
        if sha is None:
            problems.append(
                f"lane {lane}: in_progress {age_hours:.1f}h (max {lane_max_age_hours}h), "
                f"branch {branch} absent from origin"
            )
            continue
        if ahead_counter(sha) == 0:
            problems.append(
                f"lane {lane}: in_progress {age_hours:.1f}h (max {lane_max_age_hours}h), "
                f"branch {branch} has zero commits ahead of origin/main"
            )
    if unreadable:
        return _result(name, "unknown", "unreadable lane ledger(s): " + "; ".join(unreadable))
    pattern_branches = 0
    for branch in sorted(heads):
        if not any(fnmatch(branch, pat) for pat in ORPHAN_BRANCH_PATTERNS):
            continue
        pattern_branches += 1
        if ahead_counter(heads[branch]) != 0:
            continue
        tip_date = commit_dater(heads[branch])
        if tip_date is None:
            continue
        tip_age_hours = (now - tip_date).total_seconds() / 3600.0
        if tip_age_hours > orphan_age_hours:
            problems.append(
                f"orphan branch {branch}: zero commits ahead of origin/main, "
                f"tip {tip_age_hours / 24:.1f}d old (max {orphan_age_hours}h) — "
                "lane likely died at setup; no ledger claims it"
            )
    if problems:
        return _result(name, "breach", "; ".join(problems))
    return _result(
        name,
        "ok",
        f"{in_progress} in_progress lane(s) live; "
        f"{pattern_branches} lane-pattern branch(es) on origin, no orphans",
    )


# ---------------------------------------------------------------------------
# github_api_health (failure class B — external API degradation, 2026-06-10/11)
# ---------------------------------------------------------------------------


def _read_tail_lines(path: Path, max_lines: int) -> list[str]:
    """Read up to the last ``max_lines`` lines without loading the whole file."""
    step = 1 << 16
    with path.open("rb") as fh:
        fh.seek(0, 2)
        pos = fh.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= max_lines:
            read = min(step, pos)
            pos -= read
            fh.seek(pos)
            data = fh.read(read) + data
            step = min(step * 2, 1 << 22)
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def publisher_failure_streak(lines: list[str]) -> tuple[int, str]:
    """Trailing consecutive failed branch-publish passes + last error class.

    Counts ``branch publish pass failed`` markers backwards from the end of
    the log, stopping at the first ``branch publish pass complete`` marker.
    Per-attempt retry lines (``branch publish pass attempt N/M failed``) are
    NOT pass failures and are not counted.  The error class is the most
    recent ``HTTP <code>`` (or ``connectivity_failed``) seen in the window.
    """
    streak = 0
    for line in reversed(lines):
        if "branch publish pass failed" in line:
            streak += 1
        elif "branch publish pass complete" in line:
            break
    last_error = "none"
    for line in reversed(lines):
        match = re.search(r"HTTP (\d{3})", line)
        if match:
            last_error = f"HTTP {match.group(1)}"
            break
        if "connectivity_failed" in line:
            last_error = "connectivity_failed"
            break
    return streak, last_error


def check_github_api_health(
    log_path: Path,
    *,
    persist_threshold: int,
    tail_lines: int,
    probe_runner: Callable[[list[str]], int],
    probe_cmd: list[str] | None = None,
) -> CheckResult:
    """Distinguish persistent GitHub API degradation from transient blips.

    A cheap live probe (``gh api rate_limit``) plus the publisher log's
    failed-pass streak.  Breach ONLY when both agree the degradation is
    persistent: probe fails AND streak >= ``persist_threshold``.  Transient
    blips (short streak, or probe already recovered) stay visible-but-quiet:
    recorded in the detail/ledger, exit stays green.
    """
    name = "github_api_health"
    command = probe_cmd or ["gh", "api", "rate_limit"]
    probe_error = ""
    probe_ok: bool | None
    try:
        probe_ok = probe_runner(command) == 0
    except Exception as exc:  # noqa: BLE001 - a probe we cannot run is a blind spot
        probe_ok = None
        probe_error = f"{exc.__class__.__name__}: {exc}"
    if not log_path.exists():
        return _result(name, "unknown", f"publisher log missing: {log_path}")
    try:
        lines = _read_tail_lines(log_path, tail_lines)
    except OSError as exc:
        return _result(
            name, "unknown", f"publisher log unreadable: {exc.__class__.__name__}: {exc}"
        )
    streak, last_error = publisher_failure_streak(lines)
    base = (
        f"failed-pass streak={streak} (persist-threshold {persist_threshold}); "
        f"last_error={last_error}"
    )
    if probe_ok is None:
        return _result(name, "unknown", f"probe could not run ({probe_error}); {base}")
    if probe_ok:
        return _result(name, "ok", f"probe ok; {base}")
    if streak >= persist_threshold:
        return _result(name, "breach", f"persistent degradation: probe failed and {base}")
    return _result(name, "ok", f"transient degradation (no breach): probe failed but {base}")


# ---------------------------------------------------------------------------
# Report, exit contract, ledger, notification
# ---------------------------------------------------------------------------


def exit_code_for(checks: list[CheckResult]) -> int:
    """0 all ok; 1 any breach; 2 any unknown (unknown outranks breach)."""
    statuses = {c["status"] for c in checks}
    if "unknown" in statuses:
        return 2
    if "breach" in statuses:
        return 1
    return 0


def breach_summary(checks: list[CheckResult]) -> str:
    breached = [c for c in checks if c["status"] != "ok"]
    parts = [f"{c['check']}[{c['status']}]: {c['detail']}" for c in breached]
    return f"fleet-sentinel: {len(breached)} signal(s) firing — " + " | ".join(parts)


def notify(notify_cmd: str, summary: str, *, runner: Callable[[list[str]], int]) -> None:
    tokens = shlex.split(notify_cmd)
    if any("{summary}" in t for t in tokens):
        # A bare "{summary}" token becomes its own argv element — safe to pass
        # the text through verbatim.  A placeholder embedded in a larger token
        # lands inside another language's string literal (e.g. the installer's
        # default AppleScript "display notification" command), so neutralize
        # quote/backslash injection before substituting.
        embedded_safe = summary.replace("\\", "/").replace('"', "'")
        tokens = [
            summary if t == "{summary}" else t.replace("{summary}", embedded_safe) for t in tokens
        ]
    else:
        tokens.append(summary)
    try:
        runner(tokens)
    except Exception as exc:  # noqa: BLE001 - notification failure must not mask the report
        print(f"fleet-sentinel: notify-cmd failed: {exc}", file=sys.stderr)


def append_ledger(ledger: Path, report: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as fh:
        fh.write(json.dumps(report, sort_keys=True) + "\n")


def run_checks(args: argparse.Namespace, now: datetime) -> list[CheckResult]:
    selected = [c.strip() for c in args.checks.split(",") if c.strip()]
    unknown_names = set(selected) - set(ALL_CHECKS)
    if unknown_names:
        raise SystemExit(f"unknown check(s): {sorted(unknown_names)}")
    results: list[CheckResult] = []
    for name in selected:
        try:
            if name == "publisher_status":
                results.append(
                    check_publisher_status(
                        Path(args.publisher_status),
                        max_age_hours=args.publisher_max_age_hours,
                        now=now,
                    )
                )
            elif name == "boss_metrics_heartbeat":
                results.append(
                    check_boss_metrics(
                        Path(args.boss_metrics), max_age_hours=args.metrics_max_age_hours, now=now
                    )
                )
            elif name == "launchd_plists":
                results.append(check_launchd_plists(Path(args.launch_agents_dir)))
            elif name == "gh_auth":
                results.append(
                    check_gh_auth(runner=_default_command_runner, cmd=shlex.split(args.gh_auth_cmd))
                )
            elif name == "checkout_invariant":
                results.append(
                    check_checkout_invariant(
                        Path(args.repo_root), branch_reader=_default_branch_reader
                    )
                )
            elif name == "outbox_depth":
                results.append(
                    check_outbox(
                        Path(args.outbox_dir),
                        max_items=args.outbox_max,
                        max_age_days=args.outbox_max_age_days,
                        now=now,
                    )
                )
            elif name == "disk_free":
                results.append(
                    check_disk_free(Path(args.repo_root), min_free_gib=args.min_free_gib)
                )
            elif name == "lane_liveness":
                repo = Path(args.repo_root)
                results.append(
                    check_lane_liveness(
                        args.lanes_glob,
                        lane_max_age_hours=args.lane_max_age_hours,
                        orphan_age_hours=args.orphan_branch_age_hours,
                        now=now,
                        remote_heads=lambda repo=repo: _default_remote_heads(repo),
                        ahead_counter=_default_ahead_counter(repo),
                        commit_dater=_default_commit_dater(repo),
                    )
                )
            elif name == "github_api_health":
                results.append(
                    check_github_api_health(
                        Path(args.publisher_log),
                        persist_threshold=args.persist_threshold,
                        tail_lines=args.publisher_log_tail_lines,
                        probe_runner=_default_command_runner,
                        probe_cmd=shlex.split(args.rate_limit_cmd),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - a crashed check is a blind spot, not success
            results.append(
                _result(name, "unknown", f"check crashed: {exc.__class__.__name__}: {exc}")
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--publisher-status",
        # Live writer path: scripts/cache_codex_automation_github_status.py
        # refreshes this on every publisher pass.  The previous default,
        # .aragora/automation-publisher-status.json, has been an orphan since
        # its writer moved (~2026-05-24) — watching it would alarm forever on
        # stale data or, worse, stay green on a frozen healthy snapshot.
        default=str(repo_root / ".aragora" / "automation-github-status" / "latest.json"),
    )
    parser.add_argument("--publisher-max-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--boss-metrics",
        default=str(repo_root / ".aragora" / "overnight" / "boss_metrics.jsonl"),
    )
    parser.add_argument("--metrics-max-age-hours", type=float, default=48.0)
    parser.add_argument(
        "--launch-agents-dir", default=str(Path.home() / "Library" / "LaunchAgents")
    )
    parser.add_argument("--gh-auth-cmd", default="gh auth status")
    parser.add_argument("--outbox-dir", default=str(repo_root / ".aragora" / "automation-outbox"))
    parser.add_argument("--outbox-max", type=int, default=50)
    parser.add_argument("--outbox-max-age-days", type=float, default=7.0)
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument(
        "--lanes-glob",
        # Lane-ledger convention: .aragora/run-*/lanes/<lane>.json
        default=str(repo_root / ".aragora" / "run-*" / "lanes"),
        help="glob for lane-ledger directories (entries are <lane>.json inside)",
    )
    parser.add_argument("--lane-max-age-hours", type=float, default=3.0)
    parser.add_argument("--orphan-branch-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--publisher-log",
        default=str(repo_root / ".aragora" / "overnight" / "codex-automation-publisher.log"),
    )
    parser.add_argument("--publisher-log-tail-lines", type=int, default=2000)
    parser.add_argument(
        "--persist-threshold",
        type=int,
        default=3,
        help="consecutive failed publisher passes before degradation counts as persistent",
    )
    parser.add_argument("--rate-limit-cmd", default="gh api rate_limit")
    parser.add_argument(
        "--ledger",
        default=str(repo_root / ".aragora" / "fleet-sentinel" / "ledger.jsonl"),
    )
    parser.add_argument("--checks", default=",".join(ALL_CHECKS))
    parser.add_argument("--json", action="store_true", help="emit the JSON report to stdout")
    parser.add_argument(
        "--now", default=None, help="ISO-8601 timestamp override (for tests/replays)"
    )
    parser.add_argument(
        "--notify-cmd",
        default=None,
        help="command template invoked on breach; {summary} is replaced",
    )
    parser.add_argument("--no-ledger", action="store_true", help="skip the ledger append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = parse_iso(args.now) if args.now else datetime.now(timezone.utc)
    checks = run_checks(args, now)
    breaches = sum(1 for c in checks if c["status"] == "breach")
    blind = sum(1 for c in checks if c["status"] == "unknown")
    report = {
        "generated_at": (args.now or now.isoformat().replace("+00:00", "Z")),
        "checks": checks,
        "breaches": breaches,
        "blind_checks": blind,
    }
    if not args.no_ledger:
        try:
            append_ledger(Path(args.ledger), report)
        except OSError as exc:
            print(f"fleet-sentinel: ledger append failed: {exc}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for c in checks:
            print(f"{c['status']:>7}  {c['check']}: {c['detail']}")
    if (breaches or blind) and args.notify_cmd:
        notify(args.notify_cmd, breach_summary(checks), runner=_default_command_runner)
    return exit_code_for(checks)


if __name__ == "__main__":
    raise SystemExit(main())
