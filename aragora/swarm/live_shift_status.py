from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH, ShiftLedger

# Path (relative to repo root) to the canonical B0 truth artifact whose
# generated_at timestamp drives the publication-freshness signal. See
# docs/status/B0_BENCHMARK_TRUTH_STATUS.md and #6798 for the upstream
# rationale (Foreman-gate criterion: "recurring benchmark publication
# stays complete and fresh without babysitting").
BENCHMARK_TRUTH_LATEST = (
    "docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json"
)
BENCHMARK_FRESHNESS_DEFAULT_MAX_AGE_HOURS = 24.0


def _resolve_ledger_path(repo_root: Path, value: object | None) -> Path:
    candidate = Path(str(value or DEFAULT_LEDGER_PATH)).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _has_repo_metadata(repo_root: Path) -> bool:
    return (repo_root / ".git").exists()


def _run_git_command(
    repo_root: Path, *args: str, timeout: int = 5
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _infer_repo_name(repo_root: Path) -> str | None:
    proc = _run_git_command(repo_root, "remote", "get-url", "origin")
    if proc is None:
        return None
    remote = (proc.stdout or "").strip()
    if remote.startswith("git@github.com:"):
        return remote.removeprefix("git@github.com:").removesuffix(".git")
    marker = "github.com/"
    if marker in remote:
        return remote.split(marker, 1)[1].removesuffix(".git")
    return None


def _detect_swarm_process(pattern: str) -> bool | None:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return None
    try:
        proc = subprocess.run(
            [pgrep, "-f", pattern],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def _count_live_queue_depth(repo_root: Path, *, repo_name: str | None) -> int | None:
    gh = shutil.which("gh")
    if not gh or not repo_name:
        return None
    try:
        proc = subprocess.run(
            [
                gh,
                "issue",
                "list",
                "--repo",
                repo_name,
                "--label",
                "boss-ready",
                "--state",
                "open",
                "--limit",
                "500",
                "--json",
                "number",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    return len(payload) if isinstance(payload, list) else None


def _count_live_open_prs(repo_root: Path, *, repo_name: str | None) -> int | None:
    gh = shutil.which("gh")
    if not gh or not repo_name:
        return None
    try:
        proc = subprocess.run(
            [
                gh,
                "pr",
                "list",
                "--repo",
                repo_name,
                "--state",
                "open",
                "--limit",
                "500",
                "--json",
                "number",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    return len(payload) if isinstance(payload, list) else None


def _detect_observer_state(repo_root: Path) -> dict[str, Any]:
    if not _has_repo_metadata(repo_root):
        return {}
    payload: dict[str, Any] = {}
    branch_proc = _run_git_command(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    head_proc = _run_git_command(repo_root, "rev-parse", "HEAD")
    origin_main_proc = _run_git_command(repo_root, "rev-parse", "origin/main")
    rev_list_proc = _run_git_command(
        repo_root, "rev-list", "--left-right", "--count", "origin/main...HEAD"
    )
    status_proc = _run_git_command(repo_root, "status", "--short")

    branch = (
        (branch_proc.stdout or "").strip() if branch_proc and branch_proc.returncode == 0 else ""
    )
    if branch:
        payload["observer_branch"] = branch

    head = (head_proc.stdout or "").strip() if head_proc and head_proc.returncode == 0 else ""
    if head:
        payload["observer_head"] = head

    origin_main = (
        (origin_main_proc.stdout or "").strip()
        if origin_main_proc and origin_main_proc.returncode == 0
        else ""
    )
    if origin_main:
        payload["observer_origin_main_head"] = origin_main

    if rev_list_proc and rev_list_proc.returncode == 0:
        counts = (rev_list_proc.stdout or "").strip().split()
        if len(counts) == 2:
            behind_count: int | None
            ahead_count: int | None
            try:
                behind_count = int(counts[0])
                ahead_count = int(counts[1])
            except ValueError:
                behind_count = ahead_count = None
            else:
                payload["observer_behind_origin_main"] = behind_count
                payload["observer_ahead_of_origin_main"] = ahead_count

    has_uncommitted_changes = bool((status_proc.stdout or "").strip()) if status_proc else None
    if has_uncommitted_changes is not None:
        payload["observer_has_uncommitted_changes"] = has_uncommitted_changes

    warning_parts: list[str] = []
    if payload.get("observer_has_uncommitted_changes"):
        warning_parts.append("dirty checkout")
    payload_behind = payload.get("observer_behind_origin_main")
    if isinstance(payload_behind, int) and payload_behind > 0:
        warning_parts.append(f"{payload_behind} behind origin/main")
    payload_ahead = payload.get("observer_ahead_of_origin_main")
    if isinstance(payload_ahead, int) and payload_ahead > 0:
        warning_parts.append(f"{payload_ahead} ahead of origin/main")
    if warning_parts:
        payload["observer_warning"] = "observer checkout is " + ", ".join(warning_parts)

    return payload


def _detect_benchmark_freshness(
    repo_root: Path,
    *,
    max_age_hours: float = BENCHMARK_FRESHNESS_DEFAULT_MAX_AGE_HOURS,
) -> dict[str, Any]:
    """Read the B0 truth artifact and report freshness against the threshold.

    Returns three keys to be merged into the shift-status payload:

    - ``current_benchmark_fresh``: bool when the artifact has a parseable
      ``generated_at`` timestamp, else ``None``.  ``True`` when the artifact
      was written within ``max_age_hours``.
    - ``current_benchmark_age_hours``: float age in hours from the artifact's
      ``generated_at`` to ``datetime.now(timezone.utc)``, rounded to 1
      decimal; ``None`` when the timestamp is unreadable.
    - ``current_benchmark_generated_at``: the raw ``generated_at`` string from
      the artifact, or ``None``.

    Closes the Foreman-gate-blocking gap recorded in #6798: the
    ``current_benchmark_fresh`` payload key already existed in the schema
    but was never populated from disk, so a stale B0 publication looked
    healthy in operator surfaces.

    All None-returns indicate "no usable signal" rather than "artifact is
    stale" — a missing artifact is its own kind of degraded state and is
    surfaced via the warning composition in ``_compose_freshness_warning``.
    """
    payload: dict[str, Any] = {
        "current_benchmark_fresh": None,
        "current_benchmark_age_hours": None,
        "current_benchmark_generated_at": None,
    }
    artifact = repo_root / BENCHMARK_TRUTH_LATEST
    if not artifact.exists():
        return payload
    try:
        data = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError):
        return payload
    generated_at = str(data.get("generated_at") or data.get("recorded_on") or "").strip()
    if not generated_at:
        return payload
    try:
        when = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return payload
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - when).total_seconds()
    age_hours = age_seconds / 3600.0
    payload["current_benchmark_fresh"] = age_hours <= max_age_hours
    payload["current_benchmark_age_hours"] = round(age_hours, 1)
    payload["current_benchmark_generated_at"] = generated_at
    return payload


def _compose_freshness_warning(payload: dict[str, Any]) -> dict[str, Any]:
    """Extend ``observer_warning`` with benchmark-staleness when present.

    Called after both ``_detect_observer_state`` and
    ``_detect_benchmark_freshness`` have populated ``payload``.  Idempotent:
    if there is no observer warning yet, this writes one keyed only on the
    benchmark-staleness signal; if there is, it appends.
    """
    fresh = payload.get("current_benchmark_fresh")
    if fresh is not False:  # True or None: nothing to warn
        return payload
    age = payload.get("current_benchmark_age_hours")
    fragment = "benchmark truth stale"
    if isinstance(age, (int, float)):
        fragment = f"{fragment} ({age}h old)"
    existing = str(payload.get("observer_warning") or "").strip()
    if existing:
        payload["observer_warning"] = existing + "; " + fragment
    else:
        payload["observer_warning"] = "shift surfaces are " + fragment
    return payload


def _load_live_shift_truth(repo_root: Path) -> dict[str, Any]:
    if not _has_repo_metadata(repo_root):
        return {}
    repo_name = _infer_repo_name(repo_root)
    live_truth: dict[str, Any] = {}
    boss_running = _detect_swarm_process(r"aragora\.cli\.main.*swarm boss-loop")
    if boss_running is not None:
        live_truth["current_boss_running"] = boss_running
    merge_running = _detect_swarm_process(r"aragora\.cli\.main.*swarm merge-arbiter")
    if merge_running is not None:
        live_truth["current_merge_running"] = merge_running
    queue_size = _count_live_queue_depth(repo_root, repo_name=repo_name)
    if queue_size is not None:
        live_truth["current_queue_size"] = queue_size
    open_prs = _count_live_open_prs(repo_root, repo_name=repo_name)
    if open_prs is not None:
        live_truth["current_open_prs"] = open_prs
    live_truth.update(_detect_observer_state(repo_root))
    live_truth.update(_detect_benchmark_freshness(repo_root))
    _compose_freshness_warning(live_truth)
    return live_truth


def load_shift_status(
    repo_root: Path,
    *,
    ledger_path: object | None = None,
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    path = _resolve_ledger_path(repo_root, ledger_path)
    existed = path.exists()
    if existed:
        payload = ShiftLedger(path=path).get_status_summary(max_age_hours=max_age_hours)
    else:
        payload = {
            "period_hours": max_age_hours,
            "total_entries": 0,
            "shifts_started": 0,
            "shifts_stopped": 0,
            "last_stop_reason": "",
            "cycle_ticks": 0,
            "prs_merged": 0,
            "pr_numbers_merged": [],
            "current_queue_size": None,
            "current_open_prs": None,
            "current_boss_running": None,
            "current_merge_running": None,
            "current_benchmark_fresh": None,
            "failure_policy": {},
            "green_shift": {},
        }
    payload.update(_load_live_shift_truth(repo_root))
    return {
        "available": existed,
        "ledger_path": str(path),
        **payload,
    }
