from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH, ShiftLedger


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
                "--label",
                "autonomous",
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
