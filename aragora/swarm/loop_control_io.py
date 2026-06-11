"""Loop Control Plane v1 - read-only collectors (IO layer).

Each collector wraps an *existing* read-only surface (operator snapshot, publisher
freshness check, launchd job status, proof-first runtime state, git worktree list)
and returns a normalized raw-signal dict consumable by
``aragora.swarm.loop_control.classify_loop``. This module is the *only* part of
the Loop Control Plane that touches the world, and it only ever reads:

* ``launchctl print`` (job liveness)
* ``git worktree list --porcelain``
* ``scripts/publisher_freshness_check.py --json`` / ``scripts/agent_bridge.py
  operator-snapshot --json`` (themselves read-only)
* ``.aragora/proof_first_shift/runtime_state.json`` (file read)
* ``.aragora/docs_drift_status.json`` (file read)
* ``.aragora/loop_budgets.json`` + ``.aragora/loop_spend/<loop_id>.json`` (file
  reads; the spend-ledger *writer* lives in ``aragora.swarm.loop_budget`` and is
  called by loops, never by collectors)

It never merges, comments, reruns, pushes, or passes an ``--apply`` flag, and it
writes nothing. Collectors degrade gracefully (``source_status`` of ``degraded``
or ``unavailable``) on missing files, rate limits, timeouts, or non-POSIX hosts
rather than raising.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aragora.swarm.loop_budget import BudgetPolicy, resolve_loop_budget
from aragora.swarm.loop_control import LOOP_SPECS, LoopKind, LoopRecord, classify_loop

LAUNCHD_LABELS: dict[LoopKind, str] = {
    LoopKind.BOSS_LOOP: "com.aragora.swarm-boss-loop",
    LoopKind.MERGE_ARBITER: "com.aragora.swarm-merge-arbiter",
    LoopKind.PUBLISHER: "com.aragora.codex-automation-publisher",
    LoopKind.WORKTREE_AUTOPILOT: "com.aragora.codex-worktree-maintainer",
}

# Collectors that may reach the network (skipped under allow_network=False).
NETWORK_TOUCHING: frozenset[LoopKind] = frozenset({LoopKind.BOSS_LOOP})

_PROOF_FIRST_STATE_FRESH_SECONDS = 900.0
# Daily launchd cadence plus slack; staler than this means the detector
# stopped firing (halted), not that it is broken.
_DOCS_DRIFT_STATE_FRESH_SECONDS = 26 * 3600.0


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Run a *read-only* command, returning None on any execution failure."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _loads(proc: subprocess.CompletedProcess[str] | None) -> dict[str, Any] | None:
    if proc is None or not proc.stdout:
        return None
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _launchctl_loaded(label: str, timeout: float) -> tuple[bool | None, str]:
    """Return ``(loaded, detail)``; ``loaded`` is None when launchctl is unusable."""
    try:
        uid = os.getuid()
    except AttributeError:
        return None, "non-posix-platform"
    proc = _run(["launchctl", "print", f"gui/{uid}/{label}"], Path.cwd(), timeout)
    if proc is None:
        return None, "launchctl-unavailable"
    if proc.returncode == 0:
        return True, "loaded"
    return False, "not-loaded"


def collect_publisher(repo_root: Path, *, timeout: float = 10.0) -> dict[str, Any]:
    proc = _run(
        [sys.executable, "scripts/publisher_freshness_check.py", "--json", "--summary-only"],
        repo_root,
        timeout,
    )
    payload = _loads(proc)
    if payload is None:
        return {"source_status": "unavailable", "error": "publisher freshness check unreadable"}
    verdict = payload.get("verdict")
    blockers = _as_list(payload.get("blockers"))
    degraded = verdict == "degraded"
    return {
        "source_status": "ok",
        "alive": bool(payload.get("launchd_loaded")),
        "operational_fault": degraded,
        "stop_reason": "; ".join(str(b) for b in blockers) if degraded else None,
        "feedback_status": str(verdict) if verdict else "unknown",
        "owner": "launchd",
    }


def collect_boss_loop(repo_root: Path, *, timeout: float = 20.0) -> dict[str, Any]:
    proc = _run(
        [
            sys.executable,
            "scripts/agent_bridge.py",
            "operator-snapshot",
            "--json",
            "--summary-only",
        ],
        repo_root,
        timeout,
    )
    payload = _loads(proc)
    if payload is None:
        return {"source_status": "unavailable", "error": "operator snapshot unreadable"}
    status = _as_dict(payload.get("boss_loop_status"))
    heartbeats = _as_dict(payload.get("agent_heartbeats"))
    count = int(heartbeats.get("count", 0) or 0)
    fresh = int(heartbeats.get("fresh_count", 0) or 0)
    health = _as_dict(payload.get("health"))
    queue_depth = payload.get("queue_depth")
    return {
        "source_status": "ok",
        "alive": bool(payload.get("boss_loop_alive")),
        "owner_stale": count > 0 and fresh == 0,
        "feedback_status": "ok" if health.get("ok", True) else "degraded",
        "ticks": queue_depth if isinstance(queue_depth, int) else None,
        "owner": str(status.get("owner") or "swarm-boss-loop"),
        "last_progress_at": payload.get("timestamp")
        if isinstance(payload.get("timestamp"), str)
        else None,
    }


def collect_merge_arbiter(repo_root: Path, *, timeout: float = 10.0) -> dict[str, Any]:
    max_runtime_s: float | None = None
    max_ticks: int | None = None
    try:
        from aragora.swarm.merge_arbiter import MergeArbiterConfig

        config = MergeArbiterConfig()
        max_runtime_s = float(config.max_runtime_hours) * 3600.0
        max_ticks = int(config.max_consecutive_failures)
    except Exception:  # noqa: BLE001 - config import is best-effort context
        pass

    loaded, detail = _launchctl_loaded(LAUNCHD_LABELS[LoopKind.MERGE_ARBITER], timeout)
    base: dict[str, Any] = {
        "max_runtime_s": max_runtime_s,
        "max_ticks": max_ticks,
        "feedback_status": "quorum",
        "human_settlement_present": False,
        "owner": "launchd",
    }
    if loaded is None:
        # In-process arbiter exposes no state file; without launchd we can only
        # report the config-grounded bounds.
        return {"source_status": "degraded", "error": detail, **base}
    base["source_status"] = "ok"
    base["alive"] = loaded
    return base


def collect_proof_first(
    repo_root: Path, *, timeout: float = 5.0, now: float | None = None
) -> dict[str, Any]:
    path = repo_root / ".aragora" / "proof_first_shift" / "runtime_state.json"
    if not path.is_file():
        return {"source_status": "unavailable", "feedback_status": "proof_freshness"}
    payload = _read_json(path)
    if payload is None:
        return {"source_status": "degraded", "error": "runtime_state unreadable"}
    now = now if now is not None else time.time()
    mtime = path.stat().st_mtime
    age = max(0.0, now - mtime)
    failure_counts = (
        "boss_restart_count",
        "merge_restart_count",
        "auth_failure_count",
        "publication_failure_count",
        "rate_limit_failure_count",
        "permission_mismatch_count",
        "runtime_failure_count",
        "github_outage_count",
    )
    failures = sum(int(payload.get(key, 0) or 0) for key in failure_counts)
    return {
        "source_status": "ok",
        "alive": age < _PROOF_FIRST_STATE_FRESH_SECONDS,
        "no_progress_ticks": failures,
        "last_progress_at": _iso(mtime),
        "feedback_status": "proof_freshness",
        "owner": str(payload.get("recovery_shift_id") or "proof-first-shift"),
    }


def collect_worktree_autopilot(repo_root: Path, *, timeout: float = 10.0) -> dict[str, Any]:
    proc = _run(["git", "worktree", "list", "--porcelain"], repo_root, timeout)
    count: int | None = None
    if proc is not None and proc.returncode == 0:
        count = sum(1 for line in proc.stdout.splitlines() if line.startswith("worktree "))
    loaded, _detail = _launchctl_loaded(LAUNCHD_LABELS[LoopKind.WORKTREE_AUTOPILOT], timeout)
    if count is None and loaded is None:
        return {"source_status": "unavailable", "feedback_status": "worktree_health"}
    return {
        "source_status": "ok" if loaded is not None else "degraded",
        "alive": loaded,
        "ticks": count,
        "feedback_status": "worktree_health",
        "owner": "launchd",
    }


def collect_nomic(repo_root: Path, *, timeout: float = 5.0) -> dict[str, Any]:
    # The nomic loop is launched manually (protected file) and exposes no
    # standing daemon to read; report unavailable rather than fabricate state.
    return {"source_status": "unavailable", "feedback_status": "none"}


def collect_docs_sync_drift(
    repo_root: Path, *, timeout: float = 5.0, now: float | None = None
) -> dict[str, Any]:
    path = repo_root / ".aragora" / "docs_drift_status.json"
    if not path.is_file():
        return {"source_status": "unavailable", "feedback_status": "docs_drift"}
    payload = _read_json(path)
    if payload is None:
        return {"source_status": "degraded", "error": "docs drift status unreadable"}
    now = now if now is not None else time.time()
    age = max(0.0, now - path.stat().st_mtime)
    outcome = str(payload.get("outcome") or "unknown")
    try:
        errors = int(payload.get("consecutive_errors", 0) or 0)
    except (TypeError, ValueError):
        errors = 0
    fault = outcome in ("error", "drift_outside_allowlist")  # detector FAULT_OUTCOMES
    waiting = outcome in ("drift_pr_open", "drift_pr_opened", "drift_detected")
    stop_reason: str | None = None
    if fault:
        raw_error = payload.get("error")
        stop_reason = (
            str(raw_error)
            if isinstance(raw_error, str) and raw_error
            else "drift outside generated-mirror allowlist"
        )
    return {
        "source_status": "ok",
        "alive": age < _DOCS_DRIFT_STATE_FRESH_SECONDS,
        "operational_fault": fault,
        "stop_reason": stop_reason,
        "waiting_only": waiting and not fault,
        "no_progress_ticks": errors,
        "last_progress_at": (
            payload.get("generated_at") if isinstance(payload.get("generated_at"), str) else None
        ),
        "feedback_status": outcome,
        "owner": "launchd",
    }


_COLLECTORS: dict[LoopKind, Callable[..., dict[str, Any]]] = {
    LoopKind.BOSS_LOOP: collect_boss_loop,
    LoopKind.MERGE_ARBITER: collect_merge_arbiter,
    LoopKind.PROOF_FIRST_SHIFT: collect_proof_first,
    LoopKind.PUBLISHER: collect_publisher,
    LoopKind.WORKTREE_AUTOPILOT: collect_worktree_autopilot,
    LoopKind.NOMIC: collect_nomic,
    LoopKind.DOCS_SYNC_DRIFT: collect_docs_sync_drift,
}


def _safe_collect(
    fn: Callable[..., dict[str, Any]], repo_root: Path, timeout: float
) -> dict[str, Any]:
    try:
        return fn(repo_root, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - collectors must never raise to the fleet
        return {"source_status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def collect_all(
    repo_root: Path | str,
    *,
    timeout: float = 15.0,
    allow_network: bool = True,
    kinds: list[LoopKind] | None = None,
) -> dict[LoopKind, dict[str, Any]]:
    """Collect raw signals for the selected loops, concurrently and read-only.

    Each loop's raw dict carries a per-loop ``budget`` resolved from the
    operator policy file plus that loop's spend-ledger snapshot (see
    ``aragora.swarm.loop_budget``); loops without a ceiling or written spend
    are reported ``degraded``/``unavailable`` rather than fabricated.
    """
    root = Path(repo_root)
    selected = list(kinds) if kinds else list(_COLLECTORS)
    policy = BudgetPolicy.load(root)
    out: dict[LoopKind, dict[str, Any]] = {}

    pending: list[LoopKind] = []
    for kind in selected:
        if not allow_network and kind in NETWORK_TOUCHING:
            out[kind] = {
                "source_status": "unavailable",
                "error": "skipped: --no-network",
                "budget": resolve_loop_budget(root, kind.value, policy),
            }
            continue
        pending.append(kind)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as executor:
            futures = {
                executor.submit(_safe_collect, _COLLECTORS[kind], root, timeout): kind
                for kind in pending
            }
            for future in as_completed(futures):
                kind = futures[future]
                raw = future.result()
                if "budget" not in raw:
                    raw["budget"] = resolve_loop_budget(root, kind.value, policy)
                out[kind] = raw
    return out


def build_records(raw_by_kind: dict[LoopKind, dict[str, Any]]) -> list[LoopRecord]:
    """Classify collected raw signals into records, in registry order."""
    records: list[LoopRecord] = []
    for kind, spec in LOOP_SPECS.items():
        if kind in raw_by_kind:
            records.append(classify_loop(spec, raw_by_kind[kind]))
    return records


def collect_fleet(
    repo_root: Path | str,
    *,
    timeout: float = 15.0,
    allow_network: bool = True,
    kinds: list[LoopKind] | None = None,
) -> list[LoopRecord]:
    """Convenience: collect raw signals and classify them into records."""
    return build_records(
        collect_all(repo_root, timeout=timeout, allow_network=allow_network, kinds=kinds)
    )
