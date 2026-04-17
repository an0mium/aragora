from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH, ShiftLedger


def _resolve_ledger_path(repo_root: Path, value: object | None) -> Path:
    candidate = Path(str(value or DEFAULT_LEDGER_PATH)).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _float_arg(value: object, *, default: float) -> float:
    if value is None or value == "":
        return default
    if not isinstance(value, (str, int, float)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_shift_status(
    repo_root: Path,
    *,
    ledger_path: object | None = None,
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Load the proof-first shift ledger status summary."""
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
    return {
        "available": existed,
        "ledger_path": str(path),
        **payload,
    }


def render_shift_status(payload: dict[str, Any]) -> str:
    """Render the ledger summary for operator-readable CLI output."""
    lines = [
        f"proof-first shift ledger: {payload.get('ledger_path', '')}",
        "available={available} period_hours={period} entries={entries} shifts={started}/{stopped} ticks={ticks}".format(
            available=payload.get("available", False),
            period=payload.get("period_hours", 24.0),
            entries=payload.get("total_entries", 0),
            started=payload.get("shifts_started", 0),
            stopped=payload.get("shifts_stopped", 0),
            ticks=payload.get("cycle_ticks", 0),
        ),
        "queue={queue} open_prs={open_prs} boss={boss} merge={merge} benchmark_fresh={benchmark}".format(
            queue=payload.get("current_queue_size"),
            open_prs=payload.get("current_open_prs"),
            boss=payload.get("current_boss_running"),
            merge=payload.get("current_merge_running"),
            benchmark=payload.get("current_benchmark_fresh"),
        ),
        "merged_prs={merged} last_stop={stop}".format(
            merged=payload.get("prs_merged", 0),
            stop=payload.get("last_stop_reason") or "-",
        ),
    ]
    green_shift = payload.get("green_shift")
    if isinstance(green_shift, dict) and green_shift:
        lines.append(
            "green_shift={green} duration_hours={hours} reason={reason}".format(
                green=green_shift.get("is_green"),
                hours=green_shift.get("duration_hours"),
                reason=green_shift.get("reason") or "-",
            )
        )
    failure_policy = payload.get("failure_policy")
    if isinstance(failure_policy, dict) and failure_policy:
        lines.append(
            "failure_policy={status} reason={reason}".format(
                status=failure_policy.get("status") or "unknown",
                reason=failure_policy.get("reason") or "-",
            )
        )
    return "\n".join(lines)


def cmd_shift_status(args: argparse.Namespace) -> None:
    """Handle ``aragora swarm shift-status``."""
    from aragora.worktree.fleet import resolve_repo_root

    repo_root = resolve_repo_root(Path.cwd())
    payload = load_shift_status(
        repo_root,
        ledger_path=getattr(args, "shift_ledger", None),
        max_age_hours=_float_arg(getattr(args, "max_age_hours", None), default=24.0),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_shift_status(payload))
