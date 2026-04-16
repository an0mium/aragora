from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH, ShiftLedger
from aragora.swarm.terminal_truth import TerminalClass, classify_from_metrics

DEFAULT_METRICS_PATH = Path(".aragora") / "overnight" / "boss_metrics.jsonl"
_SANITIZER_REJECTION_OUTCOMES = frozenset({"dropped", "quarantined"})


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_elapsed_seconds(value: object) -> str:
    elapsed = _optional_float(value)
    if elapsed is None:
        return "-"
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s"


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_ledger_status(repo_root: Path) -> dict[str, Any] | None:
    ledger_path = repo_root / Path(DEFAULT_LEDGER_PATH)
    if not ledger_path.exists():
        return None
    try:
        payload = ShiftLedger(path=ledger_path).get_status_summary()
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload else None


def _resolve_terminal_class(row: dict[str, Any]) -> str:
    existing = _optional_text(row.get("terminal_class"))
    if existing:
        return existing
    try:
        return classify_from_metrics(row).value
    except Exception:
        return TerminalClass.RESCUE_NO_DELIVERABLE.value


def _infer_repo_name(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    remote = (proc.stdout or "").strip()
    if remote.startswith("git@github.com:"):
        return remote.removeprefix("git@github.com:").removesuffix(".git")
    marker = "github.com/"
    if marker in remote:
        return remote.split(marker, 1)[1].removesuffix(".git")
    return None


def _boss_ready_queue_depth(repo_root: Path, *, boss_repo: str | None = None) -> int | None:
    gh = shutil.which("gh")
    repo = _optional_text(boss_repo) or _infer_repo_name(repo_root)
    if not gh or not repo:
        return None
    try:
        proc = subprocess.run(
            [
                gh,
                "issue",
                "list",
                "--repo",
                repo,
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
            timeout=15,
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


def load_operator_status(
    repo_root: Path,
    *,
    limit: int = 10,
    boss_repo: str | None = None,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    metrics_file = metrics_path or (repo_root / DEFAULT_METRICS_PATH)
    rows = _load_metrics_rows(metrics_file)
    ledger_status = _load_ledger_status(repo_root)
    queue_depth = _optional_int(
        ledger_status.get("current_queue_size") if isinstance(ledger_status, dict) else None
    )
    if queue_depth is None:
        queue_depth = _boss_ready_queue_depth(repo_root, boss_repo=boss_repo)

    attempted_issues = {
        issue_number
        for row in rows
        if (issue_number := _optional_int(row.get("issue_number"))) is not None
    }
    completed_issues = {
        issue_number
        for row in rows
        if (issue_number := _optional_int(row.get("issue_number"))) is not None
        and _optional_text(row.get("worker_status")) == "completed"
    }
    deferred_publish_count = sum(
        1
        for row in rows
        if "deferred" in str(row.get("publish_action") or "").lower()
        or _resolve_terminal_class(row) == TerminalClass.RESCUE_PUBLISH_DEFERRED.value
    )
    sanitizer_rejection_count = sum(
        1
        for row in rows
        if _optional_text(row.get("sanitizer_outcome")) in _SANITIZER_REJECTION_OUTCOMES
    )
    rejection_rate = sanitizer_rejection_count / len(rows) if rows else 0.0

    per_issue: dict[int, dict[str, Any]] = {}
    for row in rows:
        issue_number = _optional_int(row.get("issue_number"))
        if issue_number is None:
            continue
        bucket = per_issue.setdefault(
            issue_number,
            {
                "issue_number": issue_number,
                "attempts": 0,
                "completed_iterations": 0,
            },
        )
        bucket["attempts"] += 1
        if _optional_text(row.get("worker_status")) == "completed":
            bucket["completed_iterations"] += 1

    per_issue_success = []
    for item in per_issue.values():
        attempts = int(item["attempts"])
        completed_iterations = int(item["completed_iterations"])
        item["success_rate"] = round(completed_iterations / attempts, 4) if attempts else 0.0
        per_issue_success.append(item)
    per_issue_success.sort(
        key=lambda item: (-float(item["success_rate"]), int(item["issue_number"]))
    )

    last_iterations = []
    for row in rows[-max(1, int(limit)) :]:
        last_iterations.append(
            {
                "iteration": _optional_int(row.get("iteration")),
                "issue_number": _optional_int(row.get("issue_number")),
                "terminal_class": _resolve_terminal_class(row),
                "elapsed_seconds": _optional_float(row.get("elapsed_seconds")),
                "worker_status": _optional_text(row.get("worker_status")) or "unknown",
            }
        )

    return {
        "available": metrics_file.exists() or bool(ledger_status),
        "metrics_path": str(metrics_file),
        "ledger_status": ledger_status or {},
        "summary": {
            "unique_issues_attempted": len(attempted_issues),
            "unique_issues_completed": len(completed_issues),
            "sanitizer_rejection_rate": round(rejection_rate, 4),
            "sanitizer_rejection_count": sanitizer_rejection_count,
            "deferred_publish_count": deferred_publish_count,
            "queue_depth": queue_depth if queue_depth is not None else "unknown",
        },
        "last_iterations": last_iterations,
        "per_issue_success": per_issue_success,
    }


def render_operator_status(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "operator attempted={attempted} completed={completed} sanitizer_rejection_rate={rate:.1%} "
        "deferred_publish={deferred} queue_depth={queue_depth}".format(
            attempted=summary.get("unique_issues_attempted", 0),
            completed=summary.get("unique_issues_completed", 0),
            rate=float(summary.get("sanitizer_rejection_rate", 0.0) or 0.0),
            deferred=summary.get("deferred_publish_count", 0),
            queue_depth=summary.get("queue_depth", "unknown"),
        )
    ]
    ledger_status = (
        payload.get("ledger_status", {}) if isinstance(payload.get("ledger_status"), dict) else {}
    )
    if ledger_status:
        green_shift = (
            ledger_status.get("green_shift", {})
            if isinstance(ledger_status.get("green_shift"), dict)
            else {}
        )
        lines.append(
            "proof-first queue={queue} boss={boss} merge={merge} benchmark_fresh={benchmark} "
            "merged_prs={merged} last_stop={stop} green_shift={green}".format(
                queue=ledger_status.get("current_queue_size", "unknown"),
                boss=ledger_status.get("current_boss_running"),
                merge=ledger_status.get("current_merge_running"),
                benchmark=ledger_status.get("current_benchmark_fresh"),
                merged=ledger_status.get("prs_merged", 0),
                stop=ledger_status.get("last_stop_reason") or "-",
                green=green_shift.get("is_green"),
            )
        )

    iterations = [item for item in payload.get("last_iterations", []) if isinstance(item, dict)]
    if iterations:
        lines.append("recent iterations:")
        for item in iterations[-10:]:
            issue_text = item.get("issue_number")
            lines.append(
                "  #{issue} {terminal} elapsed={elapsed} status={status}".format(
                    issue=issue_text if issue_text not in (None, "") else "-",
                    terminal=item.get("terminal_class", "unknown"),
                    elapsed=_format_elapsed_seconds(item.get("elapsed_seconds")),
                    status=item.get("worker_status", "unknown"),
                )
            )

    per_issue_success = [
        item for item in payload.get("per_issue_success", []) if isinstance(item, dict)
    ]
    if per_issue_success:
        lines.append("per-issue success:")
        for item in per_issue_success[:10]:
            lines.append(
                "  #{issue} {completed}/{attempts} ({rate:.0%})".format(
                    issue=item.get("issue_number", "-"),
                    completed=item.get("completed_iterations", 0),
                    attempts=item.get("attempts", 0),
                    rate=float(item.get("success_rate", 0.0) or 0.0),
                )
            )

    return "\n".join(lines)
