"""Minimal swarm status API endpoint.

Reads boss_metrics.jsonl and returns a JSON summary of recent boss loop
activity: queue depth, success/failure counts, terminal class distribution,
and latest tick metadata. This is the thin operator surface for the wedge.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")
DEFAULT_WINDOW = 50
SUCCESS_TERMINAL_PREFIXES = ("success", "deliverable")


def _is_success_terminal_class(terminal_class: str) -> bool:
    """Return whether a terminal class counts as a truthful success."""
    return terminal_class.startswith(SUCCESS_TERMINAL_PREFIXES)


def _tail_jsonl(path: Path, max_lines: int) -> list[dict[str, Any]]:
    """Read the last max_lines from a JSONL file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows: list[dict[str, Any]] = []
        for line in lines[-max_lines:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
    except OSError:
        return []


def swarm_status_summary(
    *,
    metrics_path: Path | None = None,
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Build a swarm status summary from boss metrics JSONL."""
    path = metrics_path or DEFAULT_METRICS_PATH
    rows = _tail_jsonl(path, window)

    if not rows:
        return {
            "status": "no_data",
            "metrics_path": str(path),
            "window": window,
            "total_ticks": 0,
        }

    terminal_classes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    issues_attempted: set[int] = set()
    issues_succeeded: set[int] = set()
    recent_blockers: list[dict[str, Any]] = []
    latest_tick: dict[str, Any] = {}

    for row in rows:
        tc = str(row.get("terminal_class", "")).strip()
        if tc:
            terminal_classes[tc] += 1

        outcome = str(row.get("outcome", row.get("worker_status", ""))).strip()
        if outcome:
            outcomes[outcome] += 1

        issue_num = row.get("issue_number")
        if isinstance(issue_num, int):
            issues_attempted.add(issue_num)
            if tc and _is_success_terminal_class(tc):
                issues_succeeded.add(issue_num)

        # Collect failure reasons and blocker evidence (BC-03)
        failure_reason = str(row.get("failure_reason", "")).strip()
        if failure_reason:
            failure_reasons[failure_reason] += 1
        if tc and not _is_success_terminal_class(tc):
            recent_blockers.append(
                {
                    "issue_number": issue_num,
                    "terminal_class": tc,
                    "failure_reason": failure_reason or None,
                    "blocker_kind": str(row.get("blocker_kind", "")).strip() or None,
                    "issue_title": str(row.get("issue_title", "")).strip()[:80] or None,
                }
            )

        latest_tick = row

    tick_success_count = sum(
        v for k, v in terminal_classes.items() if _is_success_terminal_class(k)
    )
    issue_success_rate = (
        round(len(issues_succeeded) / len(issues_attempted), 3) if issues_attempted else 0.0
    )

    return {
        "status": "active",
        "metrics_path": str(path),
        "window": window,
        "total_ticks": len(rows),
        "unique_issues_attempted": len(issues_attempted),
        "unique_issues_succeeded": len(issues_succeeded),
        "success_rate": issue_success_rate,
        "tick_success_rate": round(tick_success_count / len(rows), 3) if rows else 0.0,
        "terminal_class_distribution": dict(terminal_classes.most_common(10)),
        "outcome_distribution": dict(outcomes.most_common(10)),
        "failure_reason_distribution": dict(failure_reasons.most_common(10)),
        "recent_blockers": recent_blockers[-10:],
        "latest_tick": {
            "timestamp": latest_tick.get("timestamp", ""),
            "issue_number": latest_tick.get("issue_number"),
            "terminal_class": latest_tick.get("terminal_class", ""),
            "elapsed_seconds": latest_tick.get("elapsed_seconds"),
        },
    }


def preflight_check(
    *,
    agent: str = "codex",
    base_ref: str = "main",
    skip_publication: bool = True,
) -> dict[str, Any]:
    """Run preflight checks and return receipt-backed result."""
    try:
        from aragora.swarm.credential_envelope import CredentialEnvelope
        from aragora.swarm.preflight import run_preflight
    except ImportError:
        return {"error": "preflight module not available"}

    import os

    envelope = CredentialEnvelope.from_environment(os.environ)
    result = run_preflight(
        repo_root=Path.cwd(),
        agent=agent,
        base_ref=base_ref,
        skip_publication=skip_publication,
        envelope=envelope,
    )
    return result.to_dict()


def list_preflight_receipts() -> list[dict[str, Any]]:
    """List cached preflight receipts."""
    try:
        from aragora.swarm.preflight import PreflightReceipt
    except ImportError:
        return []

    receipt_dir = Path.cwd() / ".aragora" / "receipts" / "preflight"
    if not receipt_dir.exists():
        return []

    receipts: list[dict[str, Any]] = []
    for receipt_file in sorted(receipt_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            receipt = PreflightReceipt.from_dict(data)
            receipts.append(
                {
                    "receipt_id": receipt.receipt_id,
                    "check_type": receipt.check_type,
                    "passed": receipt.passed,
                    "started_at": receipt.started_at,
                    "finished_at": receipt.finished_at,
                    "expires_at": receipt.expires_at,
                    "cache_key": receipt.cache_key,
                }
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            continue
    return receipts[:20]


def register_routes(app: Any) -> None:
    """Register swarm status routes on a FastAPI app."""
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.debug("FastAPI not available, skipping swarm status routes")
        return

    router = APIRouter(prefix="/api/v1/swarm", tags=["swarm"])

    @router.get("/status")
    async def get_swarm_status() -> JSONResponse:
        return JSONResponse(content=swarm_status_summary())

    @router.post("/preflight")
    async def run_swarm_preflight(
        agent: str = "codex",
        base_ref: str = "main",
        skip_publication: bool = True,
    ) -> JSONResponse:
        result = preflight_check(
            agent=agent,
            base_ref=base_ref,
            skip_publication=skip_publication,
        )
        return JSONResponse(content=result)

    @router.get("/preflight/receipts")
    async def get_preflight_receipts() -> JSONResponse:
        return JSONResponse(content=list_preflight_receipts())

    app.include_router(router)
