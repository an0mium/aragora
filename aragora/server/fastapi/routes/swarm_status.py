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
    issues_attempted: set[int] = set()
    issues_succeeded: set[int] = set()
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
            if tc and tc.startswith("success"):
                issues_succeeded.add(issue_num)

        latest_tick = row

    success_count = sum(v for k, v in terminal_classes.items() if k.startswith("success"))

    return {
        "status": "active",
        "metrics_path": str(path),
        "window": window,
        "total_ticks": len(rows),
        "unique_issues_attempted": len(issues_attempted),
        "unique_issues_succeeded": len(issues_succeeded),
        "success_rate": round(success_count / len(rows), 3) if rows else 0,
        "terminal_class_distribution": dict(terminal_classes.most_common(10)),
        "outcome_distribution": dict(outcomes.most_common(10)),
        "latest_tick": {
            "timestamp": latest_tick.get("timestamp", ""),
            "issue_number": latest_tick.get("issue_number"),
            "terminal_class": latest_tick.get("terminal_class", ""),
            "elapsed_seconds": latest_tick.get("elapsed_seconds"),
        },
    }


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

    app.include_router(router)
