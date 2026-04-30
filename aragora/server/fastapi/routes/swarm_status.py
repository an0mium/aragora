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

from aragora.swarm.live_shift_status import _detect_benchmark_freshness
from aragora.swarm.shift_ledger import DEFAULT_LEDGER_PATH as DEFAULT_SHIFT_LEDGER_PATH
from aragora.swarm.shift_ledger import ShiftLedger

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")
DEFAULT_WINDOW = 50
SUCCESS_TERMINAL_PREFIXES = ("success", "deliverable")
BLOCKER_EVIDENCE_MAX_CHARS = 240


def _is_success_terminal_class(terminal_class: str) -> bool:
    """Return whether a terminal class counts as a truthful success."""
    return terminal_class.startswith(SUCCESS_TERMINAL_PREFIXES)


def _compact_blocker_evidence(
    value: Any, *, max_chars: int = BLOCKER_EVIDENCE_MAX_CHARS
) -> str | None:
    compact = " ".join(str(value or "").split())
    if not compact:
        return None
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


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


def _load_ledger_status(
    *,
    repo_root: Path,
    ledger_path: Path | None = None,
) -> dict[str, Any] | None:
    path = ledger_path or (repo_root / Path(DEFAULT_SHIFT_LEDGER_PATH))
    if not path.exists():
        return None
    try:
        payload = ShiftLedger(path=path).get_status_summary()
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload else None


def swarm_status_summary(
    *,
    metrics_path: Path | None = None,
    repo_root: Path | None = None,
    ledger_path: Path | None = None,
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Build a swarm status summary, preferring proof-first ledger truth."""
    path = metrics_path or DEFAULT_METRICS_PATH
    root = repo_root or Path.cwd()
    rows = _tail_jsonl(path, window)
    ledger_status = _load_ledger_status(repo_root=root, ledger_path=ledger_path)

    # B0 publication freshness — Round 2026-04-30c Phase C, additive surface
    # for the Foreman-gate operability signal.  Reads `generated_at` from the
    # canonical B0 truth artifact at
    # docs/status/generated/benchmark_scorecards/.../latest.json and reports
    # three keys: current_benchmark_fresh / _age_hours / _generated_at.
    # All-None when the artifact is missing or unparseable; never errors.
    benchmark_freshness = _detect_benchmark_freshness(root)

    if not rows and not ledger_status:
        return {
            "status": "no_data",
            "metrics_path": str(path),
            "window": window,
            "total_ticks": 0,
            **benchmark_freshness,
        }

    terminal_classes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    rescue_classes: Counter[str] = Counter()
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

        rescue_class = str(row.get("rescue_class", "")).strip()
        if rescue_class:
            rescue_classes[rescue_class] += 1

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
                    "blocker_evidence": _compact_blocker_evidence(row.get("blocker_evidence")),
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
        "ledger_status": ledger_status or {},
        "queue_depth": (
            ledger_status.get("current_queue_size") if isinstance(ledger_status, dict) else None
        ),
        "boss_running": (
            ledger_status.get("current_boss_running") if isinstance(ledger_status, dict) else None
        ),
        "merge_running": (
            ledger_status.get("current_merge_running") if isinstance(ledger_status, dict) else None
        ),
        # ``benchmark_fresh`` (legacy) is the ledger's last-tick view.
        # Round 2026-04-30c Phase C adds the on-disk artifact-driven view
        # via ``current_benchmark_fresh`` / ``_age_hours`` /
        # ``_generated_at`` — these reflect what's actually published on
        # main, complementing the ledger's record.
        "benchmark_fresh": (
            ledger_status.get("current_benchmark_fresh")
            if isinstance(ledger_status, dict)
            else None
        ),
        **benchmark_freshness,
        "last_stop_reason": (
            ledger_status.get("last_stop_reason") if isinstance(ledger_status, dict) else ""
        ),
        "prs_merged_recent": (
            ledger_status.get("prs_merged") if isinstance(ledger_status, dict) else 0
        ),
        "merged_pr_numbers": (
            ledger_status.get("pr_numbers_merged") if isinstance(ledger_status, dict) else []
        ),
        "unique_issues_attempted": len(issues_attempted),
        "unique_issues_succeeded": len(issues_succeeded),
        "success_rate": issue_success_rate,
        "tick_success_rate": round(tick_success_count / len(rows), 3) if rows else 0.0,
        "terminal_class_distribution": dict(terminal_classes.most_common(10)),
        "outcome_distribution": dict(outcomes.most_common(10)),
        "failure_reason_distribution": dict(failure_reasons.most_common(10)),
        "rescue_class_summary": dict(rescue_classes.most_common(10)),
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
