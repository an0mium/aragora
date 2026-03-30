"""SwarmReporter: plain-English report generation for non-developer users."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import logging
from dataclasses import dataclass, field
from typing import Any
import re

from pathlib import Path

from aragora.harnesses.base import AnalysisType
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.terminal_truth import (
    qualify_work_order_terminal_state,
    receipt_expected_for_lane,
)

logger = logging.getLogger(__name__)

_STALE_LANE_AFTER_SECONDS = 15 * 60
_LANE_TELEMETRY = LaneTelemetryCollector()


def _parse_iso_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: Any, *, now: datetime) -> float | None:
    parsed = _parse_iso_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _seconds_between(start: Any, end: Any) -> float | None:
    start_dt = _parse_iso_timestamp(start)
    end_dt = _parse_iso_timestamp(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def _text(value: Any) -> str:
    return str(value or "").strip()


def _metadata(item: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    data = item.get("metadata")
    return data if isinstance(data, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(_text(item) for item in value if _text(item)))


def _first_list(*values: Any) -> list[str]:
    for value in values:
        items = _text_list(value)
        if items:
            return items
    return []


def _extract_receipt_id(*sources: dict[str, Any] | None) -> str:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("receipt_id", "decision_receipt_id", "last_receipt_id"):
            text = _text(source.get(key))
            if text:
                return text
        meta = _metadata(source)
        for key in ("receipt_id", "decision_receipt_id", "last_receipt_id"):
            text = _text(meta.get(key))
            if text:
                return text
    return ""


def _deliverable(source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    deliverable = source.get("deliverable")
    if isinstance(deliverable, dict):
        return deliverable
    meta = _metadata(source)
    deliverable = meta.get("deliverable")
    return deliverable if isinstance(deliverable, dict) else {}


def _extract_pr_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = _text(value).rstrip("/")
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text.startswith("#") and text[1:].isdigit():
        return int(text[1:])
    tail = text.rsplit("/", 1)[-1]
    if tail.isdigit():
        return int(tail)
    match = re.search(r"#(\d+)$", text)
    if match:
        return int(match.group(1))
    return None


def _extract_pr_link(*sources: dict[str, Any] | None) -> dict[str, Any] | None:
    url = ""
    reference = ""
    number: int | None = None
    for source in sources:
        if not isinstance(source, dict):
            continue
        meta = _metadata(source)
        deliverable = _deliverable(source)
        for candidate in (
            source.get("pr_url"),
            source.get("pull_request_url"),
            source.get("adopted_pr"),
            deliverable.get("pr_url"),
            deliverable.get("adopted_pr"),
            meta.get("pr_url"),
            meta.get("pull_request_url"),
            meta.get("adopted_pr"),
        ):
            text = _text(candidate)
            if not text:
                continue
            if not url and re.match(r"^https?://", text):
                url = text
            elif not reference:
                reference = text
            if number is None:
                number = _extract_pr_number(text)
        for candidate in (
            source.get("pr_number"),
            source.get("pull_request_number"),
            source.get("adopted_pr"),
            deliverable.get("pr_number"),
            deliverable.get("pull_request_number"),
            deliverable.get("adopted_pr"),
            meta.get("pr_number"),
            meta.get("pull_request_number"),
            meta.get("adopted_pr"),
        ):
            parsed = _extract_pr_number(candidate)
            if parsed is not None:
                number = parsed
                break
    if not url and not reference and number is None:
        return None
    return {
        "url": url or None,
        "number": number,
        "reference": reference or None,
    }


def _explicit_missing_receipt(*sources: dict[str, Any] | None) -> bool:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for error_field in ("dispatch_error", "error"):
            lowered = _text(source.get(error_field)).lower()
            if "without receipt" in lowered or "missing receipt" in lowered:
                return True
        meta = _metadata(source)
        lowered = _text(meta.get("error")).lower()
        if "without receipt" in lowered or "missing receipt" in lowered:
            return True
    return False


def _is_superseded(*sources: dict[str, Any] | None) -> bool:
    for source in sources:
        if not isinstance(source, dict):
            continue
        if _text(source.get("status")).lower() == "superseded":
            return True
        meta = _metadata(source)
        if _text(meta.get("superseded_by")) or _text(meta.get("supersedes")):
            return True
    return False


def _merge_queue_status(queue_item: dict[str, Any] | None) -> str:
    if not isinstance(queue_item, dict):
        return ""
    return _text(queue_item.get("status")).lower()


def _receipt_expected(status: str, queue_status: str) -> bool:
    if queue_status in {"validating", "integrating", "needs_human", "merged", "failed"}:
        return True
    return status in {"completed", "needs_human", "merged"}


def _merge_readiness(
    *,
    status: str,
    queue_status: str,
    stale_heartbeat: bool,
    missing_receipt: bool,
    scope_violation: bool,
    superseded: bool,
    collisions: list[str],
) -> str:
    if superseded:
        return "superseded"
    if (
        collisions
        or stale_heartbeat
        or missing_receipt
        or scope_violation
        or queue_status in {"blocked", "failed"}
    ):
        return "blocked"
    if queue_status == "merged":
        return "merged"
    if queue_status in {"validating", "integrating"}:
        return queue_status
    if queue_status == "needs_human":
        return "review"
    if status in {"completed", "needs_human"}:
        return "ready"
    if status in {"leased", "dispatched", "queued", "active"}:
        return "in_progress"
    return status or queue_status or "unknown"


def _next_action(
    *,
    readiness: str,
    lane_health: str,
    terminal_outcome: str,
    stale_heartbeat: bool,
    missing_receipt: bool,
    scope_violation: bool,
    superseded: bool,
    collisions: list[str],
    queue_status: str,
    blockers: list[str],
) -> str:
    lowered_blockers = {_text(item).lower() for item in blockers if _text(item)}
    if superseded or lane_health == "superseded":
        return "Archive the superseded lane and keep the canonical lane."
    if lane_health == "expired":
        return "Salvage or reassign the expired lane before resuming work."
    if lane_health == "stalled":
        return "Inspect the stalled lane and decide whether to salvage, supersede, or reassign it."
    if collisions:
        return "Resolve the branch or file-scope collision before integrating."
    if scope_violation:
        return "Narrow the lane scope or split ownership before it can re-enter merge review."
    if "work_order_leasing_failed" in lowered_blockers:
        return "Reconcile or regenerate the managed worktree, then requeue the lane."
    if _text(terminal_outcome).lower() == "clean_exit_no_deliverable":
        return "Inspect why the lane produced no concrete deliverable before rerunning it."
    if "merge_gate_failed" in lowered_blockers or any(
        item.startswith("merge gate blocked:") for item in lowered_blockers
    ):
        return "Fix the merge gate or verification failure before rerunning the lane."
    if stale_heartbeat:
        return "Inspect the stale lane and decide whether to salvage or reassign it."
    if missing_receipt:
        return "Attach or regenerate the completion receipt before integration."
    if queue_status == "needs_human":
        return "Review the validated lane and decide whether it should merge."
    if readiness == "review":
        return "Review the validated lane and decide whether it should merge."
    if readiness == "ready":
        return "Queue or validate this lane for merge."
    if readiness == "merged":
        return "No action needed; the lane is already merged."
    return "Monitor the lane or reconcile it if progress stalls."


def _item_timestamp(item: dict[str, Any] | None, *keys: str) -> datetime:
    if not isinstance(item, dict):
        return datetime.min.replace(tzinfo=timezone.utc)
    for key in keys:
        parsed = _parse_iso_timestamp(item.get(key))
        if parsed is not None:
            return parsed
    meta = _metadata(item)
    for key in keys:
        parsed = _parse_iso_timestamp(meta.get(key))
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


def _sort_newest(items: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: _item_timestamp(item, *keys), reverse=True)


def _telemetry_summary() -> dict[str, Any]:
    """Return a compact operator-facing snapshot of autonomy lane telemetry."""
    try:
        return {
            "throughput_7d": _LANE_TELEMETRY.get_throughput(window_days=7),
            "success_rate_7d": _LANE_TELEMETRY.get_success_rate(window_days=7),
            "false_success_candidates_7d": _LANE_TELEMETRY.get_false_success_candidate_count(
                window_days=7
            ),
            "human_intervention_rate_7d": _LANE_TELEMETRY.get_human_intervention_rate(
                window_days=7
            ),
            "merge_yield_7d": _LANE_TELEMETRY.get_merge_yield(window_days=7),
            "avg_time_to_pr_seconds_7d": _LANE_TELEMETRY.get_avg_time_to_pr(window_days=7),
            "avg_time_to_merge_seconds_7d": _LANE_TELEMETRY.get_avg_time_to_merge(window_days=7),
        }
    except Exception:
        logger.exception("Failed to summarize lane telemetry")
        return {}


def _sync_lane_telemetry_from_lanes(lanes: list[dict[str, Any]]) -> None:
    """Backfill merge outcomes into existing supervisor telemetry rows."""
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = _text(lane.get("lane_id"))
        if not lane_id:
            continue
        existing = _LANE_TELEMETRY.get_lane("supervisor_work_order", lane_id)
        if existing is None:
            continue

        deliverable_type = _text(lane.get("deliverable_type")) or existing.deliverable_type
        missing_receipt = bool(lane.get("missing_receipt"))
        readiness = _text(lane.get("merge_readiness")).lower()
        false_success_candidate = bool(existing.false_success_candidate) or bool(
            readiness == "ready" and (missing_receipt or not deliverable_type)
        )
        metadata = dict(existing.metadata)
        metadata.update(
            {
                "merge_readiness": readiness or metadata.get("merge_readiness", ""),
                "lane_health": _text(lane.get("lane_health")) or metadata.get("lane_health", ""),
                "missing_receipt": missing_receipt,
                "decision_needed": bool(lane.get("decision_needed")),
            }
        )

        _LANE_TELEMETRY.record_lane(
            LaneTelemetryRecord(
                lane_kind=existing.lane_kind,
                lane_id=existing.lane_id,
                run_id=existing.run_id,
                task_id=existing.task_id,
                work_order_id=existing.work_order_id,
                project_id=existing.project_id,
                terminal_outcome=existing.terminal_outcome,
                worker_outcome=existing.worker_outcome,
                deliverable_type=deliverable_type,
                receipt_id=_text(lane.get("receipt_id")) or existing.receipt_id,
                human_intervention_required=existing.human_intervention_required,
                duration_seconds=existing.duration_seconds,
                pr_url=_text((lane.get("pr") or {}).get("url")) or existing.pr_url,
                pr_number=(lane.get("pr") or {}).get("number") or existing.pr_number,
                merge_ref=_text(lane.get("merge_ref")) or existing.merge_ref,
                merged_at=_text(lane.get("merged_at")) or existing.merged_at,
                time_to_pr_seconds=(
                    lane.get("time_to_pr_seconds")
                    if lane.get("time_to_pr_seconds") is not None
                    else existing.time_to_pr_seconds
                ),
                time_to_merge_seconds=(
                    lane.get("time_to_merge_seconds")
                    if lane.get("time_to_merge_seconds") is not None
                    else existing.time_to_merge_seconds
                ),
                false_success_candidate=false_success_candidate,
                timestamp=existing.timestamp,
                metadata=metadata,
            )
        )


def _lane_health(
    *,
    readiness: str,
    status: str,
    terminal_outcome: str,
    lease_status: str,
    stale_heartbeat: bool,
    missing_receipt: bool,
    scope_violation: bool,
    superseded: bool,
    collisions: list[str],
    blockers: list[str],
) -> str:
    if superseded or status == "discarded":
        return "superseded"
    lowered_blockers = {_text(item).lower() for item in blockers if _text(item)}
    if lowered_blockers.intersection({"stale_lease_reaped", "expired_lease_reaped"}):
        return "expired"
    if (
        scope_violation
        or _text(status).lower() == "scope_violation"
        or lowered_blockers.intersection({"scope_violation", "work_order_leasing_failed"})
    ):
        return "blocked"
    if lease_status == "expired" or status == "timed_out":
        return "expired"
    if stale_heartbeat or status in {"dispatch_failed", "failed"}:
        return "stalled"
    if lowered_blockers.intersection({"merge_gate_failed"}) or _text(terminal_outcome).lower() in {
        "clean_exit_no_deliverable",
        "blocked",
    }:
        return "blocked"
    if readiness == "merged":
        return "merged"
    if readiness == "blocked" or missing_receipt or scope_violation or collisions:
        return "blocked"
    return "healthy"


def _available_actions(
    *,
    canonical_lane: bool,
    readiness: str,
    lane_health: str,
    missing_receipt: bool,
    scope_violation: bool,
    superseded: bool,
    collisions: list[str],
    decision: str,
) -> list[str]:
    if superseded or lane_health == "superseded" or readiness == "superseded":
        return ["archive"]
    if readiness == "merged":
        return ["archive"]

    actions: list[str] = []
    if lane_health in {"expired", "stalled"}:
        actions.extend(["salvage", "reassign", "archive"])
    if collisions or scope_violation:
        actions.extend(["request_changes", "supersede"])
    if missing_receipt:
        actions.append("attach_receipt")
    if decision == "request_changes":
        actions.extend(["request_changes", "supersede"])
    if decision == "salvage":
        actions.extend(["salvage", "archive"])
    if readiness in {"ready", "review"} and not (collisions or scope_violation or missing_receipt):
        actions.extend(["merge", "cherry_pick", "request_changes"])
    if not canonical_lane and readiness not in {"merged", "superseded"}:
        actions.append("supersede")
    if not actions:
        actions.append("monitor")
    return list(dict.fromkeys(actions))


def build_integrator_view(
    *,
    runs: list[dict[str, Any]] | None = None,
    worktrees: list[dict[str, Any]] | None = None,
    claims: list[dict[str, Any]] | None = None,
    merge_queue: list[dict[str, Any]] | None = None,
    coordination: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Normalize coordination state into an integrator-facing lane view."""
    runs = [item for item in (runs or []) if isinstance(item, dict)]
    worktrees = [item for item in (worktrees or []) if isinstance(item, dict)]
    claims = [item for item in (claims or []) if isinstance(item, dict)]
    merge_queue = [item for item in (merge_queue or []) if isinstance(item, dict)]
    coordination = coordination if isinstance(coordination, dict) else {}
    now = now or datetime.now(timezone.utc)
    integrator = coordination.get("integrator", {})
    integrator = integrator if isinstance(integrator, dict) else {}

    tasks = _sort_newest(
        [dict(item) for item in integrator.get("developer_tasks", []) if isinstance(item, dict)],
        "updated_at",
        "created_at",
    )
    leases = _sort_newest(
        [dict(item) for item in integrator.get("leases", []) if isinstance(item, dict)],
        "updated_at",
        "created_at",
        "expires_at",
    )
    receipts = _sort_newest(
        [
            dict(item)
            for item in integrator.get("completion_receipts", [])
            if isinstance(item, dict)
        ],
        "created_at",
    )
    decisions = _sort_newest(
        [
            dict(item)
            for item in integrator.get("integration_decisions", [])
            if isinstance(item, dict)
        ],
        "created_at",
    )
    salvage_candidates = _sort_newest(
        [dict(item) for item in integrator.get("salvage_candidates", []) if isinstance(item, dict)],
        "updated_at",
        "created_at",
    )

    run_by_id: dict[str, dict[str, Any]] = {}
    work_order_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        run_id = _text(run.get("run_id"))
        if run_id:
            run_by_id[run_id] = run
        for work_order in run.get("work_orders", []):
            if not isinstance(work_order, dict):
                continue
            work_order_id = _first_text(work_order.get("work_order_id"), work_order.get("task_id"))
            if run_id and work_order_id:
                work_order_by_key[(run_id, work_order_id)] = work_order

    worktrees_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worktrees_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worktrees_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in worktrees:
        branch = _text(row.get("branch"))
        if branch:
            worktrees_by_branch[branch].append(row)
        path = _text(row.get("path"))
        if path:
            worktrees_by_path[path].append(row)
        session_id = _text(row.get("session_id"))
        if session_id:
            worktrees_by_session[session_id].append(row)

    claims_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sessions_by_path: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        session_id = _text(claim.get("session_id"))
        path = _text(claim.get("path"))
        if session_id:
            claims_by_session[session_id].append(claim)
        if session_id and path:
            sessions_by_path[path].add(session_id)

    worktree_branch_counts: dict[str, int] = defaultdict(int)
    work_order_branch_counts: dict[str, int] = defaultdict(int)
    task_branch_counts: dict[str, int] = defaultdict(int)
    queue_branch_counts: dict[str, int] = defaultdict(int)
    for branch, rows_for_branch in worktrees_by_branch.items():
        worktree_branch_counts[branch] += len(rows_for_branch)
    for run in runs:
        for work_order in run.get("work_orders", []):
            if not isinstance(work_order, dict):
                continue
            branch = _text(work_order.get("branch"))
            if branch:
                work_order_branch_counts[branch] += 1
    for task in tasks:
        branch = _text(task.get("branch"))
        if branch:
            task_branch_counts[branch] += 1

    queue_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    queue_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    queue_by_receipt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    queue_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in merge_queue:
        branch = _text(item.get("branch"))
        session_id = _text(item.get("session_id"))
        meta = _metadata(item)
        receipt_id = _first_text(item.get("receipt_id"), meta.get("receipt_id"))
        task_id = _first_text(item.get("task_id"), meta.get("task_id"))
        if branch:
            queue_by_branch[branch].append(item)
            queue_branch_counts[branch] += 1
        if session_id:
            queue_by_session[session_id].append(item)
        if receipt_id:
            queue_by_receipt[receipt_id].append(item)
        if task_id:
            queue_by_task[task_id].append(item)

    scope_violations = coordination.get("scope_violations", [])
    scope_violation_by_lease: dict[str, dict[str, Any]] = {}
    scope_violation_by_session_branch: dict[tuple[str, str], dict[str, Any]] = {}
    for item in scope_violations:
        if not isinstance(item, dict):
            continue
        lease_id = _text(item.get("lease_id"))
        if lease_id:
            scope_violation_by_lease[lease_id] = item
        session_id = _text(item.get("owner_session_id"))
        branch = _text(item.get("branch"))
        if session_id or branch:
            scope_violation_by_session_branch[(session_id, branch)] = item

    tasks_by_key: dict[str, dict[str, Any]] = {}
    tasks_by_run_task: dict[tuple[str, str], dict[str, Any]] = {}
    tasks_by_session_branch: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        task_key = _text(task.get("task_key"))
        if task_key:
            tasks_by_key[task_key] = task
        run_id = _text(task.get("run_id"))
        task_id = _text(task.get("task_id"))
        if run_id and task_id:
            tasks_by_run_task[(run_id, task_id)] = task
        session_id = _text(task.get("owner_session_id"))
        branch = _text(task.get("branch"))
        if session_id or branch:
            tasks_by_session_branch[(session_id, branch)] = task

    leases_by_id: dict[str, dict[str, Any]] = {}
    leases_by_task: dict[str, dict[str, Any]] = {}
    leases_by_session_branch: dict[tuple[str, str], dict[str, Any]] = {}
    for lease in leases:
        lease_id = _text(lease.get("lease_id"))
        if lease_id and lease_id not in leases_by_id:
            leases_by_id[lease_id] = lease
        task_id = _text(lease.get("task_id"))
        if task_id and task_id not in leases_by_task:
            leases_by_task[task_id] = lease
        key = (_text(lease.get("owner_session_id")), _text(lease.get("branch")))
        if (key[0] or key[1]) and key not in leases_by_session_branch:
            leases_by_session_branch[key] = lease

    receipts_by_id: dict[str, dict[str, Any]] = {}
    receipts_by_task: dict[str, dict[str, Any]] = {}
    receipts_by_lease: dict[str, dict[str, Any]] = {}
    receipts_by_session_branch: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipts:
        receipt_id = _text(receipt.get("receipt_id"))
        if receipt_id and receipt_id not in receipts_by_id:
            receipts_by_id[receipt_id] = receipt
        task_id = _text(receipt.get("task_id"))
        if task_id and task_id not in receipts_by_task:
            receipts_by_task[task_id] = receipt
        lease_id = _text(receipt.get("lease_id"))
        if lease_id and lease_id not in receipts_by_lease:
            receipts_by_lease[lease_id] = receipt
        key = (_text(receipt.get("owner_session_id")), _text(receipt.get("branch")))
        if (key[0] or key[1]) and key not in receipts_by_session_branch:
            receipts_by_session_branch[key] = receipt

    decisions_by_receipt: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        receipt_id = _text(decision.get("receipt_id"))
        if receipt_id and receipt_id not in decisions_by_receipt:
            decisions_by_receipt[receipt_id] = decision

    salvage_by_branch: dict[str, dict[str, Any]] = {}
    salvage_by_path: dict[str, dict[str, Any]] = {}
    for candidate in salvage_candidates:
        branch = _text(candidate.get("branch"))
        if branch and branch not in salvage_by_branch:
            salvage_by_branch[branch] = candidate
        path = _text(candidate.get("worktree_path"))
        if path and path not in salvage_by_path:
            salvage_by_path[path] = candidate

    lanes: list[dict[str, Any]] = []
    seen_task_keys: set[str] = set()
    seen_work_order_keys: set[tuple[str, str]] = set()
    seen_worktree_keys: set[tuple[str, str]] = set()
    seen_queue_keys: set[tuple[str, str, str]] = set()

    def _queue_key(item: dict[str, Any] | None) -> tuple[str, str, str]:
        item = item if isinstance(item, dict) else {}
        return (
            _text(item.get("id")),
            _text(item.get("branch")),
            _text(item.get("session_id")),
        )

    def _pick_first(*candidates: Any) -> dict[str, Any] | None:
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        return item
        return None

    def _mark_lane_sources(
        *,
        task: dict[str, Any] | None = None,
        worktree_row: dict[str, Any] | None = None,
        queue_item: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(task, dict):
            task_key = _text(task.get("task_key"))
            if task_key:
                seen_task_keys.add(task_key)
            run_id = _text(task.get("run_id"))
            task_id = _text(task.get("task_id"))
            if run_id and task_id:
                seen_work_order_keys.add((run_id, task_id))
        if isinstance(worktree_row, dict):
            seen_worktree_keys.add(
                (_text(worktree_row.get("session_id")), _text(worktree_row.get("branch")))
            )
        if isinstance(queue_item, dict):
            seen_queue_keys.add(_queue_key(queue_item))

    def build_lane(
        *,
        source: str,
        run: dict[str, Any] | None = None,
        work_order: dict[str, Any] | None = None,
        task: dict[str, Any] | None = None,
        lease: dict[str, Any] | None = None,
        receipt: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        salvage: dict[str, Any] | None = None,
        worktree_row: dict[str, Any] | None = None,
        queue_item: dict[str, Any] | None = None,
        canonical_lane: bool,
    ) -> dict[str, Any]:
        work_order = work_order or {}
        task = task or {}
        lease = lease or {}
        receipt = receipt or {}
        decision = decision or {}
        salvage = salvage or {}
        worktree_row = worktree_row or {}
        queue_item = queue_item or {}
        run = run or {}
        task_meta = _metadata(task)
        work_order_meta = _metadata(work_order)
        lease_meta = _metadata(lease)
        receipt_meta = _metadata(receipt)
        queue_meta = _metadata(queue_item)
        inferred_status = ""
        if bool(worktree_row.get("has_lock")) and bool(worktree_row.get("pid_alive")):
            inferred_status = "active"
        elif bool(worktree_row.get("has_lock")):
            inferred_status = "needs_human"
        status = _first_text(
            task.get("status"),
            work_order.get("status"),
            worktree_row.get("status"),
            inferred_status,
            "unknown",
        ).lower()
        branch = _first_text(
            task.get("branch"),
            lease.get("branch"),
            receipt.get("branch"),
            work_order.get("branch"),
            worktree_row.get("branch"),
            queue_item.get("branch"),
        )
        worktree_path = _first_text(
            task.get("worktree_path"),
            lease.get("worktree_path"),
            receipt.get("worktree_path"),
            work_order.get("worktree_path"),
            worktree_row.get("path"),
            queue_meta.get("integration_workspace_path"),
        )
        session_id = _first_text(
            task.get("owner_session_id"),
            lease.get("owner_session_id"),
            receipt.get("owner_session_id"),
            lease_meta.get("owner_session_id"),
            work_order_meta.get("owner_session_id"),
            worktree_row.get("session_id"),
            queue_item.get("session_id"),
        )
        owner_agent = _first_text(
            task.get("owner_agent"),
            lease.get("owner_agent"),
            receipt.get("owner_agent"),
            work_order.get("target_agent"),
            work_order_meta.get("owner_agent"),
            worktree_row.get("agent"),
        )
        reviewer_agent = _first_text(
            task.get("reviewer_agent"),
            receipt_meta.get("reviewer_agent"),
            work_order.get("reviewer_agent"),
            work_order_meta.get("reviewer_agent"),
        )
        task_key = _first_text(
            task.get("task_key"),
            receipt_meta.get("task_key"),
            lease_meta.get("task_key"),
        )
        task_id = _first_text(
            task.get("task_id"),
            receipt.get("task_id"),
            lease.get("task_id"),
            work_order.get("work_order_id"),
            work_order.get("task_id"),
        )
        lease_id = _first_text(
            task.get("lease_id"),
            lease.get("lease_id"),
            receipt.get("lease_id"),
            work_order.get("lease_id"),
            work_order_meta.get("lease_id"),
        )
        receipt_id = _first_text(
            task.get("receipt_id"),
            receipt.get("receipt_id"),
            lease_meta.get("last_receipt_id"),
            _extract_receipt_id(work_order, queue_item),
        )
        queue_status = _merge_queue_status(queue_item)
        decision_type = _first_text(
            decision.get("decision"),
            queue_meta.get("integration_decision"),
        ).lower()
        branch_collision = bool(
            branch
            and (
                worktree_branch_counts.get(branch, 0) > 1
                or work_order_branch_counts.get(branch, 0) > 1
                or task_branch_counts.get(branch, 0) > 1
                or queue_branch_counts.get(branch, 0) > 1
            )
        )

        lease_claimed_paths = [
            _text(path) for path in lease.get("claimed_paths", []) if _text(path)
        ]
        claimed_paths = sorted(
            set(lease_claimed_paths).union(
                {
                    _text(claim.get("path"))
                    for claim in claims_by_session.get(session_id, [])
                    if _text(claim.get("path"))
                }
            )
        )
        file_scope = [
            _text(path)
            for path in (
                task.get("allowed_paths")
                or work_order.get("file_scope")
                or lease.get("allowed_globs")
                or []
            )
            if _text(path)
        ]
        collision_reasons: list[str] = []
        if branch_collision:
            collision_reasons.append(f"branch:{branch}")
        for path in claimed_paths or file_scope:
            owners = sessions_by_path.get(path, set())
            if len(owners) > 1:
                collision_reasons.append(f"path:{path}")
        collision_reasons = sorted(set(collision_reasons))

        heartbeat_source = _first_text(
            task.get("updated_at"),
            lease.get("updated_at"),
            work_order.get("last_progress_at"),
            work_order.get("last_observed_at"),
            work_order.get("dispatched_at"),
            worktree_row.get("last_activity"),
            queue_item.get("updated_at"),
            receipt.get("created_at"),
        )
        heartbeat_age_seconds = _age_seconds(heartbeat_source, now=now)
        lease_status = _text(lease.get("status")).lower()
        worktree_active = bool(worktree_row.get("has_lock")) and bool(worktree_row.get("pid_alive"))
        stale_lease_blocked = (
            status in {"leased", "dispatched", "active", "integrating"}
            and queue_status not in {"validating", "integrating"}
            and (
                lease_status in {"released", "expired"}
                or (lease_id and not lease_status and not worktree_active)
            )
        )
        lane_in_flight = (
            lease_status == "active"
            or queue_status in {"validating", "integrating"}
            or (
                status in {"leased", "dispatched", "active", "integrating"}
                and not stale_lease_blocked
                and not lease_status
                and worktree_active
            )
        )
        stale_heartbeat = bool(
            (
                (work_order or task or lease)
                and lane_in_flight
                and heartbeat_age_seconds is not None
                and heartbeat_age_seconds >= _STALE_LANE_AFTER_SECONDS
            )
            or (bool(worktree_row.get("has_lock")) and not bool(worktree_row.get("pid_alive")))
        )

        superseded = _is_superseded(task, work_order, queue_item) or status == "discarded"
        base_sha = _first_text(
            receipt.get("base_sha"),
            queue_meta.get("base_sha"),
            task.get("base_sha"),
            task_meta.get("base_sha"),
            work_order.get("base_sha"),
            work_order_meta.get("base_sha"),
            lease_meta.get("base_sha"),
        )
        head_sha = _first_text(
            receipt.get("head_sha"),
            queue_meta.get("head_sha"),
            task.get("head_sha"),
            task_meta.get("head_sha"),
            work_order.get("head_sha"),
            work_order_meta.get("head_sha"),
            salvage.get("head_sha"),
        )
        commit_shas = _first_list(
            receipt.get("commit_shas"),
            decision.get("chosen_commits"),
            queue_meta.get("commit_shas"),
            queue_meta.get("chosen_commits"),
            task.get("commit_shas"),
            task_meta.get("commit_shas"),
            work_order.get("commit_shas"),
            work_order_meta.get("commit_shas"),
        )
        changed_files = _first_list(
            receipt.get("changed_paths"),
            queue_meta.get("changed_paths"),
            queue_meta.get("changed_files"),
            task.get("changed_paths"),
            task.get("changed_files"),
            task_meta.get("changed_paths"),
            task_meta.get("changed_files"),
            work_order.get("changed_paths"),
            work_order.get("changed_files"),
            work_order_meta.get("changed_paths"),
            work_order_meta.get("changed_files"),
            salvage.get("changed_paths"),
        )
        worker_outcome = _first_text(
            work_order.get("worker_outcome"),
            task.get("worker_outcome"),
            task_meta.get("worker_outcome"),
            receipt_meta.get("worker_outcome"),
        )
        blocker_dict = next(
            (
                candidate
                for candidate in (
                    work_order.get("blocker"),
                    task.get("blocker"),
                    task_meta.get("blocker"),
                    queue_meta.get("blocker"),
                )
                if isinstance(candidate, dict)
            ),
            None,
        )
        synthetic_work_order = {
            "status": status,
            "worker_outcome": worker_outcome,
            "branch": branch,
            "commit_shas": commit_shas,
            "pr_url": _first_text(
                work_order.get("pr_url"),
                task.get("pr_url"),
                receipt.get("pr_url"),
                queue_item.get("pr_url"),
                queue_meta.get("pr_url"),
            ),
            "adopted_pr": _first_text(
                work_order.get("adopted_pr"),
                task.get("adopted_pr"),
                receipt.get("adopted_pr"),
                queue_item.get("adopted_pr"),
                queue_meta.get("adopted_pr"),
            ),
            "blockers": [
                *_text_list(work_order.get("blockers")),
                *_text_list(task.get("blocked_by")),
                *_text_list(task.get("blockers")),
                *_text_list(queue_meta.get("blockers")),
                *_text_list(receipt.get("blockers")),
                *(["stale_lease_reaped"] if stale_lease_blocked else []),
            ],
            "dispatch_error": _first_text(
                work_order.get("dispatch_error"),
                task.get("dispatch_error"),
                queue_item.get("error"),
                queue_meta.get("error"),
            ),
            "failure_reason": _first_text(
                work_order.get("failure_reason"),
                task.get("failure_reason"),
                queue_meta.get("failure_reason"),
            ),
            "blocking_question": _first_text(
                work_order.get("blocking_question"),
                task.get("blocking_question"),
                queue_meta.get("blocking_question"),
            ),
            "blocker": blocker_dict,
        }
        qualification = qualify_work_order_terminal_state(synthetic_work_order)
        terminal_outcome = qualification.terminal_outcome
        qualification_reasons = list(qualification.reasons)
        blocked_reason = qualification.blocked_reason
        reaped_receipt_backed_lane = (
            receipt_id
            and qualification.deliverable is not None
            and qualification_reasons
            and all(
                _text(reason).lower() in {"stale_lease_reaped", "expired_lease_reaped"}
                for reason in qualification_reasons
            )
        )
        if reaped_receipt_backed_lane:
            terminal_outcome = (
                "pr_adopted"
                if qualification.deliverable_type == "adopted_pr"
                else "deliverable_created"
            )
        if (
            receipt_id
            and qualification.deliverable is not None
            and terminal_outcome in {"deliverable_created", "pr_adopted"}
        ):
            qualification_reasons = [
                reason
                for reason in qualification_reasons
                if _text(reason).lower() not in {"stale_lease_reaped", "expired_lease_reaped"}
            ]
            if _text(blocked_reason).lower() in {"stale_lease_reaped", "expired_lease_reaped"}:
                blocked_reason = None
        receipt_expected = receipt_expected_for_lane(
            status=status,
            queue_status=queue_status,
            lane_in_flight=lane_in_flight,
            decision_type=decision_type,
            terminal_outcome=terminal_outcome if terminal_outcome != "unknown" else None,
            lease_id=lease_id,
            lease_status=lease_status,
            deliverable_present=qualification.deliverable is not None,
            blockers=qualification_reasons,
        )
        missing_receipt = _explicit_missing_receipt(work_order, queue_item) or (
            receipt_expected and not receipt_id
        )
        tests_run = _first_list(
            receipt.get("tests_run"),
            queue_meta.get("tests_run"),
        )
        validations_run = _first_list(
            receipt.get("validations_run"),
            queue_meta.get("validations_run"),
            tests_run,
        )
        assumptions = _first_list(
            receipt.get("assumptions"),
            queue_meta.get("assumptions"),
        )
        receipt_blockers = _first_list(
            receipt.get("blockers"),
            queue_meta.get("blockers"),
        )
        risks = _first_list(
            receipt.get("risks"),
            queue_meta.get("risks"),
        )
        artifact_hash = _first_text(
            receipt.get("artifact_hash"),
            queue_meta.get("artifact_hash"),
        )
        receipt_created_at = _first_text(receipt.get("created_at"))
        receipt_outcome = _first_text(
            receipt.get("outcome"),
            queue_meta.get("outcome"),
        )
        pr_created_at = _first_text(
            queue_item.get("pr_created_at"),
            queue_meta.get("pr_created_at"),
            queue_item.get("pull_request_created_at"),
            queue_meta.get("pull_request_created_at"),
        )
        merged_at = _first_text(
            queue_item.get("merged_at"),
            queue_meta.get("merged_at"),
        )
        if not merged_at and queue_status == "merged":
            merged_at = _first_text(queue_item.get("updated_at"), decision.get("created_at"))
        merge_ref = _first_text(
            queue_item.get("merge_sha"),
            queue_meta.get("merge_sha"),
            queue_item.get("merge_ref"),
            queue_meta.get("merge_ref"),
            next(
                (_text(item) for item in decision.get("chosen_commits", []) if _text(item)),
                "",
            ),
            decision.get("target_branch"),
        )
        time_to_pr_seconds = _seconds_between(receipt_created_at, pr_created_at)
        time_to_merge_seconds = _seconds_between(receipt_created_at, merged_at)
        confidence_value: float | None = None
        for candidate in (receipt.get("confidence"), queue_meta.get("confidence")):
            if isinstance(candidate, (int, float)):
                confidence_value = float(candidate)
                break
        receipt_status = (
            "present"
            if receipt_id
            else "missing"
            if missing_receipt
            else "pending"
            if (
                lane_in_flight
                or receipt_expected
                or any(
                    (
                        base_sha,
                        head_sha,
                        commit_shas,
                        changed_files,
                        validations_run,
                        tests_run,
                        risks,
                        artifact_hash,
                    )
                )
            )
            else ""
        )

        scope_violation_record = (
            scope_violation_by_lease.get(lease_id)
            if lease_id
            else scope_violation_by_session_branch.get((session_id, branch))
        )
        if not isinstance(scope_violation_record, dict):
            fallback_violation = task_meta.get("last_scope_violation")
            if isinstance(fallback_violation, dict):
                scope_violation_record = fallback_violation
        lowered_reasons = {_text(item).lower() for item in qualification_reasons if _text(item)}
        scope_violation = (
            isinstance(scope_violation_record, dict)
            or status == "scope_violation"
            or worker_outcome == "scope_violation"
            or "scope_violation" in lowered_reasons
        )

        terminal_blocked = terminal_outcome in {
            "blocked",
            "needs_human",
            "clean_exit_no_deliverable",
            "crash",
            "timeout",
        }

        if superseded or decision_type == "discard":
            readiness = "superseded"
        elif queue_status == "merged" or status == "merged":
            readiness = "merged"
        elif decision_type in {"merge", "cherry_pick"}:
            readiness = "merged" if queue_status == "merged" else "integrating"
        elif terminal_blocked:
            readiness = "blocked"
        elif decision_type == "pending_review":
            readiness = "review"
        elif decision_type in {"request_changes", "salvage"}:
            readiness = "blocked"
        elif (
            receipt_id
            and qualification.deliverable is not None
            and terminal_outcome in {"deliverable_created", "pr_adopted"}
            and status
            in {
                "completed",
                "needs_human",
                "changes_requested",
                "integrating",
            }
        ):
            readiness = "blocked" if missing_receipt else "review"
        else:
            readiness = _merge_readiness(
                status=status,
                queue_status=queue_status,
                stale_heartbeat=stale_heartbeat,
                missing_receipt=missing_receipt,
                scope_violation=scope_violation,
                superseded=superseded,
                collisions=collision_reasons,
            )

        effective_lease_status = lease_status
        if reaped_receipt_backed_lane and lease_status in {"expired", "released"}:
            effective_lease_status = ""

        if superseded:
            lease_health = "superseded"
        elif reaped_receipt_backed_lane:
            lease_health = "completed"
        elif lease_status == "expired" or status == "timed_out":
            lease_health = "expired"
        elif stale_heartbeat:
            lease_health = "stalled"
        elif lease_status in {"completed", "released"}:
            lease_health = lease_status
        elif lease_id or bool(worktree_row.get("has_lock")):
            lease_health = "healthy"
        elif status in {"leased", "dispatched"}:
            lease_health = "missing"
        else:
            lease_health = "idle"
        lane_health = _lane_health(
            readiness=readiness,
            status=status,
            terminal_outcome=terminal_outcome,
            lease_status=effective_lease_status,
            stale_heartbeat=stale_heartbeat,
            missing_receipt=missing_receipt,
            scope_violation=scope_violation,
            superseded=superseded,
            collisions=collision_reasons,
            blockers=qualification_reasons,
        )

        title = _first_text(
            task.get("title"),
            work_order.get("title"),
            work_order.get("description"),
            queue_item.get("title"),
            run.get("goal"),
            branch,
            session_id,
            "lane",
        )
        pr = _extract_pr_link(receipt, work_order, queue_item)
        receipt_summary = (
            {
                "status": receipt_status,
                "receipt_id": receipt_id or None,
                "task_id": task_id or None,
                "lease_id": lease_id or None,
                "agent_id": owner_agent or None,
                "session_id": session_id or None,
                "branch": branch or None,
                "worktree_path": worktree_path or None,
                "confidence": confidence_value,
                "outcome": receipt_outcome or None,
                "created_at": receipt_created_at or None,
                "artifact_hash": artifact_hash or None,
                "base_sha": base_sha or None,
                "head_sha": head_sha or None,
                "commit_shas": commit_shas,
                "changed_files": changed_files,
                "tests_run": tests_run,
                "validations_run": validations_run,
                "assumptions": assumptions,
                "blockers": receipt_blockers,
                "risks": risks,
                "pr": pr,
            }
            if receipt_status
            else None
        )
        blockers = sorted(
            {
                *[_text(item) for item in task.get("blocked_by", []) if _text(item)],
                *[_text(item) for item in work_order.get("blockers", []) if _text(item)],
                *[_text(item) for item in qualification_reasons if _text(item)],
                *collision_reasons,
            }
        )
        if blocked_reason:
            blockers.append(blocked_reason)
        if stale_heartbeat:
            blockers.append("stale_heartbeat")
        if missing_receipt:
            blockers.append("missing_receipt")
        if scope_violation:
            blockers.append("scope_violation")
        if superseded:
            blockers.append("superseded")
        if decision_type in {"request_changes", "salvage"}:
            blockers.append(f"decision:{decision_type}")
        if reaped_receipt_backed_lane:
            blockers = [
                blocker
                for blocker in blockers
                if _text(blocker).lower() not in {"stale_lease_reaped", "expired_lease_reaped"}
            ]
        blockers = sorted(set(blockers))

        available_actions = _available_actions(
            canonical_lane=canonical_lane,
            readiness=readiness,
            lane_health=lane_health,
            missing_receipt=missing_receipt,
            scope_violation=scope_violation,
            superseded=superseded,
            collisions=collision_reasons,
            decision=decision_type,
        )
        decision_needed = canonical_lane and readiness in {"ready", "review", "blocked"}
        lane_id = _first_text(
            task_key,
            work_order.get("work_order_id"),
            queue_item.get("id"),
            branch,
            session_id,
            worktree_path,
            title,
        )
        return {
            "lane_id": lane_id,
            "source": source,
            "run_id": _text(run.get("run_id")),
            "work_order_id": _text(work_order.get("work_order_id")),
            "task_key": task_key or None,
            "task_id": task_id or None,
            "title": title,
            "status": status,
            "terminal_outcome": terminal_outcome,
            "deliverable_type": qualification.deliverable_type,
            "worker_outcome": worker_outcome or None,
            "canonical_lane": canonical_lane,
            "owner_agent": owner_agent or None,
            "reviewer_agent": reviewer_agent or None,
            "owner_session_id": session_id or None,
            "branch": branch or None,
            "worktree_path": worktree_path or None,
            "lease_id": lease_id or None,
            "lease_status": lease_status or None,
            "lease_health": lease_health,
            "lane_health": lane_health,
            "heartbeat_at": heartbeat_source or None,
            "heartbeat_age_seconds": round(heartbeat_age_seconds, 1)
            if heartbeat_age_seconds is not None
            else None,
            "stale_heartbeat": stale_heartbeat,
            "claimed_paths": claimed_paths,
            "file_scope": file_scope,
            "queue_item_id": _text(queue_item.get("id")) or None,
            "merge_queue_status": queue_status or None,
            "receipt_id": receipt_id or None,
            "receipt_summary": receipt_summary,
            "integration_decision": decision_type or None,
            "integration_decision_id": _text(decision.get("decision_id")) or None,
            "decision_context": {
                "rationale": _text(decision.get("rationale")) or None,
                "followups": [_text(item) for item in decision.get("followups", []) if _text(item)],
                "chosen_commits": [
                    _text(item) for item in decision.get("chosen_commits", []) if _text(item)
                ],
            }
            if decision_type
            else None,
            "missing_receipt": missing_receipt,
            "receipt_expected": receipt_expected,
            "scope_violation": scope_violation_record,
            "scope_violation_detected": scope_violation,
            "superseded": superseded,
            "collisions": collision_reasons,
            "pr": pr,
            "merge_ref": merge_ref or None,
            "merged_at": merged_at or None,
            "time_to_pr_seconds": time_to_pr_seconds,
            "time_to_merge_seconds": time_to_merge_seconds,
            "merge_readiness": readiness,
            "decision_needed": decision_needed,
            "available_actions": available_actions,
            "salvage_candidate_id": _text(salvage.get("candidate_id")) or None,
            "blockers": blockers,
            "next_action": _next_action(
                readiness=readiness,
                lane_health=lane_health,
                terminal_outcome=terminal_outcome,
                stale_heartbeat=stale_heartbeat,
                missing_receipt=missing_receipt,
                scope_violation=scope_violation,
                superseded=superseded,
                collisions=collision_reasons,
                queue_status=queue_status,
                blockers=blockers,
            ),
        }

    for task in tasks:
        run_id = _text(task.get("run_id"))
        task_id = _text(task.get("task_id"))
        worktree_path = _text(task.get("worktree_path"))
        branch = _text(task.get("branch"))
        session_id = _text(task.get("owner_session_id"))
        receipt_id = _text(task.get("receipt_id"))
        lease_id = _text(task.get("lease_id"))
        matched_run = run_by_id.get(run_id, {})
        work_order = work_order_by_key.get((run_id, task_id), {})
        matched_lease = _pick_first(
            leases_by_id.get(lease_id),
            leases_by_task.get(task_id),
            leases_by_session_branch.get((session_id, branch)),
        )
        matched_receipt = _pick_first(
            receipts_by_id.get(receipt_id),
            receipts_by_lease.get(lease_id),
            receipts_by_task.get(task_id),
            receipts_by_session_branch.get((session_id, branch)),
        )
        effective_receipt_id = _first_text(
            receipt_id,
            matched_receipt.get("receipt_id") if isinstance(matched_receipt, dict) else "",
        )
        matched_decision = _pick_first(
            decisions_by_receipt.get(effective_receipt_id),
        )
        matched_row = _pick_first(
            worktrees_by_path.get(worktree_path, []),
            worktrees_by_session.get(session_id, []),
            worktrees_by_branch.get(branch, []),
        )
        matched_queue = _pick_first(
            queue_by_receipt.get(effective_receipt_id, []),
            queue_by_task.get(task_id, []),
            queue_by_branch.get(branch, []),
            queue_by_session.get(session_id, []),
        )
        matched_salvage = _pick_first(
            salvage_by_path.get(worktree_path),
            salvage_by_branch.get(branch),
        )
        _mark_lane_sources(task=task, worktree_row=matched_row, queue_item=matched_queue)
        lanes.append(
            build_lane(
                source="task",
                run=matched_run,
                work_order=work_order,
                task=task,
                lease=matched_lease,
                receipt=matched_receipt,
                decision=matched_decision,
                salvage=matched_salvage,
                worktree_row=matched_row,
                queue_item=matched_queue,
                canonical_lane=True,
            )
        )

    for run in runs:
        for work_order in run.get("work_orders", []):
            if not isinstance(work_order, dict):
                continue
            run_id = _text(run.get("run_id"))
            work_order_id = _first_text(work_order.get("work_order_id"), work_order.get("task_id"))
            if (run_id, work_order_id) in seen_work_order_keys:
                continue
            matched_task = tasks_by_run_task.get((run_id, work_order_id))
            if (
                isinstance(matched_task, dict)
                and _text(matched_task.get("task_key")) in seen_task_keys
            ):
                continue
            branch = _text(work_order.get("branch"))
            worktree_path = _text(work_order.get("worktree_path"))
            session_id = _first_text(
                work_order.get("owner_session_id"),
                _metadata(work_order).get("owner_session_id"),
            )
            lease_id = _first_text(
                work_order.get("lease_id"), _metadata(work_order).get("lease_id")
            )
            receipt_id = _first_text(
                work_order.get("receipt_id"),
                _metadata(work_order).get("last_receipt_id"),
            )
            matched_row = _pick_first(
                worktrees_by_path.get(worktree_path, []),
                worktrees_by_session.get(session_id, []),
                worktrees_by_branch.get(branch, []),
            )
            matched_lease = _pick_first(
                leases_by_id.get(lease_id),
                leases_by_task.get(work_order_id),
                leases_by_session_branch.get((session_id, branch)),
            )
            matched_receipt = _pick_first(
                receipts_by_id.get(receipt_id),
                receipts_by_lease.get(lease_id),
                receipts_by_task.get(work_order_id),
                receipts_by_session_branch.get((session_id, branch)),
            )
            effective_receipt_id = _first_text(
                receipt_id,
                matched_receipt.get("receipt_id") if isinstance(matched_receipt, dict) else "",
            )
            matched_decision = _pick_first(decisions_by_receipt.get(effective_receipt_id))
            queue_item = _pick_first(
                queue_by_receipt.get(effective_receipt_id, []),
                queue_by_task.get(work_order_id, []),
                queue_by_branch.get(branch, []),
                queue_by_session.get(session_id, []),
            )
            matched_salvage = _pick_first(
                salvage_by_path.get(worktree_path),
                salvage_by_branch.get(branch),
            )
            _mark_lane_sources(
                task=matched_task,
                worktree_row=matched_row,
                queue_item=queue_item,
            )
            lanes.append(
                build_lane(
                    source="swarm",
                    run=run,
                    work_order=work_order,
                    task=matched_task,
                    lease=matched_lease,
                    receipt=matched_receipt,
                    decision=matched_decision,
                    salvage=matched_salvage,
                    worktree_row=matched_row,
                    queue_item=queue_item,
                    canonical_lane=not bool(tasks),
                )
            )

    for row in worktrees:
        key = (_text(row.get("session_id")), _text(row.get("branch")))
        if key in seen_worktree_keys:
            continue
        branch = _text(row.get("branch"))
        session_id = _text(row.get("session_id"))
        matched_task = _pick_first(tasks_by_session_branch.get((session_id, branch)))
        lease_id = _first_text(
            matched_task.get("lease_id") if isinstance(matched_task, dict) else "",
        )
        receipt_id = _first_text(
            matched_task.get("receipt_id") if isinstance(matched_task, dict) else "",
        )
        matched_lease = _pick_first(
            leases_by_id.get(lease_id),
            leases_by_session_branch.get((session_id, branch)),
        )
        matched_receipt = _pick_first(
            receipts_by_id.get(receipt_id),
            receipts_by_lease.get(lease_id),
            receipts_by_session_branch.get((session_id, branch)),
        )
        effective_receipt_id = _first_text(
            receipt_id,
            matched_receipt.get("receipt_id") if isinstance(matched_receipt, dict) else "",
        )
        matched_decision = _pick_first(decisions_by_receipt.get(effective_receipt_id))
        queue_item = _pick_first(
            queue_by_receipt.get(effective_receipt_id, []),
            queue_by_task.get(
                _text(matched_task.get("task_id")) if isinstance(matched_task, dict) else "",
                [],
            ),
            queue_by_branch.get(branch, []),
            queue_by_session.get(session_id, []),
        )
        matched_salvage = _pick_first(
            salvage_by_path.get(_text(row.get("path"))),
            salvage_by_branch.get(branch),
        )
        _mark_lane_sources(task=matched_task, worktree_row=row, queue_item=queue_item)
        lanes.append(
            build_lane(
                source="fleet",
                task=matched_task,
                lease=matched_lease,
                receipt=matched_receipt,
                decision=matched_decision,
                salvage=matched_salvage,
                worktree_row=row,
                queue_item=queue_item,
                canonical_lane=False,
            )
        )

    for item in merge_queue:
        queue_key = _queue_key(item)
        if queue_key in seen_queue_keys:
            continue
        session_id = _text(item.get("session_id"))
        branch = _text(item.get("branch"))
        meta = _metadata(item)
        receipt_id = _first_text(item.get("receipt_id"), meta.get("receipt_id"))
        task_id = _first_text(item.get("task_id"), meta.get("task_id"))
        matched_task = _pick_first(
            tasks_by_session_branch.get((session_id, branch)),
            tasks_by_run_task.get((_text(meta.get("run_id")), task_id)),
        )
        lease_id = _first_text(
            meta.get("lease_id"),
            matched_task.get("lease_id") if isinstance(matched_task, dict) else "",
        )
        matched_lease = _pick_first(
            leases_by_id.get(lease_id),
            leases_by_task.get(task_id),
            leases_by_session_branch.get((session_id, branch)),
        )
        matched_receipt = _pick_first(
            receipts_by_id.get(receipt_id),
            receipts_by_lease.get(lease_id),
            receipts_by_task.get(task_id),
            receipts_by_session_branch.get((session_id, branch)),
        )
        effective_receipt_id = _first_text(
            receipt_id,
            matched_receipt.get("receipt_id") if isinstance(matched_receipt, dict) else "",
        )
        matched_decision = _pick_first(decisions_by_receipt.get(effective_receipt_id))
        matched_row = _pick_first(
            worktrees_by_session.get(session_id, []),
            worktrees_by_branch.get(branch, []),
        )
        matched_salvage = _pick_first(
            salvage_by_branch.get(branch),
            salvage_by_path.get(
                _text(matched_row.get("path")) if isinstance(matched_row, dict) else ""
            ),
        )
        _mark_lane_sources(task=matched_task, worktree_row=matched_row, queue_item=item)
        lanes.append(
            build_lane(
                source="merge_queue",
                task=matched_task,
                lease=matched_lease,
                receipt=matched_receipt,
                decision=matched_decision,
                salvage=matched_salvage,
                worktree_row=matched_row if isinstance(matched_row, dict) else None,
                queue_item=item,
                canonical_lane=False,
            )
        )

    def lane_sort_key(item: dict[str, Any]) -> tuple[int, int, str, str]:
        readiness = _text(item.get("merge_readiness"))
        priority = {
            "blocked": 0,
            "review": 1,
            "ready": 2,
            "in_progress": 3,
            "validating": 4,
            "integrating": 5,
            "merged": 6,
            "superseded": 7,
        }.get(readiness, 8)
        canonical_priority = 0 if bool(item.get("canonical_lane")) else 1
        return (
            canonical_priority,
            priority,
            _text(item.get("branch")),
            _text(item.get("lane_id")),
        )

    lanes.sort(key=lane_sort_key)
    _sync_lane_telemetry_from_lanes(lanes)

    alerts = {
        "collisions": [],
        "stalled_lanes": [],
        "expired_lanes": [],
        "stale_heartbeats": [],
        "superseded_lanes": [],
        "orphan_lanes": [],
        "needs_decision": [],
        "missing_receipts": [],
        "scope_violations": [],
        "merge_ready": [],
    }
    for lane in lanes:
        ref = {
            "lane_id": lane["lane_id"],
            "branch": lane["branch"],
            "owner_session_id": lane["owner_session_id"],
            "title": lane["title"],
        }
        if lane["collisions"]:
            alerts["collisions"].append({**ref, "reasons": lane["collisions"]})
        if lane.get("lane_health") == "stalled":
            alerts["stalled_lanes"].append(ref)
        if lane.get("lane_health") == "expired":
            alerts["expired_lanes"].append(ref)
        if lane["stale_heartbeat"]:
            alerts["stale_heartbeats"].append(ref)
        if lane["superseded"]:
            alerts["superseded_lanes"].append(ref)
        if not lane.get("canonical_lane", False):
            alerts["orphan_lanes"].append(ref)
        if lane.get("decision_needed"):
            alerts["needs_decision"].append(ref)
        if lane["missing_receipt"]:
            alerts["missing_receipts"].append(ref)
        if lane.get("scope_violation_detected"):
            alerts["scope_violations"].append(ref)
        if lane["merge_readiness"] == "ready":
            alerts["merge_ready"].append(ref)

    next_actions: list[str] = []
    for lane in lanes:
        action = _text(lane.get("next_action"))
        if not action:
            continue
        summary = f"{lane['title']}: {action}"
        if summary not in next_actions:
            next_actions.append(summary)
        if len(next_actions) >= 5:
            break

    summary = {
        "total_lanes": len(lanes),
        "ready_lanes": sum(1 for lane in lanes if lane["merge_readiness"] == "ready"),
        "blocked_lanes": sum(1 for lane in lanes if lane["merge_readiness"] == "blocked"),
        "review_lanes": sum(1 for lane in lanes if lane["merge_readiness"] == "review"),
        "in_progress_lanes": sum(1 for lane in lanes if lane["merge_readiness"] == "in_progress"),
        "healthy_lanes": sum(1 for lane in lanes if lane.get("lane_health") == "healthy"),
        "stalled_lanes": len(alerts["stalled_lanes"]),
        "expired_lanes": len(alerts["expired_lanes"]),
        "canonical_lanes": sum(1 for lane in lanes if lane.get("canonical_lane")),
        "orphan_lanes": len(alerts["orphan_lanes"]),
        "decision_lanes": len(alerts["needs_decision"]),
        "collision_lanes": len(alerts["collisions"]),
        "stale_heartbeat_lanes": len(alerts["stale_heartbeats"]),
        "superseded_lanes": len(alerts["superseded_lanes"]),
        "missing_receipt_lanes": len(alerts["missing_receipts"]),
        "scope_violation_lanes": len(alerts.get("scope_violations", [])),
        "merge_ready_lanes": len(alerts["merge_ready"]),
        "coordination_counts": coordination.get("counts", {}),
    }
    telemetry = _telemetry_summary()
    return {
        "summary": summary,
        "telemetry": telemetry,
        "next_actions": next_actions,
        "alerts": alerts,
        "lanes": lanes,
    }


def render_runner_registration_text(payload: dict[str, Any]) -> str:
    """Render a concise runner inspection or registration summary."""
    lines = [
        f"runner_id={_text(payload.get('runner_id'))}",
        "availability="
        f"{_text(payload.get('availability'))} auth_mode={_text(payload.get('auth_mode'))}",
        f"runner_type={_text(payload.get('runner_type'))}",
    ]
    profile = _text(payload.get("profile"))
    if profile:
        lines.append(f"profile={profile}")

    owner = payload.get("owner_binding", {})
    if isinstance(owner, dict):
        lines.append(
            "owner="
            f"{_text(owner.get('user_id')) or 'unbound'}"
            f" workspace={_text(owner.get('workspace_id')) or 'none'}"
        )

    capabilities = payload.get("capabilities", {})
    if isinstance(capabilities, dict):
        lines.append(
            "capabilities="
            f"exec={capabilities.get('supports_exec', False)} "
            f"review={capabilities.get('supports_review', False)} "
            f"max_parallel={capabilities.get('max_parallel_lanes', 0)}"
        )

    if payload.get("registered"):
        lines.append(f"registered_at={_text(payload.get('registered_at'))}")
    freshness_status = _text(payload.get("freshness_status"))
    if freshness_status:
        lines.append(
            "freshness="
            f"{freshness_status} heartbeat_at={_text(payload.get('heartbeat_at')) or 'none'} "
            f"stale_after={_text(payload.get('stale_after_seconds')) or 'none'}"
        )

    status_summary = _text(payload.get("status_summary"))
    if status_summary:
        lines.append(f"status_summary={status_summary}")

    next_action = _text(payload.get("next_action"))
    if next_action:
        lines.append(f"next: {next_action}")

    return "\n".join(lines)


def _work_order_status_counts(work_orders: list[dict[str, Any]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in work_orders or []:
        if not isinstance(item, dict):
            continue
        status = _text(item.get("status")) or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_boss_payload(
    *,
    run: dict[str, Any],
    integrator_view: dict[str, Any] | None = None,
    coordination: dict[str, Any] | None = None,
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable boss-facing payload for supervised swarm runs."""
    integrator_view = integrator_view if isinstance(integrator_view, dict) else {}
    coordination = coordination if isinstance(coordination, dict) else {}
    routing = routing if isinstance(routing, dict) else {}
    work_orders = [dict(item) for item in run.get("work_orders", []) if isinstance(item, dict)]
    lanes_from_integrator = [
        dict(item) for item in integrator_view.get("lanes", []) if isinstance(item, dict)
    ]
    lane_by_work_order = {
        _text(item.get("work_order_id")): item
        for item in lanes_from_integrator
        if _text(item.get("work_order_id"))
    }

    lane_summaries: list[dict[str, Any]] = []
    needs_human: list[dict[str, Any]] = []
    for item in work_orders:
        work_order_id = _text(item.get("work_order_id"))
        lane = lane_by_work_order.get(work_order_id, {})
        lane_summary = {
            "work_order_id": work_order_id or None,
            "title": _first_text(item.get("title"), lane.get("title")) or None,
            "status": _first_text(item.get("status"), lane.get("status"), "unknown"),
            "target_agent": _first_text(item.get("target_agent"), lane.get("owner_agent")) or None,
            "branch": _first_text(item.get("branch"), lane.get("branch")) or None,
            "worktree_path": _first_text(item.get("worktree_path"), lane.get("worktree_path"))
            or None,
            "lease_id": _first_text(item.get("lease_id"), lane.get("lease_id")) or None,
            "receipt_id": _extract_receipt_id(item, lane) or None,
            "review_status": _first_text(item.get("review_status")) or None,
            "next_action": _first_text(lane.get("next_action")) or None,
        }
        lane_summaries.append(lane_summary)

        escalation_reasons = [_text(reason) for reason in lane.get("blockers", []) if _text(reason)]
        for key in ("dispatch_error", "resource_error"):
            reason = _text(item.get(key))
            if reason:
                escalation_reasons.append(reason)
        if lane_summary["status"] == "needs_human" or escalation_reasons:
            needs_human.append(
                {
                    "work_order_id": lane_summary["work_order_id"],
                    "title": lane_summary["title"],
                    "reasons": sorted(set(escalation_reasons)),
                }
            )

    return {
        "mode": "boss",
        "run_id": _text(run.get("run_id")),
        "status": _text(run.get("status")),
        "goal": _text(run.get("goal")),
        "target_branch": _text(run.get("target_branch")),
        "work_order_counts": _work_order_status_counts(work_orders),
        "lanes": lane_summaries,
        "integrator_next_actions": [
            _text(item) for item in integrator_view.get("next_actions", []) if _text(item)
        ],
        "needs_human": needs_human,
        "coordination_counts": coordination.get("counts", {}),
        "integrator_summary": integrator_view.get("summary", {}),
        "integrator_telemetry": integrator_view.get("telemetry", {}),
        "routing": routing,
    }


def render_boss_text(payload: dict[str, Any]) -> str:
    """Render a concise boss-facing supervised swarm summary."""
    counts = payload.get("work_order_counts", {})
    counts_text = (
        ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        if isinstance(counts, dict) and counts
        else "none"
    )
    lines = [
        f"run_id={_text(payload.get('run_id'))}",
        f"status={_text(payload.get('status'))} target_branch={_text(payload.get('target_branch'))}",
        f"goal={_text(payload.get('goal'))}",
        f"work_orders=[{counts_text}]",
    ]

    lanes = payload.get("lanes", [])
    if isinstance(lanes, list):
        for lane in lanes[:6]:
            if not isinstance(lane, dict):
                continue
            parts = [
                _text(lane.get("work_order_id")) or "lane",
                _text(lane.get("status")) or "unknown",
            ]
            branch = _text(lane.get("branch"))
            if branch:
                parts.append(f"branch={branch}")
            worktree_path = _text(lane.get("worktree_path"))
            if worktree_path:
                parts.append(f"worktree={worktree_path}")
            receipt_id = _text(lane.get("receipt_id"))
            if receipt_id:
                parts.append(f"receipt={receipt_id}")
            lines.append("lane: " + " ".join(parts))

    routing = payload.get("routing", {})
    if isinstance(routing, dict):
        selected_runner_ids = [
            _text(item) for item in routing.get("selected_runner_ids", []) if _text(item)
        ]
        selected_runners = routing.get("selected_runners", [])
        selected_runner_parts: list[str] = []
        if isinstance(selected_runners, list):
            for item in selected_runners:
                if not isinstance(item, dict):
                    continue
                runner_id = _text(item.get("runner_id"))
                if not runner_id:
                    continue
                freshness = _text(item.get("freshness_status")) or "unknown"
                selected_runner_parts.append(f"{runner_id}({freshness})")
        if selected_runner_parts:
            lines.append("routing: selected_runners=" + ",".join(selected_runner_parts))
        elif selected_runner_ids:
            lines.append("routing: selected_runners=" + ",".join(selected_runner_ids))
        blocked_reason = _text(routing.get("blocked_reason"))
        if blocked_reason:
            lines.append(f"routing_blocked: {blocked_reason}")
        rejected_runner_ids = [
            _text(item) for item in routing.get("rejected_runner_ids", []) if _text(item)
        ]
        if rejected_runner_ids:
            lines.append("routing_rejected: " + ",".join(rejected_runner_ids))
        routing_next = _text(routing.get("next_action"))
        if routing_next:
            lines.append(f"routing_next: {routing_next}")

    next_actions = payload.get("integrator_next_actions", [])
    if isinstance(next_actions, list):
        for item in next_actions[:3]:
            text = _text(item)
            if text:
                lines.append(f"next: {text}")

    needs_human = payload.get("needs_human", [])
    if isinstance(needs_human, list):
        for item in needs_human[:3]:
            if not isinstance(item, dict):
                continue
            reasons = [_text(reason) for reason in item.get("reasons", []) if _text(reason)]
            if not reasons:
                continue
            lines.append(
                "needs_human: "
                f"{_first_text(item.get('title'), item.get('work_order_id'), 'lane')} -> "
                + "; ".join(reasons)
            )

    return "\n".join(lines)


@dataclass
class SwarmReport:
    """Plain-English report of swarm execution for non-developer users."""

    success: bool = False
    summary: str = ""
    what_was_done: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    what_to_do_next: list[str] = field(default_factory=list)

    # Details (for developer review)
    spec: SwarmSpec | None = None
    result: Any = None
    receipts: list[Any] = field(default_factory=list)
    duration_seconds: float = 0.0
    budget_spent_usd: float = 0.0

    def to_plain_text(self) -> str:
        """Render as plain text for terminal output."""
        lines = []
        lines.append("=" * 60)
        lines.append("SWARM REPORT")
        lines.append("=" * 60)
        lines.append("")

        status = "SUCCESS" if self.success else "COMPLETED WITH ISSUES"
        lines.append(f"Status: {status}")
        lines.append("")

        if self.summary:
            lines.append(self.summary)
            lines.append("")

        if self.what_was_done:
            lines.append("What was done:")
            for item in self.what_was_done:
                lines.append(f"  - {item}")
            lines.append("")

        if self.what_failed:
            lines.append("What had issues:")
            for item in self.what_failed:
                lines.append(f"  - {item}")
            lines.append("")

        if self.what_to_do_next:
            lines.append("Suggested next steps:")
            for item in self.what_to_do_next:
                lines.append(f"  - {item}")
            lines.append("")

        lines.append("-" * 60)
        if self.duration_seconds > 0:
            mins = int(self.duration_seconds // 60)
            secs = int(self.duration_seconds % 60)
            lines.append(f"Duration: {mins}m {secs}s")
        if self.budget_spent_usd > 0:
            lines.append(f"Budget spent: ${self.budget_spent_usd:.2f}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render as Markdown for docs/reports."""
        lines = []
        lines.append("# Swarm Report")
        lines.append("")

        status = "Success" if self.success else "Completed with issues"
        lines.append(f"**Status:** {status}")
        lines.append("")

        if self.summary:
            lines.append(f"> {self.summary}")
            lines.append("")

        if self.what_was_done:
            lines.append("## What was done")
            for item in self.what_was_done:
                lines.append(f"- {item}")
            lines.append("")

        if self.what_failed:
            lines.append("## Issues")
            for item in self.what_failed:
                lines.append(f"- {item}")
            lines.append("")

        if self.what_to_do_next:
            lines.append("## Next steps")
            for item in self.what_to_do_next:
                lines.append(f"- {item}")
            lines.append("")

        lines.append("---")
        details = []
        if self.duration_seconds > 0:
            mins = int(self.duration_seconds // 60)
            secs = int(self.duration_seconds % 60)
            details.append(f"Duration: {mins}m {secs}s")
        if self.budget_spent_usd > 0:
            details.append(f"Budget: ${self.budget_spent_usd:.2f}")
        if details:
            lines.append(" | ".join(details))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage/API."""
        return {
            "success": self.success,
            "summary": self.summary,
            "what_was_done": self.what_was_done,
            "what_failed": self.what_failed,
            "what_to_do_next": self.what_to_do_next,
            "duration_seconds": self.duration_seconds,
            "budget_spent_usd": self.budget_spent_usd,
            "spec": self.spec.to_dict() if self.spec else None,
        }


class SwarmReporter:
    """Generates plain-language reports from orchestration results.

    Two modes:
    1. LLM-assisted: Claude translates OrchestrationResult into plain English
    2. Template fallback: structured templates (no LLM needed)
    """

    async def generate(
        self,
        spec: SwarmSpec,
        result: Any,
        duration_seconds: float = 0.0,
    ) -> SwarmReport:
        """Generate a SwarmReport from orchestration results.

        Args:
            spec: The SwarmSpec that drove the execution.
            result: OrchestrationResult from the orchestrator.
            duration_seconds: How long execution took.

        Returns:
            A plain-English SwarmReport.
        """
        # Try LLM-assisted report generation
        report = await self._try_llm_report(spec, result, duration_seconds)
        if report is not None:
            return report

        # Fall back to template-based generation
        return self._template_report(spec, result, duration_seconds)

    async def _try_llm_report(
        self,
        spec: SwarmSpec,
        result: Any,
        duration_seconds: float,
    ) -> SwarmReport | None:
        """Try to generate report using Claude. Returns None on failure."""
        try:
            from aragora.harnesses.claude_code import ClaudeCodeHarness

            harness = ClaudeCodeHarness()
            if not await harness.initialize():
                return None

            result_summary = self._summarize_result(result, spec=spec)
            prompt = (
                "You are a CTO giving a status update to your CEO.\n"
                "Explain what your engineering team accomplished in plain, "
                "simple language. Never use jargon. Be specific about what "
                "was done -- say 'We updated the login page to show your "
                "company logo' not 'Task 3 completed successfully'.\n\n"
                f"Goal: {spec.refined_goal or spec.raw_goal}\n\n"
                f"Results:\n{result_summary}\n\n"
                "Produce a JSON object with:\n"
                '- "summary": 2-3 sentence plain English overview\n'
                '- "what_was_done": Array of bullet points (plain language)\n'
                '- "what_failed": Array of failures (plain language, empty if none)\n'
                '- "what_to_do_next": Array of actionable next steps\n\n'
                "Respond with ONLY the JSON object."
            )

            llm_result = await harness.analyze_repository(
                repo_path=Path("."),
                analysis_type=AnalysisType.GENERAL,
                prompt=prompt,
            )
            raw = llm_result.raw_output if hasattr(llm_result, "raw_output") else str(llm_result)

            import json

            data = json.loads(raw)
            return SwarmReport(
                success=self._is_success(result),
                summary=data.get("summary", ""),
                what_was_done=data.get("what_was_done", []),
                what_failed=data.get("what_failed", []),
                what_to_do_next=data.get("what_to_do_next", []),
                spec=spec,
                result=result,
                duration_seconds=duration_seconds,
                budget_spent_usd=self._extract_budget(result),
            )
        except Exception:
            logger.debug("LLM report generation failed, using template")
            return None

    def _template_report(
        self,
        spec: SwarmSpec,
        result: Any,
        duration_seconds: float,
    ) -> SwarmReport:
        """Template-based report generation (no LLM needed)."""
        total = getattr(result, "total_subtasks", 0)
        completed = getattr(result, "completed_subtasks", 0)
        failed = getattr(result, "failed_subtasks", 0)
        skipped = getattr(result, "skipped_subtasks", 0)
        success = self._is_success(result)

        goal = spec.refined_goal or spec.raw_goal

        if success:
            summary = (
                "Great news -- everything you asked for is done. "
                f"Your team finished all {total} tasks without any issues."
            )
        elif completed > 0:
            summary = (
                f'Your team made good progress on "{goal}". '
                f"They finished {completed} out of {total} tasks, "
                f"but {failed} had issues."
            )
        else:
            summary = (
                f"Your team wasn't able to complete '{goal}'. All {total} tasks ran into issues."
            )

        what_was_done = []
        what_failed = []
        assignments = getattr(result, "assignments", [])
        for assignment in assignments:
            task_title = getattr(assignment, "subtask_title", "Task")
            status = getattr(assignment, "status", "unknown")
            if status == "completed":
                what_was_done.append(task_title)
            elif status in ("failed", "error"):
                error_msg = getattr(assignment, "error", "Unknown error")
                what_failed.append(f"{task_title}: {error_msg}")

        if not what_was_done and completed > 0:
            what_was_done.append(f"{completed} tasks completed successfully")
        if not what_failed and failed > 0:
            what_failed.append(f"{failed} tasks encountered issues")

        what_to_do_next = []
        if failed > 0:
            what_to_do_next.append(
                "Some tasks had issues -- you might want to run the swarm again "
                "or have someone look into what went wrong"
            )
        if skipped > 0:
            what_to_do_next.append(
                f"{skipped} tasks were skipped and may need someone to handle them manually"
            )
        if success:
            what_to_do_next.append(
                "You might want to have someone do a quick review of the changes "
                "to make sure everything looks right"
            )

        # Add confidence level from epistemic scores if available (Phase 5)
        if hasattr(spec, "epistemic_scores") and spec.epistemic_scores:
            avg_score = spec.epistemic_scores.get("average", 0)
            if avg_score >= 0.7:
                confidence = "High"
            elif avg_score >= 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            summary += f" Confidence level: {confidence}."

        return SwarmReport(
            success=success,
            summary=summary,
            what_was_done=what_was_done,
            what_failed=what_failed,
            what_to_do_next=what_to_do_next,
            spec=spec,
            result=result,
            duration_seconds=duration_seconds,
            budget_spent_usd=self._extract_budget(result),
        )

    def _is_success(self, result: Any) -> bool:
        """Determine if the orchestration succeeded."""
        failed = getattr(result, "failed_subtasks", 0)
        total = getattr(result, "total_subtasks", 0)
        if total == 0:
            return False
        return failed == 0

    def _extract_budget(self, result: Any) -> float:
        """Extract total cost from result."""
        return getattr(result, "total_cost_usd", 0.0)

    def _summarize_result(self, result: Any, spec: SwarmSpec | None = None) -> str:
        """Produce a text summary of OrchestrationResult for LLM consumption."""
        lines = []
        total = getattr(result, "total_subtasks", 0)
        completed = getattr(result, "completed_subtasks", 0)
        failed = getattr(result, "failed_subtasks", 0)
        skipped = getattr(result, "skipped_subtasks", 0)
        lines.append(f"Total tasks: {total}")
        lines.append(f"Completed: {completed}")
        lines.append(f"Failed: {failed}")
        lines.append(f"Skipped: {skipped}")

        assignments = getattr(result, "assignments", [])
        for assignment in assignments[:15]:
            title = getattr(assignment, "subtask_title", "Unknown")
            status = getattr(assignment, "status", "unknown")
            error = getattr(assignment, "error", "")
            line = f"  [{status}] {title}"
            if error:
                line += f" - {error}"
            lines.append(line)

        if spec and spec.proactive_suggestions:
            lines.append("\nProactive suggestions made during planning:")
            for suggestion in spec.proactive_suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)
