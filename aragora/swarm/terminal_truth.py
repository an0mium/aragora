"""Shared terminal-truth qualification for swarm autonomous lanes.

Normalizes concrete deliverable detection, truthful terminal outcomes, and
receipt expectations so boss, campaign, and reporter surfaces agree on what a
terminal lane actually produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_IN_FLIGHT_STATUSES: frozenset[str] = frozenset(
    {
        "",
        "active",
        "planned",
        "queued",
        "leased",
        "dispatched",
        "integrating",
        "running",
        "waiting_conflict",
        "waiting_resource",
        "pending",
        "ready",
        "delivered",
        "waiting_for_pr",
        "waiting_for_merge",
        "needs_revision",
    }
)

_FAILURE_STATUSES: frozenset[str] = frozenset(
    {
        "failed",
        "blocked",
        "error",
        "cancelled",
        "timed_out",
        "aborted",
    }
)

_WORKER_OUTCOME_TO_TERMINAL: dict[str, str] = {
    "completed": "deliverable_created",
    "clean_exit_no_effect": "clean_exit_no_deliverable",
    "crash": "crash",
    "crash_with_salvage": "deliverable_created",  # salvaged commits are real deliverables
    "timeout_no_progress": "timeout",
    "timeout_with_salvage": "deliverable_created",  # salvaged commits are real deliverables
    "scope_violation": "blocked",
    "merge_gate_failed": "needs_human",
}

_SUCCESS_OUTCOMES: frozenset[str] = frozenset({"deliverable_created", "pr_adopted"})
_PLACEHOLDER_BLOCKERS: frozenset[str] = frozenset({"none", "null", "n/a", "na"})


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [_text(item) for item in value if _text(item)]


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = _text(item)
        if not text or text.lower() in _PLACEHOLDER_BLOCKERS or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


@dataclass(slots=True)
class TerminalQualification:
    """Normalized terminal truth for a run or lane."""

    terminal_outcome: str
    deliverable: dict[str, Any] | None
    worker_outcome: str | None = None
    blocked_reason: str | None = None
    reasons: list[str] = field(default_factory=list)
    receipt_expected: bool = False
    human_intervention_required: bool = False

    @property
    def deliverable_type(self) -> str | None:
        if not isinstance(self.deliverable, dict):
            return None
        deliverable_type = _text(self.deliverable.get("type"))
        return deliverable_type or None

    @property
    def receipt_outcome(self) -> str:
        if self.terminal_outcome in _SUCCESS_OUTCOMES:
            return "pass"
        if self.deliverable and self.terminal_outcome in {"crash", "timeout"}:
            return "blocked"
        if self.terminal_outcome in {
            "blocked",
            "needs_human",
            "clean_exit_no_deliverable",
        }:
            return "blocked"
        if self.terminal_outcome in {"crash", "timeout"}:
            return "fail"
        return "unknown"


def extract_work_order_deliverable(
    work_order: dict[str, Any],
    *,
    require_terminal_status: bool = True,
) -> dict[str, Any] | None:
    """Extract a concrete deliverable from a work-order-like mapping."""
    if not isinstance(work_order, dict):
        return None

    status = _text(work_order.get("status")).lower()
    if require_terminal_status and status in _IN_FLIGHT_STATUSES:
        return None
    if require_terminal_status and status in _FAILURE_STATUSES:
        return None

    work_order_id = work_order.get("work_order_id")

    pr_url = _text(work_order.get("pr_url"))
    if pr_url:
        return {"type": "pr", "pr_url": pr_url, "work_order_id": work_order_id}

    adopted_pr = _text(work_order.get("adopted_pr"))
    if adopted_pr:
        return {
            "type": "adopted_pr",
            "adopted_pr": adopted_pr,
            "work_order_id": work_order_id,
        }

    branch = _text(work_order.get("branch"))
    commit_shas = _text_list(work_order.get("commit_shas"))
    if branch and commit_shas:
        return {
            "type": "branch",
            "branch": branch,
            "commit_shas": commit_shas,
            "work_order_id": work_order_id,
        }
    return None


def extract_run_deliverable(run_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the first concrete deliverable from a terminal run."""
    for work_order in run_dict.get("work_orders", []):
        deliverable = extract_work_order_deliverable(work_order)
        if deliverable is not None:
            return deliverable
    return None


def extract_run_worker_outcome(run_dict: dict[str, Any]) -> str | None:
    """Return the first structured worker outcome on the run, if any."""
    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        outcome = _text(work_order.get("worker_outcome"))
        if outcome:
            return outcome
    return None


def collect_work_order_blockers(work_order: dict[str, Any]) -> list[str]:
    """Collect blocker/failure strings from a work-order-like mapping."""
    if not isinstance(work_order, dict):
        return []

    blockers: list[str] = []
    blockers.extend(_text_list(work_order.get("blockers")))
    blockers.extend(
        _text_list(
            [
                work_order.get("dispatch_error"),
                work_order.get("failure_reason"),
                work_order.get("blocking_question"),
            ]
        )
    )
    blocker = work_order.get("blocker")
    if isinstance(blocker, dict):
        blockers.extend(_text_list([blocker.get("reason"), blocker.get("question")]))
    return _ordered_unique(blockers)


def collect_run_blockers(run_dict: dict[str, Any]) -> list[str]:
    """Collect truthful blocker/failure strings from run metadata and work orders."""
    blockers: list[str] = []
    metadata = run_dict.get("metadata")
    if isinstance(metadata, dict):
        blockers.extend(_text_list(metadata.get("campaign_blockers")))

    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        blockers.extend(collect_work_order_blockers(work_order))
    return _ordered_unique(blockers)


def qualify_work_order_terminal_state(work_order: dict[str, Any]) -> TerminalQualification:
    """Normalize truthful terminal state for one work-order-like mapping."""
    status = _text(work_order.get("status")).lower()
    deliverable = extract_work_order_deliverable(work_order, require_terminal_status=False)
    worker_outcome = _text(work_order.get("worker_outcome")) or None
    blockers = collect_work_order_blockers(work_order)
    mapped_outcome = _WORKER_OUTCOME_TO_TERMINAL.get(_text(worker_outcome).lower())

    if status in {"completed", "merged"}:
        if mapped_outcome and mapped_outcome not in _SUCCESS_OUTCOMES:
            terminal_outcome = mapped_outcome
        elif deliverable:
            terminal_outcome = (
                "pr_adopted"
                if _text(deliverable.get("type")) == "adopted_pr"
                else "deliverable_created"
            )
        else:
            terminal_outcome = "clean_exit_no_deliverable"
    elif status in {"needs_human", "changes_requested"}:
        terminal_outcome = mapped_outcome or "needs_human"
    elif status in {"blocked", "scope_violation"}:
        terminal_outcome = mapped_outcome or "blocked"
    elif status in {"timed_out", "timeout"}:
        terminal_outcome = mapped_outcome or "timeout"
    elif status in {"failed", "dispatch_failed"}:
        terminal_outcome = mapped_outcome or _keyword_terminal_outcome(work_order)
    elif status in _IN_FLIGHT_STATUSES:
        terminal_outcome = "unknown"
    else:
        terminal_outcome = mapped_outcome or _keyword_terminal_outcome(work_order)

    blocked_reason = (
        blockers[0]
        if blockers
        else _default_blocked_reason(
            terminal_outcome=terminal_outcome,
            deliverable=deliverable,
            worker_outcome=worker_outcome,
        )
    )

    human_intervention_required = terminal_outcome not in _SUCCESS_OUTCOMES
    receipt_expected = (
        terminal_outcome != "unknown" or deliverable is not None or bool(blocked_reason)
    )

    return TerminalQualification(
        terminal_outcome=terminal_outcome,
        deliverable=deliverable,
        worker_outcome=worker_outcome,
        blocked_reason=blocked_reason,
        reasons=blockers,
        receipt_expected=receipt_expected,
        human_intervention_required=human_intervention_required,
    )


def qualify_run_terminal_state(run_dict: dict[str, Any]) -> TerminalQualification:
    """Normalize the truthful terminal outcome for a supervisor-style run dict."""
    status = _text(run_dict.get("status")).lower()
    blockers = collect_run_blockers(run_dict)
    work_order_qualifications = [
        qualify_work_order_terminal_state(work_order)
        for work_order in run_dict.get("work_orders", [])
        if isinstance(work_order, dict)
    ]

    deliverable = next(
        (
            qualification.deliverable
            for qualification in work_order_qualifications
            if qualification.deliverable is not None
        ),
        extract_run_deliverable(run_dict),
    )
    worker_outcome = next(
        (
            qualification.worker_outcome
            for qualification in work_order_qualifications
            if qualification.worker_outcome
        ),
        extract_run_worker_outcome(run_dict),
    )
    qualification_outcomes = {
        qualification.terminal_outcome
        for qualification in work_order_qualifications
        if qualification.terminal_outcome != "unknown"
    }

    if not qualification_outcomes:
        mapped_outcome = _WORKER_OUTCOME_TO_TERMINAL.get(_text(worker_outcome).lower())
        if status in {"completed", "merged"}:
            if deliverable:
                terminal_outcome = (
                    "pr_adopted"
                    if _text(deliverable.get("type")) == "adopted_pr"
                    else "deliverable_created"
                )
            else:
                terminal_outcome = mapped_outcome or "clean_exit_no_deliverable"
        elif status in {"needs_human", "blocked", "changes_requested"}:
            terminal_outcome = mapped_outcome or "needs_human"
        elif status in {"timed_out", "timeout"}:
            terminal_outcome = mapped_outcome or "timeout"
        elif status in {"failed", "dispatch_failed"}:
            terminal_outcome = mapped_outcome or _keyword_terminal_outcome(run_dict)
        elif status in _IN_FLIGHT_STATUSES:
            terminal_outcome = "unknown"
        else:
            terminal_outcome = mapped_outcome or _keyword_terminal_outcome(run_dict)
    elif "blocked" in qualification_outcomes:
        terminal_outcome = "blocked"
    elif "crash" in qualification_outcomes:
        terminal_outcome = "crash"
    elif "timeout" in qualification_outcomes:
        terminal_outcome = "timeout"
    elif (
        status in {"needs_human", "blocked", "changes_requested"}
        or "needs_human" in qualification_outcomes
    ):
        terminal_outcome = "needs_human"
    elif status in {"completed", "merged"}:
        if deliverable:
            terminal_outcome = (
                "pr_adopted"
                if _text(deliverable.get("type")) == "adopted_pr"
                else "deliverable_created"
            )
        else:
            terminal_outcome = "clean_exit_no_deliverable"
    elif "pr_adopted" in qualification_outcomes:
        terminal_outcome = "pr_adopted"
    elif "deliverable_created" in qualification_outcomes:
        terminal_outcome = "deliverable_created"
    elif "clean_exit_no_deliverable" in qualification_outcomes:
        terminal_outcome = "clean_exit_no_deliverable"
    elif status in _IN_FLIGHT_STATUSES:
        terminal_outcome = "unknown"
    else:
        terminal_outcome = _keyword_terminal_outcome(run_dict)

    blocked_reason = (
        blockers[0]
        if blockers
        else _default_blocked_reason(
            terminal_outcome=terminal_outcome,
            deliverable=deliverable,
            worker_outcome=worker_outcome,
        )
    )

    human_intervention_required = terminal_outcome not in _SUCCESS_OUTCOMES
    receipt_expected = (
        terminal_outcome != "unknown" or deliverable is not None or bool(blocked_reason)
    )

    return TerminalQualification(
        terminal_outcome=terminal_outcome,
        deliverable=deliverable,
        worker_outcome=worker_outcome,
        blocked_reason=blocked_reason,
        reasons=blockers,
        receipt_expected=receipt_expected,
        human_intervention_required=human_intervention_required,
    )


def receipt_expected_for_lane(
    *,
    status: str,
    queue_status: str,
    lane_in_flight: bool,
    decision_type: str,
    terminal_outcome: str | None,
    lease_id: str,
    lease_status: str,
    deliverable_present: bool,
    blockers: list[str] | None = None,
) -> bool:
    """Return whether a lane should already have an authoritative receipt."""
    normalized_status = _text(status).lower()
    normalized_queue_status = _text(queue_status).lower()
    normalized_decision = _text(decision_type).lower()
    normalized_outcome = _text(terminal_outcome).lower()
    normalized_lease_id = _text(lease_id)
    normalized_lease_status = _text(lease_status).lower()
    normalized_blockers = {_text(item).lower() for item in blockers or [] if _text(item)}

    if normalized_status == "scope_violation" or "scope_violation" in normalized_blockers:
        return False
    if deliverable_present:
        return True
    if (
        not normalized_decision
        and normalized_queue_status in {"", "queued"}
        and normalized_lease_status not in {"active"}
        and normalized_outcome
        in {
            "clean_exit_no_deliverable",
            "needs_human",
            "blocked",
            "crash",
            "timeout",
        }
    ):
        return False
    if (
        not normalized_decision
        and normalized_queue_status in {"", "queued"}
        and (
            normalized_status == "scope_violation"
            or normalized_blockers.intersection(
                {
                    "stale_lease_reaped",
                    "expired_lease_reaped",
                    "work_order_leasing_failed",
                }
            )
        )
    ):
        return False
    if (
        not deliverable_present
        and normalized_lease_id
        and normalized_outcome in {"needs_human", "blocked"}
        and (
            normalized_status == "scope_violation"
            or "scope_violation" in normalized_blockers
            or "merge_gate_failed" in normalized_blockers
            or any(item.startswith("merge gate blocked:") for item in normalized_blockers)
        )
    ):
        return False
    if normalized_lease_id and normalized_outcome and normalized_outcome != "unknown":
        if (
            normalized_outcome in {"needs_human", "blocked", "crash", "timeout"}
            and normalized_lease_status in {"expired", "released"}
            and normalized_blockers.intersection(
                {"stale_lease_reaped", "expired_lease_reaped", "work_order_leasing_failed"}
            )
        ):
            return False
        return True

    if (
        lane_in_flight
        and not normalized_decision
        and normalized_queue_status in {"", "queued"}
        and not normalized_outcome
    ):
        return False
    if normalized_queue_status in {"validating", "integrating", "needs_human", "merged", "failed"}:
        return True
    if normalized_decision:
        return True
    if normalized_outcome and normalized_outcome != "unknown":
        return bool(normalized_lease_id)
    return normalized_status in {
        "completed",
        "merged",
        "failed",
        "timed_out",
        "needs_human",
        "blocked",
        "changes_requested",
        "salvage",
        "stalled",
    } and bool(normalized_lease_id)


def _default_blocked_reason(
    *,
    terminal_outcome: str,
    deliverable: dict[str, Any] | None,
    worker_outcome: str | None,
) -> str | None:
    deliverable_present = deliverable is not None
    normalized_worker_outcome = _text(worker_outcome).lower()
    if terminal_outcome == "clean_exit_no_deliverable":
        return "Run ended without a concrete deliverable."
    if terminal_outcome == "needs_human" and deliverable_present:
        return "Run produced a deliverable but still requires human judgment before integration."
    if terminal_outcome == "crash" and deliverable_present:
        return "Worker exited non-zero after producing a recoverable deliverable; human review is required."
    if terminal_outcome == "timeout" and deliverable_present:
        return (
            "Worker timed out after producing a recoverable deliverable; human review is required."
        )
    if terminal_outcome == "blocked":
        if normalized_worker_outcome == "scope_violation":
            return "Worker touched files outside the allowed lane scope."
        return "Run blocked before safe integration."
    if terminal_outcome == "needs_human":
        return "Run requires human judgment before continuing."
    if terminal_outcome == "crash":
        return "Worker crashed before producing an acceptable terminal result."
    if terminal_outcome == "timeout":
        return "Worker timed out before producing an acceptable terminal result."
    return None


def _keyword_terminal_outcome(run_dict: dict[str, Any]) -> str:
    details = json.dumps(run_dict, sort_keys=True).lower()
    if "timeout" in details:
        return "timeout"
    if "exit_code" in details or "traceback" in details or "crash" in details:
        return "crash"
    if "needs_human" in details or "blocked" in details:
        return "needs_human"
    return "blocked"
