"""Retry routing and ping-pong helpers for the Boss loop."""

from __future__ import annotations

from typing import Any

from aragora.swarm.boss_feed import GitHubIssue, select_eligible_issue


def normalized_model_rotation(model_rotation: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in model_rotation or []:
        runner_type = str(item).strip().lower()
        if not runner_type or runner_type in seen:
            continue
        seen.add(runner_type)
        normalized.append(runner_type)
    return normalized


def selected_issues_need_retry_routing(
    issues: list[GitHubIssue],
    *,
    pending_handoff_prompts: dict[int, tuple[str, str | None]] | None = None,
    issue_attempt_counts: dict[int | str, int] | None = None,
) -> bool:
    handoffs = pending_handoff_prompts or {}
    attempt_counts = issue_attempt_counts or {}
    for issue in issues:
        issue_number = int(getattr(issue, "number", 0) or 0)
        if issue_number <= 0:
            continue
        if issue_number in handoffs:
            return True
        if int(attempt_counts.get(issue_number, 0) or 0) > 0:
            return True
    return False


def filter_mixed_retry_routing_batch(
    issues: list[GitHubIssue],
    *,
    pending_handoff_prompts: dict[int, tuple[str, str | None]] | None = None,
    issue_attempt_counts: dict[int | str, int] | None = None,
) -> list[GitHubIssue]:
    """Keep retry-routed work isolated from fresh issues in one batch."""
    if len(issues) <= 1:
        return issues

    retry_routed: list[GitHubIssue] = []
    fresh: list[GitHubIssue] = []
    for issue in issues:
        if selected_issues_need_retry_routing(
            [issue],
            pending_handoff_prompts=pending_handoff_prompts,
            issue_attempt_counts=issue_attempt_counts,
        ):
            retry_routed.append(issue)
        else:
            fresh.append(issue)
    if retry_routed and fresh:
        return retry_routed
    return issues


def requested_runner_type_for_freshness(
    selected_issues: list[GitHubIssue],
    *,
    default_target_agent: str | None = None,
    model_rotation: list[str] | tuple[str, ...] | None = None,
    pending_handoff_prompts: dict[int, tuple[str, str | None]] | None = None,
    issue_attempt_counts: dict[int | str, int] | None = None,
) -> str | None:
    if (
        selected_issues_need_retry_routing(
            selected_issues,
            pending_handoff_prompts=pending_handoff_prompts,
            issue_attempt_counts=issue_attempt_counts,
        )
        and len(normalized_model_rotation(model_rotation)) > 1
    ):
        return None
    return default_target_agent


def requested_target_agent_for_issue(
    issue_number: int,
    *,
    default_target_agent: str | None = None,
    model_rotation: list[str] | tuple[str, ...] | None = None,
    issue_attempt_counts: dict[int | str, int] | None = None,
) -> str | None:
    attempt_counts = issue_attempt_counts or {}
    attempt_count = max(0, int(attempt_counts.get(issue_number, 0) or 0))
    default_target = str(default_target_agent or "").strip().lower() or None
    if attempt_count <= 1:
        return default_target

    rotation = normalized_model_rotation(model_rotation)
    if not rotation:
        return default_target
    if default_target and default_target in rotation:
        base_index = rotation.index(default_target)
        return rotation[(base_index + attempt_count - 1) % len(rotation)]
    if default_target:
        return rotation[(attempt_count - 2) % len(rotation)]
    return rotation[(attempt_count - 2) % len(rotation)]


def extract_worker_agent(worker_result: dict[str, Any]) -> str | None:
    for key in ("target_agent", "runner_type"):
        value = str(worker_result.get(key, "")).strip().lower()
        if value:
            return value

    receipt_metadata = worker_result.get("receipt_metadata")
    if isinstance(receipt_metadata, dict):
        for key in ("actual_target_agent", "requested_target_agent", "runner_type"):
            value = str(receipt_metadata.get(key, "")).strip().lower()
            if value:
                return value

    run = worker_result.get("run")
    if not isinstance(run, dict):
        return None
    work_orders = run.get("work_orders", [])
    if not isinstance(work_orders, list):
        return None
    for work_order in work_orders:
        if not isinstance(work_order, dict):
            continue
        value = str(work_order.get("target_agent", "")).strip().lower()
        if value:
            return value
    return None


def pending_handoff_candidates(
    issues: list[GitHubIssue],
    *,
    pending_handoff_prompts: dict[int, tuple[str, str | None]],
    issue_number: int | None = None,
    skip_labels: set[str] | None = None,
    require_labels: set[str] | None = None,
    blocked_scopes: set[str] | None = None,
) -> list[GitHubIssue]:
    if not pending_handoff_prompts:
        return []

    issue_by_number = {int(issue.number): issue for issue in issues}
    candidates: list[GitHubIssue] = []
    stale_issue_numbers: list[int] = []

    for pending_issue_number in list(pending_handoff_prompts):
        issue = issue_by_number.get(int(pending_issue_number))
        if issue is None:
            stale_issue_numbers.append(pending_issue_number)
            continue
        if issue_number is not None and pending_issue_number != issue_number:
            continue
        if (
            select_eligible_issue(
                [issue],
                skip_labels=skip_labels,
                require_labels=require_labels,
                blocked_scopes=blocked_scopes,
            )
            is None
        ):
            stale_issue_numbers.append(pending_issue_number)
            continue
        candidates.append(issue)

    for stale_issue_number in stale_issue_numbers:
        pending_handoff_prompts.pop(stale_issue_number, None)

    return candidates


def extract_worker_transcript(worker_result: dict[str, Any]) -> str:
    """Extract the worker's stdout transcript from the run dict."""
    run = worker_result.get("run")
    if not isinstance(run, dict):
        return ""
    work_orders = run.get("work_orders", [])
    if not isinstance(work_orders, list):
        return ""
    parts: list[str] = []
    for work_order in work_orders:
        if not isinstance(work_order, dict):
            continue
        for key in ("stdout_tail", "transcript", "log_tail"):
            tail = str(work_order.get(key, "")).strip()
            if tail:
                parts.append(tail)
                break
    return "\n---\n".join(parts)


def extract_worker_files_changed(worker_result: dict[str, Any]) -> list[str]:
    """Extract changed file paths from the run dict."""
    run = worker_result.get("run")
    if not isinstance(run, dict):
        return []
    work_orders = run.get("work_orders", [])
    files: list[str] = []
    for work_order in work_orders:
        if not isinstance(work_order, dict):
            continue
        paths = work_order.get("changed_paths", [])
        if isinstance(paths, list):
            files.extend(str(path) for path in paths if str(path).strip())
    return files


def prepare_ping_pong_handoff(
    *,
    issue_number: int,
    issue_title: str,
    worker_result: dict[str, Any],
    reasons: list[str],
    model_rotation: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    transcript = extract_worker_transcript(worker_result)
    if len(transcript.strip()) <= 50:
        return None

    previous_agent = extract_worker_agent(worker_result) or "unknown"
    rotation = list(model_rotation or ["claude", "codex"])
    next_agent = rotation[0] if previous_agent == rotation[-1] else rotation[-1]

    from aragora.swarm.ping_pong import build_handoff_prompt

    return {
        "prompt": build_handoff_prompt(
            goal=f"[Issue #{issue_number}] {issue_title}",
            previous_transcript=transcript,
            previous_agent=previous_agent,
            next_agent=next_agent,
            round_number=1,
            files_changed=extract_worker_files_changed(worker_result),
            remaining_issues=[str(reason) for reason in reasons[:5]],
        ),
        "previous_agent": previous_agent,
        "next_agent": next_agent,
        "transcript_length": len(transcript),
    }
