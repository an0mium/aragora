"""Helpers for summarizing completed dependency results for supervisor work orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aragora.swarm.terminal_truth import qualify_work_order_terminal_state

_DEPENDENCY_ID_KEYS = ("pipeline_task_id", "work_order_id", "task_key")
_COMPLETED_DEPENDENCY_STATUSES = frozenset({"completed", "merged", "salvage"})
_TERMINAL_FAILURE_DEPENDENCY_STATUSES = frozenset(
    {"discarded", "dispatch_failed", "failed", "needs_human", "scope_violation", "timed_out"}
)
_MAX_CHANGED_PATHS = 10
_MAX_COMMIT_SHAS = 5
_MAX_VERIFICATION_RESULTS = 5
_NON_TERMINAL_NEEDS_HUMAN_REASONS = frozenset(
    {"stale_lease_reaped", "expired_lease_reaped", "needs_human"}
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _text_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for raw in value:
        text = _text(raw)
        if not text or text in values:
            continue
        values.append(text)
        if len(values) >= limit:
            break
    return values


def _normalize_verification_status(result: dict[str, Any]) -> str:
    for key in ("status", "outcome", "conclusion", "result"):
        value = _text(result.get(key))
        if value:
            return value.lower()
    passed = result.get("passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    success = result.get("success")
    if isinstance(success, bool):
        return "passed" if success else "failed"
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int):
        return "passed" if exit_code == 0 else "failed"
    return "unknown"


def _terminal_dependency_failure_reason(work_order: dict[str, Any]) -> str:
    metadata = work_order.get("metadata")
    archived_due_to = _text(metadata.get("archived_due_to")) if isinstance(metadata, dict) else ""
    return (
        _text(work_order.get("failure_reason"))
        or _text(work_order.get("dispatch_error"))
        or (_text(metadata.get("archive_reason")) if isinstance(metadata, dict) else "")
        or archived_due_to
        or _text(work_order.get("worker_outcome"))
        or _text(work_order.get("status")).lower()
    )


def _is_terminal_dependency_failure(work_order: dict[str, Any]) -> bool:
    status = _text(work_order.get("status")).lower()
    if status in {"discarded", "dispatch_failed", "failed", "scope_violation", "timed_out"}:
        return True
    if status != "needs_human":
        return False

    metadata = work_order.get("metadata")
    archived_due_to = _text(metadata.get("archived_due_to")) if isinstance(metadata, dict) else ""
    dependency_reason = _terminal_dependency_failure_reason(work_order).lower()
    if dependency_reason in _NON_TERMINAL_NEEDS_HUMAN_REASONS and not archived_due_to:
        return False
    return True


@dataclass(slots=True)
class DependencyVerificationOutcome:
    """Bounded verification summary for a completed predecessor."""

    command: str
    status: str
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
        }


@dataclass(slots=True)
class DependencyContext:
    """Stable snapshot of one predecessor work order."""

    dependency_id: str
    work_order_id: str
    pipeline_task_id: str
    title: str
    status: str
    terminal_outcome: str
    base_ref: str
    branch: str
    head_sha: str
    changed_paths: list[str] = field(default_factory=list)
    commit_shas: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    verification_outcomes: list[DependencyVerificationOutcome] = field(default_factory=list)
    failure_reason: str = ""
    blocked_reason: str = ""
    deliverable: dict[str, Any] | None = None
    terminal_failure_reason: str = ""

    @property
    def ready_for_dispatch(self) -> bool:
        return self.status in _COMPLETED_DEPENDENCY_STATUSES

    @property
    def terminal_failure(self) -> bool:
        return bool(self.terminal_failure_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_id": self.dependency_id,
            "work_order_id": self.work_order_id,
            "pipeline_task_id": self.pipeline_task_id,
            "title": self.title,
            "status": self.status,
            "terminal_outcome": self.terminal_outcome,
            "base_ref": self.base_ref,
            "branch": self.branch,
            "head_sha": self.head_sha,
            "changed_paths": list(self.changed_paths),
            "commit_shas": list(self.commit_shas),
            "tests_run": list(self.tests_run),
            "verification_outcomes": [item.to_dict() for item in self.verification_outcomes],
            "failure_reason": self.failure_reason,
            "blocked_reason": self.blocked_reason,
            "deliverable": dict(self.deliverable) if isinstance(self.deliverable, dict) else None,
        }


def dependency_ids_for_work_order(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for raw in item.get("dependency_ids", []) if isinstance(item, dict) else []:
        dependency_id = _text(raw)
        if dependency_id and dependency_id not in values:
            values.append(dependency_id)
    return values


def build_dependency_lookup(work_orders: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in work_orders:
        if not isinstance(candidate, dict):
            continue
        for key in _DEPENDENCY_ID_KEYS:
            candidate_id = _text(candidate.get(key))
            if candidate_id:
                lookup[candidate_id] = candidate
    return lookup


def build_dependency_context(
    dependency_id: str,
    work_order: dict[str, Any],
) -> DependencyContext:
    qualification = qualify_work_order_terminal_state(work_order)
    verification_outcomes: list[DependencyVerificationOutcome] = []
    for result in (
        work_order.get("verification_results", []) if isinstance(work_order, dict) else []
    ):
        if not isinstance(result, dict):
            continue
        command = _text(result.get("command"))
        if not command:
            continue
        exit_code = result.get("exit_code")
        verification_outcomes.append(
            DependencyVerificationOutcome(
                command=command,
                status=_normalize_verification_status(result),
                exit_code=exit_code if isinstance(exit_code, int) else None,
            )
        )
        if len(verification_outcomes) >= _MAX_VERIFICATION_RESULTS:
            break

    return DependencyContext(
        dependency_id=dependency_id,
        work_order_id=_text(work_order.get("work_order_id")),
        pipeline_task_id=_text(work_order.get("pipeline_task_id")),
        title=_text(work_order.get("title")),
        status=_text(work_order.get("status")).lower(),
        terminal_outcome=qualification.terminal_outcome,
        base_ref=(
            _text(work_order.get("branch"))
            or _text(work_order.get("head_sha"))
            or _text(work_order.get("initial_head"))
        ),
        branch=_text(work_order.get("branch")),
        head_sha=_text(work_order.get("head_sha")),
        changed_paths=_text_list(work_order.get("changed_paths"), limit=_MAX_CHANGED_PATHS),
        commit_shas=_text_list(work_order.get("commit_shas"), limit=_MAX_COMMIT_SHAS),
        tests_run=_text_list(work_order.get("tests_run"), limit=_MAX_VERIFICATION_RESULTS),
        verification_outcomes=verification_outcomes,
        failure_reason=_text(work_order.get("failure_reason")),
        blocked_reason=_text(qualification.blocked_reason),
        deliverable=dict(qualification.deliverable)
        if isinstance(qualification.deliverable, dict)
        else None,
        terminal_failure_reason=(
            _terminal_dependency_failure_reason(work_order)
            if _is_terminal_dependency_failure(work_order)
            else ""
        ),
    )


def build_dependency_context_payload(
    item: dict[str, Any],
    work_orders: list[dict[str, Any]],
) -> dict[str, Any]:
    dependency_ids = dependency_ids_for_work_order(item)
    if not dependency_ids:
        return {
            "dependency_ids": [],
            "contexts": [],
            "missing_dependency_ids": [],
            "ready_for_dispatch": True,
            "base_reference": None,
            "base_reference_dependency_id": None,
            "terminal_failure": None,
            "prompt_summary": "",
        }

    lookup = build_dependency_lookup(work_orders)
    contexts: list[DependencyContext] = []
    missing_dependency_ids: list[str] = []
    for dependency_id in dependency_ids:
        dependency = lookup.get(dependency_id)
        if not isinstance(dependency, dict):
            missing_dependency_ids.append(dependency_id)
            continue
        contexts.append(build_dependency_context(dependency_id, dependency))

    ready_for_dispatch = (
        not missing_dependency_ids
        and len(contexts) == len(dependency_ids)
        and all(context.ready_for_dispatch for context in contexts)
    )
    base_reference: str | None = None
    base_reference_dependency_id: str | None = None
    for context in reversed(contexts):
        if context.ready_for_dispatch and context.base_ref:
            base_reference = context.base_ref
            base_reference_dependency_id = context.dependency_id
            break

    terminal_failure: dict[str, Any] | None = None
    for context in contexts:
        if not context.terminal_failure:
            continue
        terminal_failure = {
            "dependency_id": context.dependency_id,
            "dependency_status": context.status,
            "dependency_reason": context.terminal_failure_reason,
        }
        break

    return {
        "dependency_ids": list(dependency_ids),
        "contexts": [context.to_dict() for context in contexts],
        "missing_dependency_ids": list(missing_dependency_ids),
        "ready_for_dispatch": ready_for_dispatch,
        "base_reference": base_reference,
        "base_reference_dependency_id": base_reference_dependency_id,
        "terminal_failure": terminal_failure,
        "prompt_summary": render_dependency_context_summary(contexts, missing_dependency_ids),
    }


def render_dependency_context_summary(
    contexts: list[DependencyContext],
    missing_dependency_ids: list[str] | None = None,
) -> str:
    if not contexts and not missing_dependency_ids:
        return ""

    lines = [
        "Upstream dependency context (reference only; do not widen file scope from these paths):"
    ]
    for context in contexts:
        header = f"- {context.dependency_id}: status={context.status}"
        if context.terminal_outcome:
            header += f", outcome={context.terminal_outcome}"
        if context.base_ref:
            header += f", base_ref={context.base_ref}"
        lines.append(header)
        if context.changed_paths:
            lines.append(f"  changed_paths: {', '.join(context.changed_paths)}")
        if context.commit_shas:
            lines.append(f"  commit_shas: {', '.join(context.commit_shas)}")
        if context.verification_outcomes:
            verification_bits = ", ".join(
                f"{item.command} [{item.status}]" for item in context.verification_outcomes
            )
            lines.append(f"  verification: {verification_bits}")
        elif context.tests_run:
            lines.append(f"  tests_run: {', '.join(context.tests_run)}")
        if context.blocked_reason and not context.ready_for_dispatch:
            lines.append(f"  blocked_reason: {context.blocked_reason}")
    if missing_dependency_ids:
        lines.append(f"- missing dependencies: {', '.join(missing_dependency_ids)}")
    return "\n".join(lines)


def compose_dependency_description(base_description: str, prompt_summary: str) -> str:
    if not prompt_summary:
        return base_description.strip()
    parts = [base_description.strip()] if base_description.strip() else []
    parts.append(prompt_summary.strip())
    return "\n\n".join(part for part in parts if part).strip()


__all__ = [
    "DependencyContext",
    "DependencyVerificationOutcome",
    "build_dependency_context",
    "build_dependency_context_payload",
    "build_dependency_lookup",
    "compose_dependency_description",
    "dependency_ids_for_work_order",
    "render_dependency_context_summary",
]
