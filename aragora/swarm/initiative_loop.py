from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aragora.coordination.directives import DirectiveBoard, SessionDirective
from aragora.nomic.dev_coordination import CompletionReceipt, DevCoordinationStore, WorkLease
from aragora.pipeline.decision_plan.core import PlanStatus
from aragora.swarm.initiative_models import InitiativeRecord
from aragora.swarm.initiative_store import InitiativeStore
from aragora.worktree.fleet import resolve_repo_root

STATUS_QUEUED = "queued"
STATUS_ACTIVE = "active"
STATUS_BLOCKED = "blocked"
STATUS_NEEDS_HUMAN = "needs_human"
STATUS_MERGED = "merged"
STATUS_SUPERSEDED = "superseded"

_TERMINAL_SLICE_STATUSES = {
    STATUS_NEEDS_HUMAN,
    STATUS_MERGED,
    STATUS_SUPERSEDED,
}
_RESOLVED_BOUNDARY_STATUSES = {
    STATUS_MERGED,
    STATUS_SUPERSEDED,
    PlanStatus.COMPLETED.value,
}
_ACTIVE_DIRECTIVE_STATUSES = {
    "active",
    STATUS_ACTIVE,
}
_TERMINAL_OUTCOME_MAP = {
    STATUS_NEEDS_HUMAN: STATUS_NEEDS_HUMAN,
    STATUS_MERGED: STATUS_MERGED,
    STATUS_SUPERSEDED: STATUS_SUPERSEDED,
}
_TASK_PREFIX = "initiative_task_id:"
_INITIATIVE_PREFIX = "initiative_id:"
_SLICE_PREFIX = "initiative_slice_id:"


def _text(value: object) -> str:
    return str(value or "").strip()


def _status(value: object, *, default: str = "") -> str:
    return _text(value) or default


def _terminal_status_from_metadata(metadata: dict[str, object]) -> str | None:
    for key in ("initiative_terminal_status", "terminal_status"):
        status = _status(metadata.get(key))
        if status in _TERMINAL_OUTCOME_MAP:
            return status
    return None


def _task_id_for_slice(initiative_id: str, slice_id: str) -> str:
    return f"initiative:{initiative_id}:slice:{slice_id}"


def _constraint_value(constraints: list[str], prefix: str) -> str | None:
    for item in constraints:
        text = _text(item)
        if text.startswith(prefix):
            return text[len(prefix) :].strip() or None
    return None


def _directive_task_id(directive: SessionDirective) -> str | None:
    return _constraint_value(list(directive.constraints), _TASK_PREFIX)


def _directive_is_active(directive: SessionDirective) -> bool:
    return _status(directive.status, default=STATUS_ACTIVE) in _ACTIVE_DIRECTIVE_STATUSES


def _ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _latest_receipt_by_task(receipts: list[CompletionReceipt]) -> dict[str, CompletionReceipt]:
    latest: dict[str, CompletionReceipt] = {}
    for receipt in sorted(receipts, key=lambda item: (_text(item.created_at), item.receipt_id)):
        latest[receipt.task_id] = receipt
    return latest


@dataclass(slots=True)
class InitiativeExecutionSnapshot:
    initiative: InitiativeRecord
    ready_slice_ids: list[str]
    boundary_blockers: list[str]
    slice_statuses: dict[str, str]
    checkpoint_statuses: dict[str, str]
    milestone_statuses: dict[str, str]
    owner_targets: dict[str, str]
    dispatched_slice_ids: list[str] = field(default_factory=list)

    @property
    def initiative_id(self) -> str:
        return self.initiative.initiative_id

    @property
    def status(self) -> str:
        return self.initiative.status

    def to_dict(self) -> dict[str, object]:
        grouped: dict[str, list[str]] = {
            STATUS_QUEUED: [],
            STATUS_ACTIVE: [],
            STATUS_BLOCKED: [],
            STATUS_NEEDS_HUMAN: [],
            STATUS_MERGED: [],
            STATUS_SUPERSEDED: [],
        }
        for slice_id, status in self.slice_statuses.items():
            grouped.setdefault(status, []).append(slice_id)
        return {
            "initiative_id": self.initiative_id,
            "status": self.status,
            "ready_slice_ids": list(self.ready_slice_ids),
            "boundary_blockers": list(self.boundary_blockers),
            "slice_statuses": dict(self.slice_statuses),
            "checkpoint_statuses": dict(self.checkpoint_statuses),
            "milestone_statuses": dict(self.milestone_statuses),
            "owner_targets": dict(self.owner_targets),
            "dispatched_slice_ids": list(self.dispatched_slice_ids),
            "grouped_slice_ids": grouped,
        }


class InitiativeExecutor:
    """Execute dependency-ready initiative slices through shared coordination state."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        store: InitiativeStore | None = None,
        coordination_store: DevCoordinationStore | None = None,
        directive_board: DirectiveBoard | None = None,
    ) -> None:
        resolved_repo = Path(repo_root or Path.cwd()).resolve()
        self.repo_root = resolved_repo
        self.coord_repo_root = resolve_repo_root(resolved_repo)
        self.store = store or InitiativeStore(repo_root=resolved_repo)
        self.coordination_store = coordination_store or DevCoordinationStore(
            repo_root=resolved_repo
        )
        self.directive_board = directive_board or DirectiveBoard(repo_path=self.coord_repo_root)

    def refresh(self, initiative_id: str) -> InitiativeExecutionSnapshot:
        initiative = self.store.get(initiative_id)
        if initiative is None:
            raise KeyError(f"Unknown initiative: {initiative_id}")

        directives = self.directive_board.list()
        directive_by_task = {
            task_id: directive
            for directive in directives
            if (task_id := _directive_task_id(directive)) is not None
        }
        active_leases = {
            lease.task_id: lease for lease in self.coordination_store.list_active_leases()
        }
        latest_receipts = _latest_receipt_by_task(
            self.coordination_store.list_completion_receipts(limit=1000)
        )

        direct_statuses: dict[str, str | None] = {}
        slice_context: dict[
            str,
            tuple[
                dict[str, object],
                SessionDirective | None,
                WorkLease | None,
                CompletionReceipt | None,
            ],
        ] = {}
        slice_statuses: dict[str, str] = {}
        owner_targets: dict[str, str] = {}
        for slice_record in initiative.slices:
            task_id = _task_id_for_slice(initiative.initiative_id, slice_record.slice_id)
            metadata = dict(slice_record.metadata)
            metadata["coordination_task_id"] = task_id
            directive = directive_by_task.get(task_id)
            lease = active_leases.get(task_id)
            receipt = latest_receipts.get(task_id)
            slice_context[slice_record.slice_id] = (metadata, directive, lease, receipt)
            direct_statuses[slice_record.slice_id] = self._slice_activity_status(
                current_status=slice_record.status,
                metadata=metadata,
                directive=directive,
                lease=lease,
                receipt=receipt,
            )

        for slice_record in initiative.slices:
            task_id = _task_id_for_slice(initiative.initiative_id, slice_record.slice_id)
            metadata, directive, lease, receipt = slice_context[slice_record.slice_id]
            blocked_by = [
                dep
                for dep in slice_record.dependencies
                if direct_statuses.get(dep) not in _TERMINAL_SLICE_STATUSES
            ]
            status = self._slice_status(
                direct_status=direct_statuses[slice_record.slice_id],
                blocked_by=blocked_by,
            )
            metadata["blocked_by"] = list(blocked_by)
            metadata["coordination_task_id"] = task_id

            if directive is not None:
                metadata["owner_target"] = directive.target
                owner_targets[slice_record.slice_id] = directive.target
            else:
                owner_target = _text(metadata.get("owner_target"))
                if owner_target:
                    owner_targets[slice_record.slice_id] = owner_target

            if lease is not None:
                metadata["lease_id"] = lease.lease_id
                metadata["lease_owner_session_id"] = lease.owner_session_id

            if receipt is not None:
                metadata["receipt_id"] = receipt.receipt_id
                metadata["pr_url"] = receipt.pr_url or metadata.get("pr_url") or ""
                metadata["pr_number"] = receipt.pr_number
                metadata["receipt_outcome"] = receipt.outcome

            slice_record.status = status
            slice_record.metadata = metadata
            slice_statuses[slice_record.slice_id] = status

        checkpoint_statuses = self._refresh_checkpoints(initiative, slice_statuses)
        milestone_statuses = self._refresh_milestones(
            initiative, slice_statuses, checkpoint_statuses
        )
        boundary_blockers = self._boundary_blockers(
            initiative, checkpoint_statuses, milestone_statuses
        )
        ready_slice_ids = [
            item.slice_id
            for item in initiative.slices
            if item.status == STATUS_QUEUED and not boundary_blockers
        ]

        initiative.status = self._initiative_status(
            initiative=initiative,
            ready_slice_ids=ready_slice_ids,
            boundary_blockers=boundary_blockers,
            slice_statuses=slice_statuses,
        )
        self.store.save(initiative)
        return InitiativeExecutionSnapshot(
            initiative=initiative,
            ready_slice_ids=ready_slice_ids,
            boundary_blockers=boundary_blockers,
            slice_statuses=slice_statuses,
            checkpoint_statuses=checkpoint_statuses,
            milestone_statuses=milestone_statuses,
            owner_targets=owner_targets,
        )

    def dispatch_ready_slices(
        self,
        initiative_id: str,
        *,
        owner_targets: list[str],
        assigned_by: str = "initiative-loop",
    ) -> InitiativeExecutionSnapshot:
        snapshot = self.refresh(initiative_id)
        if not snapshot.ready_slice_ids or snapshot.boundary_blockers:
            return snapshot

        initiative = snapshot.initiative
        directives = self.directive_board.list()
        busy_targets = {
            directive.target for directive in directives if _directive_is_active(directive)
        }
        active_directive_task_ids = {
            task_id
            for directive in directives
            if _directive_is_active(directive)
            and (task_id := _directive_task_id(directive)) is not None
        }
        available_targets = [
            target for target in _ordered_unique(owner_targets) if target not in busy_targets
        ]

        dispatched: list[str] = []
        target_iter = iter(available_targets)
        for slice_record in initiative.slices:
            if slice_record.slice_id not in snapshot.ready_slice_ids:
                continue
            task_id = _task_id_for_slice(initiative.initiative_id, slice_record.slice_id)
            if task_id in active_directive_task_ids:
                continue
            try:
                target = next(target_iter)
            except StopIteration:
                break
            constraints = [
                f"{_INITIATIVE_PREFIX}{initiative.initiative_id}",
                f"{_SLICE_PREFIX}{slice_record.slice_id}",
                f"{_TASK_PREFIX}{task_id}",
            ]
            constraints.extend(f"validation:{item}" for item in slice_record.validations)
            self.directive_board.assign(
                target,
                f"Initiative slice {slice_record.slice_id}: {slice_record.title}",
                scope=list(slice_record.file_scope),
                constraints=constraints,
                assigned_by=assigned_by,
                status=STATUS_ACTIVE,
            )
            slice_record.status = STATUS_ACTIVE
            slice_record.metadata = {
                **dict(slice_record.metadata),
                "owner_target": target,
                "coordination_task_id": task_id,
            }
            dispatched.append(slice_record.slice_id)

        if dispatched:
            initiative.status = STATUS_ACTIVE
            self.store.save(initiative)
        refreshed = self.refresh(initiative_id)
        refreshed.dispatched_slice_ids.extend(dispatched)
        return refreshed

    def _slice_status(
        self,
        *,
        direct_status: str | None,
        blocked_by: list[str],
    ) -> str:
        if direct_status is not None:
            return direct_status
        if blocked_by:
            return STATUS_BLOCKED
        return STATUS_QUEUED

    def _slice_activity_status(
        self,
        *,
        current_status: str,
        metadata: dict[str, object],
        directive: SessionDirective | None,
        lease: WorkLease | None,
        receipt: CompletionReceipt | None,
    ) -> str | None:
        explicit_terminal = self._explicit_terminal_status(
            current_status=current_status, metadata=metadata
        )
        if explicit_terminal is not None:
            return explicit_terminal
        if receipt is not None:
            return self._receipt_terminal_status(receipt)
        if lease is not None:
            return STATUS_ACTIVE
        if directive is not None and _directive_is_active(directive):
            return STATUS_ACTIVE
        return None

    def _explicit_terminal_status(
        self, *, current_status: str, metadata: dict[str, object]
    ) -> str | None:
        normalized = _status(current_status)
        if normalized in _TERMINAL_SLICE_STATUSES:
            return normalized
        return _terminal_status_from_metadata(metadata)

    def _receipt_terminal_status(self, receipt: CompletionReceipt) -> str:
        metadata_status = _terminal_status_from_metadata(dict(receipt.metadata))
        if metadata_status is not None:
            return metadata_status
        outcome = _status(receipt.outcome).lower()
        if "supersed" in outcome:
            return STATUS_SUPERSEDED
        if outcome == STATUS_MERGED:
            return STATUS_MERGED
        return STATUS_NEEDS_HUMAN

    def _refresh_checkpoints(
        self, initiative: InitiativeRecord, slice_statuses: dict[str, str]
    ) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for checkpoint in initiative.checkpoints:
            current_status = _status(checkpoint.status)
            if (
                current_status in _RESOLVED_BOUNDARY_STATUSES
                or current_status == STATUS_NEEDS_HUMAN
            ):
                statuses[checkpoint.checkpoint_id] = current_status
                continue
            dependencies_satisfied = all(
                slice_statuses.get(dep) in _TERMINAL_SLICE_STATUSES
                for dep in checkpoint.dependencies
            )
            status = STATUS_NEEDS_HUMAN if dependencies_satisfied else STATUS_BLOCKED
            checkpoint.status = status
            checkpoint.metadata = {
                **dict(checkpoint.metadata),
                "blocked_by": [
                    dep
                    for dep in checkpoint.dependencies
                    if slice_statuses.get(dep) not in _TERMINAL_SLICE_STATUSES
                ],
            }
            statuses[checkpoint.checkpoint_id] = status
        return statuses

    def _refresh_milestones(
        self,
        initiative: InitiativeRecord,
        slice_statuses: dict[str, str],
        checkpoint_statuses: dict[str, str],
    ) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for milestone in initiative.milestones:
            current_status = _status(milestone.status)
            if (
                current_status in _RESOLVED_BOUNDARY_STATUSES
                or current_status == STATUS_NEEDS_HUMAN
            ):
                statuses[milestone.milestone_id] = current_status
                continue
            slice_ready = all(
                slice_statuses.get(slice_id) in _TERMINAL_SLICE_STATUSES
                for slice_id in milestone.slice_ids
            )
            checkpoint_ready = all(
                checkpoint_statuses.get(checkpoint_id) in _RESOLVED_BOUNDARY_STATUSES
                for checkpoint_id in milestone.checkpoint_ids
            )
            if (
                (milestone.slice_ids or milestone.checkpoint_ids)
                and slice_ready
                and checkpoint_ready
            ):
                status = STATUS_NEEDS_HUMAN
            else:
                status = STATUS_BLOCKED
            milestone.status = status
            milestone.metadata = {
                **dict(milestone.metadata),
                "blocked_by": [
                    item
                    for item in milestone.slice_ids
                    if slice_statuses.get(item) not in _TERMINAL_SLICE_STATUSES
                ]
                + [
                    item
                    for item in milestone.checkpoint_ids
                    if checkpoint_statuses.get(item) not in _RESOLVED_BOUNDARY_STATUSES
                ],
            }
            statuses[milestone.milestone_id] = status
        return statuses

    def _boundary_blockers(
        self,
        initiative: InitiativeRecord,
        checkpoint_statuses: dict[str, str],
        milestone_statuses: dict[str, str],
    ) -> list[str]:
        blockers: list[str] = []
        blockers.extend(
            checkpoint.checkpoint_id
            for checkpoint in initiative.checkpoints
            if checkpoint_statuses.get(checkpoint.checkpoint_id) == STATUS_NEEDS_HUMAN
        )
        blockers.extend(
            milestone.milestone_id
            for milestone in initiative.milestones
            if milestone_statuses.get(milestone.milestone_id) == STATUS_NEEDS_HUMAN
        )
        return blockers

    def _initiative_status(
        self,
        *,
        initiative: InitiativeRecord,
        ready_slice_ids: list[str],
        boundary_blockers: list[str],
        slice_statuses: dict[str, str],
    ) -> str:
        explicit_status = _status(initiative.status)
        if explicit_status in {STATUS_MERGED, STATUS_SUPERSEDED}:
            return explicit_status
        status_values = list(slice_statuses.values())
        if boundary_blockers:
            return STATUS_NEEDS_HUMAN
        if STATUS_ACTIVE in status_values:
            return STATUS_ACTIVE
        if ready_slice_ids:
            return STATUS_QUEUED
        if status_values and all(status == STATUS_SUPERSEDED for status in status_values):
            return STATUS_SUPERSEDED
        if status_values and all(
            status in {STATUS_MERGED, STATUS_SUPERSEDED} for status in status_values
        ):
            return STATUS_MERGED
        if STATUS_NEEDS_HUMAN in status_values:
            return STATUS_NEEDS_HUMAN
        if STATUS_BLOCKED in status_values:
            return STATUS_BLOCKED
        return STATUS_QUEUED


__all__ = [
    "InitiativeExecutionSnapshot",
    "InitiativeExecutor",
    "STATUS_ACTIVE",
    "STATUS_BLOCKED",
    "STATUS_MERGED",
    "STATUS_NEEDS_HUMAN",
    "STATUS_QUEUED",
    "STATUS_SUPERSEDED",
]
