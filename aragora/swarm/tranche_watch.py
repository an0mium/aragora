from __future__ import annotations

from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import (
    DevCoordinationStore,
    IntegrationDecisionType,
    LeaseStatus,
)
from aragora.swarm.tranche import TrancheArtifactStore, TrancheLaneArtifact
from aragora.swarm.tranche_state import (
    LANE_STATUS_ABORTED,
    LANE_STATUS_COMPLETED,
    LANE_STATUS_DISPATCHED,
    LANE_STATUS_NEEDS_HUMAN,
    LANE_STATUS_PENDING,
    LANE_STATUS_PREPARING,
    LANE_STATUS_RETRYING,
    LANE_STATUS_REVIEW_FAILED,
    LANE_STATUS_REVIEW_PASSED,
    LANE_STATUS_REVIEWING,
    LANE_STATUS_RUNNING,
    LANE_STATUS_WAITING_FOR_MERGE,
    LANE_STATUS_WAITING_FOR_PR,
    TRANCHE_STATUS_COMPLETED,
    TRANCHE_STATUS_INTEGRATING,
    TRANCHE_STATUS_NEEDS_HUMAN,
    TRANCHE_STATUS_PLANNED,
    TRANCHE_STATUS_PREPARING,
    TRANCHE_STATUS_REVIEWING,
    TRANCHE_STATUS_RUNNING,
    LaneRunState,
    TrancheRunState,
    _utcnow,
)


def refresh_tranche_state(
    state: TrancheRunState,
    *,
    artifacts: dict[str, TrancheLaneArtifact] | None = None,
    artifact_store: TrancheArtifactStore | None = None,
    store: DevCoordinationStore | Any | None = None,
    repo_root: Path | None = None,
) -> TrancheRunState:
    refreshed = TrancheRunState.from_dict(state.to_dict())
    resolved_store = store
    if resolved_store is None and repo_root is not None:
        resolved_store = DevCoordinationStore(repo_root=Path(repo_root).resolve())

    artifact_map = _resolve_artifacts(
        refreshed.manifest_id,
        artifacts=artifacts,
        artifact_store=artifact_store,
        repo_root=repo_root,
    )

    for lane_id, lane_state in list(refreshed.lane_states.items()):
        artifact = artifact_map.get(lane_id)
        if artifact is not None:
            _apply_artifact_projection(lane_state, artifact)

        run_dict = _get_supervisor_run(resolved_store, lane_state.run_id)
        if run_dict is not None:
            _apply_run_projection(lane_state, run_dict)

    lease_map = _lease_map(
        resolved_store,
        lease_ids={
            lease_id
            for lane_state in refreshed.lane_states.values()
            if (lease_id := _optional_text(lane_state.lease_id)) is not None
        },
    )
    for lane_state in refreshed.lane_states.values():
        lease = lease_map.get(str(lane_state.lease_id or "").strip())
        if lease is not None:
            _apply_lease_projection(lane_state, lease)

        receipt = _get_completion_receipt(resolved_store, lane_state.receipt_id)
        if receipt is not None:
            _apply_receipt_projection(lane_state, receipt)

        decision = _latest_integration_decision(resolved_store, lane_state.receipt_id)
        if decision is not None:
            _apply_integration_projection(lane_state, decision)

        lane_state.last_updated = _utcnow()

    refreshed.status = _aggregate_tranche_status(
        refreshed.lane_states.values(), current=refreshed.status
    )
    refreshed.updated_at = _utcnow()
    return refreshed


def _resolve_artifacts(
    manifest_id: str,
    *,
    artifacts: dict[str, TrancheLaneArtifact] | None,
    artifact_store: TrancheArtifactStore | None,
    repo_root: Path | None,
) -> dict[str, TrancheLaneArtifact]:
    if artifacts is not None:
        return {
            str(lane_id): artifact
            for lane_id, artifact in artifacts.items()
            if artifact is not None
        }
    store = artifact_store
    if store is None and repo_root is not None:
        store = TrancheArtifactStore(repo_root=Path(repo_root).resolve())
    if store is None:
        return {}
    return {item.lane_id: item for item in store.list(manifest_id)}


def _apply_artifact_projection(lane_state: LaneRunState, artifact: TrancheLaneArtifact) -> None:
    lane_state.status = str(artifact.status or lane_state.status).strip() or lane_state.status
    lane_state.run_id = _prefer_text(lane_state.run_id, artifact.run_id)
    lane_state.worktree_path = _prefer_text(lane_state.worktree_path, artifact.worktree_path)
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    lane_state.receipt_id = _prefer_text(lane_state.receipt_id, metadata.get("receipt_id"))
    lane_state.lease_id = _prefer_text(lane_state.lease_id, metadata.get("lease_id"))
    lane_state.pr_url = _prefer_text(lane_state.pr_url, _artifact_pr_url(artifact))


def _apply_run_projection(lane_state: LaneRunState, run_dict: dict[str, Any]) -> None:
    run_status = str(run_dict.get("status", "")).strip().lower()
    mapped = {
        "planned": LANE_STATUS_PENDING,
        "active": LANE_STATUS_RUNNING,
        "completed": lane_state.status,
        "needs_human": LANE_STATUS_NEEDS_HUMAN,
    }.get(run_status)
    if mapped and lane_state.status in {
        LANE_STATUS_PENDING,
        LANE_STATUS_DISPATCHED,
        LANE_STATUS_RUNNING,
    }:
        lane_state.status = mapped

    for work_order in run_dict.get("work_orders", []):
        if not isinstance(work_order, dict):
            continue
        lane_state.worktree_path = _prefer_text(
            lane_state.worktree_path, work_order.get("worktree_path")
        )
        lane_state.receipt_id = _prefer_text(lane_state.receipt_id, work_order.get("receipt_id"))
        lane_state.lease_id = _prefer_text(lane_state.lease_id, work_order.get("lease_id"))
        if lane_state.run_id:
            break


def _apply_lease_projection(lane_state: LaneRunState, lease: Any) -> None:
    lane_state.worktree_path = _prefer_text(
        lane_state.worktree_path, getattr(lease, "worktree_path", None)
    )
    lease_status = str(getattr(lease, "status", "")).strip()
    if lease_status == LeaseStatus.ACTIVE.value and lane_state.status in {
        LANE_STATUS_PENDING,
        LANE_STATUS_DISPATCHED,
    }:
        lane_state.status = LANE_STATUS_RUNNING
    elif lease_status == LeaseStatus.EXPIRED.value:
        lane_state.status = LANE_STATUS_NEEDS_HUMAN
    elif lease_status == LeaseStatus.COMPLETED.value and lane_state.status in {
        LANE_STATUS_PENDING,
        LANE_STATUS_DISPATCHED,
        LANE_STATUS_RUNNING,
    }:
        lane_state.status = LANE_STATUS_COMPLETED


def _apply_receipt_projection(lane_state: LaneRunState, receipt: Any) -> None:
    lane_state.receipt_id = _prefer_text(
        lane_state.receipt_id, getattr(receipt, "receipt_id", None)
    )
    lane_state.lease_id = _prefer_text(lane_state.lease_id, getattr(receipt, "lease_id", None))
    lane_state.worktree_path = _prefer_text(
        lane_state.worktree_path,
        getattr(receipt, "worktree_path", None),
    )
    lane_state.pr_url = _prefer_text(lane_state.pr_url, getattr(receipt, "pr_url", None))
    if lane_state.status == LANE_STATUS_REVIEW_PASSED:
        lane_state.status = (
            LANE_STATUS_WAITING_FOR_MERGE if lane_state.pr_url else LANE_STATUS_WAITING_FOR_PR
        )


def _apply_integration_projection(lane_state: LaneRunState, decision: Any) -> None:
    value = str(getattr(decision, "decision", "")).strip()
    if value in {IntegrationDecisionType.MERGE.value, IntegrationDecisionType.CHERRY_PICK.value}:
        lane_state.status = (
            LANE_STATUS_WAITING_FOR_MERGE if lane_state.pr_url else LANE_STATUS_WAITING_FOR_PR
        )
    elif value in {
        IntegrationDecisionType.REQUEST_CHANGES.value,
        IntegrationDecisionType.DISCARD.value,
        IntegrationDecisionType.SALVAGE.value,
    }:
        lane_state.status = LANE_STATUS_NEEDS_HUMAN
    elif (
        value == IntegrationDecisionType.PENDING_REVIEW.value
        and lane_state.status == LANE_STATUS_REVIEW_PASSED
    ):
        lane_state.status = (
            LANE_STATUS_WAITING_FOR_MERGE if lane_state.pr_url else LANE_STATUS_WAITING_FOR_PR
        )


def _aggregate_tranche_status(
    lane_states: Any,
    *,
    current: str,
) -> str:
    statuses = {str(item.status).strip() for item in lane_states if getattr(item, "status", None)}
    if not statuses:
        return current or TRANCHE_STATUS_PLANNED
    if statuses <= {LANE_STATUS_COMPLETED}:
        return TRANCHE_STATUS_COMPLETED
    if statuses & {LANE_STATUS_NEEDS_HUMAN, LANE_STATUS_REVIEW_FAILED, LANE_STATUS_ABORTED}:
        return TRANCHE_STATUS_NEEDS_HUMAN
    if statuses & {LANE_STATUS_WAITING_FOR_PR, LANE_STATUS_WAITING_FOR_MERGE}:
        return TRANCHE_STATUS_INTEGRATING
    if statuses & {
        LANE_STATUS_REVIEWING,
        LANE_STATUS_REVIEW_PASSED,
        LANE_STATUS_RETRYING,
    }:
        return TRANCHE_STATUS_REVIEWING
    if statuses & {LANE_STATUS_PREPARING}:
        return TRANCHE_STATUS_PREPARING
    if statuses & {LANE_STATUS_DISPATCHED, LANE_STATUS_RUNNING, LANE_STATUS_COMPLETED}:
        return TRANCHE_STATUS_RUNNING
    return current or TRANCHE_STATUS_PLANNED


def _artifact_pr_url(artifact: TrancheLaneArtifact) -> str | None:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    deliverable = metadata.get("deliverable", {})
    if isinstance(deliverable, dict):
        pr_url = _optional_text(deliverable.get("pr_url"))
        if pr_url:
            return pr_url
    pr_url = _optional_text(metadata.get("pr_url"))
    if pr_url:
        return pr_url
    for url in getattr(artifact, "urls", []):
        text = _optional_text(url)
        if text and "/pull/" in text:
            return text
    return None


def _lease_map(store: Any | None, *, lease_ids: set[str] | None = None) -> dict[str, Any]:
    if store is None or not hasattr(store, "list_leases"):
        return {}
    relevant_lease_ids = {item for item in (lease_ids or set()) if _optional_text(item)}
    if not relevant_lease_ids:
        return {}
    return {
        str(item.lease_id): item
        for item in store.list_leases(limit=None)
        if getattr(item, "lease_id", None) and str(item.lease_id) in relevant_lease_ids
    }


def _get_supervisor_run(store: Any | None, run_id: str | None) -> dict[str, Any] | None:
    if store is None or not run_id or not hasattr(store, "get_supervisor_run"):
        return None
    record = store.get_supervisor_run(run_id)
    return record if isinstance(record, dict) else None


def _get_completion_receipt(store: Any | None, receipt_id: str | None) -> Any | None:
    if store is None or not receipt_id or not hasattr(store, "get_completion_receipt"):
        return None
    return store.get_completion_receipt(receipt_id)


def _latest_integration_decision(store: Any | None, receipt_id: str | None) -> Any | None:
    if store is None or not receipt_id or not hasattr(store, "list_integration_decisions"):
        return None
    decisions = store.list_integration_decisions(receipt_id=receipt_id, limit=1)
    return decisions[0] if decisions else None


def _prefer_text(current: Any, candidate: Any) -> str | None:
    current_text = _optional_text(current)
    if current_text:
        return current_text
    return _optional_text(candidate)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
