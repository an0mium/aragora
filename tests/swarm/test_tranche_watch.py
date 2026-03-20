from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aragora.nomic.dev_coordination import (
    CompletionReceipt,
    IntegrationDecision,
    IntegrationDecisionType,
    WorkLease,
)
from aragora.swarm.tranche import TrancheLaneArtifact
from aragora.swarm.tranche_state import LaneRunState, TrancheRunState
from aragora.swarm.tranche_watch import (
    DriverAlreadyClaimedError,
    claim_driver,
    heartbeat_driver,
    refresh_tranche_state,
    release_driver,
)


def _utc(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(UTC).isoformat()


def _make_state(*, lane_statuses: dict[str, str]) -> TrancheRunState:
    state = TrancheRunState(
        manifest_id="m1",
        status="running",
        autonomy_mode="adaptive",
    )
    for lane_id, status in lane_statuses.items():
        state.lane_states[lane_id] = LaneRunState(lane_id=lane_id, status=status)
    return state


def _make_artifact(
    *,
    lane_id: str,
    status: str,
    run_id: str | None = None,
    timestamp: str = "2026-03-19T12:00:00+00:00",
    metadata: dict | None = None,
) -> TrancheLaneArtifact:
    return TrancheLaneArtifact(
        lane_id=lane_id,
        source_ref=f"lane:{lane_id}",
        status=status,
        run_id=run_id,
        timestamp=timestamp,
        metadata=metadata or {},
    )


class _FakeArtifactStore:
    def __init__(self, artifacts: dict[str, TrancheLaneArtifact]) -> None:
        self._artifacts = dict(artifacts)

    def list(self, manifest_id: str) -> list[TrancheLaneArtifact]:
        assert manifest_id == "m1"
        return list(self._artifacts.values())


class _FakeCoordinationStore:
    def __init__(
        self,
        *,
        runs: dict[str, dict] | None = None,
        leases: list[WorkLease] | None = None,
        receipts: dict[str, CompletionReceipt] | None = None,
        decisions: dict[str, list[IntegrationDecision]] | None = None,
    ) -> None:
        self._runs = runs or {}
        self._leases = leases or []
        self._receipts = receipts or {}
        self._decisions = decisions or {}
        self.list_leases_calls = 0

    def get_supervisor_run(self, run_id: str) -> dict | None:
        return self._runs.get(run_id)

    def list_leases(
        self, *, statuses: list[str] | None = None, limit: int | None = 500
    ) -> list[WorkLease]:
        self.list_leases_calls += 1
        items = list(self._leases)
        if statuses is None:
            return items
        allowed = set(statuses)
        return [item for item in items if item.status in allowed]

    def get_completion_receipt(self, receipt_id: str) -> CompletionReceipt | None:
        return self._receipts.get(receipt_id)

    def list_integration_decisions(
        self,
        *,
        only_pending: bool = False,
        receipt_id: str | None = None,
        limit: int | None = None,
    ) -> list[IntegrationDecision]:
        items = list(self._decisions.get(receipt_id or "", []))
        if only_pending:
            items = [
                item
                for item in items
                if item.decision == IntegrationDecisionType.PENDING_REVIEW.value
            ]
        if isinstance(limit, int) and limit > 0:
            return items[:limit]
        return items


def test_refresh_updates_lane_status_from_artifact_store():
    state = _make_state(lane_statuses={"a": "dispatched"})
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")

    refreshed = refresh_tranche_state(state, artifacts={"a": artifact})

    assert refreshed.lane_states["a"].status == "completed"
    assert refreshed.lane_states["a"].run_id == "run-1"


def test_refresh_skips_lease_lookup_without_known_lease_ids() -> None:
    state = _make_state(lane_statuses={"a": "completed"})
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    store = _FakeCoordinationStore(
        leases=[
            WorkLease(
                lease_id="lease-1",
                task_id="task-1",
                title="task-1",
                owner_agent="codex",
                owner_session_id="sess-1",
                branch="feat-branch",
                worktree_path="/tmp/worktree",
                status="active",
            )
        ]
    )

    refreshed = refresh_tranche_state(state, artifacts={"a": artifact}, store=store)

    assert refreshed.lane_states["a"].lease_id is None
    assert store.list_leases_calls == 0


def test_refresh_uses_receipt_and_integration_state_for_waiting_for_merge():
    state = _make_state(lane_statuses={"a": "review_passed"})
    artifact = _make_artifact(
        lane_id="a",
        status="review_passed",
        run_id="run-1",
        metadata={"receipt_id": "receipt-1", "lease_id": "lease-1"},
    )
    receipt = CompletionReceipt(
        receipt_id="receipt-1",
        lease_id="lease-1",
        task_id="task-1",
        owner_agent="codex",
        owner_session_id="sess-1",
        branch="feat-branch",
        worktree_path="/tmp/worktree",
        pr_url="https://github.com/org/repo/pull/42",
    )
    decision = IntegrationDecision(
        decision_id="decision-1",
        lease_id="lease-1",
        receipt_id="receipt-1",
        decision=IntegrationDecisionType.MERGE.value,
        target_branch="main",
        rationale="approved",
    )
    store = _FakeCoordinationStore(
        receipts={"receipt-1": receipt},
        decisions={"receipt-1": [decision]},
    )

    refreshed = refresh_tranche_state(state, artifacts={"a": artifact}, store=store)

    lane = refreshed.lane_states["a"]
    assert lane.receipt_id == "receipt-1"
    assert lane.lease_id == "lease-1"
    assert lane.pr_url == "https://github.com/org/repo/pull/42"
    assert lane.status == "waiting_for_merge"


def test_refresh_marks_tranche_completed_when_all_lanes_completed():
    state = _make_state(lane_statuses={"a": "running", "b": "review_passed"})
    artifacts = {
        "a": _make_artifact(lane_id="a", status="completed", run_id="run-a"),
        "b": _make_artifact(
            lane_id="b",
            status="review_passed",
            run_id="run-b",
            metadata={"receipt_id": "receipt-b", "lease_id": "lease-b"},
        ),
    }
    receipt = CompletionReceipt(
        receipt_id="receipt-b",
        lease_id="lease-b",
        task_id="task-b",
        owner_agent="codex",
        owner_session_id="sess-b",
        branch="feat-b",
        worktree_path="/tmp/worktree-b",
        pr_url="https://github.com/org/repo/pull/99",
    )
    decision = IntegrationDecision(
        decision_id="decision-b",
        lease_id="lease-b",
        receipt_id="receipt-b",
        decision=IntegrationDecisionType.CHERRY_PICK.value,
        target_branch="main",
        rationale="approved",
    )
    store = _FakeCoordinationStore(
        receipts={"receipt-b": receipt},
        decisions={"receipt-b": [decision]},
    )

    refreshed = refresh_tranche_state(state, artifacts=artifacts, store=store)

    assert refreshed.lane_states["a"].status == "completed"
    assert refreshed.lane_states["b"].status == "waiting_for_merge"
    assert refreshed.status == "integrating"


def test_refresh_marks_needs_human_for_request_changes_decision():
    state = _make_state(lane_statuses={"a": "review_passed"})
    artifact = _make_artifact(
        lane_id="a",
        status="review_passed",
        metadata={"receipt_id": "receipt-1", "lease_id": "lease-1"},
    )
    receipt = CompletionReceipt(
        receipt_id="receipt-1",
        lease_id="lease-1",
        task_id="task-1",
        owner_agent="codex",
        owner_session_id="sess-1",
        branch="feat-branch",
        worktree_path="/tmp/worktree",
    )
    decision = IntegrationDecision(
        decision_id="decision-1",
        lease_id="lease-1",
        receipt_id="receipt-1",
        decision=IntegrationDecisionType.REQUEST_CHANGES.value,
        target_branch="main",
        rationale="follow-up required",
    )
    store = _FakeCoordinationStore(
        receipts={"receipt-1": receipt},
        decisions={"receipt-1": [decision]},
    )

    refreshed = refresh_tranche_state(state, artifacts={"a": artifact}, store=store)

    assert refreshed.lane_states["a"].status == "needs_human"
    assert refreshed.status == "needs_human"


def test_driver_claim_succeeds_when_no_active_driver():
    state = TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive")

    updated = claim_driver(state, session_id="sess-1")

    assert updated.driver_session == "sess-1"
    assert updated.driver_heartbeat is not None
    assert updated.session_history[-1]["session_id"] == "sess-1"


def test_driver_claim_fails_when_active_driver_with_heartbeat():
    state = TrancheRunState(
        manifest_id="m1",
        status="running",
        autonomy_mode="adaptive",
        driver_session="sess-1",
        driver_heartbeat=datetime.now(UTC),
    )

    with pytest.raises(DriverAlreadyClaimedError):
        claim_driver(state, session_id="sess-2")


def test_stale_driver_can_be_taken_over():
    state = TrancheRunState(
        manifest_id="m1",
        status="running",
        autonomy_mode="adaptive",
        driver_session="sess-1",
        driver_heartbeat=datetime.now(UTC) - timedelta(minutes=10),
    )

    updated = claim_driver(state, session_id="sess-2", takeover_timeout_seconds=300)

    assert updated.driver_session == "sess-2"
    assert updated.session_history[-1]["session_id"] == "sess-2"


def test_release_driver_clears_driver_session():
    state = claim_driver(
        TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive"),
        session_id="sess-1",
    )

    updated = release_driver(state, session_id="sess-1")

    assert updated.driver_session is None
    assert updated.driver_heartbeat is None
    assert updated.session_history[-1]["detached_at"]


def test_heartbeat_driver_updates_timestamp():
    state = claim_driver(
        TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive"),
        session_id="sess-1",
    )
    before = state.driver_heartbeat

    updated = heartbeat_driver(state, session_id="sess-1")

    assert updated.driver_session == "sess-1"
    assert updated.driver_heartbeat is not None
    assert before is not None
    assert updated.driver_heartbeat >= before


@pytest.mark.asyncio
async def test_watch_tick_triggers_review_when_lane_completes():
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "completed"})
    artifact_store = _FakeArtifactStore({"a": _make_artifact(lane_id="a", status="completed")})
    mock_rev = AsyncMock(return_value={"status": "passed", "tier": 1})

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="adaptive",
        review_fn=mock_rev,
        artifact_store=artifact_store,
    )

    assert new_state.lane_states["a"].status in ("reviewing", "review_passed")
    mock_rev.assert_awaited_once()


@pytest.mark.asyncio
async def test_watch_tick_dispatches_pending_lane_when_idle() -> None:
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "pending"})
    mock_run = AsyncMock(
        return_value={
            "results": [
                {
                    "lane_id": "a",
                    "status": "running",
                    "run_id": "run-1",
                    "worktree_path": "/tmp/worktree-a",
                }
            ]
        }
    )

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="adaptive",
        run_fn=mock_run,
    )

    mock_run.assert_awaited_once()
    lane = new_state.lane_states["a"]
    assert lane.status == "running"
    assert lane.run_id == "run-1"
    assert lane.worktree_path == "/tmp/worktree-a"


@pytest.mark.asyncio
async def test_watch_tick_skips_dispatch_when_lane_is_already_active() -> None:
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "running", "b": "pending"})
    mock_run = AsyncMock()

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="adaptive",
        run_fn=mock_run,
    )

    mock_run.assert_not_awaited()
    assert new_state.lane_states["b"].status == "pending"


@pytest.mark.asyncio
async def test_watch_tick_skips_rereview_for_terminal_completed_lane_and_dispatches_next() -> None:
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "completed", "b": "pending"})
    state.lane_states["a"].receipt_id = "receipt-a"
    state.lane_states["a"].pr_url = "https://github.com/org/repo/pull/42"
    store = _FakeCoordinationStore(
        decisions={
            "receipt-a": [
                IntegrationDecision(
                    decision_id="decision-a",
                    lease_id="lease-a",
                    receipt_id="receipt-a",
                    decision=IntegrationDecisionType.MERGE.value,
                    target_branch="main",
                    rationale="merged",
                )
            ]
        }
    )
    mock_review = AsyncMock()
    mock_run = AsyncMock(
        return_value={"results": [{"lane_id": "b", "status": "running", "run_id": "run-b"}]}
    )

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="adaptive",
        run_fn=mock_run,
        review_fn=mock_review,
        store=store,
    )

    mock_run.assert_awaited_once()
    mock_review.assert_not_awaited()
    assert new_state.lane_states["a"].status == "completed"
    assert new_state.lane_states["b"].status == "running"


@pytest.mark.asyncio
async def test_watch_tick_marks_tranche_completed_when_all_lanes_done():
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "completed", "b": "completed"})
    artifact_store = _FakeArtifactStore(
        {
            "a": _make_artifact(lane_id="a", status="completed", run_id="run-a"),
            "b": _make_artifact(lane_id="b", status="completed", run_id="run-b"),
        }
    )

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="adaptive",
        artifact_store=artifact_store,
    )

    assert new_state.status == "completed"


@pytest.mark.asyncio
async def test_watch_tick_fire_and_forget_auto_advances():
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "review_passed"})
    artifact_store = _FakeArtifactStore({"a": _make_artifact(lane_id="a", status="review_passed")})
    mock_integrate = AsyncMock(return_value={"recommendation": "merge", "executed": True})

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="fire_and_forget",
        integrate_fn=mock_integrate,
        artifact_store=artifact_store,
    )

    mock_integrate.assert_awaited_once()
    assert new_state.lane_states["a"].status == "completed"


@pytest.mark.asyncio
async def test_watch_tick_integrates_when_receipt_projection_precedes_integrate() -> None:
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "review_passed"})
    artifact_store = _FakeArtifactStore(
        {
            "a": _make_artifact(
                lane_id="a",
                status="review_passed",
                metadata={"receipt_id": "receipt-1", "lease_id": "lease-1"},
            )
        }
    )
    receipt = CompletionReceipt(
        receipt_id="receipt-1",
        lease_id="lease-1",
        task_id="task-1",
        owner_agent="codex",
        owner_session_id="sess-1",
        branch="feat-branch",
        worktree_path="/tmp/worktree",
        pr_url="https://github.com/org/repo/pull/42",
    )
    store = _FakeCoordinationStore(receipts={"receipt-1": receipt})
    mock_integrate = AsyncMock(return_value={"recommendation": "merge", "executed": True})

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="fire_and_forget",
        integrate_fn=mock_integrate,
        artifact_store=artifact_store,
        store=store,
    )

    mock_integrate.assert_awaited_once()
    assert new_state.lane_states["a"].status == "completed"


@pytest.mark.asyncio
async def test_watch_tick_marks_tranche_needs_human_when_cascade_flags_manual_repair() -> None:
    from aragora.swarm.tranche_watch import watch_tick

    state = _make_state(lane_statuses={"a": "review_passed", "b": "waiting_for_merge"})
    artifact_store = _FakeArtifactStore(
        {
            "a": _make_artifact(lane_id="a", status="review_passed"),
            "b": _make_artifact(lane_id="b", status="waiting_for_merge"),
        }
    )
    mock_integrate = AsyncMock(
        return_value={
            "recommendation": "merge",
            "executed": True,
            "cascade_report": {
                "merged_lane_id": "a",
                "downstream": [
                    {
                        "lane_id": "b",
                        "pr_url": "https://github.com/org/repo/pull/99",
                        "action": "needs_restack",
                        "reason": "Downstream PR was closed after the upstream merge.",
                    }
                ],
                "clean": False,
                "needs_human": True,
            },
        }
    )

    new_state = await watch_tick(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        autonomy_mode="fire_and_forget",
        integrate_fn=mock_integrate,
        artifact_store=artifact_store,
    )

    mock_integrate.assert_awaited_once()
    assert new_state.lane_states["a"].status == "completed"
    assert new_state.lane_states["b"].status == "needs_human"
    assert new_state.status == "needs_human"


@pytest.mark.asyncio
async def test_watch_loop_exits_on_aborted_status() -> None:
    from aragora.swarm.tranche_watch import watch_loop

    state = _make_state(lane_statuses={"a": "running"})
    state.status = "aborted"

    result = await watch_loop(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        interval_seconds=0,
        max_ticks=5,
    )

    assert result.status == "aborted"


@pytest.mark.asyncio
async def test_watch_loop_refreshes_driver_heartbeat() -> None:
    from aragora.swarm.tranche_watch import watch_loop

    state = claim_driver(
        TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive"),
        session_id="sess-1",
    )
    before = state.driver_heartbeat

    result = await watch_loop(
        state,
        manifest=SimpleNamespace(manifest_id="m1"),
        interval_seconds=0,
        max_ticks=1,
        driver_session_id="sess-1",
    )

    assert before is not None
    assert result.driver_heartbeat is not None
    assert result.driver_heartbeat >= before
