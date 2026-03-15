"""Focused tests for hardened orchestrator budget enforcement."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.autonomous_orchestrator import AgentAssignment, Track, reset_orchestrator
from aragora.nomic.hardened_orchestrator import HardenedOrchestrator
from aragora.nomic.task_decomposer import SubTask


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_orchestrator()
    yield
    reset_orchestrator()


def _make_subtask(id: str = "sub-1") -> SubTask:
    return SubTask(
        id=id,
        title=f"Task {id}",
        description=f"Description for {id}",
        file_scope=[],
        estimated_complexity="medium",
    )


def _make_assignment(subtask: SubTask | None = None) -> AgentAssignment:
    return AgentAssignment(
        subtask=subtask or _make_subtask(),
        track=Track.DEVELOPER,
        agent_type="claude",
    )


@pytest.mark.asyncio
async def test_projected_over_budget_skips_assignment_before_execution():
    orch = HardenedOrchestrator(budget_limit_usd=1.0)
    orch._budget_spent_usd = 0.95
    orch._total_cost_usd = 0.95

    assignment = _make_assignment()
    orch._active_assignments.append(assignment)

    allowed = orch._check_budget_allows(assignment)

    assert allowed is False
    assert assignment.status == "skipped"
    assert assignment.result == {"reason": "budget_exceeded"}


@pytest.mark.asyncio
async def test_inflight_budget_reservation_blocks_second_assignment():
    orch = HardenedOrchestrator(budget_limit_usd=0.15)
    first = _make_assignment(_make_subtask("first"))
    second = _make_assignment(_make_subtask("second"))
    orch._active_assignments.extend([first, second])

    assert orch._check_budget_allows(first) is True
    assert orch._budget_reserved_usd == pytest.approx(0.10)

    allowed = orch._check_budget_allows(second)

    assert allowed is False
    assert second.status == "skipped"
    assert second.result == {"reason": "budget_exceeded"}


@pytest.mark.asyncio
async def test_budget_spend_is_visible_after_execution():
    orch = HardenedOrchestrator(
        budget_limit_usd=2.0,
        use_worktree_isolation=False,
        enable_gauntlet_validation=False,
    )
    assignment = _make_assignment()
    orch._active_assignments.append(assignment)

    with patch(
        "aragora.nomic.autonomous_orchestrator.AutonomousOrchestrator._execute_single_assignment",
        new_callable=AsyncMock,
    ) as mock_parent:
        assignment.status = "completed"
        await orch._execute_single_assignment(assignment, max_cycles=1)

    mock_parent.assert_awaited_once()
    event = orch._spectate_events[-1]
    assert event["type"] == "budget_update"
    assert event["subtask"] == assignment.subtask.id
    assert event["total_spent"] == 0.1
    assert event["limit"] == 2.0
