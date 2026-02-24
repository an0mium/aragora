"""
Tests for the ApprovalFlowManager and related approval flow logic.

Verifies:
- Approval flow creation and state tracking
- Multi-approver workflows (require N of M)
- State transitions: PENDING -> APPROVED / REJECTED / ESCALATED / RE_DEBATE
- Duplicate vote prevention
- Timeout-based auto-escalation
- Event listener notifications
- Governance store persistence
- get_status and list_pending operations
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.integrations.approval_flow import (
    ApprovalDecision,
    ApprovalFlow,
    ApprovalFlowManager,
    ApprovalState,
    ApprovalVote,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> ApprovalFlowManager:
    return ApprovalFlowManager(default_timeout=86400)


@pytest.fixture
def short_timeout_manager() -> ApprovalFlowManager:
    """Manager with very short timeout for escalation tests."""
    return ApprovalFlowManager(default_timeout=1)


# ---------------------------------------------------------------------------
# Creation tests
# ---------------------------------------------------------------------------


class TestApprovalFlowCreation:
    """Tests for creating new approval flows."""

    @pytest.mark.asyncio
    async def test_create_returns_flow_id(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="123.456",
        )
        assert flow_id.startswith("af_")
        assert len(flow_id) > 3

    @pytest.mark.asyncio
    async def test_created_flow_is_pending(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="123.456",
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.state == ApprovalState.PENDING.value

    @pytest.mark.asyncio
    async def test_create_with_custom_required_approvers(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="teams",
            channel_id="conv-1",
            thread_id="msg-1",
            required_approvers=3,
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.required_approvers == 3

    @pytest.mark.asyncio
    async def test_create_stores_metadata(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="123.456",
            metadata={"requested_by": "user123"},
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.metadata.get("requested_by") == "user123"

    @pytest.mark.asyncio
    async def test_create_with_custom_timeout(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="123.456",
            timeout_seconds=3600,
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.timeout_seconds == 3600

    @pytest.mark.asyncio
    async def test_required_approvers_minimum_one(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="123.456",
            required_approvers=0,
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.required_approvers >= 1


# ---------------------------------------------------------------------------
# Single-approver decision tests
# ---------------------------------------------------------------------------


class TestSingleApproverDecisions:
    """Tests for single-approver (N=1) approval flows."""

    @pytest.mark.asyncio
    async def test_approve_transitions_to_approved(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        result = await manager.record_decision(flow_id, "user1", "approved")
        assert result is not None
        assert result.state == ApprovalState.APPROVED.value

    @pytest.mark.asyncio
    async def test_reject_transitions_to_rejected(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        result = await manager.record_decision(flow_id, "user1", "rejected", reason="Not good enough")
        assert result is not None
        assert result.state == ApprovalState.REJECTED.value

    @pytest.mark.asyncio
    async def test_escalate_transitions_to_escalated(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        result = await manager.record_decision(flow_id, "user1", "escalated")
        assert result is not None
        assert result.state == ApprovalState.ESCALATED.value

    @pytest.mark.asyncio
    async def test_redebate_transitions_to_re_debate(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        result = await manager.record_decision(flow_id, "user1", "re_debate")
        assert result is not None
        assert result.state == ApprovalState.RE_DEBATE.value


# ---------------------------------------------------------------------------
# Multi-approver tests
# ---------------------------------------------------------------------------


class TestMultiApproverWorkflow:
    """Tests for multi-approver (N of M) workflows."""

    @pytest.mark.asyncio
    async def test_stays_pending_until_threshold(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
            required_approvers=3,
        )
        # First approval
        result = await manager.record_decision(flow_id, "user1", "approved")
        assert result is not None
        assert result.state == ApprovalState.PENDING.value

        # Second approval
        result = await manager.record_decision(flow_id, "user2", "approved")
        assert result is not None
        assert result.state == ApprovalState.PENDING.value

        # Third approval - should transition
        result = await manager.record_decision(flow_id, "user3", "approved")
        assert result is not None
        assert result.state == ApprovalState.APPROVED.value

    @pytest.mark.asyncio
    async def test_single_rejection_rejects(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
            required_approvers=3,
        )
        # First approval
        await manager.record_decision(flow_id, "user1", "approved")
        # Rejection by second user
        result = await manager.record_decision(flow_id, "user2", "rejected")
        assert result is not None
        assert result.state == ApprovalState.REJECTED.value

    @pytest.mark.asyncio
    async def test_approval_count_tracking(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
            required_approvers=3,
        )
        await manager.record_decision(flow_id, "user1", "approved")
        await manager.record_decision(flow_id, "user2", "approved")
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.approval_count == 2


# ---------------------------------------------------------------------------
# Duplicate vote tests
# ---------------------------------------------------------------------------


class TestDuplicateVotePrevention:
    """Tests for preventing duplicate votes."""

    @pytest.mark.asyncio
    async def test_duplicate_vote_ignored(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
            required_approvers=2,
        )
        await manager.record_decision(flow_id, "user1", "approved")
        # Same user votes again
        result = await manager.record_decision(flow_id, "user1", "approved")
        assert result is not None
        assert result.approval_count == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_has_voted_check(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert not flow.has_voted("user1")

        await manager.record_decision(flow_id, "user1", "approved")
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.has_voted("user1")


# ---------------------------------------------------------------------------
# Timeout and escalation tests
# ---------------------------------------------------------------------------


class TestTimeoutEscalation:
    """Tests for timeout-based auto-escalation."""

    @pytest.mark.asyncio
    async def test_expired_flow_auto_escalates_on_status_check(
        self, short_timeout_manager: ApprovalFlowManager
    ) -> None:
        flow_id = await short_timeout_manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
            timeout_seconds=0,  # Immediate timeout
        )

        # Manually set created_at to the past
        flow = short_timeout_manager._flows[flow_id]
        flow.created_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        result = short_timeout_manager.get_status(flow_id)
        assert result is not None
        assert result.state == ApprovalState.ESCALATED.value

    @pytest.mark.asyncio
    async def test_non_expired_flow_stays_pending(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        flow = manager.get_status(flow_id)
        assert flow is not None
        assert flow.state == ApprovalState.PENDING.value

    def test_is_expired_property(self) -> None:
        flow = ApprovalFlow(
            flow_id="af_test",
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="t1",
            timeout_seconds=10,
            created_at=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat(),
        )
        assert flow.is_expired is True

    def test_is_not_expired(self) -> None:
        flow = ApprovalFlow(
            flow_id="af_test",
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="t1",
            timeout_seconds=86400,
        )
        assert flow.is_expired is False


# ---------------------------------------------------------------------------
# Event listener tests
# ---------------------------------------------------------------------------


class TestEventListeners:
    """Tests for event listener notifications."""

    @pytest.mark.asyncio
    async def test_creation_emits_event(self, manager: ApprovalFlowManager) -> None:
        events: list[tuple[str, ApprovalFlow]] = []
        manager.add_event_listener(lambda et, f: events.append((et, f)))

        await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        assert len(events) == 1
        assert events[0][0] == "approval_created"

    @pytest.mark.asyncio
    async def test_decision_emits_event(self, manager: ApprovalFlowManager) -> None:
        events: list[tuple[str, ApprovalFlow]] = []
        manager.add_event_listener(lambda et, f: events.append((et, f)))

        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        await manager.record_decision(flow_id, "user1", "approved")
        # Should have "approval_created" + "decision_recorded"
        assert len(events) == 2
        assert events[1][0] == "decision_recorded"

    @pytest.mark.asyncio
    async def test_failing_listener_does_not_crash(self, manager: ApprovalFlowManager) -> None:
        def bad_listener(et: str, f: ApprovalFlow) -> None:
            raise ValueError("boom")

        manager.add_event_listener(bad_listener)
        # Should not raise
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        assert flow_id is not None


# ---------------------------------------------------------------------------
# Lookup and listing tests
# ---------------------------------------------------------------------------


class TestLookupAndListing:
    """Tests for get_status_by_debate and list_pending."""

    @pytest.mark.asyncio
    async def test_get_by_debate_id(self, manager: ApprovalFlowManager) -> None:
        await manager.create_approval(
            debate_id="d42", receipt_id="r42", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        flow = manager.get_status_by_debate("d42")
        assert flow is not None
        assert flow.debate_id == "d42"

    @pytest.mark.asyncio
    async def test_get_by_debate_id_not_found(self, manager: ApprovalFlowManager) -> None:
        flow = manager.get_status_by_debate("nonexistent")
        assert flow is None

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, manager: ApprovalFlowManager) -> None:
        flow = manager.get_status("nonexistent")
        assert flow is None

    @pytest.mark.asyncio
    async def test_list_pending_returns_only_pending(self, manager: ApprovalFlowManager) -> None:
        f1 = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        f2 = await manager.create_approval(
            debate_id="d2", receipt_id="r2", channel="slack",
            channel_id="C02", thread_id="t2",
        )
        # Approve f1
        await manager.record_decision(f1, "user1", "approved")

        pending = manager.list_pending()
        assert len(pending) == 1
        assert pending[0].flow_id == f2

    @pytest.mark.asyncio
    async def test_list_pending_filter_by_channel(self, manager: ApprovalFlowManager) -> None:
        await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        await manager.create_approval(
            debate_id="d2", receipt_id="r2", channel="teams",
            channel_id="conv-1", thread_id="m1",
        )

        slack_pending = manager.list_pending(channel="slack")
        assert len(slack_pending) == 1
        assert slack_pending[0].channel == "slack"

        teams_pending = manager.list_pending(channel="teams")
        assert len(teams_pending) == 1
        assert teams_pending[0].channel == "teams"


# ---------------------------------------------------------------------------
# Invalid decision tests
# ---------------------------------------------------------------------------


class TestInvalidDecisions:
    """Tests for edge cases in decision recording."""

    @pytest.mark.asyncio
    async def test_invalid_decision_value(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        result = await manager.record_decision(flow_id, "user1", "invalid_value")
        assert result is None

    @pytest.mark.asyncio
    async def test_decision_on_nonexistent_flow(self, manager: ApprovalFlowManager) -> None:
        result = await manager.record_decision("nonexistent", "user1", "approved")
        assert result is None

    @pytest.mark.asyncio
    async def test_decision_on_already_decided_flow(self, manager: ApprovalFlowManager) -> None:
        flow_id = await manager.create_approval(
            debate_id="d1", receipt_id="r1", channel="slack",
            channel_id="C01", thread_id="t1",
        )
        await manager.record_decision(flow_id, "user1", "approved")
        # Try to add another decision after approval
        result = await manager.record_decision(flow_id, "user2", "rejected")
        assert result is not None
        # State should still be approved (ignored second decision)
        assert result.state == ApprovalState.APPROVED.value


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for ApprovalFlow serialization."""

    def test_to_dict(self) -> None:
        flow = ApprovalFlow(
            flow_id="af_test",
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="t1",
        )
        d = flow.to_dict()
        assert d["flow_id"] == "af_test"
        assert d["debate_id"] == "d1"
        assert d["state"] == "pending"
        assert isinstance(d["votes"], list)

    def test_vote_in_serialization(self) -> None:
        vote = ApprovalVote(user_id="u1", decision="approved", reason="Looks good")
        flow = ApprovalFlow(
            flow_id="af_test",
            debate_id="d1",
            receipt_id="r1",
            channel="slack",
            channel_id="C01",
            thread_id="t1",
            votes=[vote],
        )
        d = flow.to_dict()
        assert len(d["votes"]) == 1
        assert d["votes"][0]["user_id"] == "u1"
        assert d["votes"][0]["decision"] == "approved"
        assert d["votes"][0]["reason"] == "Looks good"
