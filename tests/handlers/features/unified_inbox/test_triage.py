"""Tests for the Unified Inbox triage module.

Covers all public functions in aragora/server/handlers/features/unified_inbox/triage.py:
- run_triage            - Batch triage of messages with store persistence
- triage_single_message - Single message triage via debate or heuristic fallback
- _action_from_priority - Map priority tier strings to TriageAction enums

Tests include:
- Happy path for each priority tier (critical, high, low, medium/other)
- Debate-based triage path (Arena importable via sys.modules mock)
- Heuristic fallback triage (Arena ImportError)
- Batch triage with multiple messages
- Batch triage with partial failures (RuntimeError, ValueError, TypeError, KeyError,
  AttributeError, OSError)
- Store save_triage_result and update_message_triage calls
- Message mutation (triage_action, triage_rationale set on message object)
- Empty message list
- Context and tenant_id pass-through
- Confidence levels (0.85 debate vs 0.7 heuristic)
- Agent involvement (debate vs heuristic)
- Debate summary presence / absence
- triage_to_record converter function invocation
- Store failure scenarios (save fails, update fails)
- Edge cases: unknown priority tiers, empty strings, case sensitivity
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    EmailProvider,
    TriageAction,
    TriageResult,
    UnifiedMessage,
)
from aragora.server.handlers.features.unified_inbox.triage import (
    _action_from_priority,
    run_triage,
    triage_single_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_message(
    msg_id: str = "msg-001",
    priority_tier: str = "medium",
    subject: str = "Test Subject",
    sender_email: str = "alice@example.com",
    sender_name: str = "Alice",
    snippet: str = "Preview text here",
) -> UnifiedMessage:
    """Create a UnifiedMessage for testing."""
    return UnifiedMessage(
        id=msg_id,
        account_id="acct-001",
        provider=EmailProvider.GMAIL,
        external_id=f"ext-{msg_id}",
        subject=subject,
        sender_email=sender_email,
        sender_name=sender_name,
        recipients=["bob@example.com"],
        cc=[],
        received_at=_NOW,
        snippet=snippet,
        body_preview="Full preview text here",
        is_read=False,
        is_starred=False,
        has_attachments=False,
        labels=["inbox"],
        priority_tier=priority_tier,
    )


def _make_store() -> MagicMock:
    """Create a mock store with async save/update methods."""
    store = MagicMock()
    store.save_triage_result = AsyncMock(return_value=None)
    store.update_message_triage = AsyncMock(return_value=None)
    return store


def _triage_to_record(result: TriageResult) -> dict[str, Any]:
    """Mock triage_to_record conversion function."""
    return {
        "message_id": result.message_id,
        "recommended_action": result.recommended_action.value,
        "confidence": result.confidence,
        "rationale": result.rationale,
    }


def _fake_debate_module() -> MagicMock:
    """Build a fake aragora.debate module whose Arena/Environment/DebateProtocol are MagicMocks."""
    mod = MagicMock()
    mod.Arena = MagicMock()
    mod.Environment = MagicMock()
    mod.DebateProtocol = MagicMock()
    return mod


# ===========================================================================
# _action_from_priority
# ===========================================================================


class TestActionFromPriority:
    """Tests for the _action_from_priority helper."""

    def test_critical_returns_respond_urgent(self):
        assert _action_from_priority("critical") == TriageAction.RESPOND_URGENT

    def test_high_returns_respond_normal(self):
        assert _action_from_priority("high") == TriageAction.RESPOND_NORMAL

    def test_low_returns_archive(self):
        assert _action_from_priority("low") == TriageAction.ARCHIVE

    def test_medium_returns_defer(self):
        assert _action_from_priority("medium") == TriageAction.DEFER

    def test_unknown_string_returns_defer(self):
        assert _action_from_priority("unknown") == TriageAction.DEFER

    def test_empty_string_returns_defer(self):
        assert _action_from_priority("") == TriageAction.DEFER

    def test_uppercase_critical_returns_defer(self):
        """Priority matching is case-sensitive; 'Critical' != 'critical'."""
        assert _action_from_priority("Critical") == TriageAction.DEFER

    def test_uppercase_high_returns_defer(self):
        assert _action_from_priority("HIGH") == TriageAction.DEFER

    def test_uppercase_low_returns_defer(self):
        assert _action_from_priority("LOW") == TriageAction.DEFER

    def test_none_like_string_returns_defer(self):
        assert _action_from_priority("none") == TriageAction.DEFER

    def test_whitespace_returns_defer(self):
        assert _action_from_priority("  ") == TriageAction.DEFER

    def test_numeric_string_returns_defer(self):
        assert _action_from_priority("1") == TriageAction.DEFER


# ===========================================================================
# triage_single_message - debate path (Arena importable)
# ===========================================================================


class TestTriageSingleMessageDebatePath:
    """Tests for triage_single_message when aragora.debate is importable.

    We inject a fake module into sys.modules so the ``from aragora.debate
    import Arena, Environment, DebateProtocol`` inside the function body
    succeeds, exercising the debate code path (confidence 0.85, agents
    populated, debate_summary present).
    """

    @pytest.fixture(autouse=True)
    def _inject_debate_module(self):
        """Temporarily inject a fake aragora.debate module."""
        fake = _fake_debate_module()
        original = sys.modules.get("aragora.debate")
        sys.modules["aragora.debate"] = fake
        yield
        # Restore
        if original is None:
            sys.modules.pop("aragora.debate", None)
        else:
            sys.modules["aragora.debate"] = original

    @pytest.mark.asyncio
    async def test_critical_returns_respond_urgent(self):
        msg = _make_message(priority_tier="critical")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.RESPOND_URGENT

    @pytest.mark.asyncio
    async def test_high_returns_respond_normal(self):
        msg = _make_message(priority_tier="high")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.RESPOND_NORMAL

    @pytest.mark.asyncio
    async def test_low_returns_archive(self):
        msg = _make_message(priority_tier="low")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.ARCHIVE

    @pytest.mark.asyncio
    async def test_medium_returns_defer(self):
        msg = _make_message(priority_tier="medium")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.DEFER

    @pytest.mark.asyncio
    async def test_confidence_is_085(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_agents_involved(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.agents_involved == ["support_analyst", "product_expert"]

    @pytest.mark.asyncio
    async def test_debate_summary_present(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.debate_summary == "Multi-agent analysis completed"

    @pytest.mark.asyncio
    async def test_message_id_matches(self):
        msg = _make_message(msg_id="msg-42")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.message_id == "msg-42"

    @pytest.mark.asyncio
    async def test_rationale_contains_priority_tier(self):
        msg = _make_message(priority_tier="critical")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert "critical" in result.rationale

    @pytest.mark.asyncio
    async def test_no_suggested_response(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.suggested_response is None

    @pytest.mark.asyncio
    async def test_no_delegate_to(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.delegate_to is None

    @pytest.mark.asyncio
    async def test_no_schedule_for(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.schedule_for is None

    @pytest.mark.asyncio
    async def test_rationale_mentions_sender_analysis(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert "sender analysis" in result.rationale


# ===========================================================================
# triage_single_message - heuristic fallback path (Arena NOT importable)
# ===========================================================================


class TestTriageSingleMessageHeuristicPath:
    """Tests for triage_single_message when aragora.debate is NOT importable.

    Setting sys.modules["aragora.debate"] = None causes ``from aragora.debate
    import ...`` to raise ImportError, so the function falls through to the
    heuristic path (confidence 0.7, no agents, no debate summary).
    """

    @pytest.fixture(autouse=True)
    def _block_debate_module(self):
        """Force aragora.debate to be un-importable."""
        original = sys.modules.get("aragora.debate")
        sys.modules["aragora.debate"] = None  # type: ignore[assignment]
        yield
        if original is None:
            sys.modules.pop("aragora.debate", None)
        else:
            sys.modules["aragora.debate"] = original

    @pytest.mark.asyncio
    async def test_critical_returns_respond_urgent(self):
        msg = _make_message(priority_tier="critical")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.RESPOND_URGENT

    @pytest.mark.asyncio
    async def test_high_returns_respond_normal(self):
        msg = _make_message(priority_tier="high")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.RESPOND_NORMAL

    @pytest.mark.asyncio
    async def test_low_returns_archive(self):
        msg = _make_message(priority_tier="low")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.ARCHIVE

    @pytest.mark.asyncio
    async def test_medium_returns_defer(self):
        msg = _make_message(priority_tier="medium")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.recommended_action == TriageAction.DEFER

    @pytest.mark.asyncio
    async def test_confidence_is_07(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_no_agents_involved(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.agents_involved == []

    @pytest.mark.asyncio
    async def test_no_debate_summary(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.debate_summary is None

    @pytest.mark.asyncio
    async def test_rationale_contains_heuristic(self):
        msg = _make_message(priority_tier="high")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert "Heuristic" in result.rationale

    @pytest.mark.asyncio
    async def test_rationale_contains_priority_tier(self):
        msg = _make_message(priority_tier="low")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert "low" in result.rationale

    @pytest.mark.asyncio
    async def test_message_id_matches(self):
        msg = _make_message(msg_id="msg-99")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert result.message_id == "msg-99"


# ===========================================================================
# run_triage - batch triage with store persistence
# ===========================================================================


class TestRunTriage:
    """Tests for the run_triage batch function."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self):
        store = _make_store()
        results = await run_triage([], {}, "tenant-1", store, _triage_to_record)
        assert results == []
        store.save_triage_result.assert_not_called()
        store.update_message_triage.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_message_returns_one_result(self):
        store = _make_store()
        msg = _make_message(priority_tier="critical")
        results = await run_triage([msg], {}, "tenant-1", store, _triage_to_record)
        assert len(results) == 1
        assert results[0].recommended_action == TriageAction.RESPOND_URGENT

    @pytest.mark.asyncio
    async def test_multiple_messages_returns_all_results(self):
        store = _make_store()
        msgs = [
            _make_message(msg_id="msg-1", priority_tier="critical"),
            _make_message(msg_id="msg-2", priority_tier="high"),
            _make_message(msg_id="msg-3", priority_tier="low"),
        ]
        results = await run_triage(msgs, {}, "tenant-1", store, _triage_to_record)
        assert len(results) == 3
        actions = [r.recommended_action for r in results]
        assert TriageAction.RESPOND_URGENT in actions
        assert TriageAction.RESPOND_NORMAL in actions
        assert TriageAction.ARCHIVE in actions

    @pytest.mark.asyncio
    async def test_store_save_called_per_message(self):
        store = _make_store()
        msgs = [_make_message(msg_id="msg-1"), _make_message(msg_id="msg-2")]
        await run_triage(msgs, {}, "tenant-1", store, _triage_to_record)
        assert store.save_triage_result.call_count == 2

    @pytest.mark.asyncio
    async def test_store_update_called_per_message(self):
        store = _make_store()
        msgs = [_make_message(msg_id="msg-1"), _make_message(msg_id="msg-2")]
        await run_triage(msgs, {}, "tenant-1", store, _triage_to_record)
        assert store.update_message_triage.call_count == 2

    @pytest.mark.asyncio
    async def test_store_save_receives_tenant_id(self):
        store = _make_store()
        msg = _make_message()
        await run_triage([msg], {}, "my-tenant", store, _triage_to_record)
        call_args = store.save_triage_result.call_args
        assert call_args[0][0] == "my-tenant"

    @pytest.mark.asyncio
    async def test_store_save_receives_record_dict(self):
        store = _make_store()
        msg = _make_message(priority_tier="high")
        await run_triage([msg], {}, "t1", store, _triage_to_record)
        record = store.save_triage_result.call_args[0][1]
        assert isinstance(record, dict)
        assert record["recommended_action"] == "respond_normal"

    @pytest.mark.asyncio
    async def test_store_update_receives_tenant_and_msg_id(self):
        store = _make_store()
        msg = _make_message(msg_id="msg-42")
        await run_triage([msg], {}, "tenant-x", store, _triage_to_record)
        call_args = store.update_message_triage.call_args
        assert call_args[0][0] == "tenant-x"
        assert call_args[0][1] == "msg-42"

    @pytest.mark.asyncio
    async def test_store_update_receives_action_value_string(self):
        store = _make_store()
        msg = _make_message(priority_tier="critical")
        await run_triage([msg], {}, "tenant-1", store, _triage_to_record)
        action_str = store.update_message_triage.call_args[0][2]
        assert action_str == "respond_urgent"

    @pytest.mark.asyncio
    async def test_store_update_receives_rationale(self):
        store = _make_store()
        msg = _make_message()
        await run_triage([msg], {}, "tenant-1", store, _triage_to_record)
        rationale = store.update_message_triage.call_args[0][3]
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    @pytest.mark.asyncio
    async def test_message_triage_action_mutated(self):
        store = _make_store()
        msg = _make_message(priority_tier="critical")
        assert msg.triage_action is None
        await run_triage([msg], {}, "tenant-1", store, _triage_to_record)
        assert msg.triage_action == TriageAction.RESPOND_URGENT

    @pytest.mark.asyncio
    async def test_message_triage_rationale_mutated(self):
        store = _make_store()
        msg = _make_message(priority_tier="high")
        assert msg.triage_rationale is None
        await run_triage([msg], {}, "tenant-1", store, _triage_to_record)
        assert msg.triage_rationale is not None
        assert len(msg.triage_rationale) > 0

    @pytest.mark.asyncio
    async def test_triage_to_record_converter_called(self):
        store = _make_store()
        mock_converter = MagicMock(return_value={"converted": True})
        msg = _make_message()
        await run_triage([msg], {}, "tenant-1", store, mock_converter)
        mock_converter.assert_called_once()
        arg = mock_converter.call_args[0][0]
        assert isinstance(arg, TriageResult)

    @pytest.mark.asyncio
    async def test_context_forwarded_to_triage_single_message(self):
        store = _make_store()
        msg = _make_message()
        ctx = {"user_prefs": {"timezone": "UTC"}}
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
        ) as mock_triage:
            mock_triage.return_value = TriageResult(
                message_id=msg.id,
                recommended_action=TriageAction.DEFER,
                confidence=0.85,
                rationale="test",
                suggested_response=None,
                delegate_to=None,
                schedule_for=None,
                agents_involved=[],
                debate_summary=None,
            )
            await run_triage([msg], ctx, "tenant-1", store, _triage_to_record)
            mock_triage.assert_called_once_with(msg, ctx, "tenant-1")

    @pytest.mark.asyncio
    async def test_tenant_id_forwarded_to_triage_single_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
        ) as mock_triage:
            mock_triage.return_value = TriageResult(
                message_id=msg.id,
                recommended_action=TriageAction.DEFER,
                confidence=0.7,
                rationale="test",
                suggested_response=None,
                delegate_to=None,
                schedule_for=None,
                agents_involved=[],
                debate_summary=None,
            )
            await run_triage([msg], {}, "special-tenant", store, _triage_to_record)
            assert mock_triage.call_args[0][2] == "special-tenant"

    @pytest.mark.asyncio
    async def test_results_order_matches_message_order(self):
        store = _make_store()
        msgs = [
            _make_message(msg_id="a", priority_tier="critical"),
            _make_message(msg_id="b", priority_tier="low"),
            _make_message(msg_id="c", priority_tier="high"),
        ]
        results = await run_triage(msgs, {}, "t1", store, _triage_to_record)
        assert [r.message_id for r in results] == ["a", "b", "c"]


# ===========================================================================
# run_triage - error handling
# ===========================================================================


class TestRunTriageErrorHandling:
    """Tests for run_triage error resilience (each caught exception type)."""

    @pytest.mark.asyncio
    async def test_runtime_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=RuntimeError("crash"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_value_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=ValueError("bad"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_type_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=TypeError("wrong"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_key_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=KeyError("missing"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_attribute_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=AttributeError("no attr"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_os_error_skips_message(self):
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=OSError("disk"),
        ):
            results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_partial_failure_keeps_successful_results(self):
        """First message fails, second succeeds."""
        store = _make_store()
        msgs = [
            _make_message(msg_id="fail-1"),
            _make_message(msg_id="ok-2", priority_tier="high"),
        ]
        call_count = 0

        async def _side_effect(message, context, tenant_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return TriageResult(
                message_id=message.id,
                recommended_action=TriageAction.RESPOND_NORMAL,
                confidence=0.85,
                rationale="ok",
                suggested_response=None,
                delegate_to=None,
                schedule_for=None,
                agents_involved=[],
                debate_summary=None,
            )

        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            results = await run_triage(msgs, {}, "t1", store, _triage_to_record)
        assert len(results) == 1
        assert results[0].message_id == "ok-2"

    @pytest.mark.asyncio
    async def test_store_save_failure_result_still_in_list(self):
        """results.append runs before store.save_triage_result, so RuntimeError
        from the store does not remove the already-appended result."""
        store = _make_store()
        store.save_triage_result = AsyncMock(side_effect=RuntimeError("db down"))
        msg = _make_message()
        results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_store_update_failure_result_still_in_list(self):
        """results.append runs before store.update_message_triage, so OSError
        from the store does not remove the already-appended result."""
        store = _make_store()
        store.update_message_triage = AsyncMock(side_effect=OSError("update fail"))
        msg = _make_message()
        results = await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_triage_error_leaves_message_unmutated(self):
        """If triage_single_message fails, message.triage_action stays None."""
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=ValueError("boom"),
        ):
            await run_triage([msg], {}, "t1", store, _triage_to_record)
        assert msg.triage_action is None
        assert msg.triage_rationale is None

    @pytest.mark.asyncio
    async def test_triage_error_skips_store_calls(self):
        """Store methods not called when triage_single_message fails."""
        store = _make_store()
        msg = _make_message()
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=KeyError("missing"),
        ):
            await run_triage([msg], {}, "t1", store, _triage_to_record)
        store.save_triage_result.assert_not_called()
        store.update_message_triage.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_messages_fail_returns_empty(self):
        """When every message raises, result list is empty."""
        store = _make_store()
        msgs = [_make_message(msg_id=f"m-{i}") for i in range(5)]
        with patch(
            "aragora.server.handlers.features.unified_inbox.triage.triage_single_message",
            new_callable=AsyncMock,
            side_effect=TypeError("all bad"),
        ):
            results = await run_triage(msgs, {}, "t1", store, _triage_to_record)
        assert results == []


# ===========================================================================
# TriageResult structural validation
# ===========================================================================


class TestTriageResultStructure:
    """Validate structural properties of TriageResult objects."""

    @pytest.mark.asyncio
    async def test_result_is_triage_result_instance(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert isinstance(result, TriageResult)

    @pytest.mark.asyncio
    async def test_recommended_action_is_triage_action_enum(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert isinstance(result.recommended_action, TriageAction)

    @pytest.mark.asyncio
    async def test_confidence_between_0_and_1(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_rationale_is_nonempty_string(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    @pytest.mark.asyncio
    async def test_agents_involved_is_list(self):
        msg = _make_message()
        result = await triage_single_message(msg, {}, "tenant-1")
        assert isinstance(result.agents_involved, list)

    @pytest.mark.asyncio
    async def test_message_id_is_string(self):
        msg = _make_message(msg_id="xyz")
        result = await triage_single_message(msg, {}, "tenant-1")
        assert isinstance(result.message_id, str)
        assert result.message_id == "xyz"
