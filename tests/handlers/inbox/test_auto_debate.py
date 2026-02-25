"""
Tests for inbox auto-debate trigger.

Tests:
- Trigger eligibility evaluation (priority, cooldown, rate limit)
- Debate creation from inbox items
- Event emission on trigger
- Rate limiting behavior
- Cooldown behavior
"""

import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from aragora.server.handlers.inbox.auto_debate import (
    InboxDebateTrigger,
    DebateTriggerResult,
    get_inbox_debate_trigger,
    process_reprioritization_debates,
    _DEBATE_COOLDOWN_SECONDS,
    _MAX_DEBATES_PER_HOUR,
)


class TestInboxDebateTrigger:
    """Tests for InboxDebateTrigger eligibility checks."""

    def test_critical_priority_triggers(self):
        trigger = InboxDebateTrigger()
        should, reason = trigger.should_trigger("email-1", "critical")
        assert should is True
        assert "eligible" in reason

    def test_non_critical_priority_skipped(self):
        trigger = InboxDebateTrigger()
        should, reason = trigger.should_trigger("email-1", "high")
        assert should is False
        assert "below threshold" in reason

    def test_non_critical_with_force_triggers(self):
        trigger = InboxDebateTrigger()
        should, reason = trigger.should_trigger("email-1", "high", force=True)
        assert should is True

    def test_cooldown_prevents_retrigger(self):
        trigger = InboxDebateTrigger()
        trigger._cooldowns["email-1"] = time.time()

        should, reason = trigger.should_trigger("email-1", "critical")
        assert should is False
        assert "cooldown" in reason

    def test_expired_cooldown_allows_trigger(self):
        trigger = InboxDebateTrigger()
        trigger._cooldowns["email-1"] = time.time() - _DEBATE_COOLDOWN_SECONDS - 1

        should, reason = trigger.should_trigger("email-1", "critical")
        assert should is True

    def test_rate_limit_prevents_trigger(self):
        trigger = InboxDebateTrigger()
        trigger._recent_triggers = [time.time()] * _MAX_DEBATES_PER_HOUR

        should, reason = trigger.should_trigger("email-2", "critical")
        assert should is False
        assert "rate limit" in reason

    def test_old_triggers_dont_count(self):
        trigger = InboxDebateTrigger()
        # All triggers from > 1 hour ago
        old_time = time.time() - 3700
        trigger._recent_triggers = [old_time] * _MAX_DEBATES_PER_HOUR

        should, reason = trigger.should_trigger("email-2", "critical")
        assert should is True

    def test_different_emails_independent_cooldowns(self):
        trigger = InboxDebateTrigger()
        trigger._cooldowns["email-1"] = time.time()

        should, _ = trigger.should_trigger("email-2", "critical")
        assert should is True


class TestTriggerDebate:
    """Tests for the actual debate triggering."""

    @pytest.mark.asyncio
    async def test_trigger_records_cooldown(self):
        trigger = InboxDebateTrigger()

        with patch(
            "aragora.server.handlers.playground._run_inline_mock_debate",
            return_value={"id": "debate-123", "topic": "test"},
        ), patch(
            "aragora.events.dispatcher.dispatch_event",
        ):
            result = await trigger.trigger_debate(
                email_id="email-1",
                subject="Urgent: Budget approval needed",
                body_preview="We need to approve the Q3 budget...",
                sender="cfo@company.com",
            )

        assert result.triggered is True
        assert "email-1" in trigger._cooldowns
        assert len(trigger._recent_triggers) == 1

    @pytest.mark.asyncio
    async def test_trigger_emits_event(self):
        trigger = InboxDebateTrigger()

        with patch(
            "aragora.server.handlers.playground._run_inline_mock_debate",
            return_value={"id": "debate-456"},
        ), patch(
            "aragora.events.dispatcher.dispatch_event"
        ) as mock_dispatch:
            trigger._emit_trigger_event("email-1", "debate-456", "Test Subject")

        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "inbox_debate_triggered"
        assert call_args[0][1]["email_id"] == "email-1"
        assert call_args[0][1]["debate_id"] == "debate-456"

    @pytest.mark.asyncio
    async def test_trigger_result_structure(self):
        """Verify DebateTriggerResult has expected fields."""
        result = DebateTriggerResult(
            triggered=False,
            debate_id=None,
            reason="test reason",
            email_id="email-1",
        )
        assert result.triggered is False
        assert result.debate_id is None
        assert result.reason == "test reason"
        assert result.email_id == "email-1"


class TestProcessReprioritizationDebates:
    """Tests for the high-level reprioritization debate processing."""

    @pytest.mark.asyncio
    async def test_no_debates_when_disabled(self):
        results = await process_reprioritization_debates(
            changes=[{"email_id": "e1", "new_priority": "critical"}],
            email_cache={},
            auto_debate=False,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_skips_non_critical_changes(self):
        results = await process_reprioritization_debates(
            changes=[{"email_id": "e1", "new_priority": "high"}],
            email_cache={},
            auto_debate=True,
        )
        assert len(results) == 1
        assert results[0].triggered is False
        assert "below threshold" in results[0].reason

    @pytest.mark.asyncio
    async def test_processes_critical_changes(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "subject": "Urgent Decision",
            "snippet": "Need input on...",
            "from": "boss@company.com",
        }

        with patch.object(
            InboxDebateTrigger,
            "trigger_debate",
            new_callable=AsyncMock,
            return_value=DebateTriggerResult(
                triggered=True,
                debate_id="d-123",
                reason="ok",
                email_id="e1",
            ),
        ):
            results = await process_reprioritization_debates(
                changes=[{"email_id": "e1", "new_priority": "critical"}],
                email_cache=mock_cache,
                auto_debate=True,
            )

        assert len(results) == 1
        assert results[0].triggered is True
        assert results[0].debate_id == "d-123"


class TestGetInboxDebateTrigger:
    """Tests for the singleton accessor."""

    def test_returns_same_instance(self):
        # Reset singleton
        import aragora.server.handlers.inbox.auto_debate as mod
        mod._trigger = None

        t1 = get_inbox_debate_trigger()
        t2 = get_inbox_debate_trigger()
        assert t1 is t2

        # Cleanup
        mod._trigger = None
