"""Tests for events/security_dispatcher.py — security event dispatcher."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from aragora.events.security_dispatcher import (
    DispatcherConfig,
    DispatcherStats,
    SecurityDispatcher,
    get_security_dispatcher,
    set_security_dispatcher,
    start_security_dispatcher,
    stop_security_dispatcher,
)
from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventEmitter,
    SecurityEventType,
    SecurityFinding,
    SecuritySeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    severity: SecuritySeverity = SecuritySeverity.HIGH,
    event_type: SecurityEventType = SecurityEventType.VULNERABILITY_DETECTED,
    repository: str | None = "org/repo",
    findings: list[SecurityFinding] | None = None,
    event_id: str = "evt-1",
) -> SecurityEvent:
    return SecurityEvent(
        id=event_id,
        event_type=event_type,
        severity=severity,
        repository=repository,
        findings=findings or [],
    )


def _make_critical_finding() -> SecurityFinding:
    return SecurityFinding(
        id="f-1",
        finding_type="vulnerability",
        severity=SecuritySeverity.CRITICAL,
        title="SQL Injection",
        description="Unsanitised input in query builder",
    )


def _make_high_findings(n: int) -> list[SecurityFinding]:
    return [
        SecurityFinding(
            id=f"f-{i}",
            finding_type="vulnerability",
            severity=SecuritySeverity.HIGH,
            title=f"Finding {i}",
            description=f"Description {i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDispatcherConfig:
    def test_defaults(self):
        cfg = DispatcherConfig()
        assert cfg.min_severity == SecuritySeverity.HIGH
        assert cfg.critical_finding_threshold == 1
        assert cfg.high_finding_threshold == 3
        assert cfg.repository_cooldown_seconds == 300
        assert cfg.max_concurrent_debates == 5
        assert cfg.debate_confidence_threshold == 0.7
        assert cfg.debate_timeout_seconds == 300
        assert cfg.auto_start is False

    def test_always_trigger_types(self):
        cfg = DispatcherConfig()
        assert SecurityEventType.CRITICAL_CVE in cfg.always_trigger_types
        assert SecurityEventType.THREAT_DETECTED in cfg.always_trigger_types


class TestDispatcherStats:
    def test_defaults(self):
        stats = DispatcherStats()
        assert stats.events_received == 0
        assert stats.events_filtered == 0
        assert stats.debates_triggered == 0
        assert stats.debates_completed == 0
        assert stats.debates_failed == 0
        assert stats.last_event_time is None
        assert stats.last_debate_time is None


# ---------------------------------------------------------------------------
# SecurityDispatcher — init and lifecycle
# ---------------------------------------------------------------------------


class TestDispatcherLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        assert not dispatcher.is_running

        await dispatcher.start()
        assert dispatcher.is_running

        await dispatcher.stop()
        assert not dispatcher.is_running

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()
        await dispatcher.start()  # should not raise
        assert dispatcher.is_running
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_debates(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        # Simulate a pending task using a long-running coroutine
        async def _hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_hang())
        dispatcher._pending_debates["fake-id"] = task

        await dispatcher.stop()
        assert len(dispatcher._pending_debates) == 0

    def test_set_severity_threshold(self):
        dispatcher = SecurityDispatcher()
        dispatcher.set_severity_threshold(SecuritySeverity.MEDIUM)
        assert dispatcher.config.min_severity == SecuritySeverity.MEDIUM

    def test_set_custom_trigger(self):
        callback = AsyncMock(return_value="debate-1")
        dispatcher = SecurityDispatcher()
        dispatcher.set_custom_trigger(callback)
        assert dispatcher._custom_trigger_callback is callback


# ---------------------------------------------------------------------------
# _should_trigger_debate
# ---------------------------------------------------------------------------


class TestShouldTriggerDebate:
    def test_always_trigger_type(self):
        dispatcher = SecurityDispatcher()
        event = _make_event(
            event_type=SecurityEventType.CRITICAL_CVE, severity=SecuritySeverity.LOW
        )
        assert dispatcher._should_trigger_debate(event) is True

    def test_severity_below_threshold(self):
        dispatcher = SecurityDispatcher(config=DispatcherConfig(min_severity=SecuritySeverity.HIGH))
        event = _make_event(severity=SecuritySeverity.MEDIUM)
        assert dispatcher._should_trigger_debate(event) is False

    def test_severity_at_threshold(self):
        dispatcher = SecurityDispatcher(config=DispatcherConfig(min_severity=SecuritySeverity.HIGH))
        event = _make_event(
            severity=SecuritySeverity.HIGH,
            findings=[_make_critical_finding()],
        )
        assert dispatcher._should_trigger_debate(event) is True

    def test_critical_finding_count_triggers(self):
        dispatcher = SecurityDispatcher(
            config=DispatcherConfig(
                min_severity=SecuritySeverity.HIGH,
                critical_finding_threshold=1,
            )
        )
        event = _make_event(
            severity=SecuritySeverity.HIGH,
            findings=[_make_critical_finding()],
        )
        assert dispatcher._should_trigger_debate(event) is True

    def test_high_finding_count_triggers(self):
        dispatcher = SecurityDispatcher(
            config=DispatcherConfig(
                min_severity=SecuritySeverity.HIGH,
                high_finding_threshold=3,
            )
        )
        event = _make_event(
            severity=SecuritySeverity.HIGH,
            findings=_make_high_findings(3),
        )
        assert dispatcher._should_trigger_debate(event) is True

    def test_high_finding_count_below_threshold(self):
        dispatcher = SecurityDispatcher(
            config=DispatcherConfig(
                min_severity=SecuritySeverity.HIGH,
                high_finding_threshold=5,
                critical_finding_threshold=10,
            )
        )
        event = _make_event(
            severity=SecuritySeverity.HIGH,
            findings=_make_high_findings(2),
        )
        assert dispatcher._should_trigger_debate(event) is False

    def test_is_critical_property_triggers(self):
        dispatcher = SecurityDispatcher(config=DispatcherConfig(min_severity=SecuritySeverity.HIGH))
        event = _make_event(severity=SecuritySeverity.CRITICAL)
        assert dispatcher._should_trigger_debate(event) is True


# ---------------------------------------------------------------------------
# _check_cooldown
# ---------------------------------------------------------------------------


class TestCheckCooldown:
    def test_no_repository(self):
        dispatcher = SecurityDispatcher()
        assert dispatcher._check_cooldown(None) is True

    def test_no_prior_cooldown(self):
        dispatcher = SecurityDispatcher()
        assert dispatcher._check_cooldown("org/repo") is True

    def test_in_cooldown(self):
        dispatcher = SecurityDispatcher(config=DispatcherConfig(repository_cooldown_seconds=300))
        dispatcher._repository_cooldowns["org/repo"] = datetime.now(timezone.utc)
        assert dispatcher._check_cooldown("org/repo") is False

    def test_cooldown_expired(self):
        dispatcher = SecurityDispatcher(config=DispatcherConfig(repository_cooldown_seconds=10))
        dispatcher._repository_cooldowns["org/repo"] = datetime.now(timezone.utc) - timedelta(
            seconds=20
        )
        assert dispatcher._check_cooldown("org/repo") is True


# ---------------------------------------------------------------------------
# _handle_event
# ---------------------------------------------------------------------------


class TestHandleEvent:
    @pytest.mark.asyncio
    async def test_not_running_skips(self):
        dispatcher = SecurityDispatcher()
        event = _make_event()
        await dispatcher._handle_event(event)
        assert dispatcher._stats.events_received == 1

    @pytest.mark.asyncio
    async def test_filtered_event(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(
            emitter=emitter,
            config=DispatcherConfig(min_severity=SecuritySeverity.CRITICAL),
        )
        await dispatcher.start()
        event = _make_event(severity=SecuritySeverity.LOW)
        await dispatcher._handle_event(event)
        assert dispatcher._stats.events_filtered == 1
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_cooldown_blocks_debate(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        # Set cooldown
        dispatcher._repository_cooldowns["org/repo"] = datetime.now(timezone.utc)

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        await dispatcher._handle_event(event)
        assert dispatcher._stats.events_filtered == 1
        assert dispatcher._stats.debates_triggered == 0
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_max_concurrent_blocks(self):
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        config = DispatcherConfig(max_concurrent_debates=1)
        dispatcher = SecurityDispatcher(emitter=emitter, config=config)
        await dispatcher.start()

        # Fill up pending debates with a long-running coroutine
        async def _hang():
            await asyncio.sleep(3600)

        dispatcher._pending_debates["existing"] = asyncio.create_task(_hang())

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        await dispatcher._handle_event(event)
        assert dispatcher._stats.events_filtered == 1
        await dispatcher.stop()


# ---------------------------------------------------------------------------
# _trigger_debate and _run_debate
# ---------------------------------------------------------------------------


class TestTriggerDebate:
    @pytest.mark.asyncio
    async def test_custom_trigger_callback(self):
        callback = AsyncMock(return_value="debate-42")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        dispatcher.set_custom_trigger(callback)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        await dispatcher._handle_event(event)

        # Allow task to complete
        await asyncio.sleep(0.05)

        callback.assert_called_once_with(event)
        assert dispatcher._stats.debates_triggered == 1
        assert dispatcher._stats.debates_completed == 1
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_debate_failure_increments_failed(self):
        callback = AsyncMock(side_effect=RuntimeError("Arena down"))
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        dispatcher.set_custom_trigger(callback)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        await dispatcher._handle_event(event)
        await asyncio.sleep(0.05)

        assert dispatcher._stats.debates_failed == 1
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_sets_cooldown_on_trigger(self):
        callback = AsyncMock(return_value="debate-1")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        dispatcher.set_custom_trigger(callback)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
            repository="org/repo",
        )
        await dispatcher._handle_event(event)
        await asyncio.sleep(0.05)

        assert "org/repo" in dispatcher._repository_cooldowns
        await dispatcher.stop()


# ---------------------------------------------------------------------------
# get_stats / get_pending_debates
# ---------------------------------------------------------------------------


class TestStats:
    def test_get_stats(self):
        dispatcher = SecurityDispatcher()
        stats = dispatcher.get_stats()
        assert stats["events_received"] == 0
        assert stats["debates_pending"] == 0
        assert stats["config"]["min_severity"] == "high"
        assert stats["last_event_time"] is None

    @pytest.mark.asyncio
    async def test_stats_after_events(self):
        callback = AsyncMock(return_value="debate-1")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        dispatcher.set_custom_trigger(callback)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        await dispatcher._handle_event(event)
        await asyncio.sleep(0.05)

        stats = dispatcher.get_stats()
        assert stats["events_received"] == 1
        assert stats["debates_triggered"] == 1
        assert stats["last_event_time"] is not None
        assert stats["last_debate_time"] is not None
        await dispatcher.stop()

    def test_get_pending_debates_empty(self):
        dispatcher = SecurityDispatcher()
        assert dispatcher.get_pending_debates() == []


# ---------------------------------------------------------------------------
# Global instance management
# ---------------------------------------------------------------------------


class TestGlobalInstance:
    @pytest.fixture(autouse=True)
    def _reset_global(self):
        set_security_dispatcher(None)  # type: ignore[arg-type]
        yield
        set_security_dispatcher(None)  # type: ignore[arg-type]

    def test_get_creates_default(self):
        d = get_security_dispatcher()
        assert isinstance(d, SecurityDispatcher)

    def test_get_returns_same(self):
        d1 = get_security_dispatcher()
        d2 = get_security_dispatcher()
        assert d1 is d2

    def test_set_replaces_instance(self):
        custom = SecurityDispatcher(config=DispatcherConfig(max_concurrent_debates=99))
        set_security_dispatcher(custom)
        assert get_security_dispatcher() is custom

    @pytest.mark.asyncio
    async def test_start_convenience(self):
        d = await start_security_dispatcher()
        assert d.is_running
        await d.stop()

    @pytest.mark.asyncio
    async def test_start_with_config(self):
        cfg = DispatcherConfig(max_concurrent_debates=2)
        d = await start_security_dispatcher(config=cfg)
        assert d.config.max_concurrent_debates == 2
        assert d.is_running
        await d.stop()

    @pytest.mark.asyncio
    async def test_stop_convenience(self):
        d = await start_security_dispatcher()
        assert d.is_running
        await stop_security_dispatcher()
        assert not d.is_running

    @pytest.mark.asyncio
    async def test_stop_when_none(self):
        # Should not raise
        await stop_security_dispatcher()
