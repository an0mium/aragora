"""Tests for events/security_dispatcher.py — security event dispatcher."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
    register_security_debate_runner,
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

    def test_existing_debate_correlation_does_not_trigger_again(self):
        dispatcher = SecurityDispatcher()
        event = _make_event(
            event_type=SecurityEventType.CRITICAL_CVE,
            severity=SecuritySeverity.CRITICAL,
        )
        event.debate_requested = True
        event.debate_id = "debate-existing"

        assert dispatcher._should_trigger_debate(event) is False

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
# _run_debate default path (no custom callback)
# ---------------------------------------------------------------------------


class TestDefaultRunnerPath:
    """Regression tests for the default dispatch path (no custom callback).

    Before P4a E7a this path imported aragora.debate.orchestrator.Arena
    directly. It now routes through the same get_security_debate_runner
    registry hook used by SecurityEventEmitter, so aragora.events never
    imports aragora.debate.
    """

    @pytest.mark.asyncio
    async def test_default_path_uses_registered_runner(self):
        runner = AsyncMock(return_value="debate-default-1")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        runner.assert_awaited_once_with(
            event,
            confidence_threshold=dispatcher.config.debate_confidence_threshold,
            timeout_seconds=dispatcher.config.debate_timeout_seconds,
        )
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_supports_legacy_one_arg_registered_runner(self):
        """Registered runner callbacks that predate dispatcher kwargs remain valid."""
        calls: list[SecurityEvent] = []

        async def legacy_runner(event: SecurityEvent) -> str:
            calls.append(event)
            return "debate-legacy-1"

        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=legacy_runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        assert calls == [event]
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        assert event.debate_requested is True
        assert event.debate_id == "debate-legacy-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_keeps_kwargs_for_modern_registered_runner(self):
        calls: list[tuple[SecurityEvent, float, int]] = []

        async def modern_runner(
            event: SecurityEvent,
            *,
            confidence_threshold: float,
            timeout_seconds: int,
        ) -> str:
            calls.append((event, confidence_threshold, timeout_seconds))
            return "debate-modern-1"

        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=modern_runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        assert calls == [
            (
                event,
                dispatcher.config.debate_confidence_threshold,
                dispatcher.config.debate_timeout_seconds,
            )
        ]
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        assert event.debate_id == "debate-modern-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_passes_event_positionally_to_modern_runner(self):
        calls: list[tuple[SecurityEvent, float, int]] = []

        async def modern_runner(
            security_event: SecurityEvent,
            *,
            confidence_threshold: float,
            timeout_seconds: int,
        ) -> str:
            calls.append((security_event, confidence_threshold, timeout_seconds))
            return "debate-modern-positional-1"

        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=modern_runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        assert calls == [
            (
                event,
                dispatcher.config.debate_confidence_threshold,
                dispatcher.config.debate_timeout_seconds,
            )
        ]
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        assert event.debate_id == "debate-modern-positional-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_sets_event_debate_correlation(self):
        """The dispatcher must stamp debate_requested/debate_id on success.

        Mirrors SecurityEventEmitter._trigger_security_debate's existing
        caller-side mutation, and restores what the pre-E7a Arena delegate
        used to do internally inside security_debate.run_security_debate.
        """
        runner = AsyncMock(return_value="debate-correlated-1")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        assert event.debate_requested is False
        assert event.debate_id is None

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        assert event.debate_requested is True
        assert event.debate_id == "debate-correlated-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_none_result_does_not_inflate_completed(self):
        """A runner that declines (returns None) without raising must not
        be counted as completed, and must not stamp debate correlation."""
        runner = AsyncMock(return_value=None)
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=runner,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        runner.assert_awaited_once()
        assert dispatcher._stats.debates_completed == 0
        assert dispatcher._stats.debates_failed == 0
        assert event.debate_requested is False
        assert event.debate_id is None
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_without_registered_runner_fails_gracefully(self):
        """True cold path: no composition root has registered a runner.

        There is no lazy-import fallback inside aragora.events (P4a
        security-debate-unification removed it - see
        register_security_debate_runner's docstring); a truly cold registry
        must fail soft (counted as failed, no exception escapes) rather than
        importing aragora.debate itself.
        """
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with (
            patch(
                "aragora.events.security_dispatcher.get_security_debate_runner",
                return_value=None,
            ),
            patch.object(
                logging.getLogger("aragora.events.security_dispatcher"), "exception"
            ) as mock_log,
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        assert dispatcher._stats.debates_failed == 1
        assert dispatcher._stats.debates_completed == 0
        assert event.debate_requested is False
        assert event.debate_id is None
        mock_log.assert_called_once()
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_cancelled_default_path_clears_unresolved_debate_request(self):
        runner = AsyncMock(side_effect=asyncio.CancelledError)
        dispatcher = SecurityDispatcher()
        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )
        event.debate_requested = True
        event.debate_id = "cancelled-before-completion"

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=runner,
        ):
            result = await dispatcher._run_debate(event)

        assert result is None
        assert event.debate_requested is False
        assert event.debate_id is None

    @pytest.mark.asyncio
    async def test_default_path_registers_via_composition_root_not_lazy_import(self):
        """A composition root (e.g. importing aragora.debate.security_response,
        as aragora.debate.orchestrator and aragora.debate.event_subscribers'
        bootstrap_debate_event_subscribers do) registers the default runner
        BEFORE the dispatcher ever runs -- there is no events-side lazy-import
        fallback to fall back on if that has not happened (see
        test_default_path_without_registered_runner_fails_gracefully)."""
        import aragora.debate.security_response as security_response_mod

        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with (
            patch(
                "aragora.events.security_dispatcher.get_security_debate_runner",
                return_value=security_response_mod.trigger_security_debate,
            ),
            patch(
                "aragora.debate.security_debate.run_security_debate",
                new_callable=AsyncMock,
            ) as mock_run,
        ):
            mock_run.return_value.debate_id = "debate-composition-root-1"
            mock_run.return_value.consensus_reached = True
            mock_run.return_value.confidence = 0.9
            mock_run.return_value.metadata = {"security_confidence_threshold_met": True}

            with patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ):
                await dispatcher._handle_event(event)
                await asyncio.sleep(0.05)

        mock_run.assert_awaited_once()
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        assert event.debate_requested is True
        assert event.debate_id == "debate-composition-root-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_does_not_require_arena_import(self):
        """The default path must not import aragora.debate.orchestrator directly."""
        runner = AsyncMock(return_value="debate-default-2")
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
        )

        with patch(
            "aragora.events.security_dispatcher.get_security_debate_runner",
            return_value=runner,
        ):
            # Simulate Arena being completely unimportable: if the default
            # path still imported it directly, this would raise ImportError
            # and the debate would fail instead of completing.
            with patch.dict("sys.modules", {"aragora.debate.orchestrator": None}):
                await dispatcher._handle_event(event)
                await asyncio.sleep(0.05)

        runner.assert_awaited_once()
        assert dispatcher._stats.debates_completed == 1
        assert dispatcher._stats.debates_failed == 0
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_path_uses_debate_defaults_and_rich_context(self):
        """Regression for p4a-security-debate-unification.

        The dispatcher's default path resolves to the registered
        aragora.debate.security_response.trigger_security_debate runner,
        which forwards to aragora.debate.security_debate.run_security_debate
        unmodified. That function builds its DebateProtocol from
        DEBATE_DEFAULTS.security_debate_rounds/consensus (not a hardcoded
        rounds=3/consensus="majority") and includes security_event_type,
        source, and severity in Environment.context alongside the existing
        security_event_id/repository/scan_id/findings. This proves the
        dispatcher default path gets the exact same DEBATE_DEFAULTS-driven
        config and rich context as the emitter's auto-debate path, since
        both share the one registered runner.
        """
        from aragora.debate.config.defaults import DEBATE_DEFAULTS
        from aragora.debate.security_response import trigger_security_debate

        mock_agent = MagicMock()
        mock_agent.name = "security-auditor"

        mock_result = MagicMock()
        mock_result.debate_id = "debate-rich-context-1"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.metadata = {}

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        emitter = SecurityEventEmitter(enable_auto_debate=False)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        event = _make_event(
            severity=SecuritySeverity.CRITICAL,
            event_type=SecurityEventType.CRITICAL_CVE,
            findings=[_make_critical_finding()],
        )

        with (
            patch(
                "aragora.events.security_dispatcher.get_security_debate_runner",
                return_value=trigger_security_debate,
            ),
            patch(
                "aragora.debate.security_debate.get_security_debate_agents",
                new_callable=AsyncMock,
                return_value=[mock_agent],
            ),
            patch(
                "aragora.debate.orchestrator.Arena",
                return_value=mock_arena,
            ) as mock_arena_cls,
            patch(
                "aragora.debate.security_response._store_security_debate_result",
                new_callable=AsyncMock,
            ),
        ):
            await dispatcher._handle_event(event)
            await asyncio.sleep(0.05)

        mock_arena_cls.assert_called_once()
        protocol = mock_arena_cls.call_args.kwargs["protocol"]
        assert protocol.rounds == DEBATE_DEFAULTS.security_debate_rounds == 3
        assert protocol.consensus == DEBATE_DEFAULTS.security_debate_consensus == "majority"

        environment = mock_arena_cls.call_args.kwargs["environment"]
        context = json.loads(environment.context)
        assert context["security_event_type"] == event.event_type.value
        assert context["source"] == event.source
        assert context["severity"] == event.severity.value

        assert dispatcher._stats.debates_completed == 1
        assert event.debate_id == "debate-rich-context-1"
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_default_emitter_and_dispatcher_do_not_double_trigger(self):
        """The default emitter auto-debate path should not race the dispatcher."""
        import aragora.events.security_events as security_events

        original_runner = security_events.get_security_debate_runner()
        runner = AsyncMock(return_value="debate-single-1")
        register_security_debate_runner(runner)

        emitter = SecurityEventEmitter(enable_auto_debate=True)
        dispatcher = SecurityDispatcher(emitter=emitter)
        await dispatcher.start()

        try:
            event = _make_event(
                severity=SecuritySeverity.CRITICAL,
                event_type=SecurityEventType.CRITICAL_CVE,
            )
            await emitter.emit(event)
            await asyncio.sleep(0.05)

            runner.assert_awaited_once()
            assert event.debate_id == "debate-single-1"
        finally:
            await dispatcher.stop()
            register_security_debate_runner(original_runner)


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
