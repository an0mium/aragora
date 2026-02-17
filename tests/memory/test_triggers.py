"""Tests for MemoryTriggerEngine -- reactive rules on memory events."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.memory.triggers import (
    MemoryTrigger,
    MemoryTriggerEngine,
    TriggerResult,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture()
def engine() -> MemoryTriggerEngine:
    return MemoryTriggerEngine()


@pytest.fixture()
def bare_engine() -> MemoryTriggerEngine:
    """Engine with builtins removed for isolated testing."""
    eng = MemoryTriggerEngine()
    for name in list(eng._triggers.keys()):
        eng.unregister(name)
    return eng


# -----------------------------------------------------------------------
# Register/fire/enable/disable lifecycle: 8 tests
# -----------------------------------------------------------------------


class TestLifecycle:
    def test_register_trigger(self, bare_engine: MemoryTriggerEngine) -> None:
        trigger = MemoryTrigger(name="test", event="test_event")
        bare_engine.register(trigger)
        assert bare_engine.get_trigger("test") is not None

    def test_register_overwrites(self, bare_engine: MemoryTriggerEngine) -> None:
        t1 = MemoryTrigger(name="test", event="event_a")
        t2 = MemoryTrigger(name="test", event="event_b")
        bare_engine.register(t1)
        bare_engine.register(t2)
        assert bare_engine.get_trigger("test").event == "event_b"

    def test_unregister_existing(self, bare_engine: MemoryTriggerEngine) -> None:
        bare_engine.register(MemoryTrigger(name="test", event="e"))
        assert bare_engine.unregister("test") is True
        assert bare_engine.get_trigger("test") is None

    def test_unregister_nonexistent(self, bare_engine: MemoryTriggerEngine) -> None:
        assert bare_engine.unregister("nonexistent") is False

    def test_list_triggers(self, engine: MemoryTriggerEngine) -> None:
        triggers = engine.list_triggers()
        assert len(triggers) >= 5  # at least 5 builtins

    def test_enable_trigger(self, bare_engine: MemoryTriggerEngine) -> None:
        trigger = MemoryTrigger(name="test", event="e", enabled=False)
        bare_engine.register(trigger)
        bare_engine.enable("test")
        assert bare_engine.get_trigger("test").enabled is True

    def test_disable_trigger(self, bare_engine: MemoryTriggerEngine) -> None:
        trigger = MemoryTrigger(name="test", event="e", enabled=True)
        bare_engine.register(trigger)
        bare_engine.disable("test")
        assert bare_engine.get_trigger("test").enabled is False

    @pytest.mark.asyncio
    async def test_disabled_trigger_does_not_fire(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        action = AsyncMock()
        trigger = MemoryTrigger(name="test", event="e", action=action, enabled=False)
        bare_engine.register(trigger)
        result = await bare_engine.fire("e", {})
        assert result == []
        action.assert_not_called()


# -----------------------------------------------------------------------
# Each built-in trigger fires on correct conditions: 5 tests
# -----------------------------------------------------------------------


class TestBuiltinTriggers:
    @pytest.mark.asyncio
    async def test_high_surprise_investigate(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "high_surprise", {"item_id": "x", "surprise": 0.85}
        )
        assert "high_surprise_investigate" in triggered

    @pytest.mark.asyncio
    async def test_stale_knowledge_revalidate(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "stale_knowledge",
            {"item_id": "x", "days_since_access": 10, "confidence": 0.3},
        )
        assert "stale_knowledge_revalidate" in triggered

    @pytest.mark.asyncio
    async def test_contradiction_detected(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "contradiction", {"description": "A contradicts B"}
        )
        assert "contradiction_detected" in triggered

    @pytest.mark.asyncio
    async def test_consolidation_merge(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "consolidation", {"item_count": 5, "avg_surprise": 0.1}
        )
        assert "consolidation_merge" in triggered

    @pytest.mark.asyncio
    async def test_pattern_emergence(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "new_pattern", {"surprise_ema_trend": "decreasing", "pattern": "repeated fix"}
        )
        assert "pattern_emergence" in triggered


# -----------------------------------------------------------------------
# Condition filtering works: 5 tests
# -----------------------------------------------------------------------


class TestConditionFiltering:
    @pytest.mark.asyncio
    async def test_condition_blocks_trigger(self, engine: MemoryTriggerEngine) -> None:
        # high_surprise trigger requires surprise > 0.7
        triggered = await engine.fire(
            "high_surprise", {"item_id": "x", "surprise": 0.3}
        )
        assert "high_surprise_investigate" not in triggered

    @pytest.mark.asyncio
    async def test_stale_knowledge_needs_both_conditions(
        self, engine: MemoryTriggerEngine
    ) -> None:
        # Only days_since_access > 7 but confidence is high
        triggered = await engine.fire(
            "stale_knowledge",
            {"item_id": "x", "days_since_access": 10, "confidence": 0.9},
        )
        assert "stale_knowledge_revalidate" not in triggered

    @pytest.mark.asyncio
    async def test_consolidation_needs_min_items(
        self, engine: MemoryTriggerEngine
    ) -> None:
        triggered = await engine.fire(
            "consolidation", {"item_count": 1, "avg_surprise": 0.1}
        )
        assert "consolidation_merge" not in triggered

    @pytest.mark.asyncio
    async def test_pattern_wrong_trend(self, engine: MemoryTriggerEngine) -> None:
        triggered = await engine.fire(
            "new_pattern", {"surprise_ema_trend": "increasing"}
        )
        assert "pattern_emergence" not in triggered

    @pytest.mark.asyncio
    async def test_custom_condition(self, bare_engine: MemoryTriggerEngine) -> None:
        action = AsyncMock()
        trigger = MemoryTrigger(
            name="custom",
            event="test_event",
            condition=lambda ctx: ctx.get("value", 0) > 10,
            action=action,
        )
        bare_engine.register(trigger)

        await bare_engine.fire("test_event", {"value": 5})
        action.assert_not_called()

        await bare_engine.fire("test_event", {"value": 15})
        action.assert_called_once()


# -----------------------------------------------------------------------
# Multiple triggers on same event: 4 tests
# -----------------------------------------------------------------------


class TestMultipleTriggers:
    @pytest.mark.asyncio
    async def test_two_triggers_same_event(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        action1 = AsyncMock()
        action2 = AsyncMock()
        bare_engine.register(MemoryTrigger(name="t1", event="e", action=action1))
        bare_engine.register(MemoryTrigger(name="t2", event="e", action=action2))

        triggered = await bare_engine.fire("e", {})
        assert len(triggered) == 2
        action1.assert_called_once()
        action2.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_events_dont_cross(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        action1 = AsyncMock()
        action2 = AsyncMock()
        bare_engine.register(MemoryTrigger(name="t1", event="e1", action=action1))
        bare_engine.register(MemoryTrigger(name="t2", event="e2", action=action2))

        triggered = await bare_engine.fire("e1", {})
        assert triggered == ["t1"]
        action1.assert_called_once()
        action2.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_disabled_one_enabled(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        action1 = AsyncMock()
        action2 = AsyncMock()
        bare_engine.register(
            MemoryTrigger(name="t1", event="e", action=action1, enabled=False)
        )
        bare_engine.register(
            MemoryTrigger(name="t2", event="e", action=action2, enabled=True)
        )

        triggered = await bare_engine.fire("e", {})
        assert triggered == ["t2"]

    @pytest.mark.asyncio
    async def test_three_triggers_mixed_conditions(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        actions = [AsyncMock(), AsyncMock(), AsyncMock()]
        bare_engine.register(
            MemoryTrigger(
                name="t1", event="e", condition=lambda c: True, action=actions[0]
            )
        )
        bare_engine.register(
            MemoryTrigger(
                name="t2", event="e", condition=lambda c: False, action=actions[1]
            )
        )
        bare_engine.register(
            MemoryTrigger(
                name="t3", event="e", condition=None, action=actions[2]
            )
        )

        triggered = await bare_engine.fire("e", {})
        assert "t1" in triggered
        assert "t2" not in triggered
        assert "t3" in triggered


# -----------------------------------------------------------------------
# Error in one trigger doesn't block others: 4 tests
# -----------------------------------------------------------------------


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_action_error_doesnt_block_next(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        working = AsyncMock()
        bare_engine.register(MemoryTrigger(name="fail", event="e", action=failing))
        bare_engine.register(MemoryTrigger(name="ok", event="e", action=working))

        triggered = await bare_engine.fire("e", {})
        assert "fail" in triggered
        assert "ok" in triggered
        working.assert_called_once()

    @pytest.mark.asyncio
    async def test_condition_error_doesnt_block_next(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        def bad_condition(ctx: dict) -> bool:
            raise ValueError("bad condition")

        working = AsyncMock()
        bare_engine.register(
            MemoryTrigger(
                name="bad", event="e", condition=bad_condition, action=AsyncMock()
            )
        )
        bare_engine.register(MemoryTrigger(name="ok", event="e", action=working))

        triggered = await bare_engine.fire("e", {})
        assert "ok" in triggered
        working.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_error_logged_in_fire_log(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        failing = AsyncMock(side_effect=ValueError("oops"))
        bare_engine.register(MemoryTrigger(name="fail", event="e", action=failing))

        await bare_engine.fire("e", {})
        log = bare_engine.get_fire_log()
        assert len(log) == 1
        assert log[0].trigger_name == "fail"
        assert log[0].success is False
        assert "oops" in log[0].error

    @pytest.mark.asyncio
    async def test_condition_error_logged_in_fire_log(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        def bad(ctx: dict) -> bool:
            raise TypeError("bad type")

        bare_engine.register(
            MemoryTrigger(name="bad", event="e", condition=bad, action=AsyncMock())
        )

        await bare_engine.fire("e", {})
        log = bare_engine.get_fire_log()
        assert len(log) == 1
        assert log[0].success is False
        assert "bad type" in log[0].error


# -----------------------------------------------------------------------
# Fire log tracking: 4 tests
# -----------------------------------------------------------------------


class TestFireLog:
    @pytest.mark.asyncio
    async def test_fire_log_records_success(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        action = AsyncMock()
        bare_engine.register(MemoryTrigger(name="test", event="e", action=action))
        await bare_engine.fire("e", {})

        log = bare_engine.get_fire_log()
        assert len(log) == 1
        assert log[0].trigger_name == "test"
        assert log[0].success is True
        assert log[0].error is None

    @pytest.mark.asyncio
    async def test_fire_log_accumulates(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        bare_engine.register(MemoryTrigger(name="test", event="e", action=AsyncMock()))
        await bare_engine.fire("e", {})
        await bare_engine.fire("e", {})

        log = bare_engine.get_fire_log()
        assert len(log) == 2

    @pytest.mark.asyncio
    async def test_clear_fire_log(self, bare_engine: MemoryTriggerEngine) -> None:
        bare_engine.register(MemoryTrigger(name="test", event="e", action=AsyncMock()))
        await bare_engine.fire("e", {})
        bare_engine.clear_fire_log()
        assert bare_engine.get_fire_log() == []

    @pytest.mark.asyncio
    async def test_fire_log_no_action_trigger(
        self, bare_engine: MemoryTriggerEngine
    ) -> None:
        bare_engine.register(MemoryTrigger(name="test", event="e", action=None))
        await bare_engine.fire("e", {})

        log = bare_engine.get_fire_log()
        assert len(log) == 1
        assert log[0].success is True
