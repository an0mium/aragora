"""Tests for the extended hooks system.

Covers HookPriority, HookType, RegisteredHook, HookManager (register,
unregister, trigger, trigger_sync, enable/disable, one-time hooks,
priority ordering, error handling), create_hook_manager, create_logging_hooks,
create_checkpoint_hooks, create_finding_hooks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from aragora.debate.hooks.manager import (
    HookManager,
    HookPriority,
    HookType,
    RegisteredHook,
    create_checkpoint_hooks,
    create_finding_hooks,
    create_hook_manager,
    create_logging_hooks,
)


# ---------------------------------------------------------------------------
# HookPriority
# ---------------------------------------------------------------------------


class TestHookPriority:
    def test_ordering(self):
        assert HookPriority.CRITICAL < HookPriority.HIGH
        assert HookPriority.HIGH < HookPriority.NORMAL
        assert HookPriority.NORMAL < HookPriority.LOW
        assert HookPriority.LOW < HookPriority.CLEANUP


# ---------------------------------------------------------------------------
# HookType
# ---------------------------------------------------------------------------


class TestHookType:
    def test_debate_lifecycle_hooks(self):
        assert HookType.PRE_DEBATE.value == "pre_debate"
        assert HookType.POST_DEBATE.value == "post_debate"
        assert HookType.PRE_ROUND.value == "pre_round"
        assert HookType.POST_ROUND.value == "post_round"

    def test_audit_hooks(self):
        assert HookType.ON_FINDING.value == "on_finding"
        assert HookType.ON_CONTRADICTION.value == "on_contradiction"

    def test_propulsion_hooks(self):
        assert HookType.ON_READY.value == "on_ready"
        assert HookType.ON_PROPEL.value == "on_propel"
        assert HookType.ON_ESCALATE.value == "on_escalate"
        assert HookType.ON_MOLECULE_COMPLETE.value == "on_molecule_complete"


# ---------------------------------------------------------------------------
# RegisteredHook
# ---------------------------------------------------------------------------


class TestRegisteredHook:
    def test_defaults(self):
        cb = lambda: None
        h = RegisteredHook(callback=cb, priority=HookPriority.NORMAL, name="test")
        assert h.once is False
        assert h.name == "test"


# ---------------------------------------------------------------------------
# HookManager — register/unregister
# ---------------------------------------------------------------------------


class TestRegisterUnregister:
    def test_register_returns_unregister_fn(self):
        mgr = HookManager()
        unregister = mgr.register("test_hook", lambda: None)
        assert callable(unregister)

    def test_register_with_hook_type_enum(self):
        mgr = HookManager()
        mgr.register(HookType.PRE_DEBATE, lambda: None, name="pre")
        assert mgr.has_hooks(HookType.PRE_DEBATE)

    def test_register_with_string(self):
        mgr = HookManager()
        mgr.register("custom_hook", lambda: None)
        assert mgr.has_hooks("custom_hook")

    def test_unregister_by_name(self):
        mgr = HookManager()
        mgr.register("test_hook", lambda: None, name="my_hook")
        assert mgr.unregister("test_hook", "my_hook") is True
        assert not mgr.has_hooks("test_hook")

    def test_unregister_nonexistent(self):
        mgr = HookManager()
        assert mgr.unregister("test_hook", "does_not_exist") is False

    def test_unregister_via_returned_fn(self):
        mgr = HookManager()
        unregister = mgr.register("test_hook", lambda: None)
        unregister()
        assert not mgr.has_hooks("test_hook")

    def test_clear_specific_type(self):
        mgr = HookManager()
        mgr.register("a", lambda: None)
        mgr.register("b", lambda: None)
        mgr.clear("a")
        assert not mgr.has_hooks("a")
        assert mgr.has_hooks("b")

    def test_clear_all(self):
        mgr = HookManager()
        mgr.register("a", lambda: None)
        mgr.register("b", lambda: None)
        mgr.clear()
        assert not mgr.has_hooks("a")
        assert not mgr.has_hooks("b")

    def test_get_hooks_names(self):
        mgr = HookManager()
        mgr.register("test", lambda: None, name="hook1")
        mgr.register("test", lambda: None, name="hook2")
        names = mgr.get_hooks("test")
        assert "hook1" in names
        assert "hook2" in names


# ---------------------------------------------------------------------------
# HookManager — trigger (async)
# ---------------------------------------------------------------------------


class TestTriggerAsync:
    @pytest.mark.asyncio
    async def test_trigger_sync_callback(self):
        results = []
        mgr = HookManager()
        mgr.register("test", lambda value=None: results.append(value))
        await mgr.trigger("test", value=42)
        assert results == [42]

    @pytest.mark.asyncio
    async def test_trigger_async_callback(self):
        results = []
        mgr = HookManager()

        async def async_hook(value=None):
            results.append(value)

        mgr.register("test", async_hook)
        await mgr.trigger("test", value="hello")
        assert results == ["hello"]

    @pytest.mark.asyncio
    async def test_trigger_returns_results(self):
        mgr = HookManager()
        mgr.register("test", lambda: "result1")
        mgr.register("test", lambda: "result2")
        results = await mgr.trigger("test")
        assert results == ["result1", "result2"]

    @pytest.mark.asyncio
    async def test_trigger_no_hooks_returns_empty(self):
        mgr = HookManager()
        results = await mgr.trigger("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_trigger_disabled_returns_empty(self):
        mgr = HookManager()
        mgr.register("test", lambda: "result")
        mgr.disable()
        results = await mgr.trigger("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_trigger_error_isolation(self):
        mgr = HookManager()
        mgr.register("test", lambda: (_ for _ in ()).throw(ValueError("fail")), name="bad")
        mgr.register("test", lambda: "ok", name="good")
        results = await mgr.trigger("test")
        # Bad hook returns None, good hook returns "ok"
        assert len(results) == 2
        assert "ok" in results

    @pytest.mark.asyncio
    async def test_trigger_with_hook_type_enum(self):
        mgr = HookManager()
        called = []
        mgr.register(HookType.ON_FINDING, lambda: called.append(True))
        await mgr.trigger(HookType.ON_FINDING)
        assert called == [True]


# ---------------------------------------------------------------------------
# HookManager — priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    @pytest.mark.asyncio
    async def test_hooks_execute_in_priority_order(self):
        order = []
        mgr = HookManager()
        mgr.register("test", lambda: order.append("low"), priority=HookPriority.LOW)
        mgr.register("test", lambda: order.append("critical"), priority=HookPriority.CRITICAL)
        mgr.register("test", lambda: order.append("normal"), priority=HookPriority.NORMAL)
        await mgr.trigger("test")
        assert order == ["critical", "normal", "low"]


# ---------------------------------------------------------------------------
# HookManager — one-time hooks
# ---------------------------------------------------------------------------


class TestOnceHooks:
    @pytest.mark.asyncio
    async def test_once_hook_removed_after_trigger(self):
        mgr = HookManager()
        mgr.register("test", lambda: "once", once=True)
        results1 = await mgr.trigger("test")
        results2 = await mgr.trigger("test")
        assert results1 == ["once"]
        assert results2 == []

    def test_once_hook_removed_after_sync_trigger(self):
        mgr = HookManager()
        mgr.register("test", lambda: "once", once=True)
        results1 = mgr.trigger_sync("test")
        results2 = mgr.trigger_sync("test")
        assert results1 == ["once"]
        assert results2 == []


# ---------------------------------------------------------------------------
# HookManager — trigger_sync
# ---------------------------------------------------------------------------


class TestTriggerSync:
    def test_sync_trigger(self):
        mgr = HookManager()
        mgr.register("test", lambda: "result")
        results = mgr.trigger_sync("test")
        assert results == ["result"]

    def test_sync_trigger_disabled(self):
        mgr = HookManager()
        mgr.register("test", lambda: "result")
        mgr.disable()
        results = mgr.trigger_sync("test")
        assert results == []

    def test_sync_trigger_skips_async(self):
        mgr = HookManager()

        async def async_fn():
            return "async"

        mgr.register("test", async_fn)
        results = mgr.trigger_sync("test")
        # Coroutine should be skipped, returning None
        assert results == [None]

    def test_sync_trigger_error_isolation(self):
        mgr = HookManager()
        mgr.register("test", lambda: 1 / 0, name="bad")
        mgr.register("test", lambda: "ok", name="good")
        results = mgr.trigger_sync("test")
        assert len(results) == 2
        assert results[1] == "ok"


# ---------------------------------------------------------------------------
# HookManager — enable/disable
# ---------------------------------------------------------------------------


class TestEnableDisable:
    def test_enable_after_disable(self):
        mgr = HookManager()
        mgr.register("test", lambda: "result")
        mgr.disable()
        mgr.enable()
        results = mgr.trigger_sync("test")
        assert results == ["result"]

    def test_stats(self):
        mgr = HookManager()
        mgr.register("a", lambda: None)
        mgr.register("a", lambda: None)
        mgr.register("b", lambda: None)
        stats = mgr.stats
        assert stats["a"] == 2
        assert stats["b"] == 1


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------


class TestErrorHandler:
    @pytest.mark.asyncio
    async def test_custom_error_handler_called(self):
        errors = []
        mgr = HookManager()
        mgr.set_error_handler(lambda name, exc: errors.append((name, str(exc))))
        mgr.register("test", lambda: 1 / 0, name="divzero")
        await mgr.trigger("test")
        assert len(errors) == 1
        assert errors[0][0] == "divzero"

    @pytest.mark.asyncio
    async def test_error_handler_failure_doesnt_cascade(self):
        def bad_handler(name, exc):
            raise RuntimeError("handler also fails")

        mgr = HookManager()
        mgr.set_error_handler(bad_handler)
        mgr.register("test", lambda: 1 / 0, name="bad")
        mgr.register("test", lambda: "ok", name="good")
        # Should not raise
        results = await mgr.trigger("test")
        assert "ok" in results


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_create_hook_manager(self):
        mgr = create_hook_manager()
        assert isinstance(mgr, HookManager)

    def test_create_hook_manager_with_error_handler(self):
        handler = MagicMock()
        mgr = create_hook_manager(error_handler=handler)
        assert mgr._error_handler is handler

    def test_create_logging_hooks(self):
        mgr = HookManager()
        create_logging_hooks(mgr)
        # Should register hooks for all HookType values
        for hook_type in HookType:
            assert mgr.has_hooks(hook_type)

    def test_create_checkpoint_hooks(self):
        mgr = HookManager()
        checkpoint_fn = MagicMock()
        create_checkpoint_hooks(mgr, checkpoint_fn)
        assert mgr.has_hooks(HookType.POST_ROUND)

    def test_create_finding_hooks(self):
        mgr = HookManager()
        on_finding = MagicMock()
        create_finding_hooks(mgr, on_finding, severity_threshold=5.0)
        assert mgr.has_hooks(HookType.ON_FINDING)
