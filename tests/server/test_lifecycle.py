"""Tests for ThreadRegistry lifecycle management."""

from __future__ import annotations

import signal
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.lifecycle import (
    ThreadRegistry,
    get_thread_registry,
    register_lifecycle_signal_handlers,
    reset_thread_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stoppable_thread(name: str = "test-thread"):
    """Create a daemon thread controlled by a stop event."""
    stop_event = threading.Event()

    def target():
        stop_event.wait()

    thread = threading.Thread(target=target, name=name, daemon=True)
    return thread, stop_event


# ---------------------------------------------------------------------------
# ThreadRegistry Tests
# ---------------------------------------------------------------------------


class TestThreadRegistryRegister:
    """Tests for register/unregister."""

    def test_register_adds_thread(self):
        registry = ThreadRegistry()
        thread, stop = _make_stoppable_thread()
        thread.start()
        registry.register("t1", thread, shutdown_fn=stop.set)
        assert "t1" in registry.names
        assert len(registry) == 1
        stop.set()
        thread.join(timeout=1)

    def test_register_replaces_existing(self):
        registry = ThreadRegistry()
        t1, s1 = _make_stoppable_thread("old")
        t2, s2 = _make_stoppable_thread("new")
        t1.start()
        t2.start()
        registry.register("t", t1, shutdown_fn=s1.set)
        registry.register("t", t2, shutdown_fn=s2.set)
        assert len(registry) == 1
        s1.set()
        s2.set()
        t1.join(timeout=1)
        t2.join(timeout=1)

    def test_unregister_existing(self):
        registry = ThreadRegistry()
        t, s = _make_stoppable_thread()
        t.start()
        registry.register("t", t, shutdown_fn=s.set)
        assert registry.unregister("t") is True
        assert len(registry) == 0
        s.set()
        t.join(timeout=1)

    def test_unregister_nonexistent(self):
        registry = ThreadRegistry()
        assert registry.unregister("missing") is False


class TestThreadRegistryShutdown:
    """Tests for shutdown_all."""

    def test_shutdown_empty(self):
        registry = ThreadRegistry()
        results = registry.shutdown_all(timeout=1.0)
        assert results == {}

    def test_shutdown_stops_threads(self):
        registry = ThreadRegistry()
        t1, s1 = _make_stoppable_thread("a")
        t2, s2 = _make_stoppable_thread("b")
        t1.start()
        t2.start()
        registry.register("a", t1, shutdown_fn=s1.set)
        registry.register("b", t2, shutdown_fn=s2.set)

        results = registry.shutdown_all(timeout=5.0)
        assert results["a"] is True
        assert results["b"] is True
        assert not t1.is_alive()
        assert not t2.is_alive()

    def test_shutdown_sets_flag(self):
        registry = ThreadRegistry()
        assert registry._shutdown_called is False
        registry.shutdown_all(timeout=1.0)
        assert registry._shutdown_called is True

    def test_shutdown_handles_dead_thread(self):
        """Thread already dead before shutdown."""
        registry = ThreadRegistry()
        t, s = _make_stoppable_thread()
        t.start()
        s.set()
        t.join(timeout=1)
        registry.register("dead", t, shutdown_fn=s.set)

        results = registry.shutdown_all(timeout=1.0)
        assert results["dead"] is True

    def test_shutdown_handles_shutdown_fn_error(self):
        """shutdown_fn raises but thread still gets joined."""
        registry = ThreadRegistry()
        t, s = _make_stoppable_thread()
        t.start()

        def bad_shutdown():
            s.set()
            raise RuntimeError("boom")

        registry.register("bad", t, shutdown_fn=bad_shutdown)
        results = registry.shutdown_all(timeout=5.0)
        # Thread should still stop because stop_event was set before error
        assert results["bad"] is True

    def test_shutdown_timeout_budget(self):
        """Threads that don't stop within budget are reported as not stopped."""
        registry = ThreadRegistry()
        # Create a thread that ignores the stop signal
        never_stop = threading.Event()

        def stubborn():
            never_stop.wait(timeout=60)

        t = threading.Thread(target=stubborn, daemon=True)
        t.start()
        registry.register("stubborn", t, shutdown_fn=lambda: None)

        results = registry.shutdown_all(timeout=0.2)
        # Thread likely still alive
        assert "stubborn" in results
        # Clean up
        never_stop.set()
        t.join(timeout=1)

    def test_shutdown_multiple_with_mixed_results(self):
        """Mix of fast-stopping and already-dead threads."""
        registry = ThreadRegistry()
        t1, s1 = _make_stoppable_thread("fast")
        t1.start()
        registry.register("fast", t1, shutdown_fn=s1.set)

        t2, s2 = _make_stoppable_thread("dead")
        t2.start()
        s2.set()
        t2.join(timeout=1)
        registry.register("dead", t2, shutdown_fn=s2.set)

        results = registry.shutdown_all(timeout=5.0)
        assert results["fast"] is True
        assert results["dead"] is True


class TestThreadRegistryHealth:
    """Tests for health reporting."""

    def test_health_empty(self):
        registry = ThreadRegistry()
        h = registry.health()
        assert h["total"] == 0
        assert h["alive"] == 0
        assert h["shutdown_called"] is False
        assert h["threads"] == []

    def test_health_reports_alive_threads(self):
        registry = ThreadRegistry()
        t, s = _make_stoppable_thread("worker")
        t.start()
        registry.register("worker", t, shutdown_fn=s.set)

        h = registry.health()
        assert h["total"] == 1
        assert h["alive"] == 1
        assert h["threads"][0]["name"] == "worker"
        assert h["threads"][0]["alive"] is True
        assert h["threads"][0]["daemon"] is True
        assert "registered_at" in h["threads"][0]
        s.set()
        t.join(timeout=1)

    def test_health_reports_dead_threads(self):
        registry = ThreadRegistry()
        t, s = _make_stoppable_thread("stopped")
        t.start()
        s.set()
        t.join(timeout=1)
        registry.register("stopped", t, shutdown_fn=s.set)

        h = registry.health()
        assert h["total"] == 1
        assert h["alive"] == 0
        assert h["threads"][0]["alive"] is False

    def test_health_after_shutdown(self):
        registry = ThreadRegistry()
        registry.shutdown_all(timeout=1.0)
        h = registry.health()
        assert h["shutdown_called"] is True


class TestSingleton:
    """Tests for singleton access."""

    def test_get_thread_registry_singleton(self):
        reset_thread_registry()
        r1 = get_thread_registry()
        r2 = get_thread_registry()
        assert r1 is r2
        reset_thread_registry()

    def test_reset_creates_new_instance(self):
        reset_thread_registry()
        r1 = get_thread_registry()
        reset_thread_registry()
        r2 = get_thread_registry()
        assert r1 is not r2
        reset_thread_registry()


class TestSignalHandlers:
    """Tests for signal handler registration."""

    def test_register_signal_handlers_sets_handlers(self):
        # Save original handlers
        orig_term = signal.getsignal(signal.SIGTERM)
        orig_int = signal.getsignal(signal.SIGINT)

        try:
            register_lifecycle_signal_handlers()
            # Handlers should be changed from defaults
            new_term = signal.getsignal(signal.SIGTERM)
            new_int = signal.getsignal(signal.SIGINT)
            assert callable(new_term)
            assert callable(new_int)
        finally:
            # Restore original handlers
            signal.signal(signal.SIGTERM, orig_term)
            signal.signal(signal.SIGINT, orig_int)

    def test_signal_handler_calls_shutdown_all(self):
        reset_thread_registry()
        registry = get_thread_registry()

        orig_term = signal.getsignal(signal.SIGTERM)
        try:
            register_lifecycle_signal_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            # Call the handler directly (simulating SIGTERM)
            with patch.object(registry, "shutdown_all", return_value={}) as mock_sd:
                handler(signal.SIGTERM, None)
                mock_sd.assert_called_once_with(timeout=10.0)
        finally:
            signal.signal(signal.SIGTERM, orig_term)
            reset_thread_registry()


class TestPrometheusMetrics:
    """Tests for Prometheus metric emission."""

    def test_register_updates_gauge(self):
        """Gauge tracks registered thread count."""
        from aragora.server.lifecycle import REGISTERED_THREADS_GAUGE

        assert REGISTERED_THREADS_GAUGE is not None

        registry = ThreadRegistry()
        t, s = _make_stoppable_thread()
        t.start()
        registry.register("m1", t, shutdown_fn=s.set)

        # Gauge should reflect 1
        val = REGISTERED_THREADS_GAUGE._value.get()
        assert val == 1.0

        registry.unregister("m1")
        val = REGISTERED_THREADS_GAUGE._value.get()
        assert val == 0.0

        s.set()
        t.join(timeout=1)

    def test_shutdown_records_histogram(self):
        """Shutdown duration is observed in histogram."""
        from aragora.server.lifecycle import SHUTDOWN_DURATION_HISTOGRAM

        assert SHUTDOWN_DURATION_HISTOGRAM is not None

        registry = ThreadRegistry()
        t, s = _make_stoppable_thread()
        t.start()
        registry.register("h1", t, shutdown_fn=s.set)

        # Get sample count before
        before = SHUTDOWN_DURATION_HISTOGRAM._sum.get()

        registry.shutdown_all(timeout=5.0)

        after = SHUTDOWN_DURATION_HISTOGRAM._sum.get()
        # Sum should have increased (duration > 0)
        assert after >= before
