"""Tests for context_strategies base classes."""

from __future__ import annotations

import asyncio

import pytest

from aragora.debate.context_strategies.base import CachingStrategy, ContextStrategy


# ---------------------------------------------------------------------------
# Concrete implementations for testing abstract base
# ---------------------------------------------------------------------------


class _SimpleStrategy(ContextStrategy):
    """Minimal concrete strategy for testing."""

    name = "simple"
    default_timeout = 2.0

    def __init__(self, result: str | None = "ok") -> None:
        self._result = result

    async def gather(self, task: str, **kwargs) -> str | None:  # type: ignore[override]
        return self._result


class _SlowStrategy(ContextStrategy):
    """Strategy that sleeps longer than timeout."""

    name = "slow"
    default_timeout = 0.1

    async def gather(self, task: str, **kwargs) -> str | None:  # type: ignore[override]
        await asyncio.sleep(5)
        return "should not reach"


class _ErrorStrategy(ContextStrategy):
    """Strategy that raises during gather."""

    name = "error"

    async def gather(self, task: str, **kwargs) -> str | None:  # type: ignore[override]
        raise RuntimeError("boom")


class _SimpleCaching(CachingStrategy):
    """Minimal caching strategy for testing."""

    name = "caching_test"
    max_cache_size = 3

    async def gather(self, task: str, **kwargs) -> str | None:  # type: ignore[override]
        cached = self.get_cached(task, **kwargs)
        if cached:
            return cached
        result = f"result-for-{task}"
        self.set_cached(task, result, **kwargs)
        return result


# ---------------------------------------------------------------------------
# ContextStrategy tests
# ---------------------------------------------------------------------------


class TestContextStrategy:
    """Tests for abstract ContextStrategy."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ContextStrategy()  # type: ignore[abstract]

    def test_concrete_strategy_name(self) -> None:
        s = _SimpleStrategy()
        assert s.name == "simple"

    def test_concrete_strategy_default_timeout(self) -> None:
        s = _SimpleStrategy()
        assert s.default_timeout == 2.0

    def test_is_available_default_true(self) -> None:
        s = _SimpleStrategy()
        assert s.is_available() is True

    @pytest.mark.asyncio
    async def test_gather_returns_result(self) -> None:
        s = _SimpleStrategy(result="hello")
        assert await s.gather("task") == "hello"

    @pytest.mark.asyncio
    async def test_gather_returns_none(self) -> None:
        s = _SimpleStrategy(result=None)
        assert await s.gather("task") is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_success(self) -> None:
        s = _SimpleStrategy(result="ok")
        result = await s.gather_with_timeout("task", timeout=5.0)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_gather_with_timeout_uses_default(self) -> None:
        s = _SimpleStrategy(result="ok")
        # default_timeout is 2.0, should succeed
        result = await s.gather_with_timeout("task")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_gather_with_timeout_times_out(self) -> None:
        s = _SlowStrategy()
        result = await s.gather_with_timeout("task", timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_default_timeout_fires(self) -> None:
        s = _SlowStrategy()
        # default_timeout is 0.1s, sleep is 5s
        result = await s.gather_with_timeout("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_catches_exceptions(self) -> None:
        s = _ErrorStrategy()
        result = await s.gather_with_timeout("task", timeout=5.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_passes_kwargs(self) -> None:
        class _KwargsStrategy(ContextStrategy):
            name = "kwargs"

            async def gather(self, task: str, **kwargs) -> str | None:  # type: ignore[override]
                return f"{task}:{kwargs.get('mode', 'default')}"

        s = _KwargsStrategy()
        result = await s.gather_with_timeout("t", timeout=5.0, mode="fast")
        assert result == "t:fast"


# ---------------------------------------------------------------------------
# CachingStrategy tests
# ---------------------------------------------------------------------------


class TestCachingStrategy:
    """Tests for CachingStrategy caching layer."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            CachingStrategy()  # type: ignore[abstract]

    def test_cache_key_deterministic(self) -> None:
        s = _SimpleCaching()
        k1 = s._get_cache_key("hello")
        k2 = s._get_cache_key("hello")
        assert k1 == k2

    def test_cache_key_different_for_different_tasks(self) -> None:
        s = _SimpleCaching()
        k1 = s._get_cache_key("task-a")
        k2 = s._get_cache_key("task-b")
        assert k1 != k2

    def test_cache_key_different_for_different_kwargs(self) -> None:
        s = _SimpleCaching()
        k1 = s._get_cache_key("task", mode="a")
        k2 = s._get_cache_key("task", mode="b")
        assert k1 != k2

    def test_cache_key_length(self) -> None:
        s = _SimpleCaching()
        key = s._get_cache_key("some task")
        assert len(key) == 16  # sha256 hex[:16]

    def test_get_cached_miss(self) -> None:
        s = _SimpleCaching()
        assert s.get_cached("nonexistent") is None

    def test_set_and_get_cached(self) -> None:
        s = _SimpleCaching()
        s.set_cached("task-1", "value-1")
        assert s.get_cached("task-1") == "value-1"

    def test_cache_eviction_fifo(self) -> None:
        s = _SimpleCaching()
        # max_cache_size = 3
        s.set_cached("t1", "v1")
        s.set_cached("t2", "v2")
        s.set_cached("t3", "v3")
        # Cache is full; adding one more evicts oldest (t1)
        s.set_cached("t4", "v4")
        assert s.get_cached("t1") is None
        assert s.get_cached("t2") == "v2"
        assert s.get_cached("t4") == "v4"

    def test_cache_eviction_preserves_newest(self) -> None:
        s = _SimpleCaching()
        for i in range(10):
            s.set_cached(f"task-{i}", f"val-{i}")
        # Only last 3 should remain
        assert len(s._cache) == 3
        assert s.get_cached("task-9") == "val-9"
        assert s.get_cached("task-8") == "val-8"
        assert s.get_cached("task-7") == "val-7"
        assert s.get_cached("task-6") is None

    def test_clear_cache_all(self) -> None:
        s = _SimpleCaching()
        s.set_cached("t1", "v1")
        s.set_cached("t2", "v2")
        s.clear_cache()
        assert s.get_cached("t1") is None
        assert s.get_cached("t2") is None
        assert len(s._cache) == 0

    def test_clear_cache_specific_task(self) -> None:
        s = _SimpleCaching()
        s.set_cached("t1", "v1")
        s.set_cached("t2", "v2")
        s.clear_cache("t1")
        assert s.get_cached("t1") is None
        assert s.get_cached("t2") == "v2"

    def test_clear_cache_nonexistent_task_no_error(self) -> None:
        s = _SimpleCaching()
        s.clear_cache("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_gather_uses_cache(self) -> None:
        s = _SimpleCaching()
        r1 = await s.gather("my-task")
        assert r1 == "result-for-my-task"
        # Second call should return cached
        r2 = await s.gather("my-task")
        assert r2 == "result-for-my-task"

    def test_max_cache_size_class_attribute(self) -> None:
        assert _SimpleCaching.max_cache_size == 3
