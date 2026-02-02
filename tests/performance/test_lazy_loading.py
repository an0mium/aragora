"""
Comprehensive tests for lazy loading framework.

Tests the lazy initialization utilities in aragora/performance/lazy_loading.py including:
- Lazy initialization behavior
- Thread safety
- Memory efficiency
- Error handling
- Configuration options
- N+1 query detection
- Prefetch operations

Run with: pytest tests/performance/test_lazy_loading.py -v
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.performance.lazy_loading import (
    AUTO_PREFETCH_BATCH_DELAY_MS,
    AutoPrefetchBatcher,
    LazyDescriptor,
    LazyLoader,
    LazyLoadStats,
    LazyValue,
    N_PLUS_ONE_THRESHOLD,
    N_PLUS_ONE_WINDOW_MS,
    get_lazy_load_stats,
    get_n_plus_one_config,
    lazy_property,
    prefetch,
    register_prefetch,
    reset_lazy_load_stats,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_stats():
    """Reset lazy load stats before each test."""
    reset_lazy_load_stats()
    yield
    reset_lazy_load_stats()


@pytest.fixture
def sample_async_factory():
    """Create a sample async factory function."""
    call_count = 0

    async def factory() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.001)  # Simulate async work
        return f"value_{call_count}"

    factory.call_count = lambda: call_count
    return factory


# =============================================================================
# Lazy Initialization Tests
# =============================================================================


class TestLazyValueInitialization:
    """Tests for lazy value initialization behavior."""

    @pytest.mark.asyncio
    async def test_value_not_loaded_until_first_access(self):
        """Object is not created until first access."""
        factory_called = False

        async def factory():
            nonlocal factory_called
            factory_called = True
            return "test_value"

        lazy_val = LazyValue(factory, "test_prop")

        assert not factory_called
        assert not lazy_val.is_loaded

    @pytest.mark.asyncio
    async def test_value_loaded_on_first_access(self):
        """Object is created on first access."""
        factory_called = False

        async def factory():
            nonlocal factory_called
            factory_called = True
            return "test_value"

        lazy_val = LazyValue(factory, "test_prop")
        result = await lazy_val.get()

        assert factory_called
        assert lazy_val.is_loaded
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_subsequent_accesses_return_same_instance(self):
        """Subsequent accesses return same instance."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        lazy_val = LazyValue(factory, "test_prop")

        result1 = await lazy_val.get()
        result2 = await lazy_val.get()
        result3 = await lazy_val.get()

        # All results should be the same object
        assert result1 is result2
        assert result2 is result3
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_factory_function_called_only_once(self):
        """Factory function is called only once."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "value"

        lazy_val = LazyValue(factory, "test_prop")

        # Multiple accesses
        for _ in range(10):
            await lazy_val.get()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_set_value_directly(self):
        """Values can be set directly without calling factory."""
        factory_called = False

        async def factory():
            nonlocal factory_called
            factory_called = True
            return "factory_value"

        lazy_val = LazyValue(factory, "test_prop")
        lazy_val.set("preset_value")

        result = await lazy_val.get()

        assert not factory_called
        assert lazy_val.is_loaded
        assert result == "preset_value"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe lazy loading."""

    @pytest.mark.asyncio
    async def test_concurrent_access_from_multiple_tasks(self):
        """Concurrent access from multiple async tasks."""
        call_count = 0
        call_lock = asyncio.Lock()

        async def factory():
            nonlocal call_count
            async with call_lock:
                call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return "shared_value"

        lazy_val = LazyValue(factory, "test_prop")

        # Create many concurrent tasks
        tasks = [lazy_val.get() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should get the same value
        assert all(r == "shared_value" for r in results)
        # Factory should be called only once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_race_conditions_in_initialization(self):
        """No race conditions during initialization."""
        results = []
        call_count = 0

        async def slow_factory():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Slow initialization
            return {"instance": call_count}

        lazy_val = LazyValue(slow_factory, "test_prop")

        async def access_value():
            result = await lazy_val.get()
            results.append(result)

        # Concurrent initialization attempts
        await asyncio.gather(*[access_value() for _ in range(20)])

        # All tasks should get the exact same instance
        first_result = results[0]
        assert all(r is first_result for r in results)
        assert call_count == 1

    def test_n_plus_one_detection_thread_safe(self):
        """N+1 detection is thread-safe with concurrent access."""
        from aragora.performance.lazy_loading import _detect_n_plus_one, _n_plus_one_tracker

        property_name = "thread_safe_test"
        detection_count = 0
        detection_lock = threading.Lock()

        def trigger_detection():
            nonlocal detection_count
            for _ in range(10):
                detected = _detect_n_plus_one(property_name)
                if detected:
                    with detection_lock:
                        detection_count += 1
                time.sleep(0.001)

        threads = [threading.Thread(target=trigger_detection) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Detection should have occurred multiple times across threads
        assert detection_count > 0


# =============================================================================
# Memory Efficiency Tests
# =============================================================================


class TestMemoryEfficiency:
    """Tests for memory efficiency of lazy loading."""

    @pytest.mark.asyncio
    async def test_lazy_objects_dont_consume_memory_until_accessed(self):
        """Lazy objects don't consume memory until accessed."""
        large_data_created = False

        async def create_large_data():
            nonlocal large_data_created
            large_data_created = True
            return [0] * 1000000  # 1M integers

        lazy_val = LazyValue(create_large_data, "large_data")

        # Before access, large data shouldn't be created
        gc.collect()
        assert not large_data_created

        # After access, data is created
        result = await lazy_val.get()
        assert large_data_created
        assert len(result) == 1000000

    @pytest.mark.asyncio
    async def test_descriptor_uses_weak_key_dictionary(self):
        """Descriptor uses WeakKeyDictionary for cache - allowing model cleanup."""

        class TestModel:
            @lazy_property
            async def data(self):
                return "model_data"

        # Create and access model
        model = TestModel()
        lazy_value = model.data
        assert isinstance(lazy_value, LazyValue)

        # Get the descriptor's cache
        descriptor = TestModel.__dict__["data"]
        assert isinstance(descriptor, LazyDescriptor)

        # The cache should use WeakKeyDictionary
        assert isinstance(descriptor._cache, weakref.WeakKeyDictionary)

        # Model is in cache
        assert model in descriptor._cache

        # Create another model and verify separate caching
        model2 = TestModel()
        _ = model2.data

        # Both models should be in cache
        assert len(descriptor._cache) == 2
        assert model in descriptor._cache
        assert model2 in descriptor._cache

    @pytest.mark.asyncio
    async def test_stats_track_load_times(self):
        """Stats properly track load times."""
        reset_lazy_load_stats()

        async def timed_factory():
            await asyncio.sleep(0.01)
            return "value"

        lazy_val = LazyValue(timed_factory, "timed_prop")
        await lazy_val.get()

        stats = get_lazy_load_stats()
        assert stats["total_loads"] == 1
        assert stats["avg_load_time_ms"] > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in lazy loading."""

    @pytest.mark.asyncio
    async def test_factory_function_raises_exception(self):
        """Factory function exception is properly propagated."""

        async def failing_factory():
            raise ValueError("Factory failed")

        lazy_val = LazyValue(failing_factory, "failing_prop")

        with pytest.raises(ValueError, match="Factory failed"):
            await lazy_val.get()

    @pytest.mark.asyncio
    async def test_recovery_after_failed_initialization(self):
        """Can recover after failed initialization by creating new LazyValue."""
        call_count = 0

        async def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First call fails")
            return "success"

        lazy_val = LazyValue(sometimes_fails, "retry_prop")

        # First call fails
        with pytest.raises(RuntimeError):
            await lazy_val.get()

        # Value is not marked as loaded after failure
        assert not lazy_val.is_loaded

        # Create a new lazy value for retry
        lazy_val2 = LazyValue(sometimes_fails, "retry_prop")
        result = await lazy_val2.get()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_exception_propagation_to_concurrent_waiters(self):
        """Exception is propagated to all concurrent waiters."""

        async def slow_failing_factory():
            await asyncio.sleep(0.05)
            raise ValueError("Shared failure")

        lazy_val = LazyValue(slow_failing_factory, "shared_fail_prop")

        async def access_value():
            try:
                return await lazy_val.get()
            except ValueError as e:
                return str(e)

        results = await asyncio.gather(*[access_value() for _ in range(10)])

        # All should have received the exception
        assert all(r == "Shared failure" for r in results)

    @pytest.mark.asyncio
    async def test_loader_in_invalid_state_error(self):
        """Raises error if loader is in invalid state."""

        async def factory():
            return "value"

        lazy_val = LazyValue(factory, "test_prop")

        # Manually set loading to True without setting future
        lazy_val._loading = True
        lazy_val._load_future = None

        with pytest.raises(RuntimeError, match="invalid state"):
            await lazy_val.get()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for configuration options."""

    def test_get_n_plus_one_config(self):
        """N+1 detection config is accessible."""
        config = get_n_plus_one_config()

        assert "threshold" in config
        assert "window_ms" in config
        assert "auto_prefetch_enabled" in config
        assert config["threshold"] == N_PLUS_ONE_THRESHOLD
        assert config["window_ms"] == N_PLUS_ONE_WINDOW_MS

    def test_stats_include_config(self):
        """Stats include configuration info."""
        stats = get_lazy_load_stats()

        assert "config" in stats
        assert stats["config"]["threshold"] == N_PLUS_ONE_THRESHOLD

    def test_reset_stats_clears_counters(self):
        """Reset stats clears all counters."""
        from aragora.performance.lazy_loading import _lazy_load_stats

        # Set some values
        _lazy_load_stats.total_loads = 100
        _lazy_load_stats.n_plus_one_detections = 5

        reset_lazy_load_stats()

        stats = get_lazy_load_stats()
        assert stats["total_loads"] == 0
        assert stats["n_plus_one_detections"] == 0


# =============================================================================
# Lazy Property Decorator Tests
# =============================================================================


class TestLazyPropertyDecorator:
    """Tests for the lazy_property decorator."""

    @pytest.mark.asyncio
    async def test_decorator_without_arguments(self):
        """Decorator works without arguments."""

        class Model:
            def __init__(self, id: str):
                self.id = id

            @lazy_property
            async def data(self):
                return f"data_for_{self.id}"

        model = Model("123")
        lazy_val = model.data

        assert isinstance(lazy_val, LazyValue)
        result = await lazy_val.get()
        assert result == "data_for_123"

    @pytest.mark.asyncio
    async def test_decorator_with_prefetch_key(self):
        """Decorator works with prefetch_key argument."""

        class Model:
            @lazy_property(prefetch_key="custom_key")
            async def data(self):
                return "value"

        descriptor = Model.__dict__["data"]
        assert isinstance(descriptor, LazyDescriptor)
        assert descriptor.prefetch_key == "custom_key"

    @pytest.mark.asyncio
    async def test_different_instances_have_separate_values(self):
        """Different instances have separate lazy values."""

        class Model:
            def __init__(self, id: str):
                self.id = id

            @lazy_property
            async def data(self):
                return f"data_for_{self.id}"

        model1 = Model("1")
        model2 = Model("2")

        result1 = await model1.data.get()
        result2 = await model2.data.get()

        assert result1 == "data_for_1"
        assert result2 == "data_for_2"

    @pytest.mark.asyncio
    async def test_descriptor_set_value(self):
        """Descriptor allows setting value directly."""

        class Model:
            @lazy_property
            async def data(self):
                return "factory_value"

        model = Model()
        model.data = "preset_value"

        result = await model.data.get()
        assert result == "preset_value"

    def test_accessing_descriptor_on_class(self):
        """Accessing descriptor on class returns the descriptor."""

        class Model:
            @lazy_property
            async def data(self):
                return "value"

        assert isinstance(Model.data, LazyDescriptor)


# =============================================================================
# N+1 Detection Tests
# =============================================================================


class TestNPlusOneDetection:
    """Tests for N+1 query detection."""

    @pytest.mark.asyncio
    async def test_detects_n_plus_one_pattern(self):
        """Detects N+1 query pattern when threshold exceeded."""
        from aragora.performance.lazy_loading import _detect_n_plus_one, _n_plus_one_tracker

        property_name = "n_plus_one_test_prop"
        _n_plus_one_tracker.clear()

        # Trigger many detections within window
        detections = 0
        for _ in range(N_PLUS_ONE_THRESHOLD + 5):
            if _detect_n_plus_one(property_name):
                detections += 1

        assert detections > 0

    @pytest.mark.asyncio
    async def test_logs_warning_on_n_plus_one(self):
        """Logs warning when N+1 pattern detected."""
        from aragora.performance.lazy_loading import _n_plus_one_tracker

        class Model:
            @lazy_property
            async def items(self):
                return ["item"]

        _n_plus_one_tracker.clear()

        with patch("aragora.performance.lazy_loading.logger") as mock_logger:
            # Create and access multiple models rapidly
            for i in range(N_PLUS_ONE_THRESHOLD + 2):
                model = Model()
                await model.items.get()

            # Should have logged a warning
            assert mock_logger.warning.called

    @pytest.mark.asyncio
    async def test_old_timestamps_cleaned_up(self):
        """Old timestamps are cleaned from N+1 tracker."""
        from aragora.performance.lazy_loading import _detect_n_plus_one, _n_plus_one_tracker

        property_name = "cleanup_test_prop"
        _n_plus_one_tracker.clear()

        # Trigger some detections
        for _ in range(3):
            _detect_n_plus_one(property_name)

        # Wait for window to expire
        time.sleep(N_PLUS_ONE_WINDOW_MS / 1000 + 0.01)

        # Trigger again - old entries should be cleaned
        _detect_n_plus_one(property_name)

        # Only recent entry should remain
        assert len(_n_plus_one_tracker[property_name]) <= 2


# =============================================================================
# Prefetch Tests
# =============================================================================


class TestPrefetch:
    """Tests for prefetch functionality."""

    @pytest.mark.asyncio
    async def test_prefetch_with_registered_function(self):
        """Prefetch uses registered function for batch loading."""
        batch_load_called = False
        batch_size = 0

        class User:
            def __init__(self, id: str):
                self.id = id

            @lazy_property(prefetch_key="user_posts")
            async def posts(self):
                return [f"post_for_{self.id}"]

        async def batch_load_posts(users):
            nonlocal batch_load_called, batch_size
            batch_load_called = True
            batch_size = len(users)
            return {u: [f"batch_post_for_{u.id}"] for u in users}

        register_prefetch("user_posts", batch_load_posts)

        users = [User("1"), User("2"), User("3")]
        await prefetch(users, "posts")

        assert batch_load_called
        assert batch_size == 3

        # Values should be prefetched
        for user in users:
            result = await user.posts.get()
            assert result == [f"batch_post_for_{user.id}"]

    @pytest.mark.asyncio
    async def test_prefetch_empty_list(self):
        """Prefetch handles empty list gracefully."""
        await prefetch([], "some_property")  # Should not raise

    @pytest.mark.asyncio
    async def test_prefetch_fallback_to_individual_loads(self):
        """Prefetch falls back to individual loads when no batch function."""

        class Model:
            def __init__(self, id: str):
                self.id = id

            @lazy_property
            async def data(self):
                return f"data_{self.id}"

        models = [Model("1"), Model("2")]

        with patch("aragora.performance.lazy_loading.logger") as mock_logger:
            await prefetch(models, "data")
            mock_logger.debug.assert_called()  # Logs fallback

        # Values should still be loaded
        for model in models:
            result = await model.data.get()
            assert result == f"data_{model.id}"

    @pytest.mark.asyncio
    async def test_prefetch_non_lazy_property_warning(self):
        """Prefetch warns when property is not a lazy_property."""

        class Model:
            @property
            def regular_prop(self):
                return "value"

        models = [Model()]

        with patch("aragora.performance.lazy_loading.logger") as mock_logger:
            await prefetch(models, "regular_prop")
            mock_logger.warning.assert_called()


# =============================================================================
# LazyLoader Tests
# =============================================================================


class TestLazyLoader:
    """Tests for LazyLoader class."""

    @pytest.mark.asyncio
    async def test_register_and_use_prefetch_function(self):
        """Can register and use prefetch functions."""
        loader = LazyLoader()

        class Item:
            def __init__(self, id: str):
                self.id = id

            @lazy_property(prefetch_key="item_details")
            async def details(self):
                return {"id": self.id}

        async def batch_load_details(items):
            return {i: {"id": i.id, "batch": True} for i in items}

        loader.register_prefetch("item_details", batch_load_details)

        items = [Item("a"), Item("b")]
        await loader.prefetch(items, "details")

        for item in items:
            result = await item.details.get()
            assert result["batch"] is True


# =============================================================================
# AutoPrefetchBatcher Tests
# =============================================================================


class TestAutoPrefetchBatcher:
    """Tests for automatic prefetch batching."""

    @pytest.mark.asyncio
    async def test_batcher_initialization(self):
        """Batcher initializes properly."""
        batcher = AutoPrefetchBatcher()
        assert batcher._pending == {}
        assert batcher._batch_futures == {}

    @pytest.mark.asyncio
    async def test_batcher_add_returns_false_when_disabled(self):
        """Batcher returns False when auto prefetch is disabled."""
        batcher = AutoPrefetchBatcher()

        async def loader():
            return "value"

        lazy_val = LazyValue(loader, "test")

        with patch("aragora.performance.lazy_loading.N_PLUS_ONE_AUTO_PREFETCH", False):
            result = await batcher.add_to_batch("test", object(), lazy_val, loader)
            assert result is False


# =============================================================================
# LazyLoadStats Tests
# =============================================================================


class TestLazyLoadStats:
    """Tests for LazyLoadStats dataclass."""

    def test_stats_initial_values(self):
        """Stats have correct initial values."""
        stats = LazyLoadStats()
        assert stats.total_loads == 0
        assert stats.n_plus_one_detections == 0
        assert stats.prefetch_hits == 0
        assert stats.load_times_ms == []

    def test_avg_load_time_empty(self):
        """Average load time is 0 for empty list."""
        stats = LazyLoadStats()
        assert stats.avg_load_time_ms == 0.0

    def test_avg_load_time_calculation(self):
        """Average load time calculated correctly."""
        stats = LazyLoadStats(load_times_ms=[10.0, 20.0, 30.0])
        assert stats.avg_load_time_ms == 20.0

    @pytest.mark.asyncio
    async def test_prefetch_hits_tracked(self):
        """Prefetch hits are tracked when accessing already loaded values."""

        async def factory():
            return "value"

        lazy_val = LazyValue(factory, "test")

        # First access - loads the value
        await lazy_val.get()
        initial_hits = get_lazy_load_stats()["prefetch_hits"]

        # Second access - should increment prefetch hits
        await lazy_val.get()
        assert get_lazy_load_stats()["prefetch_hits"] == initial_hits + 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for lazy loading framework."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Complete workflow: define, access, prefetch."""
        call_counts: dict[str, int] = {}

        class User:
            def __init__(self, id: str):
                self.id = id
                call_counts[id] = 0

            @lazy_property(prefetch_key="user_profile")
            async def profile(self):
                call_counts[self.id] += 1
                await asyncio.sleep(0.001)
                return {"id": self.id, "name": f"User {self.id}"}

        async def batch_load_profiles(users):
            return {u: {"id": u.id, "name": f"User {u.id}", "batch": True} for u in users}

        register_prefetch("user_profile", batch_load_profiles)

        # Create users
        users = [User(str(i)) for i in range(5)]

        # Prefetch all profiles
        await prefetch(users, "profile")

        # Access all profiles - should not call factory
        for user in users:
            profile = await user.profile.get()
            assert profile["batch"] is True
            assert call_counts[user.id] == 0  # Factory not called

    @pytest.mark.asyncio
    async def test_mixed_access_patterns(self):
        """Mixed access: some prefetched, some not."""

        class Item:
            def __init__(self, id: str):
                self.id = id

            @lazy_property
            async def value(self):
                return f"value_{self.id}"

        items = [Item("1"), Item("2"), Item("3")]

        # Manually set one value
        items[0].value = "preset_1"

        # Access all
        results = [await item.value.get() for item in items]

        assert results[0] == "preset_1"
        assert results[1] == "value_2"
        assert results[2] == "value_3"

    @pytest.mark.asyncio
    async def test_stats_accumulate_across_operations(self):
        """Stats accumulate correctly across multiple operations."""
        reset_lazy_load_stats()

        class Model:
            def __init__(self, id: str):
                self.id = id

            @lazy_property
            async def data(self):
                return f"data_{self.id}"

        # Create and access several models
        for i in range(5):
            model = Model(str(i))
            await model.data.get()

        stats = get_lazy_load_stats()
        assert stats["total_loads"] == 5
        assert len(get_lazy_load_stats()["config"]) > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_none_value_from_factory(self):
        """Handles None value from factory correctly."""

        async def factory():
            return None

        lazy_val = LazyValue(factory, "none_prop")
        result = await lazy_val.get()

        assert result is None
        assert lazy_val.is_loaded

    @pytest.mark.asyncio
    async def test_empty_collection_from_factory(self):
        """Handles empty collection from factory."""

        async def factory():
            return []

        lazy_val = LazyValue(factory, "empty_prop")
        result = await lazy_val.get()

        assert result == []
        assert lazy_val.is_loaded

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self):
        """Handles complex nested structures."""

        async def factory():
            return {
                "users": [
                    {"id": 1, "posts": [{"title": "Hello"}]},
                    {"id": 2, "posts": []},
                ],
                "metadata": {"count": 2},
            }

        lazy_val = LazyValue(factory, "complex_prop")
        result = await lazy_val.get()

        assert len(result["users"]) == 2
        assert result["metadata"]["count"] == 2

    @pytest.mark.asyncio
    async def test_very_fast_factory(self):
        """Handles very fast factory functions."""
        call_count = 0

        async def instant_factory():
            nonlocal call_count
            call_count += 1
            return "instant"

        lazy_val = LazyValue(instant_factory, "fast_prop")

        # Rapid access
        results = await asyncio.gather(*[lazy_val.get() for _ in range(100)])

        assert all(r == "instant" for r in results)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_factory_returns_callable(self):
        """Factory can return callable objects."""

        async def factory():
            return lambda x: x * 2

        lazy_val = LazyValue(factory, "callable_prop")
        fn = await lazy_val.get()

        assert callable(fn)
        assert fn(5) == 10
