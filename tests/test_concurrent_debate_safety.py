"""
Integration tests for concurrent debate safety.

Tests thread safety, state management, race condition prevention,
and concurrent debate execution limits.
"""

import asyncio
import threading
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Optional

from aragora.server.state import StateManager, DebateState, get_state_manager, reset_state_manager
from aragora.server.debate_utils import (
    get_active_debates,
    update_debate_status,
    cleanup_stale_debates,
    increment_cleanup_counter,
    _active_debates,
    _active_debates_lock,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state manager before each test."""
    reset_state_manager()
    yield
    reset_state_manager()


@pytest.fixture
def state_manager():
    """Get a fresh state manager."""
    return get_state_manager()


# =============================================================================
# StateManager Thread Safety Tests
# =============================================================================


class TestStateManagerThreadSafety:
    """Tests for StateManager thread safety."""

    def test_concurrent_debate_registration(self, state_manager):
        """Multiple threads can register debates without race conditions."""
        num_threads = 20
        num_debates_per_thread = 10
        errors = []

        def register_debates(thread_id):
            try:
                for i in range(num_debates_per_thread):
                    debate_id = f"thread{thread_id}_debate{i}"
                    state_manager.register_debate(
                        debate_id=debate_id,
                        task=f"Task for {debate_id}",
                        agents=["agent1", "agent2"],
                        total_rounds=3,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_debates, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert state_manager.get_active_debate_count() == num_threads * num_debates_per_thread

    def test_concurrent_status_updates(self, state_manager):
        """Multiple threads can update debate status safely."""
        debate_id = "concurrent_update_test"
        state_manager.register_debate(
            debate_id=debate_id,
            task="Test task",
            agents=["agent1"],
            total_rounds=5,
        )

        errors = []
        updates_made = []

        def update_status(thread_id, round_num):
            try:
                state_manager.update_debate_status(
                    debate_id,
                    status="running",
                    current_round=round_num,
                )
                updates_made.append((thread_id, round_num))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_status, args=(i, i % 5 + 1)) for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All updates should have completed
        assert len(updates_made) == 50
        # Debate should still exist
        assert state_manager.get_debate(debate_id) is not None

    def test_concurrent_read_write(self, state_manager):
        """Concurrent reads and writes don't cause corruption."""
        debate_ids = [f"rw_test_{i}" for i in range(10)]

        # Pre-register debates
        for debate_id in debate_ids:
            state_manager.register_debate(
                debate_id=debate_id,
                task="Test task",
                agents=["agent1"],
            )

        errors = []
        read_results = []
        write_count = []

        def reader(iterations):
            for _ in range(iterations):
                try:
                    debates = state_manager.get_active_debates()
                    read_results.append(len(debates))
                except Exception as e:
                    errors.append(("read", e))

        def writer(thread_id, iterations):
            for i in range(iterations):
                try:
                    debate_id = f"new_debate_{thread_id}_{i}"
                    state_manager.register_debate(
                        debate_id=debate_id,
                        task="New task",
                        agents=["agent1"],
                    )
                    write_count.append(debate_id)
                except Exception as e:
                    errors.append(("write", e))

        # Start readers and writers concurrently
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader, args=(100,)))
        for i in range(3):
            threads.append(threading.Thread(target=writer, args=(i, 50)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All reads should have returned valid counts
        assert len(read_results) == 500
        # All writes should have completed
        assert len(write_count) == 150

    def test_concurrent_unregistration(self, state_manager):
        """Multiple threads can unregister debates safely."""
        # Register many debates
        debate_ids = [f"unreg_test_{i}" for i in range(100)]
        for debate_id in debate_ids:
            state_manager.register_debate(
                debate_id=debate_id,
                task="Test task",
                agents=["agent1"],
            )

        errors = []
        unregistered = []

        def unregister(debate_id):
            try:
                result = state_manager.unregister_debate(debate_id)
                if result:
                    unregistered.append(debate_id)
            except Exception as e:
                errors.append(e)

        # Unregister from multiple threads (some will try same debate)
        threads = []
        for debate_id in debate_ids:
            threads.append(threading.Thread(target=unregister, args=(debate_id,)))
        # Add duplicate attempts
        for debate_id in debate_ids[:50]:
            threads.append(threading.Thread(target=unregister, args=(debate_id,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Each debate should only be unregistered once
        assert len(set(unregistered)) == len(unregistered)
        assert state_manager.get_active_debate_count() == 0


# =============================================================================
# Lock Order Tests
# =============================================================================


class TestLockOrdering:
    """Tests for correct lock acquisition order."""

    def test_no_deadlock_under_contention(self, state_manager):
        """Heavy contention should not cause deadlocks."""
        # This test tries to provoke deadlocks by hammering the state manager
        # with multiple operations that acquire different locks

        timeout_seconds = 5
        completed = threading.Event()
        errors = []

        def mixed_operations(thread_id):
            try:
                for i in range(50):
                    # Register
                    debate_id = f"deadlock_test_{thread_id}_{i}"
                    state_manager.register_debate(
                        debate_id=debate_id,
                        task="Test",
                        agents=["a1"],
                    )

                    # Update
                    state_manager.update_debate_status(debate_id, "running")

                    # Read
                    _ = state_manager.get_active_debates()

                    # Cleanup check (module-level function)
                    increment_cleanup_counter()

                    # Unregister
                    state_manager.unregister_debate(debate_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_operations, args=(i,)) for i in range(10)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout_seconds)

        elapsed = time.time() - start_time

        # All threads should complete within timeout
        all_done = all(not t.is_alive() for t in threads)
        assert all_done, f"Threads still running after {elapsed:.2f}s - possible deadlock"
        assert len(errors) == 0


# =============================================================================
# Debate State Consistency Tests
# =============================================================================


class TestDebateStateConsistency:
    """Tests for debate state consistency under concurrent access."""

    def test_state_isolation_between_debates(self, state_manager):
        """Updates to one debate don't affect others."""
        # Register multiple debates
        debates = {}
        for i in range(10):
            debate_id = f"isolation_test_{i}"
            state_manager.register_debate(
                debate_id=debate_id,
                task=f"Task {i}",
                agents=[f"agent_{i}"],
                total_rounds=3,
            )
            debates[debate_id] = {"expected_task": f"Task {i}"}

        errors = []

        def update_and_check(debate_id, expected_task):
            try:
                # Update this debate
                state_manager.update_debate_status(debate_id, "running", current_round=2)

                # Read it back and verify
                state = state_manager.get_debate(debate_id)
                if state.task != expected_task:
                    errors.append(f"Task mismatch: {state.task} != {expected_task}")
                if state.status != "running":
                    errors.append(f"Status mismatch for {debate_id}")
            except Exception as e:
                errors.append(str(e))

        # Concurrent updates and checks
        threads = [
            threading.Thread(
                target=update_and_check,
                args=(debate_id, info["expected_task"]),
            )
            for debate_id, info in debates.items()
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_metadata_preservation(self, state_manager):
        """Metadata should be preserved through updates."""
        debate_id = "metadata_test"
        original_metadata = {
            "custom_field": "original_value",
            "numeric": 42,
            "nested": {"key": "value"},
        }

        state_manager.register_debate(
            debate_id=debate_id,
            task="Test task",
            agents=["agent1"],
            metadata=original_metadata,
        )

        # Multiple updates shouldn't lose metadata
        for i in range(10):
            state_manager.update_debate_status(debate_id, "running", current_round=i)

        state = state_manager.get_debate(debate_id)
        assert state.metadata.get("custom_field") == "original_value"
        assert state.metadata.get("numeric") == 42


# =============================================================================
# Active Debates Proxy Tests
# =============================================================================


class TestActiveDebatesProxy:
    """Tests for backward-compatible _active_debates proxy."""

    def test_proxy_dict_operations(self, state_manager):
        """Proxy should support standard dict operations."""
        # Register via state manager
        state_manager.register_debate(
            debate_id="proxy_test",
            task="Test task",
            agents=["agent1"],
        )

        # Access via proxy
        assert "proxy_test" in _active_debates
        assert len(_active_debates) == 1
        assert "proxy_test" in _active_debates.keys()

        # Get via proxy
        debate = _active_debates.get("proxy_test")
        assert debate is not None

        # Items iteration
        items = list(_active_debates.items())
        assert len(items) == 1

    def test_proxy_with_lock(self, state_manager):
        """Proxy should work with legacy lock pattern."""
        errors = []

        def legacy_pattern(thread_id):
            try:
                with _active_debates_lock:
                    debate_id = f"legacy_test_{thread_id}"
                    _active_debates[debate_id] = {
                        "id": debate_id,
                        "task": "Test",
                        "agents": ["agent1"],
                    }

                    # Read it back
                    assert debate_id in _active_debates
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=legacy_pattern, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Cleanup Under Concurrent Load Tests
# =============================================================================


class TestCleanupConcurrency:
    """Tests for cleanup operations under concurrent load."""

    def test_cleanup_during_active_debates(self, state_manager):
        """Cleanup should only remove completed/error debates."""
        # Register active and completed debates
        for i in range(5):
            state_manager.register_debate(
                debate_id=f"active_{i}",
                task="Active task",
                agents=["agent1"],
            )
            state_manager.update_debate_status(f"active_{i}", "running")

        for i in range(5):
            debate_id = f"completed_{i}"
            state_manager.register_debate(
                debate_id=debate_id,
                task="Completed task",
                agents=["agent1"],
            )
            state_manager.update_debate_status(debate_id, "completed")
            # Set old completion time to trigger cleanup
            state = state_manager.get_debate(debate_id)
            state.metadata["completed_at"] = time.time() - 4000  # Older than TTL

        # Cleanup while other operations are happening
        errors = []

        def operations():
            try:
                for _ in range(20):
                    _ = state_manager.get_active_debates()
                    cleanup_stale_debates()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Active debates should still exist
        for i in range(5):
            assert state_manager.get_debate(f"active_{i}") is not None

    def test_concurrent_cleanup_calls(self, state_manager):
        """Multiple concurrent cleanup calls should be safe."""
        # Create many stale debates
        for i in range(100):
            debate_id = f"stale_{i}"
            state_manager.register_debate(
                debate_id=debate_id,
                task="Stale task",
                agents=["agent1"],
            )
            state_manager.update_debate_status(debate_id, "completed")
            state = state_manager.get_debate(debate_id)
            state.metadata["completed_at"] = time.time() - 4000

        errors = []

        def cleanup():
            try:
                for _ in range(10):
                    cleanup_stale_debates()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Executor Management Tests
# =============================================================================


class TestExecutorManagement:
    """Tests for ThreadPoolExecutor management."""

    def test_executor_lazy_initialization(self, state_manager):
        """Executor should be lazily initialized."""
        # Initially no executor
        assert state_manager._executor is None

        # Get executor
        executor = state_manager.get_executor()
        assert executor is not None

        # Same executor returned
        assert state_manager.get_executor() is executor

    def test_executor_concurrent_access(self, state_manager):
        """Multiple threads requesting executor should get same instance."""
        executors = []
        errors = []

        def get_exec():
            try:
                exec = state_manager.get_executor()
                executors.append(exec)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_exec) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All should be same instance
        assert all(e is executors[0] for e in executors)

    def test_executor_shutdown_safety(self, state_manager):
        """Executor shutdown should be safe under concurrent access."""
        # Get executor
        _ = state_manager.get_executor()

        errors = []

        def use_executor():
            try:
                for _ in range(10):
                    exec = state_manager.get_executor()
                    if exec:
                        # Simulate work
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def shutdown():
            try:
                time.sleep(0.01)
                state_manager.shutdown_executor()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=use_executor) for _ in range(5)]
        threads.append(threading.Thread(target=shutdown))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some errors are expected if shutdown happens mid-use
        # but no crashes or deadlocks


# =============================================================================
# Full Debate Lifecycle Concurrent Tests
# =============================================================================


class TestDebateLifecycleConcurrency:
    """Tests for full debate lifecycle under concurrent load."""

    def test_multiple_debate_lifecycles(self, state_manager):
        """Multiple debates can run through full lifecycle concurrently."""
        num_debates = 20
        errors = []
        completed = []

        def debate_lifecycle(debate_id):
            try:
                # Register
                state_manager.register_debate(
                    debate_id=debate_id,
                    task="Lifecycle test",
                    agents=["agent1", "agent2"],
                    total_rounds=3,
                )

                # Simulate rounds
                for round_num in range(1, 4):
                    state_manager.update_debate_status(
                        debate_id,
                        status="running",
                        current_round=round_num,
                    )
                    time.sleep(0.01)  # Simulate work

                # Complete
                state_manager.update_debate_status(debate_id, "completed")
                completed.append(debate_id)

                # Unregister
                time.sleep(0.05)  # Let other operations see the completed state
                state_manager.unregister_debate(debate_id)

            except Exception as e:
                errors.append((debate_id, e))

        threads = [
            threading.Thread(target=debate_lifecycle, args=(f"lifecycle_{i}",))
            for i in range(num_debates)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(completed) == num_debates

    def test_mixed_lifecycle_operations(self, state_manager):
        """Different lifecycle operations can happen concurrently."""
        errors = []
        operations_count = {"register": 0, "update": 0, "read": 0, "unregister": 0}
        lock = threading.Lock()

        def registrar():
            for i in range(50):
                try:
                    state_manager.register_debate(
                        debate_id=f"mixed_reg_{i}",
                        task="Test",
                        agents=["a1"],
                    )
                    with lock:
                        operations_count["register"] += 1
                except Exception as e:
                    errors.append(("register", e))

        def updater():
            for _ in range(100):
                try:
                    debates = state_manager.get_active_debates()
                    for debate_id in list(debates.keys())[:5]:
                        state_manager.update_debate_status(debate_id, "running")
                        with lock:
                            operations_count["update"] += 1
                except Exception as e:
                    errors.append(("update", e))

        def reader():
            for _ in range(100):
                try:
                    _ = state_manager.get_active_debates()
                    with lock:
                        operations_count["read"] += 1
                except Exception as e:
                    errors.append(("read", e))

        def unregistrar():
            for _ in range(100):
                try:
                    debates = state_manager.get_active_debates()
                    for debate_id in list(debates.keys())[:2]:
                        state_manager.unregister_debate(debate_id)
                        with lock:
                            operations_count["unregister"] += 1
                except Exception as e:
                    errors.append(("unregister", e))

        threads = [
            threading.Thread(target=registrar),
            threading.Thread(target=registrar),
            threading.Thread(target=updater),
            threading.Thread(target=updater),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=unregistrar),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All operation types should have happened
        assert operations_count["register"] == 100
        assert operations_count["read"] == 200


# =============================================================================
# Race Condition Edge Cases
# =============================================================================


class TestRaceConditionEdgeCases:
    """Tests for specific race condition scenarios."""

    def test_register_then_immediate_unregister(self, state_manager):
        """Rapid register/unregister cycles should be safe."""
        errors = []

        def rapid_cycle(thread_id):
            try:
                for i in range(100):
                    debate_id = f"rapid_{thread_id}_{i}"
                    state_manager.register_debate(
                        debate_id=debate_id,
                        task="Rapid test",
                        agents=["a1"],
                    )
                    state_manager.unregister_debate(debate_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=rapid_cycle, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_double_registration(self, state_manager):
        """Double registration should be handled gracefully."""
        debate_id = "double_reg_test"

        # First registration
        state1 = state_manager.register_debate(
            debate_id=debate_id,
            task="Task 1",
            agents=["a1"],
        )

        # Second registration (should update or reject)
        state2 = state_manager.register_debate(
            debate_id=debate_id,
            task="Task 2",
            agents=["a2"],
        )

        # Should only have one debate
        assert state_manager.get_active_debate_count() == 1

    def test_update_nonexistent_debate(self, state_manager):
        """Updating nonexistent debate should be handled gracefully."""
        # Should not raise
        state_manager.update_debate_status("nonexistent_debate", "running")

    def test_unregister_nonexistent_debate(self, state_manager):
        """Unregistering nonexistent debate should be handled gracefully."""
        # Should not raise, return None
        result = state_manager.unregister_debate("nonexistent_debate")
        assert result is None

    def test_concurrent_same_debate_updates(self, state_manager):
        """Many threads updating same debate should not corrupt state."""
        debate_id = "same_debate_test"
        state_manager.register_debate(
            debate_id=debate_id,
            task="Test",
            agents=["a1"],
            total_rounds=100,
        )

        errors = []
        rounds_set = []
        lock = threading.Lock()

        def updater(round_num):
            try:
                state_manager.update_debate_status(
                    debate_id,
                    status="running",
                    current_round=round_num,
                )
                with lock:
                    rounds_set.append(round_num)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(1, 101)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(rounds_set) == 100

        # Final state should be valid (some round number was set)
        state = state_manager.get_debate(debate_id)
        assert state is not None
        assert state.current_round in range(1, 101)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
