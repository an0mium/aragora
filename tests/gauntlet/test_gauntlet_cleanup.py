"""
Tests for gauntlet cleanup and memory management.

Verifies that the gauntlet runs cleanup works correctly under concurrent
access and prevents memory leaks.
"""

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


class TestGauntletCleanup:
    """Tests for the gauntlet runs cleanup functionality."""

    def test_cleanup_removes_old_entries(self):
        """Verify that cleanup removes entries older than MAX_AGE."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
            _GAUNTLET_MAX_AGE_SECONDS,
        )

        # Clear any existing entries
        _gauntlet_runs.clear()

        # Add old entries (older than MAX_AGE)
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 100
        for i in range(10):
            _gauntlet_runs[f"old-{i}"] = {
                "created_at": old_time,
                "status": "running",
            }

        # Add recent entries
        for i in range(5):
            _gauntlet_runs[f"recent-{i}"] = {
                "created_at": time.time(),
                "status": "running",
            }

        assert len(_gauntlet_runs) == 15

        _cleanup_gauntlet_runs()

        # Old entries should be removed, recent should remain
        assert len(_gauntlet_runs) == 5
        for key in _gauntlet_runs:
            assert key.startswith("recent-")

    def test_cleanup_removes_completed_after_ttl(self):
        """Verify that completed runs are removed after TTL expires."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
            _GAUNTLET_COMPLETED_TTL,
        )

        _gauntlet_runs.clear()

        # Add completed entries past TTL
        old_completed_time = (
            datetime.now() - timedelta(seconds=_GAUNTLET_COMPLETED_TTL + 100)
        ).isoformat()
        for i in range(5):
            _gauntlet_runs[f"completed-old-{i}"] = {
                "created_at": time.time() - 1000,  # Recent enough to not trigger MAX_AGE
                "completed_at": old_completed_time,
                "status": "completed",
            }

        # Add recently completed entries
        recent_completed_time = datetime.now().isoformat()
        for i in range(3):
            _gauntlet_runs[f"completed-recent-{i}"] = {
                "created_at": time.time(),
                "completed_at": recent_completed_time,
                "status": "completed",
            }

        assert len(_gauntlet_runs) == 8

        _cleanup_gauntlet_runs()

        # Old completed should be removed, recent should remain
        assert len(_gauntlet_runs) == 3
        for key in _gauntlet_runs:
            assert key.startswith("completed-recent-")

    def test_cleanup_fifo_eviction_when_over_limit(self):
        """Verify FIFO eviction when over MAX_GAUNTLET_RUNS_IN_MEMORY."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
            MAX_GAUNTLET_RUNS_IN_MEMORY,
        )

        _gauntlet_runs.clear()

        # Add more entries than the limit (all recent to avoid time-based cleanup)
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 100):
            _gauntlet_runs[f"entry-{i:05d}"] = {
                "created_at": time.time(),
                "status": "running",
            }

        assert len(_gauntlet_runs) == MAX_GAUNTLET_RUNS_IN_MEMORY + 100

        _cleanup_gauntlet_runs()

        # Should be at or below the limit
        assert len(_gauntlet_runs) <= MAX_GAUNTLET_RUNS_IN_MEMORY

        # Oldest entries (lowest numbers) should be removed first (FIFO)
        remaining_keys = list(_gauntlet_runs.keys())
        remaining_nums = [int(k.split("-")[1]) for k in remaining_keys]
        # Higher numbers should remain
        assert min(remaining_nums) >= 100

    def test_cleanup_handles_iso_timestamp(self):
        """Verify cleanup correctly parses ISO format timestamps."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
            _GAUNTLET_MAX_AGE_SECONDS,
        )

        _gauntlet_runs.clear()

        # Add entry with ISO timestamp (old)
        old_dt = datetime.now() - timedelta(seconds=_GAUNTLET_MAX_AGE_SECONDS + 100)
        _gauntlet_runs["iso-old"] = {
            "created_at": old_dt.isoformat(),
            "status": "running",
        }

        # Add entry with ISO timestamp (recent)
        _gauntlet_runs["iso-recent"] = {
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }

        _cleanup_gauntlet_runs()

        assert "iso-old" not in _gauntlet_runs
        assert "iso-recent" in _gauntlet_runs

    def test_cleanup_handles_missing_timestamp(self):
        """Verify cleanup handles entries without timestamp gracefully."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
        )

        _gauntlet_runs.clear()

        # Add entry without any timestamp
        _gauntlet_runs["no-timestamp"] = {
            "status": "running",
        }

        # Add entry with valid timestamp
        _gauntlet_runs["with-timestamp"] = {
            "created_at": time.time(),
            "status": "running",
        }

        # Should not raise
        _cleanup_gauntlet_runs()

        # Entry without timestamp should remain (no way to determine age)
        assert "no-timestamp" in _gauntlet_runs
        assert "with-timestamp" in _gauntlet_runs


class TestGauntletCleanupConcurrency:
    """Tests for concurrent access to gauntlet cleanup."""

    def test_concurrent_additions_during_cleanup(self):
        """Verify cleanup works correctly with concurrent additions."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
        )

        _gauntlet_runs.clear()
        errors = []
        added_during_cleanup = []

        def add_entries(start_idx):
            """Add entries while cleanup might be running."""
            try:
                for i in range(50):
                    key = f"concurrent-{start_idx}-{i}"
                    _gauntlet_runs[key] = {
                        "created_at": time.time(),
                        "status": "running",
                    }
                    added_during_cleanup.append(key)
                    time.sleep(0.001)  # Small delay to interleave
            except Exception as e:
                errors.append(str(e))

        def run_cleanup():
            """Run cleanup multiple times."""
            try:
                for _ in range(10):
                    _cleanup_gauntlet_runs()
                    time.sleep(0.005)
            except Exception as e:
                errors.append(str(e))

        # Start threads
        threads = [
            threading.Thread(target=add_entries, args=(0,)),
            threading.Thread(target=add_entries, args=(1,)),
            threading.Thread(target=run_cleanup),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_concurrent_cleanup_calls(self):
        """Verify multiple concurrent cleanup calls don't corrupt state."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
            _GAUNTLET_MAX_AGE_SECONDS,
        )

        _gauntlet_runs.clear()
        errors = []

        # Add mix of old and new entries
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 100
        for i in range(100):
            if i % 2 == 0:
                _gauntlet_runs[f"old-{i}"] = {"created_at": old_time, "status": "running"}
            else:
                _gauntlet_runs[f"new-{i}"] = {"created_at": time.time(), "status": "running"}

        def cleanup_thread():
            try:
                for _ in range(20):
                    _cleanup_gauntlet_runs()
            except Exception as e:
                errors.append(str(e))

        # Run multiple concurrent cleanups
        threads = [threading.Thread(target=cleanup_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent cleanup: {errors}"

        # All old entries should be removed
        for key in list(_gauntlet_runs.keys()):
            assert key.startswith("new-"), f"Old entry {key} not cleaned up"

    def test_high_throughput_additions(self):
        """Test cleanup performance under high throughput."""
        from aragora.server.handlers.gauntlet import (
            _gauntlet_runs,
            _cleanup_gauntlet_runs,
        )

        _gauntlet_runs.clear()
        errors = []
        add_count = [0]

        def rapid_add(thread_id):
            try:
                for i in range(200):
                    key = f"rapid-{thread_id}-{i}"
                    _gauntlet_runs[key] = {
                        "created_at": time.time(),
                        "status": "pending",
                    }
                    add_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        def periodic_cleanup():
            try:
                for _ in range(50):
                    _cleanup_gauntlet_runs()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            # 5 adder threads
            for i in range(5):
                futures.append(executor.submit(rapid_add, i))
            # 2 cleanup threads
            futures.append(executor.submit(periodic_cleanup))
            futures.append(executor.submit(periodic_cleanup))

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(str(e))

        assert len(errors) == 0, f"Errors during high throughput test: {errors}"
        # Should have added entries (some may be cleaned up)
        assert add_count[0] == 1000  # 5 threads * 200 entries


class TestQuotaLockAtomicity:
    """Tests for the quota check atomicity using the lock."""

    def test_quota_lock_prevents_race(self):
        """Verify quota lock prevents race conditions."""
        from aragora.server.handlers.gauntlet import _quota_lock

        counter = [0]
        max_quota = 10
        over_quota_count = [0]

        def check_and_increment():
            """Simulate quota check-and-increment."""
            with _quota_lock:
                if counter[0] < max_quota:
                    # Simulate some processing time
                    time.sleep(0.001)
                    counter[0] += 1
                else:
                    over_quota_count[0] += 1

        # Run many concurrent quota checks
        threads = [threading.Thread(target=check_and_increment) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Counter should never exceed max_quota
        assert counter[0] == max_quota
        assert over_quota_count[0] == 40  # 50 - 10 = 40 rejections

    def test_quota_lock_without_race(self):
        """Verify behavior without the lock (for comparison)."""
        counter = [0]
        max_quota = 10
        errors = []

        def check_and_increment_unsafe():
            """Simulate unsafe quota check-and-increment."""
            if counter[0] < max_quota:
                # Without lock, this creates a race
                time.sleep(0.001)
                counter[0] += 1

        # This test demonstrates the race condition that the lock prevents
        # We DON'T assert on exact values because it's racy by design
        threads = [threading.Thread(target=check_and_increment_unsafe) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Counter may exceed max_quota without the lock (race condition)
        # This is expected behavior showing why we need the lock
        # Just verify it ran without crashing
        assert counter[0] > 0


@pytest.fixture
def clean_gauntlet_state():
    """Fixture to ensure clean gauntlet state for each test."""
    from aragora.server.handlers.gauntlet import _gauntlet_runs

    _gauntlet_runs.clear()
    yield
    _gauntlet_runs.clear()
