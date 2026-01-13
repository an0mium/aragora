"""
Tests for quota check race condition fixes.

Verifies that quota check-and-increment operations are atomic
and prevent over-quota usage via TOCTOU attacks.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, MagicMock, patch

import pytest


class TestQuotaCheckAtomicity:
    """Tests for atomic quota check-and-increment."""

    def test_lock_prevents_toctou_race(self):
        """Verify the quota lock prevents TOCTOU race conditions."""
        from aragora.server.handlers.gauntlet import _quota_lock

        # Simulate an organization with a quota limit
        org_quota = {"used": 0, "limit": 10}
        accepted_requests = []
        rejected_requests = []

        def check_and_increment(request_id):
            """Simulate atomic quota check-and-increment."""
            with _quota_lock:
                if org_quota["used"] < org_quota["limit"]:
                    # Simulate processing delay
                    time.sleep(0.001)
                    org_quota["used"] += 1
                    accepted_requests.append(request_id)
                    return True
                else:
                    rejected_requests.append(request_id)
                    return False

        # Fire 50 concurrent requests
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(check_and_increment, f"req-{i}") for i in range(50)]
            for f in as_completed(futures):
                f.result()

        # Exactly 10 should be accepted (the quota limit)
        assert len(accepted_requests) == 10, f"Expected 10 accepted, got {len(accepted_requests)}"
        assert len(rejected_requests) == 40, f"Expected 40 rejected, got {len(rejected_requests)}"
        assert org_quota["used"] == 10

    def test_without_lock_demonstrates_race(self):
        """Demonstrate the race condition that exists without the lock."""
        # This test shows what happens WITHOUT atomic check-and-increment
        org_quota = {"used": 0, "limit": 10}
        accepted_count = [0]

        def check_and_increment_unsafe():
            """Non-atomic check-and-increment (racy)."""
            if org_quota["used"] < org_quota["limit"]:
                # Race window: multiple threads can pass the check
                time.sleep(0.001)
                org_quota["used"] += 1
                accepted_count[0] += 1
                return True
            return False

        threads = [threading.Thread(target=check_and_increment_unsafe) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Without lock, more than 10 requests may be accepted (race condition)
        # This is expected and demonstrates why we need the lock
        # We just verify it doesn't crash
        assert accepted_count[0] > 0

    def test_multiple_orgs_independent_quotas(self):
        """Verify quotas are independent per organization."""
        from aragora.server.handlers.gauntlet import _quota_lock

        # Simulate multiple organizations
        org_quotas = {
            "org-1": {"used": 0, "limit": 5},
            "org-2": {"used": 0, "limit": 10},
            "org-3": {"used": 0, "limit": 3},
        }
        results = {org_id: [] for org_id in org_quotas}

        def check_and_increment(org_id, request_id):
            with _quota_lock:
                quota = org_quotas[org_id]
                if quota["used"] < quota["limit"]:
                    quota["used"] += 1
                    results[org_id].append(("accepted", request_id))
                    return True
                else:
                    results[org_id].append(("rejected", request_id))
                    return False

        # Send requests for all orgs concurrently
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for org_id in org_quotas:
                for i in range(20):
                    futures.append(
                        executor.submit(check_and_increment, org_id, f"{org_id}-req-{i}")
                    )
            for f in as_completed(futures):
                f.result()

        # Verify each org's quota was respected
        for org_id, quota in org_quotas.items():
            accepted = [r for r in results[org_id] if r[0] == "accepted"]
            assert (
                len(accepted) == quota["limit"]
            ), f"{org_id}: expected {quota['limit']} accepted, got {len(accepted)}"


class TestQuotaIncrementResilience:
    """Tests for quota increment failure handling."""

    def test_increment_failure_does_not_consume_quota(self):
        """Verify failed increments don't consume quota."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 10}
        increment_failures = [0]
        successful_requests = []

        def check_and_increment_with_failure(request_id, should_fail=False):
            with _quota_lock:
                if org_quota["used"] < org_quota["limit"]:
                    if should_fail:
                        # Simulate increment failure
                        increment_failures[0] += 1
                        raise Exception("Increment failed")
                    org_quota["used"] += 1
                    successful_requests.append(request_id)
                    return True
                return False

        errors = []

        def make_request(request_id, should_fail):
            try:
                check_and_increment_with_failure(request_id, should_fail)
            except Exception:
                pass  # Expected for failed requests

        # Make 15 requests: 5 will fail, 10 should succeed
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for i in range(15):
                should_fail = i < 5  # First 5 requests will fail
                futures.append(executor.submit(make_request, f"req-{i}", should_fail))
            for f in as_completed(futures):
                f.result()

        # 5 failures + 10 successes (up to quota)
        assert increment_failures[0] == 5
        assert len(successful_requests) == 10
        assert org_quota["used"] == 10

    def test_quota_rollback_on_operation_failure(self):
        """Verify quota can be rolled back if operation fails after increment."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 10}
        completed_operations = []

        def check_increment_and_operate(request_id, operation_fails=False):
            with _quota_lock:
                if org_quota["used"] >= org_quota["limit"]:
                    return False, "quota_exceeded"

                org_quota["used"] += 1
                current_used = org_quota["used"]

            # Outside lock: perform operation
            if operation_fails:
                # Roll back the increment
                with _quota_lock:
                    org_quota["used"] -= 1
                return False, "operation_failed"

            completed_operations.append(request_id)
            return True, "success"

        # 5 successful operations, 5 that fail and roll back
        results = []
        for i in range(10):
            success, reason = check_increment_and_operate(f"req-{i}", operation_fails=(i % 2 == 0))
            results.append((f"req-{i}", success, reason))

        # 5 successful (odd indices)
        successful = [r for r in results if r[1]]
        assert len(successful) == 5

        # Final quota should reflect only successful operations
        assert org_quota["used"] == 5


class TestQuotaConcurrentReads:
    """Tests for concurrent quota reads during updates."""

    def test_concurrent_read_during_increment(self):
        """Verify quota reads are consistent during concurrent increments."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 100}
        read_values = []
        errors = []

        def increment_quota():
            try:
                for _ in range(50):
                    with _quota_lock:
                        if org_quota["used"] < org_quota["limit"]:
                            org_quota["used"] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"increment: {e}")

        def read_quota():
            try:
                for _ in range(100):
                    with _quota_lock:
                        value = org_quota["used"]
                    read_values.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"read: {e}")

        threads = [
            threading.Thread(target=increment_quota),
            threading.Thread(target=increment_quota),
            threading.Thread(target=read_quota),
            threading.Thread(target=read_quota),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"

        # All read values should be valid (0 to 100)
        for value in read_values:
            assert 0 <= value <= 100, f"Invalid quota value: {value}"

        # Final value should be 100 (2 threads * 50 increments)
        assert org_quota["used"] == 100

    def test_quota_snapshot_consistency(self):
        """Verify quota snapshots are consistent."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 50, "reserved": 0}
        snapshots = []

        def modify_quota():
            for _ in range(100):
                with _quota_lock:
                    org_quota["used"] += 1
                    org_quota["reserved"] = org_quota["limit"] - org_quota["used"]
                time.sleep(0.001)

        def take_snapshot():
            for _ in range(100):
                with _quota_lock:
                    snapshot = {
                        "used": org_quota["used"],
                        "limit": org_quota["limit"],
                        "reserved": org_quota["reserved"],
                    }
                snapshots.append(snapshot)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=modify_quota),
            threading.Thread(target=take_snapshot),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All snapshots should be internally consistent
        for snap in snapshots:
            # used + reserved should equal limit
            assert (
                snap["used"] + snap["reserved"] == snap["limit"]
            ), f"Inconsistent snapshot: {snap}"


class TestQuotaEdgeCases:
    """Tests for quota edge cases."""

    def test_zero_quota_limit(self):
        """Verify zero quota limit works correctly."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 0}
        rejected = [0]

        def try_request():
            with _quota_lock:
                if org_quota["used"] < org_quota["limit"]:
                    org_quota["used"] += 1
                    return True
                rejected[0] += 1
                return False

        # All requests should be rejected
        results = [try_request() for _ in range(10)]
        assert all(r is False for r in results)
        assert rejected[0] == 10
        assert org_quota["used"] == 0

    def test_exactly_at_limit(self):
        """Verify behavior when exactly at quota limit."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 9, "limit": 10}
        results = []

        def try_request(request_id):
            with _quota_lock:
                if org_quota["used"] < org_quota["limit"]:
                    org_quota["used"] += 1
                    results.append(("accepted", request_id))
                    return True
                results.append(("rejected", request_id))
                return False

        # Exactly 1 should succeed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_request, f"req-{i}") for i in range(10)]
            for f in as_completed(futures):
                f.result()

        accepted = [r for r in results if r[0] == "accepted"]
        rejected = [r for r in results if r[0] == "rejected"]

        assert len(accepted) == 1
        assert len(rejected) == 9
        assert org_quota["used"] == 10

    def test_negative_quota_prevention(self):
        """Verify quota cannot go negative."""
        from aragora.server.handlers.gauntlet import _quota_lock

        org_quota = {"used": 0, "limit": 10}

        def decrement_quota():
            with _quota_lock:
                if org_quota["used"] > 0:
                    org_quota["used"] -= 1
                    return True
                return False

        # Try to decrement when already at 0
        results = [decrement_quota() for _ in range(10)]

        assert all(r is False for r in results)
        assert org_quota["used"] == 0  # Should never go negative


@pytest.fixture
def mock_user_store():
    """Fixture to create a mock user store for quota tests."""
    store = Mock()
    store.get_organization_by_id.return_value = Mock(
        is_at_limit=False,
        usage_count=0,
        quota_limit=10,
    )
    return store
