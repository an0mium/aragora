"""Comprehensive tests for metrics tracking module.

Covers all public functions:
  - get_start_time()
  - track_verification() / get_verification_stats()
  - track_request() / get_request_stats()

Test classes:
  TestGetStartTime           - Server start time retrieval
  TestTrackVerification      - Verification outcome tracking
  TestGetVerificationStats   - Verification stats snapshots and derived metrics
  TestTrackRequest           - Request counting and error tracking
  TestGetRequestStats        - Request stats snapshots
  TestEviction               - MAX_TRACKED_ENDPOINTS eviction behaviour
  TestThreadSafety           - Concurrent access from multiple threads
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

import aragora.server.handlers.metrics.tracking as tracking_mod
from aragora.server.handlers.metrics.tracking import (
    MAX_TRACKED_ENDPOINTS,
    get_request_stats,
    get_start_time,
    get_verification_stats,
    track_request,
    track_verification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tracking_state():
    """Reset all module-level tracking state before each test."""
    # Save originals
    orig_request_counts = tracking_mod._request_counts.copy()
    orig_error_counts = tracking_mod._error_counts.copy()
    orig_verification_stats = tracking_mod._verification_stats.copy()

    # Clear state
    tracking_mod._request_counts.clear()
    tracking_mod._error_counts.clear()
    tracking_mod._verification_stats.update(
        {
            "total_claims_processed": 0,
            "z3_verified": 0,
            "z3_disproved": 0,
            "z3_timeout": 0,
            "z3_translation_failed": 0,
            "confidence_fallback": 0,
            "total_verification_time_ms": 0.0,
        }
    )

    yield

    # Restore originals
    tracking_mod._request_counts.clear()
    tracking_mod._request_counts.update(orig_request_counts)
    tracking_mod._error_counts.clear()
    tracking_mod._error_counts.update(orig_error_counts)
    tracking_mod._verification_stats.clear()
    tracking_mod._verification_stats.update(orig_verification_stats)


# ---------------------------------------------------------------------------
# TestGetStartTime
# ---------------------------------------------------------------------------


class TestGetStartTime:
    """Tests for get_start_time()."""

    def test_returns_float(self):
        """Start time should be a float timestamp."""
        result = get_start_time()
        assert isinstance(result, float)

    def test_start_time_is_in_the_past(self):
        """Start time should be less than or equal to current time."""
        assert get_start_time() <= time.time()

    def test_start_time_is_module_level(self):
        """Start time should be the module-level _start_time value."""
        assert get_start_time() == tracking_mod._start_time


# ---------------------------------------------------------------------------
# TestTrackVerification
# ---------------------------------------------------------------------------


class TestTrackVerification:
    """Tests for track_verification()."""

    def test_increments_total_claims(self):
        """Tracking a verification should increment total_claims_processed."""
        track_verification("z3_verified")
        assert tracking_mod._verification_stats["total_claims_processed"] == 1

    def test_increments_specific_status(self):
        """Tracking should increment the specific status counter."""
        track_verification("z3_verified")
        assert tracking_mod._verification_stats["z3_verified"] == 1

    def test_increments_z3_disproved(self):
        """z3_disproved status should be tracked correctly."""
        track_verification("z3_disproved")
        assert tracking_mod._verification_stats["z3_disproved"] == 1

    def test_increments_z3_timeout(self):
        """z3_timeout status should be tracked correctly."""
        track_verification("z3_timeout")
        assert tracking_mod._verification_stats["z3_timeout"] == 1

    def test_increments_z3_translation_failed(self):
        """z3_translation_failed status should be tracked correctly."""
        track_verification("z3_translation_failed")
        assert tracking_mod._verification_stats["z3_translation_failed"] == 1

    def test_increments_confidence_fallback(self):
        """confidence_fallback status should be tracked correctly."""
        track_verification("confidence_fallback")
        assert tracking_mod._verification_stats["confidence_fallback"] == 1

    def test_unknown_status_still_increments_total(self):
        """An unrecognised status should still increment total_claims_processed."""
        track_verification("unknown_status")
        assert tracking_mod._verification_stats["total_claims_processed"] == 1

    def test_unknown_status_does_not_create_key(self):
        """An unrecognised status should NOT create a new key in the stats dict."""
        track_verification("unknown_status")
        assert "unknown_status" not in tracking_mod._verification_stats

    def test_accumulates_verification_time(self):
        """Verification time in ms should be accumulated."""
        track_verification("z3_verified", verification_time_ms=42.5)
        track_verification("z3_verified", verification_time_ms=17.5)
        assert tracking_mod._verification_stats["total_verification_time_ms"] == 60.0

    def test_default_verification_time_is_zero(self):
        """Default verification time should be 0.0."""
        track_verification("z3_verified")
        assert tracking_mod._verification_stats["total_verification_time_ms"] == 0.0

    def test_multiple_different_statuses(self):
        """Multiple different statuses should each increment independently."""
        track_verification("z3_verified")
        track_verification("z3_disproved")
        track_verification("z3_timeout")
        assert tracking_mod._verification_stats["total_claims_processed"] == 3
        assert tracking_mod._verification_stats["z3_verified"] == 1
        assert tracking_mod._verification_stats["z3_disproved"] == 1
        assert tracking_mod._verification_stats["z3_timeout"] == 1


# ---------------------------------------------------------------------------
# TestGetVerificationStats
# ---------------------------------------------------------------------------


class TestGetVerificationStats:
    """Tests for get_verification_stats()."""

    def test_returns_dict(self):
        """Should return a dict."""
        stats = get_verification_stats()
        assert isinstance(stats, dict)

    def test_empty_stats_have_zero_totals(self):
        """With no tracking, all counters should be zero."""
        stats = get_verification_stats()
        assert stats["total_claims_processed"] == 0
        assert stats["z3_verified"] == 0
        assert stats["z3_disproved"] == 0

    def test_avg_verification_time_zero_when_no_claims(self):
        """Average verification time should be 0.0 when no claims processed."""
        stats = get_verification_stats()
        assert stats["avg_verification_time_ms"] == 0.0

    def test_z3_success_rate_zero_when_no_claims(self):
        """Success rate should be 0.0 when no claims processed."""
        stats = get_verification_stats()
        assert stats["z3_success_rate"] == 0.0

    def test_avg_verification_time_calculated(self):
        """Average verification time should be total_time / total_claims."""
        track_verification("z3_verified", verification_time_ms=100.0)
        track_verification("z3_disproved", verification_time_ms=200.0)
        stats = get_verification_stats()
        # (100 + 200) / 2 = 150.0
        assert stats["avg_verification_time_ms"] == 150.0

    def test_z3_success_rate_calculated(self):
        """z3_success_rate should be z3_verified / total_claims_processed."""
        track_verification("z3_verified")
        track_verification("z3_disproved")
        track_verification("z3_verified")
        track_verification("z3_timeout")
        stats = get_verification_stats()
        # 2 verified / 4 total = 0.5
        assert stats["z3_success_rate"] == 0.5

    def test_z3_success_rate_rounded_to_four_decimals(self):
        """z3_success_rate should be rounded to 4 decimal places."""
        track_verification("z3_verified")
        track_verification("z3_disproved")
        track_verification("z3_disproved")
        stats = get_verification_stats()
        # 1/3 = 0.3333...
        assert stats["z3_success_rate"] == 0.3333

    def test_avg_verification_time_rounded_to_two_decimals(self):
        """avg_verification_time_ms should be rounded to 2 decimal places."""
        track_verification("z3_verified", verification_time_ms=1.0)
        track_verification("z3_verified", verification_time_ms=2.0)
        track_verification("z3_verified", verification_time_ms=3.0)
        stats = get_verification_stats()
        # (1+2+3)/3 = 2.0
        assert stats["avg_verification_time_ms"] == 2.0

    def test_returns_snapshot_not_reference(self):
        """Returned dict should be a copy, not a reference to internal state."""
        stats = get_verification_stats()
        stats["z3_verified"] = 9999
        # Internal state should be unchanged
        assert tracking_mod._verification_stats["z3_verified"] == 0

    def test_all_expected_keys_present(self):
        """Returned stats should contain all base keys plus derived keys."""
        stats = get_verification_stats()
        expected_keys = {
            "total_claims_processed",
            "z3_verified",
            "z3_disproved",
            "z3_timeout",
            "z3_translation_failed",
            "confidence_fallback",
            "total_verification_time_ms",
            "avg_verification_time_ms",
            "z3_success_rate",
        }
        assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestTrackRequest
# ---------------------------------------------------------------------------


class TestTrackRequest:
    """Tests for track_request()."""

    def test_tracks_single_request(self):
        """A single request should be counted."""
        track_request("/api/v1/test")
        assert tracking_mod._request_counts["/api/v1/test"] == 1

    def test_tracks_multiple_requests_same_endpoint(self):
        """Multiple requests to same endpoint should accumulate."""
        track_request("/api/v1/test")
        track_request("/api/v1/test")
        track_request("/api/v1/test")
        assert tracking_mod._request_counts["/api/v1/test"] == 3

    def test_tracks_different_endpoints(self):
        """Different endpoints should be tracked independently."""
        track_request("/api/v1/a")
        track_request("/api/v1/b")
        assert tracking_mod._request_counts["/api/v1/a"] == 1
        assert tracking_mod._request_counts["/api/v1/b"] == 1

    def test_error_increments_error_count(self):
        """Error requests should increment the error counter."""
        track_request("/api/v1/test", is_error=True)
        assert tracking_mod._error_counts["/api/v1/test"] == 1

    def test_error_also_increments_request_count(self):
        """Error requests should also increment the request counter."""
        track_request("/api/v1/test", is_error=True)
        assert tracking_mod._request_counts["/api/v1/test"] == 1

    def test_non_error_does_not_increment_error_count(self):
        """Non-error requests should NOT appear in error_counts."""
        track_request("/api/v1/test", is_error=False)
        assert "/api/v1/test" not in tracking_mod._error_counts

    def test_mixed_error_and_success(self):
        """Mix of error and success requests counted correctly."""
        track_request("/api/v1/test", is_error=False)
        track_request("/api/v1/test", is_error=True)
        track_request("/api/v1/test", is_error=False)
        assert tracking_mod._request_counts["/api/v1/test"] == 3
        assert tracking_mod._error_counts["/api/v1/test"] == 1

    def test_default_is_not_error(self):
        """Default is_error should be False."""
        track_request("/api/v1/test")
        assert "/api/v1/test" not in tracking_mod._error_counts


# ---------------------------------------------------------------------------
# TestGetRequestStats
# ---------------------------------------------------------------------------


class TestGetRequestStats:
    """Tests for get_request_stats()."""

    def test_returns_dict(self):
        """Should return a dict."""
        stats = get_request_stats()
        assert isinstance(stats, dict)

    def test_empty_stats(self):
        """With no tracking, totals should be zero and snapshot empty."""
        stats = get_request_stats()
        assert stats["total_requests"] == 0
        assert stats["total_errors"] == 0
        assert stats["counts_snapshot"] == []

    def test_total_requests_sum(self):
        """total_requests should be the sum of all endpoint counts."""
        track_request("/a")
        track_request("/a")
        track_request("/b")
        stats = get_request_stats()
        assert stats["total_requests"] == 3

    def test_total_errors_sum(self):
        """total_errors should be the sum of all endpoint error counts."""
        track_request("/a", is_error=True)
        track_request("/a", is_error=True)
        track_request("/b", is_error=True)
        stats = get_request_stats()
        assert stats["total_errors"] == 3

    def test_counts_snapshot_contents(self):
        """counts_snapshot should contain (endpoint, count) tuples."""
        track_request("/api/v1/test")
        track_request("/api/v1/test")
        stats = get_request_stats()
        assert ("/api/v1/test", 2) in stats["counts_snapshot"]

    def test_all_expected_keys_present(self):
        """Returned stats should contain required keys."""
        stats = get_request_stats()
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "counts_snapshot" in stats


# ---------------------------------------------------------------------------
# TestEviction
# ---------------------------------------------------------------------------


class TestEviction:
    """Tests for MAX_TRACKED_ENDPOINTS eviction behavior."""

    def test_max_tracked_endpoints_constant(self):
        """MAX_TRACKED_ENDPOINTS should be 1000."""
        assert MAX_TRACKED_ENDPOINTS == 1000

    def test_eviction_when_at_capacity(self):
        """When at MAX_TRACKED_ENDPOINTS, adding a new endpoint evicts old ones."""
        # Fill to capacity
        for i in range(MAX_TRACKED_ENDPOINTS):
            tracking_mod._request_counts[f"/ep/{i}"] = 1

        assert len(tracking_mod._request_counts) == MAX_TRACKED_ENDPOINTS

        # Add one more (new endpoint, not existing)
        track_request("/new_endpoint")

        # Should have evicted ~10% (100) and added the new one
        # Total should be 1000 - 100 + 1 = 901
        assert len(tracking_mod._request_counts) <= MAX_TRACKED_ENDPOINTS
        assert "/new_endpoint" in tracking_mod._request_counts

    def test_eviction_removes_first_entries(self):
        """Eviction should remove the first ~10% of keys (approximate LRU)."""
        for i in range(MAX_TRACKED_ENDPOINTS):
            tracking_mod._request_counts[f"/ep/{i}"] = 1

        # Also set error counts for some of the first entries
        tracking_mod._error_counts["/ep/0"] = 1
        tracking_mod._error_counts["/ep/1"] = 1

        track_request("/new_endpoint")

        # First entries (0..99) should have been evicted
        assert "/ep/0" not in tracking_mod._request_counts
        assert "/ep/1" not in tracking_mod._request_counts
        # Error counts for evicted entries should also be removed
        assert "/ep/0" not in tracking_mod._error_counts
        assert "/ep/1" not in tracking_mod._error_counts

    def test_no_eviction_for_existing_endpoint(self):
        """When tracking an existing endpoint at capacity, no eviction happens."""
        for i in range(MAX_TRACKED_ENDPOINTS):
            tracking_mod._request_counts[f"/ep/{i}"] = 1

        # Track an existing endpoint
        track_request("/ep/0")

        # All endpoints should still be there (no eviction needed)
        assert len(tracking_mod._request_counts) == MAX_TRACKED_ENDPOINTS
        assert tracking_mod._request_counts["/ep/0"] == 2

    def test_eviction_removes_at_least_one(self):
        """Eviction should remove at least 1 entry even for small dicts."""
        # We need a scenario where len(keys) // 10 == 0, but max(1, ...) forces 1.
        # With MAX_TRACKED_ENDPOINTS = 1000, this won't happen naturally, but
        # we can verify the logic path by filling exactly to capacity and checking
        # the new endpoint is added.
        for i in range(MAX_TRACKED_ENDPOINTS):
            tracking_mod._request_counts[f"/ep/{i}"] = 1

        track_request("/brand_new")
        assert "/brand_new" in tracking_mod._request_counts


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safety of tracking functions."""

    def test_concurrent_request_tracking(self):
        """Concurrent track_request calls should not lose updates."""
        num_threads = 10
        requests_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(requests_per_thread):
                track_request("/concurrent")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * requests_per_thread
        assert tracking_mod._request_counts["/concurrent"] == expected

    def test_concurrent_verification_tracking(self):
        """Concurrent track_verification calls should not lose updates."""
        num_threads = 10
        verifications_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(verifications_per_thread):
                track_verification("z3_verified", verification_time_ms=1.0)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * verifications_per_thread
        assert tracking_mod._verification_stats["total_claims_processed"] == expected
        assert tracking_mod._verification_stats["z3_verified"] == expected
        assert tracking_mod._verification_stats["total_verification_time_ms"] == float(expected)

    def test_concurrent_error_tracking(self):
        """Concurrent error tracking should not lose error counts."""
        num_threads = 8
        requests_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(requests_per_thread):
                track_request("/error_ep", is_error=True)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * requests_per_thread
        assert tracking_mod._request_counts["/error_ep"] == expected
        assert tracking_mod._error_counts["/error_ep"] == expected
