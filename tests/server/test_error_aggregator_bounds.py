"""Tests for bounded data structures in error aggregator to prevent OOM.

These tests verify that the timeline deque respects its maxlen and properly
evicts old entries to prevent unbounded memory growth in production.
"""

import pytest
from collections import deque

from aragora.server.error_aggregator import (
    ErrorAggregator,
    ErrorSignature,
    reset_error_aggregator,
)


class TestErrorAggregatorTimelineBounds:
    """Test that the timeline deque has proper bounds."""

    def test_timeline_has_maxlen(self) -> None:
        """Verify that _timeline deque is initialized with maxlen."""
        aggregator = ErrorAggregator()
        assert aggregator._timeline.maxlen == 100000

    def test_timeline_evicts_old_entries_at_capacity(self) -> None:
        """Verify that old entries are evicted when timeline reaches maxlen."""
        # Create aggregator with small timeline for testing
        aggregator = ErrorAggregator()
        # Override with small maxlen for testing
        aggregator._timeline = deque(maxlen=100)

        # Record more errors than maxlen
        for i in range(150):
            aggregator.record(
                error=f"Test error {i}",
                component="test.component",
            )

        # Timeline should not exceed maxlen
        assert len(aggregator._timeline) == 100

    def test_timeline_preserves_recent_entries(self) -> None:
        """Verify that most recent entries are preserved when evicting."""
        aggregator = ErrorAggregator()
        aggregator._timeline = deque(maxlen=10)

        # Record 15 errors with unique components to avoid normalization grouping
        for i in range(15):
            aggregator.record(
                error="Test error",
                component=f"component_{chr(ord('a') + i)}",  # component_a, component_b, etc.
            )

        # Timeline should contain the last 10 entries
        assert len(aggregator._timeline) == 10

        # The most recent components (indices 5-14) should be in timeline
        timeline_components = [sig.component for _, sig in aggregator._timeline]
        for i in range(5, 15):
            expected_component = f"component_{chr(ord('a') + i)}"
            assert expected_component in timeline_components, (
                f"Expected {expected_component} to be in timeline"
            )

    def test_basic_functionality_with_bounds(self) -> None:
        """Verify that basic error recording still works with bounded timeline."""
        aggregator = ErrorAggregator()

        # Record an error
        sig, is_new = aggregator.record(
            error="Test error message",
            component="test.module",
            context={"key": "value"},
        )

        # Verify it was recorded
        assert is_new is True
        assert sig.error_type == "Error"
        assert sig.component == "test.module"

        # Verify we can retrieve it
        error = aggregator.get_error(sig.fingerprint)
        assert error is not None
        assert error.count == 1

    def test_get_stats_works_with_bounded_timeline(self) -> None:
        """Verify get_stats works correctly with bounded timeline."""
        aggregator = ErrorAggregator()
        aggregator._timeline = deque(maxlen=100)

        # Record some errors with unique components to create distinct signatures
        for i in range(50):
            aggregator.record(error="Test error", component=f"component_{i}")

        stats = aggregator.get_stats()
        assert stats.unique_errors == 50
        assert stats.total_occurrences == 50

    def test_get_error_rate_works_with_bounded_timeline(self) -> None:
        """Verify get_error_rate works correctly with bounded timeline."""
        aggregator = ErrorAggregator()
        aggregator._timeline = deque(maxlen=100)

        # Record some errors
        for i in range(20):
            aggregator.record(error=f"Error {i}", component="test")

        # Should be able to calculate error rate
        rate = aggregator.get_error_rate(minutes=5)
        assert rate > 0

    def test_clear_works_with_bounded_timeline(self) -> None:
        """Verify clear() works correctly with bounded timeline."""
        aggregator = ErrorAggregator()
        aggregator._timeline = deque(maxlen=100)

        # Record some errors with unique components to create distinct signatures
        for i in range(50):
            aggregator.record(error="Test error", component=f"component_{i}")

        assert len(aggregator._timeline) == 50
        assert len(aggregator._errors) == 50

        # Clear should work
        aggregator.clear()
        assert len(aggregator._timeline) == 0
        assert len(aggregator._errors) == 0


class TestErrorAggregatorContextBounds:
    """Test that context tracking has proper bounds."""

    def test_contexts_dict_is_bounded(self) -> None:
        """Verify that contexts dict doesn't grow unbounded."""
        aggregator = ErrorAggregator()

        # Record many errors with unique contexts
        sig, _ = aggregator.record(
            error="Repeated error",
            component="test",
            context={"key0": "value0"},
        )

        # Record same error with many different context values
        for i in range(1500):
            aggregator.record(
                error="Repeated error",
                component="test",
                context={f"key{i}": f"value{i}"},
            )

        # Get the aggregated error
        error = aggregator.get_error(sig.fingerprint)
        assert error is not None

        # Contexts should be capped at 1000
        assert len(error.contexts) <= 1000


@pytest.fixture(autouse=True)
def cleanup_aggregator() -> None:
    """Reset global aggregator after each test."""
    yield
    reset_error_aggregator()
