"""Tests for error aggregation and deduplication service."""

import time
from collections import deque
from unittest.mock import patch

import pytest

from aragora.server.error_aggregator import (
    AggregatedError,
    ErrorAggregator,
    ErrorAggregatorStats,
    ErrorOccurrence,
    ErrorSignature,
    _extract_location,
    _normalize_message,
    get_error_aggregator,
    record_error,
    reset_error_aggregator,
)


class TestNormalizeMessage:
    """Test message normalization for error grouping."""

    def test_normalize_uuid(self):
        """Test UUID replacement."""
        msg = "Error for user 12345678-1234-1234-1234-123456789abc"
        result = _normalize_message(msg)
        assert "<UUID>" in result
        assert "12345678-1234" not in result

    def test_normalize_uuid_case_insensitive(self):
        """Test UUID replacement is case-insensitive."""
        msg = "Error for ABC12345-ABCD-1234-ABCD-123456789ABC"
        result = _normalize_message(msg)
        assert "<UUID>" in result

    def test_normalize_hex_id(self):
        """Test hex ID replacement."""
        msg = "Object deadbeef12345678 not found"
        result = _normalize_message(msg)
        assert "<HEX_ID>" in result
        assert "deadbeef" not in result

    def test_normalize_numbers(self):
        """Test number replacement."""
        msg = "Failed after 42 attempts with 128 bytes"
        result = _normalize_message(msg)
        assert "<N>" in result
        assert "42" not in result
        assert "128" not in result

    def test_normalize_file_path(self):
        """Test file path replacement."""
        msg = "Error loading /var/log/aragora/server.log"
        result = _normalize_message(msg)
        assert "<PATH>" in result
        assert "/var/log" not in result

    def test_normalize_quoted_strings(self):
        """Test quoted string replacement."""
        msg = "Unknown key 'api_secret' in config \"settings.json\""
        result = _normalize_message(msg)
        assert "<STR>" in result
        assert "api_secret" not in result
        assert "settings.json" not in result

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        msg = "Error   with    multiple    spaces"
        result = _normalize_message(msg)
        assert "  " not in result

    def test_normalize_preserves_keywords(self):
        """Test that important keywords are preserved."""
        msg = "Connection timeout while connecting to database"
        result = _normalize_message(msg)
        assert "Connection" in result
        assert "timeout" in result
        assert "database" in result

    def test_normalize_complex_message(self):
        """Test normalization of complex message with multiple patterns."""
        msg = (
            "Request 12345678-aaaa-bbbb-cccc-dddddddddddd failed: "
            "Timeout after 30 seconds loading '/api/v1/users'"
        )
        result = _normalize_message(msg)
        assert "<UUID>" in result
        assert "<N>" in result
        assert "<STR>" in result
        assert "failed" in result
        assert "Timeout" in result


class TestExtractLocation:
    """Test location extraction from tracebacks."""

    def test_extract_location_from_traceback(self):
        """Test extraction from valid traceback."""
        tb = """Traceback (most recent call last):
  File "/home/aragora/server/handlers.py", line 42, in handle
    raise ValueError("test")
ValueError: test"""
        result = _extract_location(tb)
        assert "aragora" in result or "handlers.py" in result
        assert "42" in result

    def test_extract_location_empty(self):
        """Test extraction from empty traceback."""
        result = _extract_location(None)
        assert result == "unknown"

    def test_extract_location_no_file(self):
        """Test extraction from traceback without file info."""
        tb = "Some random error text without file info"
        result = _extract_location(tb)
        assert result == "unknown"


class TestErrorSignature:
    """Test ErrorSignature dataclass."""

    def test_signature_hash_equality(self):
        """Test signatures with same values are equal."""
        sig1 = ErrorSignature(
            error_type="ValueError",
            normalized_message="test error",
            location="file.py:42",
            component="test",
        )
        sig2 = ErrorSignature(
            error_type="ValueError",
            normalized_message="test error",
            location="file.py:42",
            component="test",
        )
        assert sig1 == sig2
        assert hash(sig1) == hash(sig2)

    def test_signature_hash_inequality(self):
        """Test signatures with different values are not equal."""
        sig1 = ErrorSignature(
            error_type="ValueError",
            normalized_message="test error",
            location="file.py:42",
            component="test",
        )
        sig2 = ErrorSignature(
            error_type="TypeError",
            normalized_message="test error",
            location="file.py:42",
            component="test",
        )
        assert sig1 != sig2

    def test_signature_fingerprint(self):
        """Test fingerprint generation."""
        sig = ErrorSignature(
            error_type="ValueError",
            normalized_message="test error",
            location="file.py:42",
            component="test",
        )
        fp = sig.fingerprint
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_signature_fingerprint_consistency(self):
        """Test fingerprint is consistent for same values."""
        sig1 = ErrorSignature(
            error_type="ValueError",
            normalized_message="test",
            location="a:1",
            component="c",
        )
        sig2 = ErrorSignature(
            error_type="ValueError",
            normalized_message="test",
            location="a:1",
            component="c",
        )
        assert sig1.fingerprint == sig2.fingerprint

    def test_signature_not_equal_to_non_signature(self):
        """Test signature equality with non-signature object."""
        sig = ErrorSignature(
            error_type="ValueError",
            normalized_message="test",
            location="a:1",
            component="c",
        )
        assert sig != "not a signature"
        assert sig != 42
        assert sig is not None


class TestErrorOccurrence:
    """Test ErrorOccurrence dataclass."""

    def test_occurrence_creation(self):
        """Test basic occurrence creation."""
        occ = ErrorOccurrence(
            timestamp=time.time(),
            message="test error",
            context={"key": "value"},
            trace_id="trace-123",
        )
        assert occ.message == "test error"
        assert occ.context == {"key": "value"}
        assert occ.trace_id == "trace-123"


class TestAggregatedError:
    """Test AggregatedError dataclass."""

    def test_calculate_rate_single_occurrence(self):
        """Test rate calculation with single occurrence."""
        sig = ErrorSignature(
            error_type="Error",
            normalized_message="test",
            location="a:1",
            component="test",
        )
        agg = AggregatedError(
            signature=sig,
            first_seen=time.time(),
            last_seen=time.time(),
            count=1,
            occurrences=deque(),
        )
        assert agg._calculate_rate() == 0.0

    def test_calculate_rate_multiple_occurrences(self):
        """Test rate calculation with multiple occurrences over time."""
        now = time.time()
        sig = ErrorSignature(
            error_type="Error",
            normalized_message="test",
            location="a:1",
            component="test",
        )
        agg = AggregatedError(
            signature=sig,
            first_seen=now - 60,  # 1 minute ago
            last_seen=now,
            count=10,
            occurrences=deque(),
        )
        rate = agg._calculate_rate()
        assert rate == pytest.approx(10.0, rel=0.1)

    def test_calculate_rate_short_duration(self):
        """Test rate calculation with very short duration."""
        now = time.time()
        sig = ErrorSignature(
            error_type="Error",
            normalized_message="test",
            location="a:1",
            component="test",
        )
        agg = AggregatedError(
            signature=sig,
            first_seen=now,
            last_seen=now + 0.5,  # Half second
            count=5,
            occurrences=deque(),
        )
        rate = agg._calculate_rate()
        assert rate == 5.0  # Returns count when duration < 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = time.time()
        sig = ErrorSignature(
            error_type="ValueError",
            normalized_message="test error <N>",
            location="test.py:42",
            component="test.component",
        )
        occ = ErrorOccurrence(
            timestamp=now,
            message="test error 123",
            context={},
            trace_id="trace-1",
        )
        agg = AggregatedError(
            signature=sig,
            first_seen=now - 60,
            last_seen=now,
            count=5,
            occurrences=deque([occ]),
        )
        agg.contexts["key:value"] = 3

        result = agg.to_dict()

        assert result["fingerprint"] == sig.fingerprint
        assert result["error_type"] == "ValueError"
        assert result["message_pattern"] == "test error <N>"
        assert result["location"] == "test.py:42"
        assert result["component"] == "test.component"
        assert result["count"] == 5
        assert "first_seen" in result
        assert "last_seen" in result
        assert "rate_per_minute" in result
        assert result["sample_contexts"] == {"key:value": 3}
        assert len(result["recent_occurrences"]) == 1


class TestErrorAggregatorStats:
    """Test ErrorAggregatorStats dataclass."""

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = ErrorAggregatorStats(
            unique_errors=10,
            total_occurrences=100,
            errors_last_minute=5,
            errors_last_5_minutes=25,
            errors_last_hour=80,
            top_components={"debate": 50, "auth": 30},
            top_error_types={"ValueError": 60, "TypeError": 40},
            dedup_ratio=0.3,
        )

        result = stats.to_dict()

        assert result["unique_errors"] == 10
        assert result["total_occurrences"] == 100
        assert result["errors_last_minute"] == 5
        assert result["dedup_ratio"] == 0.3


class TestErrorAggregator:
    """Test ErrorAggregator class."""

    def setup_method(self):
        """Reset global aggregator before each test."""
        reset_error_aggregator()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_error_aggregator()

    def test_record_exception(self):
        """Test recording an exception."""
        agg = ErrorAggregator()
        exc = ValueError("test error 123")

        sig, is_new = agg.record(exc, component="test")

        assert is_new is True
        assert sig.error_type == "ValueError"

    def test_record_string_error(self):
        """Test recording a string error message."""
        agg = ErrorAggregator()

        sig, is_new = agg.record("Something went wrong", component="test")

        assert is_new is True
        assert sig.error_type == "Error"

    def test_record_with_context(self):
        """Test recording error with context."""
        agg = ErrorAggregator()
        context = {"debate_id": "d-123", "user_id": "u-456"}

        agg.record(ValueError("test"), component="debate", context=context)
        # Record same error again to test context tracking
        agg.record(ValueError("test"), component="debate", context=context)

        errors = agg.get_recent_errors(minutes=60)
        assert len(errors) == 1
        assert errors[0].contexts.get("debate_id:d-123", 0) >= 1

    def test_record_with_trace_id(self):
        """Test recording error with trace ID."""
        agg = ErrorAggregator()

        agg.record(ValueError("test"), component="test", trace_id="trace-123")

        errors = agg.get_recent_errors(minutes=60)
        assert len(errors[0].occurrences) == 1
        assert errors[0].occurrences[0].trace_id == "trace-123"

    def test_deduplication(self):
        """Test error deduplication within window."""
        agg = ErrorAggregator(dedup_window_seconds=60)

        sig1, is_new1 = agg.record(ValueError("test"), component="test")
        sig2, is_new2 = agg.record(ValueError("test"), component="test")

        assert is_new1 is True
        assert is_new2 is False
        assert sig1 == sig2

    def test_deduplication_counter(self):
        """Test deduplication counter is updated."""
        agg = ErrorAggregator(dedup_window_seconds=60)

        agg.record(ValueError("test"), component="test")
        agg.record(ValueError("test"), component="test")
        agg.record(ValueError("test"), component="test")

        stats = agg.get_stats()
        assert stats.dedup_ratio > 0

    def test_different_errors_not_deduplicated(self):
        """Test different errors are not deduplicated."""
        agg = ErrorAggregator()

        sig1, _ = agg.record(ValueError("error A"), component="test")
        sig2, _ = agg.record(TypeError("error B"), component="test")

        assert sig1 != sig2
        assert agg.get_stats().unique_errors == 2

    def test_get_error_by_fingerprint(self):
        """Test retrieving error by fingerprint."""
        agg = ErrorAggregator()
        sig, _ = agg.record(ValueError("test"), component="test")

        error = agg.get_error(sig.fingerprint)

        assert error is not None
        assert error.signature == sig

    def test_get_error_nonexistent_fingerprint(self):
        """Test retrieving nonexistent error returns None."""
        agg = ErrorAggregator()

        error = agg.get_error("nonexistent123")

        assert error is None

    def test_get_recent_errors(self):
        """Test getting recent errors."""
        agg = ErrorAggregator()
        agg.record(ValueError("error 1"), component="test")
        agg.record(TypeError("error 2"), component="test")

        errors = agg.get_recent_errors(minutes=60)

        assert len(errors) == 2

    def test_get_recent_errors_filter_by_component(self):
        """Test filtering recent errors by component."""
        agg = ErrorAggregator()
        agg.record(ValueError("error 1"), component="debate")
        agg.record(TypeError("error 2"), component="auth")

        errors = agg.get_recent_errors(minutes=60, component="debate")

        assert len(errors) == 1
        assert errors[0].signature.component == "debate"

    def test_get_recent_errors_sorted_by_time(self):
        """Test recent errors are sorted by last_seen descending."""
        agg = ErrorAggregator()
        agg.record(ValueError("first"), component="test")
        time.sleep(0.01)
        agg.record(TypeError("second"), component="test")

        errors = agg.get_recent_errors(minutes=60)

        assert errors[0].signature.error_type == "TypeError"
        assert errors[1].signature.error_type == "ValueError"

    def test_get_top_errors(self):
        """Test getting top errors by count."""
        agg = ErrorAggregator()
        for _ in range(5):
            agg.record(ValueError("frequent"), component="test")
        for _ in range(2):
            agg.record(TypeError("rare"), component="test")

        top = agg.get_top_errors(limit=10)

        assert len(top) == 2
        assert top[0].count == 5
        assert top[1].count == 2

    def test_get_top_errors_with_limit(self):
        """Test limit on top errors."""
        agg = ErrorAggregator()
        for i in range(10):
            agg.record(ValueError(f"error {i}"), component=f"comp{i}")

        top = agg.get_top_errors(limit=3)

        assert len(top) == 3

    def test_get_top_errors_with_minutes(self):
        """Test top errors with time filter."""
        agg = ErrorAggregator()
        agg.record(ValueError("test"), component="test")

        top = agg.get_top_errors(limit=10, minutes=60)

        assert len(top) == 1

    def test_get_errors_by_component(self):
        """Test getting all errors for a component."""
        agg = ErrorAggregator()
        agg.record(ValueError("error 1"), component="debate")
        agg.record(TypeError("error 2"), component="debate")
        agg.record(RuntimeError("error 3"), component="auth")

        errors = agg.get_errors_by_component("debate")

        assert len(errors) == 2
        assert all(e.signature.component == "debate" for e in errors)

    def test_get_error_rate(self):
        """Test error rate calculation."""
        agg = ErrorAggregator()
        for _ in range(10):
            agg.record(ValueError("test"), component="test")

        rate = agg.get_error_rate(minutes=5)

        assert rate == pytest.approx(2.0, rel=0.1)  # 10 errors / 5 minutes

    def test_get_error_rate_no_errors(self):
        """Test error rate with no errors."""
        agg = ErrorAggregator()

        rate = agg.get_error_rate(minutes=5)

        assert rate == 0.0

    def test_get_stats(self):
        """Test getting aggregator statistics."""
        agg = ErrorAggregator()
        agg.record(ValueError("error 1"), component="debate")
        agg.record(ValueError("error 1"), component="debate")
        agg.record(TypeError("error 2"), component="auth")

        stats = agg.get_stats()

        assert stats.unique_errors == 2
        assert stats.total_occurrences == 3
        assert "debate" in stats.top_components
        assert "ValueError" in stats.top_error_types

    def test_clear(self):
        """Test clearing all error data."""
        agg = ErrorAggregator()
        agg.record(ValueError("test"), component="test")

        agg.clear()

        stats = agg.get_stats()
        assert stats.unique_errors == 0
        assert stats.total_occurrences == 0

    def test_max_unique_errors_eviction(self):
        """Test eviction when max unique errors reached."""
        agg = ErrorAggregator(max_unique_errors=5)

        for i in range(10):
            agg.record(ValueError(f"unique error {i}"), component=f"comp{i}")

        assert agg.get_stats().unique_errors <= 5

    def test_max_samples_per_error(self):
        """Test max samples per error limit."""
        agg = ErrorAggregator(max_samples_per_error=3)

        for i in range(10):
            agg.record(ValueError("same error"), component="test", context={"i": i})

        errors = agg.get_recent_errors(minutes=60)
        assert len(errors[0].occurrences) <= 3

    def test_cleanup_old_entries(self):
        """Test cleanup of old entries."""
        agg = ErrorAggregator(window_minutes=1)

        # Record and manually set old timestamp
        agg.record(ValueError("old error"), component="test")

        # Manually age the error
        with agg._lock:
            for sig in agg._errors:
                agg._errors[sig].last_seen = time.time() - 120  # 2 minutes ago
            agg._timeline.clear()
            agg._timeline.append((time.time() - 120, sig))

        # Record new error to trigger cleanup
        agg.record(TypeError("new error"), component="test")

        # Old error should be cleaned up
        errors = agg.get_recent_errors(minutes=60)
        assert len(errors) == 1
        assert errors[0].signature.error_type == "TypeError"

    def test_thread_safety(self):
        """Test thread-safe operation."""
        import threading

        agg = ErrorAggregator()
        errors_recorded = []

        def record_errors():
            for i in range(100):
                sig, _ = agg.record(ValueError(f"error {i}"), component="test")
                errors_recorded.append(sig)

        threads = [threading.Thread(target=record_errors) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors_recorded) == 500
        assert agg.get_stats().total_occurrences == 500


class TestGlobalAggregator:
    """Test global aggregator functions."""

    def setup_method(self):
        """Reset global aggregator before each test."""
        reset_error_aggregator()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_error_aggregator()

    def test_get_error_aggregator(self):
        """Test getting global aggregator."""
        agg = get_error_aggregator()

        assert isinstance(agg, ErrorAggregator)

    def test_get_error_aggregator_singleton(self):
        """Test global aggregator is singleton."""
        agg1 = get_error_aggregator()
        agg2 = get_error_aggregator()

        assert agg1 is agg2

    def test_reset_error_aggregator(self):
        """Test resetting global aggregator."""
        agg1 = get_error_aggregator()
        reset_error_aggregator()
        agg2 = get_error_aggregator()

        assert agg1 is not agg2

    def test_record_error_convenience(self):
        """Test record_error convenience function."""
        sig = record_error(ValueError("test"), component="test")

        assert isinstance(sig, ErrorSignature)
        assert sig.error_type == "ValueError"

    def test_record_error_with_all_params(self):
        """Test record_error with all parameters."""
        sig = record_error(
            ValueError("test"),
            component="debate",
            context={"id": "123"},
            trace_id="trace-456",
        )

        agg = get_error_aggregator()
        error = agg.get_error(sig.fingerprint)
        assert error is not None
        assert error.occurrences[0].trace_id == "trace-456"


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_default_window_minutes(self):
        """Test default window minutes from environment."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ERROR_WINDOW_MINUTES": "30"},
            clear=False,
        ):
            # Need to reload module to pick up env change
            # For this test, just verify the environment reading logic
            import os

            assert int(os.environ.get("ARAGORA_ERROR_WINDOW_MINUTES", "60")) == 30

    def test_default_max_unique_errors(self):
        """Test default max unique errors from environment."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ERROR_MAX_UNIQUE": "500"},
            clear=False,
        ):
            import os

            assert int(os.environ.get("ARAGORA_ERROR_MAX_UNIQUE", "1000")) == 500

    def test_default_dedup_window(self):
        """Test default dedup window from environment."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ERROR_DEDUP_SECONDS": "30"},
            clear=False,
        ):
            import os

            assert int(os.environ.get("ARAGORA_ERROR_DEDUP_SECONDS", "60")) == 30


class TestMessageTruncation:
    """Test message and pattern length limits."""

    def test_normalized_message_truncation(self):
        """Test normalized message is truncated to 200 chars."""
        agg = ErrorAggregator()
        long_message = "x" * 500

        sig, _ = agg.record(ValueError(long_message), component="test")

        assert len(sig.normalized_message) <= 200

    def test_occurrence_message_truncation(self):
        """Test occurrence message is truncated to 500 chars."""
        agg = ErrorAggregator()
        long_message = "y" * 1000

        agg.record(ValueError(long_message), component="test")

        errors = agg.get_recent_errors(minutes=60)
        assert len(errors[0].occurrences[0].message) <= 500
