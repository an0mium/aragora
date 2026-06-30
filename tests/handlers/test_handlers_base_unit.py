"""Unit tests for base handler utilities.

Tests BoundedTTLCache, ttl_cache decorator, response helpers,
and parameter validation functions.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import (
    BoundedTTLCache,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
    get_string_param,
    get_bool_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
    ttl_cache,
    clear_cache,
    get_cache_stats,
)
from aragora.server.validation import (
    validate_path_segment,
    validate_agent_name,
    validate_debate_id,
)


class TestBoundedTTLCache:
    """Tests for BoundedTTLCache class."""

    def test_get_miss_returns_false(self) -> None:
        """Test cache miss returns (False, None)."""
        cache = BoundedTTLCache(max_entries=10)
        hit, value = cache.get("nonexistent", ttl_seconds=60)
        assert hit is False
        assert value is None

    def test_set_and_get_hit(self) -> None:
        """Test set followed by get returns cached value."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value1"

    def test_expired_entry_returns_miss(self) -> None:
        """Test expired entries are not returned."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        # Get with 0 TTL - should be expired immediately
        time.sleep(0.01)
        hit, value = cache.get("key1", ttl_seconds=0)
        assert hit is False
        assert value is None

    def test_eviction_when_full(self) -> None:
        """Test oldest entries are evicted when cache is full."""
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Cache is full, adding key4 should evict oldest
        cache.set("key4", "value4")

        # key1 should be evicted (oldest)
        assert "key1" not in cache
        # key4 should exist
        assert "key4" in cache

    def test_update_moves_to_end(self) -> None:
        """Test updating existing key moves it to end (most recently used)."""
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Update key1 - should move to end
        cache.set("key1", "updated")

        # Add new key - should evict key2 (now oldest)
        cache.set("key4", "value4")

        assert "key1" in cache  # Was updated, moved to end
        assert "key2" not in cache  # Now oldest, evicted
        assert "key4" in cache

    def test_get_moves_to_end(self) -> None:
        """Test getting a key moves it to end (LRU behavior)."""
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 - should move to end
        cache.get("key1", ttl_seconds=60)

        # Add new key - should evict key2 (now oldest)
        cache.set("key4", "value4")

        assert "key1" in cache  # Was accessed, moved to end
        assert "key2" not in cache  # Now oldest, evicted

    def test_clear_all(self) -> None:
        """Test clear removes all entries."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = cache.clear()

        assert count == 2
        assert len(cache) == 0

    def test_clear_with_prefix(self) -> None:
        """Test clear with prefix only removes matching entries."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("user:1", "alice")
        cache.set("user:2", "bob")
        cache.set("item:1", "widget")

        count = cache.clear("user:")

        assert count == 2
        assert len(cache) == 1
        assert "item:1" in cache

    def test_stats_tracking(self) -> None:
        """Test hit/miss statistics are tracked."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        cache.get("key1", ttl_seconds=60)  # Hit
        cache.get("key2", ttl_seconds=60)  # Miss
        cache.get("key1", ttl_seconds=60)  # Hit

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_len_returns_entry_count(self) -> None:
        """Test __len__ returns correct count."""
        cache = BoundedTTLCache(max_entries=10)
        assert len(cache) == 0

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2


class TestTTLCacheDecorator:
    """Tests for the ttl_cache decorator."""

    def test_caches_function_result(self) -> None:
        """Test decorated function result is cached."""
        call_count = 0

        @ttl_cache(ttl_seconds=60, key_prefix="test", skip_first=False)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once due to caching

    def test_different_args_not_cached(self) -> None:
        """Test different arguments produce different cache entries."""
        call_count = 0

        @ttl_cache(ttl_seconds=60, key_prefix="test2", skip_first=False)
        def add_one(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        result1 = add_one(1)
        result2 = add_one(2)

        assert result1 == 2
        assert result2 == 3
        assert call_count == 2  # Called twice for different args

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        clear_cache()


class TestJsonResponse:
    """Tests for json_response function."""

    def test_returns_handler_result(self) -> None:
        """Test returns HandlerResult instance."""
        result = json_response({"key": "value"})
        assert isinstance(result, HandlerResult)

    def test_default_status_200(self) -> None:
        """Test default status code is 200."""
        result = json_response({})
        assert result.status_code == 200

    def test_custom_status_code(self) -> None:
        """Test custom status code."""
        result = json_response({}, status=201)
        assert result.status_code == 201

    def test_content_type_json(self) -> None:
        """Test content type is application/json."""
        result = json_response({})
        assert result.content_type == "application/json"

    def test_body_is_json_encoded(self) -> None:
        """Test body is JSON encoded."""
        result = json_response({"name": "test", "count": 42})
        data = json.loads(result.body.decode("utf-8"))
        assert data["name"] == "test"
        assert data["count"] == 42

    def test_custom_headers(self) -> None:
        """Test custom headers are included."""
        result = json_response({}, headers={"X-Custom": "value"})
        assert result.headers["X-Custom"] == "value"


class TestErrorResponse:
    """Tests for error_response function."""

    def test_default_status_400(self) -> None:
        """Test default status code is 400."""
        result = error_response("Bad request")
        assert result.status_code == 400

    def test_custom_status_code(self) -> None:
        """Test custom status code."""
        result = error_response("Not found", status=404)
        assert result.status_code == 404

    def test_error_message_in_body(self) -> None:
        """Test error message is in response body."""
        result = error_response("Something went wrong")
        data = json.loads(result.body.decode("utf-8"))
        assert data["error"] == "Something went wrong"


class TestGetIntParam:
    """Tests for get_int_param function."""

    def test_returns_parsed_int(self) -> None:
        """Test parsing integer from params."""
        params = {"limit": ["42"]}
        assert get_int_param(params, "limit", 10) == 42

    def test_returns_default_for_missing(self) -> None:
        """Test default is returned for missing key."""
        params = {}
        assert get_int_param(params, "limit", 10) == 10

    def test_returns_default_for_invalid(self) -> None:
        """Test default is returned for non-integer value."""
        params = {"limit": ["not_a_number"]}
        assert get_int_param(params, "limit", 10) == 10

    def test_handles_empty_list(self) -> None:
        """Test handles empty list value."""
        params = {"limit": []}
        assert get_int_param(params, "limit", 10) == 10

    def test_takes_first_value(self) -> None:
        """Test takes first value from list."""
        params = {"limit": ["5", "10", "15"]}
        assert get_int_param(params, "limit", 10) == 5


class TestGetFloatParam:
    """Tests for get_float_param function."""

    def test_returns_parsed_float(self) -> None:
        """Test parsing float from params."""
        params = {"threshold": ["0.75"]}
        assert get_float_param(params, "threshold", 0.5) == 0.75

    def test_returns_default_for_missing(self) -> None:
        """Test default is returned for missing key."""
        params = {}
        assert get_float_param(params, "threshold", 0.5) == 0.5

    def test_returns_default_for_invalid(self) -> None:
        """Test default is returned for non-float value."""
        params = {"threshold": ["abc"]}
        assert get_float_param(params, "threshold", 0.5) == 0.5


class TestGetStringParam:
    """Tests for get_string_param function."""

    def test_returns_string_value(self) -> None:
        """Test returns string value."""
        params = {"name": ["alice"]}
        assert get_string_param(params, "name") == "alice"

    def test_returns_none_for_missing(self) -> None:
        """Test returns None for missing key."""
        params = {}
        assert get_string_param(params, "name") is None

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        params = {}
        assert get_string_param(params, "name", "default") == "default"


class TestGetBoolParam:
    """Tests for get_bool_param function.

    Note: get_bool_param expects string values, not list values.
    This is different from get_int_param/get_float_param which handle lists.
    """

    def test_true_values(self) -> None:
        """Test various true values."""
        # get_bool_param expects string values directly
        for val in ["true", "1", "yes", "on"]:
            params = {"enabled": val}
            assert get_bool_param(params, "enabled", False) is True

    def test_false_values(self) -> None:
        """Test various false values."""
        # Values not in the true set are considered false
        for val in ["false", "0", "no", "off"]:
            params = {"enabled": val}
            assert get_bool_param(params, "enabled", True) is False

    def test_returns_default_for_missing(self) -> None:
        """Test returns default for missing key."""
        params = {}
        assert get_bool_param(params, "enabled", True) is True
        assert get_bool_param(params, "enabled", False) is False


class TestGetClampedIntParam:
    """Tests for get_clamped_int_param function."""

    def test_returns_value_within_bounds(self) -> None:
        """Test value within bounds is returned unchanged."""
        params = {"limit": ["50"]}
        result = get_clamped_int_param(params, "limit", default=10, min_val=1, max_val=100)
        assert result == 50

    def test_clamps_to_min(self) -> None:
        """Test value below min is clamped."""
        params = {"limit": ["-5"]}
        result = get_clamped_int_param(params, "limit", default=10, min_val=1, max_val=100)
        assert result == 1

    def test_clamps_to_max(self) -> None:
        """Test value above max is clamped."""
        params = {"limit": ["500"]}
        result = get_clamped_int_param(params, "limit", default=10, min_val=1, max_val=100)
        assert result == 100


class TestGetBoundedFloatParam:
    """Tests for get_bounded_float_param function."""

    def test_clamps_to_bounds(self) -> None:
        """Test float is clamped to bounds."""
        params = {"threshold": ["1.5"]}
        result = get_bounded_float_param(params, "threshold", default=0.5, min_val=0.0, max_val=1.0)
        assert result == 1.0


class TestGetBoundedStringParam:
    """Tests for get_bounded_string_param function."""

    def test_truncates_long_string(self) -> None:
        """Test long string is truncated."""
        params = {"query": ["a" * 1000]}
        result = get_bounded_string_param(params, "query", max_length=100)
        assert len(result) == 100

    def test_returns_short_string_unchanged(self) -> None:
        """Test short string is returned unchanged."""
        params = {"query": ["hello"]}
        result = get_bounded_string_param(params, "query", max_length=100)
        assert result == "hello"

    def test_returns_none_for_missing(self) -> None:
        """Test returns None for missing key."""
        params = {}
        result = get_bounded_string_param(params, "query")
        assert result is None


class TestValidatePathSegment:
    """Tests for validate_path_segment function."""

    def test_valid_segment(self) -> None:
        """Test valid segment passes."""
        valid, error = validate_path_segment("valid-segment_123", "test")
        assert valid is True
        assert error is None

    def test_empty_segment_fails(self) -> None:
        """Test empty segment fails."""
        valid, error = validate_path_segment("", "test")
        assert valid is False
        assert "Missing" in error

    def test_path_traversal_blocked(self) -> None:
        """Test path traversal attempt is blocked."""
        valid, error = validate_path_segment("../etc", "test")
        assert valid is False
        assert "invalid" in error.lower() or "pattern" in error.lower()

    def test_slash_blocked(self) -> None:
        """Test forward slash is blocked."""
        valid, error = validate_path_segment("a/b", "test")
        assert valid is False
        assert "invalid" in error.lower() or "pattern" in error.lower()

    def test_invalid_format_blocked(self) -> None:
        """Test invalid format is blocked."""
        # Assuming SAFE_ID_PATTERN doesn't allow spaces
        valid, error = validate_path_segment("has space", "test")
        assert valid is False
        assert "invalid" in error.lower() or "pattern" in error.lower()


class TestValidateAgentName:
    """Tests for validate_agent_name function."""

    def test_valid_agent_name(self) -> None:
        """Test valid agent names pass."""
        for name in ["claude", "gemini", "grok", "agent-1", "agent_2"]:
            valid, error = validate_agent_name(name)
            assert valid is True, f"{name} should be valid"

    def test_invalid_agent_name(self) -> None:
        """Test invalid agent names fail."""
        valid, error = validate_agent_name("agent with spaces")
        assert valid is False


class TestValidateDebateId:
    """Tests for validate_debate_id function."""

    def test_valid_debate_id(self) -> None:
        """Test valid debate IDs pass."""
        for id in ["debate-123", "test_debate", "abc123"]:
            valid, error = validate_debate_id(id)
            assert valid is True, f"{id} should be valid"

    def test_traversal_blocked(self) -> None:
        """Test path traversal in debate ID is blocked."""
        valid, error = validate_debate_id("../../../etc")
        assert valid is False
        assert "invalid" in error.lower() or "pattern" in error.lower()


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_initialization(self) -> None:
        """Test HandlerResult can be initialized."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok": true}',
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b'{"ok": true}'

    def test_default_headers(self) -> None:
        """Test headers default to empty dict."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello",
        )
        assert result.headers == {}

    def test_custom_headers(self) -> None:
        """Test custom headers are preserved."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello",
            headers={"X-Custom": "value"},
        )
        assert result.headers["X-Custom"] == "value"
