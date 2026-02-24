"""
Tests for request size and complexity limits middleware.

Covers body size validation, JSON depth checking, query parameter
counting, and the combined ``validate_request`` method.
"""

from __future__ import annotations

import pytest

from aragora.server.security.request_limits import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_JSON_DEPTH,
    DEFAULT_MAX_QUERY_PARAMS,
    HTTP_BAD_REQUEST,
    HTTP_PAYLOAD_TOO_LARGE,
    RequestLimitsConfig,
    RequestLimitsMiddleware,
    check_json_depth,
    check_query_params,
)


# ============================================================================
# Body size checks
# ============================================================================


class TestContentLengthValidation:
    """Tests for Content-Length / body size validation."""

    def test_accepts_within_limit(self) -> None:
        mw = RequestLimitsMiddleware()
        ok, msg = mw.check_content_length({"Content-Length": "1024"})
        assert ok is True
        assert msg == ""

    def test_rejects_oversized(self) -> None:
        config = RequestLimitsConfig(max_body_bytes=1000)
        mw = RequestLimitsMiddleware(config)
        ok, msg = mw.check_content_length({"Content-Length": "2000"})
        assert ok is False
        assert "exceeds" in msg.lower() or "size" in msg.lower()

    def test_rejects_negative_content_length(self) -> None:
        mw = RequestLimitsMiddleware()
        ok, msg = mw.check_content_length({"Content-Length": "-1"})
        assert ok is False
        assert "negative" in msg.lower()

    def test_rejects_invalid_content_length(self) -> None:
        mw = RequestLimitsMiddleware()
        ok, msg = mw.check_content_length({"Content-Length": "abc"})
        assert ok is False

    def test_accepts_missing_content_length(self) -> None:
        mw = RequestLimitsMiddleware()
        ok, msg = mw.check_content_length({})
        assert ok is True

    def test_case_insensitive_header(self) -> None:
        config = RequestLimitsConfig(max_body_bytes=100)
        mw = RequestLimitsMiddleware(config)
        ok, _ = mw.check_content_length({"content-length": "200"})
        assert ok is False

    def test_default_limit_is_10mb(self) -> None:
        assert DEFAULT_MAX_BODY_BYTES == 10 * 1024 * 1024

    def test_path_override(self) -> None:
        config = RequestLimitsConfig(
            max_body_bytes=1000,
            path_body_overrides={"/api/upload": 100_000_000},
        )
        mw = RequestLimitsMiddleware(config)

        # Default path should reject
        ok, _ = mw.check_content_length({"Content-Length": "5000"}, path="/api/data")
        assert ok is False

        # Override path should allow
        ok, _ = mw.check_content_length({"Content-Length": "5000"}, path="/api/upload")
        assert ok is True

    def test_exactly_at_limit(self) -> None:
        config = RequestLimitsConfig(max_body_bytes=1000)
        mw = RequestLimitsMiddleware(config)
        ok, _ = mw.check_content_length({"Content-Length": "1000"})
        assert ok is True

    def test_one_over_limit(self) -> None:
        config = RequestLimitsConfig(max_body_bytes=1000)
        mw = RequestLimitsMiddleware(config)
        ok, _ = mw.check_content_length({"Content-Length": "1001"})
        assert ok is False


# ============================================================================
# JSON depth checks
# ============================================================================


class TestJsonDepthValidation:
    """Tests for JSON nesting depth validation."""

    def test_flat_dict_ok(self) -> None:
        ok, _ = check_json_depth({"a": 1, "b": 2})
        assert ok is True

    def test_nested_within_limit(self) -> None:
        obj: dict = {"a": {"b": {"c": 1}}}
        ok, _ = check_json_depth(obj, max_depth=5)
        assert ok is True

    def test_exceeds_depth(self) -> None:
        # Build a dict nested to depth 25
        obj: dict = {"val": 1}
        for _ in range(24):
            obj = {"nested": obj}
        ok, msg = check_json_depth(obj, max_depth=20)
        assert ok is False
        assert "depth" in msg.lower()

    def test_list_nesting(self) -> None:
        obj: list = [1]
        for _ in range(24):
            obj = [obj]
        ok, _ = check_json_depth(obj, max_depth=20)
        assert ok is False

    def test_mixed_nesting(self) -> None:
        obj: dict = {"a": [{"b": [{"c": 1}]}]}
        ok, _ = check_json_depth(obj, max_depth=10)
        assert ok is True

    def test_empty_containers(self) -> None:
        ok, _ = check_json_depth({})
        assert ok is True
        ok, _ = check_json_depth([])
        assert ok is True

    def test_scalar_values(self) -> None:
        ok, _ = check_json_depth("hello")
        assert ok is True
        ok, _ = check_json_depth(42)
        assert ok is True
        ok, _ = check_json_depth(None)
        assert ok is True

    def test_depth_exactly_at_limit(self) -> None:
        # Depth of exactly 3
        obj = {"a": {"b": {"c": 1}}}
        ok, _ = check_json_depth(obj, max_depth=3)
        assert ok is True

    def test_depth_one_over_limit(self) -> None:
        obj = {"a": {"b": {"c": {"d": 1}}}}
        ok, _ = check_json_depth(obj, max_depth=3)
        assert ok is False

    def test_default_depth_is_20(self) -> None:
        assert DEFAULT_MAX_JSON_DEPTH == 20

    def test_middleware_delegates(self) -> None:
        mw = RequestLimitsMiddleware(RequestLimitsConfig(max_json_depth=2))
        ok, _ = mw.check_json_depth({"a": {"b": {"c": 1}}})
        assert ok is False


# ============================================================================
# Query param checks
# ============================================================================


class TestQueryParamValidation:
    """Tests for query parameter count limits."""

    def test_empty_query_ok(self) -> None:
        ok, _ = check_query_params("")
        assert ok is True

    def test_within_limit(self) -> None:
        qs = "&".join(f"key{i}=val{i}" for i in range(10))
        ok, _ = check_query_params(qs, max_params=50)
        assert ok is True

    def test_exceeds_limit(self) -> None:
        qs = "&".join(f"key{i}=val{i}" for i in range(60))
        ok, msg = check_query_params(qs, max_params=50)
        assert ok is False
        assert "too many" in msg.lower()

    def test_duplicate_keys_count_all(self) -> None:
        """Duplicate keys should count each value."""
        qs = "&".join(f"key=val{i}" for i in range(55))
        ok, _ = check_query_params(qs, max_params=50)
        assert ok is False

    def test_default_limit_is_50(self) -> None:
        assert DEFAULT_MAX_QUERY_PARAMS == 50

    def test_middleware_delegates(self) -> None:
        mw = RequestLimitsMiddleware(RequestLimitsConfig(max_query_params=3))
        qs = "a=1&b=2&c=3&d=4"
        ok, _ = mw.check_query_params(qs)
        assert ok is False


# ============================================================================
# Combined validate_request
# ============================================================================


class TestValidateRequest:
    """Tests for the combined validate_request method."""

    def test_all_ok(self) -> None:
        mw = RequestLimitsMiddleware()
        ok, status, msg = mw.validate_request(
            headers={"Content-Length": "100"},
            path="/api/data",
            query_string="a=1&b=2",
            parsed_body={"key": "value"},
        )
        assert ok is True
        assert status == 200

    def test_body_too_large(self) -> None:
        config = RequestLimitsConfig(max_body_bytes=100)
        mw = RequestLimitsMiddleware(config)
        ok, status, msg = mw.validate_request(
            headers={"Content-Length": "9999"},
        )
        assert ok is False
        assert status == HTTP_PAYLOAD_TOO_LARGE

    def test_too_many_params(self) -> None:
        config = RequestLimitsConfig(max_query_params=2)
        mw = RequestLimitsMiddleware(config)
        ok, status, msg = mw.validate_request(
            headers={},
            query_string="a=1&b=2&c=3",
        )
        assert ok is False
        assert status == HTTP_BAD_REQUEST

    def test_json_too_deep(self) -> None:
        config = RequestLimitsConfig(max_json_depth=2)
        mw = RequestLimitsMiddleware(config)
        ok, status, msg = mw.validate_request(
            headers={},
            parsed_body={"a": {"b": {"c": 1}}},
        )
        assert ok is False
        assert status == HTTP_BAD_REQUEST

    def test_body_check_runs_first(self) -> None:
        """Body size should be checked before JSON depth."""
        config = RequestLimitsConfig(max_body_bytes=10, max_json_depth=1)
        mw = RequestLimitsMiddleware(config)
        ok, status, _ = mw.validate_request(
            headers={"Content-Length": "9999"},
            parsed_body={"a": {"b": 1}},
        )
        assert ok is False
        assert status == HTTP_PAYLOAD_TOO_LARGE  # body first, not depth
